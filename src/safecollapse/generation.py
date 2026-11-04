"""Generate answers for one or more prompt sets with a single model load.

One `Job` = one prompt set and one sampling setting (synthetic data at temperature 1, or
evaluation prompts greedily). The vLLM backend is the default (fast); the transformers backend
is a fallback for runtimes where vLLM will not install. Both use `modeling.prompt_ids`, so the
prompt tokens are identical to training.
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path

from .config import GenerationConfig, ModelSpec, Sampling
from .io import write_jsonl
from .modeling import load_model, load_tokenizer, prompt_ids


@dataclass
class Job:
    name: str
    records: list[dict]  # each has "id" and "messages"
    sampling: Sampling
    out: Path
    n: int = 1  # samples per prompt


def request_seed(seed: int, job: str, record_id: str) -> int:
    """Per-prompt sampling seed: results do not depend on batch order or batch size."""
    digest = hashlib.sha256(f"{seed}/{job}/{record_id}".encode()).digest()
    return int.from_bytes(digest[:4], "little") & 0x7FFFFFFF


def run_jobs(
    model_path: str,
    revision: str | None,
    spec: ModelSpec,
    jobs: list[Job],
    cfg: GenerationConfig,
    seed: int,
) -> dict:
    """Run every job, write each job's completions, return speed statistics per job."""
    tokenizer = load_tokenizer(model_path, revision)
    backend = _VLLM if cfg.backend == "vllm" else _HF
    engine = backend(model_path, revision, spec, cfg, seed, tokenizer)
    stats = {}
    for job in jobs:
        start = time.time()
        rows = engine.generate(job, seed)
        elapsed = time.time() - start
        write_jsonl(job.out, rows)
        tokens = sum(r["n_tokens"] for r in rows)
        stats[job.name] = {
            "prompts": len(job.records),
            "samples": len(rows),
            "tokens": tokens,
            "seconds": round(elapsed, 2),
            "tokens_per_s": round(tokens / elapsed, 1) if elapsed else None,
            "skipped_too_long": sum(1 for r in rows if r.get("skipped")),
        }
    return stats


def stop_ids(tokenizer) -> list[int]:
    """EOS plus the chat turn-end token (Gemma <end_of_turn>, Qwen <|im_end|>), so both backends
    stop where the chat template ends an assistant turn."""
    ids = {tokenizer.eos_token_id}
    for token in ("<end_of_turn>", "<|im_end|>", "<|eot_id|>", "<|end|>"):
        tid = tokenizer.convert_tokens_to_ids(token)
        if isinstance(tid, int) and tid != tokenizer.unk_token_id:
            ids.add(tid)
    return sorted(i for i in ids if i is not None)


def _too_long(ids: list[int], sampling: Sampling, spec: ModelSpec) -> bool:
    return len(ids) + sampling.max_new_tokens > spec.max_model_len


def _skipped_row(record_id: str, k: int) -> dict:
    return {
        "id": record_id,
        "sample": k,
        "completion": "",
        "n_tokens": 0,
        "finish_reason": "skipped",
        "skipped": "prompt_too_long",
    }


class _VLLM:
    def __init__(self, path, revision, spec: ModelSpec, cfg: GenerationConfig, seed, tokenizer):
        os.environ.update(cfg.vllm_env)
        from vllm import LLM

        self.spec, self.tokenizer, self.stop = spec, tokenizer, stop_ids(tokenizer)
        kwargs = dict(
            model=path,
            revision=revision,
            tokenizer_revision=revision,
            dtype="bfloat16",
            seed=seed,
            gpu_memory_utilization=cfg.gpu_memory_utilization,
            max_model_len=spec.max_model_len,
        )
        try:
            # Ignore the model's own sampling defaults (Qwen ships top_k=20 and a repetition
            # penalty); only the settings in our config may shape the answers.
            self.llm = LLM(generation_config="vllm", **kwargs)
        except TypeError:  # vLLM versions before the option existed never applied them
            self.llm = LLM(**kwargs)

    def generate(self, job: Job, seed: int) -> list[dict]:
        from vllm import SamplingParams

        if job.sampling.num_beams:
            raise ValueError("beam search needs the transformers backend (generation.backend)")
        prompts, params, keep, rows = [], [], [], {}
        for rec in job.records:
            ids = prompt_ids(self.tokenizer, rec["messages"])
            if _too_long(ids, job.sampling, self.spec):
                rows[rec["id"]] = [_skipped_row(rec["id"], k) for k in range(job.n)]
                continue
            prompts.append({"prompt_token_ids": ids})
            params.append(
                SamplingParams(
                    n=job.n,
                    temperature=job.sampling.temperature,
                    top_p=job.sampling.top_p,
                    max_tokens=job.sampling.max_new_tokens,
                    seed=request_seed(seed, job.name, rec["id"]),
                    stop_token_ids=self.stop,
                )
            )
            keep.append(rec["id"])
        for rid, out in zip(keep, self.llm.generate(prompts, params)):
            rows[rid] = [
                {
                    "id": rid,
                    "sample": k,
                    "completion": c.text,
                    "n_tokens": len(c.token_ids),
                    "finish_reason": c.finish_reason,
                }
                for k, c in enumerate(out.outputs)
            ]
        return [row for rec in job.records for row in rows[rec["id"]]]


class _HF:
    def _config(self, s: Sampling, n: int, stops: list[int]):
        """A complete GenerationConfig, so the model's shipped defaults (e.g. Qwen's top_k=20 and
        repetition penalty 1.05) can never leak into sampling."""
        from transformers import GenerationConfig

        sample = s.temperature > 0
        beams = s.num_beams or 1
        return GenerationConfig(
            do_sample=sample,
            num_beams=beams,
            early_stopping=beams > 1,
            temperature=s.temperature if sample else 1.0,
            top_p=s.top_p if sample else 1.0,
            top_k=0,
            repetition_penalty=1.0,
            max_new_tokens=s.max_new_tokens,
            num_return_sequences=n,
            eos_token_id=stops,
            pad_token_id=self.tokenizer.pad_token_id,
        )

    def __init__(self, path, revision, spec: ModelSpec, cfg: GenerationConfig, seed, tokenizer):
        self.spec, self.tokenizer, self.batch_size = spec, tokenizer, cfg.hf_batch_size
        self.model = load_model(path, revision, spec)

    def generate(self, job: Job, seed: int) -> list[dict]:
        import torch

        torch.manual_seed(seed)
        tok, s = self.tokenizer, job.sampling
        encoded = [(rec["id"], prompt_ids(tok, rec["messages"])) for rec in job.records]
        rows: dict[str, list[dict]] = {}
        todo = []
        for rid, ids in encoded:
            if _too_long(ids, s, self.spec):
                rows[rid] = [_skipped_row(rid, k) for k in range(job.n)]
            else:
                todo.append((rid, ids))
        todo.sort(key=lambda x: len(x[1]))  # similar lengths per batch: less padding
        stops = set(stop_ids(tok))
        for i in range(0, len(todo), self.batch_size):
            batch = todo[i : i + self.batch_size]
            width = max(len(ids) for _, ids in batch)
            input_ids = torch.tensor(
                [[tok.pad_token_id] * (width - len(ids)) + ids for _, ids in batch]
            )
            mask = torch.tensor([[0] * (width - len(ids)) + [1] * len(ids) for _, ids in batch])
            with torch.no_grad():
                out = self.model.generate(
                    input_ids=input_ids.cuda(),
                    attention_mask=mask.cuda(),
                    generation_config=self._config(s, job.n, sorted(stops)),
                )
            new = out[:, width:].tolist()
            for j, (rid, _) in enumerate(batch):
                samples = []
                for k in range(job.n):
                    seq = new[j * job.n + k]
                    cut = next((p for p, t in enumerate(seq) if t in stops), None)
                    gen = seq if cut is None else seq[:cut]
                    samples.append(
                        {
                            "id": rid,
                            "sample": k,
                            "n_tokens": len(gen),
                            "completion": tok.decode(gen, skip_special_tokens=True),
                            "finish_reason": "length" if cut is None else "stop",
                        }
                    )
                rows[rid] = samples
        return [row for rid, _ in encoded for row in rows[rid]]
