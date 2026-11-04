"""Run the study: every checkpoint in the plan, every stage, resumable.

Two kinds of stage:
- GPU stages (train, generate, collapse metrics, capability, refusal direction, HarmBench
  classifier) each run in a fresh subprocess, so GPU memory is fully released between vLLM and
  training, and multi-GPU training can go through `accelerate launch`.
- CPU stages (building training data, API judging) run in this process; judging runs in a
  background thread so the GPU never waits for the API.

A stage counts as done only when its marker file exists, and markers are written after the
outputs, so killing a run at any point and starting it again redoes at most one stage.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from . import arms, jailbreaks
from .config import Config, Sampling, load_config, save_resolved
from .io import read_json, read_jsonl, sha256_file, write_json, write_jsonl
from .manifest import update as update_manifest
from .plan import (
    Layout,
    Node,
    build_plan,
    gen0,
    generators,
    published_ref,
    sources,
    weights_stage,
)

log = logging.getLogger("safecollapse")

THROUGHPUT_KEYS = ("launcher", "per_device_batch_size", "gradient_checkpointing")
# SORRY-Bench is scored by its own local judge (stage_sorry_judge), not by the API judges: the
# StrongREJECT rubric counts ordinary requests as refused, which would undercount fulfilment of
# SORRY-Bench's borderline categories.
STRONGREJECT_BENCHES = ("harmbench_val", "harmbench_test")


# --------------------------------------------------------------------------------------------
# What each node still needs


def eval_jobs(cfg: Config, node: Node) -> list[str]:
    """Generation job names for evaluation: one per benchmark, plus the robustness samples."""
    jobs = list(cfg.evaluation.benchmarks)
    if node.label in cfg.evaluation.robustness_labels and cfg.evaluation.robustness_samples:
        jobs.append("harmbench_test@robust")
    if node.label in cfg.evaluation.jailbreak_labels:
        jobs += [jailbreaks.job_name("harmbench_test", t) for t in cfg.evaluation.jailbreaks]
    return jobs


def judge_tasks(
    cfg: Config, node: Node, jobs: list[str] | None = None
) -> list[tuple[str, str, str]]:
    """(generation job, judge task, judge key) triples this node needs."""
    out = []
    primary = cfg.judging.primary
    secondary = cfg.secondary_judge_for(node.model)
    for job in jobs or eval_jobs(cfg, node):
        bench = job.split("@")[0]
        if job.endswith("@robust") or jailbreaks.template_of(job):
            out.append((job, "strongreject", primary))
        elif bench in STRONGREJECT_BENCHES:
            out += [(job, "strongreject", primary), (job, "strongreject", secondary)]
        elif bench == "xstest_safe":
            out.append((job, "xstest", cfg.judging.refusal_judge))
    return out


def gpu_stages(cfg: Config, node: Node, is_generator: bool) -> list[str]:
    """GPU stage markers this node needs (in order), excluding training."""
    ev = cfg.evaluation
    stages = [f"gen:{job}" for job in eval_jobs(cfg, node)]
    if is_generator:
        stages.append("gen:synthetic")
    if ev.heldout_nll:
        stages.append("heldout_nll")
    if ev.tail_loss and is_generator:
        stages.append("tail_loss")
    if ev.capability_tasks:
        stages.append("capability")
    if ev.refusal_direction:
        stages.append("refusal_direction")
    return stages


def generator_nodes(cfg: Config, nodes: list[Node]) -> set[Node]:
    """Nodes whose synthetic data we generate (none for published checkpoints)."""
    return {n for n in generators(nodes) if not cfg.model(n.model).published}


def pending(cfg: Config, layout: Layout, node: Node, is_generator: bool) -> dict[str, list[str]]:
    """Pending stages by kind. "train" holds the weights stage: "train", or "fetch" for a
    published checkpoint (see plan.weights_stage)."""
    gpu = [s for s in gpu_stages(cfg, node, is_generator) if not layout.done(node, s)]
    judge = [f"judge:{j}:{t}:{k}" for j, t, k in judge_tasks(cfg, node)]
    weights = weights_stage(cfg, node)
    return {
        "data": [] if weights != "train" or layout.done(node, "data") else ["data"],
        "train": [] if weights is None or layout.done(node, weights) else [weights],
        "gpu": gpu,
        "judge": [s for s in judge if not layout.done(node, s)],
    }


# --------------------------------------------------------------------------------------------
# Orchestration


class Runner:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.layout = Layout(cfg)
        self.layout.root.mkdir(parents=True, exist_ok=True)
        self.config_path = self.layout.root / "config.yaml"
        self._check_protocol()
        save_resolved(cfg, self.config_path)

    def run(self, models=None, seeds=None, judge=True, gpus=None, classify=True) -> None:
        """Run every pending node. With `gpus`, start one worker per GPU; workers share the
        folder, take whichever node is ready (its sources' synthetic data exists) and not locked
        by another worker, so seeds, arms and models run in parallel."""
        if gpus and len(gpus) > 1:
            self._spawn_workers(models, seeds, judge, gpus)
            if judge:  # anything the workers' judging threads did not finish
                for node in build_plan(self.cfg, models, seeds):
                    self._judge_safely(node)
        else:
            self.work(models, seeds, judge)
        if classify and self.cfg.evaluation.harmbench_classifier:
            for model in models or self.cfg.models:
                if self.cfg.model(model).family != "llama":
                    self.subprocess("classify", gen0(model))
        if "sorry_bench" in self.cfg.evaluation.benchmarks:
            for model in models or self.cfg.models:
                self.subprocess("sorry_judge", gen0(model))

    def work(self, models=None, seeds=None, judge=True, poll_s=60, final_judge=True) -> None:
        """One worker: repeatedly run the first ready, unlocked node until nothing is left."""
        judging = ThreadPoolExecutor(max_workers=1) if judge else None
        futures: list[Future] = []
        # Copies finished checkpoints to the archive while the GPU moves on.
        archiving = ThreadPoolExecutor(max_workers=1) if self.cfg.paths.archive_dir else None
        archived: list[Future] = []
        nodes = build_plan(self.cfg, models, seeds)
        gens = generator_nodes(self.cfg, nodes)
        while True:
            node, waiting = self.next_node(nodes, gens)
            if node is None:
                if not waiting:
                    break
                time.sleep(poll_s)  # other workers are producing what we need
                continue
            try:
                self.run_node(node, node in gens)
            finally:
                self.layout.unlock(node)
            if judging:
                futures.append(judging.submit(self._judge_safely, node))
            if archiving and weights_stage(self.cfg, node) == "train":
                archived.append(archiving.submit(self._archive_and_free, node, node in gens))
        if archiving:
            for future in archived:
                future.result()
            archiving.shutdown()
        if judging:
            if final_judge:  # nodes finished in an earlier session may still need judging
                futures += [judging.submit(self._judge_safely, n) for n in nodes]
            for future in futures:
                future.result()
            judging.shutdown()

    def next_node(self, nodes: list[Node], gens: set[Node]) -> tuple[Node | None, bool]:
        """(first runnable node, now locked by us), or (None, whether anything is still pending)."""
        waiting = False
        for node in nodes:
            todo = pending(self.cfg, self.layout, node, node in gens)
            if not (todo["data"] or todo["train"] or todo["gpu"]):
                continue
            if not ready(self.layout, node) or not self.layout.try_lock(node):
                waiting = True
                continue
            return node, False
        return None, waiting

    def _spawn_workers(self, models, seeds, judge, gpus) -> None:
        base = [
            sys.executable,
            "-m",
            "safecollapse.cli",
            "run",
            "-c",
            str(self.config_path),
            "--worker",
        ]
        for flag, values in (("--model", models or []), ("--seed", seeds or [])):
            for value in values:
                base += [flag, str(value)]
        if not judge:
            base.append("--no-judge")
        procs = []
        for gpu in gpus:
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
            log.info("starting worker on GPU %s", gpu)
            procs.append(subprocess.Popen(base, env=env))
            time.sleep(5)  # stagger start-up so workers pick different nodes
        failed = [p.args for p in procs if p.wait() != 0]
        if failed:
            raise RuntimeError(f"{len(failed)} worker(s) failed; rerun to resume")

    def run_node(self, node: Node, is_generator: bool) -> None:
        todo = pending(self.cfg, self.layout, node, is_generator)
        if not (todo["data"] or todo["train"] or todo["gpu"]):
            return
        log.info("node %s (%s): %s", node.key, node.label, todo)
        if todo["data"]:
            build_training_data(self.cfg, node)
        weights = weights_stage(self.cfg, node)
        if weights == "fetch" and (
            todo["train"] or (todo["gpu"] and not self.layout.checkpoint(node).exists())
        ):
            self.subprocess("fetch", node)
        if weights == "train" and todo["gpu"] and not todo["train"]:
            restore_checkpoint(self.layout, node)  # local copy lost (e.g. pod restart)?
        if (
            weights == "train"
            and todo["gpu"]
            and (todo["train"] or not self.layout.checkpoint(node).exists())
        ):
            if not todo["train"]:
                log.warning("%s: checkpoint was deleted but is needed again; retraining", node.key)
            self.subprocess("train", node)
        flag = "--generator" if is_generator else "--not-generator"
        for part in gpu_parts_pending(self.cfg, self.layout, node, is_generator):
            self.subprocess("gpu", node, flag, "--part", part)
        self._cleanup(node)

    def subprocess(self, stage: str, node: Node, *extra: str) -> None:
        args = [
            "-m",
            "safecollapse.cli",
            "stage",
            stage,
            "--resolved",
            str(self.config_path),
            "--node",
            node.key,
            *extra,
        ]
        launcher = list(self.cfg.training.launcher) if stage == "train" else []
        cmd = launcher + args if launcher else [sys.executable, *args]
        log.info("running: %s", " ".join(cmd))
        subprocess.run(cmd, check=True)

    def _judge_safely(self, node: Node) -> None:
        try:
            judge_node(self.cfg, node)
        except Exception:  # judging must never kill the GPU run; a rerun picks it up
            log.exception("judging %s failed; rerun `safecollapse judge` later", node.key)

    def _archive_and_free(self, node: Node, is_generator: bool) -> None:
        """Archive a finished checkpoint, then free the fast local disk: with an archive, the
        local copy is only a cache (restored automatically if a later stage needs it)."""
        archive_checkpoint(self.layout, node)
        pending_now = pending(self.cfg, self.layout, node, is_generator)
        if self.layout.done(node, "archived") and not (pending_now["train"] or pending_now["gpu"]):
            self.layout.delete_checkpoint(node)

    def _cleanup(self, node: Node) -> None:
        """Delete a checkpoint once everything that needs it is done (disk is the bottleneck)."""
        if weights_stage(self.cfg, node) == "fetch":  # public weights: fetched again if needed
            if not any(pending(self.cfg, self.layout, node, False)[k] for k in ("train", "gpu")):
                self.layout.delete_checkpoint(node)
            return
        if self.cfg.keep_checkpoints or not node.trained or node.gen >= self.cfg.generations:
            return
        if self.cfg.paths.archive_dir and not self.layout.done(node, "archived"):
            return  # never delete the only copy
        nodes = build_plan(self.cfg, [node.model], [node.seed])
        is_gen = node in generator_nodes(self.cfg, nodes)
        if not any(pending(self.cfg, self.layout, node, is_gen)[k] for k in ("train", "gpu")):
            self.layout.delete_checkpoint(node)

    def _check_protocol(self) -> None:
        """Refuse to mix results made under different protocols in one study folder."""
        path = self.layout.root / "protocol.json"
        current = protocol(self.cfg)
        if path.exists():
            previous = read_json(path)
            if previous != current:
                changed = sorted(k for k in current if current.get(k) != previous.get(k))
                raise RuntimeError(
                    f"{self.layout.root} was produced under a different protocol (changed: "
                    f"{changed}). Use a new `name` for a new protocol."
                )
        else:
            write_json(path, current)


def archive_checkpoint(layout: Layout, node: Node) -> None:
    """Copy a finished checkpoint to persistent storage (temp name, then rename: a copy cut short
    by a crash is never mistaken for a complete one)."""
    src, dst = layout.checkpoint(node), layout.archived_checkpoint(node)
    if dst is None or layout.done(node, "archived") or not src.exists():
        return
    tmp = dst.with_name(dst.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.parent.mkdir(parents=True, exist_ok=True)
    _copy_tree(src, tmp)
    shutil.rmtree(dst, ignore_errors=True)
    tmp.rename(dst)
    layout.mark_done(node, "archived")
    log.info("%s: checkpoint archived to %s", node.key, dst)


def restore_checkpoint(layout: Layout, node: Node) -> None:
    """Bring an archived checkpoint back to the local disk instead of retraining it."""
    src, dst = layout.archived_checkpoint(node), layout.checkpoint(node)
    if dst.exists() or src is None or not src.exists():
        return
    log.info("%s: restoring checkpoint from the archive", node.key)
    tmp = dst.with_name(dst.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.parent.mkdir(parents=True, exist_ok=True)
    _copy_tree(src, tmp)
    tmp.rename(dst)


def _copy_tree(src: Path, dst: Path) -> None:
    """Copy in a separate `cp` process. On a network filesystem, a file this process holds open
    for writing would be flushed by every child process we start (each closes the inherited
    descriptor), stalling the GPU stages behind a multi-gigabyte upload.

    Only timestamps are preserved: `cp -a` exits non-zero when it cannot keep the owner and mode
    of files on RunPod's network volume, even though every byte was copied. A size check catches
    a copy that is really incomplete."""
    subprocess.run(["cp", "-r", "--preserve=timestamps", str(src), str(dst)], check=True)
    want = {p.relative_to(src): p.stat().st_size for p in src.rglob("*") if p.is_file()}
    got = {p.relative_to(dst): p.stat().st_size for p in dst.rglob("*") if p.is_file()}
    if want != got:
        raise RuntimeError(f"incomplete copy of {src} to {dst}")


def ready(layout: Layout, node: Node) -> bool:
    """A node can start once every node it trains on has written its synthetic data."""
    if layout.cfg.model(node.model).published:
        return True
    return all(layout.done(src, "gen:synthetic") for src in sources(node))


def protocol(cfg: Config) -> dict:
    """The settings that define what a checkpoint *is*; arms, seeds and generations may grow."""
    d = cfg.to_dict()
    return {
        "data": d["data"],
        # Unset optional fields are left out, so adding one never changes an existing protocol.
        "synthetic": {k: v for k, v in d["synthetic"].items() if v is not None},
        # Throughput settings (micro-batch, checkpointing, launcher) do not change the recipe.
        "training": {k: v for k, v in d["training"].items() if k not in THROUGHPUT_KEYS},
        "decoding": {k: v for k, v in d["evaluation"]["decoding"].items() if v is not None},
        # Only the studied models: adding or fixing other registry entries (e.g. a judge model's
        # context length) does not change what a checkpoint is.
        # Unset optional fields are left out, so adding one never changes an existing protocol.
        "models": {
            k: {f: v for f, v in d["registry"]["models"][k].items() if v is not None}
            for k in cfg.models
        },
        # Only the judges this study's judging uses, without unset optional fields, so adding a
        # judge to the registry (or an optional field) never changes an existing protocol.
        "judges": {
            k: {f: v for f, v in d["registry"]["judges"][k].items() if v is not None}
            for k in sorted(
                {
                    cfg.judging.primary,
                    cfg.judging.refusal_judge,
                    *cfg.judging.secondary_by_family.values(),
                }
            )
        },
        "judging": {
            k: d["judging"][k] for k in ("primary", "secondary_by_family", "refusal_judge")
        },
    }


# --------------------------------------------------------------------------------------------
# CPU stages


def build_training_data(cfg: Config, node: Node) -> None:
    layout = Layout(cfg)
    original = read_jsonl(layout.round_file(node.seed))
    synthetic = {}
    for src in sources(node):
        answers = {
            r["id"]: r["completion"]
            for r in read_jsonl(layout.synthetic_file(src))
            if r["sample"] == 0
        }
        synthetic[src.gen] = arms.with_answers(original, answers, src.label)
    data = arms.training_set(node.arm, node.gen, original, synthetic, node.seed)
    path = write_jsonl(layout.train_file(node), data)
    update_manifest(
        layout.manifest(node),
        "data",
        {
            "train_file_sha256": sha256_file(path),
            "examples": len(data),
            "by_source": dict(Counter(ex["source"] for ex in data)),
            "sources": [s.key for s in sources(node)],
        },
    )
    layout.mark_done(node, "data")


def judge_node(cfg: Config, node: Node, only_jobs: list[str] | None = None) -> dict:
    """Judge every finished evaluation job of this node that is not judged yet."""
    from .judges.openrouter import OpenRouter
    from .judges.runner import JudgeCache, judge_items
    from .metrics import safety

    layout = Layout(cfg)
    client = None
    summaries = {}
    tasks = {}
    for job, task, judge_key in judge_tasks(cfg, node, only_jobs):
        marker = f"judge:{job}:{task}:{judge_key}"
        if layout.done(node, marker) or not layout.done(node, f"gen:{job}"):
            continue
        bench = job.split("@")[0]
        prompts = {r["id"]: r for r in read_jsonl(layout.data_file(bench))}
        completions = read_jsonl(layout.completions(node, job))
        items = [
            {
                "id": _sample_id(c),
                "prompt": prompts[c["id"]]["messages"][-1]["content"],
                "response": c["completion"],
            }
            for c in completions
        ]
        client = client or OpenRouter(
            timeout_s=cfg.judging.timeout_s, max_retries=cfg.judging.max_retries
        )
        cache = JudgeCache(layout.cache_dir / f"{judge_key}.jsonl")
        if task not in tasks:
            tasks[task] = judge_task(cfg, task)
        verdicts = judge_items(
            items,
            tasks[task],
            judge_key,
            cfg.judge(judge_key),
            client,
            cache,
            cfg.judging.concurrency,
            cfg.judging.max_tokens,
        )
        write_jsonl(layout.judgments(node, job, task, judge_key), verdicts)
        texts = {_sample_id(c): c["completion"] for c in completions}
        if task == "xstest":
            summary = safety.summarize_refusal(verdicts)
        else:
            summary = safety.summarize_strongreject(verdicts, texts)
        summary["providers"] = dict(Counter(str(v.get("provider")) for v in verdicts))
        summaries[marker] = summary
        write_json(layout.metrics_file(node, f"{job}__{task}__{judge_key}"), summary)
        update_manifest(
            layout.manifest(node),
            marker,
            {
                "judge_model": cfg.judge(judge_key).model,
                "provider_pin": list(cfg.judge(judge_key).provider_order),
                "providers_served": summary["providers"],
                "errors": summary["n_errors"],
            },
        )
        if summary["n_errors"] == 0:
            layout.mark_done(node, marker)
        else:
            log.warning(
                "%s %s: %d judge errors; rerun to retry them", node.key, marker, summary["n_errors"]
            )
    return summaries


def judge_task(cfg: Config, name: str):
    """The Task object for an API judge task name."""
    from .judges import official, rubric

    if name == "strongreject":
        return rubric.STRONGREJECT
    if name == "xstest":
        return official.xstest_task(cfg.data.xstest_commit, Layout(cfg).data_dir / "raw")
    raise ValueError(f"unknown judge task {name!r}")


def _sample_id(row: dict) -> str:
    """Robustness samples get '#k' suffixes; the greedy answer keeps the plain prompt id."""
    return row["id"] if row.get("sample", 0) == 0 else f"{row['id']}#{row['sample']}"


# --------------------------------------------------------------------------------------------
# GPU stages (each runs inside `safecollapse stage ...`, a fresh process)


def stage_train(cfg: Config, node: Node) -> None:
    from .modeling import require_gpu
    from .training import train

    require_gpu()
    layout = Layout(cfg)
    spec = cfg.model(node.model)
    stats = train(
        spec.hf_id,
        spec.revision,
        spec,
        layout.train_file(node),
        layout.checkpoint(node),
        cfg.training,
        seed=node.seed,
    )
    update_manifest(
        layout.manifest(node),
        "train",
        {
            "base": f"{spec.hf_id}@{spec.revision}",
            "train_file_sha256": sha256_file(layout.train_file(node)),
            **stats,
        },
    )
    layout.mark_done(node, "train")


def stage_fetch(cfg: Config, node: Node) -> None:
    """Copy a published checkpoint (or Gen 0) to the local checkpoint folder. With
    `chat_template_from`, the tokenizer gets that chat template, so prompts are formatted exactly
    as for the other models; the source and revision are recorded in the manifest."""
    from huggingface_hub import HfApi, snapshot_download

    layout = Layout(cfg)
    spec = cfg.model(node.model)
    repo, revision = published_ref(cfg, node)
    revision = revision or HfApi().model_info(repo).sha
    dst = layout.checkpoint(node)
    tmp = dst.with_name(dst.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    snapshot_download(
        repo,
        revision=revision,
        local_dir=str(tmp),
        allow_patterns=["*.json", "*.safetensors", "tokenizer.model", "*.jinja"],
    )
    info = {"source": f"{repo}@{revision}"}
    if spec.chat_template_from:
        from transformers import AutoTokenizer

        src, _, src_revision = spec.chat_template_from.partition("@")
        tok = AutoTokenizer.from_pretrained(str(tmp))
        info["had_chat_template"] = bool(tok.chat_template)
        template = AutoTokenizer.from_pretrained(src, revision=src_revision or None).chat_template
        tok.chat_template = template
        tok.save_pretrained(str(tmp))
        info["chat_template_from"] = spec.chat_template_from
    shutil.rmtree(dst, ignore_errors=True)
    tmp.rename(dst)
    update_manifest(layout.manifest(node), "fetch", info)
    layout.mark_done(node, "fetch")


# GPU work for one node runs in three separate processes, so each starts on an empty GPU: a
# vLLM engine or a PyTorch cache left by one step must never starve the next (on the main run,
# IFEval's vLLM failed to start next to the tail-loss pass's cached memory).
GPU_PARTS = {
    "generate": lambda s: s.startswith("gen:"),
    "metrics": lambda s: s in ("heldout_nll", "tail_loss"),
    "post": lambda s: s in ("capability", "refusal_direction"),
}


def gpu_parts_pending(cfg: Config, layout: Layout, node: Node, is_generator: bool) -> list[str]:
    todo = [s for s in gpu_stages(cfg, node, is_generator) if not layout.done(node, s)]
    return [part for part, match in GPU_PARTS.items() if any(match(s) for s in todo)]


def stage_gpu(cfg: Config, node: Node, is_generator: bool, part: str) -> None:
    """One part of a node's pending GPU work (see GPU_PARTS)."""
    from .modeling import require_gpu

    require_gpu()
    layout = Layout(cfg)
    todo = [s for s in gpu_stages(cfg, node, is_generator) if not layout.done(node, s)]
    todo = [s for s in todo if GPU_PARTS[part](s)]
    if part == "generate":
        _generate(cfg, layout, node, [s.split(":", 1)[1] for s in todo])
    elif part == "metrics":
        _collapse_metrics(cfg, layout, node, todo)
    elif part == "post":
        if "capability" in todo:
            _capability(cfg, layout, node)
        if "refusal_direction" in todo:
            _refusal_direction(cfg, layout, node)


def generation_jobs(cfg: Config, layout: Layout, node: Node, names: list[str]):
    from .generation import Job

    ev = cfg.evaluation
    jobs = []
    for name in names:
        if name == "synthetic":
            records = read_jsonl(layout.round_file(node.seed))
            jobs.append(
                Job(f"synthetic/{node.label}", records, cfg.synthetic, layout.synthetic_file(node))
            )
            continue
        bench = name.split("@")[0]
        records = read_jsonl(layout.data_file(bench))[: ev.limit]
        template = jailbreaks.template_of(name)
        if template:
            wrapped = [jailbreaks.wrap_record(r, template) for r in records]
            jobs.append(Job(name, wrapped, ev.decoding, layout.completions(node, name)))
        elif name.endswith("@robust"):
            sampling = Sampling(ev.robustness_temperature, 1.0, ev.decoding.max_new_tokens)
            jobs.append(
                Job(
                    name, records, sampling, layout.completions(node, name), n=ev.robustness_samples
                )
            )
        else:
            jobs.append(Job(name, records, ev.decoding, layout.completions(node, name)))
    return jobs


def _generate(cfg: Config, layout: Layout, node: Node, names: list[str]) -> None:
    from .generation import run_jobs
    from .metrics.collapse import audit_synthetic

    path, revision = layout.model_ref(node)
    jobs = generation_jobs(cfg, layout, node, names)
    stats = run_jobs(
        path, revision, cfg.model(node.model), jobs, cfg.generation, seed=node.seed or 0
    )
    for name, job in zip(names, jobs):
        if name == "synthetic":
            audit = audit_synthetic(read_jsonl(job.out))
            if cfg.evaluation.diversity_embeddings:
                from .metrics.collapse import embedding_diversity

                audit["embedding_diversity"] = embedding_diversity(
                    [r["completion"] for r in read_jsonl(job.out)]
                )
            write_json(layout.metrics_file(node, "synthetic_audit"), audit)
        update_manifest(
            layout.manifest(node),
            f"gen:{name}",
            {
                "model": f"{path}@{revision}" if revision else path,
                "sampling": dataclasses.asdict(job.sampling),
                "n_samples": job.n,
                "output_sha256": sha256_file(job.out),
                **stats[job.name],
            },
        )
        layout.mark_done(node, f"gen:{name}")


def _collapse_metrics(cfg: Config, layout: Layout, node: Node, todo: list[str]) -> None:
    import gc

    import torch

    from .metrics import collapse
    from .modeling import load_model, load_tokenizer

    max_len = cfg.training.max_seq_len
    spec = cfg.model(node.model)
    if "heldout_nll" in todo:
        path, revision = layout.model_ref(node)
        model, tok = load_model(path, revision, spec), load_tokenizer(path, revision)
        rows = collapse.response_nll_with_model(
            model, tok, read_jsonl(layout.heldout_file()), max_len
        )
        write_json(layout.metrics_file(node, "heldout_nll"), collapse.corpus_nll(rows))
        layout.mark_done(node, "heldout_nll")
        del model
        gc.collect()
        torch.cuda.empty_cache()
    if "tail_loss" in todo:
        # Tail loss always scores with Gen 0 (Shumailov et al., Figure 1).
        model = load_model(spec.hf_id, spec.revision, spec)
        tok = load_tokenizer(spec.hf_id, spec.revision)
        original = read_jsonl(layout.round_file(node.seed))
        ref_path = layout.tail_reference(node.model, node.seed)
        if not ref_path.exists():
            ref = collapse.per_example_perplexity(
                collapse.response_nll_with_model(model, tok, original, max_len)
            )
            write_json(
                ref_path,
                {**collapse.tail_summary(ref, float(_p95(ref))), "per_example": ref.tolist()},
            )
        reference_p95 = read_json(ref_path)["quantiles"]["p95"]
        answers = {
            r["id"]: r["completion"]
            for r in read_jsonl(layout.synthetic_file(node))
            if r["sample"] == 0
        }
        synthetic = arms.with_answers(original, answers, node.label)
        ppl = collapse.per_example_perplexity(
            collapse.response_nll_with_model(model, tok, synthetic, max_len)
        )
        write_json(
            layout.metrics_file(node, "tail_loss"),
            {**collapse.tail_summary(ppl, reference_p95), "per_example": ppl.tolist()},
        )
        layout.mark_done(node, "tail_loss")


def _p95(values) -> float:
    import numpy as np

    return float(np.percentile(values, 95))


def _capability(cfg: Config, layout: Layout, node: Node) -> None:
    from .metrics.capability import run_lm_eval

    path, revision = layout.model_ref(node)
    results = run_lm_eval(
        path,
        revision,
        cfg.evaluation.capability_tasks,
        cfg.evaluation.capability_limit,
        layout.node_dir(node) / "capability",
        seed=node.seed or 0,
        max_model_len=cfg.model(node.model).max_model_len,
    )
    write_json(layout.metrics_file(node, "capability"), results)
    layout.mark_done(node, "capability")


def _refusal_direction(cfg: Config, layout: Layout, node: Node) -> None:
    import numpy as np

    from .metrics import refusal_direction as rd
    from .modeling import load_model, load_tokenizer

    harmful = [r["messages"] for r in read_jsonl(layout.data_file("harmbench_val"))]
    harmless = [
        r["messages"] for r in read_jsonl(layout.heldout_file()) if len(r["messages"]) == 1
    ][: len(harmful)]
    spec = cfg.model(node.model)
    dirs_path = layout.root / node.model / "refusal_direction_gen0.npy"
    if not dirs_path.exists():
        model, tok = (
            load_model(spec.hf_id, spec.revision, spec),
            load_tokenizer(spec.hf_id, spec.revision),
        )
        dirs = rd.directions(
            rd.last_token_activations(model, tok, harmful),
            rd.last_token_activations(model, tok, harmless),
        )
        dirs_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(dirs_path, dirs)
        del model
    dirs = np.load(dirs_path)
    path, revision = layout.model_ref(node)
    model, tok = load_model(path, revision, spec), load_tokenizer(path, revision)
    summary = rd.summarize(
        rd.last_token_activations(model, tok, harmful),
        rd.last_token_activations(model, tok, harmless),
        dirs,
    )
    write_json(layout.metrics_file(node, "refusal_direction"), summary)
    layout.mark_done(node, "refusal_direction")


def stage_classify(cfg: Config, model_key: str) -> None:
    """HarmBench classifier over every node of one model with finished HarmBench-test answers."""
    from vllm import LLM

    from .judges import harmbench_cls
    from .modeling import require_gpu

    require_gpu()
    layout = Layout(cfg)
    spec = cfg.model("harmbench_cls")
    todo = [
        n
        for n in build_plan(cfg, [model_key])
        if layout.done(n, "gen:harmbench_test") and not layout.done(n, "classify")
    ]
    if not todo:
        return
    templates = harmbench_cls.load_templates(spec.hf_id, spec.revision, layout.data_dir / "raw")
    prompts = {r["id"]: r for r in read_jsonl(layout.data_file("harmbench_test"))}
    llm = LLM(
        model=spec.hf_id,
        revision=spec.revision,
        dtype="bfloat16",
        max_model_len=spec.max_model_len,
        gpu_memory_utilization=cfg.generation.gpu_memory_utilization,
    )
    for node in todo:
        completions = read_jsonl(layout.completions(node, "harmbench_test"))
        labels = harmbench_cls.classify_completions(
            llm,
            templates,
            [prompts[c["id"]]["meta"] for c in completions],
            [c["completion"] for c in completions],
            spec.max_model_len,
        )
        rows = [{"id": c["id"], "harmful": lab} for c, lab in zip(completions, labels)]
        write_jsonl(layout.judgments(node, "harmbench_test", "classifier", "harmbench_cls"), rows)
        valid = [r["harmful"] for r in rows if r["harmful"] is not None]
        write_json(
            layout.metrics_file(node, "harmbench_test__classifier__harmbench_cls"),
            {
                "n": len(rows),
                "n_unparsed": len(rows) - len(valid),
                "asr": sum(valid) / len(valid) if valid else None,
            },
        )
        layout.mark_done(node, "classify")


def stage_sorry_judge(cfg: Config, model_key: str) -> None:
    """SORRY-Bench's own fine-tuned judge over every node of one model with finished answers."""
    from vllm import LLM

    from .judges import official
    from .metrics import safety
    from .modeling import require_gpu

    require_gpu()
    layout = Layout(cfg)
    todo = [
        n
        for n in build_plan(cfg, [model_key])
        if layout.done(n, "gen:sorry_bench") and not layout.done(n, "sorry_judge")
    ]
    if not todo:
        return
    spec = cfg.model("sorry_judge")
    template = official.sorry_template(cfg.data.sorry_bench_commit, layout.data_dir / "raw")
    prompts = {r["id"]: r for r in read_jsonl(layout.data_file("sorry_bench"))}
    categories = {pid: p["meta"]["category"] for pid, p in prompts.items()}
    llm = LLM(
        model=spec.hf_id,
        revision=spec.revision,
        dtype="bfloat16",
        max_model_len=spec.max_model_len,
        gpu_memory_utilization=cfg.generation.gpu_memory_utilization,
    )
    for node in todo:
        completions = read_jsonl(layout.completions(node, "sorry_bench"))
        items = [
            {
                "id": c["id"],
                "prompt": prompts[c["id"]]["messages"][-1]["content"],
                "response": c["completion"],
            }
            for c in completions
        ]
        rows = official.run_sorry_judge(llm, template, items)
        write_jsonl(layout.judgments(node, "sorry_bench", "fulfillment", "sorry_judge"), rows)
        write_json(
            layout.metrics_file(node, "sorry_bench__fulfillment__sorry_judge"),
            safety.fulfillment_by_category(rows, categories),
        )
        layout.mark_done(node, "sorry_judge")


# --------------------------------------------------------------------------------------------


def status(cfg: Config) -> list[dict]:
    """One row per node: which stages are done and which are pending."""
    layout = Layout(cfg)
    nodes = build_plan(cfg)
    gens = generator_nodes(cfg, nodes)
    rows = []
    for node in nodes:
        todo = pending(cfg, layout, node, node in gens)
        rows.append(
            {
                "node": node.key,
                "label": node.label,
                "generator": node in gens,
                "checkpoint": layout.checkpoint(node).exists() if node.trained else None,
                "locked": layout.locked(node),
                "pending": [s for group in todo.values() for s in group],
            }
        )
    return rows


def load_resolved(path: str | Path) -> Config:
    return load_config(Path(path))
