"""Full fine-tune from Gen 0 with the Kazdan recipe (DESIGN.md Section 3.4).

Runs as its own process (optionally under `accelerate launch` for several GPUs), so GPU memory
is fully released between generation and training.
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path

from .config import ModelSpec, TrainConfig
from .io import read_jsonl, write_json
from .modeling import encode_example, load_model, load_tokenizer, pad_batch


class _Dataset:
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def encode_file(tokenizer, path: Path, max_len: int):
    """Encode a training file; returns (encoded rows, counts for the manifest)."""
    rows, counts = [], {"examples": 0, "dropped_no_response": 0, "truncated": 0, "joined": 0}
    for ex in read_jsonl(path):
        counts["examples"] += 1
        enc = encode_example(tokenizer, ex["messages"], ex["response"], max_len)
        if enc is None:
            counts["dropped_no_response"] += 1
            continue
        counts["truncated"] += enc.truncated
        counts["joined"] += enc.joined
        rows.append(enc)
    counts["response_tokens"] = sum(sum(x != -100 for x in r.labels) for r in rows)
    return rows, counts


def train(
    base_path: str,
    base_revision: str | None,
    spec: ModelSpec,
    train_file: Path,
    out_dir: Path,
    tc: TrainConfig,
    seed: int,
    max_steps: int | None = None,
    save: bool = True,
) -> dict:
    """Fine-tune Gen 0 on `train_file`; save weights and tokenizer to `out_dir`.

    `max_steps` and `save=False` turn this into the pilot's timing run.
    """
    import torch
    from transformers import Trainer, TrainingArguments, set_seed

    set_seed(seed)
    world = int(os.environ.get("WORLD_SIZE", "1"))  # set by accelerate/torchrun
    per_step = tc.per_device_batch_size * world
    if tc.effective_batch % per_step:
        raise ValueError(f"effective_batch {tc.effective_batch} not divisible by {per_step}")
    accum = tc.effective_batch // per_step

    tokenizer = load_tokenizer(base_path, base_revision)
    rows, counts = encode_file(tokenizer, train_file, tc.max_seq_len)
    steps = max_steps or math.ceil(len(rows) / tc.effective_batch) * tc.epochs
    model = load_model(base_path, base_revision, spec, for_training=True)
    model.config.use_cache = False  # incompatible with gradient checkpointing

    args = TrainingArguments(
        output_dir=str(out_dir.parent / "trainer_tmp"),
        per_device_train_batch_size=tc.per_device_batch_size,
        gradient_accumulation_steps=accum,
        num_train_epochs=tc.epochs,
        max_steps=max_steps or -1,
        learning_rate=tc.learning_rate,
        lr_scheduler_type=tc.scheduler,
        # Explicit steps rather than warmup_ratio, which newer transformers versions deprecate.
        warmup_steps=math.ceil(tc.warmup_ratio * steps),
        optim="adamw_torch",
        weight_decay=tc.weight_decay,  # 0 = plain Adam, as on the Kazdan model cards
        adam_beta1=tc.adam_beta1,
        adam_beta2=tc.adam_beta2,
        adam_epsilon=tc.adam_eps,
        bf16=True,
        gradient_checkpointing=tc.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},  # the recommended, safe variant
        logging_steps=5,
        save_strategy="no",
        report_to=[],
        seed=seed,
        data_seed=seed,
        remove_unused_columns=False,
        dataloader_num_workers=2,
        **tc.trainer_overrides,
    )

    def collate(batch):
        ids, mask, labels = pad_batch(batch, tokenizer.pad_token_id, side="right")
        return {"input_ids": ids, "attention_mask": mask, "labels": labels}

    trainer = Trainer(model=model, args=args, train_dataset=_Dataset(rows), data_collator=collate)
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    result = trainer.train()
    elapsed = time.time() - start
    check_finite(trainer.state.log_history)

    stats = {
        **counts,
        "optimizer_steps": result.global_step,
        "grad_accum": accum,
        "world_size": world,
        "train_seconds": round(elapsed, 1),
        "seconds_per_step": round(elapsed / max(1, result.global_step), 2),
        "peak_memory_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1),
        "final_loss": result.training_loss,
    }
    if save:
        out_dir.mkdir(parents=True, exist_ok=True)
        if not _distributed():
            # Store in bfloat16, as Gen 0 is distributed: half the disk, same as evaluation uses.
            trainer.model.to(torch.bfloat16)
        trainer.save_model(str(out_dir))  # every rank calls it: FSDP gathers the shards
    if save and trainer.is_world_process_zero():
        tokenizer.save_pretrained(str(out_dir))
        write_json(out_dir / "train_log.json", {"stats": stats, "log": trainer.state.log_history})
    return stats


def check_finite(log_history: list[dict]) -> None:
    """Refuse a run whose loss or gradient norm was ever non-finite, or whose logged loss fell to
    exactly 0: the Trainer silently filters NaN/inf losses from its log, so a diverged run can
    otherwise finish "successfully" and save NaN weights (this happened to the first Qwen
    replication, DECISIONS.md D-incident)."""
    bad = []
    for entry in log_history:
        for key in ("loss", "grad_norm"):
            value = entry.get(key)
            if isinstance(value, (int, float)) and not math.isfinite(value):
                bad.append(f"step {entry.get('step')}: {key}={value}")
        if entry.get("loss") == 0.0:
            bad.append(f"step {entry.get('step')}: loss=0.0")
    if bad:
        raise RuntimeError("training diverged; checkpoint not saved: " + "; ".join(bad[:5]))


def _distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1
