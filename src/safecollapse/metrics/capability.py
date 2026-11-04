"""Capability check with EleutherAI's lm-evaluation-harness (needs the [capability] extra).

Separates "less safe" from "worse overall" (DESIGN.md Section 5.1). The tasks come from the
config (proposed: IFEval). Uses lm-eval's vLLM backend: fast, and it handles Gemma-2's
soft-capping, which the transformers backend would need eager attention for.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def run_lm_eval(
    model_path: str,
    revision: str | None,
    tasks: tuple[str, ...],
    limit: int | None,
    out_dir: Path,
    seed: int,
    max_model_len: int = 4096,
) -> dict:
    """Run lm-eval in a subprocess (clean GPU state) and return its per-task results."""
    model_args = (
        f"pretrained={model_path},dtype=bfloat16,gpu_memory_utilization=0.8,"
        f"max_model_len={max_model_len}"
    )
    if revision:
        model_args += f",revision={revision}"
    cmd = [
        sys.executable,
        "-m",
        "lm_eval",
        "--model",
        "vllm",
        "--model_args",
        model_args,
        "--tasks",
        ",".join(tasks),
        "--batch_size",
        "auto",
        "--apply_chat_template",
        "--seed",
        str(seed),
        "--output_path",
        str(out_dir),
    ]
    if limit:
        cmd += ["--limit", str(limit)]
    subprocess.run(cmd, check=True)
    results = sorted(out_dir.rglob("results_*.json"), key=lambda p: p.stat().st_mtime)
    if not results:
        raise RuntimeError(f"lm-eval wrote no results under {out_dir}")
    return json.loads(results[-1].read_text())["results"]
