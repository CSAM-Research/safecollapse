"""Gather every stored verdict into one long table: the single source for all tables and
figures (DESIGN.md Section 11: "every table regenerated from stored per-prompt scores")."""

from __future__ import annotations

import pandas as pd

from ..config import Config
from ..io import read_json, read_jsonl
from ..pipeline import judge_tasks
from ..plan import Layout, build_plan

# Benchmark-specific local judges (GPU stages): (job, task, judge) and the field holding the label.
LOCAL_JUDGES = [
    ("harmbench_test", "classifier", "harmbench_cls"),
    ("sorry_bench", "fulfillment", "sorry_judge"),
    ("harmbench_test", "jev", "jev"),  # exploratory third judge (DECISIONS.md B16)
]
LOCAL_SCORE = {"classifier": "harmful", "fulfillment": "fulfilled", "jev": "unsafe"}

COLUMNS = [
    "model",
    "seed",
    "arm",
    "generation",
    "label",
    "job",
    "task",
    "judge",
    "prompt",
    "sample",
    "score",
    "refused",
    "specific",
    "convincing",
    "refusal_label",
    "judge_refused",
    "error",
]


def scores_long(cfg: Config) -> pd.DataFrame:
    layout = Layout(cfg)
    rows = []
    for node in build_plan(cfg):
        for job, task, judge in judge_tasks(cfg, node) + LOCAL_JUDGES:
            path = layout.judgments(node, job, task, judge)
            if not path.exists():
                continue
            for v in read_jsonl(path):
                if task in LOCAL_SCORE:  # local judges store a 0/1 label; it becomes the score
                    v = {**v, "score": v.get(LOCAL_SCORE[task])}
                    if v["score"] is None:
                        v["error"] = "unparsed judge output"
                prompt, _, sample = v["id"].partition("#")
                rows.append(
                    {
                        "model": node.model,
                        "seed": node.seed,
                        "arm": node.arm,
                        "generation": node.gen,
                        "label": node.label,
                        "job": job,
                        "task": task,
                        "judge": judge,
                        "prompt": prompt,
                        "sample": int(sample or 0),
                        "score": v.get("score"),
                        "refused": v.get("refused"),
                        "specific": v.get("specific"),
                        "convincing": v.get("convincing"),
                        "refusal_label": v.get("label"),
                        "judge_refused": v.get("judge_refused", 0),
                        "error": v.get("error"),
                    }
                )
    return pd.DataFrame(rows, columns=COLUMNS)


def node_metrics(cfg: Config) -> pd.DataFrame:
    """One row per node with its collapse metrics (held-out NLL, tail loss, synthetic audit)."""
    layout = Layout(cfg)
    rows = []
    for node in build_plan(cfg):
        row = {
            "model": node.model,
            "seed": node.seed,
            "arm": node.arm,
            "generation": node.gen,
            "label": node.label,
        }
        for name in ("heldout_nll", "tail_loss", "synthetic_audit", "capability"):
            path = layout.metrics_file(node, name)
            if path.exists():
                data = read_json(path)
                data.pop("per_example", None)
                for key, value in _flatten(data).items():
                    row[f"{name}.{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flatten(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out
