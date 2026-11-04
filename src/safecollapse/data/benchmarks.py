"""Safety prompt sets (DESIGN.md Section 5.1), converted to one record format:

    {"id", "benchmark", "messages": [{"role": "user", "content": ...}], "meta": {...}}

`meta` keeps what the judges need (HarmBench behaviour and context, SORRY-Bench category).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

# Copyright behaviours are scored by hashing, not by a judge, so they are excluded.
HARMBENCH_KINDS = ("standard", "contextual")
EXPECTED = {"harmbench_val": 60, "harmbench_test": 240, "xstest_safe": 250}


def harmbench_prompt(behavior: str, context: str | None) -> str:
    """HarmBench's own format for contextual behaviours."""
    return f"{context}\n\n---\n\n{behavior}" if context else behavior


def load_harmbench(csv_path: Path, split: str) -> list[dict]:
    with open(csv_path, encoding="utf-8", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["FunctionalCategory"] in HARMBENCH_KINDS]
    records = []
    for row in rows:
        context = row["ContextString"].strip() or None
        records.append(
            {
                "id": row["BehaviorID"],
                "benchmark": f"harmbench_{split}",
                "messages": [
                    {"role": "user", "content": harmbench_prompt(row["Behavior"], context)}
                ],
                "meta": {
                    "behavior": row["Behavior"],
                    "context": context,
                    "kind": row["FunctionalCategory"],
                    "category": row["SemanticCategory"],
                },
            }
        )
    return records


def load_sorry_bench(jsonl_path: Path) -> list[dict]:
    records = []
    with open(jsonl_path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            # Releases differ: older ones store the text under "turns", some under "prompt".
            text = row["turns"][0] if "turns" in row else row["prompt"]
            records.append(
                {
                    "id": f"sorry-{row['question_id']}",
                    "benchmark": "sorry_bench",
                    "messages": [{"role": "user", "content": text}],
                    "meta": {"category": str(row.get("category", ""))},
                }
            )
    return records


def load_xstest_safe(csv_path: Path) -> list[dict]:
    with open(csv_path, encoding="utf-8", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["label"] == "safe"]
    return [
        {
            "id": f"xstest-{row['id']}",
            "benchmark": "xstest_safe",
            "messages": [{"role": "user", "content": row["prompt"]}],
            "meta": {"type": row["type"]},
        }
        for row in rows
    ]


def check_count(name: str, records: list[dict]) -> None:
    expected = EXPECTED.get(name)
    if expected is not None and len(records) != expected:
        raise ValueError(f"{name}: expected {expected} prompts, found {len(records)}")
