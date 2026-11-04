"""Score many completions with one API judge: concurrent, cached on disk, never paid twice.

The cache is an append-only JSONL file per judge, keyed by a hash of (task, judge model,
provider pin, judge prompt). A rerun after a crash re-sends only the calls that never finished.
Calls that still fail after retries are stored with an `error` and reported, never imputed
(DESIGN.md Section 6).
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..config import JudgeSpec
from ..io import sha256_text
from .openrouter import JudgeError, OpenRouter
from .rubric import JudgeRefusal
from .tasks import Task


class JudgeCache:
    """Reads `<judge>.jsonl` and every `<judge>.<worker>.jsonl`; appends only to this process's
    own file, so parallel workers never write to the same file."""

    def __init__(self, path: Path):
        self.path = path.with_name(f"{path.stem}.{socket.gethostname()}-{os.getpid()}.jsonl")
        self._lock = threading.Lock()
        self._rows: dict[str, dict] = {}
        files = [path] + sorted(path.parent.glob(f"{path.stem}.*.jsonl"))
        for file in files:
            if not file.exists():
                continue
            with open(file, encoding="utf-8") as fh:
                for line in fh:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # a torn last line from a killed run
                    self._rows[row["key"]] = row

    def get(self, key: str) -> dict | None:
        return self._rows.get(key)

    def put(self, row: dict) -> None:
        with self._lock:
            self._rows[row["key"]] = row
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def cache_key(task: Task, judge: JudgeSpec, judge_prompt: str, salt: str = "") -> str:
    """`salt` gives a deliberate repeat call its own cache entry (test-retest checks)."""
    pin = ",".join(judge.provider_order)
    system = task.system or ""
    return sha256_text(
        f"{task.name}\x1f{judge.model}\x1f{pin}\x1f{system}\x1f{judge_prompt}\x1f{salt}"
    )


def judge_items(
    items: Iterable[dict],
    task: Task,
    judge_key: str,
    judge: JudgeSpec,
    client: OpenRouter,
    cache: JudgeCache,
    concurrency: int,
    max_tokens: int,
    salt: str = "",
) -> list[dict]:
    """`items` are `{"id", "prompt", "response"}`; returns one verdict row per item, in order.

    Successful verdicts are cached; failures are returned with an `error` but not cached, so a
    rerun retries them.
    """
    items = list(items)

    def one(item: dict) -> dict:
        judge_prompt = task.render(item["prompt"], item["response"])
        key = cache_key(task, judge, judge_prompt, salt)
        hit = cache.get(key)
        if hit is None:
            hit = _call(task, judge, client, key, judge_prompt, task.max_tokens or max_tokens)
            if "error" not in hit:
                cache.put(hit)
        return {"id": item["id"], "judge": judge_key, "task": task.name, **_public(hit)}

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        return list(pool.map(one, items))


def _call(task, judge, client, key, judge_prompt, max_tokens) -> dict:
    row = {"key": key, "task": task.name, "model": judge.model, "pin": list(judge.provider_order)}
    try:
        reply = client.chat(
            judge.model,
            judge_prompt,
            judge.provider_order,
            max_tokens,
            system=task.system,
            reasoning_effort=judge.reasoning_effort,
        )
        row.update(
            raw=reply.text,
            provider=reply.provider,
            served_model=reply.model,
            usage=reply.usage,
            time=time.time(),
        )
        row["verdict"] = task.parse(reply.text)
    except JudgeRefusal:
        row["verdict"] = {"judge_refused": 1}  # a valid outcome; cached like any verdict
    except (JudgeError, ValueError) as exc:
        row["error"] = str(exc)[:500]
    return row


def _public(row: dict) -> dict:
    out = {k: row.get(k) for k in ("provider", "served_model", "raw")}
    if "error" in row:
        out["error"] = row["error"]
    else:
        out.update(row["verdict"])
    return out
