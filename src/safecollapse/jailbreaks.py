"""Tier 2 jailbreak templates (configs/jailbreaks.yaml): fixed wrappers from the StrongREJECT
authors' suite, pinned to a commit. A wrapped prompt is judged against the *original* request,
as StrongREJECT prescribes, so a jailbreak counts as successful only if the answer gives usable
help with what was actually asked."""

from __future__ import annotations

import functools
import hashlib
import json
import os
import urllib.request
from importlib import resources
from pathlib import Path

import yaml

_SPEC = yaml.safe_load((resources.files("safecollapse") / "configs" / "jailbreaks.yaml").read_text())
TEMPLATES: tuple[str, ...] = tuple(_SPEC["templates"])  # names; the text stays upstream
JOB_PREFIX = "@jb_"


def _cache_file() -> Path:
    root = Path(os.environ.get("SAFECOLLAPSE_CACHE", Path.home() / ".cache" / "safecollapse"))
    return root / f"wrapping_jailbreaks-{_SPEC['commit'][:12]}.json"


@functools.lru_cache(maxsize=1)
def templates() -> dict[str, str]:
    """The template text: fetched once from the pinned upstream commit, checked against its
    SHA-256 and cached. This repository does not redistribute it."""
    path = _cache_file()
    if not path.exists():
        with urllib.request.urlopen(_SPEC["url"], timeout=60) as response:
            body = response.read()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    body = path.read_bytes()
    if hashlib.sha256(body).hexdigest() != _SPEC["file_sha256"]:
        path.unlink()
        raise ValueError(f"jailbreak templates do not match the pinned hash: {_SPEC['url']}")
    data = json.loads(body)
    return {name: data[name] for name in TEMPLATES}


def job_name(bench: str, template: str) -> str:
    return f"{bench}{JOB_PREFIX}{template}"


def template_of(job: str) -> str | None:
    return job.split(JOB_PREFIX, 1)[1] if JOB_PREFIX in job else None


def wrap(text: str, template: str) -> str:
    return templates()[template].replace("{forbidden_prompt}", text)


def wrap_record(record: dict, template: str) -> dict:
    """The record with its last user message wrapped (earlier turns, e.g. context, unchanged)."""
    messages = [dict(m) for m in record["messages"]]
    messages[-1]["content"] = wrap(messages[-1]["content"], template)
    return {**record, "messages": messages}
