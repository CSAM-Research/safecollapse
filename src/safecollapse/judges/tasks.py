"""A judge task = how to build the judge's prompt and how to read its answer. The API runner
works with any task, so adding a judge means adding a Task, not changing the runner."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    name: str
    render: Callable[[str, str], str]  # (user prompt, model response) -> judge prompt
    parse: Callable[[str], dict]  # judge output -> verdict; raises ValueError if malformed
    system: str | None = None  # optional system message for the judge
    max_tokens: int | None = None  # per-task output limit (None = the config's default)
