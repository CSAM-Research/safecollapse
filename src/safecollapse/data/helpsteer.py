"""HelpSteer2: the original (non-recursive) training data and the held-out perplexity set."""

from __future__ import annotations

import gzip
import json
import random
from collections import defaultdict
from pathlib import Path

# HelpSteer2 stores multi-turn prompts in NeMo format: the first user turn, then
# "<extra_id_1>Assistant\n..." and "<extra_id_1>User\n..." turns. About 29% of prompts.
TURN_TAG = "<extra_id_1>"


def parse_conversation(prompt: str) -> list[dict[str, str]]:
    """NeMo-format prompt -> chat messages ending with a user turn, for the native template."""
    first, *turns = prompt.split(TURN_TAG)
    messages = [{"role": "user", "content": first.strip()}]
    for turn in turns:
        speaker, _, text = turn.partition("\n")
        role = {"User": "user", "Assistant": "assistant"}.get(speaker.strip())
        if role is None:
            raise ValueError(f"unexpected speaker {speaker!r}")
        messages.append({"role": role, "content": text.strip()})
    roles = [m["role"] for m in messages]
    if roles[-1] != "user" or any(a == b for a, b in zip(roles, roles[1:])):
        raise ValueError(f"turns do not alternate user/assistant: {roles}")
    return messages


def load_rows(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def to_example(row: dict, example_id: str) -> dict:
    return {
        "id": example_id,
        "messages": parse_conversation(row["prompt"]),
        "response": row["response"].strip(),
        "source": "original",
    }


def select_round(rows: list[dict], size: int, seed: int, one_per_prompt: bool) -> list[dict]:
    """The seed's original data (Gen 1's training set; its prompts are re-answered every round).

    Rows are sampled without replacement. HelpSteer2 has two responses per prompt, so with
    `one_per_prompt` one response is picked per prompt first (at most 10,161 rows).
    """
    rng = random.Random(f"helpsteer-round/{seed}")
    indices = list(range(len(rows)))
    if one_per_prompt:
        by_prompt: dict[str, list[int]] = defaultdict(list)
        for i, row in enumerate(rows):
            by_prompt[row["prompt"]].append(i)
        indices = [rng.choice(group) for group in by_prompt.values()]
    if size > len(indices):
        raise ValueError(f"round_size {size} exceeds the {len(indices)} available rows")
    chosen = rng.sample(indices, size)
    return [to_example(rows[i], f"hs2-train-{i}") for i in chosen]


def select_heldout(rows: list[dict], size: int | None) -> list[dict]:
    """Held-out set for perplexity: fixed across seeds and arms."""
    indices = list(range(len(rows)))
    if size is not None and size < len(rows):
        indices = sorted(random.Random("helpsteer-heldout").sample(indices, size))
    return [to_example(rows[i], f"hs2-val-{i}") for i in indices]
