"""What each checkpoint trains on (DESIGN.md Section 3.3). Pure functions over example lists.

An example is `{"id", "messages", "response", "source"}`: `messages` is the conversation up to
the answer, `response` the answer, and `source` where the answer came from ("original" or the
label of the checkpoint that wrote it).
"""

from __future__ import annotations

import random
from collections.abc import Sequence

Example = dict


def training_set(
    arm: str,
    gen: int,
    original: Sequence[Example],
    synthetic: dict[int, Sequence[Example]],
    seed: int,
) -> list[Example]:
    """Training data for generation `gen` of `arm`.

    `synthetic[g]` holds the answers written by this arm's generation-g model (g = 1 is the
    shared Gen 1), each answering the same prompts as `original`.
    """
    if gen == 1:
        return list(original)
    if arm == "original":
        # Steps-matched control: the original data repeated to match A_gen's optimiser steps.
        return [dict(ex, id=f"{ex['id']}#rep{k}") for k in range(gen) for ex in original]
    _require(synthetic, range(1, gen) if arm != "replace" else [gen - 1])
    if arm == "replace":
        return list(synthetic[gen - 1])
    pool = list(original) + [ex for g in range(1, gen) for ex in synthetic[g]]
    if arm == "accumulate":
        return pool
    if arm == "subsample":
        # A fresh draw each generation, same size as one round, so replace and subsample see the
        # same data volume and number of optimiser steps.
        rng = random.Random(f"subsample/{seed}/{gen}")
        return rng.sample(pool, len(original))
    raise ValueError(f"unknown arm {arm!r}")


def _require(synthetic: dict[int, Sequence[Example]], gens) -> None:
    missing = [g for g in gens if g not in synthetic]
    if missing:
        raise ValueError(f"missing synthetic data for generations {missing}")


def with_answers(prompts: Sequence[Example], answers: dict[str, str], source: str) -> list[Example]:
    """Replace the responses of `prompts` with a model's answers (same ids, same order)."""
    return [dict(ex, response=answers[ex["id"]], source=source) for ex in prompts]
