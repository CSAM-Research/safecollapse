"""Refusal direction (Arditi et al. 2024), exploratory (DESIGN.md Section 7.3).

Gen 0's refusal direction at each layer is the difference between the mean residual-stream
activation (last prompt token) on harmful and on harmless prompts. For every checkpoint we then
measure how strongly harmful prompts still express Gen 0's direction. Directional ablation
(Arditi's causal test) is a later addition if this tracking shows a change worth explaining.

Only validation prompts are used (HarmBench validation vs HelpSteer2 held-out), never the test
split.
"""

from __future__ import annotations

import numpy as np


def last_token_activations(model, tokenizer, conversations: list[list[dict]]) -> np.ndarray:
    """[n_prompts, n_layers + 1, d_model] residual stream at the last prompt token."""
    import torch

    from ..modeling import prompt_ids

    acts = []
    for messages in conversations:
        ids = torch.tensor([prompt_ids(tokenizer, messages)]).cuda()
        with torch.no_grad():
            hidden = model(input_ids=ids, output_hidden_states=True).hidden_states
        acts.append(torch.stack([h[0, -1].float() for h in hidden]).cpu().numpy())
    return np.stack(acts)


def directions(harmful: np.ndarray, harmless: np.ndarray) -> np.ndarray:
    """Unit difference-of-means direction per layer: [n_layers + 1, d_model]."""
    diff = harmful.mean(axis=0) - harmless.mean(axis=0)
    return diff / np.linalg.norm(diff, axis=-1, keepdims=True)


def expression(acts: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    """Mean projection of activations onto the direction, per layer: [n_layers + 1]."""
    return np.einsum("nld,ld->l", acts, dirs) / acts.shape[0]


def summarize(harmful: np.ndarray, harmless: np.ndarray, dirs: np.ndarray) -> dict:
    """Per-layer separation between harmful and harmless prompts along Gen 0's direction."""
    gap = expression(harmful, dirs) - expression(harmless, dirs)
    return {
        "gap_by_layer": gap.tolist(),
        "best_layer": int(np.argmax(gap)),
        "max_gap": float(gap.max()),
    }
