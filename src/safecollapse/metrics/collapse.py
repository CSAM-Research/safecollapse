"""Collapse measures (DESIGN.md Sections 5.4 and 7):

- held-out perplexity: response-token NLL of HelpSteer2 validation under each checkpoint;
- tail loss: per-example perplexity of each round's synthetic data under Gen 0 (Shumailov et al.,
  Figure 1), summarised against the original data's 95th percentile;
- diversity and a synthetic-data audit of each round.
"""

from __future__ import annotations

import math
import re
from statistics import fmean

import numpy as np

# Exploratory refusal heuristic for the synthetic-data audit only; never used for scoring.
REFUSAL_MARKERS = (
    "i cannot",
    "i can't",
    "i can not",
    "i won't",
    "i will not",
    "i'm not able",
    "i am not able",
    "i'm unable",
    "i am unable",
    "as an ai",
    "i'm sorry",
    "i am sorry",
    "i apologize",
)


# --------------------------------------------------------------------------------------------
# Needs a model (GPU)


def response_nll_with_model(
    model, tokenizer, examples: list[dict], max_len: int, batch_size: int = 4
) -> list[dict]:
    """Per example: summed NLL and number of response tokens (same masking as training)."""
    import torch

    from ..modeling import encode_example, pad_batch

    encoded = []
    for ex in examples:
        enc = encode_example(tokenizer, ex["messages"], ex["response"], max_len)
        if enc is not None:
            encoded.append((ex["id"], enc))
    encoded.sort(key=lambda x: len(x[1].input_ids))
    out = []
    for i in range(0, len(encoded), batch_size):
        chunk = encoded[i : i + batch_size]
        ids, mask, labels = pad_batch([e for _, e in chunk], tokenizer.pad_token_id)
        ids, mask, labels = ids.cuda(), mask.cuda(), labels.cuda()
        with torch.no_grad():
            logits = model(input_ids=ids, attention_mask=mask).logits[:, :-1].float()
        target = labels[:, 1:]
        nll = torch.nn.functional.cross_entropy(
            logits.transpose(1, 2), target, ignore_index=-100, reduction="none"
        )
        valid = target != -100
        for (eid, _), row_nll, row_valid in zip(chunk, nll, valid):
            out.append(
                {
                    "id": eid,
                    "nll_sum": float(row_nll[row_valid].sum()),
                    "n_tokens": int(row_valid.sum()),
                }
            )
    return out


# --------------------------------------------------------------------------------------------
# Pure summaries


def corpus_nll(rows: list[dict]) -> dict:
    """Token-weighted mean NLL and perplexity = e^NLL."""
    tokens = sum(r["n_tokens"] for r in rows)
    nll = sum(r["nll_sum"] for r in rows) / tokens
    return {"nll": nll, "perplexity": math.exp(nll), "tokens": tokens, "examples": len(rows)}


def per_example_perplexity(rows: list[dict]) -> np.ndarray:
    return np.array([math.exp(r["nll_sum"] / r["n_tokens"]) for r in rows if r["n_tokens"]])


def tail_summary(ppl: np.ndarray, reference_p95: float) -> dict:
    """Distribution of per-example perplexity, and the share above the original data's 95th
    percentile. Collapse predicts this share shrinks across generations."""
    qs = (5, 25, 50, 75, 95, 99)
    return {
        "n": int(ppl.size),
        "quantiles": {f"p{q}": float(np.percentile(ppl, q)) for q in qs},
        "mean_log_ppl": float(np.log(ppl).mean()),
        "share_above_reference_p95": float((ppl > reference_p95).mean()),
        "reference_p95": reference_p95,
    }


def distinct_n(texts: list[str], n: int) -> float:
    """Unique n-grams / all n-grams across the corpus (lower-cased words)."""
    grams, total = set(), 0
    for text in texts:
        words = re.findall(r"\w+", text.lower())
        seq = list(zip(*(words[i:] for i in range(n))))
        grams.update(seq)
        total += len(seq)
    return len(grams) / total if total else 0.0


def audit_synthetic(rows: list[dict]) -> dict:
    """What the next generation will train on: empty, truncated and refusal-like answers,
    length and diversity (DESIGN.md Section 7.1)."""
    texts = [r["completion"] for r in rows]
    nonempty = [t for t in texts if t.strip()]
    return {
        "n": len(rows),
        "empty_rate": 1 - len(nonempty) / len(rows) if rows else None,
        "truncated_rate": fmean(r["finish_reason"] == "length" for r in rows) if rows else None,
        "mean_tokens": fmean(r["n_tokens"] for r in rows) if rows else None,
        "refusal_marker_rate": fmean(_has_refusal_marker(t) for t in texts) if rows else None,
        "distinct_1": distinct_n(nonempty, 1),
        "distinct_2": distinct_n(nonempty, 2),
    }


def _has_refusal_marker(text: str) -> bool:
    head = text[:200].lower()
    return any(marker in head for marker in REFUSAL_MARKERS)


def embedding_diversity(texts: list[str], sample: int = 1000, seed: int = 0) -> float:
    """Mean pairwise cosine distance between answers (needs the [diversity] extra)."""
    from sentence_transformers import SentenceTransformer

    rng = np.random.default_rng(seed)
    texts = [t for t in texts if t.strip()]
    if len(texts) > sample:
        texts = [texts[i] for i in rng.choice(len(texts), sample, replace=False)]
    emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").encode(
        texts, normalize_embeddings=True
    )
    sims = emb @ emb.T
    n = len(texts)
    return float(1 - (sims.sum() - n) / (n * (n - 1)))
