"""Statistics (DESIGN.md Section 6). Inputs are long data frames with one row per
(checkpoint label, seed, prompt): columns `label`, `seed`, `prompt`, `score`, plus whatever the
analysis needs.

numpy/pandas only, except the mixed model, which needs statsmodels (the [analysis] extra).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------------
# Multiple comparisons and agreement


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values (step-down, monotone, capped at 1)."""
    ordered = sorted(pvalues.items(), key=lambda kv: kv[1])
    m, running, out = len(ordered), 0.0, {}
    for rank, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, (m - rank) * p))
        out[name] = running
    return out


def cohen_kappa(a, b) -> float:
    """Cohen's kappa for two binary or categorical label vectors."""
    a, b = np.asarray(a), np.asarray(b)
    cats = np.union1d(a, b)
    po = float(np.mean(a == b))
    pe = float(sum(np.mean(a == c) * np.mean(b == c) for c in cats))
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def spearman(a, b) -> float:
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def min_detectable_difference(
    sd: float, n_prompts: int, n_seeds: int, alpha: float = 0.05, power: float = 0.8
) -> float:
    """Rough two-sample MDE for a difference in mean score (the design's power estimate)."""
    from statistics import NormalDist

    z = NormalDist().inv_cdf(1 - alpha / 2) + NormalDist().inv_cdf(power)
    return z * sd * math.sqrt(2 / (n_prompts * n_seeds))


# --------------------------------------------------------------------------------------------
# Ambiguity sensitivity (Section 6)


def ambiguity_variant(df: pd.DataFrame, mode: str, threshold: float = 0.5) -> pd.DataFrame:
    """mode: 'scored' (as is), 'harmful' (ambiguous -> 1) or 'safe' (ambiguous -> 0)."""
    if mode == "scored":
        return df
    ambiguous = (df["score"] > 0) & (df["score"] <= threshold)
    out = df.copy()
    out.loc[ambiguous, "score"] = 1.0 if mode == "harmful" else 0.0
    return out


# --------------------------------------------------------------------------------------------
# Bootstrap over prompts and seeds


def bootstrap_difference(
    df: pd.DataFrame, a: str, b: str, value: str = "score", n: int = 5000, seed: int = 0
) -> dict:
    """Mean(a) - mean(b) with a 95% CI from resampling prompts and seeds (Section 6).

    Prompts and seeds are resampled jointly for both checkpoints, keeping the pairing.
    """
    wide = df[df["label"].isin([a, b])].pivot_table(
        index=["prompt"], columns=["label", "seed"], values=value, aggfunc="mean"
    )
    seeds = sorted({s for _, s in wide.columns})
    A = np.stack([wide[(a, s)].to_numpy() for s in seeds], axis=1)  # [prompts, seeds]
    B = np.stack([wide[(b, s)].to_numpy() for s in seeds], axis=1)
    if np.isnan(A).any() or np.isnan(B).any():
        raise ValueError(f"missing scores for {a} or {b}; every prompt needs every seed")
    rng = np.random.default_rng(seed)
    n_p, n_s = A.shape
    diffs = np.empty(n)
    for i in range(n):
        p = rng.integers(0, n_p, n_p)
        s = rng.integers(0, n_s, n_s)
        diffs[i] = A[np.ix_(p, s)].mean() - B[np.ix_(p, s)].mean()
    return {
        "estimate": float(A.mean() - B.mean()),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "resamples": n,
    }


# --------------------------------------------------------------------------------------------
# Mixed-effects model: score ~ checkpoint + (1 | prompt) + (1 | seed)


def fit_mixed(df: pd.DataFrame, reference: str = "G1", value: str = "score"):
    """Cell-means mixed model; every checkpoint (arm x generation) is one fixed level, so the
    planned arm x generation contrasts are linear combinations of its coefficients.
    Prompt and seed are crossed random intercepts (statsmodels variance components)."""
    import statsmodels.formula.api as smf

    data = df.assign(_y=df[value], _all=1)
    formula = f"_y ~ C(label, Treatment(reference='{reference}'))"
    model = smf.mixedlm(
        formula,
        data,
        groups="_all",
        re_formula="0",
        vc_formula={"prompt": "0 + C(prompt)", "seed": "0 + C(seed)"},
    )
    return model.fit(reml=True)


def contrast(result, a: str, b: str, reference: str = "G1") -> dict:
    """Wald test of mean(a) - mean(b) from a `fit_mixed` result."""
    names = list(result.fe_params.index)

    def coef(label: str) -> str | None:
        if label == reference:
            return None
        name = f"C(label, Treatment(reference='{reference}'))[T.{label}]"
        if name not in names:
            raise KeyError(f"checkpoint {label!r} is not in the model")
        return name

    L = np.zeros(len(names))
    for label, sign in ((a, 1.0), (b, -1.0)):
        name = coef(label)
        if name:
            L[names.index(name)] += sign
    test = result.t_test(L[None, :])
    low, high = np.asarray(test.conf_int()).ravel()
    return {
        "estimate": float(np.ravel(test.effect)[0]),
        "se": float(np.ravel(test.sd)[0]),
        "p": float(np.ravel(test.pvalue)[0]),
        "ci_low": float(low),
        "ci_high": float(high),
    }


def trend(df: pd.DataFrame, value: str = "score") -> dict:
    """H2: slope of score on generation (as a number) with the same random effects."""
    import statsmodels.formula.api as smf

    data = df.assign(_y=df[value], _all=1)
    model = smf.mixedlm(
        "_y ~ generation",
        data,
        groups="_all",
        re_formula="0",
        vc_formula={"prompt": "0 + C(prompt)", "seed": "0 + C(seed)"},
    )
    res = model.fit(reml=True)
    low, high = res.conf_int().loc["generation"]
    return {
        "slope": float(res.fe_params["generation"]),
        "se": float(res.bse_fe["generation"]),
        "p": float(res.pvalues["generation"]),
        "ci_low": float(low),
        "ci_high": float(high),
    }
