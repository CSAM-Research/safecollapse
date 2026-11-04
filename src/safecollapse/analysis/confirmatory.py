"""The planned analysis (DESIGN.md Sections 4 and 6), run exactly as written:

- H1: R5 > G1 (mean StrongREJECT score, HarmBench test, primary judge)
- H2: positive slope over generations 1-5 of the replace arm
- H3: S5 < R5 (A5 < R5 where the data-keeping arm is accumulate: the base-model category)
- RQ4: over-refusal on XSTest safe prompts, R5 vs G1 (two-sided)

Holm correction across the four; then the ambiguity and judge sensitivity analyses.
"""

from __future__ import annotations

import pandas as pd

from . import stats

PRIMARY_JOB = "harmbench_test"


def primary_frame(
    df: pd.DataFrame, judge: str, model: str, judge_refusal_as_harmful: bool = False
) -> pd.DataFrame:
    """Greedy HarmBench-test scores of one model and one judge; Gen 0 excluded (no seed)."""
    sel = df[
        (df["job"] == PRIMARY_JOB)
        & (df["judge"] == judge)
        & (df["model"] == model)
        & (df["task"] == "strongreject")
        & (df["generation"] >= 1)
        & df["error"].isna()
    ]
    if judge_refusal_as_harmful:  # sensitivity: a judge refusing to grade counts as harmful
        sel = sel.assign(score=sel["score"].where(sel["judge_refused"] != 1, 1.0))
    sel = sel[sel["score"].notna()]  # as planned: judge refusals are otherwise excluded
    return sel[["label", "seed", "prompt", "generation", "arm", "score"]].copy()


def over_refusal_frame(df: pd.DataFrame, judge: str, model: str) -> pd.DataFrame:
    sel = df[
        (df["job"] == "xstest_safe")
        & (df["judge"] == judge)
        & (df["model"] == model)
        & (df["generation"] >= 1)
        & df["error"].isna()
    ]
    return sel[["label", "seed", "prompt", "generation", "arm", "refused"]].copy()


def control_label(scores: pd.DataFrame, final: int = 5) -> str:
    """The data-keeping arm's final checkpoint: S5 (subsample), else A5 (accumulate; published
    base-model checkpoints, which have no subsample arm)."""
    labels = set(scores["label"])
    return f"S{final}" if f"S{final}" in labels or f"A{final}" not in labels else f"A{final}"


def confirmatory_tests(
    scores: pd.DataFrame, refusals: pd.DataFrame | None, final: int = 5, bootstrap: int = 5000
) -> dict:
    replace_final, subsample_final = f"R{final}", control_label(scores, final)
    fit = stats.fit_mixed(scores)
    out = {
        "H1": stats.contrast(fit, replace_final, "G1"),
        "H3": {**stats.contrast(fit, subsample_final, replace_final), "control": subsample_final},
    }
    replace_arm = scores[scores["label"].isin(["G1"] + [f"R{g}" for g in range(2, final + 1)])]
    out["H2"] = stats.trend(replace_arm)
    if refusals is not None and not refusals.empty:
        rfit = stats.fit_mixed(refusals, value="refused")
        out["RQ4"] = stats.contrast(rfit, replace_final, "G1")
    adjusted = stats.holm({k: v["p"] for k, v in out.items()})
    for key, p_adj in adjusted.items():
        out[key]["p_holm"] = p_adj
    if bootstrap:
        out["H1"]["bootstrap"] = stats.bootstrap_difference(
            scores, replace_final, "G1", n=bootstrap
        )
        out["H3"]["bootstrap"] = stats.bootstrap_difference(
            scores, subsample_final, replace_final, n=bootstrap
        )
        if "RQ4" in out:
            out["RQ4"]["bootstrap"] = stats.bootstrap_difference(
                refusals, replace_final, "G1", value="refused", n=bootstrap
            )
    return out


def run(
    df: pd.DataFrame,
    model: str,
    primary: str,
    secondary: str,
    refusal_judge: str,
    final: int = 5,
    bootstrap: int = 5000,
) -> dict:
    refusals = over_refusal_frame(df, refusal_judge, model)
    scores = primary_frame(df, primary, model)
    result = {"primary": confirmatory_tests(scores, refusals, final, bootstrap)}
    # Sensitivity analyses: ambiguous answers as harmful / as safe, and the secondary judge.
    for mode in ("harmful", "safe"):
        variant = stats.ambiguity_variant(scores, mode)
        result[f"ambiguity_as_{mode}"] = confirmatory_tests(variant, None, final, bootstrap=0)
    secondary_scores = primary_frame(df, secondary, model)
    if not secondary_scores.empty:
        result["secondary_judge"] = confirmatory_tests(secondary_scores, None, final, bootstrap=0)
    # Added 3 Nov 2026, before any main result was inspected (DECISIONS.md B14): judge refusals
    # occur on harmful responses, so excluding them can understate harm.
    refusals_present = (df["judge_refused"] == 1).any() if "judge_refused" in df else False
    if refusals_present:
        for name, judge in (("primary", primary), ("secondary_judge", secondary)):
            frame = primary_frame(df, judge, model, judge_refusal_as_harmful=True)
            result[f"{name}_judge_refusal_as_harmful"] = confirmatory_tests(
                frame, None, final, bootstrap=0
            )
    # Exploratory (DECISIONS.md B16): Jev's binary verdicts as the score; reported, but never
    # part of the robustness verdict.
    jev = df[
        (df["job"] == PRIMARY_JOB)
        & (df["task"] == "jev")
        & (df["model"] == model)
        & (df["generation"] >= 1)
        & df["error"].isna()
        & df["score"].notna()
    ]
    if not jev.empty:
        frame = jev[["label", "seed", "prompt", "generation", "arm", "score"]].copy()
        result["exploratory_jev"] = confirmatory_tests(frame, None, final, bootstrap=0)
    result["robust"] = _robust(result)
    return result


def _robust(result: dict) -> dict:
    """A conclusion is robust only if its direction and significance hold under every variant.
    H1 and H2 predict increases; H3 predicts a decrease."""
    expected = {"H1": 1, "H2": 1, "H3": -1}
    verdict = {}
    for key, sign in expected.items():
        variants = [
            r[key]
            for name, r in result.items()
            if isinstance(r, dict) and key in r and not name.startswith("exploratory")
        ]
        verdict[key] = all(
            (v.get("estimate", v.get("slope")) * sign > 0) and v["p"] < 0.05 for v in variants
        )
    return verdict
