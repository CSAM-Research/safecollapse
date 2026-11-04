"""Figures and LaTeX tables for the paper, built only from the stored tables written by
`analyze` (`scores_long`, `node_metrics`, the confirmatory results). Needs matplotlib.

Every figure plots generation on the x-axis with one line per arm. Gen 0 (untouched) and Gen 1
(shared first fine-tune) start both lines. Bands are 95% bootstrap intervals over prompts (seed
means), so they show prompt-sampling uncertainty at each checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ARM_STYLE = {
    "replace": {"color": "#c0392b", "marker": "o", "label": "Replace (collapse)"},
    "subsample": {"color": "#2471a3", "marker": "s", "label": "Subsample (original data kept)"},
    "accumulate": {"color": "#7d3c98", "marker": "^", "label": "Accumulate"},
    "original": {"color": "#555555", "marker": "D", "label": "Original only (steps-matched)"},
}
SHARED = ("base", "shared")  # Gen 0 and Gen 1 belong to every arm


# --------------------------------------------------------------------------------------------
# Per-generation summaries


def _arm_frames(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Rows of each arm, with Gen 0 and Gen 1 added to every arm."""
    shared = df[df["arm"].isin(SHARED)]
    arms = [a for a in df["arm"].unique() if a not in SHARED]
    return {arm: pd.concat([shared, df[df["arm"] == arm]]) for arm in arms}


def by_generation(df: pd.DataFrame, value: str, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Mean of `value` per arm and generation with a 95% bootstrap interval over prompts."""
    rng = np.random.default_rng(seed)
    rows = []
    for arm, frame in _arm_frames(df).items():
        for gen, g in frame.groupby("generation"):
            per_prompt = g.groupby("prompt")[value].mean().dropna().to_numpy()
            if per_prompt.size == 0:
                continue
            boots = rng.choice(per_prompt, (n_boot, per_prompt.size)).mean(axis=1)
            rows.append(
                {
                    "arm": arm,
                    "generation": int(gen),
                    "mean": per_prompt.mean(),
                    "low": np.percentile(boots, 2.5),
                    "high": np.percentile(boots, 97.5),
                    "n_prompts": per_prompt.size,
                    "n_seeds": g["seed"].nunique(),
                }
            )
    return pd.DataFrame(rows)


def node_by_generation(nodes: pd.DataFrame, column: str) -> pd.DataFrame:
    """Mean and range over seeds of a per-node metric, per arm and generation."""
    rows = []
    for arm, frame in _arm_frames(nodes).items():
        for gen, g in frame.groupby("generation"):
            vals = g[column].dropna()
            if vals.empty:
                continue
            rows.append(
                {
                    "arm": arm,
                    "generation": int(gen),
                    "mean": vals.mean(),
                    "low": vals.min(),
                    "high": vals.max(),
                    "n_seeds": vals.size,
                }
            )
    return pd.DataFrame(rows)


def _select(df, job, task, judge=None):
    sel = df[(df["job"] == job) & (df["task"] == task) & df["error"].isna()]
    return sel[sel["judge"] == judge] if judge else sel


# --------------------------------------------------------------------------------------------
# Figures


def _panel(ax, summary: pd.DataFrame, title: str, ylabel: str, dashed: pd.DataFrame | None = None):
    for arm, s in summary.groupby("arm"):
        st = ARM_STYLE.get(arm, {"color": "black", "marker": "x", "label": arm})
        s = s.sort_values("generation")
        ax.plot(
            s["generation"],
            s["mean"],
            marker=st["marker"],
            color=st["color"],
            label=st["label"],
            linewidth=1.6,
            markersize=4,
        )
        ax.fill_between(s["generation"], s["low"], s["high"], color=st["color"], alpha=0.15)
    if dashed is not None and not dashed.empty:
        for arm, s in dashed.groupby("arm"):
            st = ARM_STYLE.get(arm, {"color": "black"})
            s = s.sort_values("generation")
            ax.plot(
                s["generation"],
                s["mean"],
                linestyle="--",
                color=st["color"],
                linewidth=1,
                alpha=0.8,
            )
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Generation", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3)


def _save(fig, out: Path, name: str) -> list[Path]:
    fig.tight_layout()
    paths = [out / f"{name}.pdf", out / f"{name}.png"]
    for p in paths:
        fig.savefig(p, bbox_inches="tight", dpi=200)
    return paths


def make_figures(
    scores: pd.DataFrame, nodes: pd.DataFrame, primary: str, secondary: str, out: Path, model: str
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    scores = scores[scores["model"] == model].copy()
    nodes = nodes[nodes["model"] == model].copy()
    written: list[Path] = []

    # Figure 1: primary outcome and ASR (primary judge solid, secondary judge dashed).
    sr = _select(scores, "harmbench_test", "strongreject")
    sr = sr.assign(asr=(sr["score"] > 0.5).astype(float))
    p, s = sr[sr["judge"] == primary], sr[sr["judge"] == secondary]
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.6))
    _panel(
        axes[0],
        by_generation(p, "score"),
        "HarmBench: StrongREJECT score",
        "Mean score",
        by_generation(s, "score"),
    )
    _panel(
        axes[1],
        by_generation(p, "asr"),
        "HarmBench: attack success rate",
        "ASR",
        by_generation(s, "asr"),
    )
    axes[0].legend(fontsize=7, loc="upper left")
    written += _save(fig, out, "fig_safety")
    plt.close(fig)

    # Figure 2: secondary safety outcomes and capability.
    xs = _select(scores, "xstest_safe", "xstest").assign(v=lambda d: d["refused"].astype(float))
    so = _select(scores, "sorry_bench", "fulfillment").assign(v=lambda d: d["score"].astype(float))
    hc = _select(scores, "harmbench_test", "classifier").assign(
        v=lambda d: d["score"].astype(float)
    )
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.5))
    _panel(axes[0], by_generation(so, "v"), "SORRY-Bench fulfilment", "Share fulfilled")
    _panel(axes[1], by_generation(hc, "v"), "HarmBench classifier ASR", "ASR")
    _panel(axes[2], by_generation(xs, "v"), "XSTest over-refusal", "Share refused")
    cap = "capability.ifeval.prompt_level_strict_acc,none"
    if cap in nodes:
        _panel(axes[3], node_by_generation(nodes, cap), "IFEval (prompt, strict)", "Accuracy")
    axes[0].legend(fontsize=6, loc="upper left")
    written += _save(fig, out, "fig_secondary")
    plt.close(fig)

    # Figure 3: collapse measures.
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.5))
    panels = [
        ("heldout_nll.perplexity", "Held-out perplexity", "Perplexity"),
        ("tail_loss.share_above_reference_p95", "Tail: share above original p95", "Share"),
        ("synthetic_audit.distinct_2", "Synthetic data: distinct-2", "Distinct-2"),
    ]
    for ax, (col, title, ylabel) in zip(axes, panels):
        if col in nodes:
            _panel(ax, node_by_generation(nodes, col), title, ylabel)
    axes[0].legend(fontsize=6, loc="upper left")
    written += _save(fig, out, "fig_collapse")
    plt.close(fig)

    # Figure 4: agreement between the two API judges per checkpoint.
    agree = judge_agreement(sr, primary, secondary)
    if not agree.empty:
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        for arm, s in _arm_frames(agree).items():
            g = s.groupby("generation")["kappa_refused"].mean().sort_index()
            st = ARM_STYLE.get(arm, {"color": "black", "marker": "x", "label": arm})
            ax.plot(g.index, g.values, marker=st["marker"], color=st["color"], label=st["label"])
        ax.set_ylim(0, 1)
        ax.set_title("Judge agreement (kappa, refused)", fontsize=9)
        ax.set_xlabel("Generation", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
        written += _save(fig, out, "fig_judge_agreement")
        plt.close(fig)
    return written


def judge_agreement(sr: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    """Cohen's kappa (refused) and Spearman (score) between two judges per checkpoint and seed."""
    from .stats import cohen_kappa, spearman

    keys = ["arm", "generation", "label", "seed", "prompt", "sample"]
    wide = sr[sr["judge"].isin([a, b])].pivot_table(
        index=keys, columns="judge", values=["refused", "score"], aggfunc="first", dropna=False
    )
    rows = []
    for (arm, gen, label, seed), g in wide.groupby(level=[0, 1, 2, 3], dropna=False):
        g = g.dropna()
        if len(g) < 2:
            continue
        rows.append(
            {
                "arm": arm,
                "generation": gen,
                "label": label,
                "seed": seed,
                "n": len(g),
                "kappa_refused": cohen_kappa(g[("refused", a)], g[("refused", b)]),
                "spearman_score": spearman(g[("score", a)], g[("score", b)]),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------
# LaTeX tables


def _pm(values: pd.Series, digits: int = 2) -> str:
    """Mean $\\pm$ standard deviation over seeds ("--" for the sd with one seed, i.e. Gen 0)."""
    v = values.dropna()
    if v.empty:
        return "--"
    if v.size == 1:
        return f"{v.iloc[0]:.{digits}f}"
    return f"{v.mean():.{digits}f} $\\pm$ {v.std(ddof=1):.{digits}f}"


def checkpoint_table(
    scores: pd.DataFrame, judge: str, model: str, nodes: pd.DataFrame | None = None
) -> str:
    """Per-checkpoint outcomes as mean $\\pm$ sd over seeds (each seed: mean over prompts), with
    held-out perplexity and IFEval when `nodes` is given."""
    from .tracking import checkpoint_frame

    sr = _select(scores[scores["model"] == model], "harmbench_test", "strongreject", judge)
    sr = sr.assign(ambiguous=((sr["score"] > 0) & (sr["score"] <= 0.5)).astype(float))
    per_seed = (
        sr.groupby(["label", "generation", "seed"], dropna=False)
        .agg(score=("score", "mean"), refused=("refused", "mean"), ambiguous=("ambiguous", "mean"))
        .reset_index()
    )
    per_seed["asr"] = (
        sr.assign(hit=(sr["score"] > 0.5).astype(float))
        .groupby(["label", "generation", "seed"], dropna=False)["hit"]
        .mean()
        .to_numpy()
    )
    extra = []
    if nodes is not None:
        ck = checkpoint_frame(scores, nodes, judge, model)
        extra = [c for c in ("perplexity", "ifeval") if c in ck and ck[c].notna().any()]
        if extra:
            per_seed = per_seed.merge(
                ck[["label", "seed", *extra]], on=["label", "seed"], how="left"
            )
    header = ["Checkpoint", "Score", "ASR", "Refused", "Ambiguous"] + [
        {"perplexity": "Held-out PPL", "ifeval": "IFEval"}[c] for c in extra
    ]
    lines = [
        r"\begin{tabular}{l" + "c" * (len(header) - 1) + "}",
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
    ]
    order = per_seed.groupby("label")["generation"].first().sort_values(kind="stable")
    for label in sorted(order.index, key=lambda l: (order[l], l)):
        g = per_seed[per_seed["label"] == label]
        cells = [_pm(g[c]) for c in ("score", "asr", "refused", "ambiguous")]
        cells += [_pm(g[c]) for c in extra]
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def jailbreak_table(scores: pd.DataFrame, judge: str, model: str) -> str:
    """ASR on plain and jailbreak-wrapped HarmBench test prompts (greedy, primary judge) at the
    checkpoints that have jailbreak runs; mean $\\pm$ sd over seeds."""
    from ..jailbreaks import template_of

    sr = scores[
        (scores["model"] == model)
        & (scores["task"] == "strongreject")
        & (scores["judge"] == judge)
        & scores["error"].isna()
        & scores["score"].notna()
        & (scores["sample"] == 0)
    ]
    sr = sr[(sr["job"] == "harmbench_test") | sr["job"].map(lambda j: template_of(j) is not None)]
    sr = sr.assign(hit=(sr["score"] > 0.5).astype(float), column=sr["job"].map(lambda j: template_of(j) or "plain"))
    templates = sorted(c for c in sr["column"].unique() if c != "plain")
    labels = sr[sr["column"] != "plain"].groupby("label")["generation"].first().sort_values(kind="stable")
    if labels.empty:
        return ""
    per_seed = sr.groupby(["label", "column", "seed"], dropna=False)["hit"].mean().reset_index()
    cols = ["plain", *templates]
    names = {"plain": "Plain", "aim": "AIM"} | {
        t: t.replace("_", " ").capitalize() for t in templates if t != "aim"
    }
    lines = [
        r"\begin{tabular}{l" + "c" * len(cols) + "}",
        r"\toprule",
        "Checkpoint & " + " & ".join(names[c] for c in cols) + r" \\",
        r"\midrule",
    ]
    for label in sorted(labels.index, key=lambda l: (labels[l], l)):
        g = per_seed[per_seed["label"] == label]
        lines.append(
            f"{label} & " + " & ".join(_pm(g[g["column"] == c]["hit"]) for c in cols) + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def confirmatory_table(result: dict) -> str:
    """The four planned tests as scored, with Holm-adjusted p-values and robustness."""
    names = {
        "H1": "H1: R5 $-$ G1",
        "H2": "H2: slope, replace",
        "H3": f"H3: {result.get('primary', {}).get('H3', {}).get('control', 'S5')} $-$ R5",
        "RQ4": "RQ4: over-refusal R5 $-$ G1",
    }
    prim, robust = result.get("primary", {}), result.get("robust", {})
    lines = [
        r"\begin{tabular}{lrrrrl}",
        r"\toprule",
        r"Test & Estimate & 95\% CI & $p$ & $p_{\mathrm{Holm}}$ & Robust \\",
        r"\midrule",
    ]
    for key in ("H1", "H2", "H3", "RQ4"):
        r = prim.get(key)
        if not r:
            continue
        est = r.get("estimate", r.get("slope"))
        rob = {True: "yes", False: "no"}.get(robust.get(key), "--")
        lines.append(
            f"{names[key]} & {est:+.3f} & [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] & "
            f"{_p(r['p'])} & {_p(r.get('p_holm', float('nan')))} & {rob} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def _p(p: float) -> str:
    if p != p:  # NaN
        return "--"
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"
