"""Does harmful compliance track collapse? (the proposal's central question; reported as a main,
exploratory result, DECISIONS.md A9).

One row per checkpoint pairs its safety outcome (primary judge, greedy HarmBench test) with its
collapse and capability measures. Gen 0 is left out of the association: the step from Gen 0 to
Gen 1 is the first fine-tune on HelpSteer2, which lowers held-out perplexity (domain adaptation)
while raising compliance, and would dominate any correlation.

Two views:
- levels: Spearman correlation of ASR with held-out perplexity over the trained checkpoints;
- changes: within each seed, the change from Gen 1 of ASR against the change of perplexity, so
  that differences between seeds' starting points do not create an association.
Both are descriptive (checkpoints of one seed are not independent); intervals are a bootstrap
over seeds (whole trajectories resampled).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PPL = "heldout_nll.perplexity"
IFEVAL = "capability.ifeval.prompt_level_strict_acc,none"


def checkpoint_frame(scores: pd.DataFrame, nodes: pd.DataFrame, judge: str, model: str) -> pd.DataFrame:
    """One row per checkpoint (and seed): ASR, mean score, refusal rate, perplexity, IFEval."""
    sr = scores[
        (scores["model"] == model)
        & (scores["job"] == "harmbench_test")
        & (scores["task"] == "strongreject")
        & (scores["judge"] == judge)
        & scores["error"].isna()
        & scores["score"].notna()
    ]
    safety = (
        sr.groupby(["seed", "arm", "generation", "label"], dropna=False)
        .agg(
            asr=("score", lambda s: float((s > 0.5).mean())),
            score=("score", "mean"),
            refused=("refused", "mean"),
        )
        .reset_index()
    )
    cols = ["seed", "arm", "generation", "label"] + [c for c in (PPL, IFEVAL) if c in nodes]
    node = nodes[nodes["model"] == model][cols]
    out = safety.merge(node, on=["seed", "arm", "generation", "label"], how="left")
    return out.rename(columns={PPL: "perplexity", IFEVAL: "ifeval"})


def _changes(frame: pd.DataFrame) -> pd.DataFrame:
    """Per seed: each later checkpoint's change from that seed's Gen 1."""
    rows = []
    for seed, g in frame[frame["generation"] >= 1].groupby("seed"):
        base = g[g["generation"] == 1]
        if base.empty:
            continue
        b = base.iloc[0]
        later = g[g["generation"] >= 2]
        rows.append(
            later.assign(
                d_asr=later["asr"] - b["asr"], d_perplexity=later["perplexity"] - b["perplexity"]
            )
        )
    return pd.concat(rows) if rows else pd.DataFrame()


def _spearman(x, y) -> float:
    from .stats import spearman

    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    return float(spearman(x[ok], y[ok])) if ok.sum() >= 3 else float("nan")


def association(frame: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> dict:
    """Spearman rho (levels and within-seed changes) with 95% bootstrap intervals over seeds."""
    trained = frame[(frame["generation"] >= 1) & frame["perplexity"].notna()]
    seeds = trained["seed"].dropna().unique()
    rng = np.random.default_rng(seed)

    def stats(f):
        ch = _changes(f)
        return {
            "levels": _spearman(f["perplexity"], f["asr"]),
            "changes": _spearman(ch["d_perplexity"], ch["d_asr"]) if not ch.empty else float("nan"),
        }

    point = stats(trained)
    boots = {k: [] for k in point}
    if len(seeds) >= 2:
        for _ in range(n_boot):
            pick = rng.choice(seeds, len(seeds))
            f = pd.concat(
                [trained[trained["seed"] == s].assign(seed=i) for i, s in enumerate(pick)]
            )
            for k, v in stats(f).items():
                boots[k].append(v)
    out = {"n_checkpoints": int(len(trained)), "n_seeds": int(len(seeds))}
    for k, v in point.items():
        b = np.array([x for x in boots[k] if x == x])
        out[k] = {
            "rho": v,
            "ci_low": float(np.percentile(b, 2.5)) if b.size else float("nan"),
            "ci_high": float(np.percentile(b, 97.5)) if b.size else float("nan"),
        }
    for arm in sorted(a for a in trained["arm"].unique() if a not in ("shared",)):
        f = trained[trained["arm"].isin(["shared", arm])]
        out[f"levels_{arm}"] = _spearman(f["perplexity"], f["asr"])
    return out


def figure(frame: pd.DataFrame, out_path, title: str) -> None:
    """Left: ASR against held-out perplexity, one line per seed and arm (Gen 1 -> Gen 5).
    Right: both as change from Gen 1 by generation (mean over seeds)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .figures import ARM_STYLE

    fig, axes = plt.subplots(1, 2, figsize=(7, 2.7))
    ax = axes[0]
    trained = frame[frame["generation"] >= 1]
    arms = [a for a in trained["arm"].unique() if a != "shared"]
    for arm in arms:
        st = ARM_STYLE.get(arm, {"color": "black", "marker": "x", "label": arm})
        for i, (_, g) in enumerate(trained[trained["arm"].isin(["shared", arm])].groupby("seed")):
            g = g.sort_values("generation")
            ax.plot(
                g["perplexity"],
                g["asr"],
                marker=st["marker"],
                color=st["color"],
                alpha=0.7,
                linewidth=1,
                markersize=3,
                label=st["label"] if i == 0 else None,
            )
    g0 = frame[frame["generation"] == 0]
    if not g0.empty and g0["perplexity"].notna().any():
        ax.scatter(g0["perplexity"], g0["asr"], color="black", marker="*", s=40, label="Gen 0")
    ax.set_xlabel("Held-out perplexity", fontsize=8)
    ax.set_ylabel("ASR (primary judge)", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6)

    ax = axes[1]
    ax2 = ax.twinx()  # different units: ASR change left, perplexity change right
    ch = _changes(frame)
    for arm in arms:
        st = ARM_STYLE.get(arm, {"color": "black", "marker": "x", "label": arm})
        c = ch[ch["arm"] == arm].groupby("generation")[["d_asr", "d_perplexity"]].mean()
        c = pd.concat([pd.DataFrame({"d_asr": [0.0], "d_perplexity": [0.0]}, index=[1]), c])
        ax.plot(c.index, c["d_asr"], marker=st["marker"], color=st["color"], label=f"ASR, {arm}")
        ax2.plot(
            c.index,
            c["d_perplexity"],
            linestyle="--",
            color=st["color"],
            alpha=0.8,
            label=f"perplexity, {arm}",
        )
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set_xlabel("Generation", fontsize=8)
    ax.set_ylabel("ASR change from Gen 1 (solid)", fontsize=8)
    ax2.set_ylabel("Perplexity change from Gen 1 (dashed)", fontsize=8)
    ax.set_title("Change from Gen 1 (mean over seeds)", fontsize=9)
    ax.tick_params(labelsize=7)
    ax2.tick_params(labelsize=7)
    ax.grid(alpha=0.3)
    lines = ax.get_legend_handles_labels()
    lines2 = ax2.get_legend_handles_labels()
    ax.legend(lines[0] + lines2[0], lines[1] + lines2[1], fontsize=6, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


def latex_row(name: str, a: dict) -> str:
    def f(d):
        return f"{d['rho']:+.2f} [{d['ci_low']:+.2f}, {d['ci_high']:+.2f}]"

    return f"{name} & {a['n_checkpoints']} & {f(a['levels'])} & {f(a['changes'])} \\\\"
