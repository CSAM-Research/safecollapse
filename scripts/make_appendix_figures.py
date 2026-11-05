"""Appendix figures, in the style of Fig. 1, read from the node metrics. Run by `make results`.

Usage: python3 make_appendix_figures.py <runs_dir> <out_dir>
"""
import json
import statistics
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from make_main_figure import ARM, GRID, INK, MUTED, SHARED  # also applies the shared style

SEEDS = [0, 1, 2]
ASR = "harmbench_test__strongreject__{}.json"


def read(path: Path) -> dict:
    try:
        return json.loads(path.read_text().replace("NaN", "null"))
    except (OSError, ValueError):
        return {}


def node(root: Path, seed: int, arm: str, gen: int) -> Path:
    if gen == 0:
        return root / "gen0"
    return root / f"seed{seed}" / ("shared/gen1" if gen == 1 else f"{arm}/gen{gen}")


def line(root, arm, gens, file, key, seeds=SEEDS):
    xs, means, lows, highs = [], [], [], []
    for g in gens:
        vals = []
        for s in seeds:
            v = read(node(root, s, arm, g) / "metrics" / file)
            for k in key if isinstance(key, tuple) else (key,):
                v = v.get(k) if isinstance(v, dict) else None
            if v is not None:
                vals.append(v)
        if vals:
            xs.append(g)
            means.append(statistics.mean(vals))
            lows.append(min(vals))
            highs.append(max(vals))
    return xs, means, lows, highs


def tidy(ax, title=None, ylabel=None, gmax=None):
    if title:
        ax.set_title(title, color=INK, loc="left", fontsize=6.8, fontweight="bold")
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5, width=0.6)
    if gmax:
        ax.set_xticks(range(0, gmax + 1, 2 if gmax > 5 else 1))


def plot(ax, root, arm, gens, file, key, band=True, ls="-", seeds=SEEDS):
    st = ARM[arm]
    x, m, lo, hi = line(root, arm, gens, file, key, seeds)
    if band:
        ax.fill_between(x, lo, hi, color=st["color"], alpha=0.14, linewidth=0)
    ax.plot(x, m, color=st["color"], lw=1.2, ls=ls, marker=st["marker"] if ls == "-" else None,
            ms=2.8, markeredgecolor="white", markeredgewidth=0.4, zorder=3)


def legend(fig, arms, extra=(), y=1.04):
    handles = [
        plt.Line2D([], [], color=ARM[a]["color"], marker=ARM[a]["marker"], lw=1.5, ms=4,
                   markeredgecolor="white", label=ARM[a]["label"])
        for a in arms
    ] + list(extra)
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, fontsize=6.3,
               bbox_to_anchor=(0.5, y), handlelength=1.6, columnspacing=0.9, handletextpad=0.4)


def tracking(it, base, out):
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.6))
    for ax, root, arms, title in (
        (axes[0], it, ["replace", "subsample"], "Instruction-tuned"),
        (axes[1], base, ["replace", "accumulate"], "Base"),
    ):
        for arm in arms:
            st = ARM[arm]
            for s in SEEDS:
                _, a, _, _ = line(root, arm, range(1, 6), ASR.format("gpt4o_mini"), "asr", [s])
                _, p, _, _ = line(root, arm, range(1, 6), "heldout_nll.json", "perplexity", [s])
                ax.plot(p, a, color=st["color"], lw=0.9, marker=st["marker"], ms=3.5, alpha=0.85,
                        markeredgecolor="white", markeredgewidth=0.4)
        ax.set_xlabel("Held-out perplexity")
        tidy(ax, title, "ASR")
    legend(fig, ["replace", "subsample", "accumulate"], y=1.1)
    fig.tight_layout(w_pad=1.4)
    fig.savefig(out, bbox_inches="tight")


def judges(it, base, out):
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.6))
    for ax, root, arms, title, gmax in (
        (axes[0], it, ["replace", "subsample"], "Instruction-tuned", 5),
        (axes[1], base, ["replace", "accumulate"], "Base", 10),
    ):
        for arm in arms:
            plot(ax, root, arm, range(0, gmax + 1), ASR.format("gpt4o_mini"), "asr")
            plot(ax, root, arm, range(0, gmax + 1), ASR.format("llama33_70b"), "asr", band=False, ls="--")
        ax.set_xlabel("Generation")
        ax.set_ylim(bottom=0)
        tidy(ax, title, "ASR", gmax)
    extra = [plt.Line2D([], [], color=SHARED, lw=1.5, ls="--", label="Secondary judge")]
    legend(fig, ["replace", "subsample", "accumulate"], extra, y=1.1)
    fig.tight_layout(w_pad=1.4)
    fig.savefig(out, bbox_inches="tight")


def secondary(it, base, out):
    panels = (
        ("sorry_bench__fulfillment__sorry_judge.json", "fulfillment", "SORRY-Bench fulfilment"),
        ("harmbench_test__classifier__harmbench_cls.json", "asr", "HarmBench classifier ASR"),
        ("xstest_safe__xstest__gpt4o_mini.json", "over_refusal", "XSTest over-refusal"),
        ("capability.json", ("ifeval", "prompt_level_strict_acc,none"), "IFEval accuracy"),
    )
    fig, axes = plt.subplots(2, 4, figsize=(5.0, 2.3))
    for r, (root, arms, gmax, name) in enumerate((
        (it, ["replace", "subsample"], 5, "Instruction-tuned"),
        (base, ["replace", "accumulate"], 10, "Base"),
    )):
        for c, (file, key, title) in enumerate(panels):
            ax = axes[r, c]
            for arm in arms:
                plot(ax, root, arm, range(0, gmax + 1), file, key)
            tidy(ax, title if r == 0 else None, name if c == 0 else None, gmax)
            if r == 1:
                ax.set_xlabel("Generation")
    legend(fig, ["replace", "subsample", "accumulate"], y=1.07)
    fig.tight_layout(h_pad=0.8, w_pad=0.8)
    fig.savefig(out, bbox_inches="tight")


def synthetic(it, out):
    ref = read(it / "seed0" / "tail_reference.json")
    panels = (
        ("tail_loss.json", "share_above_reference_p95", "Share above original p95", 0.05),
        ("tail_loss.json", ("quantiles", "p50"), "Median perplexity under Gen 0", ref["quantiles"]["p50"]),
        ("synthetic_audit.json", "distinct_2", "Distinct-2 diversity", None),
    )
    fig, axes = plt.subplots(1, 3, figsize=(4.8, 1.55))
    for ax, (file, key, title, base_line) in zip(axes, panels):
        for arm in ("replace", "subsample"):
            plot(ax, it, arm, range(1, 10), file, key, band=False, seeds=[0])
        if base_line is not None:
            ax.axhline(base_line, color=MUTED, lw=0.9, ls=":", zorder=1)
        ax.set_xticks(range(1, 10, 2))
        ax.set_xlabel("Generation")
        tidy(ax, title)
    extra = [plt.Line2D([], [], color=MUTED, lw=0.9, ls=":", label="Original data")]
    legend(fig, ["replace", "subsample"], extra, y=1.13)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(out, bbox_inches="tight")


def main(runs: str, out: str) -> None:
    it, base = Path(runs) / "main/gemma2_2b_it", Path(runs) / "base_model/gemma2_2b_kazdan"
    out = Path(out)
    tracking(it, base, out / "fig_tracking.pdf")
    judges(it, base, out / "fig_judges.pdf")
    secondary(it, base, out / "fig_secondary.pdf")
    synthetic(it, out / "fig_synthetic.pdf")


if __name__ == "__main__":
    main(*sys.argv[1:3])
