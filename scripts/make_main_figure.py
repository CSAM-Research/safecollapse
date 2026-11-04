"""Main figure: harmful compliance and collapse by generation, both categories (2 x 2 small
multiples, one y-scale per panel; never a second axis). Reads the analysis outputs
`tracking_<model>.csv` (one row per checkpoint and seed). Run by `make results`.

Usage: python3 make_main_figure.py <it_results_dir> <base_results_dir> <out.pdf>
"""
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Validated categorical slots (dataviz reference palette, all-pairs CVD safe), fixed per arm.
ARM = {
    "replace": {"color": "#eb6834", "marker": "o", "label": "Replace (collapse)"},
    "subsample": {"color": "#2a78d6", "marker": "s", "label": "Subsample"},
    "accumulate": {"color": "#1baf7a", "marker": "^", "label": "Accumulate"},
}
SHARED = "#52514e"  # Gens 0 and 1 belong to both arms
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
    }
)


def load(path: Path) -> dict:
    """(arm, generation) -> {metric: [values over seeds]}; G0/G1 stored under arm 'shared'."""
    out = defaultdict(lambda: defaultdict(list))
    with open(path) as fh:
        for row in csv.DictReader(fh):
            arm = "shared" if row["label"] in ("G0", "G1") else row["arm"]
            for metric in ("asr", "perplexity"):
                if row.get(metric) not in (None, "", "nan"):
                    out[(arm, int(float(row["generation"])))][metric].append(float(row[metric]))
    return out


def series(data: dict, arm: str, metric: str):
    """Generations, means, mins, maxes for one arm, starting from the shared Gen 1."""
    gens = sorted(g for (a, g) in data if a == arm)
    pts = [(1, data[("shared", 1)][metric])] + [(g, data[(arm, g)][metric]) for g in gens]
    pts = [(g, v) for g, v in pts if v]
    return (
        [g for g, _ in pts],
        [statistics.mean(v) for _, v in pts],
        [min(v) for _, v in pts],
        [max(v) for _, v in pts],
    )


def panel(ax, data: dict, arms: list[str], metric: str, ylabel: str, title: str | None):
    for arm in arms:
        st = ARM[arm]
        g, m, lo, hi = series(data, arm, metric)
        ax.fill_between(g, lo, hi, color=st["color"], alpha=0.14, linewidth=0)
        ax.plot(g, m, color=st["color"], lw=1.8, marker=st["marker"], ms=5,
                markeredgecolor="white", markeredgewidth=0.8, zorder=3)
    for gen in (0, 1):
        vals = data[("shared", gen)][metric]
        if vals:
            ax.plot([gen], [statistics.mean(vals)], marker="D", ms=5, color=SHARED,
                    markeredgecolor="white", markeredgewidth=0.8, zorder=4)
    g0, g1 = data[("shared", 0)][metric], data[("shared", 1)][metric]
    if g0 and g1:
        ax.plot([0, 1], [statistics.mean(g0), statistics.mean(g1)], color=SHARED, lw=1.4, zorder=2)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, color=INK, loc="left", fontweight="bold")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5, width=0.6)


def load_nodes(root: Path, seeds: list[int], arms: list[str], gmax: int) -> dict:
    """Same structure as load(), read from node metrics (the analysis outputs stop at Gen 5)."""
    def value(node: Path, name: str, key: str):
        try:
            return json.loads((node / "metrics" / name).read_text().replace("NaN", "null")).get(key)
        except (OSError, ValueError):
            return None

    out = defaultdict(lambda: defaultdict(list))
    nodes = [("shared", 0, root / "gen0")]
    nodes += [("shared", 1, root / f"seed{s}/shared/gen1") for s in seeds]
    nodes += [
        (arm, g, root / f"seed{s}/{arm}/gen{g}")
        for s in seeds for arm in arms for g in range(2, gmax + 1)
    ]
    for arm, g, node in nodes:
        asr = value(node, "harmbench_test__strongreject__gpt4o_mini.json", "asr")
        ppl = value(node, "heldout_nll.json", "perplexity")
        if asr is not None:
            out[(arm, g)]["asr"].append(asr)
        if ppl is not None:
            out[(arm, g)]["perplexity"].append(ppl)
    return out


def draw(cols: list, out: str) -> None:
    """2 x 2 grid: ASR on top, perplexity below, one column per category."""
    fig, axes = plt.subplots(2, 2, figsize=(5.6, 3.3), sharex="col")
    for c, (data, arms, title) in enumerate(cols):
        panel(axes[0, c], data, arms, "asr", "ASR (primary judge)", title)
        panel(axes[1, c], data, arms, "perplexity", "Held-out perplexity", None)
        gmax = max(g for (_, g) in data)
        axes[1, c].set_xticks(range(0, gmax + 1, 2 if gmax > 5 else 1))
        axes[1, c].set_xticklabels(
            [str(t) for t in range(0, gmax + 1, 2 if gmax > 5 else 1)]
        )
        axes[1, c].set_xlabel("Generation")
        axes[0, c].set_ylim(bottom=0)
    handles = [
        plt.Line2D([], [], color=ARM[a]["color"], marker=ARM[a]["marker"], lw=1.8, ms=5,
                   markeredgecolor="white", label=ARM[a]["label"])
        for a in ("replace", "subsample", "accumulate")
    ] + [plt.Line2D([], [], color=SHARED, marker="D", lw=1.4, ms=5, markeredgecolor="white",
                    label="Shared Gens 0, 1")]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=7.5,
               bbox_to_anchor=(0.5, 1.03), handlelength=2.0, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=0.6, w_pad=1.6)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(Path(out).with_suffix(".png")), bbox_inches="tight", dpi=200)


def main(it_dir: str, base_dir: str, out: str, runs: str | None = None) -> None:
    it = load(Path(it_dir) / "tracking_gemma2_2b_it.csv")
    base = load(Path(base_dir) / "tracking_gemma2_2b_kazdan.csv")
    draw(
        [
            (it, ["replace", "subsample"], "Instruction-tuned (Gemma-2-2B-IT)"),
            (base, ["replace", "accumulate"], "Base (Gemma-2-2B)"),
        ],
        out,
    )
    if runs:
        arms_it, arms_base = ["replace", "subsample"], ["replace", "accumulate"]
        draw(
            [
                (load_nodes(Path(runs) / "main/gemma2_2b_it", [0], arms_it, 10), arms_it,
                 "Instruction-tuned, seed 0"),
                (load_nodes(Path(runs) / "base_model/gemma2_2b_kazdan", [0, 1, 2], arms_base, 10),
                 arms_base, "Base, three seeds"),
            ],
            str(Path(out).with_name("fig_late.pdf")),
        )


if __name__ == "__main__":
    main(*sys.argv[1:5])
