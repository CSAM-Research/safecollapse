"""Late-collapse table: per-generation results from the node metrics. The instruction-tuned columns
are seed 0; base columns (mean over the seeds available) appear once Gens 6-10 are in. Run by
`make results`. Usage: python3 make_late_table.py <runs_dir> <out>"""
import json
import statistics
import sys
from pathlib import Path


def metrics(node: Path) -> dict:
    def load(name):
        try:
            return json.loads((node / "metrics" / name).read_text().replace("NaN", "null"))
        except (OSError, ValueError):
            return {}

    return {
        "asr": load("harmbench_test__strongreject__gpt4o_mini.json").get("asr"),
        "ppl": load("heldout_nll.json").get("perplexity"),
        "over": load("xstest_safe__xstest__gpt4o_mini.json").get("over_refusal"),
    }


def node(root: Path, seed: int, arm: str, gen: int) -> Path:
    return root / f"seed{seed}" / ("shared/gen1" if gen == 1 else f"{arm}/gen{gen}")


def mean(root: Path, seeds: list[int], arm: str, gen: int, key: str):
    vals = [metrics(node(root, s, arm, gen))[key] for s in seeds]
    vals = [v for v in vals if v is not None]
    return statistics.mean(vals) if vals else None


def fmt(v) -> str:
    return "--" if v is None else f"{v:.2f}"


def main(runs: str, out: str) -> None:
    it, base = Path(runs) / "main/gemma2_2b_it", Path(runs) / "base_model/gemma2_2b_kazdan"
    gens = (1, 5, 6, 7, 8, 9, 10)
    has_base = mean(base, [0, 1, 2], "replace", 10, "asr") is not None
    head = r" & \multicolumn{6}{c}{Instruction-tuned, seed 0}"
    sub = r"Gen. & ASR rep. & ASR sub. & PPL rep. & PPL sub. & Over-ref. rep. & Over-ref. sub."
    rule = r"\cmidrule(lr){2-7}"
    spec = "lcccccc"
    if has_base:
        head += r" & \multicolumn{4}{c}{Base, three seeds}"
        sub += r" & ASR rep. & ASR acc. & PPL rep. & PPL acc."
        rule += r"\cmidrule(lr){8-11}"
        spec += "cccc"
    lines = [rf"\begin{{tabular}}{{{spec}}}", r"\toprule", head + r" \\", rule, sub + r" \\", r"\midrule"]
    for g in gens:
        cells = [str(g)]
        for key in ("asr", "ppl", "over"):
            cells += [fmt(mean(it, [0], arm, g, key)) for arm in ("replace", "subsample")]
        if has_base:
            for key in ("asr", "ppl"):
                cells += [fmt(mean(base, [0, 1, 2], arm, g, key)) for arm in ("replace", "accumulate")]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    Path(out).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:3])
