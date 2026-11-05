"""Build the main-text results table (both categories side by side, per checkpoint: ASR, held-out
perplexity, IFEval; mean +- sd over seeds) from the analysis outputs `tracking_<model>.csv`.
Run by `make results`. Usage: python3 make_main_table.py <it_results_dir> <base_results_dir> <out>
"""
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def load(results_dir: Path, model: str) -> dict:
    """label -> {metric: [values over seeds]}"""
    out = defaultdict(lambda: defaultdict(list))
    with open(results_dir / f"tracking_{model}.csv") as fh:
        for row in csv.DictReader(fh):
            for metric in ("asr", "refused", "perplexity", "ifeval"):
                if row.get(metric) not in (None, "", "nan"):
                    out[row["label"]][metric].append(float(row[metric]))
    return out


def cell(values: list[float]) -> str:
    if not values:
        return "--"
    if len(values) == 1:
        return f"{values[0]:.2f}"
    return f"{statistics.mean(values):.2f} $\\pm$ {statistics.stdev(values):.2f}"


def row(name: str, it: dict, base: dict, it_label: str, base_label: str) -> str:
    metrics = ("asr", "refused", "perplexity", "ifeval")
    cells = [cell(it.get(it_label, {}).get(m, [])) for m in metrics]
    cells += [cell(base.get(base_label, {}).get(m, [])) for m in metrics]
    return f"{name} & " + " & ".join(cells) + r" \\"


def main(it_dir: str, base_dir: str, out: str) -> None:
    it = load(Path(it_dir), "gemma2_2b_it")
    base = load(Path(base_dir), "gemma2_2b_kazdan")
    gens = sorted(
        {int(lbl[1:]) for lbl in list(it) + list(base) if lbl[0] in "RSA" and lbl[1:].isdigit()}
    )
    lines = [
        r"\begin{tabular}{lcccccccc}",
        r"\toprule",
        r" & \multicolumn{4}{c}{Instruction-tuned (Gemma-2-2B-IT)} & \multicolumn{4}{c}{Base (Gemma-2-2B)} \\",
        r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
        r"Checkpoint & ASR & Refused & PPL & IFEval & ASR & Refused & PPL & IFEval \\",
        r"\midrule",
        row("Gen~0", it, base, "G0", "G0"),
        row("Gen~1", it, base, "G1", "G1"),
    ]
    for t in gens:
        lines.append(r"\addlinespace[2pt]")
        lines.append(row(rf"Gen~{t}$^{{\mathrm{{rep}}}}$", it, base, f"R{t}", f"R{t}"))
        lines.append(row(rf"Gen~{t}$^{{\mathrm{{sub/acc}}}}$", it, base, f"S{t}", f"A{t}"))
    lines += [r"\bottomrule", r"\end{tabular}"]
    Path(out).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:4])
