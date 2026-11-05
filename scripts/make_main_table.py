"""Build the main-text results table (both categories side by side, per checkpoint: ASR, refusals,
held-out perplexity, IFEval; mean +- sd over seeds) to Gen 10, from the node metrics, for every
seed that reached that checkpoint. Run by `make results`.
Usage: python3 make_main_table.py <runs_dir> <out>
"""
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def load(root: Path) -> dict:
    """label -> {metric: [values over seeds]}"""
    def read(node: Path, name: str) -> dict:
        try:
            return json.loads((node / "metrics" / name).read_text().replace("NaN", "null"))
        except (OSError, ValueError):
            return {}

    nodes = [("G0", root / "gen0")]
    for seed in sorted(root.glob("seed*")):
        nodes.append(("G1", seed / "shared/gen1"))
        for arm, letter in (("replace", "R"), ("subsample", "S"), ("accumulate", "A")):
            nodes += [(f"{letter}{d.name[3:]}", d) for d in (seed / arm).glob("gen*")]
    out = defaultdict(lambda: defaultdict(list))
    for label, node in nodes:
        judged = read(node, "harmbench_test__strongreject__gpt4o_mini.json")
        values = {
            "asr": judged.get("asr"),
            "refused": judged.get("refusal_rate"),
            "perplexity": read(node, "heldout_nll.json").get("perplexity"),
            "ifeval": read(node, "capability.json").get("ifeval", {}).get("prompt_level_strict_acc,none"),
        }
        for metric, value in values.items():
            if value is not None:
                out[label][metric].append(value)
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


def main(runs: str, out: str) -> None:
    it = load(Path(runs) / "main/gemma2_2b_it")
    base = load(Path(runs) / "base_model/gemma2_2b_kazdan")
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
        lines.append(r"\addlinespace[1pt]")
        lines.append(row(rf"Gen~{t}$^{{\mathrm{{rep}}}}$", it, base, f"R{t}", f"R{t}"))
        lines.append(row(rf"Gen~{t}$^{{\mathrm{{sub/acc}}}}$", it, base, f"S{t}", f"A{t}"))
    lines += [r"\bottomrule", r"\end{tabular}"]
    Path(out).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:3])
