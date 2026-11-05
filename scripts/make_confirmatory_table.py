"""Confirmatory table: estimate, planned bootstrap 95% interval (Wald for the slope), test p-values.
Run by `make results`. Usage: python3 make_confirmatory_table.py <results_dir> <model> <arm_tag> <out>
"""
import json
import sys
from pathlib import Path


def num(x: float, sign: bool = True) -> str:
    text = f"{x:+.3f}" if sign else f"{x:.3f}"
    return text.replace("-", "$-$")


def pval(p: float) -> str:
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def main(results: str, model: str, arm: str, out: str) -> None:
    data = json.load(open(Path(results) / f"confirmatory_{model}.json"))
    p, robust = data["primary"], data["robust"]
    rep, keep = r"Gen~5$^{\mathrm{rep}}$", rf"Gen~5$^{{\mathrm{{{arm}}}}}$"
    rows = [
        ("H1", rf"H1: {rep} $-$ Gen~1"),
        ("H2", "H2: slope, replace"),
        ("H3", rf"H3: {keep} $-$ {rep}"),
        ("RQ4", rf"H4: over-refusal {rep} $-$ Gen~1"),
    ]
    lines = [r"\begin{tabular}{lrrrrl}", r"\toprule",
             r"Test & Estimate & 95\% CI & $p$ & $p_{\mathrm{Holm}}$ & Robust \\", r"\midrule"]
    for key, name in rows:
        v = p[key]
        est = v.get("estimate", v.get("slope"))
        b = v.get("bootstrap")
        lo, hi = (b["ci_low"], b["ci_high"]) if b else (v["ci_low"], v["ci_high"])
        mark = "" if b else r"$^{\dagger}$"
        flag = {True: "yes", False: "no"}.get(robust.get(key), "--")
        lines.append(f"{name} & {num(est)} & [{num(lo)}, {num(hi)}]{mark} & {pval(v['p'])} & "
                     f"{pval(v['p_holm'])} & {flag} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    Path(out).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:5])
