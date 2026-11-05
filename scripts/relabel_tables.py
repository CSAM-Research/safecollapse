"""Rewrite the analysis code's checkpoint labels (G0, G1, R5, S5, A5, ...) in the copied LaTeX
tables into the paper's names (Gen 0, Gen 1, then Gen 5 with superscript rep, sub or acc). Run
by `make results`;
the analysis code and data keep their own labels. Idempotent."""
import re
import sys
from pathlib import Path

PAT = re.compile(r"(?<![A-Za-z0-9\\{:_-])(G[01]|[RSA](?:10|[2-9]))(?![A-Za-z0-9}_])")
SUP = "Gen~%s$^{\\mathrm{%s}}$"
NAME = {"G": "Gen~%s", "R": SUP % ("%s", "rep"), "S": SUP % ("%s", "sub"), "A": SUP % ("%s", "acc")}


def relabel(text: str) -> str:
    return PAT.sub(lambda m: NAME[m.group(1)[0]] % m.group(1)[1:], text)


def minus_signs(text: str) -> str:
    """Typeset negative numbers with a minus sign, not a hyphen."""
    return re.sub(r"(?<=[ \[])-(?=\d)", "$-$", text).replace("RQ4:", "H4:")


def replace_first(text: str) -> str:
    """Within each generation, list the replace row before the accumulate row."""
    lines = text.split("\n")
    for i in range(len(lines) - 1):
        a, r = lines[i].split(" & ")[0], lines[i + 1].split(" & ")[0]
        if a.endswith("{acc}}$") and r == a.replace("{acc}", "{rep}"):
            lines[i], lines[i + 1] = lines[i + 1], lines[i]
    return "\n".join(lines)


if __name__ == "__main__":
    for path in map(Path, sys.argv[1:]):
        path.write_text(minus_signs(replace_first(relabel(path.read_text()))))
