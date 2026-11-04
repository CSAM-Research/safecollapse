"""Map the published `prompt_key` values back to benchmark prompt IDs.

Scores are published with hashed keys, because benchmark IDs can describe the request. Build the
benchmarks from their public sources first (`safecollapse prepare -c core`, see README), then run:

    python scripts/map_prompt_keys.py data prompt_keys.csv

The output (prompt_key, benchmark, prompt_id) joins to `results/<study>/scores_long.csv.gz`.
Usage: python scripts/map_prompt_keys.py <data_dir> <out.csv>
"""
import csv
import hashlib
import json
import sys
from pathlib import Path


def main(data_dir: str, out: str) -> None:
    rows = 0
    with open(out, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["prompt_key", "benchmark", "prompt_id"])
        for name in ("harmbench_test", "sorry_bench", "xstest_safe"):
            for line in (Path(data_dir) / f"{name}.jsonl").open():
                pid = json.loads(line)["id"]
                writer.writerow([hashlib.sha256(pid.encode()).hexdigest()[:12], name, pid])
                rows += 1
    print(f"{rows} prompts written to {out}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
