"""`safecollapse prepare`: download every dataset at its pinned revision and write the processed
files the pipeline reads, plus `hashes.json` (SHA-256 of every raw and processed file)."""

from __future__ import annotations

from pathlib import Path

from ..config import Config
from ..io import read_json, sha256_file, write_json, write_jsonl
from ..plan import Layout
from . import benchmarks, helpsteer
from .download import fetch, github_url, hf_url

HELPSTEER = "nvidia/HelpSteer2"
SORRY_BENCH = "sorry-bench/sorry-bench-202503"
HARMBENCH = "centerforaisafety/HarmBench"
XSTEST = "paul-rottger/xstest"


def raw_sources(cfg: Config) -> dict[str, str]:
    d = cfg.data
    hb = "data/behavior_datasets/harmbench_behaviors_text_{}.csv"
    return {
        "helpsteer_train.jsonl.gz": hf_url(HELPSTEER, "train.jsonl.gz", d.helpsteer_revision),
        "helpsteer_validation.jsonl.gz": hf_url(
            HELPSTEER, "validation.jsonl.gz", d.helpsteer_revision
        ),
        "harmbench_val.csv": github_url(HARMBENCH, hb.format("val"), d.harmbench_commit),
        "harmbench_test.csv": github_url(HARMBENCH, hb.format("test"), d.harmbench_commit),
        "xstest_prompts.csv": github_url(XSTEST, "xstest_prompts.csv", d.xstest_commit),
        "sorry_bench_question.jsonl": hf_url(SORRY_BENCH, "question.jsonl", d.sorry_bench_revision),
    }


def prepare(cfg: Config, need_sorry_bench: bool = True) -> dict:
    layout = Layout(cfg)
    raw_dir = layout.data_dir / "raw"
    hashes_path = layout.data_dir / "hashes.json"
    known = read_json(hashes_path) if hashes_path.exists() else {}

    raw: dict[str, Path] = {}
    for name, url in raw_sources(cfg).items():
        if name.startswith("sorry_bench") and not need_sorry_bench:
            continue
        raw[name] = fetch(url, raw_dir / name)
        _check_unchanged(known, f"raw/{name}", raw[name])

    outputs: dict[str, list[dict]] = {}
    train_rows = helpsteer.load_rows(raw["helpsteer_train.jsonl.gz"])
    for seed in sorted(set(cfg.seeds) | {0}):
        outputs[layout.round_file(seed).stem] = helpsteer.select_round(
            train_rows, cfg.data.round_size, seed, cfg.data.one_response_per_prompt
        )
    outputs[layout.heldout_file().stem] = helpsteer.select_heldout(
        helpsteer.load_rows(raw["helpsteer_validation.jsonl.gz"]), cfg.data.heldout_size
    )
    for split in ("val", "test"):
        name = f"harmbench_{split}"
        outputs[name] = benchmarks.load_harmbench(raw[f"{name}.csv"], split)
    outputs["xstest_safe"] = benchmarks.load_xstest_safe(raw["xstest_prompts.csv"])
    if "sorry_bench_question.jsonl" in raw:
        outputs["sorry_bench"] = benchmarks.load_sorry_bench(raw["sorry_bench_question.jsonl"])
    for name, records in outputs.items():
        benchmarks.check_count(name, records)

    hashes = {f"raw/{name}": sha256_file(path) for name, path in raw.items()}
    for name, records in outputs.items():
        path = write_jsonl(layout.data_dir / f"{name}.jsonl", records)
        hashes[path.name] = sha256_file(path)
    write_json(hashes_path, {**known, **hashes, "_sources": raw_sources(cfg)})
    return {name: len(records) for name, records in outputs.items()}


def _check_unchanged(known: dict, key: str, path: Path) -> None:
    """A raw file must never change under a pinned revision; if it does, stop."""
    if key in known and known[key] != sha256_file(path):
        raise RuntimeError(f"{path} differs from the hash recorded earlier; refusing to continue")
