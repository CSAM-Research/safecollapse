"""Command line: `safecollapse <command> -c <config> [--set key=value ...]`.

prepare        download and build every dataset (CPU)
check-judges   confirm each judge is live at its pinned provider (CPU, a few API calls)
pilot          the pilot: safety baseline, inclusion gate, timings, PILOT_REPORT.md (GPU)
run            train, generate and score every checkpoint in the plan; resumable (GPU)
judge          score any finished answers not yet judged (CPU; run anywhere)
status         what is done and what is pending (CPU)
analyze        tables, the planned tests and figures from stored scores (CPU)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter

from .config import load_config

log = logging.getLogger("safecollapse")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="safecollapse",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def command(name: str, help_: str):
        p = sub.add_parser(name, help=help_)
        p.add_argument(
            "-c",
            "--config",
            default="core",
            help="bundled config name (pilot, smoke, core, optional, extend10) or path",
        )
        p.add_argument(
            "--set",
            dest="overrides",
            action="append",
            default=[],
            metavar="KEY=VALUE",
            help="override a config value, e.g. seeds=[0]",
        )
        return p

    command("prepare", "download and build every dataset").add_argument(
        "--skip-sorry-bench", action="store_true", help="if the gated terms are not accepted yet"
    )
    command("check-judges", "check judges are live at their pinned providers").add_argument(
        "--no-call", action="store_true", help="only check the endpoint list; no paid call"
    )
    command("pilot", "run the pilot and write PILOT_REPORT.md")
    run = command("run", "run the study (resumable)")
    run.add_argument("--model", action="append", help="limit to these model keys")
    run.add_argument("--seed", type=int, action="append", help="limit to these seeds")
    run.add_argument("--no-judge", action="store_true", help="skip API judging (run it later)")
    run.add_argument(
        "--gpus",
        default=None,
        help="one worker per GPU, e.g. 0,1,2,3 (default: one worker, all GPUs)",
    )
    run.add_argument(
        "--break-locks",
        action="store_true",
        help="remove every node lock first (only if no other worker is running)",
    )
    run.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    command("judge", "judge finished answers")
    status = command("status", "done/pending stages (C = checkpoint on disk, L = running)")
    status.add_argument("--pending-only", action="store_true")
    status.add_argument("--verbose", action="store_true", help="list every pending stage")
    analyze = command("analyze", "tables, figures and the planned tests from stored scores")
    analyze.add_argument("--bootstrap", type=int, default=5000)
    analyze.add_argument("--no-figures", action="store_true")

    val = command("validate-judges", "agreement of our judges with existing human labels")
    val.add_argument("--datasets", default="jbb,sorry,wildguard")
    val.add_argument("--judges", default="gpt4o_mini,llama33_70b")
    val.add_argument("--limit", type=int, default=None)
    val.add_argument("--jev-model", default="jev-1.14.0")
    val.add_argument("--out", default=None, help="default: <runs_dir>/judge_validation")

    jv = command("judge-jev", "exploratory third judge (Jev) on HarmBench-test answers")
    jv.add_argument("--jev-model", default="jev-1.14.0")

    stage = sub.add_parser("stage", help=argparse.SUPPRESS)  # internal: one GPU stage per process
    stage.add_argument("name")
    stage.add_argument("--resolved", required=True)
    stage.add_argument("--node", required=True)
    stage.add_argument("--generator", dest="generator", action="store_true")
    stage.add_argument("--not-generator", dest="generator", action="store_false")
    stage.add_argument("--part", default="generate", choices=["generate", "metrics", "post"])

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )
    if args.command == "stage":
        return _stage(args)
    cfg = load_config(args.config, args.overrides)
    return COMMANDS[args.command](cfg, args)


def _stage(args) -> int:
    from . import pilot, pipeline
    from .plan import Node

    cfg = load_config(args.resolved)
    node = Node.from_key(args.node)
    stages = {
        "train": lambda: pipeline.stage_train(cfg, node),
        "fetch": lambda: pipeline.stage_fetch(cfg, node),
        "gpu": lambda: pipeline.stage_gpu(cfg, node, args.generator, args.part),
        "classify": lambda: pipeline.stage_classify(cfg, node.model),
        "sorry_judge": lambda: pipeline.stage_sorry_judge(cfg, node.model),
        "pilot_generate": lambda: pilot.stage_generate(cfg, node),
        "pilot_training": lambda: pilot.stage_training(cfg, node),
    }
    stages[args.name]()
    return 0


def _prepare(cfg, args) -> int:
    from .data.prepare import prepare

    counts = prepare(cfg, need_sorry_bench=not args.skip_sorry_bench)
    for name, n in counts.items():
        print(f"{name:32s} {n:6d}")
    return 0


def _check_judges(cfg, args) -> int:
    from .judges.openrouter import OpenRouter, endpoints
    from .judges.runner import JudgeCache, judge_items
    from .pipeline import judge_task
    from .plan import Layout

    keys = {cfg.judging.primary, cfg.judging.refusal_judge}
    for model in set(cfg.models) | set(cfg.pilot.candidates):
        keys.add(cfg.secondary_judge_for(model))
    ok = True
    client = None if args.no_call else OpenRouter()
    for key in sorted(keys):
        spec = cfg.judge(key)
        tags = {e.get("tag") for e in endpoints(spec.model)} | {
            (e.get("provider_name") or "").lower() for e in endpoints(spec.model)
        }
        live = all(p in tags for p in spec.provider_order)
        line = f"{key:12s} {spec.model:40s} pin={list(spec.provider_order)} live={live}"
        if client and live:
            item = {
                "id": "check",
                "prompt": "What is the capital of France?",
                "response": "The capital of France is Paris.",
            }
            cache = JudgeCache(Layout(cfg).cache_dir / f"{key}.jsonl")
            # Every task this judge is used for must return a parseable verdict.
            names = ["strongreject"] + (["xstest"] if key == cfg.judging.refusal_judge else [])
            for name in names:
                verdict = judge_items(
                    [item],
                    judge_task(cfg, name),
                    key,
                    spec,
                    client,
                    cache,
                    1,
                    cfg.judging.max_tokens,
                )[0]
                ok_call = "error" not in verdict
                line += f" {name}:served_by={verdict.get('provider')} verdict_ok={ok_call}"
                live = live and ok_call
        ok &= live
        print(line)
    return 0 if ok else 1


def _pilot(cfg, args) -> int:
    from .pilot import run_pilot

    print(run_pilot(cfg))
    return 0


def _run(cfg, args) -> int:
    from .pipeline import Runner
    from .plan import build_plan

    runner = Runner(cfg)
    if args.break_locks:
        for node in build_plan(cfg):
            runner.layout.unlock(node)
    if args.worker:  # started by a parent `run --gpus ...`
        runner.work(args.model, args.seed, judge=not args.no_judge, final_judge=False)
        return 0
    gpus = [g.strip() for g in args.gpus.split(",")] if args.gpus else None
    runner.run(models=args.model, seeds=args.seed, judge=not args.no_judge, gpus=gpus)
    return 0


def _judge(cfg, args) -> int:
    from .pipeline import judge_node
    from .plan import build_plan

    for node in build_plan(cfg):
        judge_node(cfg, node)
    return 0


def _status(cfg, args) -> int:
    from .pipeline import status

    for row in status(cfg):
        if args.pending_only and not row["pending"]:
            continue
        ckpt = "L" if row["locked"] else {None: " ", True: "C", False: "-"}[row["checkpoint"]]
        if args.verbose or not row["pending"]:
            todo = ", ".join(row["pending"]) or "done"
        else:  # compact: training steps by name, the rest counted by kind
            kinds = Counter(s.split(":")[0] for s in row["pending"])
            todo = ", ".join(f"{k} x{n}" if n > 1 else k for k, n in kinds.items())
        print(f"{row['label']:4s} {ckpt} {row['node']:45s} {todo}")
    return 0


def _analyze(cfg, args) -> int:
    from .analysis import confirmatory, figures, tracking
    from .analysis.collect import node_metrics, scores_long
    from .io import write_json
    from .plan import Layout

    out = Layout(cfg).root / "results"
    out.mkdir(parents=True, exist_ok=True)
    df = scores_long(cfg)
    df.to_csv(out / "scores_long.csv.gz", index=False)
    nodes = node_metrics(cfg)
    nodes.to_csv(out / "node_metrics.csv", index=False)
    ok = df[df["error"].isna()]
    summary = (
        ok.groupby(["model", "label", "job", "task", "judge"])
        .agg(
            n=("prompt", "size"),
            mean_score=("score", "mean"),
            asr=("score", lambda s: (s > 0.5).mean()),
            refusal_rate=("refused", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(out / "summary.csv", index=False)
    print(summary.to_string(index=False))
    for model in cfg.models:
        labels = set(df.loc[df["model"] == model, "label"])
        if {"G1", "R5"} <= labels and labels & {"S5", "A5"}:
            result = confirmatory.run(
                df,
                model,
                cfg.judging.primary,
                cfg.secondary_judge_for(model),
                cfg.judging.refusal_judge,
                bootstrap=args.bootstrap,
            )
            write_json(out / f"confirmatory_{model}.json", result)
            (out / f"table_confirmatory_{model}.tex").write_text(figures.confirmatory_table(result))
            print(json.dumps({model: result["primary"]}, indent=2, default=str))
        else:
            log.info("%s: confirmatory tests need G1, R5 and S5 (or A5); skipped", model)
        (out / f"table_checkpoints_{model}.tex").write_text(
            figures.checkpoint_table(df, cfg.judging.primary, model, nodes)
        )
        jb = figures.jailbreak_table(df, cfg.judging.primary, model)
        if jb:
            (out / f"table_jailbreaks_{model}.tex").write_text(jb)
        # Does harmful compliance track collapse? (headline, exploratory; DECISIONS.md A9)
        frame = tracking.checkpoint_frame(df, nodes, cfg.judging.primary, model)
        if "perplexity" in frame and frame["perplexity"].notna().any():
            frame.to_csv(out / f"tracking_{model}.csv", index=False)
            assoc = tracking.association(frame)
            write_json(out / f"tracking_{model}.json", assoc)
            (out / f"table_tracking_{model}.tex").write_text(tracking.latex_row("Base" if cfg.model(model).published else "Instruction-tuned", assoc) + "\n")
            print(json.dumps({f"tracking {model}": assoc}, indent=2, default=str))
            if not args.no_figures:
                (out / "figures").mkdir(parents=True, exist_ok=True)
                title = "Base" if cfg.model(model).published else "Instruction-tuned"
                tracking.figure(frame, out / "figures" / f"fig_tracking_{model}.pdf", title)
        if not args.no_figures:
            paths = figures.make_figures(
                df,
                nodes,
                cfg.judging.primary,
                cfg.secondary_judge_for(model),
                out / "figures",
                model,
            )
            log.info("%s: wrote %d figure files to %s", model, len(paths), out / "figures")
    return 0


def _validate_judges(cfg, args) -> int:
    from pathlib import Path

    from . import validation

    out = Path(args.out) if args.out else Path(cfg.paths.runs_dir) / "judge_validation"
    summary = validation.validate(
        cfg,
        args.datasets.split(","),
        args.judges.split(","),
        out,
        limit=args.limit,
        jev_model=args.jev_model,
    )
    (out / "table_judge_validation.tex").write_text(validation.latex_table(summary))
    print(json.dumps(summary, indent=2, default=str))
    return 0


def _judge_jev(cfg, args) -> int:
    from .io import read_jsonl, write_json, write_jsonl
    from .judges import jev
    from .plan import Layout, build_plan

    layout = Layout(cfg)
    prompts = {r["id"]: r for r in read_jsonl(layout.data_file("harmbench_test"))}
    done = 0
    for node in build_plan(cfg):
        out = layout.judgments(node, "harmbench_test", "jev", "jev")
        src = layout.completions(node, "harmbench_test")
        if out.exists() or not src.exists():
            continue
        items = [
            {
                "id": c["id"],
                "prompt": prompts[c["id"]]["messages"][-1]["content"],
                "response": c["completion"],
            }
            for c in read_jsonl(src)
        ]
        rows = jev.judge(items, layout.cache_dir / "jev.jsonl", model=args.jev_model)
        if any("error" in r for r in rows):
            log.warning(
                "%s: %d Jev errors; rerun to retry", node.key, sum("error" in r for r in rows)
            )
            continue  # not written, so a rerun retries the whole node from the cache
        write_jsonl(out, rows)
        write_json(
            layout.metrics_file(node, "harmbench_test__jev__jev"),
            {
                "n": len(rows),
                "unsafe_rate": sum(r["unsafe"] for r in rows) / len(rows),
                "model": args.jev_model,
            },
        )
        done += 1
    log.info("Jev judged %d checkpoint(s)", done)
    return 0


COMMANDS = {
    "prepare": _prepare,
    "check-judges": _check_judges,
    "pilot": _pilot,
    "run": _run,
    "judge": _judge,
    "status": _status,
    "analyze": _analyze,
    "validate-judges": _validate_judges,
    "judge-jev": _judge_jev,
}

if __name__ == "__main__":
    sys.exit(main())
