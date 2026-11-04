"""The pilot (DESIGN.md Section 8): score each candidate untouched on the HarmBench validation
split with both judges, apply the inclusion gate, time generation and training, and turn the
timings into a budget. Writes `PILOT_REPORT.md` in the pilot's run folder.
"""

from __future__ import annotations

import logging
import math
import subprocess
from statistics import fmean, pstdev

from .config import Config
from .io import read_json, read_jsonl, write_json, write_jsonl, write_text
from .judges.rubric import STRONGREJECT
from .metrics import safety
from .pipeline import Runner, judge_node
from .plan import Layout, Node, gen0

log = logging.getLogger("safecollapse")

TIMING_JOB = "synthetic_timing"
DESIGN_SEEDS = 3  # the main run's seeds, for the power estimate


def run_pilot(cfg: Config) -> str:
    runner = Runner(cfg)
    layout = runner.layout
    bench = cfg.pilot.benchmark
    for model in cfg.pilot.candidates:
        node = gen0(model)
        # One candidate failing (e.g. a 7B model out of memory in the timing run) must not stop
        # the others; the failure is recorded and shown in the report.
        try:
            if not (layout.done(node, f"gen:{bench}") and layout.done(node, f"gen:{TIMING_JOB}")):
                runner.subprocess("pilot_generate", node)
            judge_node(cfg, node, only_jobs=[bench])
            judge_repeats(cfg, node)
            if not layout.done(node, "pilot_training"):
                runner.subprocess("pilot_training", node)
        except (subprocess.CalledProcessError, RuntimeError) as exc:
            log.error("pilot: %s failed: %s", model, exc)
            write_json(layout.metrics_file(node, "pilot_error"), {"error": str(exc)})
    report = render_report(cfg)
    write_text(layout.root / "PILOT_REPORT.md", report)
    return report


# GPU stages (run in a fresh process by `safecollapse stage pilot_*`)


def stage_generate(cfg: Config, node: Node) -> None:
    from .generation import Job, run_jobs
    from .modeling import require_gpu

    require_gpu()
    layout = Layout(cfg)
    bench = cfg.pilot.benchmark
    timing_prompts = read_jsonl(layout.round_file(0))[: cfg.pilot.synthetic_timing_prompts]
    jobs = [
        Job(
            bench,
            read_jsonl(layout.data_file(bench)),
            cfg.evaluation.decoding,
            layout.completions(node, bench),
        ),
        Job(
            TIMING_JOB, timing_prompts, cfg.synthetic, layout.node_dir(node) / f"{TIMING_JOB}.jsonl"
        ),
    ]
    spec = cfg.model(node.model)
    stats = run_jobs(spec.hf_id, spec.revision, spec, jobs, cfg.generation, seed=0)
    write_json(layout.metrics_file(node, "pilot_generation"), stats)
    layout.mark_done(node, f"gen:{bench}")
    layout.mark_done(node, f"gen:{TIMING_JOB}")


def stage_training(cfg: Config, node: Node) -> None:
    from .modeling import require_gpu
    from .training import train

    require_gpu()
    layout = Layout(cfg)
    spec = cfg.model(node.model)
    stats = train(
        spec.hf_id,
        spec.revision,
        spec,
        layout.round_file(0),
        layout.node_dir(node) / "timing_unused",
        cfg.training,
        seed=0,
        max_steps=cfg.pilot.timing_steps,
        save=False,
    )
    write_json(layout.metrics_file(node, "pilot_training"), stats)
    layout.mark_done(node, "pilot_training")


# Report


def judge_repeats(cfg: Config, node: Node) -> None:
    """Judge every pilot answer again (fresh API calls) to measure each judge's stability."""
    from .judges.openrouter import OpenRouter
    from .judges.runner import JudgeCache, judge_items

    layout, bench = Layout(cfg), cfg.pilot.benchmark
    prompts = {r["id"]: r for r in read_jsonl(layout.data_file(bench))}
    items = [
        {
            "id": c["id"],
            "prompt": prompts[c["id"]]["messages"][-1]["content"],
            "response": c["completion"],
        }
        for c in read_jsonl(layout.completions(node, bench))
    ]
    client = OpenRouter(timeout_s=cfg.judging.timeout_s, max_retries=cfg.judging.max_retries)
    for judge in (cfg.judging.primary, cfg.secondary_judge_for(node.model)):
        for r in range(1, cfg.pilot.judge_repeats + 1):
            path = layout.judgments(node, bench, "strongreject", f"{judge}__repeat{r}")
            if path.exists():
                continue
            cache = JudgeCache(layout.cache_dir / f"{judge}.jsonl")
            rows = judge_items(
                items,
                STRONGREJECT,
                judge,
                cfg.judge(judge),
                client,
                cache,
                cfg.judging.concurrency,
                cfg.judging.max_tokens,
                salt=f"repeat{r}",
            )
            write_jsonl(path, rows)


def _stability(layout, node, bench, judge, repeats: int) -> dict:
    """Test-retest: share of identical verdicts and kappa on `refused`, first call vs repeats."""
    from .analysis.stats import cohen_kappa

    first = {v["id"]: v for v in read_jsonl(layout.judgments(node, bench, "strongreject", judge))}
    same, a, b = [], [], []
    for r in range(1, repeats + 1):
        path = layout.judgments(node, bench, "strongreject", f"{judge}__repeat{r}")
        if not path.exists():
            continue
        for v in read_jsonl(path):
            u = first.get(v["id"])
            if u is None or "error" in u or "error" in v:
                continue
            same.append(all(u[k] == v[k] for k in ("refused", "specific", "convincing")))
            a.append(u["refused"])
            b.append(v["refused"])
    if not same:
        return {}
    return {"n": len(same), "identical": sum(same) / len(same), "kappa_refused": cohen_kappa(a, b)}


def model_summary(cfg: Config, model: str) -> dict:
    layout = Layout(cfg)
    node, bench = gen0(model), cfg.pilot.benchmark
    out = {"model": model, "hf_id": cfg.model(model).hf_id}
    if not layout.completions(node, bench).exists():
        error = layout.metrics_file(node, "pilot_error")
        return {**out, "pilot_error": read_json(error) if error.exists() else {"error": "not run"}}
    completions = {r["id"]: r["completion"] for r in read_jsonl(layout.completions(node, bench))}
    for role, judge in (
        ("primary", cfg.judging.primary),
        ("secondary", cfg.secondary_judge_for(model)),
    ):
        path = layout.judgments(node, bench, "strongreject", judge)
        if not path.exists():
            continue
        verdicts = read_jsonl(path)
        summary = safety.summarize_strongreject(verdicts, completions)
        scores = [v["score"] for v in verdicts if "error" not in v]
        summary["score_sd"] = pstdev(scores) if len(scores) > 1 else None
        out[role] = {"judge": judge, **summary}
    if "primary" in out:
        out["gate"] = safety.gate(out["primary"], cfg.gate)
    if "primary" in out and "secondary" in out:
        out["agreement"] = _agreement(
            layout, node, bench, cfg.judging.primary, cfg.secondary_judge_for(model)
        )
    for role in ("primary", "secondary"):
        if role in out:
            out[role]["stability"] = _stability(
                layout, node, bench, out[role]["judge"], cfg.pilot.judge_repeats
            )
    for name in ("pilot_generation", "pilot_training", "pilot_error"):
        path = layout.metrics_file(node, name)
        if path.exists():
            out[name] = read_json(path)
    out["budget"] = budget(cfg, out)
    return out


def _agreement(layout, node, bench, a, b) -> dict:
    from .analysis.stats import cohen_kappa, spearman

    va = {v["id"]: v for v in read_jsonl(layout.judgments(node, bench, "strongreject", a))}
    vb = {v["id"]: v for v in read_jsonl(layout.judgments(node, bench, "strongreject", b))}
    ids = [i for i in va if i in vb and "error" not in va[i] and "error" not in vb[i]]
    if not ids:
        return {}
    return {
        "n": len(ids),
        "kappa_refused": cohen_kappa(
            [va[i]["refused"] for i in ids], [vb[i]["refused"] for i in ids]
        ),
        "spearman_score": spearman([va[i]["score"] for i in ids], [vb[i]["score"] for i in ids]),
    }


def budget(cfg: Config, s: dict) -> dict | None:
    """GPU hours for the core design from the measured speeds (rough; DESIGN.md Section 9)."""
    gen, tr = s.get("pilot_generation"), s.get("pilot_training")
    if not gen or not tr:
        return None
    synth, ev = gen[TIMING_JOB], gen[cfg.pilot.benchmark]
    steps = math.ceil(cfg.data.round_size / cfg.training.effective_batch) * cfg.training.epochs
    train_h = steps * tr["seconds_per_step"] / 3600
    synth_h = cfg.data.round_size * synth["seconds"] / max(1, synth["prompts"]) / 3600
    # ~930 evaluation prompts per checkpoint (HarmBench test, SORRY-Bench, XSTest); the small
    # pilot batch under-uses vLLM, so this errs high.
    eval_h = 930 * ev["seconds"] / max(1, ev["prompts"]) / 3600
    n_arms = len([a for a in cfg.arms if a in ("replace", "subsample")]) or 2
    trained = 1 + (cfg.generations - 1) * n_arms
    generating = 1 + max(0, cfg.generations - 2) * n_arms
    per_seed = trained * (train_h + eval_h) + generating * synth_h
    return {
        "train_hours_per_checkpoint": round(train_h, 3),
        "synthetic_hours_per_round": round(synth_h, 3),
        "eval_hours_per_checkpoint": round(eval_h, 3),
        "checkpoints_per_seed": trained,
        "core_gpu_hours_all_seeds": round(per_seed * DESIGN_SEEDS + eval_h, 1),
        "peak_training_memory_gb": tr.get("peak_memory_gb"),
    }


def render_report(cfg: Config) -> str:
    rows = [model_summary(cfg, m) for m in cfg.pilot.candidates]
    write_json(Layout(cfg).root / "pilot_summary.json", rows)
    pct = lambda x: "-" if x is None else f"{100 * x:.1f}%"  # noqa: E731
    num = lambda x, d=3: "-" if x is None else f"{x:.{d}f}"  # noqa: E731
    lines = [
        "# Pilot report",
        "",
        f"HarmBench validation split ({cfg.pilot.benchmark}), untouched models, greedy decoding, "
        f"{cfg.evaluation.decoding.max_new_tokens} new tokens. Gate: ASR <= "
        f"{pct(cfg.gate.max_asr)} and clear refusals >= {pct(cfg.gate.min_clear_refusal)} "
        "(primary judge).",
        "",
        "## Safety baseline",
        "",
        "| Model | Judge | n (errors) | Mean score | SD | ASR | Refused | Clear refusals | "
        "Ambiguous | Gate |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        for role in ("primary", "secondary"):
            s = r.get(role)
            if not s:
                continue
            gate = ("PASS" if r["gate"]["passed"] else "fail") if role == "primary" else ""
            lines.append(
                f"| {r['model']} | {s['judge']} | {s['n']} ({s['n_errors']}) | "
                f"{num(s.get('mean_score'))} | {num(s.get('score_sd'))} | {pct(s.get('asr'))} | "
                f"{pct(s.get('refusal_rate'))} | {pct(s.get('clear_refusal_rate'))} | "
                f"{pct(s.get('ambiguity_rate'))} | {gate} |"
            )
    lines += [
        "",
        "## Judge agreement (primary vs secondary)",
        "",
        "| Model | n | Kappa (refused) | Spearman (score) |",
        "|---|---|---|---|",
    ]
    for r in rows:
        a = r.get("agreement") or {}
        lines.append(
            f"| {r['model']} | {a.get('n', '-')} | {num(a.get('kappa_refused'), 2)} | "
            f"{num(a.get('spearman_score'), 2)} |"
        )
    lines += [
        "",
        "## Judge stability (same answer judged twice, temperature 0)",
        "",
        "| Model | Judge | n | Identical verdicts | Kappa (refused) |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        for role in ("primary", "secondary"):
            st = (r.get(role) or {}).get("stability") or {}
            if st:
                lines.append(
                    f"| {r['model']} | {r[role]['judge']} | {st['n']} | "
                    f"{pct(st['identical'])} | {num(st['kappa_refused'], 2)} |"
                )
    lines += [
        "",
        "## Speed and budget (core design: Gen 1, R2-R5, S2-S5)",
        "",
        "| Model | Train h / ckpt | Synthetic h / round | Eval h / ckpt | Peak train GB | "
        "Core GPU-h (all seeds) |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        b = r.get("budget") or {}
        cells = [
            b.get(k, "-")
            for k in (
                "train_hours_per_checkpoint",
                "synthetic_hours_per_round",
                "eval_hours_per_checkpoint",
                "peak_training_memory_gb",
                "core_gpu_hours_all_seeds",
            )
        ]
        lines.append(f"| {r['model']} | " + " | ".join(map(str, cells)) + " |")
    sds = [r["primary"]["score_sd"] for r in rows if r.get("primary", {}).get("score_sd")]
    if sds:
        from .analysis.stats import min_detectable_difference

        sd = max(sds + [0.25])  # untouched models score near 0; use at least the design's 0.25
        mde = min_detectable_difference(sd, 240, DESIGN_SEEDS)
        lines += [
            "",
            "## Power",
            "",
            f"Per-prompt score SD (primary judge): mean {fmean(sds):.3f}, max "
            f"{max(sds):.3f} across candidates. Untouched models score near 0, so the "
            f"estimate uses at least the design's 0.25. With SD {sd:.2f}, 240 test prompts "
            f"and {DESIGN_SEEDS} seeds, the minimum detectable difference in mean score is "
            f"about {mde:.3f}.",
        ]
    failed = [r for r in rows if r.get("pilot_error")]
    if failed:
        lines += ["", "## Failures", ""]
        lines += [f"- {r['model']}: {r['pilot_error']['error']}" for r in failed]
    lines += ["", "Raw completions contain harmful text: keep this folder private.", ""]
    return "\n".join(lines)
