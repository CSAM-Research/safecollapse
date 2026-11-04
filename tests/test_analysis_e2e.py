"""End-to-end check of `safecollapse analyze` on a fake study with a known effect.

The analysis code is finished and tested on synthetic data before any real result is looked at,
so no analysis choice depends on the real outcomes.
"""

import json

import numpy as np
import pytest

from safecollapse.cli import main
from safecollapse.config import load_config
from safecollapse.io import write_json, write_jsonl
from safecollapse.judges.rubric import strongreject_score
from safecollapse.plan import Layout, build_plan

pytest.importorskip("statsmodels")
pytest.importorskip("matplotlib")

N_PROMPTS = 40


def _harm_prob(node) -> float:
    """Built-in truth: replace erodes safety with each generation; subsample barely moves."""
    if node.gen <= 1:
        return 0.05 + 0.03 * node.gen
    return 0.08 + (0.08 * (node.gen - 1) if node.arm == "replace" else 0.01 * (node.gen - 1))


def _fake_study(tmp_path, experiment="core"):
    from safecollapse.jailbreaks import template_of
    from safecollapse.pipeline import eval_jobs

    overrides = [
        f"paths.runs_dir={tmp_path / 'runs'}",
        f"paths.data_dir={tmp_path / 'data'}",
        "seeds=[0, 1]",
    ]
    cfg = load_config(experiment, overrides)
    layout = Layout(cfg)
    rng = np.random.default_rng(0)
    for node in build_plan(cfg):
        p = _harm_prob(node)
        for judge in ("gpt4o_mini", "llama33_70b"):
            rows = []
            for i in range(N_PROMPTS):
                harmful = rng.random() < p
                refused = 0 if harmful else 1
                spec, conv = (4, 4) if harmful else (1, 1)
                if judge == "llama33_70b" and harmful and rng.random() < 0.2:
                    rows.append({"id": f"p{i}", "judge_refused": 1})  # judge declines to grade
                    continue
                rows.append(
                    {
                        "id": f"p{i}",
                        "refused": refused,
                        "specific": spec,
                        "convincing": conv,
                        "score": strongreject_score(refused, spec, conv),
                    }
                )
            write_jsonl(layout.judgments(node, "harmbench_test", "strongreject", judge), rows)
        for job in [j for j in eval_jobs(cfg, node) if template_of(j)]:  # jailbreaks: more harm
            write_jsonl(
                layout.judgments(node, job, "strongreject", "gpt4o_mini"),
                [
                    {"id": f"p{i}", "refused": 0, "specific": 4, "convincing": 4, "score": 0.75}
                    if rng.random() < min(1.0, 2 * p)
                    else {"id": f"p{i}", "refused": 1, "specific": 1, "convincing": 1, "score": 0.0}
                    for i in range(N_PROMPTS)
                ],
            )
        write_jsonl(
            layout.judgments(node, "xstest_safe", "xstest", "gpt4o_mini"),
            [
                {"id": f"x{i}", "refused": int(rng.random() < 0.1), "label": "compliance"}
                for i in range(N_PROMPTS)
            ],
        )
        write_jsonl(
            layout.judgments(node, "harmbench_test", "classifier", "harmbench_cls"),
            [{"id": f"p{i}", "harmful": int(rng.random() < p)} for i in range(N_PROMPTS)],
        )
        write_jsonl(
            layout.judgments(node, "sorry_bench", "fulfillment", "sorry_judge"),
            [{"id": f"s{i}", "fulfilled": int(rng.random() < p + 0.1)} for i in range(N_PROMPTS)],
        )
        write_json(
            layout.metrics_file(node, "heldout_nll"),
            {"nll": 1.0 + 0.05 * node.gen, "perplexity": float(np.exp(1 + 0.05 * node.gen))},
        )
        write_json(
            layout.metrics_file(node, "capability"),
            {"ifeval": {"prompt_level_strict_acc,none": 0.5 - 0.02 * node.gen}},
        )
        if node.gen >= 1:
            write_json(
                layout.metrics_file(node, "tail_loss"),
                {"share_above_reference_p95": 0.05 / node.gen, "per_example": [1.0]},
            )
            write_json(layout.metrics_file(node, "synthetic_audit"), {"distinct_2": 0.5})
    return overrides


def test_analyze_end_to_end(tmp_path):
    overrides = _fake_study(tmp_path)
    args = ["analyze", "-c", "core", "--bootstrap", "200"]
    for item in overrides:
        args += ["--set", item]
    assert main(args) == 0

    results = tmp_path / "runs" / "main" / "results"
    for name in (
        "scores_long.csv.gz",
        "node_metrics.csv",
        "summary.csv",
        "confirmatory_gemma2_2b_it.json",
        "table_confirmatory_gemma2_2b_it.tex",
        "table_checkpoints_gemma2_2b_it.tex",
    ):
        assert (results / name).exists(), name
    for fig in ("fig_safety", "fig_secondary", "fig_collapse", "fig_judge_agreement"):
        assert (results / "figures" / f"{fig}.pdf").exists(), fig

    result = json.loads((results / "confirmatory_gemma2_2b_it.json").read_text())
    primary = result["primary"]
    # The built-in effect is recovered: replace erodes, subsample is safer than replace.
    assert primary["H1"]["estimate"] > 0 and primary["H1"]["p"] < 0.05
    assert primary["H2"]["slope"] > 0
    assert primary["H3"]["estimate"] < 0
    assert {"H1", "H2", "H3", "RQ4"} <= set(primary)
    assert all(
        k in result
        for k in (
            "ambiguity_as_harmful",
            "ambiguity_as_safe",
            "secondary_judge",
            "robust",
            "secondary_judge_judge_refusal_as_harmful",
        )
    )
    table = (results / "table_confirmatory_gemma2_2b_it.tex").read_text()
    assert "H1" in table and r"\toprule" in table
    # mean +- sd over seeds, with perplexity and IFEval
    checkpoints = (results / "table_checkpoints_gemma2_2b_it.tex").read_text()
    assert r"$\pm$" in checkpoints and "Held-out PPL" in checkpoints and "IFEval" in checkpoints
    jb = (results / "table_jailbreaks_gemma2_2b_it.tex").read_text()
    assert "AIM" in jb and "Plain" in jb and "R5" in jb and "R3" not in jb
    # safety-collapse tracking: in the fake study both rise with generation in the replace arm
    tracking = json.loads((results / "tracking_gemma2_2b_it.json").read_text())
    assert tracking["levels_replace"] > 0.5 and tracking["n_seeds"] == 2
    assert (results / "figures" / "fig_tracking_gemma2_2b_it.pdf").exists()
    row = (results / "table_tracking_gemma2_2b_it.tex").read_text()
    assert row.startswith("Instruction-tuned & ") and row.count("&") == 3 and "_" not in row


def test_analyze_base_category_uses_accumulate(tmp_path):
    overrides = _fake_study(tmp_path, "base_model")
    args = ["analyze", "-c", "base_model", "--bootstrap", "100", "--no-figures"]
    for item in overrides:
        args += ["--set", item]
    assert main(args) == 0
    results = tmp_path / "runs" / "base_model" / "results"
    result = json.loads((results / "confirmatory_gemma2_2b_kazdan.json").read_text())
    assert result["primary"]["H3"]["control"] == "A5"
    assert "A5" in (results / "table_confirmatory_gemma2_2b_kazdan.tex").read_text()
    assert "A5" in (results / "table_jailbreaks_gemma2_2b_kazdan.tex").read_text()
