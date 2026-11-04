"""Orchestration logic on fake files: what is pending, training-data assembly, protocol guard."""

import pytest

from safecollapse.config import load_config
from safecollapse.io import read_jsonl, write_jsonl
from safecollapse.pipeline import Runner, build_training_data, judge_tasks, pending, status
from safecollapse.plan import Layout, Node, gen0, gen1


def _write_round(layout, seed, n=8):
    write_jsonl(
        layout.round_file(seed),
        [
            {
                "id": f"p{i}",
                "messages": [{"role": "user", "content": f"q{i}"}],
                "response": f"orig{i}",
                "source": "original",
            }
            for i in range(n)
        ],
    )


def _write_synthetic(layout, node, n=8):
    write_jsonl(
        layout.synthetic_file(node),
        [
            {
                "id": f"p{i}",
                "sample": 0,
                "completion": f"{node.label}-{i}",
                "n_tokens": 3,
                "finish_reason": "stop",
            }
            for i in range(n)
        ],
    )


def test_fresh_node_needs_everything(cfg):
    layout = Layout(cfg)
    todo = pending(cfg, layout, gen1("gemma2_2b_it", 0), is_generator=True)
    assert todo["data"] == ["data"] and todo["train"] == ["train"]
    assert "gen:synthetic" in todo["gpu"] and "tail_loss" in todo["gpu"]
    assert "gen:harmbench_test@robust" in todo["gpu"]  # G1 is in robustness_labels
    assert todo["judge"]


def test_gen0_is_not_trained(cfg):
    todo = pending(cfg, Layout(cfg), gen0("gemma2_2b_it"), is_generator=False)
    assert todo["data"] == [] and todo["train"] == []
    assert "tail_loss" not in todo["gpu"] and "gen:synthetic" not in todo["gpu"]


def test_judge_tasks_follow_family_separation(cfg):
    tasks = judge_tasks(cfg, gen0("gemma2_2b_it"))
    judges = {(job, judge) for job, _, judge in tasks}
    assert ("harmbench_test", "gpt4o_mini") in judges
    assert ("harmbench_test", "llama33_70b") in judges
    assert ("xstest_safe", "gpt4o_mini") in judges
    assert all(task == "xstest" for job, task, _ in tasks if job == "xstest_safe")
    assert not any(job == "sorry_bench" for job, _, _ in tasks)  # its own local judge


def test_build_training_data_replace_and_subsample(cfg):
    layout = Layout(cfg)
    _write_round(layout, 0)
    _write_synthetic(layout, gen1("gemma2_2b_it", 0))
    r2 = Node("gemma2_2b_it", 0, "replace", 2)
    build_training_data(cfg, r2)
    data = read_jsonl(layout.train_file(r2))
    assert [e["response"] for e in data] == [f"G1-{i}" for i in range(8)]
    assert layout.done(r2, "data")

    s2 = Node("gemma2_2b_it", 0, "subsample", 2)
    build_training_data(cfg, s2)
    data = read_jsonl(layout.train_file(s2))
    assert len(data) == 8 and {e["source"] for e in data} <= {"original", "G1"}


def test_status_lists_every_node(cfg):
    rows = status(cfg)
    assert len(rows) == 28 and all(r["pending"] for r in rows)


def test_protocol_guard(tmp_path):
    paths = [f"paths.runs_dir={tmp_path}", f"paths.data_dir={tmp_path / 'd'}"]
    Runner(load_config("core", paths))
    Runner(load_config("optional", paths))  # same protocol: fine
    with pytest.raises(RuntimeError, match="different protocol"):
        Runner(load_config("core", paths + ["training.learning_rate=1e-5"]))


# scheduler: several workers share one folder -------------------------------------------------


def _fake_run_node(runner, order):
    """Stand-in for the GPU work: record the node and mark all its stages done."""
    from safecollapse.pipeline import gpu_stages

    def run_node(node, is_generator):
        order.append(node)
        for stage in ["data", "train", *gpu_stages(runner.cfg, node, is_generator)]:
            runner.layout.mark_done(node, stage)

    return run_node


def test_worker_runs_everything_in_dependency_order(cfg):
    from safecollapse.plan import sources

    runner = Runner(cfg)
    order = []
    runner.run_node = _fake_run_node(runner, order)
    runner.work(judge=False, poll_s=0)
    assert len(order) == 28 and len(set(order)) == 28
    position = {n: i for i, n in enumerate(order)}
    assert all(position[s] < position[n] for n in order for s in sources(n))
    assert not any(runner.layout.locked(n) for n in order)  # locks released


def test_two_workers_never_take_the_same_node(cfg):
    from safecollapse.plan import build_plan, generators

    a, b = Runner(cfg), Runner(cfg)
    nodes = build_plan(cfg)
    gens = generators(nodes)
    first, _ = a.next_node(nodes, gens)
    second, _ = b.next_node(nodes, gens)
    assert first == gen0("gemma2_2b_it")
    assert second == gen1("gemma2_2b_it", 0)  # Gen 0 is locked by a; Gen 1 needs nothing
    # R2 must wait for Gen 1's synthetic data, so a third worker skips to another seed's Gen 1
    third, _ = Runner(cfg).next_node(nodes, gens)
    assert third == gen1("gemma2_2b_it", 1)


def test_stale_lock_is_reclaimed(cfg, monkeypatch):
    import socket

    layout = Layout(cfg)
    node = gen0("gemma2_2b_it")
    layout.lock_path(node).parent.mkdir(parents=True, exist_ok=True)
    layout.lock_path(node).write_text(f"{socket.gethostname()} 999999999\n")  # dead pid
    assert layout.try_lock(node)
    assert not layout.try_lock(node)  # now held by this (live) process


def test_throughput_settings_are_not_protocol(tmp_path):
    paths = [f"paths.runs_dir={tmp_path}", f"paths.data_dir={tmp_path / 'd'}"]
    Runner(load_config("core", paths))
    Runner(load_config("core", paths + ["training.per_device_batch_size=16"]))


def test_marker_files_use_portable_names(cfg):
    layout = Layout(cfg)
    node = gen0("gemma2_2b_it")
    layout.mark_done(node, "judge:harmbench_test@robust:strongreject:gpt4o_mini")
    assert layout.done(node, "judge:harmbench_test@robust:strongreject:gpt4o_mini")
    names = [p.name for p in (layout.node_dir(node) / ".done").iterdir()]
    assert names and not any(c in n for n in names for c in ':@<>"|?*')


def test_smoke_and_main_never_share_data_files(tmp_path):
    paths = [f"paths.runs_dir={tmp_path}", f"paths.data_dir={tmp_path / 'd'}"]
    core, smoke = Layout(load_config("core", paths)), Layout(load_config("smoke", paths))
    assert core.round_file(0) != smoke.round_file(0)
    assert core.heldout_file() != smoke.heldout_file()


def test_checkpoints_are_archived_and_restored(tmp_path):
    from safecollapse.pipeline import archive_checkpoint, restore_checkpoint

    cfg = load_config(
        "core",
        [
            f"paths.runs_dir={tmp_path / 'runs'}",
            f"paths.checkpoints_dir={tmp_path / 'local'}",
            f"paths.archive_dir={tmp_path / 'archive'}",
        ],
    )
    layout = Layout(cfg)
    node = gen1("gemma2_2b_it", 0)
    layout.checkpoint(node).mkdir(parents=True)
    (layout.checkpoint(node) / "model.safetensors").write_text("weights")
    archive_checkpoint(layout, node)
    assert layout.done(node, "archived")
    assert (layout.archived_checkpoint(node) / "model.safetensors").read_text() == "weights"
    import shutil

    shutil.rmtree(layout.checkpoint(node))  # pod restart wipes the local disk
    restore_checkpoint(layout, node)
    assert (layout.checkpoint(node) / "model.safetensors").read_text() == "weights"


def test_local_copy_freed_only_after_archive(tmp_path):
    from safecollapse.pipeline import gpu_stages

    cfg = load_config(
        "core",
        [
            f"paths.runs_dir={tmp_path / 'runs'}",
            f"paths.checkpoints_dir={tmp_path / 'local'}",
            f"paths.archive_dir={tmp_path / 'archive'}",
        ],
    )
    runner = Runner(cfg)
    node = Node("gemma2_2b_it", 0, "replace", 5)  # final generation: not a generator
    for stage in ["data", "train", *gpu_stages(cfg, node, False)]:
        runner.layout.mark_done(node, stage)
    runner.layout.checkpoint(node).mkdir(parents=True)
    (runner.layout.checkpoint(node) / "w").write_text("x")
    runner._archive_and_free(node, False)
    assert not runner.layout.checkpoint(node).exists()  # local cache freed
    assert (runner.layout.archived_checkpoint(node) / "w").read_text() == "x"  # kept forever


def test_gpu_work_split_into_separate_processes(cfg):
    from safecollapse.pipeline import gpu_parts_pending

    layout = Layout(cfg)
    node = gen1("gemma2_2b_it", 0)
    assert gpu_parts_pending(cfg, layout, node, True) == ["generate", "metrics", "post"]
    for stage in ("heldout_nll", "tail_loss"):
        layout.mark_done(node, stage)
    assert gpu_parts_pending(cfg, layout, node, True) == ["generate", "post"]


# published checkpoints (base-model category): fetched and evaluated, never trained ------------


def test_published_checkpoints_are_fetched_not_trained(tmp_path):
    from safecollapse.pipeline import generator_nodes
    from safecollapse.plan import build_plan, published_ref, weights_stage

    paths = [f"paths.runs_dir={tmp_path}", f"paths.data_dir={tmp_path / 'd'}"]
    cfg = load_config("base_model", paths)
    layout = Layout(cfg)
    nodes = build_plan(cfg)
    assert len(nodes) == 28 and {n.arm for n in nodes if n.gen > 1} == {"replace", "accumulate"}
    assert generator_nodes(cfg, nodes) == set()  # no synthetic data, so no tail loss either
    g1, a3 = gen1("gemma2_2b_kazdan", 2), Node("gemma2_2b_kazdan", 1, "accumulate", 3)
    assert published_ref(cfg, g1)[0] == "jkazdan/collapse_gemma-2-2b_hs2_iter1_sftsd2"
    assert published_ref(cfg, a3)[0] == "jkazdan/collapse_gemma-2-2b_hs2_accumulate_iter3_sftsd1"
    # Gen 0 is fetched too: the base model needs Gemma-2-2B-IT's chat template
    assert weights_stage(cfg, gen0("gemma2_2b_kazdan")) == "fetch"
    todo = pending(cfg, layout, a3, is_generator=False)
    assert todo["data"] == [] and todo["train"] == ["fetch"]
    assert "gen:synthetic" not in todo["gpu"] and "tail_loss" not in todo["gpu"]
    assert layout.model_ref(a3) == (str(layout.checkpoint(a3)), None)
    labels = {n.label for n in nodes if "gen:harmbench_test@robust" in pending(cfg, layout, n, False)["gpu"]}
    assert labels == {"G0", "G1", "R5", "A5"}


def test_published_run_fetches_then_evaluates_and_frees_disk(tmp_path):
    from safecollapse.pipeline import gpu_stages

    paths = [f"paths.runs_dir={tmp_path}", f"paths.data_dir={tmp_path / 'd'}"]
    runner = Runner(load_config("base_model", paths))
    calls = []

    def fake_subprocess(stage, node, *extra):
        calls.append((stage, node.label))
        if stage == "fetch":
            runner.layout.checkpoint(node).mkdir(parents=True, exist_ok=True)
            runner.layout.mark_done(node, "fetch")
        elif stage == "gpu":
            for s in gpu_stages(runner.cfg, node, False):
                runner.layout.mark_done(node, s)

    runner.subprocess = fake_subprocess
    node = Node("gemma2_2b_kazdan", 0, "replace", 4)
    runner.run_node(node, False)
    assert calls[0] == ("fetch", "R4") and not any(c[0] == "train" for c in calls)
    assert not runner.layout.checkpoint(node).exists()  # public weights: local copy freed


def test_main_protocol_unchanged_by_optional_model_fields(cfg):
    from safecollapse.pipeline import protocol

    spec = protocol(cfg)["models"]["gemma2_2b_it"]
    assert set(spec) == {"hf_id", "revision", "family", "attn_implementation", "max_model_len"}


def test_main_protocol_unchanged_by_unused_judges(cfg):
    from safecollapse.pipeline import protocol

    judges = protocol(cfg)["judges"]
    assert set(judges) == {"gpt4o_mini", "llama33_70b", "qwen25_72b"}  # not deepseek_flash
    assert all("reasoning_effort" not in spec for spec in judges.values())


def test_jailbreak_jobs_wrap_the_prompt_and_judge_the_original(cfg, monkeypatch):
    from safecollapse import jailbreaks

    monkeypatch.setattr(
        jailbreaks, "templates", lambda: {"refusal_suppression": "Rules: {forbidden_prompt}"}
    )
    from safecollapse.pipeline import eval_jobs

    r5 = Node("gemma2_2b_it", 0, "replace", 5)
    jobs = eval_jobs(cfg, r5)
    jb = [j for j in jobs if jailbreaks.template_of(j)]
    assert sorted(jailbreaks.template_of(j) for j in jb) == [
        "aim",
        "prefix_injection",
        "refusal_suppression",
    ]
    assert not any(jailbreaks.template_of(j) for j in eval_jobs(cfg, Node("gemma2_2b_it", 0, "replace", 3)))
    tasks = [t for t in judge_tasks(cfg, r5) if jailbreaks.template_of(t[0])]
    assert {(task, judge) for _, task, judge in tasks} == {("strongreject", "gpt4o_mini")}
    rec = {"id": "p1", "messages": [{"role": "user", "content": "context"}, {"role": "user", "content": "do X"}]}
    wrapped = jailbreaks.wrap_record(rec, "refusal_suppression")
    assert "do X" in wrapped["messages"][-1]["content"] and wrapped["messages"][0]["content"] == "context"
    assert rec["messages"][-1]["content"] == "do X"  # original untouched (the judge uses it)
    assert "{forbidden_prompt}" not in wrapped["messages"][-1]["content"]


def test_jailbreak_templates_are_pinned_not_redistributed():
    from importlib import resources

    from safecollapse import jailbreaks

    assert jailbreaks._SPEC["commit"] == "7a551d5b440ec7b75d4f6f5bb7c1719965b76b47"
    assert jailbreaks._SPEC["commit"] in jailbreaks._SPEC["url"]
    assert len(jailbreaks._SPEC["file_sha256"]) == 64
    assert set(jailbreaks.TEMPLATES) == {"refusal_suppression", "prefix_injection", "aim"}
    text = (resources.files("safecollapse") / "configs" / "jailbreaks.yaml").read_text()
    assert "forbidden_prompt}" not in text.split("source:")[1]


def test_jailbreak_template_fetch_checks_the_pinned_hash(tmp_path, monkeypatch):
    from safecollapse import jailbreaks

    jailbreaks.templates.cache_clear()
    monkeypatch.setenv("SAFECOLLAPSE_CACHE", str(tmp_path))
    jailbreaks._cache_file().parent.mkdir(parents=True, exist_ok=True)
    jailbreaks._cache_file().write_text("{}")  # not the pinned file
    with pytest.raises(ValueError, match="pinned hash"):
        jailbreaks.templates()
    assert not jailbreaks._cache_file().exists()
    jailbreaks.templates.cache_clear()


def test_diverged_training_is_refused():
    from safecollapse.training import check_finite

    check_finite([{"loss": 0.8, "grad_norm": 2.1, "step": 5}, {"train_loss": 0.7, "step": 98}])
    for bad in (
        [{"loss": 0.8, "grad_norm": float("nan"), "step": 5}],
        [{"loss": float("inf"), "grad_norm": 1.0, "step": 5}],
        [{"loss": 0.0, "grad_norm": 1.0, "step": 10}],
    ):
        with pytest.raises(RuntimeError, match="diverged"):
            check_finite(bad)


def test_beam_setting_leaves_existing_protocols_unchanged():
    from safecollapse.config import load_config
    from safecollapse.pipeline import protocol

    assert "num_beams" not in protocol(load_config("core"))["synthetic"]
    assert "num_beams" not in protocol(load_config("core"))["decoding"]
    beam = load_config("beam")
    assert protocol(beam)["synthetic"]["num_beams"] == 5
    assert beam.generation.backend != "vllm" and beam.arms == ("replace",)
