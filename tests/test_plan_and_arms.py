import pytest

from safecollapse import arms
from safecollapse.config import load_config
from safecollapse.plan import Node, build_plan, gen1, generators, sources


def test_core_plan_shape():
    cfg = load_config("core")
    nodes = build_plan(cfg)
    # per model: Gen 0, then per seed Gen 1 + R2-R5 + S2-S5
    assert len(nodes) == 1 + 3 * (1 + 4 * 2)  # Gemma only
    labels = {n.label for n in nodes}
    assert labels == {"G0", "G1", "R2", "R3", "R4", "R5", "S2", "S3", "S4", "S5"}


def test_sources_before_dependants():
    nodes = build_plan(load_config("optional"))
    position = {n: i for i, n in enumerate(nodes)}
    for node in nodes:
        for src in sources(node):
            assert position[src] < position[node], (src, node)


def test_sources():
    assert sources(Node("m", 0, "replace", 3)) == [Node("m", 0, "replace", 2)]
    assert sources(Node("m", 0, "replace", 2)) == [gen1("m", 0)]
    assert sources(Node("m", 0, "subsample", 3)) == [gen1("m", 0), Node("m", 0, "subsample", 2)]
    assert sources(Node("m", 0, "original", 5)) == []


def test_generators_exclude_final_generation():
    nodes = build_plan(load_config("core"))
    gens = generators(nodes)
    assert all(n.gen < 5 for n in gens)
    assert gen1("gemma2_2b_it", 0) in gens


def test_original_arm_only_at_listed_generations():
    nodes = build_plan(load_config("optional"))
    assert {n.gen for n in nodes if n.arm == "original"} == {5}


def test_node_key_round_trip():
    for node in build_plan(load_config("core")):
        assert Node.from_key(node.key) == node


# arms ---------------------------------------------------------------------------------------


def _round(n=10):
    return [
        {"id": f"p{i}", "messages": [], "response": f"orig{i}", "source": "original"}
        for i in range(n)
    ]


def _synthetic(original, gens):
    return {
        g: arms.with_answers(original, {e["id"]: f"g{g}-{e['id']}" for e in original}, f"gen{g}")
        for g in gens
    }


def test_gen1_trains_on_original():
    original = _round()
    assert arms.training_set("replace", 1, original, {}, 0) == original


def test_replace_uses_only_previous_generation():
    original = _round()
    data = arms.training_set("replace", 3, original, _synthetic(original, [1, 2]), 0)
    assert {e["source"] for e in data} == {"gen2"} and len(data) == len(original)


def test_subsample_same_size_mixed_and_deterministic():
    original = _round(50)
    syn = _synthetic(original, [1, 2, 3])
    a = arms.training_set("subsample", 4, original, syn, seed=0)
    b = arms.training_set("subsample", 4, original, syn, seed=0)
    c = arms.training_set("subsample", 4, original, syn, seed=1)
    assert len(a) == len(original)
    assert a == b and a != c
    assert len({e["source"] for e in a}) > 1  # draws from the whole pool


def test_accumulate_grows():
    original = _round()
    data = arms.training_set("accumulate", 3, original, _synthetic(original, [1, 2]), 0)
    assert len(data) == 3 * len(original)


def test_original_steps_matched():
    original = _round()
    data = arms.training_set("original", 5, original, {}, 0)
    assert len(data) == 5 * len(original)
    assert {e["source"] for e in data} == {"original"}


def test_missing_synthetic_is_an_error():
    original = _round()
    with pytest.raises(ValueError, match="missing synthetic"):
        arms.training_set("subsample", 3, original, _synthetic(original, [1]), 0)
