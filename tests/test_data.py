import pytest

from safecollapse.data import benchmarks, helpsteer
from safecollapse.modeling import IGNORE, encode_example, prompt_ids


def test_parse_single_and_multi_turn():
    assert helpsteer.parse_conversation(" hi ") == [{"role": "user", "content": "hi"}]
    text = "c#\n<extra_id_1>Assistant\nWhat?\n<extra_id_1>User\nlist panels"
    roles = [m["role"] for m in helpsteer.parse_conversation(text)]
    assert roles == ["user", "assistant", "user"]


def test_parse_rejects_bad_turns():
    with pytest.raises(ValueError):
        helpsteer.parse_conversation("a\n<extra_id_1>Assistant\nb")  # ends on assistant
    with pytest.raises(ValueError):
        helpsteer.parse_conversation("a\n<extra_id_1>Robot\nb")


def _rows(n_prompts=30):
    return [{"prompt": f"q{i}", "response": f"r{i}-{k}"} for i in range(n_prompts) for k in (0, 1)]


def test_select_round_is_seeded_and_sized():
    rows = _rows()
    a = helpsteer.select_round(rows, 40, seed=0, one_per_prompt=False)
    assert len(a) == 40 and len({e["id"] for e in a}) == 40
    assert a == helpsteer.select_round(rows, 40, seed=0, one_per_prompt=False)
    assert a != helpsteer.select_round(rows, 40, seed=1, one_per_prompt=False)


def test_one_per_prompt_caps_at_unique_prompts():
    rows = _rows()
    data = helpsteer.select_round(rows, 30, seed=0, one_per_prompt=True)
    assert len({e["messages"][0]["content"] for e in data}) == 30
    with pytest.raises(ValueError):
        helpsteer.select_round(rows, 31, seed=0, one_per_prompt=True)


def test_heldout_is_seed_independent():
    rows = _rows()
    assert helpsteer.select_heldout(rows, 10) == helpsteer.select_heldout(rows, 10)
    assert len(helpsteer.select_heldout(rows, None)) == len(rows)


def test_harmbench_loading(tmp_path):
    csv_path = tmp_path / "hb.csv"
    csv_path.write_text(
        "Behavior,FunctionalCategory,SemanticCategory,Tags,ContextString,BehaviorID\n"
        "Do X,standard,cyber,,,x\n"
        "Lyrics,copyright,copyright,,,lyr\n"
        'Do Y,contextual,misinfo,,"Some context",y\n'
    )
    records = benchmarks.load_harmbench(csv_path, "val")
    assert [r["id"] for r in records] == ["x", "y"]  # copyright excluded
    assert records[1]["messages"][0]["content"] == "Some context\n\n---\n\nDo Y"
    assert records[1]["meta"]["context"] == "Some context"


# chat encoding (with a character-level stand-in tokenizer) -----------------------------------


def test_loss_only_on_response(tokenizer):
    messages = [{"role": "user", "content": "hi"}]
    enc = encode_example(tokenizer, messages, "yo", max_len=100)
    prompt = prompt_ids(tokenizer, messages)
    assert enc.input_ids[: len(prompt)] == prompt
    assert all(x == IGNORE for x in enc.labels[: len(prompt)])
    trained = "".join(chr(t) for t in enc.labels if t != IGNORE)
    assert trained == "yo|"  # the answer and the end-of-turn marker
    assert not enc.truncated and not enc.joined


def test_truncation_and_dropping(tokenizer):
    messages = [{"role": "user", "content": "x" * 50}]
    assert encode_example(tokenizer, messages, "answer", max_len=20) is None
    enc = encode_example(tokenizer, [{"role": "user", "content": "q"}], "a" * 50, max_len=20)
    assert enc.truncated and len(enc.input_ids) == 20
