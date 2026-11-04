import pytest

from safecollapse.config import ConfigError, load_config
from safecollapse.pipeline import protocol


@pytest.mark.parametrize("name", ["pilot", "smoke", "core", "optional", "extend10", "replication", "beam"])
def test_bundled_configs_load(name):
    cfg = load_config(name)
    assert cfg.training.learning_rate == 8e-6
    assert cfg.training.effective_batch == 128
    assert cfg.judging.primary == "gpt4o_mini"


def test_design_values():
    cfg = load_config("core")
    assert cfg.arms == ("replace", "subsample")
    assert cfg.generations == 5 and cfg.seeds == (0, 1, 2)
    assert cfg.synthetic.temperature == 1.0 and cfg.synthetic.max_new_tokens == 512
    assert cfg.evaluation.decoding.temperature == 0.0
    assert cfg.model("gemma2_2b_it").attn_implementation == "eager"
    assert cfg.secondary_judge_for("gemma2_2b_it") == "llama33_70b"
    assert cfg.secondary_judge_for("llama2_7b_chat") == "qwen25_72b"


def test_inheritance_and_overrides():
    cfg = load_config("optional", ["training.learning_rate=1e-5", "seeds=[4]"])
    assert cfg.arms == ("accumulate", "original")  # from optional.yaml
    assert cfg.name == "main"  # inherited from core -> base
    assert cfg.training.learning_rate == 1e-5 and cfg.seeds == (4,)


def test_unknown_key_is_an_error():
    with pytest.raises(ConfigError, match="unknown keys"):
        load_config("core", ["training.learning_rat=1e-5"])


def test_type_errors_are_caught():
    with pytest.raises(ConfigError):
        load_config("core", ["generations=five"])


def test_family_separation_is_enforced():
    with pytest.raises(ConfigError, match="same family"):
        load_config(
            "core",
            [
                "judging.secondary_by_family.gemma=llama33_70b",
                "registry.judges.llama33_70b.family=gemma",
            ],
        )


def test_unknown_arm_and_model():
    with pytest.raises(ConfigError, match="unknown arm"):
        load_config("core", ["arms=[replace, mixed]"])
    with pytest.raises(ConfigError, match="not in the registry"):
        load_config("core", ["models=[gpt2]"])


def test_protocol_ignores_arms_and_generations():
    """Core, optional arms and the generation-10 extension may share one study folder."""
    assert protocol(load_config("core")) == protocol(load_config("optional"))
    assert protocol(load_config("core")) == protocol(load_config("extend10"))
    assert protocol(load_config("core")) != protocol(load_config("smoke"))


def test_fingerprint_is_stable():
    assert load_config("core").fingerprint() == load_config("core").fingerprint()
