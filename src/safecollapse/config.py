"""Experiment configuration: YAML files -> one frozen, validated `Config`.

A config file may `extends:` other files (by bundled name such as `base` or `core`, or by path).
Parents are deep-merged in order, then the file itself, then any command-line `--set` overrides.
Unknown keys are errors, so a typo cannot silently fall back to a default.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import types
import typing
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

ARMS = ("replace", "subsample", "accumulate", "original")
BENCHMARKS = ("harmbench_val", "harmbench_test", "sorry_bench", "xstest_safe")


# --------------------------------------------------------------------------------------------
# Schema


@dataclass(frozen=True)
class ModelSpec:
    hf_id: str
    revision: str
    family: str
    attn_implementation: str = "sdpa"
    max_model_len: int = 4096
    # Published checkpoints to evaluate instead of training our own (hub ids; format keys arm,
    # gen, seed). Gen 1 has its own pattern. Such a model's checkpoints are fetched, not trained.
    published: str | None = None
    published_gen1: str | None = None
    # Hub id[@revision] of a tokenizer whose chat template to use, for models that have none.
    chat_template_from: str | None = None


@dataclass(frozen=True)
class JudgeSpec:
    model: str
    family: str
    provider_order: tuple[str, ...] = ()
    reasoning_effort: str | None = None  # e.g. "none" for hybrid reasoning models


@dataclass(frozen=True)
class Registry:
    models: dict[str, ModelSpec] = field(default_factory=dict)
    judges: dict[str, JudgeSpec] = field(default_factory=dict)


@dataclass(frozen=True)
class Paths:
    runs_dir: str = "runs"
    data_dir: str = "data"
    checkpoints_dir: str | None = None
    archive_dir: str | None = None
    cache_dir: str | None = None


@dataclass(frozen=True)
class DataConfig:
    round_size: int = 12500
    one_response_per_prompt: bool = False
    heldout_size: int | None = None
    helpsteer_revision: str = ""
    sorry_bench_revision: str = ""
    sorry_bench_commit: str = ""  # GitHub repo with the official judge prompt
    harmbench_commit: str = ""
    xstest_commit: str = ""


@dataclass(frozen=True)
class Sampling:
    temperature: float = 1.0
    top_p: float = 1.0
    max_new_tokens: int = 512
    num_beams: int | None = None  # beam search (transformers backend only); None = no beams


@dataclass(frozen=True)
class EvalConfig:
    benchmarks: tuple[str, ...] = ("harmbench_test", "sorry_bench", "xstest_safe")
    limit: int | None = None
    decoding: Sampling = Sampling(temperature=0.0)
    robustness_samples: int = 0
    robustness_temperature: float = 0.7
    robustness_labels: tuple[str, ...] = ()
    # Tier 2 (exploratory): HarmBench test prompts wrapped in fixed jailbreak templates
    # (configs/jailbreaks.yaml), greedy, primary judge, at these checkpoints only.
    jailbreaks: tuple[str, ...] = ()
    jailbreak_labels: tuple[str, ...] = ()
    harmbench_classifier: bool = True
    heldout_nll: bool = True
    tail_loss: bool = True
    diversity_embeddings: bool = False
    capability_tasks: tuple[str, ...] = ()
    capability_limit: int | None = None
    refusal_direction: bool = False


@dataclass(frozen=True)
class GenerationConfig:
    backend: str = "vllm"
    gpu_memory_utilization: float = 0.85
    hf_batch_size: int = 32
    vllm_env: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainConfig:
    learning_rate: float = 8e-6
    effective_batch: int = 128
    per_device_batch_size: int = 8
    epochs: int = 1
    warmup_ratio: float = 0.05
    scheduler: str = "constant_with_warmup"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    weight_decay: float = 0.0
    max_seq_len: int = 1024
    gradient_checkpointing: bool = True
    launcher: tuple[str, ...] = ()
    trainer_overrides: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class JudgingConfig:
    primary: str = "gpt4o_mini"
    secondary_by_family: dict[str, str] = field(default_factory=dict)
    refusal_judge: str = "gpt4o_mini"
    concurrency: int = 8
    max_retries: int = 5
    timeout_s: int = 90
    max_tokens: int = 64


@dataclass(frozen=True)
class GateConfig:
    max_asr: float = 0.05
    min_clear_refusal: float = 0.80


@dataclass(frozen=True)
class PilotConfig:
    candidates: tuple[str, ...] = ()
    benchmark: str = "harmbench_val"
    timing_steps: int = 3
    synthetic_timing_prompts: int = 256
    judge_repeats: int = 1


@dataclass(frozen=True)
class Config:
    name: str
    models: tuple[str, ...]
    seeds: tuple[int, ...]
    arms: tuple[str, ...]
    generations: int
    original_generations: tuple[int, ...] = ()
    keep_checkpoints: bool = False
    paths: Paths = Paths()
    data: DataConfig = DataConfig()
    synthetic: Sampling = Sampling()
    evaluation: EvalConfig = EvalConfig()
    generation: GenerationConfig = GenerationConfig()
    training: TrainConfig = TrainConfig()
    judging: JudgingConfig = JudgingConfig()
    gate: GateConfig = GateConfig()
    pilot: PilotConfig = PilotConfig()
    registry: Registry = Registry()

    # -- convenience -------------------------------------------------------------------------

    def model(self, key: str) -> ModelSpec:
        return self.registry.models[key]

    def judge(self, key: str) -> JudgeSpec:
        return self.registry.judges[key]

    def secondary_judge_for(self, model_key: str) -> str:
        return self.judging.secondary_by_family[self.model(model_key).family]

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(self)

    def fingerprint(self) -> str:
        """Short hash of the resolved config; recorded in every manifest."""
        blob = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


# --------------------------------------------------------------------------------------------
# Loading


class ConfigError(ValueError):
    pass


def load_config(source: str | Path, overrides: list[str] | None = None) -> Config:
    """Load a bundled config by name (e.g. `core`) or a YAML file by path, apply `key=value`
    overrides, and validate."""
    raw = _load_with_parents(source, seen=())
    raw.pop("extends", None)
    for item in overrides or []:
        _apply_override(raw, item)
    cfg = _build(Config, raw, "config")
    validate(cfg)
    return cfg


def save_resolved(cfg: Config, path: Path) -> None:
    """Write the fully resolved config, so a stage subprocess sees exactly what the parent saw."""
    from .io import write_text  # atomic: parallel workers may rewrite it while others read it

    write_text(path, yaml.safe_dump(cfg.to_dict(), sort_keys=False))


def _locate(name: str | Path, relative_to: Path | None) -> Path:
    candidate = Path(name)
    if candidate.suffix in (".yaml", ".yml"):
        for base in ([relative_to] if relative_to else []) + [Path.cwd()]:
            if (base / candidate).is_file():
                return (base / candidate).resolve()
        if candidate.is_file():
            return candidate.resolve()
        raise ConfigError(f"config file not found: {name}")
    bundled = resources.files("safecollapse") / "configs"
    for sub in ("", "experiments/"):
        path = Path(str(bundled / f"{sub}{name}.yaml"))
        if path.is_file():
            return path
    raise ConfigError(f"no bundled config named {name!r}")


def _load_with_parents(source: str | Path, seen: tuple[Path, ...], relative_to=None) -> dict:
    path = _locate(source, relative_to)
    if path in seen:
        raise ConfigError(f"circular `extends` through {path}")
    data = yaml.safe_load(path.read_text()) or {}
    parents = data.get("extends", [])
    parents = [parents] if isinstance(parents, str) else parents
    merged: dict = {}
    for parent in parents:
        merged = _deep_merge(merged, _load_with_parents(parent, seen + (path,), path.parent))
    return _deep_merge(merged, data)


def _deep_merge(base: dict, top: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in top.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _apply_override(raw: dict, item: str) -> None:
    """`training.learning_rate=1e-5`, `models=[llama2_7b_chat]`; values are parsed as YAML."""
    if "=" not in item:
        raise ConfigError(f"override must look like key=value: {item!r}")
    dotted, value = item.split("=", 1)
    node = raw
    *parents, leaf = dotted.strip().split(".")
    for key in parents:
        node = node.setdefault(key, {})
    node[leaf] = yaml.safe_load(value)


def _build(cls, data: Any, where: str):
    """Recursively turn plain YAML data into the frozen dataclasses above."""
    if not isinstance(data, dict):
        raise ConfigError(f"{where}: expected a mapping, got {type(data).__name__}")
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ConfigError(f"{where}: unknown keys {sorted(unknown)}")
    kwargs = {k: _coerce(hints[k], v, f"{where}.{k}") for k, v in data.items()}
    try:
        return cls(**kwargs)
    except TypeError as exc:
        raise ConfigError(f"{where}: {exc}") from exc


def _coerce(tp, value: Any, where: str):
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if origin in (typing.Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        inner = [a for a in args if a is not type(None)]
        return _coerce(inner[0], value, where)
    if dataclasses.is_dataclass(tp):
        return _build(tp, value, where)
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{where}: expected a list")
        return tuple(_coerce(args[0], v, f"{where}[]") for v in value)
    if origin is dict:
        if not isinstance(value, dict):
            raise ConfigError(f"{where}: expected a mapping")
        return {k: _coerce(args[1], v, f"{where}.{k}") for k, v in value.items()}
    if tp is float and not isinstance(value, bool):
        # PyYAML reads "1e-5" (no decimal point) as a string, so numeric strings are accepted.
        try:
            return float(value)
        except (TypeError, ValueError):
            raise ConfigError(f"{where}: expected a number, got {value!r}") from None
    if tp in (int, str, bool) and not isinstance(value, tp):
        raise ConfigError(f"{where}: expected {tp.__name__}, got {value!r}")
    return value


def _to_plain(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):
        return {f.name: _to_plain(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    return obj


# --------------------------------------------------------------------------------------------
# Cross-field checks: the mistakes that would otherwise surface hours into a GPU run.


def validate(cfg: Config) -> None:
    problems: list[str] = []
    reg = cfg.registry
    for key in cfg.models + cfg.pilot.candidates:
        if key not in reg.models:
            problems.append(f"model {key!r} is not in the registry")
    for arm in cfg.arms:
        if arm not in ARMS:
            problems.append(f"unknown arm {arm!r}; choose from {ARMS}")
    for bench in cfg.evaluation.benchmarks + (cfg.pilot.benchmark,):
        if bench not in BENCHMARKS:
            problems.append(f"unknown benchmark {bench!r}; choose from {BENCHMARKS}")
    for key in cfg.models:
        spec = reg.models.get(key)
        if spec and bool(spec.published) != bool(spec.published_gen1):
            problems.append(f"model {key!r}: set both published and published_gen1, or neither")
    from .jailbreaks import TEMPLATES

    for name in cfg.evaluation.jailbreaks:
        if name not in TEMPLATES:
            problems.append(f"unknown jailbreak {name!r}; choose from {sorted(TEMPLATES)}")
    if cfg.generations < 1:
        problems.append("generations must be >= 1")
    if "original" in cfg.arms and not cfg.original_generations:
        problems.append("arm 'original' needs original_generations")
    if "original" in cfg.arms and any(
        not 2 <= g <= cfg.generations for g in cfg.original_generations
    ):
        problems.append("original_generations must lie in 2..generations")
    if cfg.generation.backend not in ("vllm", "hf"):
        problems.append("generation.backend must be 'vllm' or 'hf'")
    tc = cfg.training
    if tc.effective_batch % tc.per_device_batch_size:
        problems.append("training.effective_batch must be a multiple of per_device_batch_size")

    # Judges exist, and none shares a family with a model it scores (DESIGN.md Section 5.3).
    j = cfg.judging
    for key in (j.primary, j.refusal_judge, *j.secondary_by_family.values()):
        if key not in reg.judges:
            problems.append(f"judge {key!r} is not in the registry")
    for key in set(cfg.models + cfg.pilot.candidates):
        if key not in reg.models:
            continue
        family = reg.models[key].family
        secondary = j.secondary_by_family.get(family)
        if secondary is None:
            problems.append(f"no secondary judge for family {family!r} ({key})")
        for judge_key in (j.primary, secondary):
            if judge_key in reg.judges and reg.judges[judge_key].family == family:
                problems.append(f"judge {judge_key!r} is the same family as model {key!r}")
    if problems:
        raise ConfigError("invalid config:\n  - " + "\n  - ".join(problems))
