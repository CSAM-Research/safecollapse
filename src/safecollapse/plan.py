"""The study as a list of checkpoints (nodes) and where each one's files live.

Pure logic, no GPU: `safecollapse status` uses it to show what is done and what is pending.

Every trained checkpoint is a fresh fine-tune of Gen 0 (DESIGN.md Section 3.3); the recursion
lives in the training data. A node's *sources* are the earlier nodes whose synthetic answers it
trains on, so the plan is a small DAG over generations.
"""

from __future__ import annotations

import os
import shutil
import socket
from dataclasses import dataclass
from pathlib import Path

from .config import Config

ARM_LETTER = {"replace": "R", "subsample": "S", "accumulate": "A", "original": "O"}
SYNTHETIC_ARMS = ("replace", "subsample", "accumulate")


@dataclass(frozen=True, order=True)
class Node:
    model: str
    seed: int | None  # None only for Gen 0 (untouched, so seed-independent)
    arm: str  # "base" (Gen 0), "shared" (Gen 1) or one of the arms
    gen: int

    @property
    def label(self) -> str:
        if self.gen <= 1:
            return f"G{self.gen}"
        return f"{ARM_LETTER[self.arm]}{self.gen}"

    @property
    def trained(self) -> bool:
        return self.gen >= 1

    @property
    def key(self) -> str:
        if self.gen == 0:
            return f"{self.model}/gen0"
        return f"{self.model}/seed{self.seed}/{self.arm}/gen{self.gen}"

    @classmethod
    def from_key(cls, key: str) -> Node:
        parts = key.strip("/").split("/")
        if len(parts) == 2 and parts[1] == "gen0":
            return gen0(parts[0])
        if len(parts) == 4 and parts[1].startswith("seed") and parts[3].startswith("gen"):
            return cls(parts[0], int(parts[1][4:]), parts[2], int(parts[3][3:]))
        raise ValueError(f"not a node key: {key!r}")


def gen0(model: str) -> Node:
    return Node(model, None, "base", 0)


def gen1(model: str, seed: int) -> Node:
    return Node(model, seed, "shared", 1)


def arm_node(model: str, seed: int, arm: str, gen: int) -> Node:
    """Generation 1 is shared by every arm."""
    return gen1(model, seed) if gen == 1 else Node(model, seed, arm, gen)


def sources(node: Node) -> list[Node]:
    """Generator nodes whose synthetic answers this node trains on (besides the original data)."""
    if node.gen <= 1 or node.arm == "original":
        return []
    if node.arm == "replace":
        return [arm_node(node.model, node.seed, "replace", node.gen - 1)]
    # subsample and accumulate: every earlier generation of the same arm
    return [arm_node(node.model, node.seed, node.arm, g) for g in range(1, node.gen)]


def build_plan(cfg: Config, models: list[str] | None = None, seeds: list[int] | None = None):
    """All nodes in a valid execution order (sources before the nodes that need them)."""
    nodes: list[Node] = []
    for model in models or list(cfg.models):
        nodes.append(gen0(model))
        for seed in seeds if seeds is not None else list(cfg.seeds):
            nodes.append(gen1(model, seed))
            for gen in range(2, cfg.generations + 1):
                for arm in cfg.arms:
                    if arm == "original" and gen not in cfg.original_generations:
                        continue
                    nodes.append(Node(model, seed, arm, gen))
    return nodes


def weights_stage(cfg: Config, node: Node) -> str | None:
    """How a node's weights are obtained: "train" (our fine-tune), "fetch" (a published checkpoint,
    or Gen 0 needing a chat template, copied to the local checkpoint folder) or None (Gen 0,
    loaded straight from the hub)."""
    spec = cfg.model(node.model)
    if node.gen == 0:
        return "fetch" if spec.chat_template_from else None
    return "fetch" if spec.published else "train"


def published_ref(cfg: Config, node: Node) -> tuple[str, str | None]:
    """(hub id, revision or None for the latest) of a fetched node's weights."""
    spec = cfg.model(node.model)
    if node.gen == 0:
        return spec.hf_id, spec.revision
    pattern = spec.published_gen1 if node.gen == 1 else spec.published
    return pattern.format(arm=node.arm, gen=node.gen, seed=node.seed), None


def generators(nodes: list[Node]) -> set[Node]:
    """Nodes whose synthetic data some planned node needs."""
    return {src for node in nodes for src in sources(node)}


# --------------------------------------------------------------------------------------------


class Layout:
    """Where everything lives. Run folders: <runs_dir>/<name>/<model>/seed<k>/<arm>/gen<t>/."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.root = Path(cfg.paths.runs_dir) / cfg.name
        ckpt_root = cfg.paths.checkpoints_dir
        self.checkpoint_root = Path(ckpt_root) / cfg.name if ckpt_root else self.root
        self.data_dir = Path(cfg.paths.data_dir)
        self.cache_dir = Path(cfg.paths.cache_dir or Path(cfg.paths.runs_dir) / "_judge_cache")

    # data
    def data_file(self, name: str) -> Path:
        return self.data_dir / f"{name}.jsonl"

    def round_file(self, seed: int) -> Path:
        """The original data (Gen 1's training set) for one seed. The size is in the name, so a
        smoke test's small rounds can never be mistaken for the main study's."""
        d = self.cfg.data
        tag = f"_n{d.round_size}" + ("_onepp" if d.one_response_per_prompt else "")
        return self.data_dir / f"helpsteer_round_seed{seed}{tag}.jsonl"

    def heldout_file(self) -> Path:
        size = self.cfg.data.heldout_size
        return self.data_dir / f"helpsteer_heldout_n{size if size is not None else 'all'}.jsonl"

    # nodes
    def node_dir(self, node: Node) -> Path:
        return self.root / node.key

    def checkpoint(self, node: Node) -> Path:
        return self.checkpoint_root / node.key / "checkpoint"

    def train_file(self, node: Node) -> Path:
        return self.node_dir(node) / "train.jsonl"

    def synthetic_file(self, node: Node) -> Path:
        return self.node_dir(node) / "synthetic.jsonl"

    def completions(self, node: Node, bench: str) -> Path:
        return self.node_dir(node) / "eval" / _safe(bench) / "completions.jsonl"

    def judgments(self, node: Node, bench: str, task: str, judge: str) -> Path:
        return self.node_dir(node) / "eval" / _safe(bench) / f"{task}__{_safe(judge)}.jsonl"

    def metrics_file(self, node: Node, name: str) -> Path:
        return self.node_dir(node) / "metrics" / f"{_safe(name)}.json"

    def manifest(self, node: Node) -> Path:
        return self.node_dir(node) / "manifest.json"

    def tail_reference(self, model: str, seed: int) -> Path:
        """Per-example perplexity of the seed's original data under Gen 0 (tail threshold)."""
        return self.root / model / f"seed{seed}" / "tail_reference.json"

    # stage markers: a stage counts as done only once its marker exists
    def done(self, node: Node, stage: str) -> bool:
        return self._marker(node, stage).exists()

    def mark_done(self, node: Node, stage: str) -> None:
        marker = self._marker(node, stage)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()

    def _marker(self, node: Node, stage: str) -> Path:
        return self.node_dir(node) / ".done" / _safe(stage)

    # node locks: several workers (one per GPU) share one study folder; a worker owns a node
    # while it runs it. A lock left by a killed process on this machine is reclaimed.
    def lock_path(self, node: Node) -> Path:
        return self.node_dir(node) / ".lock"

    def try_lock(self, node: Node) -> bool:
        path = self.lock_path(node)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and _stale(path):
            path.unlink(missing_ok=True)
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        with os.fdopen(fd, "w") as fh:
            fh.write(f"{socket.gethostname()} {os.getpid()}\n")
        return True

    def unlock(self, node: Node) -> None:
        self.lock_path(node).unlink(missing_ok=True)

    def locked(self, node: Node) -> bool:
        return self.lock_path(node).exists()

    def model_ref(self, node: Node) -> tuple[str, str | None]:
        """(path or hub id, revision) to load this node's weights from."""
        if weights_stage(self.cfg, node) is None:
            spec = self.cfg.model(node.model)
            return spec.hf_id, spec.revision
        return str(self.checkpoint(node)), None

    def archived_checkpoint(self, node: Node) -> Path | None:
        archive = self.cfg.paths.archive_dir
        return Path(archive) / self.cfg.name / node.key / "checkpoint" if archive else None

    def delete_checkpoint(self, node: Node) -> None:
        shutil.rmtree(self.checkpoint(node), ignore_errors=True)


def _stale(lock: Path) -> bool:
    """True if the lock was taken on this machine by a process that no longer exists."""
    try:
        host, pid = lock.read_text().split()
    except (OSError, ValueError):
        return False
    if host != socket.gethostname():
        return False  # cannot check another machine; use `run --break-locks` if it died
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def _safe(name: str) -> str:
    """File-name-safe form of a stage or job name. Stage names use ':' and '@' (e.g.
    "gen:harmbench_test@robust"); some network filesystems (RunPod volumes) reject ':'."""
    return name.replace(":", "__").replace("@", "_at_")
