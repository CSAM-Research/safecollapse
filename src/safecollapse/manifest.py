"""Run manifests: everything needed to say exactly how a checkpoint and its scores were made
(DESIGN.md Section 11). One `manifest.json` per node, updated by each stage."""

from __future__ import annotations

import importlib.metadata
import platform
import time
from pathlib import Path

from .io import read_json, write_json

PACKAGES = ("safecollapse", "torch", "transformers", "accelerate", "vllm", "lm_eval")


def environment() -> dict:
    env = {"python": platform.python_version(), "platform": platform.platform(), "packages": {}}
    for name in PACKAGES:
        try:
            env["packages"][name] = importlib.metadata.version(name.replace("_", "-"))
        except importlib.metadata.PackageNotFoundError:
            pass
    try:
        import torch

        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name()
            env["gpu_count"] = torch.cuda.device_count()
            env["cuda"] = torch.version.cuda
    except ImportError:
        pass
    return env


def update(path: Path, stage: str, record: dict) -> None:
    """Add one stage's record (with timestamp and environment) to the node's manifest."""
    manifest = read_json(path) if path.exists() else {"stages": {}}
    manifest["stages"][stage] = {
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "environment": environment(),
        **record,
    }
    write_json(path, manifest)
