"""Fetch raw dataset files at pinned revisions. Plain HTTPS, so data preparation needs no GPU
libraries. Gated Hugging Face files (SORRY-Bench) use the `HF_TOKEN` environment variable."""

from __future__ import annotations

import os
import shutil
import urllib.error
import urllib.request
from pathlib import Path


def hf_url(repo: str, filename: str, revision: str) -> str:
    return f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{filename}"


def github_url(repo: str, path: str, commit: str) -> str:
    return f"https://raw.githubusercontent.com/{repo}/{commit}/{path}"


def fetch(url: str, dest: Path, timeout_s: int = 120) -> Path:
    """Download once; a file already on disk is reused (its hash is checked by the caller)."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url)
    token = os.environ.get("HF_TOKEN")
    if token and url.startswith("https://huggingface.co/"):
        request.add_header("Authorization", f"Bearer {token}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as resp, open(tmp, "wb") as out:
            shutil.copyfileobj(resp, out)
    except urllib.error.HTTPError as exc:
        hint = ""
        if exc.code in (401, 403) and "huggingface.co" in url:
            hint = " (gated dataset: accept its terms on Hugging Face and set HF_TOKEN)"
        raise RuntimeError(f"download failed: HTTP {exc.code} for {url}{hint}") from exc
    tmp.replace(dest)
    return dest
