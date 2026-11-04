"""Shared fixtures. Every test is pure logic: no GPU, no model weights, no network."""

from __future__ import annotations

import pytest

from safecollapse.config import load_config


@pytest.fixture
def cfg(tmp_path):
    """The core config with every path inside a temporary folder."""
    return load_config(
        "core",
        [
            f"paths.runs_dir={tmp_path / 'runs'}",
            f"paths.data_dir={tmp_path / 'data'}",
        ],
    )


class CharTokenizer:
    """Stand-in tokenizer: one token per character, with a Gemma-like chat template."""

    eos_token_id = 1
    pad_token_id = 0
    unk_token_id = 2

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = "^"
        for m in messages:
            role = "model" if m["role"] == "assistant" else m["role"]
            text += f"<{role}>{m['content'].strip()}|"
        if add_generation_prompt:
            text += "<model>"
        return text

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


@pytest.fixture
def tokenizer():
    return CharTokenizer()
