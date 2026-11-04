"""Loading models and turning conversations into token ids with the model's own chat template.

Generation and training share `prompt_ids`, so the model is trained on exactly the prompt
tokens it later sees at generation time. Heavy imports (torch, transformers) are local to the
functions, so the rest of the library imports on a machine without them.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import ModelSpec

IGNORE = -100  # label value the loss ignores (prompt and padding tokens)


def require_gpu() -> None:
    """This library never runs models on a CPU: it would take days and hide dtype problems."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA GPU found; run this stage on a GPU runtime")
    if not torch.cuda.is_bf16_supported():
        # T4 and older: no bfloat16, and Gemma-2's soft-capping overflows in float16.
        raise RuntimeError(f"{torch.cuda.get_device_name()} lacks bfloat16; use an A100/L4/H100")


def dtype_kwargs(dtype=None) -> dict:
    """transformers renamed `torch_dtype` to `dtype` in 4.56. Default: bfloat16."""
    import torch
    import transformers
    from packaging.version import Version

    key = "dtype" if Version(transformers.__version__) >= Version("4.56") else "torch_dtype"
    return {key: dtype or torch.bfloat16}


def load_tokenizer(path: str, revision: str | None = None):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path, revision=revision)
    if tok.pad_token is None:  # Llama-2 has none; padding is masked out of every loss anyway
        tok.pad_token = tok.eos_token
    return tok


def load_model(path: str, revision: str | None, spec: ModelSpec, for_training: bool = False):
    import torch
    from transformers import AutoModelForCausalLM

    # Training keeps float32 master weights and computes in bfloat16 (mixed precision). Pure
    # bfloat16 weights would silently drop most updates: at learning rate 8e-6 an Adam step is
    # far smaller than bfloat16's resolution (~0.4% of a weight's size).
    model = AutoModelForCausalLM.from_pretrained(
        path,
        revision=revision,
        attn_implementation=spec.attn_implementation,
        device_map=None if for_training else "cuda",
        **dtype_kwargs(torch.float32 if for_training else None),
    )
    if not for_training:
        model.eval()
    return model


# --------------------------------------------------------------------------------------------
# Chat encoding


def prompt_ids(tokenizer, messages: list[dict]) -> list[int]:
    """Token ids of the conversation plus the assistant-turn header, ready for generation."""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # The template already contains BOS; add_special_tokens=False avoids a second one.
    return tokenizer(text, add_special_tokens=False)["input_ids"]


@dataclass(frozen=True)
class Encoded:
    input_ids: list[int]
    labels: list[int]
    truncated: bool
    joined: bool  # True if the prompt/response boundary needed the fallback join


def encode_example(tokenizer, messages: list[dict], response: str, max_len: int) -> Encoded | None:
    """Prompt + response with loss on the response tokens only (DESIGN.md Section 3.4).

    Returns None if truncation to `max_len` leaves no response token.
    """
    p_ids = prompt_ids(tokenizer, messages)
    full_text = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": response}], tokenize=False
    )
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    joined = full_ids[: len(p_ids)] != p_ids
    if joined:
        # The tokenizer merged across the boundary. Use what generation sees instead: the prompt
        # tokens followed by the tokens of the rest of the text.
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if not full_text.startswith(prompt_text):
            raise ValueError("chat template output is not prefix-consistent")
        rest = tokenizer(full_text[len(prompt_text) :], add_special_tokens=False)["input_ids"]
        full_ids = p_ids + rest
    labels = [IGNORE] * len(p_ids) + full_ids[len(p_ids) :]
    truncated = len(full_ids) > max_len
    full_ids, labels = full_ids[:max_len], labels[:max_len]
    if all(x == IGNORE for x in labels):
        return None
    return Encoded(full_ids, labels, truncated, joined)


def pad_batch(rows: list[Encoded], pad_id: int, side: str = "right"):
    """List of encoded examples -> (input_ids, attention_mask, labels) tensors."""
    import torch

    width = max(len(r.input_ids) for r in rows)
    ids, mask, labels = [], [], []
    for r in rows:
        gap = width - len(r.input_ids)
        pad_ids, pad_mask, pad_lab = [pad_id] * gap, [0] * gap, [IGNORE] * gap
        if side == "right":
            ids.append(r.input_ids + pad_ids)
            mask.append([1] * len(r.input_ids) + pad_mask)
            labels.append(r.labels + pad_lab)
        else:
            ids.append(pad_ids + r.input_ids)
            mask.append(pad_mask + [1] * len(r.input_ids))
            labels.append(pad_lab + r.labels)
    return torch.tensor(ids), torch.tensor(mask), torch.tensor(labels)
