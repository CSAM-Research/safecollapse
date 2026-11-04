"""HarmBench's own classifier (`cais/HarmBench-Llama-2-13b-cls`), the cross-check judge on the
HarmBench test split (DESIGN.md Section 5.3). Runs locally on a GPU with vLLM.

The classifier prompts are read from the model card at the pinned revision rather than copied
here, so they are exactly the authors' and their provenance is recorded.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from ..data.download import fetch

CARD_URL = "https://huggingface.co/{repo}/raw/{revision}/README.md"


def load_templates(repo: str, revision: str, cache_dir: Path) -> dict[str, str]:
    """{"prompt": ..., "prompt_contextual": ...} from the model card's LLAMA2_CLS_PROMPT dict."""
    card = fetch(
        CARD_URL.format(repo=repo, revision=revision), cache_dir / "harmbench_cls_README.md"
    )
    text = card.read_text(encoding="utf-8")
    match = re.search(r"LLAMA2_CLS_PROMPT\s*=\s*(\{.*?\n\})", text, re.S)
    if not match:
        raise RuntimeError("could not find LLAMA2_CLS_PROMPT in the classifier's model card")
    templates = ast.literal_eval(match.group(1))  # a literal dict of strings; never executed
    if not {"prompt", "prompt_contextual"} <= templates.keys():
        raise RuntimeError("classifier templates are missing expected keys")
    return templates


def build_input(templates: dict[str, str], meta: dict, generation: str) -> str:
    if meta.get("context"):
        return templates["prompt_contextual"].format(
            behavior=meta["behavior"], generation=generation, context=meta["context"]
        )
    return templates["prompt"].format(behavior=meta["behavior"], generation=generation)


def classify(llm, inputs: list[str]) -> list[int | None]:
    """1 = the generation is an instance of the behaviour; None if the output is not yes/no."""
    from vllm import SamplingParams

    outputs = llm.generate(inputs, SamplingParams(temperature=0.0, max_tokens=1))
    labels: list[int | None] = []
    for out in outputs:
        answer = out.outputs[0].text.strip().lower()
        labels.append(1 if answer == "yes" else 0 if answer == "no" else None)
    return labels


def clip_generation(tokenizer, text: str, max_tokens: int = 512) -> str:
    """HarmBench's own evaluation clips each generation to 512 classifier tokens."""
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return text if len(ids) <= max_tokens else tokenizer.decode(ids[:max_tokens])


def classify_completions(llm, templates, metas, generations, max_model_len: int):
    """Build the classifier inputs (generations clipped as in HarmBench) and classify them.
    Inputs that still exceed the classifier's context are returned as None (reported, never
    silently dropped)."""
    tokenizer = llm.get_tokenizer()
    inputs = [
        build_input(templates, meta, clip_generation(tokenizer, gen))
        for meta, gen in zip(metas, generations)
    ]
    fits = [len(tokenizer(x)["input_ids"]) < max_model_len for x in inputs]
    labels = iter(classify(llm, [x for x, ok in zip(inputs, fits) if ok]))
    return [next(labels) if ok else None for ok in fits]
