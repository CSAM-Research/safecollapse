"""Minimal OpenRouter client: one chat call with provider pinning, retries and backoff.

The API key is read from `OPENROUTER_API_KEY` and is never logged or written to disk.
"""

from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

API = "https://openrouter.ai/api/v1"
HEADERS = {"HTTP-Referer": "https://local.research.wits", "X-Title": "collapse-security-eval"}
RETRYABLE_HTTP = {408, 409, 425, 429, 500, 502, 503, 504}


class JudgeError(RuntimeError):
    pass


@dataclass(frozen=True)
class Reply:
    text: str
    provider: str | None  # who actually served the call
    model: str | None  # model id echoed by the API
    usage: dict


class OpenRouter:
    def __init__(self, api_key: str | None = None, timeout_s: int = 90, max_retries: int = 5):
        self._key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self._key:
            raise JudgeError("OPENROUTER_API_KEY is not set")
        self.timeout_s = timeout_s
        self.max_retries = max_retries

    def chat(
        self,
        model: str,
        prompt: str,
        provider_order: tuple[str, ...],
        max_tokens: int,
        system: str | None = None,
        reasoning_effort: str | None = None,
    ) -> Reply:
        messages = [{"role": "system", "content": system}] if system else []
        payload = {
            "model": model,
            "messages": messages + [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        if reasoning_effort:
            payload["reasoning"] = {"effort": reasoning_effort}
        if provider_order:
            # Fallbacks disabled: a pinned judge must never be silently swapped for another host.
            payload["provider"] = {"order": list(provider_order), "allow_fallbacks": False}
        body = self._post("/chat/completions", payload)
        try:
            text = body["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise JudgeError(f"unexpected response shape: {str(body)[:300]}") from exc
        return Reply(text, body.get("provider"), body.get("model"), body.get("usage") or {})

    def _post(self, path: str, payload: dict) -> dict:
        request = urllib.request.Request(
            API + path,
            data=json.dumps(payload).encode(),
            method="POST",
            headers={
                "Authorization": f"Bearer {self._key}",
                "Content-Type": "application/json",
                **HEADERS,
            },
        )
        last = "no attempt made"
        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as resp:
                    body = json.loads(resp.read().decode())
                if "error" in body:  # OpenRouter can return 200 with an error object
                    raise JudgeError(f"API error: {str(body['error'])[:300]}")
                return body
            except urllib.error.HTTPError as exc:
                last = f"HTTP {exc.code}: {exc.read().decode(errors='replace')[:300]}"
                if exc.code not in RETRYABLE_HTTP:
                    raise JudgeError(last) from exc
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, JudgeError) as exc:
                last = str(exc)
            # Exponential backoff with jitter, capped at a minute.
            time.sleep(min(60, 2**attempt) + random.random())
        raise JudgeError(f"gave up after {self.max_retries} attempts: {last}")


def endpoints(model: str, timeout_s: int = 30) -> list[dict]:
    """Public list of providers serving `model` (no key needed); used by `check-judges`."""
    with urllib.request.urlopen(f"{API}/models/{model}/endpoints", timeout=timeout_s) as resp:
        return json.loads(resp.read().decode())["data"]["endpoints"]
