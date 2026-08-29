"""
OpenAI-SDK wrapper for scenario + label generation (Section 8.4) and the LLM-judge
naturalness score (Section 8.5).

Uses the official `openai` python SDK. `base_url` is configurable so the *same* code
talks to OpenAI, or any OpenAI-compatible endpoint (Sarvam v2, vLLM, Together, Groq...).
Set the key in the environment:

    OPENAI_API_KEY   (required)
    OPENAI_BASE_URL  (optional; overrides cfg.llm_base_url)

The client is intentionally small and robust: JSON extraction tolerates stray markdown
fences, and requests retry with exponential backoff on transient errors.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any

from thinkspark.config import DataGenConfig, env

try:  # imported lazily-friendly so `import thinkspark` never hard-fails without openai
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_FIRST_OBJ = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of an LLM response, tolerating fences/prose."""
    text = text.strip()
    m = _JSON_FENCE.search(text)
    if m:
        return json.loads(m.group(1))
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = _FIRST_OBJ.search(text)
    if not m:
        raise ValueError(f"no JSON object found in response: {text[:200]!r}")
    return json.loads(m.group(0))


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def cost_usd(self, price_in_per_1m: float, price_out_per_1m: float) -> float:
        return (self.prompt_tokens / 1_000_000.0) * price_in_per_1m + \
               (self.completion_tokens / 1_000_000.0) * price_out_per_1m


@dataclass
class LLMClient:
    model: str
    base_url: str | None = None
    temperature: float = 0.9
    max_tokens: int = 1200
    max_retries: int = 5
    price_in_per_1m: float = 0.0
    price_out_per_1m: float = 0.0

    def __post_init__(self):
        if OpenAI is None:
            raise RuntimeError("openai SDK not installed. `pip install openai>=1.40`.")
        api_key = env("OPENAI_API_KEY", required=True)
        base_url = os.environ.get("OPENAI_BASE_URL") or self.base_url
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    # ------------------------------------------------------------------ #
    @classmethod
    def for_generation(cls, cfg: DataGenConfig) -> "LLMClient":
        return cls(
            model=cfg.llm_model,
            base_url=cfg.llm_base_url,
            temperature=cfg.llm_temperature,
            max_tokens=cfg.llm_max_tokens,
            max_retries=cfg.llm_max_retries,
            price_in_per_1m=cfg.llm_price_in_per_1m_usd,
            price_out_per_1m=cfg.llm_price_out_per_1m_usd,
        )

    @classmethod
    def for_judge(cls, cfg: DataGenConfig) -> "LLMClient":
        return cls(model=cfg.judge_model, base_url=cfg.llm_base_url,
                   temperature=0.0, max_tokens=200,
                   price_in_per_1m=cfg.llm_price_in_per_1m_usd,
                   price_out_per_1m=cfg.llm_price_out_per_1m_usd)

    # ------------------------------------------------------------------ #
    def chat_json(self, system: str, user: str) -> dict[str, Any]:
        """Send a system+user turn and parse a single JSON object back (no usage)."""
        raw, _usage = self._chat_raw(system, user)
        return extract_json(raw)

    def chat_json_with_usage(
        self, system: str, user: str, max_tokens: int | None = None
    ) -> tuple[dict[str, Any], Usage]:
        """
        Send a system+user turn, parse ONE JSON object back, and return token usage so
        the caller can log cost. Used by batched scenario generation (script 02).
        """
        raw, usage = self._chat_raw(system, user, max_tokens=max_tokens)
        return extract_json(raw), usage

    # ------------------------------------------------------------------ #
    def _chat_raw(
        self, system: str, user: str, max_tokens: int | None = None
    ) -> tuple[str, Usage]:
        mt = max_tokens or self.max_tokens
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=mt,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    # Ask for JSON where the backend supports it; harmless otherwise.
                    response_format={"type": "json_object"},
                )
                return resp.choices[0].message.content or "", _usage_of(resp)
            except TypeError:
                # Endpoint doesn't accept response_format -> retry without it.
                resp = self._client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=mt,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                return resp.choices[0].message.content or "", _usage_of(resp)
            except Exception as e:  # transient API/network error -> backoff
                last_err = e
                sleep = min(2 ** attempt, 30)
                time.sleep(sleep)
        raise RuntimeError(f"LLM call failed after {self.max_retries} retries: {last_err}")


def _usage_of(resp) -> Usage:
    u = getattr(resp, "usage", None)
    if u is None:
        return Usage()
    return Usage(prompt_tokens=int(getattr(u, "prompt_tokens", 0) or 0),
                completion_tokens=int(getattr(u, "completion_tokens", 0) or 0))
