"""SHELDON's model client, set up for our model and behind the same gate as the agent's client.

SHELDON reaches Gemini through its OpenAI-compatible backend, pointed at the Gemini API's
OpenAI-compatible endpoint. Left alone that backend would
  - send no thinking setting ("no portable equivalent across proxied models"), so the model
    would run at its own default instead of the agent's thinking level: `_Backend` adds
    reasoning_effort, which the endpoint maps to Gemini 4's thinking_level of the same name;
  - price calls at SHELDON's fallback rate: `configure()` adds our list price;
  - assume OpenAI's token conventions: `_Backend._usage` adds reasoning tokens to output unless
    the totals show they are already inside it, the same never-undercount rule as the agent.
None of this edits SHELDON's code.

The real client exists only with --allow-llm and an API key, as for the agent (guard.py).
"""
from __future__ import annotations

import os

import sheldon.core.llm as sllm
from sheldon.core.cache import DiskCache

from ccagent.llm import settings
from ccagent.llm.guard import LLMNotAllowed, key_env
from ccagent.llm.real import rates

MODEL, EFFORT = settings.AGENT_MODEL, settings.AGENT_EFFORT
PROVIDER = "openai"
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


def configure() -> None:
    inp, out = rates(MODEL)  # today's price, after any scheduled change
    sllm.PRICING[MODEL] = (inp * 1e6, out * 1e6)


class SheldonLLM(sllm.LLM):
    """SHELDON's model client; the Gemini specifics live in the backend."""


def reasoning_outside_completion(usage) -> int:
    """Reasoning tokens to add to completion_tokens: all of them unless total_tokens shows
    completion_tokens already includes them."""
    details = getattr(usage, "completion_tokens_details", None)
    reasoning = getattr(details, "reasoning_tokens", 0) or 0
    prompt = getattr(usage, "prompt_tokens", 0) or 0
    completion = getattr(usage, "completion_tokens", 0) or 0
    total = getattr(usage, "total_tokens", None)
    if reasoning and total is not None and total < prompt + completion + reasoning:
        return 0
    return reasoning


class _Backend(sllm.OpenAIBackend):
    def __init__(self, api_key: str | None = None) -> None:
        super().__init__(BASE_URL, api_key)

    def _params(self, req, schema):
        params = super()._params(req, schema)
        params["reasoning_effort"] = EFFORT
        return params

    @staticmethod
    def _usage(resp):
        usage = sllm.OpenAIBackend._usage(resp)
        extra = reasoning_outside_completion(getattr(resp, "usage", None))
        if extra:
            usage.output_tokens += extra
            usage.cost_usd = sllm.estimate_cost(getattr(resp, "model", "") or MODEL, usage)
        return usage


class SheldonFake(sllm.FakeLLM, SheldonLLM):
    """SHELDON's scripted offline model, with our request settings (so tests see them)."""


def make_llm(kind: str, allow_llm: bool = False, env=None, cache_path=None, key: str = "1",
             **fake_kwargs) -> sllm.LLM:
    """'fake' -> SHELDON's scripted offline model. 'real' -> needs the flag and the key."""
    configure()
    common = dict(model=MODEL, effort=EFFORT, max_tokens=settings.MAX_TOKENS, thinking=True,
                  provider=PROVIDER)
    if kind == "fake":
        return SheldonFake(**common, **fake_kwargs)
    if kind != "real":
        raise ValueError("kind must be 'fake' or 'real'")
    env = os.environ if env is None else env
    if not allow_llm:
        raise LLMNotAllowed("real model calls need the --allow-llm flag")
    name = key_env(key)
    api_key = env.get(name)
    if not api_key:
        raise LLMNotAllowed(f"real model calls need {name} to be set")
    llm = SheldonLLM(**common, api_key=api_key, base_url=BASE_URL,
                     cache=DiskCache(cache_path, enabled=True))
    llm._backend_impl = _Backend(api_key)
    return llm
