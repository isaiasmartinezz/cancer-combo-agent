"""SHELDON's model client, set up for our model and behind the same gate as the agent's client.

SHELDON's model tables (release 7fbf027, Oct 14) predate Sonnet 5.5. Left alone it would
  - send no thinking or effort setting, so the model runs at its default effort (high), not the
    agent's medium: the two arms would not be compared at the same setting;
  - price calls at its fallback rate ($2.50 / $15 per M tokens) instead of $2 / $10.
`SheldonLLM` sets thinking and effort for our model in the one place SHELDON builds requests,
and `configure()` adds the list price. Neither edits SHELDON's code.

One SDK gap: SHELDON passes a top-level `cache_control` to `messages.parse()`, which (SDK 2.14)
takes it on `create()` only. The backend here sends the same request field via `extra_body`.

The real client exists only with --allow-llm and an API key, as for the agent (guard.py).
"""
from __future__ import annotations

import os

import sheldon.core.llm as sllm
from sheldon.core.cache import DiskCache

from ccagent.llm import settings
from ccagent.llm.guard import API_KEY_ENV, LLMNotAllowed

MODEL, EFFORT = settings.AGENT_MODEL, settings.AGENT_EFFORT


def configure() -> None:
    inp, out, _read, _write = settings.PRICES[MODEL]
    sllm.PRICING.setdefault(MODEL, (inp * 1e6, out * 1e6))


class SheldonLLM(sllm.LLM):
    def _request(self, prompt, system, model, effort, max_tokens, thinking, cache_prompt):
        req = super()._request(prompt, system, model, effort, max_tokens, thinking, cache_prompt)
        if model == MODEL:
            if thinking:
                req["thinking"] = {"type": "adaptive"}
            req["output_config"] = {"effort": effort}
        return req


class _Backend(sllm.AnthropicBackend):
    def __init__(self, api_key: str | None = None) -> None:
        super().__init__(api_key)
        parse = self.client.messages.parse

        async def parse_with_cache(*args, cache_control=None, extra_body=None, **kwargs):
            if cache_control is not None:
                extra_body = {**(extra_body or {}), "cache_control": cache_control}
            return await parse(*args, extra_body=extra_body, **kwargs)

        self.client.messages.parse = parse_with_cache


class SheldonFake(sllm.FakeLLM, SheldonLLM):
    """SHELDON's scripted offline model, with our request settings (so tests see them)."""


def make_llm(kind: str, allow_llm: bool = False, env=None, cache_path=None,
             **fake_kwargs) -> sllm.LLM:
    """'fake' -> SHELDON's scripted offline model. 'real' -> needs the flag and the key."""
    configure()
    common = dict(model=MODEL, effort=EFFORT, max_tokens=settings.MAX_TOKENS, thinking=True,
                  provider="anthropic")
    if kind == "fake":
        return SheldonFake(**common, **fake_kwargs)
    if kind != "real":
        raise ValueError("kind must be 'fake' or 'real'")
    env = os.environ if env is None else env
    if not allow_llm:
        raise LLMNotAllowed("real model calls need the --allow-llm flag")
    key = env.get(API_KEY_ENV)
    if not key:
        raise LLMNotAllowed(f"real model calls need {API_KEY_ENV} to be set")
    llm = SheldonLLM(**common, api_key=key, cache=DiskCache(cache_path, enabled=True))
    llm._backend_impl = _Backend(key)
    return llm
