"""The real model client. Only reachable through `get_client("real", allow_llm=True)`, which
also needs an API key; nothing here is imported unless a real run is requested.

Each call:
- sends the fixed system prompt (instructions + reference block) as a cached prefix, so after
  the first call of a run it is billed at the cache-read price;
- asks for adaptive thinking at the fixed effort level, and for the step's JSON schema;
- opts into the API's server-side fallback, so a request a safety classifier declines (cancer
  drug content can occasionally trip one) is retried on a fallback model instead of failing.
  The model that actually answered is logged with every call, and the share of fallback-served
  calls should be reported;
- is priced from its token counts and appended to the spend log before the result is returned,
  and refused outright once the logged total reaches the budget.
"""
import json
import time
from pathlib import Path

from ccagent.llm import settings
from ccagent.llm.client import LLMResponse

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class BudgetExceeded(RuntimeError):
    pass


class SpendLedger:
    """Every real call's cost, one JSON line each, summed across all runs."""

    def __init__(self, path: str | Path = settings.SPEND_LOG,
                 budget_usd: float = settings.BUDGET_USD):
        self.path, self.budget_usd = Path(path), budget_usd

    def total(self) -> float:
        if not self.path.exists():
            return 0.0
        return sum(json.loads(line)["cost_usd"] for line in self.path.read_text().splitlines()
                   if line.strip())

    def check(self) -> None:
        spent = self.total()
        if spent >= self.budget_usd:
            raise BudgetExceeded(f"model spend ${spent:.2f} has reached the "
                                 f"${self.budget_usd:.2f} budget; stopping")

    def record(self, entry: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(entry) + "\n")


def price(model: str, usage) -> float:
    inp, out, read, write = settings.PRICES.get(model, settings.UNKNOWN_PRICE)
    return (usage.input_tokens * inp + usage.output_tokens * out
            + (getattr(usage, "cache_read_input_tokens", 0) or 0) * read
            + (getattr(usage, "cache_creation_input_tokens", 0) or 0) * write)


class RealClient:
    def __init__(self, api_key: str, model: str | None = settings.AGENT_MODEL,
                 effort: str = settings.AGENT_EFFORT, ledger: SpendLedger | None = None,
                 sdk=None, run_id: str = ""):
        if not model:
            raise ValueError("no agent model chosen yet: set AGENT_MODEL in "
                             "ccagent/llm/settings.py and record the choice")
        self.model, self.effort, self.run_id = model, effort, run_id
        self.ledger = ledger or SpendLedger()
        if sdk is None:
            import anthropic  # only a real run loads the provider SDK

            sdk = anthropic.Anthropic(api_key=api_key)
        self.sdk = sdk

    def complete(self, system: str, prompt: str, schema: dict | None = None) -> LLMResponse:
        self.ledger.check()
        output_config = {"effort": self.effort}
        if schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": schema}
        response = self.sdk.beta.messages.create(
            model=self.model,
            max_tokens=settings.MAX_TOKENS,
            thinking={"type": "adaptive"},
            output_config=output_config,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": prompt}],
            betas=[FALLBACK_BETA],
            fallbacks="default",
        )
        usage, served = response.usage, response.model
        cost = price(served, usage)
        stopped_early = response.stop_reason in ("refusal", "max_tokens")
        text = "" if stopped_early else next(
            (b.text for b in response.content if b.type == "text"), "")
        self.ledger.record({
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "run_id": self.run_id,
            "model_requested": self.model, "model_served": served, "effort": self.effort,
            "stop_reason": response.stop_reason, "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens, "cost_usd": round(cost, 6)})
        return LLMResponse(
            text=text, input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
            cost_usd=cost,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
            model=served, stop_reason=response.stop_reason)
