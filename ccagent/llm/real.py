"""The real model client: Gemini 4 Argon through the Gemini Interactions API (google-genai).
Only reachable through `get_client("real", allow_llm=True)`, which also needs an API key;
nothing here imports the provider SDK unless a real run is requested.

Each call:
- sends the fixed system prompt (instructions + reference block) as the system instruction and
  the step's prompt as the input, so the shared prefix comes first (implicit caching, which the
  API applies on its own above 4,096 prompt tokens);
- sets the thinking level and the step's JSON schema (response_format), and store=False, so
  requests are not kept for later retrieval;
- retries a transient failure (rate limit, server error) a few times with a pause;
- treats any status other than "completed" (incomplete, failed, blocked...) as no answer, so
  the step's retry and BO fallback take over and the failure is counted;
- is priced from its token counts and appended to the spend log before the result is returned,
  and refused outright once the logged total reaches the budget.

Pricing never undercounts. The API's documentation does not say whether cached tokens are part
of total_input_tokens or whether thought tokens are part of total_output_tokens, so all input
is charged at the full input rate (no cache discount assumed) and thought tokens are added to
output unless total_tokens shows they are already inside it. The raw usage is logged with every
call, so exact cost can be recomputed later.
"""
import json
import time
from datetime import date
from pathlib import Path

from ccagent.llm import settings
from ccagent.llm.client import LLMResponse

RETRY_STATUS = {408, 429, 500, 502, 503, 504}
RETRIES, PAUSE = 4, 5.0  # seconds, doubled after each failed try


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


def rates(model: str, today: date | None = None) -> tuple[float, float]:
    """(input, output) USD per token on `today`, after any scheduled price change."""
    today = today or date.today()
    inp, out = settings.PRICES.get(model, settings.UNKNOWN_PRICE)
    for start, new in settings.PRICE_CHANGES.get(model, ()):
        if today >= date.fromisoformat(start):
            inp, out = new
    return inp, out


def billed_output(usage: dict) -> int:
    """Output tokens to charge: thoughts added unless total_tokens shows output includes them."""
    inp, out, thought = (usage.get(k) or 0 for k in
                         ("total_input_tokens", "total_output_tokens", "total_thought_tokens"))
    total = usage.get("total_tokens")
    if total is not None and thought and total < inp + out + thought:
        return out  # the totals only add up if output already counts the thoughts
    return out + thought


def price(model: str, usage: dict, today: date | None = None) -> float:
    inp_rate, out_rate = rates(model, today)
    return (usage.get("total_input_tokens") or 0) * inp_rate + billed_output(usage) * out_rate


def _usage_dict(usage) -> dict:
    fields = ("total_input_tokens", "total_output_tokens", "total_thought_tokens",
              "total_cached_tokens", "total_tool_use_tokens", "total_tokens")
    return {f: getattr(usage, f, None) for f in fields} if usage is not None else {}


def _transient(error: Exception) -> bool:
    return getattr(error, "code", None) in RETRY_STATUS


class RealClient:
    def __init__(self, api_key: str, model: str | None = settings.AGENT_MODEL,
                 effort: str = settings.AGENT_EFFORT, ledger: SpendLedger | None = None,
                 sdk=None, run_id: str = "", pause: float = PAUSE):
        if not model:
            raise ValueError("no agent model chosen yet: set AGENT_MODEL in "
                             "ccagent/llm/settings.py and record the choice")
        self.model, self.effort, self.run_id, self.pause = model, effort, run_id, pause
        self.ledger = ledger or SpendLedger()
        if sdk is None:
            from google import genai  # only a real run loads the provider SDK

            sdk = genai.Client(api_key=api_key)
        self.sdk = sdk

    def _create(self, **request):
        wait = self.pause
        for attempt in range(RETRIES):
            try:
                return self.sdk.interactions.create(**request)
            except Exception as error:  # the SDK's APIError carries the HTTP status as .code
                if not _transient(error) or attempt == RETRIES - 1:
                    raise
                time.sleep(wait)
                wait *= 2

    def complete(self, system: str, prompt: str, schema: dict | None = None) -> LLMResponse:
        self.ledger.check()
        request = dict(model=self.model, system_instruction=system, input=prompt, store=False,
                       generation_config={"thinking_level": self.effort,
                                          "max_output_tokens": settings.MAX_TOKENS})
        if schema is not None:
            request["response_format"] = {"type": "text", "mime_type": "application/json",
                                          "schema": schema}
        response = self._create(**request)
        usage = _usage_dict(getattr(response, "usage", None))
        served = getattr(response, "model", None) or self.model
        cost = price(served, usage)
        status = str(getattr(response, "status", "") or "")
        errors = [getattr(e, "message", str(e)) for e in (getattr(response, "errors", None) or [])]
        text = (getattr(response, "output_text", None) or "") if status == "completed" else ""
        self.ledger.record({
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "run_id": self.run_id,
            "model_requested": self.model, "model_served": served, "effort": self.effort,
            "status": status, "errors": errors, **usage, "cost_usd": round(cost, 6)})
        return LLMResponse(
            text=text, input_tokens=usage.get("total_input_tokens") or 0,
            output_tokens=billed_output(usage), cost_usd=cost,
            cache_read_tokens=usage.get("total_cached_tokens") or 0,
            model=served, stop_reason=status if not errors else f"{status}: {errors[0]}")
