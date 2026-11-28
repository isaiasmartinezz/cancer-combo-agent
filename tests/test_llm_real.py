"""The real client, tested against a stand-in for the provider SDK: no network, no key."""
import json
import sys
from datetime import date
from types import SimpleNamespace

import pytest

from ccagent.llm import settings
from ccagent.llm.real import BudgetExceeded, RealClient, SpendLedger, billed_output, price, rates
from ccagent.steps import SCHEMAS

ARGON = "gemini-4-argon"


def usage(inp=1000, out=200, thought=300, cached=0, total=None):
    return SimpleNamespace(total_input_tokens=inp, total_output_tokens=out,
                           total_thought_tokens=thought, total_cached_tokens=cached,
                           total_tool_use_tokens=0,
                           total_tokens=inp + out + thought if total is None else total)


class StubSDK:
    """Records each request and answers with a canned interaction."""

    def __init__(self, text='{"picks": []}', model=ARGON, status="completed", use=None,
                 errors=None, fail_first=0, fail_code=503):
        self.requests, self.fail_first, self.fail_code = [], fail_first, fail_code
        self.response = SimpleNamespace(model=model, status=status, output_text=text,
                                        usage=use or usage(), errors=errors or [])
        self.interactions = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        if self.fail_first:
            self.fail_first -= 1
            raise type("ServerError", (Exception,), {"code": self.fail_code})("busy")
        return self.response


def client(tmp_path, sdk=None, budget=48.0, model=ARGON):
    return RealClient("key", model=model, effort="medium", pause=0,
                      ledger=SpendLedger(tmp_path / "spend.jsonl", budget), sdk=sdk or StubSDK())


def test_request_shape(tmp_path):
    sdk = StubSDK()
    client(tmp_path, sdk).complete("SYSTEM", "PROMPT", SCHEMAS["select"])
    (req,) = sdk.requests
    assert req["model"] == ARGON and req["store"] is False
    assert req["system_instruction"] == "SYSTEM" and req["input"] == "PROMPT"
    assert req["generation_config"] == {"thinking_level": "medium",
                                        "max_output_tokens": settings.MAX_TOKENS}
    assert req["response_format"] == {"type": "text", "mime_type": "application/json",
                                      "schema": SCHEMAS["select"]}
    assert "temperature" not in json.dumps(req)


def test_answer_text_tokens_and_cost_with_thoughts_counted_once(tmp_path):
    r = client(tmp_path).complete("s", "p")
    assert r.text == '{"picks": []}' and r.model == ARGON
    in_rate, out_rate = rates(ARGON)
    assert r.output_tokens == 500  # 200 output + 300 thought, reported separately
    assert r.cost_usd == pytest.approx(1000 * in_rate + 500 * out_rate)


def test_thoughts_inside_output_are_not_charged_twice():
    u = {"total_input_tokens": 1000, "total_output_tokens": 500, "total_thought_tokens": 300,
         "total_tokens": 1500}  # 1000 + 500: output already holds the 300 thoughts
    assert billed_output(u) == 500
    assert billed_output({**u, "total_tokens": None}) == 800  # unknown: charge both


def test_cached_input_gets_no_discount_so_spend_is_never_undercounted():
    u = {"total_input_tokens": 10_000, "total_cached_tokens": 8_000, "total_output_tokens": 0}
    assert price(ARGON, u, date(2026, 11, 28)) == pytest.approx(10_000 * 2e-6)


def test_price_doubles_on_december_first():
    assert rates(ARGON, date(2026, 11, 30)) == (2e-6, 10e-6)
    assert rates(ARGON, date(2026, 12, 1)) == (4e-6, 20e-6)


@pytest.mark.parametrize("status", ["incomplete", "failed", "budget_exceeded", "cancelled",
                                    "requires_continuation"])
def test_any_unfinished_status_returns_no_text_so_the_step_retries(tmp_path, status):
    r = client(tmp_path, StubSDK(status=status, text='{"partial"')).complete("s", "p")
    assert r.text == "" and r.stop_reason.startswith(status)


def test_errors_are_logged_with_the_status(tmp_path):
    err = SimpleNamespace(message="blocked by safety settings")
    r = client(tmp_path, StubSDK(status="failed", errors=[err])).complete("s", "p")
    assert r.stop_reason == "failed: blocked by safety settings"
    entry = json.loads((tmp_path / "spend.jsonl").read_text())
    assert entry["errors"] == ["blocked by safety settings"] and entry["status"] == "failed"


def test_transient_errors_are_retried_and_others_raised(tmp_path):
    sdk = StubSDK(fail_first=2)
    assert client(tmp_path, sdk).complete("s", "p").text
    assert len(sdk.requests) == 3
    bad = StubSDK(fail_first=1, fail_code=400)
    with pytest.raises(Exception, match="busy"):
        client(tmp_path, bad).complete("s", "p")
    assert len(bad.requests) == 1


def test_unknown_served_model_is_priced_conservatively(tmp_path):
    r = client(tmp_path, StubSDK(model="gemini-future", use=usage(1000, 0, 0))).complete("s", "p")
    assert r.cost_usd == pytest.approx(1000 * settings.UNKNOWN_PRICE[0])
    entry = json.loads((tmp_path / "spend.jsonl").read_text())
    assert entry["model_requested"] == ARGON and entry["model_served"] == "gemini-future"
    assert entry["total_thought_tokens"] == 0  # raw usage is logged for exact recomputation


def test_every_call_is_logged_and_the_budget_stops_further_calls(tmp_path):
    in_rate = rates(ARGON)[0]
    expensive = StubSDK(use=usage(1_000_000, 0, 0))
    c = client(tmp_path, expensive, budget=2.5 * in_rate * 1_000_000)
    for _ in range(3):
        c.complete("s", "p")
    with pytest.raises(BudgetExceeded):
        c.complete("s", "p")
    assert len(expensive.requests) == 3  # the refused call never reached the API


def test_ledger_sums_across_clients_and_runs(tmp_path):
    for _ in range(2):
        client(tmp_path, StubSDK(use=usage(500_000, 0, 0))).complete("s", "p")
    assert SpendLedger(tmp_path / "spend.jsonl").total() == pytest.approx(
        2 * 500_000 * rates(ARGON)[0])


def test_no_model_chosen_means_no_client():
    with pytest.raises(ValueError, match="no agent model chosen"):
        RealClient("key", model=None, sdk=StubSDK())
    assert settings.AGENT_MODEL == ARGON and settings.AGENT_MODEL in settings.PRICES


def test_building_the_real_client_with_a_stub_does_not_load_the_sdk(tmp_path):
    client(tmp_path)
    assert "google.genai" not in sys.modules


def test_schemas_are_strict_objects():
    for schema in SCHEMAS.values():
        item = next(iter(schema["properties"].values()))["items"]
        assert schema["additionalProperties"] is False and item["additionalProperties"] is False
        assert set(item["required"]) == set(item["properties"])
