"""The real client, tested against a stand-in for the provider SDK: no network, no key."""
import json
import sys
from types import SimpleNamespace

import pytest

from ccagent.llm import settings
from ccagent.llm.real import BudgetExceeded, RealClient, SpendLedger, price
from ccagent.steps import SCHEMAS


class StubSDK:
    """Records each request and answers with a canned response."""

    def __init__(self, text='{"picks": []}', model="claude-sonnet-5-5", stop="end_turn",
                 usage=(1000, 200, 5000, 0)):
        self.requests = []
        inp, out, read, write = usage
        self.response = SimpleNamespace(
            model=model, stop_reason=stop,
            content=[SimpleNamespace(type="thinking", thinking=""),
                     SimpleNamespace(type="text", text=text)],
            usage=SimpleNamespace(input_tokens=inp, output_tokens=out,
                                  cache_read_input_tokens=read,
                                  cache_creation_input_tokens=write))
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        return self.response


def client(tmp_path, sdk=None, budget=30.0, model="claude-sonnet-5-5"):
    return RealClient("key", model=model, effort="medium",
                      ledger=SpendLedger(tmp_path / "spend.jsonl", budget), sdk=sdk or StubSDK())


def test_request_shape(tmp_path):
    sdk = StubSDK()
    client(tmp_path, sdk).complete("SYSTEM", "PROMPT", SCHEMAS["select"])
    (req,) = sdk.requests
    assert req["model"] == "claude-sonnet-5-5"
    assert req["thinking"] == {"type": "adaptive"}
    assert req["output_config"]["effort"] == "medium"
    assert req["output_config"]["format"] == {"type": "json_schema",
                                              "schema": SCHEMAS["select"]}
    assert req["system"] == [{"type": "text", "text": "SYSTEM",
                              "cache_control": {"type": "ephemeral"}}]
    assert req["messages"] == [{"role": "user", "content": "PROMPT"}]
    assert req["fallbacks"] == "default" and req["betas"] == ["server-side-fallback-2026-07-01"]
    assert "temperature" not in req and "budget_tokens" not in json.dumps(req)


def test_answer_text_tokens_and_cost(tmp_path):
    r = client(tmp_path).complete("s", "p")
    assert r.text == '{"picks": []}' and r.model == "claude-sonnet-5-5"
    expected = 1000 * 2e-6 + 200 * 10e-6 + 5000 * 0.10e-6
    assert r.cost_usd == pytest.approx(expected)
    assert (r.cache_read_tokens, r.input_tokens) == (5000, 1000)


def test_refusal_returns_no_text_so_the_step_retries_then_falls_back(tmp_path):
    r = client(tmp_path, StubSDK(stop="refusal")).complete("s", "p")
    assert r.text == "" and r.stop_reason == "refusal"


def test_fallback_model_is_logged_and_priced_conservatively(tmp_path):
    led = tmp_path / "spend.jsonl"
    r = client(tmp_path, StubSDK(model="some-future-model", usage=(1000, 0, 0, 0))).complete(
        "s", "p")
    assert r.cost_usd == pytest.approx(1000 * settings.UNKNOWN_PRICE[0])
    entry = json.loads(led.read_text())
    assert entry["model_requested"] == "claude-sonnet-5-5"
    assert entry["model_served"] == "some-future-model"


def test_every_call_is_logged_and_the_budget_stops_further_calls(tmp_path):
    expensive = StubSDK(usage=(1_000_000, 0, 0, 0))  # $2.00 per call on Sonnet 5.5
    c = client(tmp_path, expensive, budget=5.0)
    for _ in range(3):
        c.complete("s", "p")
    assert c.ledger.total() == pytest.approx(6.0)
    with pytest.raises(BudgetExceeded):
        c.complete("s", "p")
    assert len(expensive.requests) == 3  # the refused call never reached the API


def test_ledger_sums_across_clients_and_runs(tmp_path):
    for _ in range(2):
        client(tmp_path, StubSDK(usage=(500_000, 0, 0, 0))).complete("s", "p")
    assert SpendLedger(tmp_path / "spend.jsonl").total() == pytest.approx(2.0)


def test_no_model_chosen_means_no_client(tmp_path):
    with pytest.raises(ValueError, match="no agent model chosen"):
        RealClient("key", model=None, sdk=StubSDK())
    assert settings.AGENT_MODEL is None or settings.AGENT_MODEL in settings.PRICES


def test_price_table_covers_cache_reads_and_writes():
    usage = SimpleNamespace(input_tokens=0, output_tokens=0, cache_read_input_tokens=1_000_000,
                            cache_creation_input_tokens=1_000_000)
    assert price("claude-opus-5-5", usage) == pytest.approx(0.20 + 5.0)


def test_building_the_real_client_with_a_stub_does_not_load_the_sdk(tmp_path):
    client(tmp_path)
    assert "anthropic" not in sys.modules


def test_schemas_are_strict_objects():
    for schema in SCHEMAS.values():
        item = next(iter(schema["properties"].values()))["items"]
        assert schema["additionalProperties"] is False and item["additionalProperties"] is False
        assert set(item["required"]) == set(item["properties"])
