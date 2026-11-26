import socket
import sys

import pytest

from ccagent.llm import FakeLLMClient, LLMNotAllowed, get_client, guard


def test_default_client_is_the_offline_fake():
    assert isinstance(get_client(), FakeLLMClient)


def test_real_client_refused_without_flag():
    with pytest.raises(LLMNotAllowed):
        get_client("real", allow_llm=False, env={guard.API_KEY_ENV: "k"})


def test_real_client_refused_without_key():
    with pytest.raises(LLMNotAllowed):
        get_client("real", allow_llm=True, env={})


def test_real_client_built_only_with_flag_and_key(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(guard, "_build_real_client", lambda key: sentinel)
    assert get_client("real", allow_llm=True, env={guard.API_KEY_ENV: "k"}) is sentinel


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        get_client("other")


def test_importing_the_package_loads_no_provider_sdk():
    assert "anthropic" not in sys.modules
    assert "openai" not in sys.modules


def test_fake_client_replays_in_order_and_records_calls():
    fake = FakeLLMClient(replies=["a", "b"])
    assert fake.complete("s", "p1").text == "a"
    assert fake.complete("s", "p2").text == "b"
    assert fake.calls == [("s", "p1"), ("s", "p2")]


def test_fake_client_never_invents_a_reply():
    with pytest.raises(RuntimeError):
        FakeLLMClient().complete("s", "p")


def test_network_is_blocked_in_tests():
    with pytest.raises(RuntimeError):
        socket.create_connection(("example.com", 80))
