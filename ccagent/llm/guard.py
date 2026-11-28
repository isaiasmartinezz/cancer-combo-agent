"""The gate that keeps real LLM calls off until someone explicitly turns them on.

Real calls need BOTH the --allow-llm flag and an API key in the environment.
The default everywhere is the offline fake client. --key picks which key (settings.KEYS): its
environment variable and its own spend log and cap.
"""
import argparse
import os
from collections.abc import Mapping

from ccagent.llm import settings
from ccagent.llm.client import LLMClient
from ccagent.llm.fake import FakeLLMClient

API_KEY_ENV = settings.KEYS["1"][0]


def key_env(key: str = "1") -> str:
    """The environment variable that holds API key `key`."""
    if key not in settings.KEYS:
        raise ValueError(f"unknown key {key!r}; choose one of {sorted(settings.KEYS)}")
    return settings.KEYS[key][0]


def spend_log(key: str = "1") -> str:
    key_env(key)
    return settings.KEYS[key][1]


class LLMNotAllowed(RuntimeError):
    """A real model call was requested without the explicit opt-in."""


def add_llm_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--allow-llm",
        action="store_true",
        help="permit real (billed) model calls; without it only the offline fake is used",
    )
    parser.add_argument(
        "--key", choices=sorted(settings.KEYS), default="1",
        help="which API key to use; each key has its own spend log and cap",
    )


def _build_real_client(api_key: str, run_id: str = "", key: str = "1") -> LLMClient:
    # Kept separate so tests can prove the gate without building a client.
    from ccagent.llm.real import RealClient, SpendLedger

    return RealClient(api_key, run_id=run_id, ledger=SpendLedger(spend_log(key)))


def get_client(
    mode: str = "fake",
    allow_llm: bool = False,
    env: Mapping[str, str] | None = None,
    run_id: str = "",
    key: str = "1",
) -> LLMClient:
    """Return the fake client by default; the real one only if explicitly allowed."""
    if mode == "fake":
        return FakeLLMClient()
    if mode != "real":
        raise ValueError(f"unknown client mode: {mode!r}")
    if not allow_llm:
        raise LLMNotAllowed("real model calls need the --allow-llm flag")
    env = os.environ if env is None else env
    name = key_env(key)
    api_key = env.get(name, "")
    if not api_key:
        raise LLMNotAllowed(f"real model calls need {name} to be set")
    return _build_real_client(api_key, run_id, key)
