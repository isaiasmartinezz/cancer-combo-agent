"""The gate that keeps real LLM calls off until someone explicitly turns them on.

Real calls need BOTH the --allow-llm flag and an API key in the environment.
The default everywhere is the offline fake client.
"""
import argparse
import os
from collections.abc import Mapping

from ccagent.llm.client import LLMClient
from ccagent.llm.fake import FakeLLMClient

API_KEY_ENV = "LLM_API_KEY"


class LLMNotAllowed(RuntimeError):
    """A real model call was requested without the explicit opt-in."""


def add_llm_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--allow-llm",
        action="store_true",
        help="permit real (billed) model calls; without it only the offline fake is used",
    )


def _build_real_client(api_key: str, run_id: str = "") -> LLMClient:
    # Kept separate so tests can prove the gate without building a client.
    from ccagent.llm.real import RealClient

    return RealClient(api_key, run_id=run_id)


def get_client(
    mode: str = "fake",
    allow_llm: bool = False,
    env: Mapping[str, str] | None = None,
    run_id: str = "",
) -> LLMClient:
    """Return the fake client by default; the real one only if explicitly allowed."""
    if mode == "fake":
        return FakeLLMClient()
    if mode != "real":
        raise ValueError(f"unknown client mode: {mode!r}")
    if not allow_llm:
        raise LLMNotAllowed("real model calls need the --allow-llm flag")
    env = os.environ if env is None else env
    key = env.get(API_KEY_ENV, "")
    if not key:
        raise LLMNotAllowed(f"real model calls need {API_KEY_ENV} to be set")
    return _build_real_client(key, run_id)
