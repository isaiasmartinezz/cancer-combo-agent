"""Offline stand-in for a language model, used by every test and dry run."""
from collections.abc import Callable, Iterable

from ccagent.llm.client import LLMResponse


class FakeLLMClient:
    """Returns canned replies in order, or whatever `responder(system, prompt)` returns.

    Every call is recorded in `calls`. Running out of canned replies raises an error
    instead of inventing text, so a test can never pass on made-up model output.
    """

    def __init__(
        self,
        replies: Iterable[str] = (),
        responder: Callable[[str, str], str] | None = None,
    ):
        self._replies = list(replies)
        self._responder = responder
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, prompt: str) -> LLMResponse:
        self.calls.append((system, prompt))
        if self._responder is not None:
            return LLMResponse(text=self._responder(system, prompt))
        if not self._replies:
            raise RuntimeError("FakeLLMClient has no replies left")
        return LLMResponse(text=self._replies.pop(0))
