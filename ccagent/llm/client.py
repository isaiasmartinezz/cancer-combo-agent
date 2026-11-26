"""The one interface the rest of the code uses to talk to a language model."""
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class LLMResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


class LLMClient(Protocol):
    def complete(self, system: str, prompt: str) -> LLMResponse: ...
