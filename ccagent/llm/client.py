"""The one interface the rest of the code uses to talk to a language model."""
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class LLMResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    model: str = ""          # the model that actually answered (may differ after a fallback)
    stop_reason: str = ""


class LLMClient(Protocol):
    def complete(self, system: str, prompt: str, schema: dict | None = None) -> LLMResponse:
        """schema: optional JSON schema the answer must follow."""
        ...
