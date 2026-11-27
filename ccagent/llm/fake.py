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

    def complete(self, system: str, prompt: str, schema: dict | None = None) -> LLMResponse:
        self.calls.append((system, prompt))
        if self._responder is not None:
            return LLMResponse(text=self._responder(system, prompt))
        if not self._replies:
            raise RuntimeError("FakeLLMClient has no replies left")
        return LLMResponse(text=self._replies.pop(0))


def top_k_responder(system: str, prompt: str) -> str:
    """A scripted stand-in for dry runs: valid answers that never use judgment.

    hypothesize: one statistical hypothesis about the first shortlisted pair
    critique:    keep every hypothesis
    select:      the shortlist's first k rows, which is exactly Bayesian optimization's batch
    So an agent run with this responder must reproduce BO pick for pick; any difference means
    the agent and the baselines are not seeing the same candidates.
    """
    import io
    import json
    import re

    import pandas as pd

    def section(name: str) -> str:
        found = re.search(rf"## {re.escape(name)}\n(.*?)(?=\n\n## |\Z)", prompt, re.S)
        return found.group(1) if found else ""

    if prompt.startswith("Propose up to"):
        first = pd.read_csv(io.StringIO(section("Shortlist"))).iloc[0]
        return json.dumps({"hypotheses": [{
            "statement": f"{first.drug_a} with {first.drug_b} resembles pairs that scored well.",
            "drugs": [first.drug_a, first.drug_b], "level": "statistical",
            "predicted_responders": [], "rationale": "scripted", "evidence_ids": []}]})
    if prompt.startswith("Challenge each"):
        ids = [h["id"] for h in json.loads(section("Hypotheses"))]
        return json.dumps({"verdicts": [{"id": i, "verdict": "keep", "reason": "scripted"}
                                        for i in ids]})
    if prompt.startswith("Choose exactly"):
        k = int(re.match(r"Choose exactly (\d+)", prompt).group(1))
        ids = pd.read_csv(io.StringIO(section("Shortlist")))["exp_id"].iloc[:k]
        return json.dumps({"picks": [{"exp_id": e, "hypothesis_id": "explore",
                                      "level": "statistical", "reason": "scripted top-k"}
                                     for e in ids]})
    raise ValueError("top_k_responder got a prompt it does not recognize")
