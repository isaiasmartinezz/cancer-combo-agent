"""The language model used inside the agent, fixed once for every run so runs are comparable.

Chosen 2026-11-26: Claude Sonnet 5.5 at medium effort, the same setting as iteration 1's agent
runs. If it is ever changed, every earlier agent result must be rerun (see docs/decision_log.md).
"""
AGENT_MODEL: str | None = "claude-sonnet-5-5"
AGENT_EFFORT = "medium"
MAX_TOKENS = 16000
BUDGET_USD = 30.0  # hard cap per key on the summed cost of its real calls, across all runs

# One environment variable, spend log and $BUDGET_USD cap per API key (decision log, Nov 27).
# Key 1 is the original; key 2 keeps its own log, so each key's cap is checked separately.
KEYS = {"1": ("LLM_API_KEY", "runs/llm_spend.jsonl"),
        "2": ("LLM_API_KEY_2", "runs/llm_spend_key2.jsonl")}
SPEND_LOG = KEYS["1"][1]

# USD per token: (input, output, cache read, cache write), list prices as of 2026-11.
# Thinking is billed as output. Cache writes cost 1.25x input (5-minute lifetime).
PRICES = {
    "claude-sonnet-5-5": (2e-6, 10e-6, 0.10e-6, 2.5e-6),
    "claude-opus-5-5": (4e-6, 20e-6, 0.20e-6, 5e-6),
}
# A fallback may be answered by a model not listed above; price it at the highest list price
# (Fable-tier $8 / $40 per million tokens) so spend is never undercounted.
UNKNOWN_PRICE = (8e-6, 40e-6, 0.8e-6, 10e-6)
