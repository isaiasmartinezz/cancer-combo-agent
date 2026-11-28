"""The language model used inside the agent, fixed once for every run so runs are comparable.

Chosen 2026-11-27 (Isaias): Gemini 4 Argon, thinking level medium, for every arm and SHELDON.
It replaces Claude Sonnet 5.5 (chosen 2026-11-26), because the project's API credit is on the
Gemini API; no real run had been made with Sonnet in this repo. If the model is ever changed
again, every earlier result must be rerun (see docs/decision_log.md).
"""
AGENT_PROVIDER = "gemini"
AGENT_MODEL: str | None = "gemini-4-argon"
AGENT_EFFORT = "medium"  # Gemini thinking_level; Argon's default, and the level used before
MAX_TOKENS = 16000
# Hard cap per key on the summed cost of its real calls, across all runs. Each key holds $50;
# $48 leaves room for billing rounding and an estimate that runs high (decision log, Nov 27).
BUDGET_USD = 48.0

# One environment variable, spend log and $BUDGET_USD cap per API key (decision log, Nov 27).
# Key 1 is the original; key 2 keeps its own log, so each key's cap is checked separately.
KEYS = {"1": ("LLM_API_KEY", "runs/llm_spend.jsonl"),
        "2": ("LLM_API_KEY_2", "runs/llm_spend_key2.jsonl")}
SPEND_LOG = KEYS["1"][1]

# USD per token: (input, output), Gemini API paid tier, Standard, as listed on 2026-11-27.
# Thinking is billed as output. Cached input is charged at the full input rate (real.py says why).
PRICES = {
    "gemini-4-argon": (2e-6, 10e-6),
}
# Scheduled price changes: (first day, new (input, output)). Argon doubles on 2026-12-01.
PRICE_CHANGES = {
    "gemini-4-argon": [("2026-12-01", (4e-6, 20e-6))],
}
# An answer from a model not listed above is priced at the highest listed Gemini 4 rate after
# the December change ($4 / $20 per million tokens), so spend is never undercounted.
UNKNOWN_PRICE = (4e-6, 20e-6)
