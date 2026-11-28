"""The model steps (hypothesize, critique, select, and the held-out prediction after the run):
call, check the answer, retry once.

The model must answer in JSON. Code then checks what a format cannot: cited IDs were actually
shown, drugs and cell lines are on the shortlist, mechanistic hypotheses name responder lines,
critique gives one verdict per hypothesis, and select returns exactly k distinct shortlist IDs.
A failed answer gets one retry with its problems listed. If that fails too, hypothesize and
critique contribute nothing this round, and select falls back to the shortlist's first k rows,
which is exactly the batch Bayesian optimization would run.

Checks run on names as the model saw them (masked if the ablation is on); answers are then
unmasked, so everything stored uses real drug IDs.
"""
import json
from dataclasses import dataclass, field

import pandas as pd

from ccagent.hypotheses import LEVELS, MAX_NEW, Hypothesis, HypothesisBook
from ccagent.llm.client import LLMClient, LLMResponse
from ccagent.masking import Masker

VERDICTS = ("keep", "revise", "drop")
EXPLORE = "explore"


def _obj(properties: dict) -> dict:
    return {"type": "object", "properties": properties, "required": list(properties),
            "additionalProperties": False}


_STR, _STRS = {"type": "string"}, {"type": "array", "items": {"type": "string"}}
# The shapes the three answers must take. Code still checks everything a schema cannot.
SCHEMAS = {
    "hypothesize": _obj({"hypotheses": {"type": "array", "items": _obj({
        "statement": _STR, "drugs": _STRS, "level": {"type": "string", "enum": list(LEVELS)},
        "predicted_responders": _STRS, "rationale": _STR, "evidence_ids": _STRS})}}),
    "critique": _obj({"verdicts": {"type": "array", "items": _obj({
        "id": _STR, "verdict": {"type": "string", "enum": list(VERDICTS)},
        "revised_statement": _STR, "reason": _STR, "evidence_ids": _STRS})}}),
    "select": _obj({"picks": {"type": "array", "items": _obj({
        "exp_id": _STR, "hypothesis_id": _STR,
        "level": {"type": "string", "enum": list(LEVELS)}, "reason": _STR})}}),
    "predict_heldout": _obj({"predictions": {"type": "array", "items": _obj({
        "hypothesis_id": _STR, "predicted_responders": _STRS, "reason": _STR})}}),
}


@dataclass
class StepRecord:
    """One step's calls, for the run's model-call log."""
    step: str
    attempts: list[dict] = field(default_factory=list)
    ok: bool = False


def _parse(text: str, key: str) -> tuple[list | None, list[str]]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        return None, [f"the answer is not valid JSON ({e.msg})"]
    if not isinstance(data, dict) or not isinstance(data.get(key), list):
        return None, [f'the answer must be a JSON object with a list under "{key}"']
    if not all(isinstance(item, dict) for item in data[key]):
        return None, [f'every item under "{key}" must be a JSON object']
    return data[key], []


def _missing(item: dict, fields: dict[str, type], label: str) -> list[str]:
    return [f"{label}: {name!r} must be a {kind.__name__}" for name, kind in fields.items()
            if not isinstance(item.get(name), kind)]


def run_step(client: LLMClient, system: str, prompt: str, step: str, key: str,
             check) -> tuple[list | None, StepRecord]:
    """The answer format is also sent as a JSON schema, which a real model is held to."""
    # check(items) -> problems. Returns (items, record), with items None after two failures.
    record, feedback = StepRecord(step), ""
    for attempt in (1, 2):
        content = prompt + (f"\n\nYour previous answer had these problems; fix them:\n{feedback}"
                            if feedback else "")
        response: LLMResponse = client.complete(system, content, SCHEMAS[step])
        items, problems = _parse(response.text, key)
        if items is not None:
            problems = check(items)
        record.attempts.append({
            "attempt": attempt, "prompt": content, "response": response.text,
            "problems": problems, "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens, "cost_usd": response.cost_usd,
            "cache_read_tokens": response.cache_read_tokens,
            "cache_write_tokens": response.cache_write_tokens,
            "model": response.model, "stop_reason": response.stop_reason})
        if not problems:
            record.ok = True
            return items, record
        feedback = "\n".join(f"- {p}" for p in problems)
    return None, record


# ---------- the steps ----------

def hypothesize(client, system, prompt, shown_shortlist: pd.DataFrame, source_ids: set[str],
                masker: Masker, max_new: int = MAX_NEW):
    drugs = set(shown_shortlist["drug_a"]) | set(shown_shortlist["drug_b"])
    lines = set(shown_shortlist["cell_line"])
    fields = {"statement": str, "drugs": list, "level": str, "predicted_responders": list,
              "rationale": str, "evidence_ids": list}

    def check(items):
        problems = [] if 1 <= len(items) <= max_new else \
            [f"give between 1 and {max_new} hypotheses, not {len(items)}"]
        for i, h in enumerate(items, 1):
            label = f"hypothesis {i}"
            if bad := _missing(h, fields, label):
                problems += bad
                continue
            if h["level"] not in LEVELS:
                problems.append(f"{label}: level must be one of {list(LEVELS)}")
            if not 1 <= len(h["drugs"]) <= 2:
                problems.append(f"{label}: name one or two drugs")
            if bad := set(h["drugs"]) - drugs:
                problems.append(f"{label}: drugs not on the shortlist: {sorted(bad)}")
            if bad := set(h["predicted_responders"]) - lines:
                problems.append(f"{label}: cell lines not on the shortlist: {sorted(bad)}")
            if h["level"] == "mechanistic" and not h["predicted_responders"]:
                problems.append(f"{label}: a mechanistic hypothesis must name responder lines")
            if bad := set(h["evidence_ids"]) - source_ids:
                problems.append(f"{label}: cites sources that were not shown: {sorted(bad)}")
        return problems

    items, record = run_step(client, system, prompt, "hypothesize", "hypotheses", check)
    if items is not None:
        for h in items:
            h["drugs"] = [masker.unname(d) for d in h["drugs"]]
    return items, record


def critique(client, system, prompt, to_judge: list[Hypothesis], source_ids: set[str]):
    want = sorted(h.id for h in to_judge)

    def check(items):
        problems = []
        got = sorted(str(v.get("id")) for v in items)
        if got != want:
            problems.append(f"give exactly one verdict for each of {want}; got {got}")
        for v in items:
            label = f"verdict for {v.get('id')}"
            if bad := _missing(v, {"id": str, "verdict": str, "reason": str}, label):
                problems += bad
                continue
            if v["verdict"] not in VERDICTS:
                problems.append(f"{label}: verdict must be one of {list(VERDICTS)}")
            if v["verdict"] == "revise" and not str(v.get("revised_statement", "")).strip():
                problems.append(f"{label}: 'revise' needs a revised_statement")
            if bad := set(v.get("evidence_ids") or []) - source_ids:
                problems.append(f"{label}: cites sources that were not shown: {sorted(bad)}")
        return problems

    return run_step(client, system, prompt, "critique", "verdicts", check)


def select(client, system, prompt, shortlist: pd.DataFrame, options: list[Hypothesis], k: int):
    """Always returns k picks: the model's if valid, else the shortlist's first k (BO's batch)."""
    valid, hyp_ids = set(shortlist["exp_id"]), {h.id for h in options} | {EXPLORE}

    def check(items):
        problems = []
        ids = [str(p.get("exp_id")) for p in items]
        if len(ids) != k:
            problems.append(f"choose exactly {k} experiments, not {len(ids)}")
        if len(set(ids)) != len(ids):
            problems.append("an exp_id is repeated")
        if bad := set(ids) - valid:
            problems.append(f"exp_ids not on the shortlist: {sorted(bad)}")
        for p in items:
            label = f"pick {p.get('exp_id')}"
            problems += _missing(p, {"hypothesis_id": str, "level": str, "reason": str}, label)
            if p.get("hypothesis_id") not in hyp_ids:
                problems.append(f"{label}: unknown hypothesis id {p.get('hypothesis_id')!r}")
            if p.get("level") not in LEVELS:
                problems.append(f"{label}: level must be one of {list(LEVELS)}")
        return problems

    items, record = run_step(client, system, prompt, "select", "picks", check)
    if items is None:
        items = [{"exp_id": e, "hypothesis_id": EXPLORE, "level": "statistical",
                  "reason": "model answer invalid twice; fell back to the response model's "
                            "ranking"} for e in shortlist["exp_id"].iloc[:k]]
    return items, record


def predict_heldout(client, system, prompt, hypotheses: list[Hypothesis], lines: set[str]):
    """One prediction per hypothesis, naming only held-out lines. None after two failures (the
    run then has no held-out test; its picks are unaffected)."""
    want = sorted(h.id for h in hypotheses)

    def check(items):
        problems = []
        got = sorted(str(p.get("hypothesis_id")) for p in items)
        if got != want:
            problems.append(f"give exactly one prediction for each of {want}; got {got}")
        for p in items:
            label = f"prediction for {p.get('hypothesis_id')}"
            if bad := _missing(p, {"predicted_responders": list, "reason": str}, label):
                problems += bad
                continue
            if bad := set(p["predicted_responders"]) - lines:
                problems.append(f"{label}: not among the untested lines: {sorted(bad)}")
        return problems

    return run_step(client, system, prompt, "predict_heldout", "predictions", check)


# ---------- the explanation logged with each pick ----------

def explain(pick: dict, book: HypothesisBook, shortlist: pd.DataFrame) -> dict:
    """Run-log fields for one pick. A pick citing a hypothesis that does not cover its drug pair
    is logged as statistical, and so is a "mechanistic" pick whose hypothesis names no responder
    lines, so every mechanistic explanation carries a testable prediction about this pair."""
    row = shortlist.set_index("exp_id").loc[pick["exp_id"]]
    h = book.items.get(pick["hypothesis_id"])
    level, text = pick["level"], pick["reason"]
    if h is not None:
        text = f"{h.statement} | Why this experiment: {pick['reason']}"
        if not h.covers(row["drug_a"], row["drug_b"]):
            text += " | (downgraded: the cited hypothesis does not cover this drug pair)"
            level = "statistical"
        elif level == "mechanistic" and h.predicted_responders:
            text += f" | Predicted responders: {', '.join(h.predicted_responders)}"
        elif level == "mechanistic":
            level = "statistical"
            text += " | (downgraded from mechanistic: no predicted responder lines)"
    elif level == "mechanistic":
        level = "statistical"  # an explore pick has no hypothesis to carry a prediction
    return {"hypothesis_id": pick["hypothesis_id"], "explanation_level": level,
            "explanation_text": text,
            "source_ids": "; ".join(h.evidence_ids) if h else ""}
