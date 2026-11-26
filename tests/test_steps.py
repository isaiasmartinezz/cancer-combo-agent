import json

import pandas as pd

from ccagent.hypotheses import HypothesisBook
from ccagent.llm import FakeLLMClient
from ccagent.masking import Masker
from ccagent.steps import critique, explain, hypothesize, select

SHORT = pd.DataFrame({"exp_id": [f"E{i}" for i in range(1, 6)],
                      "drug_a": ["A", "A", "C", "C", "E"], "drug_b": ["B", "B", "D", "D", "F"],
                      "cell_line": ["L1", "L2", "L1", "L2", "L3"]})


def hyp(**over):
    h = {"statement": "A+B works", "drugs": ["A", "B"], "level": "mechanistic",
         "predicted_responders": ["L1"], "rationale": "r", "evidence_ids": ["PMID:1"]}
    return {**h, **over}


def answer(key, items):
    return json.dumps({key: items})


# ---------- hypothesize ----------

def test_valid_hypotheses_pass_first_time():
    fake = FakeLLMClient([answer("hypotheses", [hyp()])])
    items, rec = hypothesize(fake, "sys", "prompt", SHORT, {"PMID:1"}, Masker())
    assert rec.ok and len(rec.attempts) == 1 and items[0]["drugs"] == ["A", "B"]


def test_invented_citation_gets_one_retry_with_the_problem_listed():
    fake = FakeLLMClient([answer("hypotheses", [hyp(evidence_ids=["PMID:999"])]),
                          answer("hypotheses", [hyp()])])
    items, rec = hypothesize(fake, "sys", "prompt", SHORT, {"PMID:1"}, Masker())
    assert rec.ok and len(rec.attempts) == 2
    assert "PMID:999" in fake.calls[1][1]  # the retry prompt names the problem


def test_hypothesize_checks_drugs_lines_level_and_responders():
    bad = [hyp(drugs=["A", "Z"]), hyp(predicted_responders=["HELD"]), hyp(level="vibes"),
           hyp(predicted_responders=[]), hyp(drugs=["A", "B", "C"])]
    fake = FakeLLMClient([answer("hypotheses", bad)] * 2)
    items, rec = hypothesize(fake, "sys", "p", SHORT, {"PMID:1"}, Masker())
    assert items is None and not rec.ok
    problems = " ".join(rec.attempts[0]["problems"])
    for phrase in ("not on the shortlist: ['Z']", "cell lines not on the shortlist",
                   "level must be", "must name responder lines", "one or two drugs"):
        assert phrase in problems


def test_hypothesize_rejects_too_many_and_bad_json():
    fake = FakeLLMClient([answer("hypotheses", [hyp()] * 7), "not json"])
    items, rec = hypothesize(fake, "sys", "p", SHORT, {"PMID:1"}, Masker())
    assert items is None
    assert "between 1 and 6" in rec.attempts[0]["problems"][0]
    assert "not valid JSON" in rec.attempts[1]["problems"][0]


def test_masked_answers_are_unmasked():
    masker = Masker({"A": [], "B": [], "C": [], "D": [], "E": [], "F": []})
    shown = masker.table(SHORT)
    fake = FakeLLMClient([answer("hypotheses", [hyp(drugs=[masker.name("A"),
                                                            masker.name("B")])])])
    items, _ = hypothesize(fake, "sys", "p", shown, {"PMID:1"}, masker)
    assert items[0]["drugs"] == ["A", "B"]


# ---------- critique ----------

def test_critique_needs_one_verdict_per_hypothesis():
    book = HypothesisBook()
    judged = book.add([hyp(), hyp()], 1)
    ok = [{"id": "H1", "verdict": "keep", "reason": "x"},
          {"id": "H2", "verdict": "revise", "revised_statement": "narrower", "reason": "y"}]
    fake = FakeLLMClient([answer("verdicts", ok[:1]), answer("verdicts", ok)])
    items, rec = critique(fake, "sys", "p", judged, {"PMID:1"})
    assert rec.ok and len(rec.attempts) == 2 and [v["id"] for v in items] == ["H1", "H2"]


def test_critique_failing_twice_contributes_nothing():
    book = HypothesisBook()
    judged = book.add([hyp()], 1)
    bad = answer("verdicts", [{"id": "H1", "verdict": "revise", "reason": "x"}])
    items, rec = critique(FakeLLMClient([bad, bad]), "sys", "p", judged, set())
    assert items is None and "revised_statement" in rec.attempts[0]["problems"][0]


# ---------- select ----------

def pick(exp_id, hid="explore", level="statistical"):
    return {"exp_id": exp_id, "hypothesis_id": hid, "level": level, "reason": "r"}


def test_select_valid_picks():
    fake = FakeLLMClient([answer("picks", [pick("E3"), pick("E1")])])
    items, rec = select(fake, "sys", "p", SHORT, [], 2)
    assert rec.ok and [p["exp_id"] for p in items] == ["E3", "E1"]


def test_select_falls_back_to_the_shortlist_order_after_two_bad_answers():
    bad = answer("picks", [pick("E1"), pick("E1")])  # repeated
    worse = answer("picks", [pick("E1"), pick("E99")])  # not on the shortlist
    items, rec = select(FakeLLMClient([bad, worse]), "sys", "p", SHORT, [], 3)
    assert not rec.ok
    assert [p["exp_id"] for p in items] == ["E1", "E2", "E3"]  # BO's batch
    assert "repeated" in " ".join(rec.attempts[0]["problems"])
    assert "E99" in " ".join(rec.attempts[1]["problems"])


def test_select_rejects_unknown_hypothesis_ids():
    fake = FakeLLMClient([answer("picks", [pick("E1", "H7")]), answer("picks", [pick("E1")])])
    _, rec = select(fake, "sys", "p", SHORT, [], 1)
    assert "unknown hypothesis id 'H7'" in rec.attempts[0]["problems"][0]


# ---------- explanations ----------

def test_explanation_levels_are_enforced():
    book = HypothesisBook()
    book.add([hyp(), hyp(level="statistical", predicted_responders=[]),
              hyp(drugs=["C"], level="cross_cancer", predicted_responders=[])], 1)
    mech = explain(pick("E1", "H1", "mechanistic"), book, SHORT)
    assert mech["explanation_level"] == "mechanistic" and "L1" in mech["explanation_text"]
    no_resp = explain(pick("E1", "H2", "mechanistic"), book, SHORT)
    assert no_resp["explanation_level"] == "statistical"
    off_pair = explain(pick("E5", "H1", "mechanistic"), book, SHORT)
    assert off_pair["explanation_level"] == "statistical"
    assert "does not cover" in off_pair["explanation_text"]
    explore = explain(pick("E5", "explore", "mechanistic"), book, SHORT)
    assert explore["explanation_level"] == "statistical"
    cross = explain(pick("E3", "H3", "cross_cancer"), book, SHORT)
    assert cross["explanation_level"] == "cross_cancer" and cross["source_ids"] == "PMID:1"
