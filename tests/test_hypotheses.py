import pandas as pd
import pytest

from ccagent.hypotheses import (
    MAX_CRITIQUE,
    MAX_NEW,
    MAX_SHOWN,
    MIN_TESTS,
    Hypothesis,
    HypothesisBook,
)


def proposal(drugs, level="mechanistic", responders=(), statement=None):
    return {"statement": statement or f"{'+'.join(drugs)} works", "drugs": list(drugs),
            "level": level, "predicted_responders": list(responders), "rationale": "",
            "evidence_ids": []}


def history(rows):
    """rows: (drug_a, drug_b, cell_line, target value)."""
    return pd.DataFrame(rows, columns=["drug_a", "drug_b", "cell_line", "joint"])


def shortlist(pairs):
    return pd.DataFrame([{"drug_a": a, "drug_b": b, "cell_line": "L1"} for a, b in pairs])


# ---------- coverage and scoring ----------

def test_covers_one_drug_and_pair():
    one = Hypothesis("H1", "", ("A",), "statistical")
    pair = Hypothesis("H2", "", ("A", "B"), "statistical")
    assert one.covers("A", "C") and one.covers("C", "A") and not one.covers("B", "C")
    assert pair.covers("B", "A") and not pair.covers("A", "C")


def test_rescore_statuses_from_percentiles():
    book = HypothesisBook()
    good, bad, thin = book.add([proposal(["A", "B"]), proposal(["C", "D"]),
                                proposal(["E", "F"])], round_num=1)
    lines = ["L1", "L2", "L3", "L4"]
    hist = history([("A", "B", line, v) for line, v in zip(lines, (12, 11, 10, 9), strict=True)]
                   + [("C", "D", line, v) for line, v in zip(lines, (1, 2, 3, 4), strict=True)]
                   + [("E", "F", line, v) for line, v in zip(lines[:3], (5, 6, 7), strict=True)])
    book.rescore(hist, "joint", round_num=2)
    # A+B ranks 8-11 of the 11 results; C+D ranks 1-4
    assert (good.status, good.n_tests, good.score) == ("supported", MIN_TESTS, 0.864)
    assert (bad.status, bad.score) == ("refuted", 0.227)
    assert (thin.status, thin.n_tests) == ("open", 3)  # three tests are not enough


def test_status_needs_four_tests():
    assert MIN_TESTS == 4  # decision log, Nov 27: 2 tests label 1 in 4 no-effect ideas


def test_responder_gap_points_the_right_way():
    book = HypothesisBook()
    (h,) = book.add([proposal(["A", "B"], responders=["L1"])], 1)
    book.rescore(history([("A", "B", "L1", 9), ("A", "B", "L2", 1), ("A", "B", "L3", 2)]),
                 "joint", 2)
    assert h.responder_gap > 0


def test_rescore_reads_only_the_target_column():
    """The book never needs hidden data: a history with only returned columns is enough."""
    book = HypothesisBook()
    book.add([proposal(["A", "B"])], 1)
    hist = history([("A", "B", "L1", 1.0), ("A", "B", "L2", 2.0)])
    book.rescore(hist, "joint", 2)  # no strong flags, no cutoffs anywhere in the input
    with pytest.raises(KeyError):
        book.rescore(hist, "is_strong_joint", 2)


def test_dropped_hypotheses_stop_being_scored_or_shown():
    book = HypothesisBook()
    (h,) = book.add([proposal(["A", "B"])], 1)
    book.apply_critique([{"id": "H1", "verdict": "drop", "revised_statement": "",
                          "reason": "no support"}], 1)
    book.rescore(history([("A", "B", "L1", 9), ("A", "B", "L2", 8)]), "joint", 2)
    assert h.status == "dropped" and h.n_tests == 0
    assert book.shown(shortlist([("A", "B")])) == []
    assert book.retired() == ["H1 (dropped): A+B works"]


def test_revise_changes_statement_only():
    book = HypothesisBook()
    (h,) = book.add([proposal(["A", "B"])], 1)
    book.apply_critique([{"id": "H1", "verdict": "revise", "revised_statement": "narrower",
                          "reason": ""}], 1)
    assert (h.statement, h.drugs, h.status) == ("narrower", ("A", "B"), "open")
    with pytest.raises(ValueError):
        book.apply_critique([{"id": "H1", "verdict": "maybe"}], 1)


# ---------- the size limits ----------

def test_ids_are_assigned_by_the_book_and_new_ones_are_capped():
    book = HypothesisBook()
    accepted = book.add([proposal([f"D{i}"]) for i in range(MAX_NEW + 3)], 1)
    assert [h.id for h in accepted] == [f"H{i}" for i in range(1, MAX_NEW + 1)]
    assert sum(e["event"].startswith("rejected") for e in book.events) == 3
    assert book.add([proposal(["Z"])], 2)[0].id == f"H{MAX_NEW + 1}"


def test_shown_is_capped_relevant_and_ordered():
    book = HypothesisBook()
    for r in range(5):  # 30 hypotheses about A+B, 6 about an off-shortlist pair
        book.add([proposal(["A", "B"]) for _ in range(MAX_NEW)], r)
    book.add([proposal(["X", "Y"]) for _ in range(MAX_NEW)], 5)
    book.items["H3"].status, book.items["H3"].score = "supported", 0.7
    book.items["H4"].status = "refuted"
    shown = book.shown(shortlist([("A", "B")]))
    assert len(shown) == MAX_SHOWN
    assert shown[0].id == "H3"  # supported first
    assert all(set(h.drugs) == {"A", "B"} for h in shown)
    assert "H4" not in {h.id for h in shown}


def test_critique_sees_new_then_changed_and_never_rejudges_unchanged():
    book = HypothesisBook()
    first = book.add([proposal(["A", "B"]), proposal(["C", "D"])], 1)
    assert [h.id for h in book.to_critique(first)] == ["H1", "H2"]
    assert book.to_critique([]) == []  # nothing new, nothing tested: nothing to judge
    book.rescore(history([("A", "B", "L1", 9), ("X", "Y", "L1", 1)]), "joint", 2)
    new = book.add([proposal(["E", "F"])], 2)
    assert [h.id for h in book.to_critique(new)] == ["H3", "H1"]  # H1 has a new result
    assert book.to_critique([]) == []  # H1 already judged at this n_tests


def test_critique_is_capped_and_always_fits_every_new_hypothesis():
    book = HypothesisBook()
    old = book.add([proposal(["A", "B"]) for _ in range(MAX_NEW)], 1)
    book.to_critique(old)
    book.rescore(history([("A", "B", "L1", 9)]), "joint", 2)
    new = book.add([proposal(["C", "D"]) for _ in range(MAX_NEW)], 2)
    picked = book.to_critique(new)
    assert len(picked) == MAX_CRITIQUE
    assert {h.id for h in new} <= {h.id for h in picked}


def test_failed_critique_leaves_hypotheses_uncitable_and_first_in_line():
    book = HypothesisBook()
    judged = book.to_critique(book.add([proposal(["A", "B"])], 1))
    book.uncritique(judged, 1)  # the critique answer failed twice
    short = shortlist([("A", "B")])
    assert book.options(short, judged, critiqued_only=True) == []
    assert [h.id for h in book.options(short, judged)] == ["H1"]  # no_critique arm: citable
    new = book.add([proposal(["C", "D"])], 2)
    assert [h.id for h in book.to_critique(new)] == ["H2", "H1"]  # queued behind new ones
    assert {h.id for h in book.options(short, new, critiqued_only=True)} == {"H1", "H2"}


def test_options_add_new_hypotheses_to_those_shown():
    book = HypothesisBook()
    new = book.add([proposal(["A", "B"]), proposal(["C", "D"])], 1)
    book.apply_critique([{"id": "H2", "verdict": "drop", "revised_statement": "",
                          "reason": ""}], 1)
    assert [h.id for h in book.options(shortlist([("Q", "R")]), new)] == ["H1"]


def test_prompt_slice_stays_flat_over_a_full_run():
    """The iteration 1 cost problem: 25 rounds of new ideas must not grow what the model sees."""
    book, rows = HypothesisBook(), []
    sizes = []
    for r in range(1, 25):
        new = book.add([proposal(["A", "B"]) for _ in range(MAX_NEW)], r)
        rows += [("A", "B", f"L{r}", r % 7), ("C", "D", f"L{r}", 3)]
        book.rescore(history(rows), "joint", r)
        sizes.append((len(book.shown(shortlist([("A", "B")]))), len(book.to_critique(new)),
                      len(book.retired())))
    assert len(book.items) == 24 * MAX_NEW
    assert max(s[0] for s in sizes) <= MAX_SHOWN
    assert max(s[1] for s in sizes) <= MAX_CRITIQUE
    assert max(s[2] for s in sizes) <= 6
