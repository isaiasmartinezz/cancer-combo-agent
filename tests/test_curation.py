import pandas as pd
import pytest

from ccagent.data.annotations import FETCHED_COLS
from ccagent.data.curation import FINAL_COLS, build


def fetched(**rows):
    base = dict.fromkeys(FETCHED_COLS, "")
    return pd.DataFrame([{**base, "drug_id": d, **v} for d, v in rows.items()])


def curation(**lit):
    rows = [{"drug_id": d, "drug_class": "class", "drug_class_source": "ChEMBL",
             "lit_targets": "", "lit_mechanism": "", "lit_action_type": "", "lit_sources": ""}
            for d in ["A", "B"]]
    for row in rows:
        row.update(lit.get(row["drug_id"], {}))
    return pd.DataFrame(rows)


def signoff(status="pending"):
    return pd.DataFrame({"drug_id": ["A", "B"], "status": status, "reviewed_by": "IM",
                         "reviewed_on": "2026-11-26", "comment": ""})


FETCHED = fetched(A={"chembl_id": "C1", "mechanism": "m", "targets": "t"}, B={"chembl_id": "C2"})
LIT = {"B": {"lit_targets": "Hsp90", "lit_mechanism": "binds Hsp90",
             "lit_action_type": "INHIBITOR", "lit_sources": "PMID 1"}}


def test_literature_fills_only_drugs_without_chembl_mechanism():
    out = build(FETCHED, curation(**LIT), signoff()).set_index("drug_id")
    assert list(out.reset_index().columns) == FINAL_COLS
    assert out.loc["B", "targets"] == "Hsp90" and "PMID 1" in out.loc["B", "sources"]
    assert out.loc["A", "targets"] == "t" and "ChEMBL C1" in out.loc["A", "sources"]


def test_literature_never_overrides_a_chembl_mechanism():
    with pytest.raises(ValueError):
        build(FETCHED, curation(A=LIT["B"]), signoff())


def test_blanks_become_unknown_and_missing_atc_is_none():
    out = build(FETCHED, curation(**LIT), signoff()).set_index("drug_id")
    assert out.loc["A", "smiles"] == "unknown" and out.loc["A", "atc_codes"] == "none"


def test_pending_rows_carry_no_reviewer():
    out = build(FETCHED, curation(**LIT), signoff("pending"))
    assert (out.reviewed_by == "").all()
    out = build(FETCHED, curation(**LIT), signoff("ok"))
    assert (out.reviewed_by == "IM").all()


def test_tables_must_cover_the_same_drugs_and_statuses_must_be_known():
    with pytest.raises(ValueError):
        build(FETCHED, curation(**LIT).iloc[:1], signoff())
    with pytest.raises(ValueError):
        build(FETCHED, curation(**LIT), signoff("looks fine"))
