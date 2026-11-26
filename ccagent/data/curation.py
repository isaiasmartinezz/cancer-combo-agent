"""Merge fetched annotations with the hand-curated table and the sign-off record.

data/reference/drug_curation.csv holds every manual choice, each with its source:
  drug_class + drug_class_source   plain-words class restating ATC, ChEMBL, or a cited paper
  lit_targets / lit_mechanism / lit_action_type / lit_sources
                                   only for drugs with no ChEMBL mechanism, from cited abstracts
data/reference/drug_signoff.csv records the human review of each finished row.
"""
import pandas as pd

LIT_FIELDS = {"lit_targets": "targets", "lit_mechanism": "mechanism",
              "lit_action_type": "action_type"}
SIGNOFF_STATUSES = {"pending", "ok", "fixed"}
FINAL_COLS = [
    "drug_id", "search_name", "pubchem_cid", "smiles", "chembl_id", "chembl_name",
    "targets", "target_genes", "mechanism", "action_type", "atc_codes", "drug_class",
    "sources", "status", "reviewed_by", "reviewed_on",
]


def build(fetched: pd.DataFrame, curation: pd.DataFrame, signoff: pd.DataFrame) -> pd.DataFrame:
    fetched, curation, signoff = (t.fillna("").astype(str) for t in (fetched, curation, signoff))
    for name, table in [("curation", curation), ("signoff", signoff)]:
        if set(table.drug_id) != set(fetched.drug_id):
            diff = set(table.drug_id) ^ set(fetched.drug_id)
            raise ValueError(f"{name} drugs differ from fetched drugs: {sorted(diff)}")
    bad = set(signoff.status) - SIGNOFF_STATUSES
    if bad:
        raise ValueError(f"unknown sign-off status: {sorted(bad)}")

    out = fetched.merge(curation, on="drug_id").merge(signoff, on="drug_id")
    sources = []
    for i, row in out.iterrows():
        src = ["structure: PubChem", f"targets: ChEMBL {row.chembl_id}" if row.chembl_id else ""]
        if row.lit_sources:
            if row.mechanism:
                raise ValueError(f"{row.drug_id}: literature override but ChEMBL has a mechanism")
            for lit_col, col in LIT_FIELDS.items():
                out.at[i, col] = row[lit_col]
            src[1] = f"targets/mechanism: {row.lit_sources}"
        src.append(f"class: {row.drug_class_source}")
        sources.append("; ".join(s for s in src if s))
    out["sources"] = sources
    out["atc_codes"] = out["atc_codes"].replace("", "none")
    out = out[FINAL_COLS].replace("", "unknown")
    for col in ("reviewed_by", "reviewed_on"):
        out.loc[out.status == "pending", col] = ""
    return out
