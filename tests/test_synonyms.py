import pandas as pd

from ccagent.data.synonyms import clean_synonyms, synonyms_by_drug
from ccagent.masking import Masker


def test_cleaning_rules():
    table = clean_synonyms({
        "ERLOTINIB": ["erlotinib", "Tarceva", "OSI", "176870", "Erlotinib (Tarceva)",
                      "EGFR inhibitor erlotinib", "Shared-name"],
        "LAPATINIB": ["lapatinib", "Tykerb", "Shared-name"],
    })
    reason = dict(zip(table["synonym"], table["reason"], strict=False))
    assert reason["Tarceva"] == "" and reason["erlotinib"] == ""
    assert reason["OSI"] == "too short"
    assert reason["176870"] == "only digits"
    assert reason["Shared-name"] == "shared"
    assert reason["Erlotinib (Tarceva)"] == "contains a shorter name"
    assert reason["EGFR inhibitor erlotinib"] == "contains a shorter name"
    assert synonyms_by_drug(table)["ERLOTINIB"] == ["Tarceva", "erlotinib"]


def test_brand_names_are_masked_and_mechanism_words_kept():
    drugs = pd.DataFrame({"drug_id": ["ERLOTINIB"], "drugcomb_name": ["Erlotinib"],
                          "generic_name": ["erlotinib"]})
    ann = pd.DataFrame({"drug_id": ["ERLOTINIB"], "search_name": ["erlotinib"],
                        "chembl_name": ["ERLOTINIB"]})
    masker = Masker.from_tables(drugs, ann, {"ERLOTINIB": ["Tarceva", "CP-358774"]})
    code = masker.name("ERLOTINIB")
    out = masker.text("EGFR inhibitor erlotinib (Tarceva, CP-358774) was given.")
    assert out == f"EGFR inhibitor {code} ({code}, {code}) was given."


def test_shipped_synonyms_cover_known_brand_names():
    table = pd.read_csv("data/reference/drug_synonyms.csv")
    kept = synonyms_by_drug(table)
    lowered = {d: {n.lower() for n in names} for d, names in kept.items()}
    for drug, brand in [("ERLOTINIB", "tarceva"), ("PACLITAXEL", "taxol"),
                        ("DOXORUBICIN", "adriamycin"), ("BORTEZOMIB", "velcade")]:
        assert brand in lowered[drug]
