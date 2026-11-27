"""Every name a drug goes by, from PubChem, for the masked-drug-name ablation.

PubChem lists many synonyms per compound: brand names (Taxol), codes (NSC-125973), salt forms,
and registry numbers. Masking all of them keeps a brand name in an abstract from revealing which
drug a code stands for. A few kinds are dropped, each with a recorded reason:
  too short     fewer than MIN_LENGTH characters (would mask unrelated abbreviations)
  only digits   a bare number could match years or counts in an abstract
  shared        the same spelling is listed for two of our drugs, so it identifies neither
  contains a shorter name   e.g. "CHK1 inhibitor MK-8776": the shorter name is masked anyway,
                            and masking the whole phrase would also hide "CHK1 inhibitor"
"""
import re

import pandas as pd

MIN_LENGTH = 4
COLUMNS = ["drug_id", "synonym", "kept", "reason"]


def clean_synonyms(raw: dict[str, list[str]]) -> pd.DataFrame:
    rows = [(drug, s.strip()) for drug, names in raw.items() for s in dict.fromkeys(names)
            if s.strip()]
    table = pd.DataFrame(rows, columns=["drug_id", "synonym"])
    owners = table.groupby(table["synonym"].str.lower())["drug_id"].nunique()
    short_names = {}  # drug -> pattern matching any of its other kept-length names as a word
    for drug, group in table.groupby("drug_id"):
        names = sorted({n for n in group["synonym"] if len(n) >= MIN_LENGTH} | {drug}, key=len)
        # a lookahead, so matches may overlap: the whole phrase matching at its start must not
        # hide a shorter name later in it
        short_names[drug] = re.compile(r"(?<!\w)(?=(" + "|".join(map(re.escape, names)) +
                                       r")(?!\w))", re.IGNORECASE)
    reasons = []
    for row in table.itertuples():
        inner = [m.group(1) for m in short_names[row.drug_id].finditer(row.synonym)
                 if m.group(1).lower() != row.synonym.lower()]
        if len(row.synonym) < MIN_LENGTH:
            reasons.append("too short")
        elif re.fullmatch(r"[\d\s.,-]+", row.synonym):
            reasons.append("only digits")
        elif owners[row.synonym.lower()] > 1:
            reasons.append("shared")
        elif inner:
            reasons.append("contains a shorter name")
        else:
            reasons.append("")
    table["reason"] = reasons
    table["kept"] = table["reason"] == ""
    return table[COLUMNS].drop_duplicates(["drug_id", "synonym"]).reset_index(drop=True)


def synonyms_by_drug(table: pd.DataFrame) -> dict[str, list[str]]:
    kept = table[table["kept"].astype(bool)]
    return {d: sorted(g["synonym"]) for d, g in kept.groupby("drug_id")}
