"""Famous vs obscure drugs: does literature pull the agent toward well-studied drugs?

A drug's familiarity is the number of PubMed papers naming it in the title or abstract (its
dataset ID or generic name: the same names the literature searches use). A drug is famous with
at least FAMOUS_MIN papers. The cutoff was fixed on Nov 27, before any real run, and matches the
iteration-1 pilot so its 63% vs 36% figure stays comparable. Counts are continuous, so 1,000 is
a round number rather than a natural gap; the report also uses the raw counts, so a conclusion
never rests on a drug sitting near the line.

Counts come from scripts/fetch_drug_familiarity.py into data/reference/drug_familiarity.csv.
"""
from pathlib import Path

import pandas as pd

FAMILIARITY = Path("data/reference/drug_familiarity.csv")
FAMOUS_MIN = 1000
GROUPS = ("famous", "mixed", "obscure")


def load(path: Path = FAMILIARITY) -> dict[str, int]:
    table = pd.read_csv(path)
    return dict(zip(table["drug_id"], table["pubmed_count"].astype(int), strict=True))


def famous(counts: dict[str, int], cutoff: int = FAMOUS_MIN) -> set[str]:
    return {drug for drug, n in counts.items() if n >= cutoff}


def pair_group(drug_a: str, drug_b: str, famous_drugs: set[str]) -> str:
    """famous: both drugs famous; obscure: neither; mixed: one of each."""
    n = (drug_a in famous_drugs) + (drug_b in famous_drugs)
    return GROUPS[2 - n]


def famous_share(drug_lists, famous_drugs: set[str]) -> float:
    """Share of drug mentions that are famous drugs, over e.g. every hypothesis's drug list."""
    mentions = [d for drugs in drug_lists for d in drugs]
    return sum(d in famous_drugs for d in mentions) / len(mentions) if mentions else float("nan")
