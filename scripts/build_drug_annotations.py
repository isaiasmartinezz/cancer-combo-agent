"""Build data/reference/drug_annotations.csv from the fetched table, curation, and sign-off.
Offline. Usage: python -m scripts.build_drug_annotations
"""
from pathlib import Path

import pandas as pd

from ccagent.data.curation import build

REFERENCE = Path("data/reference")


def main() -> None:
    table = build(
        pd.read_csv(REFERENCE / "drug_annotations_fetched.csv", dtype=str),
        pd.read_csv(REFERENCE / "drug_curation.csv", dtype=str),
        pd.read_csv(REFERENCE / "drug_signoff.csv", dtype=str),
    )
    table.to_csv(REFERENCE / "drug_annotations.csv", index=False)
    print(table.status.value_counts().to_string())
    unknown = (table == "unknown").sum()
    print("unknown cells:", unknown[unknown > 0].to_dict())


if __name__ == "__main__":
    main()
