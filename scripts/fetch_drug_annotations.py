"""Fetch drug structures (PubChem) and targets/mechanisms (ChEMBL) for the 38 drugs.

Needs the internet; run once on a machine that can reach both APIs. Output:
data/reference/drug_annotations_fetched.csv (committed, so the build is reproducible offline)
and a .meta.json with the fetch date (ChEMBL releases change over time).
Usage: python -m scripts.fetch_drug_annotations
"""
import json
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from ccagent.data.annotations import FETCHED_COLS, annotate

REFERENCE = Path("data/reference")
OUT = REFERENCE / "drug_annotations_fetched.csv"


def get_json(url: str, params: dict | None) -> dict | None:
    for attempt in range(3):
        try:
            response = requests.get(url, params=params, timeout=30)
        except requests.RequestException:
            time.sleep(2 * (attempt + 1))
            continue
        time.sleep(0.25)  # stay well under both APIs' rate limits
        if response.ok:
            return response.json()
        if response.status_code == 404:
            return None
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"request kept failing: {url} {params}")


def main() -> None:
    drugs = pd.read_csv(REFERENCE / "drugs.csv", dtype=str)
    rows = []
    for d in drugs.itertuples():
        row = annotate(get_json, d.drug_id, d.generic_name, d.pubchem_cid)
        rows.append(row)
        print(f"{d.drug_id:18} {row['match'] or '-':12} {row['fetch_notes']}")
    pd.DataFrame(rows, columns=FETCHED_COLS).to_csv(OUT, index=False)
    meta = {"fetched_on": date.today().isoformat(), "sources": ["PubChem PUG REST", "ChEMBL API"]}
    OUT.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
