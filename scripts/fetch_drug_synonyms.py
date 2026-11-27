"""Fetch every listed name of the 38 drugs, for the masked-drug-name ablation.

Two sources, both needing the internet (free APIs, no model calls):
  PubChem   all synonyms of the compound's CID (codes, salts, many brand names)
  ChEMBL    synonyms of the parent molecule and every salt form ChEMBL links to it. Needed
            because a brand name is often filed under the salt: Tarceva is erlotinib
            hydrochloride, which PubChem keeps under a different CID from erlotinib.
Output: data/reference/drug_synonyms.csv (committed, so masking is reproducible offline), with
each name's source and why it was kept or dropped, plus a .meta.json with the fetch date.
Usage: python -m scripts.fetch_drug_synonyms
"""
import json
from datetime import date
from pathlib import Path

import pandas as pd

from ccagent.data.synonyms import clean_synonyms
from scripts.fetch_drug_annotations import get_json

OUT = Path("data/reference/drug_synonyms.csv")
PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/synonyms/JSON"
CHEMBL = "https://www.ebi.ac.uk/chembl/api/data"


def pubchem_synonyms(cid: str) -> list[str]:
    data = get_json(PUBCHEM.format(cid=cid), None) or {}
    return data.get("InformationList", {}).get("Information", [{}])[0].get("Synonym", [])


def chembl_synonyms(chembl_id: str) -> list[str]:
    forms = get_json(f"{CHEMBL}/molecule_form.json", {"parent_chembl_id": chembl_id}) or {}
    ids = {chembl_id} | {f["molecule_chembl_id"] for f in forms.get("molecule_forms", [])}
    names = []
    for mid in sorted(ids):
        mol = get_json(f"{CHEMBL}/molecule/{mid}.json", None) or {}
        names += [mol.get("pref_name") or ""]
        names += [s["molecule_synonym"] for s in mol.get("molecule_synonyms") or []]
    return [n for n in names if n]


def main() -> None:
    drugs = pd.read_csv("data/reference/drugs.csv", dtype=str)
    chembl = pd.read_csv("data/reference/drug_annotations.csv", dtype=str) \
        .set_index("drug_id")["chembl_id"].dropna().to_dict()
    raw, source = {}, {}
    for d in drugs.itertuples():
        pubchem = pubchem_synonyms(d.pubchem_cid)
        from_chembl = chembl_synonyms(chembl[d.drug_id]) if d.drug_id in chembl else []
        raw[d.drug_id] = pubchem + from_chembl
        for name in pubchem:
            source.setdefault((d.drug_id, name.strip()), f"PubChem CID {d.pubchem_cid}")
        for name in from_chembl:
            source.setdefault((d.drug_id, name.strip()), f"ChEMBL {chembl[d.drug_id]} + forms")
        print(f"{d.drug_id:18} PubChem {len(pubchem):4}  ChEMBL {len(from_chembl):3}", flush=True)
    table = clean_synonyms(raw)
    table.insert(2, "source", [source[(r.drug_id, r.synonym)] for r in table.itertuples()])
    table.to_csv(OUT, index=False)
    meta = {"fetched_on": date.today().isoformat(),
            "sources": ["PubChem PUG REST synonyms", "ChEMBL molecule + molecule_form"],
            "kept": int(table["kept"].sum()), "dropped": int((~table["kept"]).sum())}
    OUT.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"-> {OUT}: {meta}")


if __name__ == "__main__":
    main()
