"""Look up each drug's structure (PubChem) and targets and mechanism (ChEMBL).

Matching rules (from iteration 1, where free-text search matched MRK-003 to a different
compound): exact ChEMBL preferred name, then exact synonym, then the ChEMBL ID that PubChem
lists for the same compound. Otherwise the drug is "not found"; nothing is guessed.
Mechanisms are looked up by parent compound, because ChEMBL often files them under a salt form.

All network access goes through a `get(url, params) -> dict | None` function passed in, so the
logic is tested offline against fake responses.
"""
from collections.abc import Callable
from urllib.parse import quote

PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
CHEMBL = "https://www.ebi.ac.uk/chembl/api/data"

Getter = Callable[[str, dict | None], dict | None]

FETCHED_COLS = [
    "drug_id", "search_name", "pubchem_cid", "smiles", "chembl_id", "chembl_name", "match",
    "targets", "target_chembl_ids", "target_genes", "mechanism", "action_type", "atc_codes",
    "fetch_notes",
]


def _join(values) -> str:
    return "; ".join(dict.fromkeys(v for v in values if v))


def pubchem_structure(get: Getter, name: str) -> tuple[str, str]:
    """(CID, SMILES) for a compound name, or ("", "")."""
    for prop in ("SMILES", "CanonicalSMILES"):
        data = get(f"{PUBCHEM}/compound/name/{quote(name)}/property/{prop}/JSON", None)
        if data:
            row = data["PropertyTable"]["Properties"][0]
            smiles = next((v for k, v in row.items() if "SMILES" in k), "")
            return str(row.get("CID", "")), smiles
    return "", ""


def pubchem_chembl_xref(get: Getter, cid: str) -> str:
    if not cid:
        return ""
    data = get(f"{PUBCHEM}/compound/cid/{cid}/synonyms/JSON", None)
    if not data:
        return ""
    synonyms = data["InformationList"]["Information"][0].get("Synonym", [])
    return next((s for s in synonyms if s.startswith("CHEMBL") and s[6:].isdigit()), "")


def find_molecule(get: Getter, name: str, cid: str) -> tuple[dict | None, str]:
    for params in (
        {"pref_name__iexact": name},
        {"molecule_synonyms__molecule_synonym__iexact": name},
    ):
        data = get(f"{CHEMBL}/molecule.json", {**params, "limit": 5})
        if data and data.get("molecules"):
            return data["molecules"][0], "name" if "pref_name__iexact" in params else "synonym"
    xref = pubchem_chembl_xref(get, cid)
    if xref:
        mol = get(f"{CHEMBL}/molecule/{xref}.json", None)
        if mol:
            return mol, "pubchem_xref"
    return None, ""


def target_details(get: Getter, target_id: str) -> tuple[str, set[str]]:
    """(preferred name, gene symbols). First gene symbol per component is the primary one."""
    target = get(f"{CHEMBL}/target/{target_id}.json", None) or {}
    genes = set()
    for component in target.get("target_components", []):
        for synonym in component.get("target_component_synonyms", []):
            if synonym.get("syn_type") == "GENE_SYMBOL":
                genes.add(synonym["component_synonym"].upper())
                break
    return target.get("pref_name") or target_id, genes


def annotate(get: Getter, drug_id: str, search_name: str, expected_cid: str = "") -> dict:
    row = dict.fromkeys(FETCHED_COLS, "")
    row.update(drug_id=drug_id, search_name=search_name)
    notes = []
    row["pubchem_cid"], row["smiles"] = pubchem_structure(get, search_name)
    if not row["smiles"]:
        notes.append("no PubChem structure")
    if expected_cid and row["pubchem_cid"] and row["pubchem_cid"] != str(expected_cid):
        notes.append(f"PubChem CID {row['pubchem_cid']} differs from reference {expected_cid}")

    mol, match = find_molecule(get, search_name, row["pubchem_cid"])
    if mol is None:
        row["fetch_notes"] = "; ".join(notes + ["not found in ChEMBL"])
        return row
    parent = (mol.get("molecule_hierarchy") or {}).get("parent_chembl_id") or mol[
        "molecule_chembl_id"
    ]
    row.update(
        chembl_id=parent,
        chembl_name=mol.get("pref_name") or "",
        match=match,
        atc_codes="; ".join(mol.get("atc_classifications") or []),
    )
    data = get(f"{CHEMBL}/mechanism.json", {"parent_molecule_chembl_id": parent, "limit": 100})
    mechs = (data or {}).get("mechanisms", [])
    target_ids = sorted({m["target_chembl_id"] for m in mechs if m.get("target_chembl_id")})
    names, genes = {}, set()
    for tid in target_ids:
        names[tid], g = target_details(get, tid)
        genes |= g
    row.update(
        mechanism=_join(m.get("mechanism_of_action") for m in mechs),
        action_type=_join(m.get("action_type") for m in mechs),
        targets=_join(names[m["target_chembl_id"]] for m in mechs if m.get("target_chembl_id")),
        target_chembl_ids="; ".join(target_ids),
        target_genes="; ".join(sorted(genes)),
    )
    if not mechs:
        notes.append("no mechanism record in ChEMBL")
    row["fetch_notes"] = "; ".join(notes)
    return row
