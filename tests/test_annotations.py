"""Matching logic against fake PubChem/ChEMBL responses (the real APIs are never called)."""
from ccagent.data.annotations import CHEMBL, PUBCHEM, annotate


class FakeAPI:
    def __init__(self, responses):
        self.responses, self.calls = responses, []

    def __call__(self, url, params):
        key = (url, tuple(sorted((params or {}).items())))
        self.calls.append(key)
        return self.responses.get(key)


def key(url, **params):
    return (url, tuple(sorted(params.items())))


def structure(name, cid, smiles):
    return {key(f"{PUBCHEM}/compound/name/{name}/property/SMILES/JSON"):
            {"PropertyTable": {"Properties": [{"CID": cid, "SMILES": smiles}]}}}


def molecule(chembl_id, parent=None, name="X", atc=()):
    return {"molecule_chembl_id": chembl_id, "pref_name": name, "atc_classifications": list(atc),
            "molecule_hierarchy": {"parent_chembl_id": parent or chembl_id}}


def mechanism(parent, *mechs):
    return {key(f"{CHEMBL}/mechanism.json", parent_molecule_chembl_id=parent, limit=100):
            {"mechanisms": list(mechs)}}


def target(tid, name, *genes):
    def synonyms(gene):
        return [{"syn_type": "UNIPROT", "component_synonym": "P1"},
                {"syn_type": "GENE_SYMBOL", "component_synonym": gene},
                {"syn_type": "GENE_SYMBOL", "component_synonym": "ALT"}]

    comps = [{"target_component_synonyms": synonyms(g)} for g in genes]
    return {key(f"{CHEMBL}/target/{tid}.json"): {"pref_name": name, "target_components": comps}}


def by_name(name, mol):
    return {key(f"{CHEMBL}/molecule.json", pref_name__iexact=name, limit=5): {"molecules": [mol]}}


def test_exact_name_match_uses_parent_for_mechanisms():
    api = FakeAPI({
        **structure("erlotinib", 176870, "C#C"),
        **by_name("erlotinib", molecule("CHEMBL_SALT", parent="CHEMBL553", atc=["L01EB02"])),
        **mechanism("CHEMBL553", {"mechanism_of_action": "EGFR inhibitor",
                                  "action_type": "INHIBITOR", "target_chembl_id": "T1"}),
        **target("T1", "Epidermal growth factor receptor", "egfr"),
    })
    row = annotate(api, "ERLOTINIB", "erlotinib", "176870")
    assert row["chembl_id"] == "CHEMBL553" and row["match"] == "name"
    assert row["targets"] == "Epidermal growth factor receptor"
    assert row["target_genes"] == "EGFR"  # first gene symbol only, upper-cased
    assert row["atc_codes"] == "L01EB02" and row["fetch_notes"] == ""


def test_synonym_then_pubchem_cross_reference():
    syn = key(f"{CHEMBL}/molecule.json", molecule_synonyms__molecule_synonym__iexact="mk-1",
              limit=5)
    api = FakeAPI({**structure("mk-1", 5, "CC"), syn: {"molecules": [molecule("CHEMBL9")]}})
    assert annotate(api, "MK1", "mk-1")["match"] == "synonym"

    api = FakeAPI({
        **structure("mk-2", 6, "CC"),
        key(f"{PUBCHEM}/compound/cid/6/synonyms/JSON"):
            {"InformationList": {"Information": [{"Synonym": ["mk-2", "CHEMBLX", "CHEMBL77"]}]}},
        key(f"{CHEMBL}/molecule/CHEMBL77.json"): molecule("CHEMBL77"),
    })
    row = annotate(api, "MK2", "mk-2")
    assert row["match"] == "pubchem_xref" and row["chembl_id"] == "CHEMBL77"


def test_no_free_text_search_and_nothing_guessed():
    api = FakeAPI(structure("obscure", 7, "CC"))
    row = annotate(api, "OBS", "obscure")
    assert row["chembl_id"] == "" and row["targets"] == ""
    assert "not found in ChEMBL" in row["fetch_notes"]
    assert not any("search" in url for url, _ in api.calls)


def test_missing_mechanism_and_cid_mismatch_are_noted():
    api = FakeAPI({**structure("x", 10, "CC"), **by_name("x", molecule("CHEMBL1"))})
    row = annotate(api, "X", "x", expected_cid="11")
    assert "no mechanism record in ChEMBL" in row["fetch_notes"]
    assert "differs from reference 11" in row["fetch_notes"]
