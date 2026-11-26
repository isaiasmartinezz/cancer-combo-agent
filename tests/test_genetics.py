import numpy as np
import pandas as pd
import pytest

from ccagent.data.genetics import build_features

CELL_MAP = pd.DataFrame({
    "cell_line": ["A", "B", "C", "D", "E"],
    "cmp_model_id": ["M1", "M2", "M3", "M4", ""],
    "unknown_genes": ["", "", "", "BRCA1", ""],
})
MODELS = pd.DataFrame({
    "model_id": ["M1", "M2", "M3", "M4"],
    "mutation_data": ["True", "True", "True", "False"],
    "msi_status": ["MSS", "MSI", None, "MSS"],
    "cancer_type": ["Colorectal Carcinoma"] * 4,
})


def mutation(model, gene, change="p.V600E"):
    return {"model_id": model, "gene_symbol": gene, "protein_mutation": change,
            "cancer_driver": "t", "duplicate": "f"}


MUTATIONS = pd.DataFrame([
    mutation("M1", "TP53"), mutation("M2", "TP53"), mutation("M3", "TP53"),  # in 3 lines
    mutation("M1", "BRAF"),                                                  # 1 line, not a target
    mutation("M2", "EGFR"),                                                  # 1 line, a target
    {**mutation("M3", "KRAS"), "cancer_driver": "f"},                        # not a driver
])
CNV = pd.DataFrame({"model_id": ["M1", "M2"], "symbol": ["EGFR", "EGFR"],
                    "cn_category": ["Amplification", "Neutral"]})
RNA = pd.DataFrame({"model_id": ["M1", "M1", "M2"], "gene_symbol": ["EGFR", "ERBB2", "EGFR"],
                    "rsem_tpm": [6.0, 8.0, 4.0], "data_source": "s", "duplicate": "f"})
TARGETS = {"ERLOTINIB": "EGFR", "LAPATINIB": "EGFR; ERBB2", "CARBOPLATIN": "unknown"}


def build(rna=RNA):
    return build_features(CELL_MAP, MODELS, MUTATIONS, CNV, rna, TARGETS)


def test_mutation_panel_is_recurrent_genes_plus_mutated_targets():
    feats, _, _ = build()
    assert sorted(c for c in feats if c.startswith("mut_")) == ["mut_EGFR", "mut_TP53"]
    assert feats.loc["B", "mut_EGFR"] == 1 and feats.loc["A", "mut_EGFR"] == 0


def test_unsequenced_and_unmatched_lines_are_unknown_not_zero():
    feats, texpr, summaries = build()
    assert np.isnan(feats.loc["D", "mut_TP53"]) and feats.loc["D", "has_genetics"] == 0
    assert np.isnan(feats.loc["E", "mut_TP53"]) and feats.loc["E", "cmp_model_id"] == "unknown"
    assert texpr.loc["E"].isna().all()
    assert "genetics unknown" in summaries.loc["E", "summary"]


def test_copy_number_and_missing_copy_number():
    feats, _, _ = build()
    assert feats.loc["A", "amp_EGFR"] == 1 and feats.loc["B", "amp_EGFR"] == 0
    assert np.isnan(feats.loc["C", "amp_EGFR"])  # no CNV rows for M3


def test_target_expression_is_mean_over_target_genes():
    _, texpr, _ = build()
    assert texpr.loc["A", "LAPATINIB"] == 7.0 and texpr.loc["A", "ERLOTINIB"] == 6.0
    assert np.isnan(texpr.loc["A", "CARBOPLATIN"])  # no protein target genes


def test_summary_names_mutations_and_unknown_genes():
    _, _, summaries = build()
    assert "TP53 V600E" in summaries.loc["A", "summary"]
    assert "amplified drug targets: EGFR" in summaries.loc["A", "summary"]
    assert "BRCA1: unknown" in summaries.loc["D", "summary"]


def test_unscaled_expression_is_refused():
    with pytest.raises(ValueError, match="log2"):
        build(RNA.assign(rsem_tpm=[600.0, 800.0, 400.0]))
