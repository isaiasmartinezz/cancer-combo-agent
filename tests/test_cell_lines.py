import gzip
import io

import pandas as pd
import pytest

from ccagent.data.cell_lines import match_cell_lines, normalize
from ccagent.data.cmp import filter_rnaseq

MODELS = pd.DataFrame({
    "model_id": ["M1", "M2", "M3", "M4"],
    "model_name": ["HT-29", "LoVo", "MSTO-211H", "Dup"],
    "tissue": ["Large Intestine", "Large Intestine", "Lung", "Skin"],
    "cancer_type": ["Colorectal Carcinoma"] * 2 + ["Mesothelioma", "Melanoma"],
})
NO_OVERRIDES = pd.DataFrame(columns=["cell_line", "cmp_model_id", "expected_cmp_name",
                                     "unknown_genes", "reason"])


def lines(*rows):
    return pd.DataFrame(rows, columns=["cell_line", "tissue"])


def test_normalize_ignores_case_and_punctuation():
    assert normalize("HT-29") == normalize("ht29") == "ht29"


def test_exact_name_and_tissue_match():
    out = match_cell_lines(lines(("HT29", "large_intestine"), ("LOVO", "large_intestine")),
                           MODELS, NO_OVERRIDES)
    assert list(out.cmp_model_id) == ["M1", "M2"] and set(out.match) == {"name"}


def test_tissue_disagreement_is_an_error():
    with pytest.raises(ValueError, match="tissue"):
        match_cell_lines(lines(("HT29", "lung")), MODELS, NO_OVERRIDES)


def test_no_or_ambiguous_name_match_needs_an_override():
    with pytest.raises(ValueError, match="override"):
        match_cell_lines(lines(("MSTO", "lung")), MODELS, NO_OVERRIDES)
    two = pd.concat([MODELS, MODELS.iloc[[3]].assign(model_id="M5")])
    with pytest.raises(ValueError, match="2 CMP name matches"):
        match_cell_lines(lines(("DUP", "skin")), two, NO_OVERRIDES)


def test_override_must_name_the_expected_model():
    good = pd.DataFrame([{"cell_line": "MSTO", "cmp_model_id": "M3",
                          "expected_cmp_name": "MSTO-211H", "unknown_genes": "", "reason": "r"}])
    out = match_cell_lines(lines(("MSTO", "lung")), MODELS, good)
    assert out.iloc[0].cmp_model_id == "M3" and out.iloc[0].match == "override"
    with pytest.raises(ValueError, match="expected"):
        match_cell_lines(lines(("MSTO", "lung")), MODELS,
                         good.assign(expected_cmp_name="Something else"))


def test_override_without_model_means_unknown():
    none = pd.DataFrame([{"cell_line": "X", "cmp_model_id": "", "expected_cmp_name": "",
                          "unknown_genes": "", "reason": "no confident match"}])
    out = match_cell_lines(lines(("X", "skin")), MODELS, none)
    assert out.iloc[0].match == "none" and out.iloc[0].cmp_model_id == ""


def test_rnaseq_filter_keeps_only_wanted_models():
    text = ("model_id,gene_symbol,rsem_tpm,data_source,duplicate,extra\n"
            "M1,TP53,5.1,s,f,x\nM9,TP53,4.0,s,f,x\nM1,EGFR,7.2,s,f,x\n")
    out = io.StringIO()
    seen, kept = filter_rnaseq(io.BytesIO(gzip.compress(text.encode())), {"M1"}, out)
    assert (seen, kept) == (3, 2)
    assert out.getvalue().splitlines() == [
        "model_id,gene_symbol,rsem_tpm,data_source,duplicate",
        "M1,TP53,5.1,s,f", "M1,EGFR,7.2,s,f"]
