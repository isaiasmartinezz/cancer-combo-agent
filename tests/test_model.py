import numpy as np
import pandas as pd
import pytest

from ccagent.features import FINGERPRINT_BITS, FeatureSpace, drug_features
from ccagent.model import ResponseModel

ANNOTATIONS = pd.DataFrame({
    "drug_id": ["A", "B", "C"],
    "smiles": ["CCO", "c1ccccc1", "unknown"],
    "targets": ["EGFR", "EGFR; TP53", "unknown"],
})
LINES = ["L1", "L2"]
TISSUE = {"L1": "lung", "L2": "skin"}
GENETICS = pd.DataFrame({"has_genetics": [1, 0], "mut_TP53": [1.0, np.nan]}, index=LINES)
EXPRESSION = pd.DataFrame({"A": [6.0, np.nan], "B": [8.0, 4.0], "C": [np.nan, np.nan]}, index=LINES)


def space(**kwargs):
    return FeatureSpace(LINES, ANNOTATIONS, **kwargs)


def experiments(*rows):
    return pd.DataFrame(rows, columns=["drug_a", "drug_b", "cell_line"])


def test_drug_without_a_structure_gets_an_empty_fingerprint():
    drugs = drug_features(ANNOTATIONS)
    assert drugs.loc["C", [c for c in drugs if c.startswith("fp_")]].sum() == 0
    assert drugs.loc["A", [c for c in drugs if c.startswith("fp_")]].sum() > 0
    assert "target_unknown" not in drugs.columns  # "unknown" is not a target


def test_drug_order_cannot_change_an_experiment_s_features():
    features = space(tissue=TISSUE, target_expression=EXPRESSION)
    forwards = features.transform(experiments(("A", "B", "L1")))
    backwards = features.transform(experiments(("B", "A", "L1")))
    assert (forwards == backwards).all()


def test_cell_lines_and_tissue_and_genetics_enter_the_features():
    plain = space().transform(experiments(("A", "B", "L1")))
    rich = space(tissue=TISSUE, genetics=GENETICS).transform(experiments(("A", "B", "L1")))
    assert rich.shape[1] > plain.shape[1]
    assert plain.shape[1] == 2 * FINGERPRINT_BITS - FINGERPRINT_BITS + len(ANNOTATIONS) + 2 + 2


def test_unknown_genetics_become_zero_not_missing():
    rich = space(genetics=GENETICS).transform(experiments(("A", "B", "L2")))
    assert np.isfinite(rich).all()


def test_unknown_expression_falls_back_to_the_column_average():
    features = space(target_expression=EXPRESSION)
    # L2 has no value for drug A, so it takes A's mean over the lines that do (6.0)
    assert features.expression.loc["L2", "A"] == 6.0
    assert np.isfinite(features.transform(experiments(("A", "B", "L2")))).all()


def history(n=12, seed=0):
    rng = np.random.default_rng(seed)
    rows = [("A", "B", "L1"), ("A", "C", "L1"), ("B", "C", "L2"), ("A", "B", "L2")] * (n // 4)
    table = pd.DataFrame(rows, columns=["drug_a", "drug_b", "cell_line"])
    table["synergy"] = rng.normal(size=len(table))
    return table


def model(seed=0):
    return ResponseModel(LINES, features=space(tissue=TISSUE), seed=seed, n_trees=20)


def test_predict_before_fit_is_an_error():
    with pytest.raises(RuntimeError):
        model().predict(experiments(("A", "B", "L1")))


def test_prediction_and_uncertainty_per_candidate():
    candidates = experiments(("A", "B", "L1"), ("B", "C", "L2"))
    mean, std = model().fit(history(), "synergy").predict(candidates)
    assert mean.shape == std.shape == (2,)
    assert (std >= 0).all()


def test_same_seed_and_data_give_the_same_prediction():
    candidates = experiments(("A", "B", "L1"))
    first = model(7).fit(history(), "synergy").predict(candidates)
    second = model(7).fit(history(), "synergy").predict(candidates)
    assert first[0] == second[0] and first[1] == second[1]


def test_the_model_learns_something_from_the_results_it_is_given():
    past = history(16)
    past["synergy"] = np.where(past["cell_line"] == "L1", 10.0, 0.0)
    mean, _ = model().fit(past, "synergy").predict(experiments(("A", "B", "L1"), ("A", "B", "L2")))
    assert mean[0] > mean[1] + 5
