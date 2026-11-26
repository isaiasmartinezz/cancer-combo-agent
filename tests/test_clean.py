import numpy as np
import pandas as pd
import pytest

from ccagent.data.clean import OUTPUT_COLS, clean

DRUG_MAP = {"Alpha": "A", "Beta": "B", "Gamma": "C"}


def raw_row(drug_row, drug_col, line="L1", tissue="lung", loewe=1.0, inhibition=10.0):
    return {
        "drug_row": drug_row, "drug_col": drug_col, "cell_line_name": line,
        "tissue_name": tissue, "css_ri": inhibition, "synergy_loewe": loewe,
        "synergy_bliss": 0.0, "synergy_zip": 0.0, "synergy_hsa": 0.0,
    }


def run(rows):
    return clean(pd.DataFrame(rows), DRUG_MAP)


def test_both_drug_orders_are_one_experiment_and_replicates_average():
    table, report = run([
        raw_row("Beta", "Alpha", loewe=2.0, inhibition=20.0),
        raw_row("Alpha", "Beta", loewe=4.0, inhibition=40.0),
    ])
    assert len(table) == 1
    row = table.iloc[0]
    assert (row.drug_a, row.drug_b) == ("A", "B")
    assert row.n_replicates == 2
    assert row.synergy_loewe == 3.0 and row.inhibition == 30.0
    assert row.synergy_loewe_sd == pytest.approx(np.std([2.0, 4.0], ddof=1))


def test_single_replicate_has_no_spread():
    table, _ = run([raw_row("Alpha", "Beta")])
    assert np.isnan(table.iloc[0].synergy_loewe_sd)


def test_single_drug_rows_are_ignored():
    table, report = run([raw_row("Alpha", "Beta"), raw_row("Gamma", None)])
    assert len(table) == 1 and report["combination_rows"] == 1


def test_missing_outcome_is_dropped_and_counted():
    table, report = run([raw_row("Alpha", "Beta"), raw_row("Alpha", "Gamma", loewe=np.nan)])
    assert len(table) == 1 and report["dropped_missing_outcome"] == 1


def test_unknown_drug_name_is_an_error():
    with pytest.raises(ValueError, match="Delta"):
        run([raw_row("Alpha", "Delta")])


def test_drug_paired_with_itself_is_an_error():
    with pytest.raises(ValueError):
        run([raw_row("Alpha", "Alpha")])


def test_cell_line_with_two_tissues_is_an_error():
    with pytest.raises(ValueError):
        run([raw_row("Alpha", "Beta", tissue="lung"), raw_row("Alpha", "Gamma", tissue="skin")])


def test_ids_are_sorted_unique_and_columns_fixed():
    table, _ = run([
        raw_row("Gamma", "Beta", line="L2"), raw_row("Alpha", "Beta", line="L2"),
        raw_row("Alpha", "Beta", line="L1"),
    ])
    assert list(table.columns) == OUTPUT_COLS
    assert list(table.exp_id) == ["E00000", "E00001", "E00002"]
    assert list(zip(table.drug_a, table.drug_b, table.cell_line, strict=True)) == [
        ("A", "B", "L1"), ("A", "B", "L2"), ("B", "C", "L2"),
    ]


def test_more_than_one_run_of_replicates_is_flagged():
    four = [raw_row("Alpha", "Beta")] * 4
    eight = [raw_row("Alpha", "Gamma")] * 8
    table, report = run(four + eight)
    flags = dict(zip(table.drug_b, table.duplicate_screen, strict=True))
    assert flags == {"B": False, "C": True}
    assert report["duplicate_screen_experiments"] == 1
