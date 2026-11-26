"""Checks on the real built dataset. Skipped until `make data` has been run."""
from pathlib import Path

import pandas as pd
import pytest

from ccagent.data.clean import cell_line_table
from ccagent.data.split import choose_heldout

BUILT = Path("data/processed/oneil.csv")
pytestmark = [
    pytest.mark.data,
    pytest.mark.skipif(not BUILT.exists(), reason="dataset not built; run `make data`"),
]


@pytest.fixture(scope="module")
def table():
    return pd.read_csv(BUILT)


def test_expected_size(table):
    assert len(table) == 22_737
    assert pd.concat([table.drug_a, table.drug_b]).nunique() == 38
    assert table.cell_line.nunique() == 39


def test_one_row_per_experiment(table):
    assert not table.duplicated(["drug_a", "drug_b", "cell_line"]).any()
    assert table.exp_id.is_unique
    assert (table.drug_a < table.drug_b).all()


def test_every_drug_is_in_the_reference(table):
    reference = set(pd.read_csv("data/reference/drugs.csv").drug_id)
    assert set(table.drug_a) | set(table.drug_b) == reference


def test_no_missing_outcomes(table):
    assert table[["synergy_loewe", "inhibition"]].notna().all().all()


def test_committed_heldout_lines_reproduce(table):
    committed = Path("data/reference/heldout_lines.txt").read_text().split()
    assert choose_heldout(cell_line_table(table)) == committed
    assert set(committed) <= set(table.cell_line)


def test_duplicate_screen_experiments(table):
    flagged = table[table.duplicate_screen]
    assert len(flagged) == 315
    assert set(table.n_replicates) == {4, 8}
