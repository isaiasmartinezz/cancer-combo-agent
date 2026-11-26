"""The rebuilt lab must reproduce iteration 1's numbers. Skipped until `make data` has run."""
from pathlib import Path

import pytest

from ccagent.baselines import random_chooser
from ccagent.evaluation import add_strong_flags, visible_truth
from ccagent.lab import SimLab
from ccagent.loop import run_selection
from ccagent.strong import strong_labels

pytestmark = [
    pytest.mark.data,
    pytest.mark.skipif(not Path("data/processed/oneil.csv").exists(), reason="run `make data`"),
]


@pytest.fixture(scope="module")
def truth():
    return visible_truth()


def test_strong_fraction_is_five_percent(truth):
    labels, _ = strong_labels(truth)
    assert len(truth) == 16_907
    assert labels.sum() == pytest.approx(0.05 * len(truth), abs=20)


def test_random_baseline_matches_iteration_1(truth):
    found = []
    for seed in range(5):
        log = add_strong_flags(run_selection(SimLab.from_files(), random_chooser, seed), truth)
        found.append(int(log.strong_joint_within_line.sum()))
    assert found == [11, 8, 8, 16, 10]
