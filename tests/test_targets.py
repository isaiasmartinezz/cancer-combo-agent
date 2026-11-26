import numpy as np
import pandas as pd
import pytest

from ccagent.targets import joint_score, line_adjusted, with_target


def history(rows):
    return pd.DataFrame(rows, columns=["cell_line", "synergy", "inhibition"])


def test_a_strong_result_in_a_dull_line_beats_a_weak_one_in_a_sensitive_line():
    # every L2 score is 100 higher, so without adjustment L2 would win every comparison
    rows = [("L1", v, 0) for v in range(20)] + [("L2", 100 + v, 0) for v in range(20)]
    h = history(rows)
    adjusted = line_adjusted(h["synergy"], h["cell_line"]).to_numpy()
    assert adjusted[19] > adjusted[20]  # L1's best now outranks L2's worst
    assert list(adjusted[:20]) == sorted(adjusted[:20])  # order within a line is untouched


def test_the_line_effect_shrinks_as_a_line_gathers_more_results():
    def gap(n):
        rows = [("L1", v, 0) for v in range(n)] + [("L2", 100 + v, 0) for v in range(n)]
        h = history(rows)
        adjusted = line_adjusted(h["synergy"], h["cell_line"]).to_numpy()
        return adjusted[n:].mean() - adjusted[:n].mean()

    assert gap(3) > gap(20) > gap(100) > 0


def test_a_line_with_one_result_is_not_fully_adjusted_away():
    h = history([("L1", 10, 0), ("L1", 12, 0), ("L1", 14, 0), ("L2", 40, 0)])
    adjusted = line_adjusted(h["synergy"], h["cell_line"])
    assert adjusted.iloc[3] > adjusted.iloc[:3].max()  # the single high result still stands out


def test_joint_is_the_weaker_of_the_two_scores():
    h = history([("L1", 10, 9), ("L1", 8, 2), ("L1", 3, 7), ("L1", 1, 1)])
    joint = joint_score(h)
    assert joint.idxmax() == 0 and joint.idxmin() == 3


def test_with_target_adds_joint_only_when_asked():
    h = history([("L1", 1, 2), ("L1", 3, 4)])
    assert "joint" not in with_target(h, "synergy").columns
    assert "joint" in with_target(h, "joint").columns


def test_with_target_handles_a_single_row_and_rejects_unknown_targets():
    assert with_target(history([("L1", 1, 2)]), "joint")["joint"].tolist() == [0.0]
    assert with_target(history([]), "joint").empty
    with pytest.raises(ValueError):
        with_target(history([("L1", 1, 2)]), "nonsense")


def test_adjustment_survives_a_line_whose_scores_are_all_equal():
    h = history([("L1", 5, 1), ("L1", 5, 2), ("L2", 9, 3)])
    assert np.isfinite(line_adjusted(h["synergy"], h["cell_line"])).all()
