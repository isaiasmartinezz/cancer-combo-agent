import numpy as np
import pandas as pd
import pytest

from ccagent.strong import strong_labels


def truth(n_lines=20, per_line=200, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for li in range(n_lines):
        shift = 30 * li / n_lines  # some lines are generally more drug-sensitive
        for _ in range(per_line):
            rows.append({"cell_line": f"L{li}", "synergy": rng.normal(),
                         "inhibition": rng.normal(40 + shift, 10)})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("definition", ["joint_within_line", "joint", "synergy", "inhibition"])
def test_about_five_percent_are_strong(definition):
    labels, info = strong_labels(truth(), definition)
    assert abs(labels.mean() - 0.05) < 0.005
    assert info["fraction_strong"] == pytest.approx(labels.mean())


def test_strong_needs_both_scores_high():
    t = truth()
    labels, info = strong_labels(t, "joint")
    strong = t[labels]
    assert (strong.synergy >= info["synergy_cutoff"]).all()
    assert (strong.inhibition >= info["inhibition_cutoff"]).all()


def test_within_line_ranking_spreads_strong_across_lines():
    t = truth()
    within, _ = strong_labels(t, "joint_within_line")
    overall, _ = strong_labels(t, "joint")
    top5 = [f"L{i}" for i in range(15, 20)]  # the most drug-sensitive lines
    share = lambda labels: t[labels].cell_line.isin(top5).mean()  # noqa: E731
    assert share(within) < 0.4 < share(overall)


def test_unknown_definition_is_an_error():
    with pytest.raises(ValueError):
        strong_labels(truth(), "other")
