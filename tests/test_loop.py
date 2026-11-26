import pytest

from ccagent.baselines import random_chooser
from ccagent.lab import SimLab
from ccagent.loop import LOG_COLS, run_selection
from tests.helpers import synthetic_table

TABLE = synthetic_table()


def lab(budget=30):
    return SimLab(TABLE, heldout=["L3"], budget=budget)


def first_k(candidates, history, k, rng):
    return list(candidates.exp_id[:k]), [{"explanation_level": "none"}] * k


def test_spends_exact_budget_in_rounds_of_ten():
    log = run_selection(lab(30), random_chooser, seed=0)
    assert len(log) == 30 and log.exp_id.is_unique
    assert log.groupby("round").size().tolist() == [10, 10, 10]
    assert list(log.columns) == LOG_COLS


def test_last_round_is_partial_when_budget_is_not_a_multiple():
    log = run_selection(lab(25), random_chooser, seed=0)
    assert log.groupby("round").size().tolist() == [10, 10, 5]


def test_first_batch_is_shared_across_methods():
    a = run_selection(lab(), random_chooser, seed=3)
    b = run_selection(lab(), first_k, seed=3)
    assert list(a[a["round"] == 0].exp_id) == list(b[b["round"] == 0].exp_id)
    assert list(a[a["round"] > 0].exp_id) != list(b[b["round"] > 0].exp_id)


def test_same_seed_same_run():
    assert run_selection(lab(), random_chooser, seed=1).equals(
        run_selection(lab(), random_chooser, seed=1)
    )


def test_history_given_to_chooser_is_only_past_results():
    seen = []

    def spy(candidates, history, k, rng):
        seen.append(set(history.exp_id))
        assert not seen[-1] & set(candidates.exp_id)
        return first_k(candidates, history, k, rng)

    run_selection(lab(30), spy, seed=0)
    assert [len(s) for s in seen] == [10, 20]


def test_chooser_returning_wrong_count_is_an_error():
    def short(candidates, history, k, rng):
        return list(candidates.exp_id[: k - 1]), [{}] * (k - 1)

    with pytest.raises(ValueError):
        run_selection(lab(), short, seed=0)
