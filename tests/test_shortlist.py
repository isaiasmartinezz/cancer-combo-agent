import numpy as np
import pandas as pd

from ccagent.shortlist import EXPLORE_FRACTION, make_shortlist, upper_confidence_bound

N = 200
CANDIDATES = pd.DataFrame({
    "exp_id": [f"E{i:05d}" for i in range(N)],
    "drug_a": "A", "drug_b": "B",
    "cell_line": [f"L{i % 4}" for i in range(N)],
})
RNG_SEED = 3


class FakeModel:
    """Predicts from fixed arrays, so the shortlist's logic is tested on its own."""

    def __init__(self, mean, std):
        self.mean, self.std = np.asarray(mean, float), np.asarray(std, float)
        self.fitted = 0

    def fit(self, history, target):
        self.fitted += 1
        return self

    def predict(self, candidates):
        return self.mean, self.std


def model():
    rng = np.random.default_rng(0)
    return FakeModel(rng.normal(size=N), rng.uniform(0.1, 2.0, size=N))


def shortlist(**kwargs):
    return make_shortlist(CANDIDATES, pd.DataFrame(), np.random.default_rng(RNG_SEED),
                          model=model(), **kwargs)


def test_shortlist_has_the_asked_for_size_and_no_repeats():
    short = shortlist()
    assert len(short) == 50 and short["exp_id"].is_unique
    assert set(short["exp_id"]) <= set(CANDIDATES["exp_id"])


def test_its_first_rows_are_exactly_what_bayesian_optimization_would_run():
    # this is what makes "agent vs BO" and "agent vs random-from-shortlist" the same comparison
    m = model()
    best = np.argsort(-upper_confidence_bound(m.mean, m.std), kind="stable")[:10]
    assert list(shortlist()["exp_id"][:10]) == list(CANDIDATES["exp_id"].iloc[best])


def test_it_keeps_room_for_the_most_uncertain_candidates():
    m = model()
    short = shortlist()
    n_explore = round(50 * EXPLORE_FRACTION)
    exploratory = list(short["exp_id"])[-n_explore:]
    chosen_for_value = set(np.argsort(-upper_confidence_bound(m.mean, m.std),
                                      kind="stable")[:50 - n_explore])
    rest = [i for i in range(N) if i not in chosen_for_value]
    most_uncertain = [CANDIDATES["exp_id"].iloc[i]
                      for i in sorted(rest, key=lambda i: -m.std[i])[:n_explore]]
    assert exploratory == most_uncertain


def test_predictions_are_shown_alongside_each_candidate():
    short = shortlist()
    assert {"pred_mean", "pred_std"} <= set(short.columns)
    assert short["pred_mean"].iloc[0] >= short["pred_mean"].iloc[-1]


def test_without_a_model_the_shortlist_is_a_random_sample():
    short = make_shortlist(CANDIDATES, pd.DataFrame(), np.random.default_rng(1), model=None)
    assert len(short) == 50 and short["exp_id"].is_unique
    assert "pred_mean" not in short.columns


def test_it_never_asks_for_more_candidates_than_exist():
    few = CANDIDATES.iloc[:7]
    short = make_shortlist(few, pd.DataFrame(), np.random.default_rng(1),
                           model=FakeModel(np.zeros(7), np.ones(7)))
    assert len(short) == 7


def test_tissue_is_shown_when_known():
    short = shortlist(tissue={"L0": "lung"})
    assert short["tissue"].isin(["lung", "unknown"]).all()
