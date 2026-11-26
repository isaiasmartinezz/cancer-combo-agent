import numpy as np
import pandas as pd
import pytest

from ccagent import baselines
from ccagent.baselines import ModelChooser, make_baseline, random_chooser
from ccagent.shortlist import upper_confidence_bound
from tests.test_shortlist import CANDIDATES, FakeModel

HISTORY = pd.DataFrame({"cell_line": ["L0", "L1"], "synergy": [1.0, 2.0],
                        "inhibition": [3.0, 4.0]})
MEAN = np.linspace(-1, 1, len(CANDIDATES))
STD = np.abs(np.sin(np.arange(len(CANDIDATES))))


@pytest.fixture
def fake_model(monkeypatch):
    model = FakeModel(MEAN, STD)
    monkeypatch.setattr(baselines, "ResponseModel", lambda *a, **k: model)
    return model


def choose(method, k=10, **kwargs):
    chooser = make_baseline(method, **kwargs)
    return chooser(CANDIDATES, HISTORY, k, np.random.default_rng(5))


def test_every_method_returns_one_pick_and_one_note_each():
    for method in ("random", "uncertainty", "bo", "shortlist_random"):
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(baselines, "ResponseModel", lambda *a, **k: FakeModel(MEAN, STD))
            picks, notes = choose(method)
        assert len(picks) == len(notes) == 10 and len(set(picks)) == 10


def test_random_picks_from_everything_untested():
    picks, _ = choose("random")
    assert set(picks) <= set(CANDIDATES["exp_id"])


def test_uncertainty_takes_the_least_certain(fake_model):
    picks, _ = choose("uncertainty")
    expected = CANDIDATES["exp_id"].iloc[np.argsort(-STD, kind="stable")[:10]]
    assert picks == list(expected)


def test_bayesian_optimization_takes_the_best_score(fake_model):
    picks, _ = choose("bo")
    expected = CANDIDATES["exp_id"].iloc[
        np.argsort(-upper_confidence_bound(MEAN, STD), kind="stable")[:10]]
    assert picks == list(expected)


def test_shortlist_random_stays_inside_the_shortlist(fake_model):
    picks, notes = choose("shortlist_random")
    best = set(CANDIDATES["exp_id"].iloc[
        np.argsort(-upper_confidence_bound(MEAN, STD), kind="stable")[:45]])
    assert set(picks) <= set(CANDIDATES["exp_id"])
    assert len(set(picks) & best) < 10  # not simply the model's top picks
    assert "shortlist" in notes[0]["explanation_text"]


def test_the_model_is_built_once_and_reused(fake_model):
    chooser = ModelChooser("bo")
    rng = np.random.default_rng(0)
    chooser(CANDIDATES, HISTORY, 5, rng)
    first = chooser.model
    chooser(CANDIDATES, HISTORY, 5, rng)
    assert chooser.model is first and first.fitted == 2  # refitted each round, rebuilt never


def test_an_unknown_method_is_refused():
    with pytest.raises(ValueError):
        ModelChooser("vibes")


def test_random_chooser_needs_no_model():
    picks, notes = random_chooser(CANDIDATES, HISTORY, 3, np.random.default_rng(0))
    assert len(picks) == 3 and notes[0]["explanation_level"] == "none"
