"""Selection methods that use no language model: the ladder the agent has to beat.

  random            pick at random from every untested experiment
  uncertainty       pick the ones the response model is least sure about
  bo                Bayesian optimization: highest predicted value plus kappa * uncertainty
  shortlist_random  pick at random from the agent's own 50-candidate shortlist. This is the
                    fair bar for the agent: it answers "does the language model choose better
                    than chance within the list it is handed?"
"""
import numpy as np
import pandas as pd

from ccagent.model import ResponseModel
from ccagent.shortlist import KAPPA, make_shortlist, upper_confidence_bound
from ccagent.targets import with_target

METHODS = ("random", "uncertainty", "bo", "shortlist_random")
MODEL_METHODS = ("uncertainty", "bo", "shortlist_random")


def random_chooser(candidates: pd.DataFrame, history: pd.DataFrame, k: int,
                   rng: np.random.Generator):
    picks = list(rng.choice(candidates["exp_id"].to_numpy(), size=k, replace=False))
    return picks, [{"explanation_level": "none", "explanation_text": "random pick"}] * k


class ModelChooser:
    """Shared by the three methods that use the response model."""

    def __init__(self, method: str, target: str = "joint", tissue: dict | None = None,
                 kappa: float = KAPPA, seed: int = 0, with_genetics: bool = True):
        if method not in MODEL_METHODS:
            raise ValueError(f"method must be one of {MODEL_METHODS}")
        self.method, self.target, self.tissue = method, target, tissue
        self.kappa, self.seed, self.with_genetics = kappa, seed, with_genetics
        self.model: ResponseModel | None = None

    def __call__(self, candidates: pd.DataFrame, history: pd.DataFrame, k: int,
                 rng: np.random.Generator):
        if self.model is None:  # the first call's candidates hold every visible cell line
            self.model = ResponseModel(candidates["cell_line"].unique(), tissue=self.tissue,
                                       with_genetics=self.with_genetics, seed=self.seed)
        past = with_target(history, self.target)
        if self.method == "shortlist_random":
            shortlist = make_shortlist(candidates, past, rng, model=self.model,
                                       target=self.target, kappa=self.kappa)
            chosen = rng.choice(len(shortlist), size=k, replace=False)
            note = {"explanation_level": "none",
                    "explanation_text": "random pick from the 50-candidate shortlist"}
            return list(shortlist["exp_id"].iloc[chosen]), [note] * k

        mean, std = self.model.fit(past, self.target).predict(candidates)
        score = std if self.method == "uncertainty" \
            else upper_confidence_bound(mean, std, self.kappa)
        best = np.argsort(-score, kind="stable")[:k]
        notes = [{"explanation_level": "none",
                  "explanation_text": f"{self.method}: predicted {mean[i]:.2f}, "
                                      f"uncertainty {std[i]:.2f}"} for i in best]
        return list(candidates["exp_id"].iloc[best]), notes


def make_baseline(method: str, **kwargs):
    if method == "random":
        return random_chooser
    return ModelChooser(method, **kwargs)
