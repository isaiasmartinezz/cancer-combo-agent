"""The 50 candidates the agent chooses from, shared with the random-from-shortlist control.

Mostly the model's best bets by the same score Bayesian optimization uses, plus a few of the
most uncertain so the list is not purely exploitative. The list is ordered by that score, so its
first k rows are exactly the batch Bayesian optimization would run: the agent and its control
start from the same 50, and "did the language model beat the model's own ranking?" is a fair
question to ask of the results.
"""
import numpy as np
import pandas as pd

SHORTLIST_SIZE = 50
EXPLORE_FRACTION = 0.1
KAPPA = 1.0  # how much the score rewards uncertainty; tuned on seeds 100-104, never on 0-4


def upper_confidence_bound(mean: np.ndarray, std: np.ndarray, kappa: float = KAPPA) -> np.ndarray:
    return mean + kappa * std


def make_shortlist(candidates: pd.DataFrame, history: pd.DataFrame, rng: np.random.Generator,
                   model=None, target: str = "joint", n: int = SHORTLIST_SIZE,
                   explore_fraction: float = EXPLORE_FRACTION, kappa: float = KAPPA,
                   tissue: dict | None = None) -> pd.DataFrame:
    n = min(n, len(candidates))
    if model is None:  # before a model exists, the shortlist is a random sample
        shortlist = candidates.iloc[rng.choice(len(candidates), size=n, replace=False)] \
            .reset_index(drop=True)
    else:
        mean, std = model.fit(history, target).predict(candidates)
        order = np.argsort(-upper_confidence_bound(mean, std, kappa), kind="stable")
        n_explore = max(1, round(n * explore_fraction))
        best = order[:n - n_explore]
        rest = np.setdiff1d(np.arange(len(candidates)), best)
        unsure = rest[np.argsort(-std[rest], kind="stable")[:n_explore]]
        keep = np.concatenate([best, unsure])
        shortlist = candidates.iloc[keep].reset_index(drop=True)
        shortlist["pred_mean"] = mean[keep].round(2)
        shortlist["pred_std"] = std[keep].round(2)
    if tissue:
        shortlist.insert(4, "tissue", shortlist["cell_line"].map(tissue).fillna("unknown"))
    return shortlist
