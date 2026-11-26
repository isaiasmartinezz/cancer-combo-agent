"""Selection methods that use no language model."""
import numpy as np
import pandas as pd


def random_chooser(candidates: pd.DataFrame, history: pd.DataFrame, k: int,
                   rng: np.random.Generator):
    picks = list(rng.choice(candidates["exp_id"].to_numpy(), size=k, replace=False))
    return picks, [{"explanation_level": "none", "explanation_text": "random pick"}] * k
