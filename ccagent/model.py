"""The response model: predicts an untested experiment's outcome, with an honest uncertainty.

A random forest, retrained from scratch on everything seen so far each time `fit` is called.
The prediction is the average over the trees and the uncertainty is their spread, so an
experiment unlike anything tested yet gets a wide spread.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from ccagent.features import FeatureSpace

N_TREES = 100
MAX_FEATURES = 0.3


class ResponseModel:
    def __init__(self, cell_lines, tissue: dict | None = None, with_genetics: bool = True,
                 seed: int = 0, n_trees: int = N_TREES, features: FeatureSpace | None = None):
        self.features = features or FeatureSpace.from_files(
            cell_lines, tissue=tissue, with_genetics=with_genetics
        )
        self.seed, self.n_trees = seed, n_trees
        self.forest: RandomForestRegressor | None = None

    def fit(self, history: pd.DataFrame, outcome: str) -> "ResponseModel":
        self.forest = RandomForestRegressor(
            n_estimators=self.n_trees, max_features=MAX_FEATURES, min_samples_leaf=1,
            n_jobs=-1, random_state=self.seed,
        )
        self.forest.fit(self.features.transform(history),
                        pd.to_numeric(history[outcome]).to_numpy())
        return self

    def predict(self, candidates: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """(predicted value, uncertainty) for each candidate."""
        if self.forest is None:
            raise RuntimeError("the model must be fitted before predicting")
        x = self.features.transform(candidates)
        per_tree = np.stack([tree.predict(x) for tree in self.forest.estimators_])
        return per_tree.mean(axis=0), per_tree.std(axis=0)
