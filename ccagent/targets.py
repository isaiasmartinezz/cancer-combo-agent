"""What a method tries to maximize, computed only from results the lab has already returned.

  synergy / inhibition   the raw score
  joint                  "good on both, for this cell line": each score has its cell line's
                         observed mean removed (shrunk toward the overall mean so a line with one
                         or two results is not over-adjusted), is standardized, and the joint value
                         is the smaller of the two.

This mirrors the evaluation definition (both scores high, ranked within line) without ever using
hidden results or the evaluation cutoffs.
"""
import numpy as np
import pandas as pd

TARGETS = ("synergy", "inhibition", "joint")
SHRINK = 2  # a line's own mean counts like SHRINK extra results sitting at the overall mean
DESCRIPTIONS = {
    "synergy": "synergy",
    "inhibition": "growth inhibition",
    "joint": "joint score (both synergy and growth inhibition high relative to other experiments "
             "in the same cell line; the smaller of the two standardized, line-adjusted scores)",
}


def line_adjusted(values: pd.Series, lines: pd.Series, shrink: int = SHRINK) -> pd.Series:
    """Remove each cell line's own average level, then standardize."""
    values = pd.to_numeric(values).astype(float)
    grouped = values.groupby(lines)
    n, mean = grouped.transform("size"), grouped.transform("mean")
    residual = values - (n * mean + shrink * values.mean()) / (n + shrink)
    sd = residual.std()
    return (residual - residual.mean()) / (sd if sd and np.isfinite(sd) else 1.0)


def joint_score(history: pd.DataFrame) -> pd.Series:
    synergy = line_adjusted(history["synergy"], history["cell_line"])
    inhibition = line_adjusted(history["inhibition"], history["cell_line"])
    return np.minimum(synergy, inhibition)


def with_target(history: pd.DataFrame, target: str) -> pd.DataFrame:
    """A copy of history with a `joint` column added when that is the target."""
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}")
    if target != "joint" or history is None or len(history) == 0:
        return history
    out = history.copy()
    out["joint"] = joint_score(out).round(4) if len(out) > 1 else 0.0
    return out
