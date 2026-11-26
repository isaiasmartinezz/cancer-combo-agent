"""What counts as a "strong" combination. Evaluation only: never visible to any method.

Agreed with the research team (Nov 2026):
  joint_within_line (primary)  Loewe synergy in the top 10% AND inhibition (CSS) above a cutoff
                               set so about 5% of experiments qualify, both ranked within each
                               experiment's own cell line (so finding generally drug-sensitive lines
                               does not count as finding synergy).
  joint (secondary)            the same rule ranked over all experiments.
  synergy / inhibition         top 5% of that single score (extra checks).
Computed over the visible (non-held-out) experiments.
"""
import numpy as np
import pandas as pd

PRIMARY = "joint_within_line"
DEFINITIONS = ("joint_within_line", "joint", "synergy", "inhibition")
SYNERGY_TOP = 0.10
TARGET_FRACTION = 0.05
SINGLE_QUANTILE = 0.95


def _joint(syn: pd.Series, inh: pd.Series) -> tuple[pd.Series, dict]:
    """Synergy fixed at its top 10%; pick the inhibition cutoff giving closest to 5% overall."""
    syn_cut = float(syn.quantile(1 - SYNERGY_TOP))
    syn_ok = syn >= syn_cut
    passing = np.sort(inh[syn_ok].to_numpy())
    cutoffs = np.unique(passing)[::-1]
    fraction = (len(passing) - np.searchsorted(passing, cutoffs, side="left")) / len(syn)
    inh_cut = float(cutoffs[int(np.argmin(np.abs(fraction - TARGET_FRACTION)))])
    labels = syn_ok & (inh >= inh_cut)
    return labels, {
        "synergy_cutoff": syn_cut,
        "inhibition_cutoff": inh_cut,
        "fraction_strong": float(labels.mean()),
    }


def strong_labels(truth: pd.DataFrame, definition: str = PRIMARY) -> tuple[pd.Series, dict]:
    """truth: index exp_id; columns cell_line, synergy, inhibition. Returns (labels, cutoffs)."""
    if definition in ("synergy", "inhibition"):
        cut = float(truth[definition].quantile(SINGLE_QUANTILE))
        labels = truth[definition] >= cut
        return labels, {f"{definition}_cutoff": cut, "fraction_strong": float(labels.mean())}
    if definition == "joint":
        return _joint(truth["synergy"], truth["inhibition"])
    if definition == "joint_within_line":
        pct = truth.groupby("cell_line")[["synergy", "inhibition"]].rank(pct=True)
        return _joint(pct["synergy"], pct["inhibition"])
    raise ValueError(f"definition must be one of {DEFINITIONS}")
