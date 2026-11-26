"""Scoring a finished run. Reads the data on its own; nothing here is passed to a method."""
from pathlib import Path

import pandas as pd

from ccagent.lab import DATA, HELDOUT, SYNERGY, read_heldout
from ccagent.strong import DEFINITIONS, strong_labels


def visible_truth(
    data: Path = DATA, heldout: Path = HELDOUT, synergy_col: str = SYNERGY
) -> pd.DataFrame:
    table = pd.read_csv(data)
    table = table[~table["cell_line"].isin(read_heldout(heldout))]
    return (
        table.rename(columns={synergy_col: "synergy"})
        .set_index("exp_id")[["drug_a", "drug_b", "cell_line", "synergy", "inhibition"]]
    )


def add_strong_flags(log: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    out = log.copy()
    for definition in DEFINITIONS:
        labels, _ = strong_labels(truth, definition)
        out[f"strong_{definition}"] = out["exp_id"].map(labels).astype(bool)
    return out
