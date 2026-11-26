"""The simulated lab: the only way any method learns an experiment's outcome.

Rules it enforces
- Held-out cell lines do not exist as far as a method can tell (not listed, cannot be run).
- An experiment can be run once. A batch cannot repeat an ID or exceed the remaining budget.
- `run` returns only the two outcomes a real lab would report: synergy and inhibition.
The lab has no method that reveals untested outcomes. Evaluation reads the data separately.
"""
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

CANDIDATE_COLS = ["exp_id", "drug_a", "drug_b", "cell_line"]
DATA = Path("data/processed/oneil.csv")
HELDOUT = Path("data/reference/heldout_lines.txt")
SYNERGY = "synergy_loewe"  # primary score; Bliss is the robustness check


class LabError(ValueError):
    pass


def read_heldout(path: Path = HELDOUT) -> list[str]:
    return path.read_text().split()


class SimLab:
    def __init__(
        self,
        table: pd.DataFrame,
        heldout: Iterable[str],
        budget: int = 250,
        synergy_col: str = SYNERGY,
    ):
        heldout = set(heldout)
        missing = heldout - set(table["cell_line"])
        if missing:
            raise LabError(f"held-out lines not in data: {sorted(missing)}")
        visible = table[~table["cell_line"].isin(heldout)].set_index("exp_id").sort_index()
        self._outcomes = visible[[synergy_col, "inhibition"]].rename(
            columns={synergy_col: "synergy"}
        )
        self._design = visible[CANDIDATE_COLS[1:]]
        self._budget = budget
        self._done: set[str] = set()
        self.batches: list[list[str]] = []  # every batch in order, for replay and audit

    @classmethod
    def from_files(cls, data: Path = DATA, heldout: Path = HELDOUT, **kwargs) -> "SimLab":
        return cls(pd.read_csv(data), read_heldout(heldout), **kwargs)

    def candidates(self) -> pd.DataFrame:
        """Every visible experiment not yet run, in exp_id order."""
        todo = self._design.index[~self._design.index.isin(self._done)]
        return self._design.loc[todo].reset_index()[CANDIDATE_COLS]

    def budget_remaining(self) -> int:
        return self._budget - len(self._done)

    def run(self, exp_ids: Iterable[str]) -> pd.DataFrame:
        exp_ids = list(exp_ids)
        if not exp_ids:
            raise LabError("empty batch")
        if len(set(exp_ids)) != len(exp_ids):
            raise LabError("batch repeats an exp_id")
        unknown = set(exp_ids) - set(self._design.index)
        if unknown:
            raise LabError(f"unknown or held-out exp_ids: {sorted(unknown)[:5]}")
        repeats = set(exp_ids) & self._done
        if repeats:
            raise LabError(f"already run: {sorted(repeats)[:5]}")
        if len(exp_ids) > self.budget_remaining():
            raise LabError(f"over budget: asked {len(exp_ids)}, {self.budget_remaining()} left")
        self._done.update(exp_ids)
        self.batches.append(exp_ids)
        return self._outcomes.loc[exp_ids].reset_index()
