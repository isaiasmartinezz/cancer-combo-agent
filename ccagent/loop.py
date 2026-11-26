"""The experiment loop shared by every method, so all are compared under the same rules.

Round 0 is the same random batch for every method with a given seed. After that, each round the
method (a "chooser") sees the untested candidates and its own results so far, and picks k.
"""
from collections.abc import Callable

import numpy as np
import pandas as pd

from ccagent.lab import SimLab

BATCH_SIZE = 10
LOG_COLS = [
    "round", "exp_id", "drug_a", "drug_b", "cell_line", "synergy", "inhibition",
    "hypothesis_id", "explanation_level", "explanation_text", "source_ids",
]

# chooser(candidates, history, k, rng) -> (exp_ids, notes); one note dict per pick
Chooser = Callable[[pd.DataFrame, pd.DataFrame, int, np.random.Generator], tuple[list, list]]


def initial_batch(candidates: pd.DataFrame, seed: int, k: int) -> list[str]:
    rng = np.random.default_rng(seed)
    return list(rng.choice(candidates["exp_id"].to_numpy(), size=k, replace=False))


def run_selection(
    lab: SimLab, chooser: Chooser, seed: int, batch_size: int = BATCH_SIZE
) -> pd.DataFrame:
    """Run until the budget is spent; return the run log (one row per experiment)."""
    rng = np.random.default_rng(seed + 1)  # separate stream from the shared first batch
    batches = []
    history = pd.DataFrame(columns=LOG_COLS)
    round_num = 0
    while lab.budget_remaining() > 0:
        candidates = lab.candidates()
        k = min(batch_size, lab.budget_remaining(), len(candidates))
        if round_num == 0:
            picks = initial_batch(candidates, seed, k)
            notes = [{"explanation_level": "none", "explanation_text": "shared first batch"}] * k
        else:
            picks, notes = chooser(candidates, history, k, rng)
            if len(picks) != k or len(notes) != k:
                raise ValueError(
                    f"chooser returned {len(picks)} picks and {len(notes)} notes for k={k}"
                )
        results = lab.run(picks)
        batch = (
            candidates.set_index("exp_id").loc[picks].reset_index()
            .merge(results, on="exp_id")
            .join(pd.DataFrame(notes))
        )
        batch["round"] = round_num
        batches.append(batch.reindex(columns=LOG_COLS))
        history = pd.concat(batches, ignore_index=True)
        round_num += 1
    return history
