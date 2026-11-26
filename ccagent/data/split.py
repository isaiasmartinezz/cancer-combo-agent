"""Choose the held-out cell lines (never shown to the agent; used for the mechanism test).

About a quarter of the lines, spread across tissues in proportion to their size
(largest-remainder rounding). Ties and picks use one fixed seed. The list is chosen once and
committed in data/reference/heldout_lines.txt; the build checks that it still reproduces.
"""
import numpy as np
import pandas as pd

HELDOUT_N = 10
HELDOUT_SEED = 2026


def choose_heldout(lines: pd.DataFrame, n: int = HELDOUT_N, seed: int = HELDOUT_SEED) -> list[str]:
    rng = np.random.default_rng(seed)
    members = lines.groupby("tissue")["cell_line"].apply(sorted)  # tissues in sorted order
    share = members.apply(len) * n / len(lines)
    quota = np.floor(share).astype(int)

    # Remainders decide who gets the leftover slots; a random draw per tissue breaks ties.
    tiebreak = {t: rng.random() for t in share.index}
    by_remainder = sorted(share.index, key=lambda t: (-(share[t] - quota[t]), tiebreak[t]))
    for t in by_remainder[: n - quota.sum()]:
        quota[t] += 1

    picks = []
    for tissue, names in members.items():
        shuffled = rng.permutation(names)  # drawn for every tissue, even with quota 0
        picks.extend(shuffled[: quota[tissue]])
    return sorted(str(p) for p in picks)
