import pandas as pd

from ccagent.data.split import choose_heldout

LINES = pd.DataFrame({
    "cell_line": [f"{t}{i}" for t, k in [("lung", 8), ("skin", 6), ("ovary", 4), ("bone", 2)]
                  for i in range(k)],
    "tissue": [t for t, k in [("lung", 8), ("skin", 6), ("ovary", 4), ("bone", 2)]
               for _ in range(k)],
})


def test_same_seed_same_lines():
    assert choose_heldout(LINES, n=5, seed=1) == choose_heldout(LINES, n=5, seed=1)


def test_picks_n_real_lines():
    picks = choose_heldout(LINES, n=5, seed=1)
    assert len(picks) == len(set(picks)) == 5
    assert set(picks) <= set(LINES.cell_line)


def test_each_tissue_gets_at_least_its_whole_share():
    picks = choose_heldout(LINES, n=5, seed=1)
    counts = LINES.set_index("cell_line").loc[picks, "tissue"].value_counts()
    # shares at n=5 of 20 lines: lung 2.0, skin 1.5, ovary 1.0, bone 0.5
    assert counts.get("lung", 0) >= 2 and counts.get("skin", 0) >= 1
    assert counts.get("ovary", 0) >= 1
