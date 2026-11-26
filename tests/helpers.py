"""Small synthetic datasets for tests."""
import numpy as np
import pandas as pd


def synthetic_table(n_lines=4, n_drugs=6, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    drugs = [f"D{i}" for i in range(n_drugs)]
    rows = []
    for li in range(n_lines):
        for i, a in enumerate(drugs):
            for b in drugs[i + 1 :]:
                rows.append({"drug_a": a, "drug_b": b, "cell_line": f"L{li}", "tissue": "t",
                             "synergy_loewe": rng.normal(), "inhibition": rng.normal(50, 10)})
    table = pd.DataFrame(rows)
    table.insert(0, "exp_id", [f"E{i:05d}" for i in range(len(table))])
    return table
