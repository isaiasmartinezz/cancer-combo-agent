"""Build the cell-line map (committed) and genetics features (data/processed/). Offline.
Usage: python -m scripts.build_cell_lines
"""
from pathlib import Path

import pandas as pd

from ccagent.data.cell_lines import match_cell_lines
from ccagent.data.genetics import build_features

RAW = Path("data/raw/cmp")
REFERENCE = Path("data/reference")
OUT = Path("data/processed")


def main() -> None:
    models = pd.read_csv(RAW / "model_list_latest.csv.gz", low_memory=False)
    cell_map = match_cell_lines(pd.read_csv(OUT / "cell_lines.csv"), models,
                                pd.read_csv(REFERENCE / "cell_line_overrides.csv", dtype=str))
    cell_map.to_csv(REFERENCE / "cell_line_map.csv", index=False)

    drugs = pd.read_csv(REFERENCE / "drug_annotations.csv", dtype=str)
    feats, texpr, summaries = build_features(
        cell_map, models,
        pd.read_csv(RAW / "mutations_summary_latest.csv.gz", low_memory=False),
        pd.read_csv(RAW / "cnv_summary_latest.csv.gz", low_memory=False),
        pd.read_csv(RAW / "rnaseq_oneil_lines.csv", low_memory=False),
        dict(zip(drugs["drug_id"], drugs["target_genes"], strict=True)),
    )
    feats.to_csv(OUT / "cell_line_features.csv")
    texpr.round(3).to_csv(OUT / "target_expression.csv")
    summaries.to_csv(OUT / "cell_line_summaries.csv")
    print(cell_map["match"].value_counts().to_string())
    print(f"{len(feats)} lines; with mutation data: {int(feats.has_genetics.sum())}; "
          f"mutation panel: {sum(c.startswith('mut_') for c in feats)} genes")


if __name__ == "__main__":
    main()
