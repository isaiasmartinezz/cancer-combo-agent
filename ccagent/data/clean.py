"""Turn raw O'Neil rows into one row per (drug pair, cell line).

Rules
- Drug names are mapped to screen IDs through data/reference/drugs.csv; an unknown name is an error.
- Pair order is alphabetical, so (A, B) and (B, A) are the same experiment.
- Replicates of the same experiment are averaged; their standard deviation is kept as a noise check.
- Rows missing the primary synergy score or inhibition are dropped and counted.
- The screen ran 4 replicates per experiment. Experiments with more were screened in a second
  run, which reads systematically lower; they are averaged like the rest and flagged in
  `duplicate_screen` so results can be checked without them.
"""
from pathlib import Path

import numpy as np
import pandas as pd

SYNERGY_COLS = ["synergy_loewe", "synergy_bliss", "synergy_zip", "synergy_hsa"]
PRIMARY_SYNERGY = "synergy_loewe"  # agreed with the research team; Bliss is the robustness check
KEYS = ["drug_a", "drug_b", "cell_line", "tissue"]
OUTPUT_COLS = [
    "exp_id", *KEYS, "inhibition", *SYNERGY_COLS,
    "n_replicates", "synergy_loewe_sd", "inhibition_sd", "duplicate_screen",
]
NA_VALUES = ["NULL", "NA", "\\N"]
REPLICATES_PER_RUN = 4


def load_drug_map(path: Path) -> dict[str, str]:
    table = pd.read_csv(path)
    if table["drugcomb_name"].duplicated().any() or table["drug_id"].duplicated().any():
        raise ValueError("drug reference table has duplicate names")
    return dict(zip(table["drugcomb_name"], table["drug_id"], strict=True))


def clean(raw: pd.DataFrame, drug_map: dict[str, str]) -> tuple[pd.DataFrame, dict]:
    report = {"raw_rows": len(raw)}
    raw = raw[raw["drug_col"].notna()].copy()
    report["combination_rows"] = len(raw)

    unknown = (set(raw["drug_row"]) | set(raw["drug_col"])) - set(drug_map)
    if unknown:
        raise ValueError(f"no reference entry for: {sorted(unknown)}")
    a, b = raw["drug_row"].map(drug_map), raw["drug_col"].map(drug_map)
    if (a == b).any():
        raise ValueError("a combination row pairs a drug with itself")
    raw["drug_a"] = np.where(a < b, a, b)
    raw["drug_b"] = np.where(a < b, b, a)
    raw = raw.rename(
        columns={"cell_line_name": "cell_line", "tissue_name": "tissue", "css_ri": "inhibition"}
    )

    tissues = raw.groupby("cell_line")["tissue"].nunique()
    if (tissues > 1).any():
        bad = list(tissues[tissues > 1].index)
        raise ValueError(f"cell lines with more than one tissue: {bad}")

    grouped = raw.groupby(KEYS)
    out = grouped[SYNERGY_COLS + ["inhibition"]].mean()
    out["n_replicates"] = grouped.size()
    out["synergy_loewe_sd"] = grouped[PRIMARY_SYNERGY].std(ddof=1)
    out["inhibition_sd"] = grouped["inhibition"].std(ddof=1)
    out["duplicate_screen"] = out["n_replicates"] > REPLICATES_PER_RUN
    out = out.reset_index()

    before = len(out)
    out = out.dropna(subset=[PRIMARY_SYNERGY, "inhibition"])
    report["dropped_missing_outcome"] = before - len(out)

    out = out.sort_values(["drug_a", "drug_b", "cell_line"]).reset_index(drop=True)
    out.insert(0, "exp_id", [f"E{i:05d}" for i in range(len(out))])
    out = out[OUTPUT_COLS]

    replicated = out[out["n_replicates"] > 1]
    report.update(
        experiments=len(out),
        drugs=int(pd.concat([out["drug_a"], out["drug_b"]]).nunique()),
        cell_lines=int(out["cell_line"].nunique()),
        pairs=int(out[["drug_a", "drug_b"]].drop_duplicates().shape[0]),
        duplicate_screen_experiments=int(out["duplicate_screen"].sum()),
        replicates_per_experiment=out["n_replicates"].value_counts().sort_index().to_dict(),
        median_loewe_replicate_sd=float(replicated["synergy_loewe_sd"].median())
        if len(replicated) else None,
    )
    return out, report


def cell_line_table(table: pd.DataFrame) -> pd.DataFrame:
    return (
        table.groupby(["cell_line", "tissue"]).size().rename("n_experiments").reset_index()
    )
