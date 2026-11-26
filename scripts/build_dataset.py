"""Build the clean experiment table from the raw O'Neil rows.

Outputs (data/processed/, not committed): oneil.csv, cell_lines.csv, build_report.json.
Fails if the committed held-out list no longer reproduces. Usage: python -m scripts.build_dataset
"""
import json
from pathlib import Path

import pandas as pd

from ccagent.data.clean import NA_VALUES, cell_line_table, clean, load_drug_map
from ccagent.data.split import choose_heldout

RAW = Path("data/raw/drugcomb_oneil.csv")
REFERENCE = Path("data/reference")
OUT = Path("data/processed")


def main() -> None:
    raw = pd.read_csv(RAW, na_values=NA_VALUES, low_memory=False)
    table, report = clean(raw, load_drug_map(REFERENCE / "drugs.csv"))
    lines = cell_line_table(table)

    committed = (REFERENCE / "heldout_lines.txt").read_text().split()
    if choose_heldout(lines) != committed:
        raise SystemExit("held-out lines no longer reproduce the committed list; investigate")
    report["heldout_lines"] = committed
    report["agent_visible_experiments"] = int((~table["cell_line"].isin(committed)).sum())

    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "oneil.csv", index=False)
    lines.to_csv(OUT / "cell_lines.csv", index=False)
    (OUT / "build_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
