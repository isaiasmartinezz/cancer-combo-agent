"""Download Cell Model Passports files into data/raw/cmp/ and record them in the manifest.

Needs the internet. Builds the cell-line map in between, because the RNA-seq stream keeps only
our models. Usage: python -m scripts.fetch_cell_lines
"""
import json
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from ccagent.data.cell_lines import match_cell_lines
from ccagent.data.cmp import BASE, RNASEQ, RNASEQ_OUT, SMALL_FILES, filter_rnaseq, sha256

RAW = Path("data/raw/cmp")
MANIFEST = Path("data/reference/cmp_manifest.json")


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for name in SMALL_FILES:
        response = requests.get(BASE + name, timeout=300)
        response.raise_for_status()
        (RAW / name).write_bytes(response.content)
        print(f"saved {name} ({len(response.content):,} bytes)")

    cell_map = match_cell_lines(pd.read_csv("data/processed/cell_lines.csv"),
                                pd.read_csv(RAW / "model_list_latest.csv.gz", low_memory=False),
                                pd.read_csv("data/reference/cell_line_overrides.csv", dtype=str))
    wanted = set(cell_map["cmp_model_id"].dropna()) - {""}
    tmp = RAW / (RNASEQ_OUT + ".partial")
    with requests.get(BASE + RNASEQ, stream=True, timeout=300) as response, \
            open(tmp, "w", newline="") as out:
        response.raise_for_status()
        response.raw.decode_content = False  # the file itself is gzip
        seen, kept = filter_rnaseq(response.raw, wanted, out)
    tmp.rename(RAW / RNASEQ_OUT)
    print(f"rnaseq: scanned {seen:,} rows, kept {kept:,}")

    files = {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)}
             for p in sorted(RAW.iterdir()) if p.suffix in (".gz", ".csv")}
    MANIFEST.write_text(json.dumps({"downloaded_on": date.today().isoformat(), "source": BASE,
                                    "files": files}, indent=2) + "\n")
    print(f"-> {MANIFEST}")


if __name__ == "__main__":
    main()
