"""Cell Model Passports (Wellcome Sanger) public download files.

The DepMap portal puts automated downloads behind bot verification, which we do not bypass;
CMP publishes the same kinds of data as plain files. "latest" files change between releases, so
every download's size and sha256 are recorded in data/reference/cmp_manifest.json.
"""
import csv
import gzip
import hashlib
from collections.abc import Iterable
from pathlib import Path
from typing import IO

BASE = "https://cog.sanger.ac.uk/cmp/download/"
SMALL_FILES = ["model_list_latest.csv.gz", "mutations_summary_latest.csv.gz",
               "cnv_summary_latest.csv.gz"]
RNASEQ = "rnaseq_latest.csv.gz"  # ~1.2 GB; streamed, only our models kept
RNASEQ_OUT = "rnaseq_oneil_lines.csv"
RNASEQ_COLS = ["model_id", "gene_symbol", "rsem_tpm", "data_source", "duplicate"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def filter_rnaseq(gz_stream: IO[bytes], wanted: Iterable[str], out: IO[str]) -> tuple[int, int]:
    """Copy rows of the gzipped RNA-seq file whose model_id is wanted. Returns (seen, kept)."""
    wanted = set(wanted)
    reader = csv.DictReader(gzip.open(gz_stream, "rt", encoding="utf-8"))
    writer = csv.writer(out)
    writer.writerow(RNASEQ_COLS)
    seen = kept = 0
    for row in reader:
        seen += 1
        if row["model_id"] in wanted:
            writer.writerow([row[c] for c in RNASEQ_COLS])
            kept += 1
    return seen, kept
