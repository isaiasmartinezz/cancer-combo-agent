"""Stream DrugComb v1.5 and keep only the O'Neil study rows.

The full file (~1.4 GB) is streamed and never stored. Its md5 is computed on the fly and
checked against the value published on Zenodo, so a changed or truncated download fails.
"""
import csv
import hashlib
from collections.abc import Iterable, Iterator
from pathlib import Path

URL = "https://zenodo.org/records/15235991/files/summary_v_1_5.csv?download=1"
EXPECTED_MD5 = "cd3e24758c10ffaea0124b6e80626c32"
STUDY = "ONEIL"


class ChecksumMismatch(RuntimeError):
    pass


def hashed_lines(chunks: Iterable[bytes], digest) -> Iterator[str]:
    """Decode byte chunks into text lines, feeding every byte into `digest`."""
    buffer = b""
    for chunk in chunks:
        digest.update(chunk)
        buffer += chunk
        *complete, buffer = buffer.split(b"\n")
        for line in complete:
            yield line.decode("utf-8") + "\n"
    if buffer:
        yield buffer.decode("utf-8")


def filter_study(lines: Iterable[str], study: str = STUDY) -> Iterator[list[str]]:
    """Yield the header, then only rows whose study_name equals `study`."""
    reader = csv.reader(lines)
    header = next(reader)
    col = header.index("study_name")
    yield header
    for row in reader:
        if len(row) > col and row[col] == study:
            yield row


def fetch(chunks: Iterable[bytes], out: Path, expected_md5: str = EXPECTED_MD5) -> int:
    """Write the filtered rows to `out`; return the number of study rows kept."""
    digest = hashlib.md5()
    tmp = out.with_suffix(".partial")
    out.parent.mkdir(parents=True, exist_ok=True)
    n = -1  # header is not a data row
    with open(tmp, "w", newline="") as f:
        writer = csv.writer(f)
        for row in filter_study(hashed_lines(chunks, digest)):
            writer.writerow(row)
            n += 1
    if digest.hexdigest() != expected_md5:
        tmp.unlink()
        raise ChecksumMismatch(f"md5 {digest.hexdigest()} != expected {expected_md5}")
    tmp.rename(out)
    return n
