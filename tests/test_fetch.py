import csv
import hashlib

import pytest

from ccagent.data.fetch import ChecksumMismatch, fetch

CSV = (
    "block_id,drug_row,study_name\n"
    "1,a,ONEIL\n"
    "2,b,ALMANAC\n"
    "3,c,ONEIL\n"
).encode()


def chunks(data, size):
    return [data[i : i + size] for i in range(0, len(data), size)]


@pytest.mark.parametrize("size", [1, 7, 1000])  # chunk edges fall mid-line and mid-row
def test_keeps_only_study_rows(tmp_path, size):
    out = tmp_path / "out.csv"
    n = fetch(chunks(CSV, size), out, expected_md5=hashlib.md5(CSV).hexdigest())
    rows = list(csv.reader(out.open()))
    assert n == 2
    assert rows == [
        ["block_id", "drug_row", "study_name"], ["1", "a", "ONEIL"], ["3", "c", "ONEIL"],
    ]


def test_bad_checksum_fails_and_leaves_no_file(tmp_path):
    out = tmp_path / "out.csv"
    with pytest.raises(ChecksumMismatch):
        fetch(chunks(CSV, 5), out, expected_md5="0" * 32)
    assert not out.exists()
    assert not out.with_suffix(".partial").exists()
