"""Download the O'Neil rows of DrugComb v1.5 into data/raw/drugcomb_oneil.csv.

Needs the internet; never run by tests. Usage: python -m scripts.fetch_oneil
"""
from pathlib import Path

import requests

from ccagent.data.fetch import URL, fetch

OUT = Path("data/raw/drugcomb_oneil.csv")


def main() -> None:
    with requests.get(URL, stream=True, timeout=60) as response:
        response.raise_for_status()
        n = fetch(response.iter_content(chunk_size=1 << 20), OUT)
    print(f"kept {n:,} O'Neil rows -> {OUT} (md5 of full file verified)")


if __name__ == "__main__":
    main()
