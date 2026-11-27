"""Fill the PubMed cache with every search the agent can make, so agent runs stay offline.

Needs the internet (NCBI E-utilities, free; no model calls). Re-running only fetches what is
missing. Usage: python -m scripts.prefetch_literature
"""
import argparse
import time

import pandas as pd

from ccagent.agent import PAPERS_PER_QUERY
from ccagent.lab import DATA
from ccagent.literature import Exclusions, Literature, PubMed, drug_names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--papers", type=int, default=PAPERS_PER_QUERY)
    args = parser.parse_args()
    lit = Literature(PubMed(allow_network=True), Exclusions.from_file(), drug_names())
    table = pd.read_csv(DATA, usecols=["drug_a", "drug_b"])
    drugs = sorted(set(table["drug_a"]) | set(table["drug_b"]))
    pairs = sorted(set(zip(table["drug_a"], table["drug_b"], strict=True)))
    started = time.time()
    for i, drug in enumerate(drugs, 1):
        lit.for_drug(drug, args.papers)
        print(f"drug {i}/{len(drugs)} {drug}", flush=True)
    for i, (a, b) in enumerate(pairs, 1):
        lit.for_pair(a, b, args.papers)
        if i % 50 == 0 or i == len(pairs):
            print(f"pair {i}/{len(pairs)} ({time.time() - started:.0f}s)", flush=True)
    print(f"done: {len(drugs)} drugs, {len(pairs)} pairs; "
          f"{len(set(lit.excluded_log))} excluded papers seen")


if __name__ == "__main__":
    main()
