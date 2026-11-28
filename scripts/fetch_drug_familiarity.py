"""Count PubMed papers naming each drug (title or abstract), for the famous-vs-obscure analysis.

Needs the internet (NCBI E-utilities, free; no model calls; one request per drug). Same name
clause as the literature searches (dataset ID or generic name). Writes
data/reference/drug_familiarity.csv with the query, count, famous flag and fetch date.
Usage: python -m scripts.fetch_drug_familiarity
"""
import json
import time

import pandas as pd

from ccagent.familiarity import FAMILIARITY, FAMOUS_MIN
from ccagent.literature import drug_names, drug_term, http_fetch


def count(query: str, fetch=http_fetch) -> int:
    found = json.loads(fetch("esearch.fcgi", {"term": query, "retmax": 0, "retmode": "json"}))
    return int(found["esearchresult"]["count"])


def build(fetch=http_fetch, names: dict[str, list[str]] | None = None) -> pd.DataFrame:
    names = names or drug_names()
    rows = []
    for drug in sorted(names):
        query = drug_term(names[drug])
        n = count(query, fetch)
        rows.append({"drug_id": drug, "query": query, "pubmed_count": n,
                     "famous": n >= FAMOUS_MIN, "fetched": time.strftime("%Y-%m-%d")})
    return pd.DataFrame(rows)


def main() -> None:
    table = build()
    FAMILIARITY.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(FAMILIARITY, index=False)
    print(table[["drug_id", "pubmed_count", "famous"]].sort_values("pubmed_count")
          .to_string(index=False))
    print(f"{int(table['famous'].sum())} famous, {int((~table['famous']).sum())} obscure "
          f"(cutoff {FAMOUS_MIN} papers); wrote {FAMILIARITY}")


if __name__ == "__main__":
    main()
