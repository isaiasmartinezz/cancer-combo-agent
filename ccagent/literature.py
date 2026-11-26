"""Literature for the agent: PubMed search with a local cache and a leakage filter.

Two layers, kept apart so each can be tested on its own:
  PubMed      raw search results, cached on disk. Agent runs read the cache only; the network is
              used only when a caller passes allow_network=True (the prefetch script), so a run
              always sees the same papers and cannot reach the internet mid-run.
  Literature  what the agent is allowed to see: papers with an abstract, one per title, and none
              matching the exclusion list. The filter runs on every read, so editing the list
              takes effect without re-downloading anything.
"""
import hashlib
import json
import re
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

EXCLUSIONS = Path("data/reference/excluded_sources.txt")
CACHE_DIR = Path("data/cache/pubmed")
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
OVERFETCH = 10  # extra results per search so filtering does not leave the agent short

# fetch(endpoint, params) -> response text
Fetch = Callable[[str, dict], str]


class CacheMiss(RuntimeError):
    """A search was not in the cache and the network was not allowed."""


@dataclass(frozen=True)
class Paper:
    pmid: str
    title: str
    abstract: str
    year: str = ""
    doi: str = ""
    journal: str = ""

    @property
    def source_id(self) -> str:
        return f"PMID:{self.pmid}"


# ---------- exclusion list ----------

@dataclass(frozen=True)
class Exclusions:
    pmids: frozenset[str]
    dois: frozenset[str]
    patterns: tuple[re.Pattern, ...]

    @classmethod
    def parse(cls, text: str) -> "Exclusions":
        pmids, dois, patterns = set(), set(), []
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            kind, sep, value = (s.strip() for s in line.partition(":"))
            if not sep or not value:
                raise ValueError(f"bad exclusion rule: {raw!r}")
            if kind == "pmid":
                pmids.add(value)
            elif kind == "doi":
                dois.add(value.lower())
            elif kind == "pattern":
                patterns.append(re.compile(value, re.IGNORECASE))
            else:
                raise ValueError(f"unknown exclusion kind {kind!r} in {raw!r}")
        return cls(frozenset(pmids), frozenset(dois), tuple(patterns))

    @classmethod
    def from_file(cls, path: Path = EXCLUSIONS) -> "Exclusions":
        return cls.parse(Path(path).read_text())

    def reason(self, paper: Paper) -> str | None:
        """Why a paper is excluded, or None if the agent may see it."""
        if paper.pmid in self.pmids:
            return f"pmid {paper.pmid}"
        if paper.doi and paper.doi.lower() in self.dois:
            return f"doi {paper.doi}"
        text = f"{paper.title} {paper.abstract}"
        for pattern in self.patterns:
            if pattern.search(text):
                return f"pattern {pattern.pattern}"
        return None


# ---------- PubMed (raw, cached) ----------

def parse_pubmed_xml(xml_text: str) -> list[Paper]:
    papers = []
    for art in ET.fromstring(xml_text).iter("PubmedArticle"):
        title_el = art.find(".//ArticleTitle")
        year = art.findtext(".//PubDate/Year") or (art.findtext(".//PubDate/MedlineDate") or "")[:4]
        doi = next((i.text or "" for i in art.iter("ArticleId") if i.get("IdType") == "doi"), "")
        papers.append(Paper(
            pmid=art.findtext(".//MedlineCitation/PMID", ""),
            title="".join(title_el.itertext()).strip() if title_el is not None else "",
            abstract=" ".join("".join(t.itertext()).strip() for t in art.iter("AbstractText")),
            year=year,
            doi=doi,
            journal=art.findtext(".//Journal/Title", ""),
        ))
    return papers


def http_fetch(endpoint: str, params: dict, tries: int = 3, delay: float = 0.34) -> str:
    """Real E-utilities call (NCBI allows 3 requests a second without a key)."""
    import requests  # imported here so nothing loads a network library unless it is used

    params = {**params, "db": "pubmed", "tool": "cancer-combo-agent"}
    for attempt in range(tries):
        time.sleep(delay * (attempt + 1))
        try:
            response = requests.get(f"{EUTILS}/{endpoint}", params=params, timeout=30)
            if response.ok:
                return response.text
        except requests.RequestException:
            pass
    raise RuntimeError(f"PubMed {endpoint} failed after {tries} tries")


class PubMed:
    def __init__(self, cache_dir: Path = CACHE_DIR, fetch: Fetch | None = None,
                 allow_network: bool = False):
        self.cache_dir = Path(cache_dir)
        self.fetch = fetch or http_fetch
        self.allow_network = allow_network

    def _path(self, query: str, retmax: int) -> Path:
        key = hashlib.sha1(f"{query}|{retmax}|relevance".encode()).hexdigest()[:16]
        return self.cache_dir / f"{key}.json"

    def search(self, query: str, retmax: int) -> list[Paper]:
        """Up to retmax papers, most relevant first. Unfiltered: use Literature for the agent."""
        path = self._path(query, retmax)
        if path.exists():
            return [Paper(**p) for p in json.loads(path.read_text())["papers"]]
        if not self.allow_network:
            raise CacheMiss(f"not cached: {query!r} (run the literature prefetch first)")
        # E-utilities default to newest first, which returns off-topic recent papers for
        # well-studied drugs; relevance order is far better.
        found = json.loads(self.fetch("esearch.fcgi", {
            "term": query, "retmax": retmax, "retmode": "json", "sort": "relevance"}))
        ids = found["esearchresult"]["idlist"]
        papers = parse_pubmed_xml(
            self.fetch("efetch.fcgi", {"id": ",".join(ids), "retmode": "xml"})) if ids else []
        by_id = {p.pmid: p for p in papers}
        papers = [by_id[i] for i in ids if i in by_id]  # keep the relevance order
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "query": query, "retmax": retmax, "fetched": time.strftime("%Y-%m-%d"),
            "papers": [asdict(p) for p in papers]}, indent=1))
        return papers


# ---------- what the agent sees ----------

def _title_key(title: str) -> str:
    return re.sub(r"\W+", " ", title.lower()).strip()


def drug_term(names: Iterable[str], field: str = "tiab") -> str:
    """PubMed clause matching any of a drug's names, e.g. ABT-888 or veliparib."""
    return "(" + " OR ".join(f'"{n}"[{field}]' for n in sorted(set(names))) + ")"


MECHANISM = '(mechanism[tiab] OR "mode of action"[tiab] OR inhibitor[tiab] OR target[tiab])'


class Literature:
    def __init__(self, pubmed: PubMed, exclusions: Exclusions, names: dict[str, list[str]]):
        """names: drug_id -> every name to search it by (the dataset ID and the generic name)."""
        self.pubmed, self.exclusions, self.names = pubmed, exclusions, names
        self.excluded_log: list[tuple[str, str]] = []  # (source_id, reason), for the run audit

    def search(self, query: str, n: int) -> list[Paper]:
        kept, titles = [], set()
        for paper in self.pubmed.search(query, n + OVERFETCH):
            reason = self.exclusions.reason(paper)
            if reason:
                self.excluded_log.append((paper.source_id, reason))
                continue
            # No abstract: nothing to reason from. Repeated title: another edition of a review.
            key = _title_key(paper.title)
            if not paper.abstract or key in titles:
                continue
            kept.append(paper)
            titles.add(key)
        return kept[:n]

    def _term(self, drug: str, field: str = "tiab") -> str:
        return drug_term(self.names.get(drug, [drug]), field)

    def for_drug(self, drug: str, n: int) -> list[Paper]:
        """Mechanism papers for one drug: title matches first, so a famous drug does not return
        passing mentions; then abstract matches, so a rarely named compound still gets some."""
        papers = self.search(f"{self._term(drug, 'ti')} AND {MECHANISM} AND cancer[tiab]", n)
        if len(papers) < n:
            seen = {p.pmid for p in papers}
            extra = self.search(f"{self._term(drug)} AND {MECHANISM}", n)
            papers += [p for p in extra if p.pmid not in seen][: n - len(papers)]
        return papers

    def for_pair(self, drug_a: str, drug_b: str, n: int) -> list[Paper]:
        """Papers naming both drugs."""
        return self.search(f"{self._term(drug_a)} AND {self._term(drug_b)}", n)


def drug_names(drugs_csv: Path = Path("data/reference/drugs.csv")) -> dict[str, list[str]]:
    import pandas as pd

    table = pd.read_csv(drugs_csv, dtype=str).fillna("")
    return {row.drug_id: sorted({row.drug_id, row.generic_name} - {""})
            for row in table.itertuples()}
