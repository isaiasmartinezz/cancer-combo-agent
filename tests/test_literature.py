import json

import pytest

from ccagent.literature import (
    EXCLUSIONS,
    MECHANISM,
    CacheMiss,
    Exclusions,
    Literature,
    Paper,
    PubMed,
    drug_term,
    parse_pubmed_xml,
)


def article(pmid, title, abstract="Some biology.", year="2012", doi=""):
    ids = f'<ArticleId IdType="doi">{doi}</ArticleId>' if doi else ""
    abstract_xml = (f"<Abstract><AbstractText>{abstract}</AbstractText></Abstract>"
                    if abstract else "")
    return (f"<PubmedArticle><MedlineCitation><PMID>{pmid}</PMID><Article>"
            f"<Journal><Title>J Test</Title><JournalIssue><PubDate><Year>{year}</Year></PubDate>"
            f"</JournalIssue></Journal><ArticleTitle>{title}</ArticleTitle>{abstract_xml}"
            f"</Article></MedlineCitation><PubmedData><ArticleIdList>{ids}</ArticleIdList>"
            f"</PubmedData></PubmedArticle>")


class FakeEutils:
    """Answers esearch/efetch from a dict of query -> articles, and records every call."""

    def __init__(self, results: dict[str, list[tuple]]):
        self.results, self.calls = results, []
        self.by_id = {a[0]: a for arts in results.values() for a in arts}

    def __call__(self, endpoint, params):
        self.calls.append((endpoint, params))
        if endpoint == "esearch.fcgi":
            ids = [a[0] for a in self.results.get(params["term"], [])][: params["retmax"]]
            return json.dumps({"esearchresult": {"idlist": ids}})
        # efetch returns articles in PMID order, not relevance order, like the real service
        ids = sorted(params["id"].split(","))
        return "<PubmedArticleSet>" + "".join(article(*self.by_id[i]) for i in ids) + \
            "</PubmedArticleSet>"


RULES = Exclusions.parse("""
# comment
pmid: 26983881
doi: 10.1158/1535-7163.MCT-15-0843
pattern: o['’]?neil
pattern: predict\\w*\\s+(\\S+\\s+){0,3}synerg
""")


def paper(pmid="1", title="Title", abstract="Abstract", doi=""):
    return Paper(pmid=pmid, title=title, abstract=abstract, doi=doi)


# ---------- exclusions ----------

def test_exclusions_by_pmid_doi_and_pattern():
    assert RULES.reason(paper(pmid="26983881")) == "pmid 26983881"
    assert RULES.reason(paper(doi="10.1158/1535-7163.mct-15-0843")).startswith("doi")
    assert RULES.reason(paper(abstract="data from O'Neil et al.")).startswith("pattern")
    assert RULES.reason(paper(title="Deep models predict drug synergy")).startswith("pattern")
    assert RULES.reason(paper(abstract="PARP inhibition sensitizes cells to platinum")) is None


def test_exclusion_patterns_stay_tight():
    # a biology paper about a gene "predictive of" response must survive
    text = "BRCA1 loss was predictive of response to the combination in xenografts"
    assert RULES.reason(paper(abstract=text)) is None


def test_bad_exclusion_rules_fail_loudly():
    with pytest.raises(ValueError):
        Exclusions.parse("pmd: 123")
    with pytest.raises(ValueError):
        Exclusions.parse("pattern:")


def test_shipped_exclusion_list_blocks_the_source_screen():
    rules = Exclusions.from_file(EXCLUSIONS)
    assert "26983881" in rules.pmids
    assert rules.reason(paper(abstract="We used the DrugComb portal")) is not None
    assert rules.reason(paper(title="Machine learning of drug synergy")) is not None


# ---------- PubMed parsing and cache ----------

def test_parse_pubmed_xml():
    xml = "<PubmedArticleSet>" + article("42", "A <i>title</i>", "Text.", "2010", "10.1/x") + \
        "</PubmedArticleSet>"
    (p,) = parse_pubmed_xml(xml)
    assert (p.pmid, p.title, p.abstract, p.year, p.doi) == ("42", "A title", "Text.", "2010",
                                                             "10.1/x")
    assert p.source_id == "PMID:42"


def test_pubmed_cache_is_offline_by_default(tmp_path):
    with pytest.raises(CacheMiss):
        PubMed(tmp_path, fetch=FakeEutils({})).search("anything", 5)


def test_pubmed_fetches_once_then_reads_cache(tmp_path):
    fake = FakeEutils({"q": [("3", "Third"), ("1", "First")]})
    online = PubMed(tmp_path, fetch=fake, allow_network=True)
    first = online.search("q", 5)
    assert [p.pmid for p in first] == ["3", "1"]  # relevance order kept, not efetch order
    assert len(fake.calls) == 2
    offline = PubMed(tmp_path, fetch=fake)  # no network: must come from the cache
    assert offline.search("q", 5) == first
    assert len(fake.calls) == 2


# ---------- what the agent sees ----------

def literature(tmp_path, results):
    pubmed = PubMed(tmp_path, fetch=FakeEutils(results), allow_network=True)
    return Literature(pubmed, RULES, {"ABT-888": ["ABT-888", "veliparib"]})


def test_search_filters_excluded_abstractless_and_repeated(tmp_path):
    lit = literature(tmp_path, {"q": [
        ("1", "Keep me"),
        ("26983881", "The screen itself"),
        ("2", "No abstract", ""),
        ("3", "Keep  me!"),  # same title, different punctuation: another edition
        ("4", "Also kept", "Data reused from O'Neil 2016"),
        ("5", "Second keeper"),
    ]})
    assert [p.pmid for p in lit.search("q", 10)] == ["1", "5"]
    assert {sid for sid, _ in lit.excluded_log} == {"PMID:26983881", "PMID:4"}


def test_search_respects_n(tmp_path):
    lit = literature(tmp_path, {"q": [(str(i), f"Paper {i}") for i in range(1, 8)]})
    assert [p.pmid for p in lit.search("q", 3)] == ["1", "2", "3"]


def test_drug_term_uses_every_name():
    assert drug_term(["veliparib", "ABT-888"], "ti") == '("ABT-888"[ti] OR "veliparib"[ti])'


def test_for_drug_tops_up_from_abstract_matches(tmp_path):
    names = drug_term(["ABT-888", "veliparib"], "ti")
    title_q = f"{names} AND {MECHANISM} AND cancer[tiab]"
    abstract_q = f"{drug_term(['ABT-888', 'veliparib'])} AND {MECHANISM}"
    lit = literature(tmp_path, {title_q: [("1", "Veliparib title")],
                                abstract_q: [("1", "Veliparib title"), ("2", "In abstract"),
                                             ("3", "Another")]})
    assert [p.pmid for p in lit.for_drug("ABT-888", 2)] == ["1", "2"]


def test_for_pair_queries_both_drugs(tmp_path):
    q = f"{drug_term(['ABT-888', 'veliparib'])} AND {drug_term(['CARBOPLATIN'])}"
    lit = literature(tmp_path, {q: [("9", "Veliparib with carboplatin")]})
    assert [p.pmid for p in lit.for_pair("ABT-888", "CARBOPLATIN", 3)] == ["9"]


def test_literature_never_touches_network_in_a_run(tmp_path):
    lit = Literature(PubMed(tmp_path), RULES, {})
    with pytest.raises(CacheMiss):
        lit.for_pair("A", "B", 3)
