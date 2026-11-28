import json

import pandas as pd

from ccagent.familiarity import FAMOUS_MIN, famous, famous_share, load, pair_group
from scripts.fetch_drug_familiarity import build


def test_cutoff_is_fixed_at_1000():
    assert FAMOUS_MIN == 1000  # decision log, Nov 27: fixed before any real run


def test_famous_is_at_least_the_cutoff():
    assert famous({"A": 1000, "B": 999, "C": 50000}) == {"A", "C"}


def test_pair_groups():
    f = {"A", "B"}
    assert [pair_group("A", "B", f), pair_group("A", "X", f), pair_group("X", "Y", f)] == \
        ["famous", "mixed", "obscure"]


def test_famous_share_counts_mentions():
    assert famous_share([["A", "X"], ["A"]], {"A"}) == 2 / 3


def test_build_counts_each_drug_with_the_literature_name_clause(tmp_path):
    asked = []

    def fake(endpoint, params):
        asked.append(params["term"])
        return json.dumps({"esearchresult": {"count": "1200" if "veliparib" in params["term"]
                                             else "40"}})
    table = build(fetch=fake, names={"ABT-888": ["ABT-888", "veliparib"], "MK-1": ["MK-1"]})
    assert asked == ['("ABT-888"[tiab] OR "veliparib"[tiab])', '("MK-1"[tiab])']
    assert table.set_index("drug_id")["famous"].to_dict() == {"ABT-888": True, "MK-1": False}
    table.to_csv(tmp_path / "f.csv", index=False)
    assert load(tmp_path / "f.csv") == {"ABT-888": 1200, "MK-1": 40}
    assert isinstance(table, pd.DataFrame)
