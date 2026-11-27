"""The agent end to end on a small synthetic lab, with a fake model and fake literature."""
import hashlib
import json
import re

import numpy as np
import pandas as pd
import pytest

from ccagent import agent as agent_module
from ccagent import baselines
from ccagent.agent import AgentChooser, AgentConfig
from ccagent.baselines import ModelChooser
from ccagent.hypotheses import MAX_SHOWN
from ccagent.lab import SimLab
from ccagent.literature import Exclusions, Literature, PubMed
from ccagent.llm import FakeLLMClient, top_k_responder
from ccagent.loop import run_selection
from ccagent.masking import Masker
from tests.helpers import synthetic_table

HELD = "ZZHELD"
BUDGET = 60


def table():
    t = synthetic_table(n_lines=6, n_drugs=8, seed=1)
    t["cell_line"] = t["cell_line"].replace({"L5": HELD})
    return t


class HashModel:
    """A deterministic stand-in for the forest: scores come from the exp_id, uncertainty
    shrinks as results accumulate, so the shortlist changes from round to round."""

    def __init__(self, *args, **kwargs):
        self.n = 0

    def fit(self, history, target):
        self.n = len(history)
        return self

    def predict(self, candidates):
        h = np.array([int(hashlib.md5(e.encode()).hexdigest()[:8], 16) for e in
                      candidates["exp_id"]], float)
        return (h % 1000) / 1000, ((h // 1000) % 1000) / 1000 / (1 + self.n / 10)


@pytest.fixture(autouse=True)
def fake_forest(monkeypatch):
    monkeypatch.setattr(agent_module, "ResponseModel", HashModel)
    monkeypatch.setattr(baselines, "ResponseModel", HashModel)


def fake_fetch(endpoint, params):
    """Any search returns three papers whose IDs depend on the query."""
    if endpoint == "esearch.fcgi":
        base = int(hashlib.md5(params["term"].encode()).hexdigest()[:6], 16)
        return json.dumps({"esearchresult": {"idlist": [str(base + i) for i in range(3)]}})
    articles = "".join(
        f"<PubmedArticle><MedlineCitation><PMID>{i}</PMID><Article><ArticleTitle>Paper {i} on "
        f"D1</ArticleTitle><Abstract><AbstractText>D1 inhibits growth.</AbstractText></Abstract>"
        f"</Article></MedlineCitation></PubmedArticle>" for i in params["id"].split(","))
    return f"<PubmedArticleSet>{articles}</PubmedArticleSet>"


def make_agent(tmp_path, ablation="none", client=None, masker=None):
    drugs = [f"D{i}" for i in range(8)]
    lit = None if ablation == "no_literature" else Literature(
        PubMed(tmp_path, fetch=fake_fetch, allow_network=True), Exclusions.parse(""),
        {d: [d] for d in drugs})
    if ablation == "masked_names" and masker is None:
        masker = Masker({d: [] for d in drugs})
    return AgentChooser(
        client=client or FakeLLMClient(responder=top_k_responder),
        annotations=pd.DataFrame({"drug_id": drugs, "drug_class": "kinase inhibitor",
                                  "targets": "KIN1"}),
        line_summaries={f"L{i}": f"line {i}; KRAS mutant" for i in range(5)} | {HELD: "secret"},
        heldout={HELD}, literature=lit, masker=masker or Masker(),
        config=AgentConfig(ablation=ablation))


def run(chooser, seed=0):
    return run_selection(SimLab(table(), [HELD], budget=BUDGET), chooser, seed=seed)


# ---------- the wiring check ----------

@pytest.mark.parametrize("ablation", ["none", "no_literature", "no_critique", "masked_names"])
def test_top_k_agent_reproduces_bayesian_optimization(tmp_path, ablation):
    for seed in (0, 1):
        bo = run(ModelChooser("bo"), seed)
        agent = run(make_agent(tmp_path, ablation), seed)
        assert list(agent["exp_id"]) == list(bo["exp_id"])


# ---------- what the model is shown ----------

def test_prompts_never_show_heldout_lines_or_outcome_labels(tmp_path):
    agent = make_agent(tmp_path)
    run(agent)
    texts = [agent.system] + [c["prompt"] for c in agent.calls]
    assert texts and all(HELD not in t and "secret" not in t for t in texts)
    assert all("strong" not in t.lower() for t in texts)


def test_literature_reaches_the_prompts_only_when_enabled(tmp_path):
    with_lit, without = make_agent(tmp_path), make_agent(tmp_path, "no_literature")
    run(with_lit)
    run(without)
    hyp = [c["prompt"] for c in with_lit.calls if c["step"] == "hypothesize"]
    assert all("[PMID:" in p for p in hyp)
    assert all("(no literature retrieved)" in c["prompt"] for c in without.calls
               if c["step"] == "hypothesize")
    assert with_lit.papers and not without.papers


def test_no_critique_ablation_makes_no_critique_calls(tmp_path):
    agent = make_agent(tmp_path, "no_critique")
    run(agent)
    assert {c["step"] for c in agent.calls} == {"hypothesize", "select"}


def test_masked_run_shows_no_drug_names(tmp_path):
    agent = make_agent(tmp_path, "masked_names")
    run(agent)
    for text in [agent.system] + [c["prompt"] for c in agent.calls]:
        assert not re.search(r"\bD[0-7]\b", text)
    assert all(d.startswith("D") for h in agent.book.items.values() for d in h.drugs)


def test_hypotheses_shown_stay_capped_over_the_run(tmp_path):
    agent = make_agent(tmp_path)
    run(agent)
    assert len(agent.book.items) >= 5  # one new hypothesis per round
    for call in agent.calls:
        section = call["prompt"].split("## Existing hypotheses\n")[-1].split("\n\n## ")[0]
        assert section.count('"id": "H') <= MAX_SHOWN


def test_every_pick_carries_an_explanation(tmp_path):
    log = run(make_agent(tmp_path))
    picked = log[log["round"] > 0]
    assert picked["hypothesis_id"].notna().all()
    assert set(picked["explanation_level"]) <= {"mechanistic", "cross_cancer", "statistical"}


def test_bad_model_answers_fall_back_to_bo_and_are_logged(tmp_path):
    agent = make_agent(tmp_path, client=FakeLLMClient(responder=lambda s, p: "no"))
    bo = run(ModelChooser("bo"))
    log = run(agent)
    assert list(log["exp_id"]) == list(bo["exp_id"])
    assert len(agent.calls) == 5 * 2 * 2  # 5 rounds x (hypothesize + select) x 2 attempts
    assert all(c["problems"] for c in agent.calls)


def test_ablation_settings_are_checked(tmp_path):
    with pytest.raises(ValueError):
        AgentConfig(ablation="vibes")
    with pytest.raises(ValueError):  # masked ablation without a masking Masker
        make_agent(tmp_path, "masked_names", masker=Masker())
    with pytest.raises(ValueError):  # a masking Masker outside the masked ablation
        make_agent(tmp_path, "none", masker=Masker({"D0": []}))
