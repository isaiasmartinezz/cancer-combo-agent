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


# ---------- the held-out prediction after the run ----------

def mechanistic_responder(system, prompt):
    """top-k picks, but every new hypothesis is mechanistic and predicts responders, and the
    held-out step names line L0... which is visible, so the first answer must be refused."""
    out = json.loads(top_k_responder(system, prompt))
    if prompt.startswith("Propose up to"):
        for h in out["hypotheses"]:
            h["level"], h["predicted_responders"] = "mechanistic", ["L1"]
    if prompt.startswith("The run is over"):
        bad = "Your previous answer had these problems" not in prompt
        for p in out["predictions"]:
            p["predicted_responders"] = ["L0"] if bad else [HELD]
    return json.dumps(out)


def test_heldout_prediction_runs_once_after_the_loop_and_names_only_heldout_lines(tmp_path):
    agent = make_agent(tmp_path, client=FakeLLMClient(responder=mechanistic_responder))
    run(agent)
    before = len(agent.calls)
    preds = agent.predict_heldout()
    final = agent.calls[before:]
    assert [c["step"] for c in final] == ["predict_heldout"] * 2  # refused once, then fixed
    assert final[0]["problems"] and not final[1]["problems"]
    assert {p["hypothesis_id"] for p in preds} == {h.id for h in agent.book.active()}
    assert all(p["predicted_responders"] == [HELD] for p in preds)
    # the only prompt that names the held-out line, and it shows genetics, not results
    assert all(HELD not in c["prompt"] for c in agent.calls[:before])
    assert "secret" in final[0]["prompt"] and "synergy," not in final[0]["prompt"].split(
        "## Untested")[1]


def test_no_mechanistic_hypotheses_means_no_heldout_call(tmp_path):
    agent = make_agent(tmp_path)
    run(agent)
    n = len(agent.calls)
    assert agent.predict_heldout() == [] and len(agent.calls) == n


def test_masked_heldout_prompt_hides_drug_names(tmp_path):
    agent = make_agent(tmp_path, "masked_names",
                       client=FakeLLMClient(responder=mechanistic_responder))
    run(agent)
    agent.predict_heldout()
    final = [c["prompt"] for c in agent.calls if c["step"] == "predict_heldout"]
    assert final and not any(re.search(r"\bD[0-7]\b", p) for p in final)


# ---------- literature design (Nov 27) ----------

def test_evidence_rotates_across_the_shortlist(tmp_path):
    agent = make_agent(tmp_path)
    pairs = [(f"D{i}", f"D{i + 1}") for i in range(6)]
    short = pd.DataFrame({"drug_a": [a for a, _ in pairs], "drug_b": [b for _, b in pairs]})
    chosen = []
    for r in range(1, 5):
        agent.round = r
        chosen.append(agent.evidence_pairs(short))
    assert chosen[0] == pairs[:3] and chosen[1] == pairs[3:6]
    assert chosen[2] == pairs[:3]  # nothing fresh left: back to the best pairs
    assert chosen[3] == pairs[3:6]


def test_rotation_off_repeats_the_top_pairs(tmp_path):
    agent = make_agent(tmp_path)
    agent.config.rotate_evidence = False
    short = pd.DataFrame({"drug_a": ["D0", "D1", "D2", "D3"], "drug_b": ["D1", "D2", "D3", "D4"]})
    agent.round = 1
    first = agent.evidence_pairs(short)
    agent.round = 2
    assert agent.evidence_pairs(short) == first


def test_reading_note_only_with_literature(tmp_path):
    from ccagent.prompts import READING_NOTE

    with_lit, without = make_agent(tmp_path), make_agent(tmp_path, "no_literature")
    run(with_lit)
    run(without)
    lit_prompts = [c["prompt"] for c in with_lit.calls if c["step"] in ("hypothesize",
                                                                         "critique")]
    assert lit_prompts and all(READING_NOTE in p for p in lit_prompts)
    assert not any(READING_NOTE in c["prompt"] for c in without.calls)
