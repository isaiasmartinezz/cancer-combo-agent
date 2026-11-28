"""SHELDON on a small synthetic lab: its tools, its refusals, and one full offline run."""
import asyncio
import json

import pandas as pd
import pytest

pytest.importorskip("sheldon")

from ccagent import baselines, sheldon_arm  # noqa: E402
from ccagent.baselines import ModelChooser  # noqa: E402
from ccagent.lab import SimLab  # noqa: E402
from ccagent.literature import Exclusions, Literature, PubMed  # noqa: E402
from ccagent.llm.guard import LLMNotAllowed  # noqa: E402
from ccagent.llm.sheldon_client import EFFORT, MODEL, SheldonLLM, make_llm  # noqa: E402
from ccagent.loop import run_selection  # noqa: E402
from ccagent.prompts import LeakError  # noqa: E402
from ccagent.sheldon_arm import OneilCombo, scripted_bo_responder  # noqa: E402
from scripts.run_sheldon import build_config  # noqa: E402
from tests.test_agent import HELD, HashModel, fake_fetch, fetch_naming_heldout, table  # noqa: E402,E501

BUDGET = 60


def truth(t):
    t = t[t["cell_line"] != HELD]
    return t.rename(columns={"synergy_loewe": "synergy"}).set_index("exp_id")[
        ["drug_a", "drug_b", "cell_line", "synergy", "inhibition"]]


@pytest.fixture(autouse=True)
def synthetic(monkeypatch, tmp_path):
    t = table()
    drugs = sorted(set(t["drug_a"]) | set(t["drug_b"]))
    lit = Literature(PubMed(tmp_path, fetch=fake_fetch, allow_network=True), Exclusions.parse(""),
                     {d: [d] for d in drugs})
    ctx = sheldon_arm.Context(
        table=t, heldout=[HELD], truth=truth(t),
        annotations=pd.DataFrame({"drug_id": drugs, "drug_class": "kinase inhibitor",
                                  "targets": "KIN1"}),
        line_summaries={f"L{i}": f"line {i}; KRAS mutant" for i in range(5)} | {HELD: "secret"},
        literature=lit, model_factory=HashModel)
    monkeypatch.setattr(sheldon_arm, "make_context", lambda literature=True: ctx)
    monkeypatch.setattr(baselines, "ResponseModel", HashModel)
    monkeypatch.setenv("SHELDON_RUNS", str(tmp_path / "runs"))
    return ctx


def bench(**opts):
    b = OneilCombo(seeds=[0], budget=BUDGET, **opts)
    task = b.load()[0]
    return b, task, {p.name: p for p in b.probes(task)}


def call(probes, name, **kw):
    return asyncio.run(probes[name].run(**kw))


def shortlist_ids(result):
    rows = result.summary.split(sheldon_arm.SHORTLIST_HEADER)[1].strip().splitlines()[1:]
    return [r.split(",")[0] for r in rows]


# ---------- the wiring check ----------

def test_running_the_shortlist_top_10_reproduces_bayesian_optimization():
    b, task, probes = bench()
    while not b.objective_exhausted(task):
        ids = shortlist_ids(call(probes, "get_shortlist"))[:10]
        assert call(probes, "run_experiments", exp_ids=ids).ok
    bo = run_selection(SimLab(table(), [HELD], budget=BUDGET), ModelChooser("bo"), seed=0)
    assert list(b.run_log(task.id)["exp_id"]) == list(bo["exp_id"])


def test_the_shared_first_batch_is_run_before_sheldon_starts():
    b, task, _ = bench()
    bo = run_selection(SimLab(table(), [HELD], budget=BUDGET), ModelChooser("bo"), seed=0)
    first = b.run_log(task.id)
    assert list(first["exp_id"]) == list(bo[bo["round"] == 0]["exp_id"])
    shown = task.question.split("## Results already returned\n")[1]
    for row in first.itertuples():
        assert f"{row.drug_a},{row.drug_b},{row.cell_line}," in shown
    assert f"so {BUDGET - 10} remain" in task.question


# ---------- refusals ----------

def test_lab_refuses_bad_batches_without_spending_budget():
    b, task, probes = bench()
    left = b.trials[task.id].lab.budget_remaining()
    assert not call(probes, "run_experiments", exp_ids=["E00001"]).ok  # no shortlist yet
    ids = shortlist_ids(call(probes, "get_shortlist"))
    off = sorted(set(table()["exp_id"]) - set(ids) - set(b.run_log(task.id)["exp_id"]))
    assert not call(probes, "run_experiments", exp_ids=ids[:11]).ok   # too many
    assert not call(probes, "run_experiments", exp_ids=[off[0]]).ok   # off the shortlist
    assert not call(probes, "run_experiments", exp_ids=[ids[0], ids[0]]).ok  # repeated
    assert not call(probes, "run_experiments", exp_ids=[]).ok
    assert b.trials[task.id].lab.budget_remaining() == left
    assert call(probes, "run_experiments", exp_ids=ids[:3]).ok
    assert not call(probes, "run_experiments", exp_ids=ids[3:5]).ok  # shortlist is stale now
    assert len(b.trials[task.id].refused) == 6


def test_heldout_lines_cannot_be_run_or_seen():
    b, task, probes = bench(shortlist_only=False)
    held = table()[table()["cell_line"] == HELD]["exp_id"].iloc[0]
    assert not call(probes, "run_experiments", exp_ids=[held]).ok
    assert "not a cell line" in call(probes, "cell_line_info", cell_lines=[HELD]).summary
    texts = [task.question, call(probes, "get_shortlist").summary,
             call(probes, "results_so_far").summary,
             call(probes, "cell_line_info", cell_lines=["L1", "L2"]).summary,
             call(probes, "drug_literature", drug="D1").summary,
             call(probes, "pair_literature", drug_a="D1", drug_b="D2").summary]
    for text in texts:
        assert HELD not in text and "secret" not in text
        assert "strong_" not in text and "cutoff" not in text.lower()


@pytest.mark.parametrize("hide", [True, False])
def test_literature_tools_redact_heldout_lines_or_refuse(synthetic, monkeypatch, tmp_path, hide):
    drugs = sorted(set(table()["drug_a"]) | set(table()["drug_b"]))
    lit = Literature(PubMed(tmp_path / "naming", fetch=fetch_naming_heldout, allow_network=True),
                     Exclusions.parse(""), {d: [d] for d in drugs},
                     hide_lines=[HELD] if hide else ())
    monkeypatch.setattr(synthetic, "literature", lit)
    _, _, probes = bench()
    for name, args in (("drug_literature", {"drug": "D1"}),
                       ("pair_literature", {"drug_a": "D1", "drug_b": "D2"})):
        if hide:
            text = call(probes, name, **args).summary
            assert HELD not in text and "[another cell line]" in text
        else:
            with pytest.raises(LeakError):
                call(probes, name, **args)


def test_literature_tools_only_when_enabled():
    _, _, with_lit = bench()
    _, task, without = bench(literature=False)
    assert {"drug_literature", "pair_literature"} <= set(with_lit)
    assert not {"drug_literature", "pair_literature"} & set(without)
    for name, args in (("drug_literature", {"drug": "D1"}),
                       ("pair_literature", {"drug_a": "D1", "drug_b": "D2"})):
        text = call(with_lit, name, **args).summary
        assert "[PMID:" in text and text.startswith("How to read the literature")
    assert "literature" not in task.question


# ---------- the full loop, offline ----------

def test_full_sheldon_loop_runs_offline_and_is_scored(tmp_path):
    from sheldon.runner import run_experiment

    cfg = build_config(0, 5.0, 4, True, tmp_path / "logs")
    cfg.params["benchmark_opts"]["budget"] = BUDGET
    llm = make_llm("fake", tool_responder=scripted_bo_responder)
    summary = asyncio.run(run_experiment(cfg, llm=llm))
    assert summary["errors"] == 0
    log = pd.read_csv(tmp_path / "logs" / "sheldon_joint_seed0.csv")
    assert 10 < len(log) <= BUDGET
    assert summary["metrics"]["experiments_run"] == len(log)
    assert summary["score"] == log["strong_joint_within_line"].sum()
    seen = "\n".join(llm.prompts + llm.systems)
    assert HELD not in seen and "secret" not in seen
    assert not any(n in seen for n in ("web_search", "web_fetch", "run_shell"))


def test_config_offers_only_the_benchmarks_tools(tmp_path):
    cfg = build_config(3, 4.0, 40, False, tmp_path)
    assert cfg.params["probes"] == [] and cfg.model == MODEL and cfg.effort == EFFORT
    assert cfg.params["benchmark_opts"] == {"seeds": [3], "literature": False, "target": "joint",
                                            "log_dir": str(tmp_path)}
    json.dumps(cfg.to_dict(), default=str)


# ---------- the model client ----------

def test_requests_carry_adaptive_thinking_and_our_effort():
    llm = SheldonLLM(model=MODEL, effort=EFFORT)
    req = llm._request("hi", "sys", MODEL, EFFORT, 1000, True, True)
    assert req["thinking"] == {"type": "adaptive"}
    assert req["output_config"] == {"effort": EFFORT}


def test_real_sheldon_client_is_gated():
    with pytest.raises(LLMNotAllowed):
        make_llm("real", allow_llm=False, env={"LLM_API_KEY": "k"})
    with pytest.raises(LLMNotAllowed):
        make_llm("real", allow_llm=True, env={})


def test_sheldon_prices_our_model_at_list_price():
    import sheldon.core.llm as sllm

    make_llm("fake")
    assert sllm.PRICING[MODEL] == pytest.approx((2.0, 10.0))
