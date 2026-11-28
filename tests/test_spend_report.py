import json

from scripts.spend_report import ESTIMATE, report


def write(path, entries):
    path.write_text("".join(json.dumps(e) + "\n" for e in entries))


def test_measured_cost_replaces_the_estimate_and_projects_the_rest(tmp_path):
    k1, k2 = tmp_path / "k1.jsonl", tmp_path / "k2.jsonl"
    write(k1, [{"run_id": "agent_real_none_joint_seed0", "cost_usd": 1.5},
               {"run_id": "agent_real_none_joint_seed0", "cost_usd": 1.5}])  # one seed, $3
    plan = [("1", "agent_real_none_joint", range(0, 3)),
            ("2", "agent_real_no_literature_joint", range(0, 2))]
    text = report({"1": k1, "2": k2}, cap=10.0, plan=plan)
    assert "key 1: spent $3.00 of $10.00; plan left 2 seeds ~$6.00 -> ~$9.00 (fits)" in text
    est = 2 * ESTIMATE["agent_real_no_literature_joint"]
    assert f"key 2: spent $0.00 of $10.00; plan left 2 seeds ~${est:.2f}" in text
    assert "agent_real_none_joint: 1, $3.00" in text


def test_over_the_cap_is_flagged(tmp_path):
    k1 = tmp_path / "k1.jsonl"
    write(k1, [{"run_id": "agent_real_none_joint_seed0", "cost_usd": 4.0}])
    text = report({"1": k1}, cap=10.0, plan=[("1", "agent_real_none_joint", range(0, 3))])
    assert "OVER the cap" in text


def test_practice_seed_checks_count_as_spend_but_not_as_a_seed_rate(tmp_path):
    k1 = tmp_path / "k1.jsonl"
    write(k1, [{"run_id": "sheldon_joint_seed100", "cost_usd": 1.0}])
    text = report({"1": k1}, cap=50.0, plan=[("1", "sheldon_joint", range(0, 1))])
    assert "spent $1.00" in text and "sheldon_joint: 0, -" in text
    assert f"~${ESTIMATE['sheldon_joint']:.2f}" in text  # the estimate still prices seed 0


def test_each_key_is_capped_at_48_dollars():
    from ccagent.llm import settings
    from ccagent.llm.real import SpendLedger
    assert settings.BUDGET_USD == 48.0 and len(settings.KEYS) == 2
    assert SpendLedger("unused.jsonl").budget_usd == 48.0
