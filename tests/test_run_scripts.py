"""The run scripts never pay twice for a finished seed, and use the chosen key's spend log."""
import sys

import pytest

from ccagent import agent as agent_module
from ccagent.lab import SimLab
from scripts import run_agent
from tests.test_agent import BUDGET, HELD, HashModel, make_agent, table
from tests.test_sheldon_arm import truth


@pytest.fixture
def synthetic_run(monkeypatch, tmp_path):
    built = []
    monkeypatch.setattr(agent_module, "ResponseModel", HashModel)  # as in test_agent

    def build(client, ablation, seed, target):
        built.append(seed)
        return make_agent(tmp_path / f"lit{len(built)}", client=client)

    monkeypatch.setattr(run_agent, "build_agent", build)
    monkeypatch.setattr(run_agent, "visible_truth", lambda: truth(table()))
    monkeypatch.setattr(run_agent.SimLab, "from_files",
                        classmethod(lambda cls, **kw: SimLab(table(), [HELD], budget=BUDGET)))

    def main(*args):
        monkeypatch.setattr(sys, "argv", ["run_agent", "--out", str(tmp_path / "out"), *args])
        run_agent.main()
    return main, built, tmp_path / "out"


def test_finished_seeds_are_skipped_unless_overwritten(synthetic_run, capsys):
    main, built, out = synthetic_run
    main("--seeds", "0", "1")
    assert built == [0, 1]
    main("--seeds", "0", "1", "2")
    assert built == [0, 1, 2]  # 0 and 1 not run again
    assert capsys.readouterr().out.count("already finished") == 2
    main("--seeds", "0", "--overwrite")
    assert built == [0, 1, 2, 0]


def test_a_seed_missing_its_run_log_is_not_finished(synthetic_run):
    main, built, out = synthetic_run
    main("--seeds", "0")
    (csv,) = out.glob("*_seed0.csv")
    csv.unlink()  # as if the run had crashed before writing it
    main("--seeds", "0")
    assert built == [0, 0]


def test_key_flag_is_offered_and_checked():
    parser = run_agent.argparse.ArgumentParser()
    run_agent.add_llm_args(parser)
    assert parser.parse_args(["--key", "2"]).key == "2"
    assert parser.parse_args([]).key == "1"
    with pytest.raises(SystemExit):
        parser.parse_args(["--key", "3"])


def test_sheldon_skips_a_finished_seed(monkeypatch, tmp_path, capsys):
    pytest.importorskip("sheldon")
    from scripts import run_sheldon

    (tmp_path / "sheldon_joint_seed0.csv").write_text("exp_id\n")
    monkeypatch.setattr(sys, "argv", ["run_sheldon", "--seeds", "0", "--out", str(tmp_path)])
    monkeypatch.setattr(run_sheldon, "build_config",
                        lambda *a, **k: pytest.fail("a finished seed was started again"))
    run_sheldon.main()
    assert "already finished" in capsys.readouterr().out
