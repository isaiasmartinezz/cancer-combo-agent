import pytest

from ccagent.lab import LabError, SimLab
from tests.helpers import synthetic_table

TABLE = synthetic_table()  # 4 lines x 15 pairs = 60 experiments


def make_lab(budget=20):
    return SimLab(TABLE, heldout=["L3"], budget=budget)


def test_heldout_lines_are_invisible():
    lab = make_lab()
    assert "L3" not in set(lab.candidates().cell_line)
    heldout_id = TABLE[TABLE.cell_line == "L3"].exp_id.iloc[0]
    with pytest.raises(LabError):
        lab.run([heldout_id])


def test_candidates_show_design_only():
    assert list(make_lab().candidates().columns) == ["exp_id", "drug_a", "drug_b", "cell_line"]


def test_run_returns_only_the_two_outcomes():
    lab = make_lab()
    out = lab.run(lab.candidates().exp_id[:3])
    assert list(out.columns) == ["exp_id", "synergy", "inhibition"]
    expected = TABLE.set_index("exp_id").loc[out.exp_id, "synergy_loewe"].to_numpy()
    assert (out.synergy.to_numpy() == expected).all()


def test_no_repeats_across_or_within_batches():
    lab = make_lab()
    first = list(lab.candidates().exp_id[:2])
    lab.run(first)
    assert not set(first) & set(lab.candidates().exp_id)
    with pytest.raises(LabError):
        lab.run(first[:1])
    with pytest.raises(LabError):
        lab.run([lab.candidates().exp_id[0]] * 2)


def test_budget_cannot_be_exceeded():
    lab = make_lab(budget=5)
    lab.run(lab.candidates().exp_id[:5])
    assert lab.budget_remaining() == 0
    with pytest.raises(LabError):
        lab.run(lab.candidates().exp_id[:1])


def test_failed_batch_changes_nothing():
    lab = make_lab(budget=5)
    ids = list(lab.candidates().exp_id[:6])
    with pytest.raises(LabError):
        lab.run(ids)
    assert lab.budget_remaining() == 5 and lab.batches == []


def test_unknown_heldout_line_is_an_error():
    with pytest.raises(LabError):
        SimLab(TABLE, heldout=["nope"])


def test_lab_has_no_public_way_to_reveal_outcomes():
    public = {name for name in dir(make_lab()) if not name.startswith("_")}
    assert public == {"batches", "budget_remaining", "candidates", "from_files", "run"}
