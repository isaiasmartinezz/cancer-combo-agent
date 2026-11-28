import pandas as pd
import pytest

from ccagent.hypotheses import HypothesisBook
from ccagent.literature import Paper
from ccagent.masking import Masker
from ccagent.prompts import (
    ABSTRACT_CHARS,
    SYSTEM,
    LeakError,
    check_inputs,
    critique_prompt,
    fmt_literature,
    fmt_results,
    heldout_prompt,
    hypothesize_prompt,
    reference_block,
    select_prompt,
)

NAMES = {"ABT-888": ["veliparib", "Veliparib"], "CARBOPLATIN": ["carboplatin"],
         "SN-38": ["SN-38", "7-ethyl-10-hydroxycamptothecin"]}
SENTINEL = 987.654  # a value that exists only in the hidden columns


def shortlist():
    return pd.DataFrame({
        "exp_id": ["E1", "E2"], "drug_a": ["ABT-888", "CARBOPLATIN"],
        "drug_b": ["CARBOPLATIN", "SN-38"], "cell_line": ["L1", "L2"],
        "tissue": ["ovary", "lung"], "pred_mean": [0.4, 0.1], "pred_std": [0.2, 0.3]})


def history_with_traps():
    """A results table carrying columns the agent must never see, with a sentinel value."""
    return pd.DataFrame({
        "drug_a": ["ABT-888", "CARBOPLATIN", "ABT-888"],
        "drug_b": ["CARBOPLATIN", "SN-38", "SN-38"],
        "cell_line": ["L1", "L2", "L3"], "synergy": [5.0, -2.0, 1.0],
        "inhibition": [40.0, 10.0, 20.0], "joint": [0.9, -0.5, 0.1],
        "strong_joint_within_line": [True, False, False], "is_strong_joint": [True, False, False],
        "synergy_cutoff": [SENTINEL] * 3, "hidden_outcome": [SENTINEL] * 3})


def all_prompts(masker=None):
    masker = masker or Masker()
    book = HypothesisBook()
    new = book.add([{"statement": "veliparib plus carboplatin works in BRCA lines",
                     "drugs": ["ABT-888", "CARBOPLATIN"], "level": "mechanistic",
                     "predicted_responders": ["L1"], "responder_feature": "BRCA1 loss",
                     "rationale": "PARP trapping",
                     "evidence_ids": ["PMID:1"]}], 1)
    lit = {"ABT-888 + CARBOPLATIN": [Paper("1", "Veliparib and carboplatin",
                                           "Veliparib potentiated carboplatin in BRCA1 cells.")]}
    hist, short = history_with_traps(), shortlist()
    return [
        hypothesize_prompt(short, hist, "joint score", "joint", new, ["H9 (refuted): x"], lit,
                           masker, 6, 14),
        critique_prompt(new, hist, short, "joint", lit, masker),
        select_prompt(short, hist, "joint", new, 2, masker),
    ]


# ---------- leakage ----------

def test_answer_formats_show_single_braces():
    """Templates escape JSON braces as {{ }}; every one must go through .format()."""
    held = heldout_prompt([], {"X": "genetics unknown"}, "joint score", Masker())
    for prompt in [*all_prompts(), held]:
        assert "{{" not in prompt and "}}" not in prompt
        assert '{"' in prompt  # the answer format is still there


def test_no_prompt_contains_strong_flags_cutoffs_or_hidden_values():
    for prompt in [SYSTEM, *all_prompts()]:
        assert "strong" not in prompt.lower()
        assert "cutoff" not in prompt.lower()
        assert "hidden" not in prompt.lower()
        assert str(SENTINEL) not in prompt


def test_results_show_only_returned_columns_and_the_target():
    text = fmt_results(history_with_traps(), shortlist(), "joint", Masker())
    header = text.splitlines()[2]
    assert header == "drug_a,drug_b,cell_line,synergy,inhibition,joint"


def test_heldout_lines_refused_in_shortlist_or_results():
    check_inputs({"HELD1"}, shortlist(), history_with_traps())  # fine
    with pytest.raises(LeakError):
        check_inputs({"L2"}, shortlist())
    with pytest.raises(LeakError):
        check_inputs({"L3"}, shortlist(), history_with_traps())


def test_reference_block_lists_only_visible_lines():
    ann = pd.DataFrame({"drug_id": ["ABT-888"], "drug_class": ["PARP inhibitor"],
                        "targets": ["PARP1"]})
    block = reference_block(ann, {"L1": "ovarian; BRCA1 mut", "HELD1": "secret line"},
                            {"L1"}, Masker())
    assert "L1: ovarian; BRCA1 mut" in block
    assert "HELD1" not in block and "secret" not in block


def test_abstracts_show_their_end_from_a_sentence_start():
    background = "Background words here. " * 40
    finding = "We found the pair synergistic in KRAS mutant cells."
    text = fmt_literature({"x": [Paper("1", "T", background + finding)]}, Masker())
    shown = text.rsplit("\n", 1)[1]
    assert shown.endswith(finding) and shown.startswith("...Background")
    assert len(shown) <= ABSTRACT_CHARS + 3
    short = fmt_literature({"x": [Paper("1", "T", finding)]}, Masker())
    assert short.endswith(finding) and "..." not in short


def test_result_numbers_are_rounded_but_ranked_at_full_precision():
    history = pd.DataFrame({"drug_a": ["A", "A"], "drug_b": ["B", "C"], "cell_line": ["L1"] * 2,
                            "synergy": [12.874675113616199, 12.8744], "inhibition": [80.0, 70.0],
                            "joint": [0.12345, 0.12341]})
    text = fmt_results(history, history, "joint", Masker())
    assert "12.875" in text and "12.8746" not in text
    assert text.index("A,B,L1") < text.index("A,C,L1")  # the higher one is still first


# ---------- masked names ----------

def test_masking_hides_every_name_in_every_prompt():
    masker = Masker(NAMES)
    for prompt in all_prompts(masker):
        lowered = prompt.lower()
        for drug, names in NAMES.items():
            for name in [drug, *names]:
                assert name.lower() not in lowered, (name, prompt[:200])
    assert masker.name("ABT-888").startswith("Drug ")


def test_masking_round_trips_and_respects_word_edges():
    masker = Masker(NAMES)
    code = masker.name("SN-38")
    assert masker.unname(code) == "SN-38"
    assert masker.text("SN-38 and 7-ethyl-10-hydroxycamptothecin") == f"{code} and {code}"
    assert masker.text("PSN-389 is unrelated") == "PSN-389 is unrelated"
    vel = masker.name("ABT-888")
    assert masker.text("veliparib-induced and Veliparib-loaded") == \
        f"{vel}-induced and {vel}-loaded"


def test_mask_codes_are_not_alphabetical():
    names = {f"D{i:02d}": [] for i in range(20)}
    masker = Masker(names)
    assert [masker.name(d) for d in sorted(names)] != [f"Drug {i:02d}" for i in range(1, 21)]


def test_masker_from_tables_collects_every_spelling():
    drugs = pd.DataFrame({"drug_id": ["AZD1775"], "drugcomb_name": ["MK-1775"],
                          "generic_name": ["adavosertib"]})
    ann = pd.DataFrame({"drug_id": ["AZD1775"], "search_name": ["adavosertib"],
                        "chembl_name": ["ADAVOSERTIB"]})
    masker = Masker.from_tables(drugs, ann)
    assert "MK-1775" not in masker.text("MK-1775 (adavosertib, AZD1775)")


def test_identity_masker_changes_nothing():
    assert all_prompts(Masker())[0].count("ABT-888") > 0
