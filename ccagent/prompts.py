"""The text the agent's language model sees: one fixed reference block, and three task prompts.

Leakage rules, each enforced here and locked by a test:
- Results come only from the run history, and only the columns a lab returns (plus the method's
  own target computed from them). Strong flags, cutoffs and hidden outcomes are never formatted,
  even if a caller passes a table that contains them.
- No held-out cell line may appear in the shortlist, the results, or the reference block.
  `check_inputs` refuses to build a prompt otherwise. (A held-out line can still be named inside a
  retrieved abstract; that is published literature, not a result.)
- Held-out lines appear in exactly one prompt: the prediction made after the run is over
  (`heldout_prompt`), which shows their names and genetics, never a result.
- Every drug name passes through the masker, which is the identity unless the masked-name
  ablation is on.
"""
import json

import pandas as pd

from ccagent.hypotheses import Hypothesis, max_responders
from ccagent.literature import Paper, line_pattern
from ccagent.masking import Masker

# The end of an abstract, where its results and conclusion usually are (decision log, Nov 27):
# finding-type wording is in the last 800 characters of 45% of cached abstracts, the first 400
# of 12%, and anywhere in 55%. The title already says what the paper is about.
ABSTRACT_CHARS = 800
DECIMALS = 3  # numbers in result tables; replicate noise is far larger than 0.001
N_BEST, N_WORST, MAX_PAIRS = 8, 4, 12
SHORTLIST_COLS = ["exp_id", "drug_a", "drug_b", "cell_line", "tissue", "pred_mean", "pred_std"]
RESULT_COLS = ["drug_a", "drug_b", "cell_line", "synergy", "inhibition"]


class LeakError(RuntimeError):
    """A prompt would have shown something the agent must never see."""


SYSTEM = """You help a cancer screening lab decide which drug-combination experiments to run \
next. Each experiment tests one drug pair in one cancer cell line. The lab reports two numbers: \
a synergy score (higher means the drugs work better together than either alone predicts) and a \
growth-inhibition score.

Base every claim on the material given to you: the reference tables, the literature excerpts, \
and the results the lab has returned. Cite literature only by the IDs shown with it (for \
example PMID:12345), and never cite an ID that was not shown. When a claim rests on general \
pharmacology rather than a provided source, say so and leave it uncited. Prefer specific, \
testable mechanisms, and say plainly when the evidence is thin.

Be brief: one sentence per hypothesis statement, reason, or verdict explanation; at most two \
sentences per rationale. Answer with JSON only, in the shape the task describes."""

HYPOTHESIZE = """Propose up to {max_new} new hypotheses about which drug pairs on the shortlist \
will score high on the target: {target}.

Each hypothesis is a JSON object with:
- "statement": one sentence
- "drugs": one or two drug names, written exactly as on the shortlist
- "level": "mechanistic" (a specific pathway in these cells), "cross_cancer" (shown in other \
tumor types), or "statistical" (similar drugs did this in results so far)
- "predicted_responders": for mechanistic hypotheses, the cell lines (any in the reference \
list) that have the responder feature, at most {max_responders}; empty for other levels
- "responder_feature": for mechanistic hypotheses, the genetic or tissue feature the responders \
share, in a few words (for example "BRAF V600E" or "ovarian tissue"); a line whose genetics are \
unknown fits only a tissue feature; "" for other levels
- "rationale": at most two sentences
- "evidence_ids": IDs of the literature excerpts it rests on (may be empty)
Do not repeat an existing or retired idea. Answer as {{"hypotheses": [...]}}."""

CRITIQUE = """Challenge each hypothesis below against the literature and the results so far. \
Look for claims the cited sources do not support, results that contradict it, and mechanisms \
that do not apply to the predicted cell lines, including responders that lack the stated \
responder feature. Give exactly one verdict per hypothesis:
{{"verdicts": [{{"id": "H1", "verdict": "keep" | "revise" | "drop", "revised_statement": \
"(only for revise)", "reason": "one sentence", "evidence_ids": []}}]}}"""

SELECT = """Choose exactly {k} experiments from the shortlist, by exp_id, for the next batch. \
Balance testing the most promising hypotheses against experiments that would tell competing \
hypotheses apart; a few "explore" picks for untested drug classes are fine. Each pick names the \
hypothesis it tests (or "explore"), an explanation level, and a one-sentence reason:
{{"picks": [{{"exp_id": "E00001", "hypothesis_id": "H1" | "explore", "level": "mechanistic" | \
"cross_cancer" | "statistical", "reason": "..."}}]}}"""


PREDICT_HELDOUT = """The run is over and no more experiments will be chosen. Below are cell \
lines this run never tested, with their genetics only (no results exist for you), and your \
mechanistic hypotheses. For each hypothesis, name the untested lines that have its \
responder feature, so the mechanism should make the pair score high on the target: {target}. \
Use only the mechanism and each line's genetics and tissue; name at most {max_responders}, and \
an empty list is right when no line fits. One prediction per hypothesis:
{{"predictions": [{{"hypothesis_id": "H1", "predicted_responders": ["LINE"], "reason": \
"one sentence"}}]}}"""


# Shown with the literature only (the no-literature arm's prompts are unchanged). In iteration 1,
# hypotheses with literature named famous drugs 63% of the time, against 36% without it.
READING_NOTE = """How to read the literature: how much has been published about a drug \
reflects how much it has been studied, not how well it combines. Research compounds with few \
papers are as likely to score high in this screen as famous drugs. Papers describe other cells \
and conditions; when this lab's results disagree with a paper, go with the results. Class \
evidence is about other members of a drug's class, not the compound itself."""


# ---------- leakage check ----------

def check_papers(heldout: set[str], papers) -> None:
    check_text(heldout, *(f"{p.title} {p.abstract}" for p in papers))


def check_inputs(heldout: set[str], *tables: pd.DataFrame | None) -> None:
    for table in tables:
        if table is None or len(table) == 0 or "cell_line" not in table:
            continue
        leaked = sorted(set(table["cell_line"]) & set(heldout))
        if leaked:
            raise LeakError(f"held-out cell lines would reach the prompt: {leaked}")


def check_text(heldout: set[str], *texts: str) -> None:
    """The text guard for what we supply: no held-out line name, in any spelling, in paper
    text. Text the model wrote itself is not checked (it may name a well-known line from its own
    training, which is not a leak of ours); the held-out prediction prompt names them on purpose."""
    pattern = line_pattern(heldout)
    for text in texts:
        if pattern is not None and (found := pattern.search(text)):
            raise LeakError(f"a held-out cell line would reach the prompt: {found.group(0)!r}")


# ---------- reference block (fixed for a run) ----------

def reference_block(annotations: pd.DataFrame, line_summaries: dict[str, str],
                    visible_lines: set[str], masker: Masker) -> str:
    """Every drug's class and targets, and the genetics of the visible cell lines only."""
    drugs = [f"- {masker.name(r.drug_id)}: {r.drug_class}; targets: {r.targets}"
             for r in annotations.fillna("unknown").sort_values("drug_id").itertuples()]
    if masker.enabled:
        drugs.sort()
    lines = [f"- {line}: {line_summaries.get(line, 'genetics unknown')}"
             for line in sorted(visible_lines)]
    return masker.text("## Drugs\n" + "\n".join(drugs)) + \
        "\n\n## Cell lines (genetics; results only where the lab has returned them)\n" + \
        "\n".join(lines)


# ---------- sections ----------

def excerpt(abstract: str, n: int = ABSTRACT_CHARS) -> str:
    """The last n characters, starting at a sentence (or else a word) so no word is cut."""
    if len(abstract) <= n:
        return abstract
    tail = abstract[-n:]
    start = tail.find(". ", 0, n // 4)
    start = start + 2 if start >= 0 else tail.find(" ") + 1
    return "..." + tail[start:]


def rounded(table: pd.DataFrame) -> pd.DataFrame:
    """Result tables as shown to a model: scores to DECIMALS places, nothing else changed."""
    out = table.copy()
    for col in ("synergy", "inhibition", "joint"):
        if col in out:
            out[col] = pd.to_numeric(out[col]).round(DECIMALS)
    return out


def fmt_shortlist(shortlist: pd.DataFrame, masker: Masker) -> str:
    cols = [c for c in SHORTLIST_COLS if c in shortlist.columns]
    return masker.table(shortlist[cols]).to_csv(index=False)


def fmt_results(history: pd.DataFrame | None, shortlist: pd.DataFrame, target: str,
                masker: Masker) -> str:
    """A compact summary of returned results: best, worst, and per-pair summaries for drugs on
    the shortlist. Only returned columns and the target are read; nothing else can leak."""
    if history is None or len(history) == 0:
        return "(no experiments run yet)"
    cols = RESULT_COLS + ([target] if target not in RESULT_COLS else [])
    h = history[cols].copy()
    h[target] = pd.to_numeric(h[target])
    h["pct_in_line"] = h.groupby("cell_line")[target].rank(pct=True).round(2)
    ranked = masker.table(rounded(h.sort_values(target, ascending=False, kind="stable")))
    out = [f"{len(h)} experiments run so far.",
           f"Highest {target}:\n{ranked.head(N_BEST)[cols].to_csv(index=False)}",
           f"Lowest {target}:\n{ranked.tail(N_WORST)[cols].to_csv(index=False)}"]
    drugs = set(shortlist["drug_a"]) | set(shortlist["drug_b"])
    related = h[h["drug_a"].isin(drugs) | h["drug_b"].isin(drugs)]
    if len(related):
        g = related.groupby(["drug_a", "drug_b"])
        best = related.loc[g[target].idxmax()].set_index(["drug_a", "drug_b"])["cell_line"]
        pairs = pd.DataFrame({"lines_tested": g.size(),
                              "mean_pct_in_line": g["pct_in_line"].mean().round(2),
                              "best_line": best})
        pairs = pairs.sort_values("mean_pct_in_line", ascending=False, kind="stable") \
            .head(MAX_PAIRS).reset_index()
        out.append("Tested pairs involving shortlisted drugs (pct_in_line: rank within the cell "
                   f"line, 1 = best):\n{masker.table(pairs).to_csv(index=False)}")
    return "\n".join(out)


def fmt_hypotheses(hypotheses: list[Hypothesis], masker: Masker) -> str:
    if not hypotheses:
        return "(none yet)"
    rows = [{"id": h.id, "statement": masker.text(h.statement),
             "drugs": [masker.name(d) for d in h.drugs], "level": h.level,
             "predicted_responders": list(h.predicted_responders),
             "responder_feature": masker.text(h.responder_feature),
             "rationale": masker.text(h.rationale), "evidence_ids": list(h.evidence_ids),
             "status": h.status, "n_tests": h.n_tests, "score": h.score}
            for h in hypotheses]
    return json.dumps(rows, indent=1)


def fmt_literature(sections: dict[str, list[Paper]], masker: Masker) -> str:
    blocks = []
    for heading, papers in sections.items():
        if not papers:
            continue
        blocks.append(f"### {masker.text(heading)}")
        for p in papers:
            abstract = excerpt(p.abstract)
            blocks.append(masker.text(f"[{p.source_id}] ({p.year}) {p.title}\n{abstract}"))
    return "\n\n".join(blocks) or "(no literature retrieved)"


# ---------- the three task prompts ----------

def _literature_section(literature, masker, reading_note: bool) -> str:
    body = fmt_literature(literature, masker)
    return f"## Literature\n{READING_NOTE}\n\n{body}" if reading_note \
        else f"## Literature\n{body}"


def hypothesize_prompt(shortlist, history, target_text, target, shown, retired, literature,
                       masker, max_new, max_responders: int,
                       reading_note: bool = False) -> str:
    parts = [HYPOTHESIZE.format(max_new=max_new, target=target_text,
                                max_responders=max_responders),
             f"## Shortlist\n{fmt_shortlist(shortlist, masker)}",
             f"## Results so far\n{fmt_results(history, shortlist, target, masker)}",
             f"## Existing hypotheses\n{fmt_hypotheses(shown, masker)}"]
    if retired:
        parts.append("## Retired ideas (do not repeat)\n" + masker.text("\n".join(retired)))
    parts.append(_literature_section(literature, masker, reading_note))
    return "\n\n".join(parts)


def critique_prompt(to_judge, history, shortlist, target, literature, masker,
                    reading_note: bool = False) -> str:
    return "\n\n".join([
        CRITIQUE.format(),  # .format() turns the escaped {{ }} into JSON braces
        f"## Hypotheses\n{fmt_hypotheses(to_judge, masker)}",
        f"## Results so far\n{fmt_results(history, shortlist, target, masker)}",
        _literature_section(literature, masker, reading_note),
    ])


def select_prompt(shortlist, history, target, options, k, masker) -> str:
    return "\n\n".join([
        SELECT.format(k=k),
        f"## Shortlist\n{fmt_shortlist(shortlist, masker)}",
        f"## Hypotheses\n{fmt_hypotheses(options, masker)}",
        f"## Results so far\n{fmt_results(history, shortlist, target, masker)}",
    ])


def heldout_prompt(hypotheses, heldout_summaries: dict[str, str], target_text, masker) -> str:
    """The one prompt that names held-out lines: genetics only, after every pick is made."""
    lines = "\n".join(f"- {line}: {heldout_summaries[line]}" for line in sorted(heldout_summaries))
    return "\n\n".join([
        PREDICT_HELDOUT.format(target=target_text,
                               max_responders=max_responders(len(heldout_summaries))),
        f"## Untested cell lines (genetics only)\n{lines}",
        f"## Mechanistic hypotheses\n{fmt_hypotheses(hypotheses, masker)}",
    ])
