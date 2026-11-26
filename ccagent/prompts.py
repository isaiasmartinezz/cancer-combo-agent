"""The text the agent's language model sees: one fixed reference block, and three task prompts.

Leakage rules, each enforced here and locked by a test:
- Results come only from the run history, and only the columns a lab returns (plus the method's
  own target computed from them). Strong flags, cutoffs and hidden outcomes are never formatted,
  even if a caller passes a table that contains them.
- No held-out cell line may appear in the shortlist, the results, or the reference block.
  `check_inputs` refuses to build a prompt otherwise. (A held-out line can still be named inside a
  retrieved abstract; that is published literature, not a result.)
- Every drug name passes through the masker, which is the identity unless the masked-name
  ablation is on.
"""
import json

import pandas as pd

from ccagent.hypotheses import Hypothesis
from ccagent.literature import Paper
from ccagent.masking import Masker

ABSTRACT_CHARS = 400  # the key finding is usually in the first sentences
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
- "predicted_responders": for mechanistic hypotheses, the shortlist cell lines that should \
respond, chosen for a reason you state; empty for other levels
- "rationale": at most two sentences
- "evidence_ids": IDs of the literature excerpts it rests on (may be empty)
Do not repeat an existing or retired idea. Answer as {{"hypotheses": [...]}}."""

CRITIQUE = """Challenge each hypothesis below against the literature and the results so far. \
Look for claims the cited sources do not support, results that contradict it, and mechanisms \
that do not apply to the predicted cell lines. Give exactly one verdict per hypothesis:
{{"verdicts": [{{"id": "H1", "verdict": "keep" | "revise" | "drop", "revised_statement": \
"(only for revise)", "reason": "one sentence", "evidence_ids": []}}]}}"""

SELECT = """Choose exactly {k} experiments from the shortlist, by exp_id, for the next batch. \
Balance testing the most promising hypotheses against experiments that would tell competing \
hypotheses apart; a few "explore" picks for untested drug classes are fine. Each pick names the \
hypothesis it tests (or "explore"), an explanation level, and a one-sentence reason:
{{"picks": [{{"exp_id": "E00001", "hypothesis_id": "H1" | "explore", "level": "mechanistic" | \
"cross_cancer" | "statistical", "reason": "..."}}]}}"""


# ---------- leakage check ----------

def check_inputs(heldout: set[str], *tables: pd.DataFrame | None) -> None:
    for table in tables:
        if table is None or len(table) == 0 or "cell_line" not in table:
            continue
        leaked = sorted(set(table["cell_line"]) & set(heldout))
        if leaked:
            raise LeakError(f"held-out cell lines would reach the prompt: {leaked}")


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
    ranked = masker.table(h.sort_values(target, ascending=False, kind="stable"))
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
            abstract = p.abstract[:ABSTRACT_CHARS] + ("..." if len(p.abstract) > ABSTRACT_CHARS
                                                      else "")
            blocks.append(masker.text(f"[{p.source_id}] ({p.year}) {p.title}\n{abstract}"))
    return "\n\n".join(blocks) or "(no literature retrieved)"


# ---------- the three task prompts ----------

def hypothesize_prompt(shortlist, history, target_text, target, shown, retired, literature,
                       masker, max_new) -> str:
    parts = [HYPOTHESIZE.format(max_new=max_new, target=target_text),
             f"## Shortlist\n{fmt_shortlist(shortlist, masker)}",
             f"## Results so far\n{fmt_results(history, shortlist, target, masker)}",
             f"## Existing hypotheses\n{fmt_hypotheses(shown, masker)}"]
    if retired:
        parts.append("## Retired ideas (do not repeat)\n" + masker.text("\n".join(retired)))
    parts.append(f"## Literature\n{fmt_literature(literature, masker)}")
    return "\n\n".join(parts)


def critique_prompt(to_judge, history, shortlist, target, literature, masker) -> str:
    return "\n\n".join([
        CRITIQUE,
        f"## Hypotheses\n{fmt_hypotheses(to_judge, masker)}",
        f"## Results so far\n{fmt_results(history, shortlist, target, masker)}",
        f"## Literature\n{fmt_literature(literature, masker)}",
    ])


def select_prompt(shortlist, history, target, options, k, masker) -> str:
    return "\n\n".join([
        SELECT.format(k=k),
        f"## Shortlist\n{fmt_shortlist(shortlist, masker)}",
        f"## Hypotheses\n{fmt_hypotheses(options, masker)}",
        f"## Results so far\n{fmt_results(history, shortlist, target, masker)}",
    ])
