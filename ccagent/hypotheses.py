"""The agent's hypotheses: what it believes, how results change that, and what it is shown.

Status rule. Uses only results the lab has returned, never the hidden evaluation cutoffs.
Each returned experiment gets its percentile rank, on the method's target, among all experiments
run so far. A hypothesis "covers" an experiment when the drugs fit: a one-drug hypothesis covers
any pair containing that drug; a multi-drug one covers pairs whose two drugs are both in its set.
  fewer than MIN_TESTS covered experiments      open
  mean percentile >= SUPPORT                    supported
  mean percentile <= REFUTE                     refuted
  otherwise                                     open
A hypothesis the critique drops is kept for the record but no longer scored or shown.
Mechanistic hypotheses also get responder_gap: mean percentile in the cell lines they predicted
would respond minus in the other lines (positive means the prediction pointed the right way).

Size limits. Every hypothesis ever made is stored, but each round the model sees a bounded
slice, so prompts stay flat across the run instead of growing with it:
  at most MAX_NEW new hypotheses accepted per round
  at most MAX_SHOWN active hypotheses shown (about drugs on the shortlist, supported first)
  at most MAX_RETIRED one-line reminders of refuted or dropped ideas
  at most MAX_CRITIQUE sent to critique: new ones first, then ones with new results since their
  last critique. MAX_NEW <= MAX_CRITIQUE, so a new hypothesis is always critiqued before it can
  be selected.
"""
from dataclasses import asdict, dataclass, field

import pandas as pd

LEVELS = ("mechanistic", "cross_cancer", "statistical")
# 4, not iteration 1's 2 (decision log, Nov 27): with no real effect and random ranks, 2 tests
# label a hypothesis supported 25% of the time and refuted 24%; 4 tests, 16% each.
MIN_TESTS = 4
SUPPORT, REFUTE = 0.65, 0.35
MAX_NEW, MAX_SHOWN, MAX_RETIRED, MAX_CRITIQUE = 6, 12, 6, 8
# A mechanistic hypothesis names the feature that defines its responders and may name at most
# half the lines it chooses from (decision log, Nov 27): it cannot hedge by naming nearly all of
# them, and a comparison group is always left. Common features are why the cap is not smaller:
# TP53 is mutated in about half the visible lines.
RESPONDER_SHARE = 0.5


def max_responders(n_lines: int) -> int:
    return max(1, int(n_lines * RESPONDER_SHARE))
_SHOW_ORDER = {"supported": 0, "open": 1}

assert MAX_NEW <= MAX_CRITIQUE, "every new hypothesis must fit in one round's critique"


@dataclass
class Hypothesis:
    id: str
    statement: str
    drugs: tuple[str, ...]
    level: str
    predicted_responders: tuple[str, ...] = ()
    responder_feature: str = ""  # e.g. "BRAF V600E": what the responders have in common
    rationale: str = ""
    evidence_ids: tuple[str, ...] = ()
    round_added: int = 0
    status: str = "open"
    n_tests: int = 0
    score: float | None = None
    responder_gap: float | None = None
    critiqued_at: int | None = None  # n_tests when last critiqued; None = never

    def covers(self, drug_a: str, drug_b: str) -> bool:
        drugs = set(self.drugs)
        if len(drugs) == 1:
            return drug_a in drugs or drug_b in drugs
        return {drug_a, drug_b} <= drugs


@dataclass
class HypothesisBook:
    items: dict[str, Hypothesis] = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)  # every change, for the run record
    _before_critique: dict = field(default_factory=dict, repr=False)  # for uncritique

    # ---------- changes ----------

    def add(self, proposals: list[dict], round_num: int) -> list[Hypothesis]:
        """Accept up to MAX_NEW proposals. The book assigns IDs, so the model cannot collide with
        or overwrite an earlier hypothesis. Returns the accepted hypotheses."""
        accepted = []
        for proposal in proposals[:MAX_NEW]:
            if proposal["level"] not in LEVELS:
                raise ValueError(f"unknown level {proposal['level']!r}")
            h = Hypothesis(
                id=f"H{len(self.items) + 1}",
                statement=proposal["statement"],
                drugs=tuple(proposal["drugs"]),
                level=proposal["level"],
                predicted_responders=tuple(proposal.get("predicted_responders", ())),
                responder_feature=proposal.get("responder_feature", ""),
                rationale=proposal.get("rationale", ""),
                evidence_ids=tuple(proposal.get("evidence_ids", ())),
                round_added=round_num,
            )
            self.items[h.id] = h
            accepted.append(h)
            self._log(round_num, h.id, "added", level=h.level)
        for _ in proposals[MAX_NEW:]:
            self._log(round_num, None, "rejected: over the per-round limit")
        return accepted

    def apply_critique(self, verdicts: list[dict], round_num: int) -> None:
        """verdicts: {id, verdict: keep | revise | drop, revised_statement, reason}."""
        for v in verdicts:
            h = self.items.get(v["id"])
            if h is None or h.status == "dropped":
                continue
            if v["verdict"] == "drop":
                h.status = "dropped"
            elif v["verdict"] == "revise":
                h.statement = v["revised_statement"]
            elif v["verdict"] != "keep":
                raise ValueError(f"unknown verdict {v['verdict']!r}")
            self._log(round_num, h.id, f"critique: {v['verdict']}", reason=v.get("reason", ""))

    def rescore(self, history: pd.DataFrame, target: str, round_num: int) -> None:
        """Re-score every hypothesis not dropped against all results returned so far."""
        if history is None or len(history) == 0:
            return
        pct = pd.to_numeric(history[target]).rank(pct=True).to_numpy()
        pairs = list(zip(history["drug_a"], history["drug_b"], strict=True))
        lines = history["cell_line"].to_numpy()
        for h in self.active():
            mask = [h.covers(a, b) for a, b in pairs]
            covered = pct[mask]
            h.n_tests = len(covered)
            h.score = round(float(covered.mean()), 3) if h.n_tests else None
            h.responder_gap = None
            if h.level == "mechanistic" and h.n_tests:
                responder = pd.Series(lines[mask]).isin(h.predicted_responders).to_numpy()
                if responder.any() and not responder.all():
                    h.responder_gap = round(
                        float(covered[responder].mean() - covered[~responder].mean()), 3)
            new = "open" if h.n_tests < MIN_TESTS else (
                "supported" if h.score >= SUPPORT else "refuted" if h.score <= REFUTE else "open")
            if new != h.status:
                self._log(round_num, h.id, f"status: {h.status} -> {new}",
                          n_tests=h.n_tests, score=h.score)
                h.status = new

    # ---------- what the model sees ----------

    def active(self) -> list[Hypothesis]:
        return [h for h in self.items.values() if h.status != "dropped"]

    def shown(self, shortlist: pd.DataFrame, k: int = MAX_SHOWN) -> list[Hypothesis]:
        """Supported and open hypotheses about pairs on this shortlist: supported first, then
        higher score, then newer."""
        pairs = set(zip(shortlist["drug_a"], shortlist["drug_b"], strict=True))
        relevant = [h for h in self.active() if h.status in _SHOW_ORDER
                    and any(h.covers(a, b) for a, b in pairs)]
        relevant.sort(key=lambda h: (_SHOW_ORDER[h.status], -(h.score or 0), -h.round_added))
        return relevant[:k]

    def retired(self, k: int = MAX_RETIRED) -> list[str]:
        """One-line reminders of failed ideas, newest first, so they are not proposed again."""
        gone = [h for h in self.items.values() if h.status in ("refuted", "dropped")]
        gone.sort(key=lambda h: -h.round_added)
        return [f"{h.id} ({h.status}): {h.statement[:120]}" for h in gone[:k]]

    def to_critique(self, new: list[Hypothesis], k: int = MAX_CRITIQUE) -> list[Hypothesis]:
        """New hypotheses first, then older ones never critiqued (a critique answer failed), then
        those with new results since their last critique (most new results first). Marks the
        returned ones as critiqued at their current n_tests; `uncritique` undoes that."""
        new_ids = {h.id for h in new}
        changed = [h for h in self.active() if h.id not in new_ids and (
            h.critiqued_at is None or (h.n_tests > 0 and h.critiqued_at != h.n_tests))]
        changed.sort(key=lambda h: (h.critiqued_at is not None,
                                    -(h.n_tests - (h.critiqued_at or 0))))
        picked = ([h for h in new if h.status != "dropped"] + changed)[:k]
        self._before_critique = {h.id: h.critiqued_at for h in picked}
        for h in picked:
            h.critiqued_at = h.n_tests
        return picked

    def uncritique(self, judged: list[Hypothesis], round_num: int) -> None:
        """The critique answer failed twice: these were not judged after all. Never-critiqued
        ones stay uncitable and are first in line next round."""
        for h in judged:
            h.critiqued_at = self._before_critique.get(h.id)
        self._log(round_num, None, "critique failed; not judged: "
                  + ", ".join(h.id for h in judged))

    def options(self, shortlist: pd.DataFrame, new: list[Hypothesis],
                critiqued_only: bool = False) -> list[Hypothesis]:
        """Hypotheses a pick may cite this round: those shown, plus this round's surviving new
        ones (a brand-new hypothesis has no results yet, so it would not rank into `shown`).
        With critique on, only hypotheses that have been critiqued may be cited."""
        out = {h.id: h for h in self.shown(shortlist)}
        out.update({h.id: h for h in new if h.status != "dropped"})
        return [h for h in out.values() if not critiqued_only or h.critiqued_at is not None]

    # ---------- record ----------

    def _log(self, round_num: int, hid: str | None, event: str, **extra) -> None:
        self.events.append({"round": round_num, "id": hid, "event": event, **extra})

    def to_dict(self) -> dict:
        return {"hypotheses": [asdict(h) for h in self.items.values()], "events": self.events}
