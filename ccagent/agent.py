"""The literature-informed agent, as a chooser for the shared experiment loop.

Each round:
  1. Shortlist  the response model's 50 candidates, ordered by BO's score (`make_shortlist`)
  2. Retrieve   cached literature for the first few drug pairs on the shortlist
  3. Hypothesize  re-score the hypothesis list on returned results, then ask for new ones
  4. Critique   judge new hypotheses and ones with new results; drop or revise them
  5. Select     choose k from the shortlist, each tagged with a hypothesis and explanation
After the last round, `predict_heldout` asks once which held-out lines each mechanistic
hypothesis should reach, from their genetics alone (the plan's mechanism test, section 9.3).

The agent sees only what the loop hands every method (untested candidates and its own
results), plus the reference tables and the literature cache. Ablations:
  no_literature  step 2 returns nothing
  no_critique    step 4 is skipped
  masked_names   every drug name the model sees is replaced by a code (`masking.py`)
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ccagent import prompts, steps
from ccagent.hypotheses import MAX_NEW, Hypothesis, HypothesisBook, max_responders
from ccagent.literature import Literature, Paper
from ccagent.llm.client import LLMClient
from ccagent.masking import Masker
from ccagent.model import ResponseModel
from ccagent.shortlist import KAPPA, make_shortlist
from ccagent.targets import DESCRIPTIONS, with_target

ABLATIONS = ("none", "no_literature", "no_critique", "masked_names")
EVIDENCE_PAIRS = 3  # drug pairs per round that get literature (iteration 1's budget setting)
PAPERS_PER_QUERY = 2
ROTATE_ROUNDS = 2   # a pair that got literature waits this many rounds before getting it again


@dataclass
class AgentConfig:
    target: str = "joint"
    ablation: str = "none"
    kappa: float = KAPPA
    seed: int = 0
    with_genetics: bool = True
    evidence_pairs: int = EVIDENCE_PAIRS
    papers_per_query: int = PAPERS_PER_QUERY
    # Literature design (decision log, Nov 27). All three only change what the literature arm
    # sees; with no_literature the prompts are unchanged.
    class_evidence: bool = True   # fill thin pairs with papers on the drug classes together
    rotate_evidence: bool = True  # spread literature over the shortlist's pairs across rounds
    reading_note: bool = True     # say that publication volume is not evidence of synergy

    def __post_init__(self):
        if self.ablation not in ABLATIONS:
            raise ValueError(f"ablation must be one of {ABLATIONS}")


@dataclass
class AgentChooser:
    client: LLMClient
    annotations: pd.DataFrame           # drug_annotations.csv
    line_summaries: dict[str, str]      # cell_line -> one-line genetics summary
    heldout: set[str]
    literature: Literature | None = None
    tissue: dict | None = None
    masker: Masker = field(default_factory=Masker)
    config: AgentConfig = field(default_factory=AgentConfig)
    # state
    book: HypothesisBook = field(default_factory=HypothesisBook)
    papers: dict[str, Paper] = field(default_factory=dict)  # every paper shown, by source_id
    calls: list[dict] = field(default_factory=list)          # every model call, for the log
    round: int = 0
    model: ResponseModel | None = None
    system: str | None = None
    visible: set[str] = field(default_factory=set)  # cell lines a hypothesis may name
    evidence_round: dict = field(default_factory=dict)  # (drug_a, drug_b) -> last round shown

    def __post_init__(self):
        if self.config.ablation == "masked_names" and not self.masker.enabled:
            raise ValueError("the masked_names ablation needs a Masker built from the drug tables")
        if self.config.ablation != "masked_names" and self.masker.enabled:
            raise ValueError("a masking Masker is only allowed in the masked_names ablation")
        if self.config.ablation != "no_literature" and self.literature is None:
            raise ValueError("literature is required unless the ablation is no_literature")

    # ---------- the loop calls this ----------

    def __call__(self, candidates: pd.DataFrame, history: pd.DataFrame, k: int,
                 rng: np.random.Generator):
        cfg, m = self.config, self.masker
        self.round += 1
        if self.model is None:  # the first call's candidates hold every visible cell line
            visible = self.visible = set(candidates["cell_line"])
            # same line order as ModelChooser, so the features (and BO's scores) are identical
            self.model = ResponseModel(candidates["cell_line"].unique(), tissue=self.tissue,
                                       with_genetics=cfg.with_genetics, seed=cfg.seed)
            self.system = prompts.SYSTEM + "\n\n" + prompts.reference_block(
                self.annotations, self.line_summaries, visible, m)
        past = with_target(history, cfg.target)

        # 1. shortlist (same call, same model and settings as the shortlist_random control)
        short = make_shortlist(candidates, past, rng, model=self.model, target=cfg.target,
                               kappa=cfg.kappa, tissue=self.tissue)
        prompts.check_inputs(self.heldout, short, past)
        shown = m.table(short)

        # 2. retrieve
        literature = self._retrieve(short)
        source_ids = {p.source_id for papers in literature.values() for p in papers}

        # 3. hypothesize
        self.book.rescore(past, cfg.target, self.round)
        note = cfg.reading_note and bool(literature)
        prompt = prompts.hypothesize_prompt(
            short, past, DESCRIPTIONS[cfg.target], cfg.target, self.book.shown(short),
            self.book.retired(), literature, m, MAX_NEW, max_responders(len(self.visible)),
            reading_note=note)
        proposals, record = steps.hypothesize(self.client, self.system, prompt, shown,
                                              source_ids, m, lines=self.visible)
        self._record(record)
        new = self.book.add(proposals or [], self.round)

        # 4. critique (sees this round's papers plus earlier papers the judged ones cite)
        if cfg.ablation != "no_critique":
            to_judge = self.book.to_critique(new)
            if to_judge:
                lit = self._with_cited(literature, to_judge)
                ids = {p.source_id for papers in lit.values() for p in papers}
                prompt = prompts.critique_prompt(to_judge, past, short, cfg.target, lit, m,
                                                 reading_note=note)
                verdicts, record = steps.critique(self.client, self.system, prompt, to_judge,
                                                  ids)
                self._record(record)
                if verdicts is None:  # failed twice: nothing was judged, so nothing passes
                    self.book.uncritique(to_judge, self.round)
                else:
                    self.book.apply_critique(verdicts, self.round)

        # 5. select (with critique on, an uncritiqued hypothesis cannot be cited)
        options = self.book.options(short, new, critiqued_only=cfg.ablation != "no_critique")
        prompt = prompts.select_prompt(short, past, cfg.target, options, k, m)
        picks, record = steps.select(self.client, self.system, prompt, short, options, k)
        self._record(record)
        return [p["exp_id"] for p in picks], [steps.explain(p, self.book, short) for p in picks]

    # ---------- after the run ----------

    def predict_heldout(self) -> list[dict] | None:
        """Held-out predictions for every mechanistic hypothesis still standing. Called once the
        loop has finished, so nothing it shows can influence a pick. [] if there is nothing to
        predict; None if the model's answer failed twice."""
        hyps = [h for h in self.book.active() if h.level == "mechanistic"]
        if not hyps or self.system is None:
            return []
        lines = {line: self.line_summaries.get(line, "genetics unknown")
                 for line in sorted(self.heldout)}
        prompt = prompts.heldout_prompt(hyps, lines, DESCRIPTIONS[self.config.target],
                                        self.masker)
        self.round = "final"
        items, record = steps.predict_heldout(self.client, self.system, prompt, hyps, set(lines))
        self._record(record)
        if items is None:
            return None
        by_id = {h.id: h for h in hyps}
        return [{"hypothesis_id": p["hypothesis_id"], "drugs": by_id[p["hypothesis_id"]].drugs,
                 "statement": by_id[p["hypothesis_id"]].statement,
                 "responder_feature": by_id[p["hypothesis_id"]].responder_feature,
                 "status": by_id[p["hypothesis_id"]].status,
                 "predicted_responders": sorted(p["predicted_responders"]),
                 "reason": p["reason"]} for p in items]

    # ---------- helpers ----------

    def evidence_pairs(self, short: pd.DataFrame) -> list[tuple[str, str]]:
        """The shortlist's drug pairs that get literature this round, best first. With rotation,
        a pair shown in the last ROTATE_ROUNDS rounds gives way to the next one, so evidence
        reaches more of the shortlist instead of the same top pairs every round."""
        k = self.config.evidence_pairs
        pairs = list(dict.fromkeys(zip(short["drug_a"], short["drug_b"], strict=True)))
        if not self.config.rotate_evidence:
            return pairs[:k]
        last = {p: self.evidence_round.get(p, -ROTATE_ROUNDS - 1) for p in pairs}
        fresh = [p for p in pairs if self.round - last[p] > ROTATE_ROUNDS]
        stale = sorted((p for p in pairs if p not in fresh), key=lambda p: last[p])  # oldest first
        chosen = (fresh + stale)[:k]
        for p in chosen:
            self.evidence_round[p] = self.round
        return chosen

    def _retrieve(self, short: pd.DataFrame) -> dict[str, list[Paper]]:
        if self.config.ablation == "no_literature":
            return {}
        n, out = self.config.papers_per_query, {}
        for a, b in self.evidence_pairs(short):
            out[f"{a}: mechanism"] = self.literature.for_drug(a, n)
            out[f"{b}: mechanism"] = self.literature.for_drug(b, n)
            together = self.literature.for_pair(a, b, n)
            out[f"{a} + {b}: together"] = together
            if self.config.class_evidence and len(together) < n:
                out[f"{a} + {b}: their drug classes together (not these exact compounds)"] = \
                    self.literature.for_pair_by_class(a, b, n - len(together),
                                                      skip=[p.pmid for p in together])
        for papers in out.values():
            prompts.check_papers(self.heldout, papers)  # redacted by Literature(hide_lines=...)
            for p in papers:
                self.papers.setdefault(p.source_id, p)
        return out

    def _with_cited(self, literature, judged: list[Hypothesis]) -> dict[str, list[Paper]]:
        have = {p.source_id for papers in literature.values() for p in papers}
        cited = sorted({i for h in judged for i in h.evidence_ids} - have)
        earlier = [self.papers[i] for i in cited if i in self.papers]
        return {**literature, "Earlier papers cited by these hypotheses": earlier} if earlier \
            else literature

    def _record(self, record: steps.StepRecord) -> None:
        for attempt in record.attempts:
            self.calls.append({"round": self.round, "step": record.step, **attempt})
