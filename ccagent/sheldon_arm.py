"""SHELDON as another method on the same simulated lab.

SHELDON (Joshi, Shethia & Lam, arXiv:2610.18025; github.com/stanford-oval/sheldon) is a
reasoning harness that searches a hypothesis space and an observation space together. Here the
O'Neil screen is registered as a SHELDON benchmark ("oneil-combo"): one task per seed, "find as
many strong combinations as you can within the experiment budget".

What SHELDON gets is what our agent gets, offered as tools instead of prompts:
  get_shortlist     the same 50-candidate shortlist (BO's order), from the same response model
  run_experiments   up to 10 exp_ids from the latest shortlist; returns synergy and inhibition only
  results_so_far    the same compact results summary the agent is shown
  drug_info         class and targets (drug_annotations.csv)
  cell_line_info    genetics summaries for visible lines only
  drug_literature / pair_literature   the same cached, filtered PubMed papers (2 per query)
No web search, web fetch or shell: the run must set params.probes = [] (checked at build time).
The shared first batch for the seed is run before SHELDON starts and shown in the task, as for
every other method. Held-out lines cannot be run or seen; the strong cutoffs never reach a tool.
Scoring happens afterwards, on the experiments SHELDON actually ran, with the same definitions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sheldon.benchmarks.base import Benchmark
from sheldon.core.probes import CallableProbe, Probe, ProbeResult
from sheldon.core.registry import benchmarks
from sheldon.core.types import Prediction, RunContext, Score, Task

from ccagent import prompts
from ccagent.evaluation import add_strong_flags
from ccagent.lab import LabError, SimLab
from ccagent.literature import CacheMiss, Literature
from ccagent.loop import BATCH_SIZE, LOG_COLS, initial_batch
from ccagent.masking import Masker
from ccagent.model import ResponseModel
from ccagent.shortlist import KAPPA, make_shortlist
from ccagent.strong import PRIMARY
from ccagent.targets import DESCRIPTIONS, with_target

NAME = "oneil-combo"
PAPERS_PER_QUERY = 2  # as the agent
NO_MASK = Masker()


SHORTLIST_HEADER = "Shortlist, best first:"


def scripted_bo_responder(prompt: str, tools: list[dict]) -> dict | None:
    """Offline stand-in for SHELDON's tool calls: ask for the shortlist, run its first 10 (BO's
    batch), repeat. With SHELDON's fake model this is a wiring check, like the agent's top-k
    fake. Reads only the last tool result, so it holds no state between calls."""
    names = {t.get("name") for t in tools}
    if "run_experiments" not in names:
        return None
    if SHORTLIST_HEADER in prompt:
        rows = prompt.split(SHORTLIST_HEADER, 1)[1].strip().splitlines()[1:]
        ids = [r.split(",", 1)[0] for r in rows if r.strip()][:BATCH_SIZE]
        return {"name": "run_experiments", "input": {"exp_ids": ids, "reason": "BO's batch"}}
    if "budget is spent" in prompt or "0 experiments left" in prompt:
        return None
    return {"name": "get_shortlist", "input": {}}


@dataclass
class Context:
    """Everything the benchmark reads. Built once per run by `make_context` (tests replace it)."""
    table: pd.DataFrame                 # the screen, for the lab
    heldout: list[str]
    truth: pd.DataFrame                 # visible outcomes, for scoring only
    annotations: pd.DataFrame
    line_summaries: dict[str, str]
    tissue: dict | None = None
    literature: Literature | None = None
    model_factory: Any = ResponseModel


def make_context(literature: bool = True) -> Context:
    """The real data, read the same way as scripts/run_agent.py."""
    from ccagent.evaluation import visible_truth
    from ccagent.lab import DATA, read_heldout
    from ccagent.literature import Exclusions, PubMed, class_terms, drug_names
    from scripts.run_baselines import cell_line_tissue

    summaries = pd.read_csv("data/processed/cell_line_summaries.csv")
    return Context(
        table=pd.read_csv(DATA), heldout=read_heldout(), truth=visible_truth(),
        annotations=pd.read_csv("data/reference/drug_annotations.csv"),
        line_summaries=dict(zip(summaries["cell_line"], summaries["summary"], strict=True)),
        tissue=cell_line_tissue(),
        literature=Literature(PubMed(), Exclusions.from_file(), drug_names(), class_terms())
        if literature else None)


@dataclass
class Trial:
    """One seed's lab and everything SHELDON has done in it."""
    seed: int
    lab: SimLab
    model: Any
    history: pd.DataFrame
    shortlist: pd.DataFrame | None = None
    round: int = 0
    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    refused: list[str] = field(default_factory=list)  # every refused run_experiments call


@benchmarks.register(NAME)
class OneilCombo(Benchmark):
    name = NAME
    splits = ("eval",)
    answer_kind = "text"
    answer_format = ("End with one line starting `SUMMARY:` that states, in a sentence, which "
                     "combinations you found most promising and why. Your score is the strong "
                     "combinations among the experiments you ran, not this line.")
    objective_probe = "run_experiments"
    probes_essential = True  # the lab is only reachable through the tools
    blurb = "choose drug-combination experiments in a replayed O'Neil 2016 screen"

    def __init__(self, root: Any = None, **opts: Any) -> None:
        super().__init__(root, **opts)
        self.seeds = [int(s) for s in opts.get("seeds", [0])]
        self.budget = int(opts.get("budget", 250))
        self.batch = int(opts.get("batch_size", BATCH_SIZE))
        self.target = str(opts.get("target", "joint"))
        self.kappa = float(opts.get("kappa", KAPPA))
        self.shortlist_only = bool(opts.get("shortlist_only", True))
        self.use_literature = bool(opts.get("literature", True))
        self.log_dir = opts.get("log_dir")  # where score() writes each seed's run log
        self.ctx = make_context(self.use_literature)
        self.trials: dict[str, Trial] = {}

    # -- tasks ------------------------------------------------------------

    def load(self, split: str = "eval", limit: int | None = None, seed: int = 0) -> list[Task]:
        tasks = []
        for s in self.seeds:
            trial = self._start(s)
            tid = f"seed{s}"
            self.trials[tid] = trial
            tasks.append(Task(id=tid, benchmark=self.name, question=self._question(trial),
                              payload={"seed": s}, answer=None))
        return self.subsample(tasks, limit, seed)

    def _start(self, seed: int) -> Trial:
        """A fresh lab with the shared first batch already run (loop.run_selection's round 0)."""
        lab = SimLab(self.ctx.table, self.ctx.heldout, budget=self.budget)
        candidates = lab.candidates()
        model = self.ctx.model_factory(candidates["cell_line"].unique(), tissue=self.ctx.tissue,
                                       with_genetics=True, seed=seed)
        trial = Trial(seed=seed, lab=lab, model=model,
                      history=pd.DataFrame(columns=LOG_COLS),
                      rng=np.random.default_rng(seed + 1))
        k = min(self.batch, lab.budget_remaining(), len(candidates))
        self._record(trial, initial_batch(candidates, seed, k), "shared first batch", "none")
        return trial

    def _question(self, trial: Trial) -> str:
        drugs = sorted(set(self.ctx.annotations["drug_id"]))
        lines = sorted(set(trial.lab.candidates()["cell_line"]))
        first = trial.history[prompts.RESULT_COLS].to_csv(index=False)
        return "\n".join([
            "# Choose drug-combination experiments",
            "",
            "You are helping a cancer screening lab. Each experiment tests one drug pair in one "
            "cancer cell line and returns two numbers: a synergy score (Loewe; higher means the "
            "drugs work better together than expected) and growth inhibition (percent).",
            "",
            f"Goal: find as many strong combinations as you can within {self.budget} "
            f"experiments in total. A strong combination scores well on {DESCRIPTIONS[self.target]}"
            " relative to other experiments in the same cell line. "
            f"{len(trial.history)} experiments have already been run (below), so "
            f"{trial.lab.budget_remaining()} remain.",
            "",
            "## How to run experiments",
            "- `get_shortlist` gives the 50 untested experiments a statistical model currently "
            "rates most worth testing, with its predicted score and uncertainty. "
            + ("You can only run experiments from the latest shortlist; ask again after new "
               "results, because it changes." if self.shortlist_only else ""),
            f"- `run_experiments` runs up to {self.batch} at a time and returns their results. "
            "Each experiment can be run once. This is the only way to learn an outcome.",
            "- `results_so_far`, `drug_info` and `cell_line_info` are free."
            + (" `drug_literature` and `pair_literature` return published abstracts (free)."
               if self.use_literature else ""),
            "",
            f"Drugs ({len(drugs)}): {', '.join(drugs)}",
            f"Cell lines ({len(lines)}): {', '.join(lines)}",
            "",
            "## Results already returned",
            first,
            "## Answer",
            self.answer_format,
        ])

    # -- tools ------------------------------------------------------------

    def probes(self, task: Task) -> list[Probe]:
        trial = self.trials[task.id]
        ann = self.ctx.annotations.set_index("drug_id")
        visible = set(trial.lab.candidates()["cell_line"]) | set(trial.history["cell_line"])

        async def get_shortlist() -> ProbeResult:
            trial.shortlist = self._shortlist(trial)
            return ProbeResult(summary=(
                f"{trial.lab.budget_remaining()} experiments left. Shortlist, best first:\n"
                + prompts.fmt_shortlist(trial.shortlist, NO_MASK)), limit=20000)

        async def run_experiments(exp_ids: list[str], reason: str = "") -> ProbeResult:
            problem = self._refusal(trial, exp_ids)
            if problem:
                trial.refused.append(problem)
                return ProbeResult(summary=problem, ok=False)
            try:
                rows = self._record(trial, list(exp_ids), reason or "sheldon", "sheldon")
            except LabError as e:  # unknown or held-out IDs: the lab's own refusal
                trial.refused.append(str(e))
                return ProbeResult(summary=str(e), ok=False)
            return ProbeResult(summary=(
                f"{len(rows)} results ({trial.lab.budget_remaining()} experiments left):\n"
                + rows[prompts.RESULT_COLS].to_csv(index=False)), limit=8000)

        async def results_so_far() -> ProbeResult:
            short = trial.shortlist if trial.shortlist is not None else trial.history
            return ProbeResult(summary=prompts.fmt_results(
                with_target(trial.history, self.target), short, self.target, NO_MASK),
                limit=12000)

        async def drug_info(drugs: list[str]) -> ProbeResult:
            lines = [f"{d}: {ann.at[d, 'drug_class']}; targets: {ann.at[d, 'targets']}"
                     if d in ann.index else f"{d}: unknown drug" for d in drugs]
            return ProbeResult(summary="\n".join(lines))

        async def cell_line_info(cell_lines: list[str]) -> ProbeResult:
            lines = [f"{c}: {self.ctx.line_summaries.get(c, 'no summary')}" if c in visible
                     else f"{c}: not a cell line in this screen" for c in cell_lines]
            return ProbeResult(summary="\n".join(lines), limit=8000)

        async def drug_literature(drug: str) -> ProbeResult:
            return self._papers(f"{drug}: mechanism",
                                lambda: self.ctx.literature.for_drug(drug, PAPERS_PER_QUERY))

        async def pair_literature(drug_a: str, drug_b: str) -> ProbeResult:
            """As the agent: papers naming both, topped up with class evidence when thin."""
            lit, n = self.ctx.literature, PAPERS_PER_QUERY
            try:
                together = lit.for_pair(drug_a, drug_b, n)
                by_class = lit.for_pair_by_class(drug_a, drug_b, n - len(together),
                                                 skip=[p.pmid for p in together]) \
                    if len(together) < n else []
            except CacheMiss:
                together, by_class = [], []
            sections = {f"{drug_a} + {drug_b}: together": together,
                        f"{drug_a} + {drug_b}: their drug classes together (not these exact "
                        "compounds)": by_class}
            if not together and not by_class:
                return ProbeResult(summary="no cached literature for this query")
            return ProbeResult(summary=prompts.READING_NOTE + "\n\n" +
                               prompts.fmt_literature(sections, NO_MASK), limit=6000)

        def tool(name, description, fn, properties=None, required=()):
            return CallableProbe(name=name, description=description, fn=fn, max_strength=0.9,
                                 input_schema={"type": "object", "properties": properties or {},
                                               "required": list(required),
                                               "additionalProperties": False})

        ids = {"type": "array", "items": {"type": "string"}}
        out = [
            tool("get_shortlist", "The 50 untested experiments the statistical model rates most "
                 "worth testing now, best first, with predicted score and uncertainty.",
                 get_shortlist),
            tool("run_experiments", f"Run up to {self.batch} experiments (exp_ids from the latest "
                 "shortlist) and get their synergy and inhibition. Uses the experiment budget.",
                 run_experiments, {"exp_ids": ids, "reason": {"type": "string",
                                   "description": "One sentence on why these."}}, ["exp_ids"]),
            tool("results_so_far", "Summary of every result returned so far: best and worst, and "
                 "per-pair summaries.", results_so_far),
            tool("drug_info", "Class and targets for the named drugs.", drug_info,
                 {"drugs": ids}, ["drugs"]),
            tool("cell_line_info", "Tissue and genetics summary for the named cell lines.",
                 cell_line_info, {"cell_lines": ids}, ["cell_lines"]),
        ]
        if self.use_literature:
            out += [
                tool("drug_literature", "Published abstracts on one drug's mechanism.",
                     drug_literature, {"drug": {"type": "string"}}, ["drug"]),
                tool("pair_literature", "Published abstracts on two drugs used together.",
                     pair_literature, {"drug_a": {"type": "string"},
                                       "drug_b": {"type": "string"}}, ["drug_a", "drug_b"]),
            ]
        return out

    def _shortlist(self, trial: Trial) -> pd.DataFrame:
        past = with_target(trial.history, self.target)
        short = make_shortlist(trial.lab.candidates(), past, trial.rng, model=trial.model,
                               target=self.target, kappa=self.kappa, tissue=self.ctx.tissue)
        prompts.check_inputs(set(self.ctx.heldout), short, past)
        return short

    def _refusal(self, trial: Trial, exp_ids: list[str]) -> str | None:
        exp_ids = list(exp_ids or [])
        if not exp_ids:
            return "no exp_ids given"
        if len(exp_ids) > self.batch:
            return f"at most {self.batch} experiments per call; you asked for {len(exp_ids)}"
        if self.shortlist_only:
            if trial.shortlist is None:
                return "call get_shortlist first; experiments come from the latest shortlist"
            off = sorted(set(exp_ids) - set(trial.shortlist["exp_id"]))
            if off:
                return (f"not on the latest shortlist: {off[:5]}. Call get_shortlist for the "
                        "current one.")
        left = trial.lab.budget_remaining()
        if left == 0:
            return "the experiment budget is spent"
        if len(exp_ids) > left:
            return f"only {left} experiments left; you asked for {len(exp_ids)}"
        done = set(trial.history["exp_id"])
        repeats = sorted(set(exp_ids) & done)
        if repeats or len(set(exp_ids)) != len(exp_ids):
            return f"already run or repeated: {(repeats or exp_ids)[:5]}"
        return None

    def _record(self, trial: Trial, exp_ids: list[str], why: str, level: str) -> pd.DataFrame:
        candidates = trial.lab.candidates().set_index("exp_id")
        results = trial.lab.run(exp_ids)
        batch = candidates.loc[exp_ids].reset_index().merge(results, on="exp_id")
        batch["round"] = trial.round
        batch["hypothesis_id"] = None
        batch["explanation_level"] = level
        batch["explanation_text"] = why
        trial.history = pd.concat([trial.history, batch.reindex(columns=LOG_COLS)],
                                  ignore_index=True)
        trial.round += 1
        trial.shortlist = None  # results change the shortlist; it must be asked for again
        return batch

    def _papers(self, heading: str, fetch) -> ProbeResult:
        try:
            papers = fetch()
        except CacheMiss:
            papers = []
        text = prompts.fmt_literature({heading: papers}, NO_MASK) if papers \
            else "no cached literature for this query"
        return ProbeResult(summary=text, limit=6000)

    # -- budget -----------------------------------------------------------

    def objective_exhausted(self, task: Task) -> bool:
        return self.trials[task.id].lab.budget_remaining() == 0

    # -- scoring ----------------------------------------------------------

    def run_log(self, task_id: str) -> pd.DataFrame:
        return add_strong_flags(self.trials[task_id].history, self.ctx.truth)

    async def score(self, task: Task, pred: Prediction, ctx: RunContext) -> Score:
        trial = self.trials[task.id]
        log = self.run_log(task.id)
        if self.log_dir:
            out = Path(self.log_dir)
            out.mkdir(parents=True, exist_ok=True)
            log.to_csv(out / f"sheldon_{self.target}_seed{trial.seed}.csv", index=False)
        strong = {c.removeprefix("strong_"): float(log[c].sum())
                  for c in log.columns if c.startswith("strong_")}
        return Score(task_id=task.id, score=strong[PRIMARY], metrics={
            "experiments_run": float(len(log)), "budget_used": len(log) / self.budget,
            "refused_calls": float(len(trial.refused)),
            **{f"strong_{k}": v for k, v in strong.items()}},
            detail={"seed": trial.seed, "summary": str(pred.answer or "")[:500],
                    "refusals": trial.refused[:20]})

    def check(self) -> list[str]:
        return []
