# Design review: every choice in the study, in one place

Purpose: review the design before the real runs, and later use it as the source for the
report's methods section. Each item says what was chosen, why, and where it lives in the code.
The full history and checks for each choice are in `docs/decision_log.md`.

Status tags:
- **[decide]** an open choice that has to be settled before the real runs, because it changes
  what the model sees in every arm
- **[confirm]** chosen already; worth a look
- **[fixed]** settled with the research team, or verified against iteration 1; changing it
  means redoing verified work

The exact prompts are in `docs/prompt_examples/`, rendered from the real data by
`python -m scripts.render_prompts` (round 2 of seed 0 for each arm, plus the held-out prompt).
The hypotheses in those files are placeholders from the scripted model, not real model output.

---

## 1. Decisions needed before the real runs

### 1.1 [decided: last 800 characters] Which part of each abstract the model sees
**Now:** the first 400 characters of each abstract (`prompts.ABSTRACT_CHARS`), as in iteration 1.
**Problem:** the median cached abstract is 1,638 characters, so the model sees about the first
quarter, which is usually background. Across all 2,665 cached abstracts, finding-type wording
("we found", "results show", "significantly", "synergistic", "in conclusion") appears in the
first 400 characters of 12% of abstracts, against 32% of the last 400 characters.
Example from the round-2 prompt: the dactolisib + cisplatin paper is cut off at "However, its
therapeutic outcomes have remained unsatisfactory due to severe side effects, a short el...",
before it says what it found.
**Options:**
- (a) keep the first 400 characters (comparable with iteration 1);
- (b) show the last 400 characters, usually the results and conclusion;
- (c) show the first 150 and last 250 characters, giving some context plus the finding.

Options (b) and (c) cost the same tokens as (a).
**Recommendation:** (b). The title already says what the paper is about, and the ending says
what it found. Literature arm only (the no-literature arm is unchanged). This is a principled
fix, not tuned on any score. It would be recorded as part of literature v3, and it plausibly
explains some of the pilot's "literature hurt" result: the model mostly saw generic background
on well-known drugs.

### 1.2 [decided: 3 decimals] Rounding numbers in the results tables
**Now:** synergy and inhibition are printed at full precision (`12.874675113616199`) in the
results summary and in SHELDON's tool output.
**Proposal:** round to 1 decimal in every arm, the same for all methods. It saves roughly
5% of tokens per prompt and is easier to read. The lab's own measurement noise is far larger
than 0.1. The response model and the scoring are not affected, only the text.
**Recommendation:** yes.

---

## 2. The study design

| Item | Choice | Why | Status |
|---|---|---|---|
| Data | O'Neil 2016 screen (DrugComb v1.5): 38 drugs, 39 cell lines, 22,737 experiments (583 pairs) after cleaning | Largest public screen with both synergy and inhibition | [fixed] |
| Replicates | Averaged, spread kept | One result per experiment, like a lab report | [fixed] |
| Duplicate screening runs | Averaged and flagged (315 experiments); sensitivity check planned | Two batches of the same experiment disagree | [fixed] |
| Held-out cell lines | 10 lines (A2780, HT29, KPL1, LOVO, NCIH23, NCIH520, RPMI7951, SKMEL30, UWB1289+BRCA1, ZR751) never run and never shown, except in the held-out prompt after the run | Mechanism test on cells the agent never saw | [fixed] |
| Visible pool | 29 lines, 16,907 experiments | | [fixed] |
| "Strong" (primary) | Loewe synergy in the top 10% *and* inhibition above a cutoff giving ~5% overall, both ranked within each cell line | Agreed with the research team (Nov 13); within-line ranking so finding drug-sensitive lines doesn't count as synergy | [fixed] |
| "Strong" (secondary) | Same rule over all experiments; synergy-only and inhibition-only top 5% | Sensitivity checks | [fixed] |
| Budget | 250 experiments, 25 batches of 10 | Iteration 1's setting; ~1.5% of the pool | [fixed] |
| First batch | Same 10 random experiments for every method, per seed | Every method starts from the same information | [fixed] |
| Seeds | 0–4 for reporting; 100–104 for tuning only | No tuning on reported seeds | [fixed] |
| What methods maximize | "Joint" score: synergy and inhibition each adjusted for their cell line's mean (shrunk for lines with few results) and standardized; the smaller of the two | Mirrors the strong definition using only returned results | [confirm] |
| Spread | Sample SD (n−1) | Correct for 5 seeds; Nov updates used population SD | [fixed] |
| Libraries | Pinned (scikit-learn 1.9.1, etc.) | Baselines change with the scikit-learn version | [fixed] |

## 3. Baselines and the shortlist

| Item | Choice | Why | Status |
|---|---|---|---|
| Response model | Random forest on drug features (structure fingerprint, ChEMBL targets, identity), cell-line features (identity, tissue, genetics) and the two drugs' target-gene expression in that line; uncertainty = spread across trees | Simple, fast, honest uncertainty | [fixed] |
| Random | 10 random per batch | Floor | [fixed] |
| Uncertainty sampling | Highest model uncertainty | Pure exploration | [fixed] |
| BO | Highest predicted + kappa × uncertainty (UCB) | Standard optimizer | [fixed] |
| kappa | 1.0, tuned on seeds 100–104; rechecked Nov 27 (0 → 41.4, 0.5 → 36.2, **1 → 43.6**, 2 → 30.4, 4 → 17.0) | Not tuned on reported seeds | [fixed] |
| Shortlist | 50 candidates: 45 best by BO's score plus 5 of the most uncertain, ordered by BO's score | The agent's choice set; its first 10 rows are BO's batch, so "did the agent beat BO's ranking?" is a fair question | [fixed] |
| Random from shortlist | 10 random picks from the same 50 | Separates "the shortlist is good" from "the agent chose well"; currently nearly ties BO (45.2 vs 47.2) | [fixed] |

## 4. The agent (`ccagent/agent.py`)

### 4.1 Each round
1. **Shortlist:** the same call as the random-from-shortlist control.
2. **Retrieve:** cached literature for 3 drug pairs from the shortlist: 2 mechanism papers per
   drug, 2 papers naming both drugs, topped up with class papers when fewer than 2 name both.
3. **Hypothesize:** re-score existing hypotheses on returned results, then ask for up to 6 new ones.
4. **Critique:** keep, revise or drop new hypotheses and ones with new results.
5. **Select:** exactly 10 from the shortlist, each tagged with a hypothesis (or "explore"), a
   level, and a one-sentence reason.

After experiment 250: one **held-out prediction** call (section 4.6).

[confirm] **The select step sees hypotheses and results, but no literature.** The literature
reaches the picks only through hypotheses. This keeps prompts small and makes every pick
traceable to a stated idea. It also means a paper can only affect a pick if the model turned it
into a hypothesis.

### 4.2 System prompt (fixed for the run; `docs/prompt_examples/none_system.txt`)
- Role: helping a cancer screening lab choose experiments, with both scores explained.
- Rules: base claims on the material given; cite only shown IDs; mark general-pharmacology claims
  as uncited; prefer specific, testable mechanisms; say when evidence is thin.
- Brevity: one sentence per statement or reason, two per rationale; JSON only.
- Reference block: every drug's class and targets (signed-off annotations), plus genetics
  summaries for the 29 visible lines only.

[confirm] Wording is our own (not iteration 1's). Sent as a cached prefix, so after the first
call it costs a tenth of the normal input price.

### 4.3 Hypothesize prompt (`none_round2_hypothesize.txt`)
- **Asks for:** up to 6 hypotheses about which shortlist pairs will score high on the joint
  score. Each has a statement, 1–2 drugs, a level, predicted responder lines (mechanistic
  only), a rationale and evidence IDs.
- **Levels:** `mechanistic` (a specific pathway in these cells), `cross_cancer` (shown in other
  tumor types), `statistical` (similar drugs did this in results so far).
- **Shows:** the shortlist with predicted mean and uncertainty; a results summary (best 8, worst
  4, per-pair summaries for up to 12 pairs involving shortlisted drugs, with within-line
  percentile); up to 12 existing hypotheses; up to 6 retired ideas; literature with the reading
  note.

[confirm] The shortlist shows the model's predicted mean and uncertainty, so the agent can see
BO's own view. This is deliberate: the question is whether it improves on that view, not
whether it can do without it.

### 4.4 Critique prompt (`none_round2_critique.txt`)
- Challenge each hypothesis against the literature and the results: claims the sources don't
  support, results that contradict it, mechanisms that don't fit the predicted lines.
- One verdict per hypothesis: keep, revise (with a new statement) or drop, with a one-sentence reason.
- Shows: the hypotheses being judged, the results summary, this round's literature plus earlier
  papers those hypotheses cite.

[confirm] A revised hypothesis keeps its ID, test history and score. Only the wording changes.

### 4.5 Select prompt (`none_round2_select.txt`)
- Choose exactly 10 by exp_id, balancing the most promising hypotheses against experiments
  that tell competing hypotheses apart. A few "explore" picks are allowed.
- Shows: the shortlist, the hypotheses that may be cited, the results summary.

### 4.6 Held-out prediction (`heldout_prediction.txt`)
- After the last pick: for each surviving mechanistic hypothesis, which of the 10 held-out
  lines should respond, from genetics and tissue only. No results exist for those lines in any
  prompt.
- Scored afterwards against the real outcomes in those lines, with a permutation test (shuffle
  which lines were predicted).

[confirm] Only mechanistic hypotheses are asked. If a run ends with very few of them, the test
for that run is weak. We'll know after the smoke run.

### 4.7 Answer checks and fallbacks (`ccagent/steps.py`)
- Each answer must match a JSON schema (sent to the API), and code then checks what a schema
  can't: cited IDs were shown, drugs and lines are on the shortlist, mechanistic hypotheses
  name responders, one verdict per hypothesis, and 10 distinct shortlist IDs.
- A bad answer gets one retry with its problems listed. If the retry also fails: hypothesize
  and critique add nothing that round; select falls back to BO's batch (the shortlist's first 10).
- Every fallback is logged, and the report counts them per arm.

[confirm] The fallback is BO, so a broken agent drifts toward BO's score rather than below it.
The report must state the fallback rate, or BO's picks could be credited to the agent.

## 5. Hypotheses (`ccagent/hypotheses.py`)

| Rule | Choice | Status |
|---|---|---|
| Coverage | A one-drug hypothesis covers any pair containing that drug; a two-drug one covers only that pair | [confirm]: the one-drug rule is broad (a drug appears in ~31 of the 583 pairs) |
| Score | Mean within-run percentile (on the joint score) of the experiments it covers | [fixed] |
| Status | Fewer than 4 tests: open; score ≥ 0.65: supported; ≤ 0.35: refuted; else open | [decided Nov 27]: minimum raised from 2 to 4 (no-effect ideas mislabelled 25% → 16% each way); 0.65 / 0.35 kept, untuned |
| Responder gap | Mechanistic only: mean percentile in predicted responder lines minus in other lines | [fixed] |
| IDs | Assigned by code, not the model | [fixed] |
| Limits per round | 6 new accepted, 12 shown, 6 retired reminders, 8 critiqued (new ones first) | [fixed]: keeps prompts flat; every new hypothesis is critiqued before it can be cited |
| Dropped | Kept for the record, no longer scored or shown | [fixed] |
| Explanation label of a pick | The level of the cited hypothesis; "statistical" if the hypothesis doesn't cover the pick's pair; "explore" if none | [fixed] (changed from iteration 1: affects 3–15 of 240 picks there) |

## 6. Literature (`ccagent/literature.py`)

| Item | Choice | Status |
|---|---|---|
| Source | PubMed, relevance order, fetched once into a cache (624 searches + class searches; 911 files); runs never touch the internet | [fixed] |
| Drug names searched | Dataset ID and generic name | [fixed] |
| Leakage exclusions | The O'Neil paper (PMID and DOI), papers naming O'Neil, the Merck screen or DrugComb, and computational synergy-prediction papers (`data/reference/excluded_sources.txt`) | [fixed] |
| Publication-date cutoff | None: 30 of the 69 cached papers that name two screen drugs and a screen cell line are from 2016 or earlier, so a cutoff would not remove the risk | [decided Nov 27]: say this in the limitations |
| Held-out line names in papers | Replaced with "[another cell line]" in every paper, in the agent and SHELDON (any spelling: HT-29, NCI-H23, ZR-75-1...); a guard refuses any paper text that still names one | [decided Nov 27] |
| Pairs per round | 3, with rotation (a pair waits 2 rounds before getting literature again) | v3 |
| Papers | 2 per drug (mechanism), 2 per pair; class papers fill a thin pair | v3 |
| Class terms | From each drug's annotated class (`drug_class_terms.csv`); mitomycin by name only; multi-target kinase inhibitors use their first two targets; paclitaxel "taxane" | v3, [confirm] |
| Low-information filter | Drops approval notices, pharmacokinetics, bioequivalence, formulation, errata, retractions by title (35 papers) | v3 |
| Reading note | Shown with literature: publication volume reflects study, not synergy; obscure compounds are as likely to score high; this lab's results outweigh papers; class evidence is about other members of the class | v3, [confirm] wording |
| Abstract excerpt | Last 800 characters, from a sentence start | v3 (decided Nov 27) |

[confirm] **v3 was designed after looking at the seed-0 pilot logs.** That's recorded. The
report should say so, and the pilot shouldn't be pooled with the new runs.

## 7. Ablations

| Arm | What changes | Question it answers | Planned seeds |
|---|---|---|---|
| No literature | Retrieval returns nothing; prompts otherwise identical (no reading note) | Does literature help? | 0–4 |
| Masked names | Every name a drug goes by (4,261 names from PubChem/ChEMBL incl. salt forms and brands) becomes "Drug NN", in tables and abstracts; codes shuffled | Is the effect from recognising famous drugs or from reading the papers? | 0–2 |
| No critique | Critique step skipped | Does self-critique earn its cost? | 0–4 |

**Decided Nov 27 (before any real run):** the two primary comparisons are agent vs BO and agent
vs no-literature, on strong finds (primary definition), reported as per-seed paired differences
with their range. Everything else is secondary. Five seeds detect only differences of roughly
20–25 strong finds, so process measures carry the "did literature change the reasoning"
question: famous-drug share of hypotheses, mechanistic share, citations per hypothesis, picks
testing a hypothesis vs explore, held-out accuracy. For masked names the main measure is the
famous-drug share; masking hides names, not identity (class and targets stay visible).

## 8. SHELDON arm (`ccagent/sheldon_arm.py`)

| Item | Choice | Status |
|---|---|---|
| Task text | Lab description, goal ("as many strong as you can within 250"), how the tools work, drug and line lists, the shared first batch | [confirm] |
| Tools | get_shortlist, run_experiments (≤10, latest shortlist only), results_so_far, drug_info, cell_line_info, drug_literature, pair_literature | [fixed]: same information as the agent |
| Removed | SHELDON's web search, web fetch, shell | [fixed]: would reach unfiltered papers |
| Model | Same model and effort as the agent (fixed via a subclass; SHELDON's code unedited) | [fixed] |
| Stopping | Budget spent, 40-round cap, or per-seed dollar cap | [fixed] |
| Literature excerpts | Same formatting as the agent (so section 1.1 applies here too) | follows 1.1 |

## 9. Model and cost controls

| Item | Choice | Status |
|---|---|---|
| Model | Claude Sonnet 5.5, adaptive thinking, effort medium; same for every arm and SHELDON | [fixed] |
| Gate | Real calls need `--allow-llm` and `LLM_API_KEY` | [fixed] |
| Spend cap | $30 per key, each with its own spend log (`--key 1`: LLM_API_KEY, runs/llm_spend.jsonl; `--key 2`: LLM_API_KEY_2, runs/llm_spend_key2.jsonl), checked before every call; a seed is skipped if a whole seed won't fit | [built Nov 27] |
| Finished seeds | Skipped by both run scripts (never paid for twice); the run log is written last, so a crashed seed is rerun; `--overwrite` forces a rerun | [built Nov 27] |
| Run order | Smoke run, literature 1–4, no literature 0–4, SHELDON short check, masked names 0–2: a budget shortfall cuts a masked seed, not a primary comparison | [decided Nov 27] |
| Repeatability | No temperature with adaptive thinking, so a rerun differs; every answer is logged | [report] |
| Logged | Every prompt, answer, problem, token count, cost and answering model | [fixed] |

## 10. Evaluation and analyses (to build, offline)

**Primary (named Nov 27, before any real run):** agent vs BO and agent vs no-literature, strong
finds (joint_within_line), per-seed paired differences over **seeds 0–9** with their range
(extended from 0–4 when a second $50 key became available). Seeds are dropped only for cost,
never on results; the report states how many ran.

| Analysis | What it shows |
|---|---|
| Strong found, mean ± SD per method | Headline table (sample SD) |
| Paired differences by seed | The two primary comparisons; the rest secondary |
| Discovery curves and area under them | How fast each method finds strong combinations |
| Famous vs obscure | Famous-drug share of hypotheses and picks, strong finds by pair group (definition fixed before runs) |
| Hypothesis process | Mechanistic / cross-cancer / statistical share; one-drug vs two-drug share; citations per hypothesis; status counts |
| Responder honesty | Whether each named responder's genetics show the stated feature; responders per hypothesis |
| Mechanism test | Held-out predictions vs real outcomes, permutation test; split by in-run status; count of hypotheses behind each number |
| Picks | Share testing a hypothesis vs explore; distinct pairs; final explanation level after downgrades |
| Answer checks | Retry rate per step; critique failures; select fallback rate (BO's batch); fallback-model answers |
| Critique | Keep / revise / drop rates per arm |
| Literature | Papers per pair group; redactions of held-out lines per run |
| SHELDON (2 seeds) | Strong finds, experiments used, cost, strong finds per dollar |
| Sensitivity | Secondary strong definitions, Bliss synergy, duplicate-screen handling, kappa |
| Cost | Dollars and tokens per seed per arm |

## 11. Limitations to state in the report
- 10 seeds for the two primary arms, 5 for the ablations, 2 for SHELDON; ten seeds detect
  differences of roughly 14–17 strong finds (five, roughly 20–25), so process measures carry
  much of the argument.
- Much of the gain over random comes from the shortlist: random-from-shortlist finds 39.6 ±
  12.0 against BO's 47.3 ± 10.5 over seeds 0–9 (on seeds 0–4 alone they looked equal).
- One screen, one model, one prompt design; a simulation is not a real lab.
- Literature v3 was designed after seeing the seed-0 pilot logs; seed 0 is also a test seed.
- No date cutoff on literature: the risk it would address is papers on held-out lines, and
  30 of 69 such papers predate 2016; held-out line names are redacted instead.
- The model may know published results for these drugs and lines from its training,
  separate from the provided papers; the masked-names arm checks name recognition only, since
  class and targets stay visible.
- The in-run responder gap uses results the model may have seen; only the held-out
  prediction is a clean mechanism test.
- The same model critiques its own hypotheses; the no-critique arm measures whether it helps.
- Status cutoffs (0.65 / 0.35) are untuned; the 4-test minimum was set from a simulation.
- Runs are not exactly repeatable (no temperature with adaptive thinking); every answer is logged.
- SHELDON is compared on strong finds, picks and cost only.
