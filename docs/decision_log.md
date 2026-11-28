# Decision log

One entry per design choice: what was decided, why, and what it replaces. Newest last.

## 2026-11-25: Rebuild in a fresh repository
**Decision:** The project is rebuilt step by step in this repository. The earlier repository (CS224V_Final_Project) is treated as "iteration 1": a reference for design and results, not a source to copy from.
**Why:** Iteration 1 grew fast. Rebuilding lets every piece be re-derived, tested, and documented on purpose.

## 2026-11-25: No real model calls until explicitly enabled
**Decision:** All code talks to a model through one interface (`ccagent/llm`). The default is an offline fake. A real client needs both the `--allow-llm` flag and an API key in the environment. Tests block network access entirely.
**Why:** Keeps cost and leakage risks out of development, and makes every test reproducible.

## 2026-11-25: Data source = DrugComb v1.5 (Zenodo mirror)
**Decision:** The O'Neil 2016 screen is taken from DrugComb v1.5 `summary_v_1_5.csv` (Zenodo record 15235991), keeping rows with `study_name == "ONEIL"`. Details in `docs/data_sources.md`.
**Why:** Same source as iteration 1, so results stay comparable. drugcomb.org itself was unreachable when the data was first fetched.

## 2026-11-25: Drug IDs and names
**Decision:** Drugs keep the screen's IDs (e.g. ZOLINZA, MK-4827) as their key. `data/reference/drugs.csv` adds a generic name (vorinostat, niraparib) for literature search.
**Why:** The IDs match the source data and iteration 1. Generic names find far more literature than internal codes.

## 2026-11-25: Replicates averaged, spread kept
**Decision:** Replicates of the same experiment are averaged (as in iteration 1). The replicate standard deviation of Loewe synergy and of inhibition is kept.
**Why:** The spread shows how noisy a single "strong" label is (median Loewe SD 2.4).

## 2026-11-25: Duplicate screening runs averaged and flagged
**Decision:** 315 experiments screened in two runs are averaged like the rest and flagged in `duplicate_screen`. The main result is to be rerun without them as a sensitivity check.
**Why:** The second run reads systematically lower (see `docs/data_sources.md`). Averaging keeps the dataset identical to iteration 1, and the flag makes the check possible.

## 2026-11-25: Same held-out cell lines as iteration 1
**Decision:** The held-out procedure (10 lines, stratified by tissue, seed 2026) is reimplemented. The build fails if it stops reproducing the committed list.
**Why:** Keeps the mechanism test comparable with iteration 1.

## 2026-11-25: Targets from ChEMBL, not DrugComb
**Decision:** DrugComb's target-name columns are dropped.
**Why:** They contain implausible targets (for example, thrombopoietin for 5-FU).

## 2026-11-25: Lab, loop, and "strong" label rebuilt; verified against iteration 1
**Decision:** `ccagent/lab.py` is the only way to learn outcomes. It hides held-out lines and refuses repeats, overspending, and unknown IDs, and it returns only synergy (Loewe) and inhibition. It has no method that reveals untested outcomes; evaluation (`ccagent/evaluation.py`) reads the data separately. `ccagent/loop.py` runs every method under the same rules: 250 experiments, batches of 10, the same random first batch per seed. `ccagent/strong.py` keeps the definition agreed with the research team (top 10% Loewe AND CSS cutoff giving 5%, ranked within cell line; secondary: same rule over all experiments).
**Check:** Random selection reproduces iteration 1 exactly for seeds 0 to 4: the same 250 picks, the same strong flags, and the same counts (11, 8, 8, 16, 10). `tests/test_reproduce.py` locks this in.

## 2026-11-25: Drug annotations: fetched, curated, and signed off as three separate tables
**Decision:** `scripts/fetch_drug_annotations.py` re-fetches structures (PubChem) and targets/mechanisms (ChEMBL) with iteration 1's matching rules: exact name, then exact synonym, then PubChem's ChEMBL cross-reference, never free-text search. Mechanisms are looked up by parent compound. The result is committed as `drug_annotations_fetched.csv` with its fetch date. Manual choices live in `data/reference/drug_curation.csv`, each with a source: plain-words drug class, and targets/mechanism from cited abstracts for the 4 drugs ChEMBL lacks (GELDANAMYCIN, L778123, MK-4541, MRK-003). Literature can only fill a drug with no ChEMBL mechanism; the build refuses otherwise. `drug_signoff.csv` records Isaias's review of each row (pending / ok / fixed, reviewer, date).
**Why:** Every value is traceable to a database or a citation, manual choices are reviewable as data instead of code, and no annotation counts as checked until a person has read it. In iteration 1 all 38 rows were still "needs sign-off".
**Known gap:** the 4 literature drugs have no target gene symbols (ChEMBL has none), so expression features are unknown for them.

## 2026-11-25: Drug annotation review
**Decision:** The 38 annotations were reviewed by an AI check: the 4 literature-sourced drugs against their 8 cited abstracts, the 3 indirect ChEMBL matches against the fetched records, and the rest for name/ID consistency plus standard pharmacology. `drug_signoff.csv` records this as "AI check, not yet confirmed by a person" until Isaias confirms.
**Changed:** L778123's mechanism no longer says it "blocks Ras prenylation"; PMID 12479371 found no K-Ras prenylation inhibition in patients.
**Open:** PACLITAXEL stays pending. "Tubulin inhibitor" (ChEMBL's wording) does not distinguish microtubule stabilizers (taxanes) from destabilizers (vinca alkaloids); a cited wording should replace it before agent runs.

## 2026-11-25: Cell-line genetics rebuilt; matching decisions moved to data
**Decision:** Automatic matching to Cell Model Passports needs a unique name match and agreeing tissue. The 4 lines that need judgment live in `data/reference/cell_line_overrides.csv` with reasons. Each override states the CMP model name it expects, so a changed CMP release fails the build instead of silently changing genetics. Features: driver-mutation panel (genes mutated in at least 3 of our lines, plus mutated drug targets), amplification/deletion of drug-target genes, mean target-gene expression per line and drug, and a one-line summary per line for the LLM.
**Check:** With the Nov 13 CMP files, every feature value, missing-value pattern, and summary is identical to iteration 1.
**Known limitation:** For drugs whose ChEMBL target is a large complex (metformin: 51 genes; bortezomib: 38), "target expression" averages over every subunit. To be revisited in the response-model step.

## 2026-11-25: Response model and baselines rebuilt
**Decision:** The response model is a random forest (100 trees, `max_features` 0.3), retrained from scratch each round; the uncertainty is the spread across trees. Features: both drugs' Morgan fingerprints (radius 2, 256 bits), ChEMBL target multi-hot, and drug identity, all summed over the pair so order cannot matter; plus cell-line identity, tissue, Cell Model Passports genetics, and the smaller and larger of the two drugs' target-gene expression in that line. Methods: `uncertainty` (highest spread), `bo` (highest predicted value + kappa x spread, kappa = 1.0), and `shortlist_random` (10 at random from the agent's own 50-candidate shortlist).
**Why the shortlist matters:** the shortlist is ordered by the same score BO uses, so its first 10 rows *are* BO's batch. The agent picks from that same list, which makes "agent vs BO" and "agent vs random-from-shortlist" a fair pair of comparisons rather than a comparison of different candidate pools. A test locks this property in.
**Check:** Run against iteration 1's code in one environment, the rebuilt code makes identical picks and identical strong flags (bo and uncertainty, seed 0).

## 2026-11-25: Library versions pinned; iteration 1's numbers depend on them
**Decision:** `requirements.txt` pins numpy, pandas, scipy, scikit-learn and rdkit to the versions that produced the reported results (scikit-learn 1.9.1).
**Why:** The random forest's output changes with the scikit-learn version. Run under scikit-learn 1.7.2, iteration 1's *own* code gives 30 strong combinations for BO on seed 0 instead of the reported 40, and 27 instead of 16 for uncertainty sampling. The code is unchanged; only the library differs. Any reported number is therefore tied to the pinned environment, and the write-up should say which versions were used.
**Open:** confirm on the recording machine (scikit-learn 1.9.1) that the rebuilt baselines reproduce 10.6 / 15.4 / 45.2 / 47.2 across seeds 0-4.

## 2026-11-26: Baselines reproduce iteration 1 exactly
**Result:** On the recording machine (scikit-learn 1.9.1), all four methods reproduce iteration 1 seed for seed, not just on average: random 11, 8, 8, 16, 10; uncertainty 16, 11, 19, 19, 12; BO 40, 41, 43, 51, 61; random-from-shortlist 62, 25, 58, 41, 40. The genetics also rebuild identically from a fresh Cell Model Passports download, so the pipeline is verified end to end from its live sources.

## 2026-11-26: Spread reported as the sample standard deviation
**Decision:** Variation across seeds is reported with the sample standard deviation (dividing by n-1), not the population one (dividing by n) that iteration 1 used.
**Effect on reported numbers:** random 2.9 -> 3.3, uncertainty 3.4 -> 3.8, BO 7.9 -> 8.8, random-from-shortlist 13.4 -> 15.0. The means are unchanged. The Nov 13 and Nov 20 updates to the research team quote the old figures, so the write-up should state the convention.
**Why:** With 5 runs, dividing by n understates how much the result varies between seeds. The wider figures also make the headline comparison plainer: BO (47.2 +/- 8.8) and random-from-shortlist (45.2 +/- 15.0) are indistinguishable, so the shortlist, not the optimizer's ranking, is doing most of the work.

## 2026-11-26: Agent literature layer (`ccagent/literature.py`)
**Decision:** PubMed (E-utilities, relevance order) behind a disk cache, and a separate filter for what the agent may see: papers with an abstract, one per title, none matching `data/reference/excluded_sources.txt` (the O'Neil paper by PMID and DOI, papers naming O'Neil, the Merck screen or DrugComb, and computational synergy-prediction papers). Drugs are searched by their dataset ID and generic name (`drugs.csv`). Single-drug search uses title matches first, topped up from abstract matches.
**Offline by default:** a cache miss raises an error unless the caller passes `allow_network=True`, which only the prefetch step does. An agent run therefore never reaches the internet and always sees the same papers.
**Why:** Same retrieval and exclusion rules as iteration 1, so the two existing agent runs stay comparable, and a fixed cache makes reruns reproducible. Every exclusion is logged with its rule, for the run audit.
**Check:** Over iteration 1's 331 cached searches, the rebuilt filter keeps exactly the same papers in the same order as iteration 1's code (643 papers; 4 distinct papers excluded).

## 2026-11-26: Hypothesis list with size limits (`ccagent/hypotheses.py`)
**Decision:** Every hypothesis is stored, but each round the model sees a bounded slice: at most 6 new hypotheses accepted, 12 active ones shown (about drugs on the shortlist, supported first), 6 one-line reminders of failed ideas, and 8 sent to critique (new ones first, then ones with new results since their last critique). The book assigns hypothesis IDs itself. Status comes from returned results only: mean percentile, on the method's target, of the experiments a hypothesis covers; fewer than 2 tests = open, >= 0.65 supported, <= 0.35 refuted. Iteration 1's untuned cutoffs are kept for comparability.
**Why:** Iteration 1's first version showed and critiqued every hypothesis every round (100+ by the end), so prompts grew from about 12k to 43k tokens and runs cost $6-8. Because at most 6 new hypotheses are accepted and 8 are critiqued, every new hypothesis is critiqued before a pick can cite it.
**Check:** Replaying iteration 1's four real agent runs (490 hypotheses) through the rebuilt list gives the same IDs, statuses, test counts, scores and responder gaps for every hypothesis. For the two runs with iteration 1's size limits, it also picks exactly the hypotheses iteration 1 sent to critique in every round. The two earlier unlimited runs differ there, as expected.

## 2026-11-26: Agent prompts, answer checks, and masked names (`ccagent/prompts.py`, `steps.py`, `masking.py`)
**Decision:** The model gets one fixed reference block per run (every drug's class and targets; genetics of the visible cell lines only) and three task prompts (hypothesize, critique, select). It answers in JSON. Code checks what a format cannot: cited IDs were shown, drugs and cell lines are on the shortlist, mechanistic hypotheses name responder lines, one verdict per critiqued hypothesis, exactly k distinct shortlist IDs. A failed answer gets one retry with its problems listed. After a second failure, hypothesize and critique add nothing that round, and select falls back to the shortlist's first k rows, which is BO's batch.
**Leakage controls, each with a test:** the results section is built from a fixed list of returned columns plus the target, so strong flags, cutoffs or hidden values cannot reach a prompt even if a caller passes them; prompt building refuses any shortlist or results table containing a held-out line; the reference block lists visible lines only.
**Masked-name ablation:** every name a drug goes by (dataset ID, DrugComb name, generic name, ChEMBL name) becomes "Drug NN", in tables and in literature text. Codes are shuffled with a fixed seed so they carry no alphabetical hint. Answers are unmasked at the boundary, so stored hypotheses and run logs use real IDs. Known limit: a brand name or synonym not in the tables can still slip through an abstract.
**Changed from iteration 1:** a pick that cites a hypothesis not covering its drug pair is logged as statistical, not mechanistic or cross-cancer. In iteration 1's two size-limited runs this affects 3 and 15 of 240 picks.
**Check:** On all 303 real model answers logged by iteration 1's four agent runs, the rebuilt checks accept and reject exactly the same answers as iteration 1's checks did (19 rejected on the first attempt).

## 2026-11-26: The agent as a chooser in the shared loop (`ccagent/agent.py`)
**Decision:** The agent is one more chooser for `run_selection`, so it runs under exactly the same rules as the baselines (shared first batch, 25 batches of 10, the lab's checks). Each round: shortlist (the same `make_shortlist` call, model and settings as the random-from-shortlist control), cached literature for the first 3 distinct drug pairs on the shortlist (2 mechanism papers per drug, 2 papers on the pair), hypothesize, critique, select. Ablations: `no_literature`, `no_critique`, `masked_names`. Every model call (prompt, answer, problems, tokens, cost) is kept for the run's call log, and the hypothesis list is saved with its event history.
**Literature cache:** `scripts/prefetch_literature.py` fetches every search the agent can make (38 drugs, every drug pair in the screen) once; agent runs then read the cache only.
**Check:** With a scripted model that always selects the shortlist's first 10 rows, the agent reproduces BO pick for pick on seeds 0-4 of the real data (same 250 experiments in the same order; 40, 41, 43, 51, 61 strong). This shows the agent and the baselines see the same candidates, and that any difference in a real run comes from the model's choices. Offline tests repeat the check on a synthetic lab for every ablation, and check that held-out lines and outcome labels never reach a prompt, that masked runs show no drug names, that the shown hypotheses stay capped, and that unusable answers fall back to BO.
**Literature check:** After the prefetch (38 drugs, 583 drug pairs, 624 cached searches; 4 excluded papers seen), the same scripted run with literature on, and with masked names, again reproduces BO on seeds 0-4, with literature in every hypothesize prompt and no cache misses.
**Found:** in masked runs, brand names not in the names table slip through abstracts: Tarceva (erlotinib), Taxol (paclitaxel) and Adriamycin (doxorubicin) appear in 80, 62 and 48 prompts over five seeds. The masked-name ablation is not valid until these are masked too.

## 2026-11-26: Masked names cover every listed synonym (`ccagent/data/synonyms.py`, `scripts/fetch_drug_synonyms.py`)
**Decision:** The masked-name ablation masks every name PubChem lists for the drug's CID and every name ChEMBL lists for the parent molecule and all its salt forms (`data/reference/drug_synonyms.csv`, fetched 2026-11-26, each name with its source). ChEMBL's salt forms are needed because brand names are often filed under the salt: Tarceva is erlotinib hydrochloride, a different PubChem CID from erlotinib. Dropped, with the reason recorded: names under 4 characters, bare numbers, names listed for two of our drugs (none), and phrases that contain a shorter name of the same drug (for example "CHK1 inhibitor MK-8776", so "CHK1 inhibitor" stays visible). 4,261 names kept, 1,365 dropped.
**Also fixed:** a name joined to another word by a hyphen ("paclitaxel-loaded", "Adavosertib-encapsulated") was not masked, because the word-boundary rule treated hyphens as part of the word. Boundaries are now letters and digits only; "PSN-389" still does not match "SN-38".
**Check:** Over five scripted masked runs, no generic name, dataset ID or common brand name appears anywhere in a prompt, even as a raw substring, except inside one related compound's name (17-O-demethylgeldanamycin, 4 prompts). The masked run still reproduces BO on seeds 0-4. Four kept names are also English words (ASTA, Vespid, Taxus, TAXUS); masking them costs nothing.

## 2026-11-26: Real model client, spend log, and budget cap (`ccagent/llm/real.py`, `settings.py`)
**Decision:** The real client is reachable only through the existing gate (`--allow-llm` plus `LLM_API_KEY`) and additionally refuses to start until `AGENT_MODEL` is set in `ccagent/llm/settings.py`, so the model choice has to be made and recorded before any real run. Each call sends the fixed system prompt (instructions plus reference block) as a cached prefix, uses adaptive thinking at a fixed effort, asks for the step's JSON schema (code still checks everything a schema cannot), and opts into the API's server-side fallback for requests a safety classifier declines. A declined or truncated answer is treated as invalid, so the step retries once and then falls back like any other bad answer. The model that actually answered is logged with every call.
**Spend:** every real call is priced from its token counts and appended to `runs/llm_spend.jsonl`. A call is refused once the logged total reaches the budget ($30, the study budget set in iteration 1), and `run_agent` skips a seed whose expected cost (default $3) would pass it. An answer from a model not in the price table is priced at the highest list price, so spend is never undercounted.
**Check:** tested against a stand-in for the provider SDK (request shape, cost arithmetic including cache reads and writes, refusal handling, fallback logging, budget stop, ledger summed across runs). The provider SDK is pinned in `requirements.txt` but not installed in the development environment; no real call has been made.

## 2026-11-26: Agent model fixed: Claude Sonnet 5.5, effort medium
**Decision (Isaias):** every real agent run uses `claude-sonnet-5-5` with adaptive thinking at effort `medium`, set once in `ccagent/llm/settings.py`.
**Why:** It is the setting iteration 1's two agent runs used, so they stay comparable, and at about $2-3 per run with the hypothesis cap, five seeds on two arms fits the $30 budget. If the model or effort ever changes, every earlier agent result is rerun.

## 2026-11-27: SHELDON added as a method arm (Isaias)
**Decision (Isaias):** SHELDON (Joshi, Shethia & Lam 2026; stanford-oval/sheldon, release 7fbf027) runs as a further method on the same simulated lab, registered as the SHELDON benchmark `oneil-combo` (`ccagent/sheldon_arm.py`, `scripts/run_sheldon.py`). This reverses the handoff's note that SHELDON was dropped.
**Why:** It is a published hypothesis-and-observation search harness, so it tests whether a general scientific-reasoning loop does as well as our purpose-built agent with the same information.

## 2026-11-27: SHELDON gets the agent's information as tools, and nothing else
**Decision:** SHELDON sees the shared first batch in its task and has seven tools: the same 50-candidate shortlist (BO order, same response model), run up to 10 experiments from the latest shortlist, the agent's results summary, drug class/targets, genetics for visible lines, and the agent's cached literature (2 papers per drug or pair query). SHELDON's web search, web fetch and shell are removed (`probes = []`, checked when the config is built).
**Why:** The comparison is about how the information is used, not what is available. Restricting picks to the current shortlist keeps SHELDON's choice set the same as the agent's; a free-text literature search was not offered because it would reach papers outside the leakage-filtered cache.
**Check:** a scripted SHELDON that runs the shortlist's first 10 each time reproduces BO pick for pick on a synthetic lab; refused calls spend no budget; held-out lines and outcome labels appear in no tool output.

## 2026-11-27: SHELDON runs at the agent's model and effort, priced at list
**Decision:** SHELDON's tables predate Sonnet 5.5, so without a fix it would send no effort setting (the model then runs at its default, high) and price calls at its fallback rate ($2.50/$15). A subclass of SHELDON's client sets adaptive thinking and effort `medium` for our model, and the list price ($2/$10) is added; SHELDON's own code is not edited. Its `cache_control` field is passed through `extra_body` because the SDK's `parse()` does not accept it.
**Why:** Same model and setting as the agent, or the two arms are not comparable. Iteration 1 found and fixed the same two gaps (re-derived here from SHELDON's source).

## 2026-11-27: SHELDON runs until the experiment budget is spent
**Decision:** The round cap is 40 and the loop stops when the 250 experiments are used (SHELDON's `objective_exhausted`), on a per-seed dollar cap, or when SHELDON has nothing left to test. Spend is recorded in the shared spend log at list prices, so the $30 budget counts it.
**Why:** Iteration 1's single SHELDON run stopped at its 8-round cap after 118 of 250 experiments (20 strong, $3.91), so it was not comparable with methods that used all 250. Where a run still stops early, it is reported with the experiments it used.

## 2026-11-27: Held-out prediction step added to the agent
**Decision:** after the 250th experiment, the agent makes one more call: for every mechanistic hypothesis still standing, which held-out lines should respond, from their names, tissue and genetics only (`AgentChooser.predict_heldout`, saved as `_heldout_predictions.json`). Same checks as the other steps (one prediction per hypothesis, held-out lines only, one retry). The masked arm sees codes, not drug names.
**Why:** the plan's mechanism test (section 9.3) needs these predictions from every real run; adding them later would mean paying for the runs twice. It is the only prompt that names a held-out line, and it comes after the last pick, so it cannot affect selection.

## 2026-11-27: Literature design v3 (Isaias asked to improve the literature arm)
**Diagnosis (iteration 1, seed 0, literature arm against no literature):** with literature, 63% of hypothesis drug mentions were famous drugs, against 36% without; picks on famous pairs hit 17% against 47% for the no-literature arm's obscure pairs. Every drug got two mechanism papers, so the gap is not missing drug papers: papers naming *both* drugs averaged 2.0 per famous pair, 1.3 per mixed pair and 0.25 per pair of research compounds, and 53 of 288 mechanism papers were approval notices, pharmacokinetics or formulation papers.
**Decision:** three changes to what the literature arm sees, none tuned on any score:
1. *Class evidence:* when fewer than 2 papers name both drugs, the rest are papers on the two drug classes used together (class terms in `data/reference/drug_class_terms.csv`, taken from each drug's annotated class wording; drugs whose class is too general, e.g. mitomycin, are searched by name only; paclitaxel uses "taxane" while its annotation is pending). Labelled "their drug classes together (not these exact compounds)".
2. *Low-information titles dropped* (approval notices, pharmacokinetics, bioequivalence, formulations, errata), logged separately from leakage exclusions.
3. *Rotation:* a pair that got literature waits two rounds before getting it again, so evidence spreads over the shortlist instead of re-reading the top pairs.
Plus a reading note shown with the literature: publication volume reflects how much a drug is studied, not how well it combines; the lab's results outweigh papers.
**Why:** the literature arm was steering toward drugs that are written about, not drugs that combine well in this screen. The no-literature arm's prompts are unchanged, so with versus without literature still isolates the literature.
**Caveat for the report:** the diagnosis used iteration 1's seed 0, which is also one of the five test seeds. No setting was chosen by its score, but the report should say the design was motivated by seed 0 and give the per-seed results. SHELDON's pair tool gets the same class top-up and reading note.

## 2026-11-27: Drug annotations confirmed by a person
**Decision (Isaias):** Isaias reviewed the annotations and confirmed them; `drug_signoff.csv` now records Isaias as reviewer for all 38 rows (37 ok, 1 fixed), and `drug_annotations.csv` was rebuilt from it. Paclitaxel was the last one: its class already reads "taxane" (ATC L01CD01), which separates the microtubule stabilizer from the vinca alkaloids (destabilizers, ATC L01CA), so it was marked ok with the wording unchanged. The literature class search uses "taxane" only, since "tubulin inhibitor" would also match the vinca alkaloids.
**Why:** the handoff recorded the annotations as AI-checked only; the report can now say they were checked by a person.

## 2026-11-27: Iteration 1's agent and SHELDON runs reported as pilot results
**Decision (Isaias):** iteration 1's seed-0 agent runs (v1 and v2 memory, with and without literature) and its SHELDON run are kept and reported as pilot results, labelled as an earlier code version, not pooled with the new runs. No money is spent rerunning them.
**Why:** they cost about $22 and motivated the literature v3 design; they are a different code version and library environment, so they inform the story but not the headline comparison.

## 2026-11-27: SHELDON stays in, following the original plan (Isaias)
**Decision (Isaias):** SHELDON is part of the project as in the Nov 2 plan; the handoff's "dropped" note is set aside.
