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
