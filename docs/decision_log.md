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
