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
