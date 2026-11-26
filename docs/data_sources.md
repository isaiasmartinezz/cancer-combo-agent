# Data sources

Every dataset used in this project, where it came from, and how to rebuild it.
Raw and processed data are not committed (see `.gitignore`); scripts rebuild them.

## Drug-combination screen
- **Study:** O'Neil et al., "An Unbiased Oncology Compound Screen to Identify Novel Combination Strategies", Molecular Cancer Therapeutics, 2016 (doi 10.1158/1535-7163.MCT-15-0843).
- **Obtained through:** DrugComb v1.5, `summary_v_1_5.csv`, Zenodo record 15235991 (https://zenodo.org/records/15235991). Rows kept: `study_name == "ONEIL"`.
- **File checksum (full DrugComb file, md5):** `cd3e24758c10ffaea0124b6e80626c32`
- **DrugComb reference:** Zheng et al., Nucleic Acids Research, 2021 (doi 10.1093/nar/gkab438).
- **Rebuild:** `make data` (downloads only if `data/raw/drugcomb_oneil.csv` is missing).

### What the build produces (`data/processed/`, numbers from `build_report.json`)
| | |
|---|---|
| Raw O'Neil rows | 92,208 (all drug combinations; no single-drug rows) |
| Experiments (drug pair x cell line) | 22,737 |
| Drugs / cell lines / drug pairs | 38 / 39 / 583 |
| Rows dropped for a missing outcome | 0 |
| Replicates per experiment | 4 (22,422 experiments) or 8 (315) |
| Median replicate SD of Loewe synergy | 2.4 |
| Held-out cell lines | 10 (`data/reference/heldout_lines.txt`) |
| Experiments the agent can choose from | 16,907 |

Columns of `oneil.csv`: `exp_id`, `drug_a`, `drug_b` (alphabetical), `cell_line`, `tissue`,
`inhibition` (DrugComb `css_ri`), `synergy_loewe`, `synergy_bliss`, `synergy_zip`, `synergy_hsa`
(replicate means), `n_replicates`, `synergy_loewe_sd`, `inhibition_sd`, `duplicate_screen`.

### Known issues
- **Duplicate screening runs.** 315 experiments (21 drug pairs x 15 cell lines) were screened in
  two runs of 4 replicates. The second run reads lower on average (Loewe about 8 points, CSS about
  11.5 points; run-to-run correlation 0.79 for Loewe), and the replicate SD for these experiments is
  7.6 versus 2.4 elsewhere. They are averaged and flagged in `duplicate_screen`.
- **DrugComb target names are not used.** The `drug_*_target_name` columns contain implausible
  targets (for example, thrombopoietin listed for 5-FU). Drug targets come from ChEMBL instead.

## Reference tables (`data/reference/`, committed)
- `drugs.csv`: each DrugComb drug name mapped to the screen's drug ID, a generic name used for
  literature search, and its PubChem CID. 36 of 38 DrugComb names resolve to the same PubChem CID;
  the two platinum drugs resolve to a second PubChem record of the same compound (see `note`).
- `heldout_lines.txt`: the 10 held-out cell lines (see `ccagent/data/split.py`).

## Cell-line genetics
- **Source:** Cell Model Passports, Wellcome Sanger Institute (https://cellmodelpassports.sanger.ac.uk), public files at https://cog.sanger.ac.uk/cmp/download/: `model_list_latest`, `mutations_summary_latest` (cancer-driver mutations), `cnv_summary_latest`, `rnaseq_latest` (streamed; only our models kept). The DepMap portal was not used: its automated downloads sit behind bot verification.
- **Rebuild:** `python -m scripts.fetch_cell_lines` (internet; records sizes and sha256 in `data/reference/cmp_manifest.json`), then `python -m scripts.build_cell_lines` (offline).
- **Matching** (`data/reference/cell_line_map.csv`): 35 of 39 lines match by name (ignoring case and punctuation) with agreeing tissue. 4 are decided by hand in `cell_line_overrides.csv`: MSTO = MSTO-211H, LNCAP = LNCaP-Clone-FGC, UWB1289+BRCA1 = UWB1.289 profile with BRCA1 unknown, COLO320DM = no confident match (genetics unknown).
- **Coverage:** 34 lines have mutation data. Unknown values stay unknown, never filled with guesses.
- **Note:** CMP classifies MSTO-211H as mesothelioma; the O'Neil screen lists it under lung.
