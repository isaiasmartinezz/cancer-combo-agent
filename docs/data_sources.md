# Data sources

Every dataset used in this project, where it came from, and how to rebuild it.
Raw and processed data are not committed (see `.gitignore`); scripts rebuild them.

## Drug-combination screen
- **Study:** O'Neil et al., "An Unbiased Oncology Compound Screen to Identify Novel Combination Strategies", Molecular Cancer Therapeutics, 2016 (doi 10.1158/1535-7163.MCT-15-0843).
- **Obtained through:** DrugComb v1.5, `summary_v_1_5.csv`, Zenodo record 15235991 (https://zenodo.org/records/15235991). Rows kept: `study_name == "ONEIL"`.
- **File checksum (full DrugComb file, md5):** `cd3e24758c10ffaea0124b6e80626c32`
- **DrugComb reference:** Zheng et al., Nucleic Acids Research, 2021 (doi 10.1093/nar/gkab438).
- **Size in iteration 1:** 38 drugs, 39 cell lines, 22,737 drug-pair/cell-line experiments.
- **Script:** to be added in the data step (`scripts/fetch_oneil.py`).
