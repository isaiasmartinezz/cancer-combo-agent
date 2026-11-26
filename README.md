# Cancer-Combo-Agent

Literature-informed experiment selection for cancer drug combinations.
Stanford CS 224V, Fall 2026.

A simulated lab built from the O'Neil 2016 drug-combination screen lets different
methods choose which combinations to test under a fixed experiment budget. We compare
random selection, uncertainty sampling, Bayesian optimization, and a literature-informed
agent on how many strong combinations each one finds.

## Setup
```bash
make setup
source .venv/bin/activate
```

## Checks
```bash
make check    # lint + tests; offline only
```

Real model calls are off by default. They require the `--allow-llm` flag and an API key
(see `.env.example`).

## Reproducing the baselines
```bash
make data                                   # build the experiment table
python -m scripts.build_cell_lines          # cell-line genetics
python -m scripts.run_baselines             # 4 methods x 5 seeds, about 7 minutes
```
Results depend on the pinned library versions in `requirements.txt`; see `docs/decision_log.md`.

## Layout
- `ccagent/`: source code
- `tests/`: offline tests
- `docs/`: decision log and data sources
- `scripts/`: data and run scripts
