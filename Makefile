.PHONY: setup lint test check data
PY ?= python

setup:
	python3.11 -m venv .venv
	.venv/bin/pip install -r requirements.txt

lint:
	$(PY) -m ruff check .

test:
	$(PY) -m pytest

# Run before every commit. Offline only: no network, no LLM calls.
check: lint test

# Rebuild the dataset. Downloads only if the raw file is missing (needs the internet).
data:
	test -f data/raw/drugcomb_oneil.csv || $(PY) -m scripts.fetch_oneil
	$(PY) -m scripts.build_dataset
