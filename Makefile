.PHONY: setup lint test check
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
