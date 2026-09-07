.PHONY: help setup doctor run digest week verify review test coverage smoke smoke-quick clean

# The interpreter every target runs. Bare `python` is right once the venv is
# activated and in CI; elsewhere pass the venv's explicitly:
#   make coverage PYTHON=.venv/bin/python
PYTHON ?= python

help:
	@echo "make setup    install dependencies"
	@echo "make doctor   preflight everything"
	@echo "make run      process the inbox"
	@echo "make digest   combined digest to stdout"
	@echo "make week     write this week's digest to data/outbox"
	@echo "make verify   confirm every artifact still opens"
	@echo "make review   what the review cadence says is due"
	@echo "make test     run the test suite"
	@echo "make coverage the suite with line+branch coverage, failing under the floor"
	@echo "make smoke    drive every CLI route against a throwaway project"
	@echo "make smoke-quick  the same routes, one check each"
	@echo "make clean    remove work scratch and caches"

setup:
	$(PYTHON) -m pip install -r requirements.txt

doctor:
	$(PYTHON) run.py doctor

run:
	$(PYTHON) run.py run

digest:
	$(PYTHON) run.py digest

week:
	$(PYTHON) run.py digest --days 7 --out data/outbox/digest-$$(date +%Y-%m-%d).md

week-html:
	$(PYTHON) run.py digest --days 7 --format html --out data/outbox/digest-$$(date +%Y-%m-%d).html

verify:
	$(PYTHON) run.py verify

review:
	$(PYTHON) run.py review

test:
	$(PYTHON) -m pytest tests/ -q

# Line AND branch coverage, and a floor CI enforces. The number is the last
# measured value rounded down, never a hope: raise it when the measurement
# rises, and never lower it to make a build pass.
coverage:
	$(PYTHON) -m pytest tests/ -q --cov=src/plaud_bridge --cov-branch \
		--cov-report=term-missing --cov-fail-under=$(COVERAGE_FLOOR)

# Measured 2026-09-07: 9346 statements, 15 missed; 3050 branches, 76
# partial; 99.27% combined. Rounded down, and it only moves up.
COVERAGE_FLOOR ?= 99

smoke:
	$(PYTHON) scripts/smoke.py

smoke-quick:
	$(PYTHON) scripts/smoke.py --quick

clean:
	rm -rf data/work .pytest_cache
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
