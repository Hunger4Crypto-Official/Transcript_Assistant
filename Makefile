.PHONY: help setup doctor run digest week verify review test smoke smoke-quick clean

help:
	@echo "make setup    install dependencies"
	@echo "make doctor   preflight everything"
	@echo "make run      process the inbox"
	@echo "make digest   combined digest to stdout"
	@echo "make week     write this week's digest to data/outbox"
	@echo "make verify   confirm every artifact still opens"
	@echo "make review   what the review cadence says is due"
	@echo "make test     run the test suite"
	@echo "make smoke    drive every CLI route against a throwaway project"
	@echo "make smoke-quick  the same routes, one check each"
	@echo "make clean    remove work scratch and caches"

setup:
	pip install -r requirements.txt

doctor:
	python run.py doctor

run:
	python run.py run

digest:
	python run.py digest

week:
	python run.py digest --days 7 --out data/outbox/digest-$$(date +%Y-%m-%d).md

week-html:
	python run.py digest --days 7 --format html --out data/outbox/digest-$$(date +%Y-%m-%d).html

verify:
	python run.py verify

review:
	python run.py review

test:
	python -m pytest tests/ -q

smoke:
	python scripts/smoke.py

smoke-quick:
	python scripts/smoke.py --quick

clean:
	rm -rf data/work .pytest_cache
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
