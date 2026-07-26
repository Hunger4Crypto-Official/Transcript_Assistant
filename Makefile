.PHONY: help setup doctor run digest week test clean

help:
	@echo "make setup    install dependencies"
	@echo "make doctor   preflight everything"
	@echo "make run      process the inbox"
	@echo "make digest   combined digest to stdout"
	@echo "make week     write this week's digest to data/outbox"
	@echo "make test     run the test suite"
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

test:
	python -m pytest tests/ -q

clean:
	rm -rf data/work .pytest_cache
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
