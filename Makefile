.PHONY: setup run run-nollm test eval repro cost lint

setup:
	python3 -m pip install -e .

run:
	PYTHONPATH=src python3 -m sentinel run

run-nollm:
	PYTHONPATH=src python3 -m sentinel run --no-llm

test:
	pytest -q || test $$? -eq 5

eval:
	PYTHONPATH=src python3 -m sentinel eval

repro:
	bash scripts/repro.sh

cost:
	bash scripts/cost.sh

lint:
	ruff check src tests
	ruff format --check src tests
	mypy src
