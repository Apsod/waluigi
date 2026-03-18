.PHONY: all lint typecheck test check

all: check

lint:
	uv run --with ruff ruff check src test examples

typecheck:
	uv run --with mypy mypy src test

test:
	uv run python -m unittest discover -s test -v

example:
	uv run --with polars examples/polars_pipeline.py

check: lint typecheck test
