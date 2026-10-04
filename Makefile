ENV := .venv
PYTHON := $(ENV)/bin/python

DEFINITION := data/input/functions_definition.json
INPUT := data/input/function_calling_tests.json
OUTPUT := data/output/function_calls.json

.PHONY: all install run debug clean fclean lint lint-strict

all: run

install:
	uv sync

run:
	uv run python -m src --functions_definition $(DEFINITION) --input $(INPUT) --output $(OUTPUT)

debug:
	uv run python -m pdb -m src

clean:
	rm -rf data/output
	rm -rf __pycache__ .mypy_cache .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "Call_me_maybe.egg-info" -exec rm -rf {} +

fclean: clean
	rm -rf $(ENV)

lint: install
	uv run flake8 src
	uv run mypy --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs src

lint-strict: install
	uv run flake8 src
	uv run mypy --strict src
