ENV := .venv
PYTHON := $(ENV)/bin/python

.PHONY: all install run debug clean lint lint-strict

all: run

install:
	uv sync

run:
	uv run python -m src

debug:
	uv run python -m pdb -m src

clean:
	rm -rf __pycache__ .mypy_cache .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "Call_me_maybe.egg-info" -exec rm -rf {} +

fclean: clean
	rm -rf $(ENV)

lint:
	uv run flake8 src
	uv run mypy --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs src

lint-strict:
	uv run flake8 src
	uv run mypy --strict src
