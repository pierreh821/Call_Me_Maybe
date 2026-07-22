ENV := .venv/
PYTHON := $(ENV)/bin/python
export PATH := $(HOME)/.local/bin:$(PATH)

.PHONY: all install run debug clean lint lint-strict

all: install


install:
	uv sync

run:
	uv run python -m src

debug:
	uv run python -m pdb -m src

clean:
	rm -rf __pycache__ .mypy_cache .pytest_cache
	rm -rf $(ENV)

lint:
	flake8
	mypy --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs .

lint-strict:
	flake8
	mypy --strict .
