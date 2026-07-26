ENV := .venv
PYTHON := $(ENV)/bin/python
export PATH := $(HOME)/.local/bin:$(PATH)

.PHONY: all install run debug clean lint lint-strict

all: install


install:
	uv sync

run:
	uv run python -m main

debug:
	uv run python -m pdb -m main

clean:
	rm -rf __pycache__ .mypy_cache .pytest_cache
	rm -rf $(ENV)

lint:
	$(PYTHON) -m flake8 --exclude=".venv .src"
	$(PYTHON) -m mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs --exclude=".venv .src"

lint-strict:
	$(PYTHON) -m flake8 --exclude=".venv .src"
	$(PYTHON) -m mypy . --strict --exclude=".venv .src"
