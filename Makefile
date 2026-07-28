ENV := .venv
PYTHON := $(ENV)/bin/python

SRC := src
MODELS := models

.PHONY: all install run debug clean lint lint-strict

all: install


install:
	uv sync

run:
	uv run python -m $(SRC)/main

debug:
	uv run python -m pdb -m $(SRC)/main

clean:
	rm -rf __pycache__ .mypy_cache .pytest_cache
	rm -rf $(ENV)

lint:
	$(PYTHON) -m flake8 $(SRC) $(MODELS)
	$(PYTHON) -m mypy $(SRC) $(MODELS) --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	$(PYTHON) -m flake8 $(SRC) $(MODELS)
	$(PYTHON) -m mypy $(SRC) $(MODELS) --strict
