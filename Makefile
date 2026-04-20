PYTHON = python3
MYPY_FLAGS = --warn-return-any --warn-unused-ignores \
			 --ignore-missing-imports --disallow-untyped-defs \
			 --check-untyped-defs

.PHONY: install run debug clean lint lint-strict

install:
	uv sync

run:
	uv run $(PYTHON) -m src

debug:
	uv run $(PYTHON) -m pdb -m src

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +

lint:
	uv run $(PYTHON) -m flake8 .
	uv run $(PYTHON) -m mypy . $(MYPY_FLAGS)

lint-strict:
	uv run $(PYTHON) -m flake8 .
	uv run $(PYTHON) -m mypy . --strict
