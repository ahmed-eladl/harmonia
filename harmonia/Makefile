.PHONY: install test lint build clean check doctor

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

lint:
	ruff check harmonia/ tests/ || true

build:
	python -m build

clean:
	rm -rf dist/ build/ *.egg-info harmonia/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true

check:
	harmonia check

doctor:
	harmonia doctor
