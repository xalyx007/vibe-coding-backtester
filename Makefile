.PHONY: setup test lint clean docs build install dev

# Default Python interpreter
PYTHON := python3

# Installation
setup:
	$(PYTHON) -m pip install -e ".[dev]"

# Testing
test:
	$(PYTHON) -m pytest tests/

test-unit:
	$(PYTHON) -m pytest tests/unit/

test-integration:
	$(PYTHON) -m pytest tests/integration/

test-coverage:
	$(PYTHON) -m pytest --cov=backtester tests/

# Linting and formatting
lint:
	$(PYTHON) -m flake8 backtester tests
	$(PYTHON) -m black --check backtester tests

format:
	$(PYTHON) -m black backtester tests

# Documentation
docs:
	$(PYTHON) -m mkdocs build

docs-serve:
	$(PYTHON) -m mkdocs serve

# Cleaning
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf site/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Building and distribution
build:
	$(PYTHON) -m pip install --upgrade build
	$(PYTHON) -m build

# Installation
install:
	$(PYTHON) -m pip install .

dev:
	$(PYTHON) -m pip install -e ".[dev]"

# Help
help:
	@echo "Available commands:"
	@echo "  setup           - Install the package in development mode"
	@echo "  test            - Run all tests"
	@echo "  test-unit       - Run unit tests"
	@echo "  test-integration - Run integration tests"
	@echo "  test-coverage   - Run tests with coverage report"
	@echo "  lint            - Check code style with flake8 and black"
	@echo "  format          - Format code with black"
	@echo "  docs            - Build documentation"
	@echo "  docs-serve      - Serve documentation locally"
	@echo "  clean           - Remove build artifacts"
	@echo "  build           - Build distribution packages"
	@echo "  install         - Install the package"
	@echo "  dev             - Install the package in development mode with dev dependencies" 