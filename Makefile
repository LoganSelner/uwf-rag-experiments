# --- Shell & Defaults ---
SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

# --- Vars ---
UV      ?= uv
CONFIG  ?= configs/base.yaml
DIRS    ?= results/*

# --- Phony ---
.PHONY: help bootstrap update env test test-all experiment compare \
	fmt fmt-check lint typecheck qa precommit clean deep-clean

help: ## Show available targets
	@awk '\
	  BEGIN { FS=":.*##"; printf "\nTargets:\n" } \
	  /^[a-zA-Z0-9_.-]+:.*##/ { \
	    name = $$1; desc = $$2; \
	    gsub(/^[ \t]+|[ \t]+$$/, "", desc); \
	    if (!seen[name]++) printf "  \033[36m%-16s\033[0m %s\n", name, desc; \
	  }' $(MAKEFILE_LIST)

# ---------- Setup ----------
bootstrap: ## Install Python (if needed), sync deps, install git hooks
	$(UV) python install
	$(UV) sync
	$(UV) run pre-commit install

update: ## Upgrade locked package versions (respecting constraints)
	$(UV) lock --upgrade
	$(UV) sync
	$(UV) run pre-commit autoupdate
	$(UV) run pre-commit install

env: ## Print tool versions
	@echo "Python:  $$($(UV) run python -V)"
	@echo "uv:      $$($(UV) --version)"
	@echo "Ruff:    $$($(UV) run ruff --version || true)"
	@echo "Mypy:    $$($(UV) run mypy --version || true)"
	@echo "pytest:  $$($(UV) run pytest --version | head -n1 || true)"

# ---------- RAG workflow ----------
experiment: ## Run an experiment (CONFIG=configs/base.yaml)
	$(UV) run python scripts/run_experiment.py $(CONFIG)

compare: ## Compare experiment results (DIRS="results/a results/b")
	$(UV) run python scripts/compare.py $(DIRS)

# ---------- Code quality ----------
test: ## Run fast tests (skip slow)
	$(UV) run pytest -m "not slow"

test-all: ## Run all tests including slow
	$(UV) run pytest

fmt: ## Auto-fix lint + format
	$(UV) run ruff check --select I --fix src scripts tests || true
	$(UV) run ruff check --fix src scripts tests || true
	$(UV) run ruff format .

fmt-check: ## Check formatting (CI gate)
	$(UV) run ruff check --select I src scripts tests
	$(UV) run ruff format --check .

lint: ## Ruff lint
	$(UV) run ruff check .

typecheck: ## Mypy type check
	$(UV) run mypy

qa: fmt-check typecheck lint test ## Full quality gate

precommit: ## Run all pre-commit hooks on tracked files
	$(UV) run pre-commit run --all-files

# ---------- Housekeeping ----------
clean: ## Remove caches
	-rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache .coverage
	-find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	-find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +
	-find . -type d -name ".mypy_cache" -prune -exec rm -rf {} +

deep-clean: clean ## Also remove env and build artifacts
	-rm -rf .venv htmlcov coverage.xml .dist build dist *.egg-info
