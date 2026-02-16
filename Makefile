# --- Shell & Defaults ---
SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

# --- Vars ---
UV      ?= uv

# --- Phony ---
.PHONY: help bootstrap update env test fmt fmt-check lint typecheck qa \
	clean deep-clean

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

test: ## Pytest
	$(UV) run pytest

# ---------- Code quality (dev UX uses project env; CI uses pre-commit manual) ----------
fmt: ## Apply fixes now (ruff imports + format)
	$(UV) run ruff check --select I --fix src tests || true
	$(UV) run ruff format .

fmt-check: ## Non-mutating gate (CI/local)
	$(UV) run ruff check --select I src tests
	$(UV) run ruff format --check .

lint: ## Ruff lint
	$(UV) run ruff check .

typecheck: ## Mypy
	$(UV) run mypy

qa: fmt-check typecheck lint test ## Full quality gate

# ---------- Housekeeping ----------
clean: ## Remove caches
	-rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache .coverage
	-find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	-find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +
	-find . -type d -name ".mypy_cache" -prune -exec rm -rf {} +

deep-clean: clean ## Also remove build artifacts
	-rm -rf build dist *.egg-info
