# --- Shell & Defaults ---
SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

# --- Vars ---
UV      ?= uv
HOST    ?= 0.0.0.0
PORT    ?= 8000
APP     ?= app.main:app
IMAGE   ?= phase0-app
TAG     ?= local
FULL_IMAGE := $(IMAGE):$(TAG)
CONTAINER_NAME ?= phase0
DOCKER  ?= docker

# --- Phony ---
.PHONY: help bootstrap update env dev serve test fmt fmt-check lint typecheck qa \
	clean deep-clean \
	docker-build docker-rebuild docker-run docker-run-d docker-stop docker-logs docker-shell

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

# ---------- App ----------
dev: ## FastAPI with auto-reload
	$(UV) run uvicorn $(APP) --reload --host $(HOST) --port $(PORT)

serve: ## FastAPI like prod (no reload)
	$(UV) run uvicorn $(APP) --host $(HOST) --port $(PORT)

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

# ---------- Docker (uv multi-stage Dockerfile) ----------
docker-build: ## Build image (cached)
	DOCKER_BUILDKIT=1 $(DOCKER) build -t $(FULL_IMAGE) .

docker-rebuild: ## Build image (no cache)
	DOCKER_BUILDKIT=1 $(DOCKER) build --no-cache -t $(FULL_IMAGE) .

docker-run: ## Run foreground
	$(DOCKER) run --rm --init -p $(PORT):$(PORT) $(FULL_IMAGE)

docker-run-d: ## Run detached
	$(DOCKER) run -d --rm --init -p $(PORT):$(PORT) --name $(CONTAINER_NAME) $(FULL_IMAGE)

docker-stop: ## Stop detached
	-$(DOCKER) stop $(CONTAINER_NAME)

docker-logs: ## Follow logs
	$(DOCKER) logs -f $(CONTAINER_NAME)

docker-shell: ## Shell in image
	$(DOCKER) run --rm -it --entrypoint /bin/bash $(FULL_IMAGE)

# ---------- Housekeeping ----------
clean: ## Remove caches
	-rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache .coverage
	-find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	-find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +
	-find . -type d -name ".mypy_cache" -prune -exec rm -rf {} +

deep-clean: clean ## Also remove build artifacts
	-rm -rf build dist *.egg-info
