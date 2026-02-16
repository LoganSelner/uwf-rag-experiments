![CI](https://github.com/LoganSelner/phase0/actions/workflows/ci.yml/badge.svg)

# FastAPI + uv template

A minimal, modern FastAPI starter that uses **[uv](https://github.com/astral-sh/uv)** for dependency and virtualenv management, **pre-commit** for quality gates, and a clean `src/` layout. Ships with a Dockerfile and a small health-check route.

---

## Features

* ⚡ **FastAPI** app at `src/app/main.py` with a `/health` endpoint
* 🧪 **Tests** via `pytest` (see `tests/test_app.py`)
* 🧰 **uv** for fast, reproducible installs (`pyproject.toml` + `uv.lock`)
* 🧹 **pre-commit** hooks: ruff (lint + format), and more
* 🐳 **Dockerfile** for production-like images (multi-stage, small runtime)
* 🗂️ `src/` layout with `app` package (`app.main:app`)

---

## Prerequisites

* **Python 3.13** (a `.python-version` file is included if you use pyenv)
* **uv** (once per machine). Install options:

  * macOS/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
  * Windows (PowerShell): `irm https://astral.sh/uv/install.ps1 | iex`

---

## Quickstart (local dev)

```bash
make bootstrap
# → installs a compatible Python, syncs deps (incl. dev), installs git hooks

make dev
# → http://localhost:8000
# Visit: http://localhost:8000/health
# Docs:  http://localhost:8000/docs  (Swagger UI)
#        http://localhost:8000/redoc

make qa
# (non-mutating) fmt-check + typecheck + pytest
```

### One-liners you’ll use a lot

```bash
# Run tests
uv run pytest

# Type-check
uv run mypy

# Lint + format (non-destructive checks)
uv run ruff check . && uv run ruff format --check .

# Apply fixes (lint + format)
uv run ruff check --fix . && uv run ruff format .

```

---

## Makefile shortcuts

Run `make help` to see everything. Common targets:

| Target                | What it does                                             |
| --------------------- | -------------------------------------------------------- |
| `make bootstrap`      | Install Python (if needed), sync deps, install git hooks |
| `make update`         | Upgrade locked package versions (respecting constraints) |
| `make env`            | Print tool versions                                      |
| `make dev`            | Start FastAPI with auto-reload                           |
| `make serve`          | Start FastAPI like prod (no reload)                      |
| `make test`           | Run pytest                                               |
| `make fmt`            | Apply formatting fixes (ruff imports + format)           |
| `make fmt-check`      | Non-mutating QA check for CI/local                       |
| `make lint`           | Ruff lint                                                |
| `make typecheck`      | Mypy                                                     |
| `make qa`             | Full quality gate (lint+format+types+tests)              |
| `make docker-build`   | Build Docker image (cached)                              |
| `make docker-rebuild` | Build without cache                                      |
| `make docker-run`     | Run in foreground (maps port 8000)                       |
| `make docker-run-d`   | Run detached                                             |
| `make docker-stop`    | Stop detached container                                  |
| `make docker-logs`    | Follow logs                                              |
| `make docker-shell`   | Shell into the image                                     |
| `make clean`          | Remove caches                                            |
| `make deep-clean`     | Also remove build artifacts                              |

### Examples:
```bash
make dev PORT=9001
make docker-build TAG=pr-42
make docker-run-d CONTAINER_NAME=phase0
```

> If a target isn’t available on your machine, run `make help` to confirm the list and descriptions.

---

## Rename this template

Prefer simple search/replace? Do this once and you’re done:

1. pyproject.toml → in [project], change name = "phase0" to your project name.

2. README.md → replace the few occurrences of phase0 in examples.

3. Makefile → if it mentions phase0, replace those occurrences too.

---

## Docker

Build a compact image with a prebuilt virtualenv and run it:

```bash
# Build
make docker-build

# Run
make docker-run
# App will listen on 0.0.0.0:8000 → http://localhost:8000
```

The image’s default command is:

```
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## Project layout

```
.
├─ src/app/
│  ├─ __init__.py
│  └─ main.py          # FastAPI app with /health
├─ tests/
│  └─ test_app.py      # smoke test
├─ pyproject.toml      # deps, tool config, build backend
├─ uv.lock             # lockfile (commit this)
├─ Makefile            # dev workflow
├─ Dockerfile          # multi-stage build (non-root runtime)
└─ README.md

```

* **App entrypoint**: `app.main:app`
* **Health check**: `GET /health` → `{ "ok": true }`

---

## pre-commit (quality gates)

This repo is configured to run a suite of checks (ruff lint, ruff format, etc.).

* Run once on all files:

  ```bash
  uv run pre-commit run --all-files
  ```
* Install into your git hooks (so it runs automatically on commit):

  ```bash
  uv run pre-commit install
  ```


---

## Troubleshooting

* **No files to check**

  * Only tracked files are checked. git add -A, or run make fmt.
* **Port already in use**

  * Change `--port` or stop the other process.
* **Python version mismatch**

  * This project targets **Python 3.13**. If you’re on another version, use pyenv or update your interpreter.
* **Fresh env**

  ```bash
  rm -rf .venv
  make bootstrap
  ```

## License
[MIT](./LICENSE)
