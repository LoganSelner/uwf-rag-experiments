![CI](https://github.com/LoganSelner/phase0/actions/workflows/ci.yml/badge.svg)

# UWF RAG Experiments

A Python project using **[uv](https://github.com/astral-sh/uv)** for dependency and virtualenv management, **pre-commit** for quality gates, and a clean `src/` layout.

---

## Features

* 🧪 **Tests** via `pytest`
* 🧰 **uv** for fast, reproducible installs (`pyproject.toml` + `uv.lock`)
* 🧹 **pre-commit** hooks: ruff (lint + format), and more
* 🗂️ `src/` layout

---

## Prerequisites

* **Python 3.11** (a `.python-version` file is included if you use pyenv)
* **uv** (once per machine). Install options:

  * macOS/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
  * Windows (PowerShell): `irm https://astral.sh/uv/install.ps1 | iex`

---

## Quickstart (local dev)

```bash
make bootstrap
# → installs a compatible Python, syncs deps (incl. dev), installs git hooks

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
| `make test`           | Run pytest                                               |
| `make fmt`            | Apply formatting fixes (ruff imports + format)           |
| `make fmt-check`      | Non-mutating QA check for CI/local                       |
| `make lint`           | Ruff lint                                                |
| `make typecheck`      | Mypy                                                     |
| `make qa`             | Full quality gate (lint+format+types+tests)              |
| `make clean`          | Remove caches                                            |
| `make deep-clean`     | Also remove build artifacts                              |

> If a target isn't available on your machine, run `make help` to confirm the list and descriptions.

---

## Rename this template

Prefer simple search/replace? Do this once and you’re done:

1. pyproject.toml → in [project], change name = "phase0" to your project name.

2. README.md → replace the few occurrences of phase0 in examples.

3. Makefile → if it mentions phase0, replace those occurrences too.

---

## Project layout

```
.
├─ src/                 # source code
├─ tests/               # tests
├─ pyproject.toml       # deps, tool config, build backend
├─ uv.lock              # lockfile (commit this)
├─ Makefile             # dev workflow
└─ README.md
```

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
* **Python version mismatch**

  * This project targets **Python 3.11**. If you’re on another version, use pyenv or update your interpreter.
* **Fresh env**

  ```bash
  rm -rf .venv
  make bootstrap
  ```

## License
[MIT](./LICENSE)
