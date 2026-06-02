"""Git metadata helpers.

Captures repository state (SHA, dirty status) for experiment
reproducibility. These are intentionally best-effort — the
functions return safe defaults when git is unavailable.
"""

from __future__ import annotations

import subprocess


def get_git_sha() -> str:
    """Return the short current git SHA, or ``'unknown'`` when unavailable."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def get_git_dirty() -> bool:
    """Return True if the working tree has uncommitted changes."""
    try:
        output = (
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        return bool(output)
    except (subprocess.SubprocessError, FileNotFoundError):
        return False
