"""Tests for core.git — git metadata helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from ragbench.core.git import get_git_dirty, get_git_sha


class TestGetGitSha:
    @patch("ragbench.core.git.subprocess.check_output")
    def test_returns_trimmed_sha(self, mock_sub: MagicMock) -> None:
        mock_sub.return_value = "abc1234\n"
        assert get_git_sha() == "abc1234"

    @patch("ragbench.core.git.subprocess.check_output", side_effect=FileNotFoundError)
    def test_returns_unknown_on_error(self, mock_sub: MagicMock) -> None:
        assert get_git_sha() == "unknown"


class TestGetGitDirty:
    @patch("ragbench.core.git.subprocess.check_output")
    def test_clean_repo(self, mock_sub: MagicMock) -> None:
        mock_sub.return_value = b""
        assert get_git_dirty() is False

    @patch("ragbench.core.git.subprocess.check_output")
    def test_dirty_repo(self, mock_sub: MagicMock) -> None:
        mock_sub.return_value = b" M src/foo.py\n"
        assert get_git_dirty() is True

    @patch("ragbench.core.git.subprocess.check_output", side_effect=FileNotFoundError)
    def test_no_git_available(self, mock_sub: MagicMock) -> None:
        assert get_git_dirty() is False
