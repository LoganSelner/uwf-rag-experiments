"""Tests for src/core/filtering.py — shared metadata-filter matching."""

from __future__ import annotations

from ragbench.core.filtering import metadata_matches


class TestMetadataMatches:
    def test_empty_filters_match_anything(self) -> None:
        assert metadata_matches({"source_name": "kb"}, {}) is True

    def test_scalar_equality(self) -> None:
        md = {"source_name": "kb", "page": 3}
        assert metadata_matches(md, {"source_name": "kb"}) is True
        assert metadata_matches(md, {"source_name": "hb"}) is False
        assert metadata_matches(md, {"missing": 1}) is False

    def test_all_criteria_must_hold(self) -> None:
        md = {"source_name": "kb", "page": 3}
        assert metadata_matches(md, {"source_name": "kb", "page": 3}) is True
        assert metadata_matches(md, {"source_name": "kb", "page": 9}) is False

    def test_list_is_membership(self) -> None:
        # $in semantics: a list value matches if the metadata value is a member.
        assert metadata_matches({"s": "kb"}, {"s": ["kb", "hb"]}) is True
        assert metadata_matches({"s": "kb"}, {"s": ["hb", "other"]}) is False
        # Tuples and sets behave the same.
        assert metadata_matches({"s": "kb"}, {"s": ("kb", "hb")}) is True
        assert metadata_matches({"s": "kb"}, {"s": {"kb"}}) is True

    def test_missing_key_against_list_is_false(self) -> None:
        assert metadata_matches({}, {"s": ["kb", "hb"]}) is False
