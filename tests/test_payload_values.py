"""A model-authored payload's absent values, and where its paths may land.

Guards the 2026-08-19 live defect: a scaffold written into a directory named
``None`` at the root of the source tree.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.runtime.payload_values import (
    MISSING_SENTINELS,
    is_missing,
    payload_path,
    payload_text,
    payload_value,
)


@pytest.mark.parametrize("spelling", sorted(MISSING_SENTINELS))
def test_every_sentinel_reads_as_absent(spelling: str) -> None:
    assert is_missing(spelling)
    assert is_missing(spelling.upper())
    assert is_missing(f"  {spelling}  ")


def test_content_survives() -> None:
    """Zero, False and an empty container are values somebody chose."""
    for value in (0, 0.0, False, [], {}, "0", "false", "nothing much"):
        assert not is_missing(value), value


def test_the_defect_itself() -> None:
    """The string "None" no longer resolves to a directory called None."""
    assert payload_path({"output_dir": "None"}, "output_dir", root="/tmp") is None
    assert payload_text({"output_dir": "None"}, "output_dir") == ""


def test_relative_paths_resolve_under_the_root_not_the_cwd(tmp_path: Path) -> None:
    resolved = payload_path({"out": "build/x"}, "out", root=tmp_path)
    assert resolved == (tmp_path / "build" / "x").resolve()


def test_a_path_that_climbs_out_is_refused(tmp_path: Path) -> None:
    assert payload_path({"out": "../elsewhere"}, "out", root=tmp_path) is None
    assert payload_path({"out": "/etc"}, "out", root=tmp_path) is None


def test_outside_root_is_available_when_the_caller_asks(tmp_path: Path) -> None:
    resolved = payload_path({"out": "/etc"}, "out", root=tmp_path, allow_outside_root=True)
    assert resolved == Path("/etc").resolve()


def test_keys_are_tried_in_order() -> None:
    payload = {"new": "null", "old": "kept"}
    assert payload_value(payload, "new", "old") == "kept"
    assert payload_value(payload, "absent", default="fallback") == "fallback"


def test_a_non_mapping_payload_is_not_an_error() -> None:
    assert payload_value(None, "any") is None
    assert payload_text("a string", "any", default="d") == "d"
