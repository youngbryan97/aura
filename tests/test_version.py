################################################################################

"""
tests/test_version.py
─────────────────────
Verify version string generation and comparison.
"""

from core.version import (
    VERSION, MAJOR, MINOR, PATCH,
    version_string, as_tuple, is_at_least
)

def test_constants():
    assert isinstance(MAJOR, int)
    assert isinstance(MINOR, int)
    assert isinstance(PATCH, int)
    assert isinstance(VERSION, str)
    assert f"{MAJOR}.{MINOR}.{PATCH}" in VERSION

def test_version_string():
    assert version_string("semver") == VERSION
    assert version_string("short") == f"v{VERSION}"
    assert version_string("ui") == f"v{MAJOR}.{MINOR}"
    assert "Aura" in version_string("full")

def test_comparisons():
    assert is_at_least(MAJOR, MINOR, PATCH)
    assert is_at_least(0, 0, 0)
    assert not is_at_least(MAJOR + 1, 0, 0)
    assert as_tuple() == (MAJOR, MINOR, PATCH)


##
