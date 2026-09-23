"""What she learned about a world is kept in a root that has never held it.

`remember` made the directory before it opened its governed scope and wrote
inside it. On the desktop the directory already existed, so the only sign was a
governance violation logged on every call. In a run with a fresh state root,
which is every campaign, the directory was refused under enforced governance,
nothing was written, and `recall` came back empty for the whole run.
"""

from __future__ import annotations

import pytest

from core.runtime import what_she_learned

pytestmark = pytest.mark.unit


def test_a_fresh_root_keeps_what_she_learned_under_enforced_governance(tmp_path, monkeypatch) -> None:
    kept_in = tmp_path / "state" / "worlds"
    monkeypatch.setattr(what_she_learned, "_KEPT_IN", kept_in)
    monkeypatch.setenv("AURA_GOVERNANCE_MODE", "production")
    assert not kept_in.exists()
    assert what_she_learned.remember("the tower", {"moves": ["left", "up"]}) is True
    assert what_she_learned.recall("the tower") == {"moves": ["left", "up"]}
