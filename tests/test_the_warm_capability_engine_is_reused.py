"""One engine, one catalog load.

LIVE, 2026-08-22. A single chat turn logged "Refreshing skill registry" twenty
times in forty-five seconds, each rebuilding and probing all 79 skills.
`_ensure_catalog_loaded` guards against reloading, so twenty loads meant
twenty engines: several callers fall back to `CapabilityEngine()` when none is
passed to them, and a fresh engine has a cold catalog.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.capability_engine import live_capability_engine


def test_the_accessor_returns_nothing_outside_a_runtime():
    """It must not build one to answer the question."""
    assert live_capability_engine() is None


def test_the_accessor_returns_the_registered_engine(monkeypatch):
    sentinel = object()

    class Container:
        @staticmethod
        def get(name, default=None):
            return sentinel if name == "capability_engine" else default

    import core.container as container_module

    monkeypatch.setattr(container_module, "ServiceContainer", Container)
    assert live_capability_engine() is sentinel


@pytest.mark.parametrize(
    "path",
    [
        "core/self/capability_sources.py",
        "core/self/capability_lexicon.py",
        "core/conversation/capability_denial.py",
    ],
)
def test_the_fallback_sites_ask_for_the_warm_engine_first(path: str):
    source = Path(path).read_text(encoding="utf-8")
    assert "live_capability_engine()" in source, path
    # Never a bare construction where a warm one would do.
    assert "engine = CapabilityEngine()" not in source, path
