"""A lane whose model cannot be admitted is not registered.

LIVE, 2026-08-20. On a 64GB host the deep solver needs 48.4GB beside a
resident 25.3GB cortex against a 46.1GB lane budget, so admission refused
every load. The lane was registered anyway. A chat turn offered five tools
generated no text at all, twice, and ended in "I couldn't get to an answer
I'd stand behind" — the model had never been asked, because the endpoint that
was asked could not exist.
"""

from __future__ import annotations

import pytest

from core.brain.inference_gate import local_deep_solver_enabled


def test_a_64gb_host_cannot_host_the_deep_solver() -> None:
    assert local_deep_solver_enabled(64.0) is False


def test_a_large_host_can() -> None:
    assert local_deep_solver_enabled(256.0) is True


@pytest.mark.parametrize("setting", ["1", "true", "on", "yes"])
def test_an_explicit_yes_overrides_the_memory_class(monkeypatch, setting: str) -> None:
    monkeypatch.setenv("AURA_ENABLE_LOCAL_DEEP_SOLVER", setting)
    assert local_deep_solver_enabled(8.0) is True


@pytest.mark.parametrize("setting", ["0", "false", "off", "no"])
def test_an_explicit_no_overrides_a_large_host(monkeypatch, setting: str) -> None:
    monkeypatch.setenv("AURA_ENABLE_LOCAL_DEEP_SOLVER", setting)
    assert local_deep_solver_enabled(512.0) is False


def test_the_router_registration_asks_before_it_builds() -> None:
    """The predicate is consulted where the lane is created, not downstream."""
    from pathlib import Path

    source = Path("core/brain/llm_health_router.py").read_text(encoding="utf-8")
    registration = source[source.index("# Deep solver (72B)") :]
    registration = registration[: registration.index("# Brainstem")]
    assert "local_deep_solver_enabled" in registration
    assert registration.index("local_deep_solver_enabled") < registration.index(
        "router.register"
    )
