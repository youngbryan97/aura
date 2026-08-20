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


def test_every_site_that_can_select_the_deep_lane_asks_first() -> None:
    """Two decisions can route to the solver, and only one was gated.

    Stopping the registration at boot left the inference gate's own
    deep_handoff untouched, so a foreground chat turn still logged "Routing
    to Solver", spent a load admission that could not be granted, and came
    back empty — twice — ending in an apology.
    """
    from pathlib import Path

    gate = Path("core/brain/inference_gate.py").read_text(encoding="utf-8")
    assert "if deep_handoff and not local_deep_solver_enabled():" in gate

    decision = gate[gate.index('if requested_tier == "secondary":') :]
    decision = decision[: decision.index("strict_primary_proof_lane = False")]
    assert "local_deep_solver_enabled()" in decision
    assert decision.index("local_deep_solver_enabled()") < decision.index(
        "if deep_handoff and not explicit_background:"
    )


def test_a_refused_deep_handoff_falls_back_to_the_resident_lane() -> None:
    from pathlib import Path

    gate = Path("core/brain/inference_gate.py").read_text(encoding="utf-8")
    block = gate[gate.index("if deep_handoff and not local_deep_solver_enabled():") :]
    block = block[: block.index("if deep_handoff and not explicit_background:")]
    assert "deep_handoff = False" in block
    assert 'requested_tier = "primary"' in block
