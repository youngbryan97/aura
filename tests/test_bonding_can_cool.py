"""A quantity that cannot fall is a clock.

Bonding rose by a fixed amount on every user-facing turn and by nothing on any
other turn, so it could only ever increase. Across a 480-turn recording of the
subject core it correlated with the frame index to four decimal places — a
relationship that never cools, and in the coupling matrix the clock wearing the
self-state's name, correlating at one with every other counter in every other
domain.
"""

from __future__ import annotations

import asyncio

from core.phases.bonding_phase import BondingPhase
from core.state.aura_state import AuraState


def _turn(phase: BondingPhase, state: AuraState, objective: str) -> None:
    state.cognition.current_origin = "user"
    asyncio.run(phase.execute(state, objective=objective))


def test_a_long_silence_lets_bonding_settle(monkeypatch) -> None:
    """The same exchange, after a gap ten times their usual one."""
    import core.phases.bonding_phase as module

    clock = {"now": 1_000.0}
    monkeypatch.setattr(module.time, "time", lambda: clock["now"])

    phase = BondingPhase()
    state = AuraState()
    for _ in range(8):
        clock["now"] += 60.0
        _turn(phase, state, "a message of ordinary length")
    built = state.identity.bonding_level
    assert built > 0.05, "a run of exchanges did not build bonding at all"

    clock["now"] += 60.0 * 200.0
    _turn(phase, state, "a message of ordinary length")
    assert state.identity.bonding_level < built, (
        "bonding did not settle across a silence two hundred times their usual gap"
    )


def test_a_steady_rhythm_of_ordinary_exchanges_does_not_run_away(monkeypatch) -> None:
    """What the defect looked like: the same turn, forever, always up."""
    import core.phases.bonding_phase as module

    clock = {"now": 1_000.0}
    monkeypatch.setattr(module.time, "time", lambda: clock["now"])

    phase = BondingPhase()
    state = AuraState()
    for _ in range(4):
        clock["now"] += 60.0
        _turn(phase, state, "a message of ordinary length")
    early = state.identity.bonding_level
    for _ in range(60):
        clock["now"] += 60.0
        _turn(phase, state, "a message of ordinary length")
    late = state.identity.bonding_level
    # It may still build — a relationship of many exchanges is deeper than one
    # of few. What it must not do is build at the undamped rate for ever.
    drift_per_turn = (late - early) / 60.0
    assert 0.0 <= drift_per_turn <= 0.0001, (
        f"an unchanging rhythm climbs {drift_per_turn:.6f} a turn"
    )


def test_a_richer_exchange_still_builds(monkeypatch) -> None:
    """Settling must not cost the phase the thing it is for."""
    import core.phases.bonding_phase as module

    clock = {"now": 1_000.0}
    monkeypatch.setattr(module.time, "time", lambda: clock["now"])

    phase = BondingPhase()
    state = AuraState()
    for _ in range(8):
        clock["now"] += 60.0
        _turn(phase, state, "short")
    settled = state.identity.bonding_level
    clock["now"] += 60.0
    # Over fifty words, which is the length the phase counts as a longer turn.
    _turn(phase, state, " ".join(f"word{index}" for index in range(60)))
    assert state.identity.bonding_level > settled
