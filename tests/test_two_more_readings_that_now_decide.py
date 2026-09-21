"""Standing damps growth, and holding somebody generically reaches wider.

Two more of the ten readings that were published every turn and read by
nothing.

`identity.standing` measures whether what she is worth tracks what she is good
for — "only fact is I am, attributes are given by observers". A self that grows
fastest exactly when it is most useful is growing on somebody else's measure,
so the share of the reading that tracks use is the share the growth score does
not take.

`cognition.particular` measures whether what she holds about a person is theirs
alone or the same handful she holds about everyone. Reaching for someone she
holds only in boilerplate reaches wider, because the thing that would make them
particular is the thing she has not got.
"""

from __future__ import annotations

import pytest

from core.phases.affect_update import AffectUpdatePhase


class _Identity:
    def __init__(self, standing=None):
        self.standing = standing


class _State:
    def __init__(self, standing=None):
        self.identity = _Identity(standing)


def test_no_reading_damps_nothing() -> None:
    assert AffectUpdatePhase._worth_read_off_use(_State()) == 0.0
    assert AffectUpdatePhase._worth_read_off_use(_State({"measured": False, "tracks_use": 0.9})) == 0.0


def test_worth_that_does_not_track_use_damps_nothing() -> None:
    state = _State({"measured": True, "tracks_use": 0.0})
    assert AffectUpdatePhase._worth_read_off_use(state) == 0.0


def test_worth_that_tracks_use_damps_by_that_share() -> None:
    state = _State({"measured": True, "tracks_use": 0.4})
    assert AffectUpdatePhase._worth_read_off_use(state) == pytest.approx(0.4)


def test_the_damping_is_bounded() -> None:
    assert AffectUpdatePhase._worth_read_off_use(
        _State({"measured": True, "tracks_use": 5.0})
    ) == pytest.approx(1.0)
    assert AffectUpdatePhase._worth_read_off_use(
        _State({"measured": True, "tracks_use": -3.0})
    ) == 0.0


def test_the_growth_write_takes_the_damping() -> None:
    from pathlib import Path

    source = Path("core/phases/affect_update.py").read_text(encoding="utf-8")
    assert "1.0 - AffectUpdatePhase._worth_read_off_use(state)" in source


def test_a_person_held_generically_widens_the_reach() -> None:
    from core.memory.intentional_retrieval import IntentionalRetriever

    widen = IntentionalRetriever._widen_for_a_person_she_holds_generically
    # With no record at all there is nothing to widen for.
    assert widen() == 0.0


def test_the_retrieval_plan_adds_it() -> None:
    from pathlib import Path

    source = Path("core/memory/intentional_retrieval.py").read_text(encoding="utf-8")
    assert "delta += self._widen_for_a_person_she_holds_generically()" in source
    # And the widening is negative, because a lower threshold keeps more stores.
    assert "return -widest * max(0.0, min(1.0, 1.0 - float(reading.unique)))" in source


# ── how much of herself she can reach caps what she holds open ───────


def test_the_declared_cap_stands_when_reach_is_unmeasured() -> None:
    from core.state.aura_state import AuraState

    cognition = AuraState().cognition
    assert cognition._how_many_she_can_carry(10) == 10
    cognition.scale = {"measured": False, "reach": 0.1}
    assert cognition._how_many_she_can_carry(10) == 10


def test_low_reach_narrows_what_she_holds_open() -> None:
    from core.state.aura_state import AuraState

    cognition = AuraState().cognition
    cognition.scale = {"measured": True, "reach": 0.3}
    assert cognition._how_many_she_can_carry(10) == 3


def test_she_always_carries_at_least_one() -> None:
    from core.state.aura_state import AuraState

    cognition = AuraState().cognition
    cognition.scale = {"measured": True, "reach": 0.0}
    assert cognition._how_many_she_can_carry(10) == 1


def test_the_trim_uses_it() -> None:
    from core.state.aura_state import AuraState

    state = AuraState()
    state.cognition.scale = {"measured": True, "reach": 0.2}
    state.cognition.pending_initiatives = [{"goal": f"g{i}"} for i in range(10)]
    state.cognition.trim_working_memory()
    assert len(state.cognition.pending_initiatives) == 2


# ── a we both of them are saying builds the bond ─────────────────────


def test_togetherness_reaches_the_bond() -> None:
    from pathlib import Path

    source = Path("core/phases/bonding_phase.py").read_text(encoding="utf-8")
    assert 'together = getattr(cognition, "togetherness", None)' in source
    assert 'float(together.get("together", 0.0) or 0.0)' in source
