"""Whether anything but the body can move what she intends.

run_031's interventional graph had one strongly connected component and a
vertex connectivity of 1 against a bar of 2, so `robust_recurrence_kappa`
failed. The cut vertex was I: remove interoception and nothing in the system
could reach D at all, because I held D's only replicated inbound edge.

The reason was in the proposers. Every urgency a branch declares is a constant
chosen at that branch, and the only thing that moved without one was a step
function of wall-clock age, so nothing a proposer could not see reached
deliberation's own state. `free_energy.get_action_urgency` is a measured
reading of prediction error and arousal, which is recurrent cognition and
affect, and it was reaching the workspace and not this.
"""

from __future__ import annotations

import pytest

from core.agency.initiative_arbiter import InitiativeArbiter
from core.container import ServiceContainer


@pytest.fixture(autouse=True)
def _restore_the_container():
    """Put back whatever was registered. A test that leaves a stand-in in the
    container is the order-dependence these tests are about."""
    had = ServiceContainer.get("free_energy", default=None)
    yield
    ServiceContainer.register("free_energy", had)


class _Cognition:
    working_memory: list = []
    active_goals: list = []
    pending_initiatives: list = []
    current_origin = ""


class _MinimalState:
    """Enough of a state for the arbiter to score one initiative."""

    def __init__(self) -> None:
        self.cognition = _Cognition()
        self.identity = type("I", (), {"values": [], "traits": {}})()
        self.affect = type("A", (), {"valence": 0.0, "arousal": 0.5, "emotions": {}})()
        self.motivation = type("M", (), {"budgets": {}})()
        self.response_modifiers: dict = {}


class _FreeEnergy:
    """Stands in for the engine, returning whatever it is set to."""

    def __init__(self, value: float = 0.5) -> None:
        self.value = value

    def get_action_urgency(self) -> float:
        return self.value


def _settled(arbiter: InitiativeArbiter, engine: _FreeEnergy) -> None:
    """Enough readings that a middle of them exists."""
    for value in (0.2, 0.5, 0.8):
        engine.value = value
        arbiter._read_pressure_shift()


def test_a_declared_urgency_stands_alone_when_nothing_measures_the_pressure():
    ServiceContainer.register("free_energy", None)
    arbiter = InitiativeArbiter()
    arbiter._shift = arbiter._read_pressure_shift()
    assert arbiter._score_urgency({"urgency": 0.6, "timestamp": 0}, None) == 0.6


def test_a_state_that_is_pressing_raises_what_she_intends():
    engine = _FreeEnergy()
    ServiceContainer.register("free_energy", engine)
    arbiter = InitiativeArbiter()
    _settled(arbiter, engine)
    declared = {"urgency": 0.6, "timestamp": 0}

    engine.value = 0.9
    arbiter._shift = arbiter._read_pressure_shift()
    pressing = arbiter._score_urgency(declared, None)

    engine.value = 0.1
    arbiter._shift = arbiter._read_pressure_shift()
    at_rest = arbiter._score_urgency(declared, None)

    assert pressing > 0.6 > at_rest, (pressing, at_rest)


def test_every_initiative_in_one_moment_moves_by_the_same_amount():
    """The proposer's ranking is what the explicit-urgency branch protects.

    A shift read per initiative rather than per pass would reorder them
    whenever the engine's reading moved between two calls, which it does.
    """
    engine = _FreeEnergy()
    ServiceContainer.register("free_energy", engine)
    arbiter = InitiativeArbiter()
    _settled(arbiter, engine)
    low = {"urgency": 0.3, "timestamp": 0}
    high = {"urgency": 0.7, "timestamp": 0}

    engine.value = 0.9
    arbiter._shift = arbiter._read_pressure_shift()
    moved_low = arbiter._score_urgency(low, None)
    moved_high = arbiter._score_urgency(high, None)

    assert moved_high > moved_low
    assert (moved_high - 0.7) == pytest.approx(moved_low - 0.3)


def test_the_shift_is_centred_on_her_own_middle_and_not_on_a_number():
    """It averages to nothing over a life, so it cannot inflate urgency."""
    engine = _FreeEnergy()
    ServiceContainer.register("free_energy", engine)
    arbiter = InitiativeArbiter()
    for value in (0.2, 0.4, 0.6):
        engine.value = value
        arbiter._read_pressure_shift()
    engine.value = 0.4
    assert arbiter._read_pressure_shift() == 0.0


def test_what_decided_is_what_the_recorder_reads():
    """The arbiter's score lived in its own receipt and reached nothing.

    `core/subject/state.py` reads `cognition.pending_initiatives[*].urgency`,
    which is the proposer's constant, so a shift applied in the arbiter moved
    the ranking and left the recorded column exactly where it was.
    """
    from core.subject.state import _urgency_of

    engine = _FreeEnergy()
    ServiceContainer.register("free_energy", engine)
    arbiter = InitiativeArbiter()
    _settled(arbiter, engine)
    engine.value = 0.9
    arbiter._shift = arbiter._read_pressure_shift()

    initiative = {"urgency": 0.6, "timestamp": 0}
    arbiter._score_urgency(initiative, None)
    assert _urgency_of([initiative]) == 0.6

    import asyncio

    asyncio.run(arbiter.score_initiative(initiative, _MinimalState()))
    assert initiative["urgency"] == 0.6, "the proposal must stay the proposal"
    assert initiative["decided_urgency"] > 0.6
    assert _urgency_of([initiative]) == initiative["decided_urgency"]


def test_the_proposal_is_never_overwritten_so_the_shift_cannot_compound():
    """`_explicit_urgency` reads `urgency`. Writing the decided value back
    would make the next pass treat this pass's shift as a declaration."""
    import asyncio

    engine = _FreeEnergy()
    ServiceContainer.register("free_energy", engine)
    arbiter = InitiativeArbiter()
    _settled(arbiter, engine)
    initiative = {"urgency": 0.5, "timestamp": 0}
    decided = []
    for _ in range(4):
        engine.value = 0.9
        arbiter._shift = arbiter._read_pressure_shift()
        asyncio.run(arbiter.score_initiative(initiative, _MinimalState()))
        decided.append(initiative["decided_urgency"])
    assert initiative["urgency"] == 0.5
    assert max(decided) - min(decided) < 0.5, decided


def test_two_readings_are_not_a_middle():
    engine = _FreeEnergy()
    ServiceContainer.register("free_energy", engine)
    arbiter = InitiativeArbiter()
    engine.value = 0.2
    assert arbiter._read_pressure_shift() == 0.0
    engine.value = 0.9
    assert arbiter._read_pressure_shift() == 0.0
