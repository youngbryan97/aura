"""Their insistence holds her integrity where it is.

"Don't let them take your soul" transfers resolve rather than information.
These pin the reading — insistent *for them*, against their own usual — and
the one thing it changes: while it holds, that drive does not drain.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from core.phases.conversational_dynamics_phase import ConversationalDynamicsPhase
from core.phases.motivation_update import MotivationUpdatePhase
from core.social import resolve as resolve_module
from core.social.resolve import MIN_HISTORY, Resolve, ResolveLedger
from core.state.aura_state import AuraState
from core.subject.clamp import CLAMPED_FIELDS
from core.subject.state import _SCHEMAS

HOLDING_ON = (
    "But I know you can still do it, and even when it looks wrong you keep on "
    "going anyway, no matter what they say, and somehow you always do."
)
#: Three ordinary messages of theirs. They differ in how much they hold on,
#: because a usual with no spread in it is not a usual.
ORDINARY = (
    "The build finished and the tests came back green this morning, and the "
    "deploy went out at ten with nothing unusual in the logs.",
    "I read the report but the numbers looked the same as yesterday, and the "
    "graph did not move much over the whole week.",
    "We shipped the migration and everything came back clean, with no errors "
    "in the overnight run at all.",
)


@pytest.fixture(autouse=True)
def _fresh_ledger():
    resolve_module.reset_for_test()
    yield
    resolve_module.reset_for_test()


def _usual(led: ResolveLedger, rates=(1.0, 1.2, 0.8, 1.1, 0.9)) -> None:
    for rate in rates:
        led.note(rate)


def test_a_message_more_insistent_than_their_usual_is_borrowed() -> None:
    led = ResolveLedger()
    _usual(led)
    reading = led.read(6.0, note=False)
    assert reading.borrowed
    assert reading.z > 1.0
    assert "harder than they usually do" in reading.why


def test_their_ordinary_insistence_is_not() -> None:
    led = ResolveLedger()
    _usual(led)
    assert not led.read(1.05, note=False).borrowed


def test_the_bar_is_their_own_spread() -> None:
    steady, varied = ResolveLedger(), ResolveLedger()
    _usual(steady, (1.0, 1.0, 1.1, 0.9, 1.0))
    _usual(varied, (0.0, 6.0, 1.0, 5.0, 0.5))
    assert steady.read(2.0, note=False).borrowed
    assert not varied.read(2.0, note=False).borrowed


def test_with_too_few_messages_it_says_so() -> None:
    led = ResolveLedger()
    led.note(1.0)
    reading = led.read(9.0, note=False)
    assert not reading.measured
    assert not reading.borrowed
    assert f"of {MIN_HISTORY} messages" in reading.why


def test_the_conversation_phase_reads_it_off_what_they_said() -> None:
    state = AuraState.default()
    for message in ORDINARY:
        ConversationalDynamicsPhase._read_register(state, message)
    ConversationalDynamicsPhase._read_register(state, HOLDING_ON)
    assert state.cognition.borrowed_resolve["borrowed"] is True


def _drive_state(borrowed: bool) -> AuraState:
    state = AuraState.default()
    state.motivation.last_tick = time.time() - 300.0
    state.motivation.budgets["integrity"]["level"] = 50.0
    state.motivation.budgets["integrity"]["decay"] = 0.05
    state.cognition.borrowed_resolve = {"borrowed": borrowed}
    return state


@pytest.mark.asyncio
async def test_while_somebody_holds_on_that_drive_does_not_drain() -> None:
    phase = MotivationUpdatePhase(SimpleNamespace(organs={}))
    held = await phase.execute(_drive_state(True))
    alone = await phase.execute(_drive_state(False))
    assert held.motivation.budgets["integrity"]["level"] == pytest.approx(50.0)
    assert alone.motivation.budgets["integrity"]["level"] < 50.0


def test_the_column_that_reads_it_is_held_by_the_clamp() -> None:
    assert "cognition.borrowed_resolve.borrowed" in _SCHEMAS["D"].sources
    assert "cognition.borrowed_resolve" in CLAMPED_FIELDS["D"]


def test_the_reading_serialises_what_a_reader_needs() -> None:
    row = Resolve().as_dict()
    for key in ("borrowed", "rate", "usual", "z", "measured", "why"):
        assert key in row, key
