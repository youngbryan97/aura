"""The agency ledger hears about what somebody else did.

`core/agency/intention_loop.py` records her own actions and says outcomes she
only watched arrive through perception. Nothing on the perception path recorded
them, so in the runtime and in every battery recording the ledger held her own
actions alone: the share of what happened that she did read 1.0 for a whole
2,400-turn life and the last actor never changed. The affect phase, which feels
each percept once, now records a percept that names the user.
"""

from __future__ import annotations

import pytest

from core.agency.authorship import (
    SELF,
    Event,
    actor_of_percept,
    get_agency_ledger,
    reset_agency_ledger_for_test,
)
from core.phases.affect_update import AffectUpdatePhase

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _fresh_ledger():
    reset_agency_ledger_for_test()
    yield
    reset_agency_ledger_for_test()


@pytest.mark.parametrize("source", ["user", "chat", "conversation", "voice:desk", "api:client", "USER"])
def test_a_percept_from_the_user_names_the_user(source: str) -> None:
    assert actor_of_percept({"source": source}) == "user"


@pytest.mark.parametrize("source", ["memory_retrieval", "action_grounding", "", "subject_core_turn"])
def test_a_percept_from_her_own_processing_names_nobody(source: str) -> None:
    assert actor_of_percept({"source": source}) is None


def test_what_the_user_did_is_watched_and_not_authored() -> None:
    ledger = get_agency_ledger()
    ledger.observe(Event(what="intention", actor=SELF, verified=True))
    assert ledger.authored_share == 1.0

    percepts = [
        {"type": "interaction", "source": "chat", "content": "Bryan sent a message"},
        {"type": "memory", "source": "memory_retrieval", "content": "something recalled"},
    ]
    assert AffectUpdatePhase._record_what_others_did(percepts) == 1
    assert ledger.observed == 1
    assert ledger.acted == 1
    assert ledger.authored_share == 0.5
    assert ledger.last_event is not None and ledger.last_event.actor == "user"
    assert ledger.by_capability == {"intention": [1, 1]}


def test_nothing_that_names_nobody_reaches_the_ledger() -> None:
    assert AffectUpdatePhase._record_what_others_did([{"type": "error", "source": "action_grounding"}, "noise"]) == 0
    assert get_agency_ledger().observed == 0
