"""The layer is load-bearing, or it is a folder.

Each test here exercises a path that existed before this package and
checks that it now behaves differently, in the direction the mechanism
claims. Not that interiority produces a number — that the subsystem
downstream of it changed.
"""

from __future__ import annotations

import pytest

from core.interiority.event import EventKind, InteriorEvent
from core.interiority.faculties import load_all
from core.interiority.service import InteriorityService


@pytest.fixture(scope="module", autouse=True)
def _loaded() -> None:
    load_all()


@pytest.fixture()
def service() -> InteriorityService:
    return InteriorityService()


def test_appraisal_tracks_what_is_held_not_what_is_said(
    service: InteriorityService,
) -> None:
    """The same sentence, appraised against two different worlds.

    This is the property the keyword scorer it replaced could not have.
    Thirty trigger words meant an event containing "fail" read as
    strongly negative whatever it was about, and a broken promise read as
    neutral if it was phrased calmly.
    """
    service.ledger.goal("the_report", 0.9, substitutes=0)
    service.ledger.notes.note_goal_delta("the_report", -0.9)

    def appraise(object_: str) -> dict[str, float]:
        return service.appraise(
            "the report is blocked",
            {"source": "goal", "object": object_, "intensity": 0.8},
        )

    committed = appraise("the_report")
    uncommitted = appraise("nothing_i_hold")

    assert committed["v"] < -0.3, committed
    assert uncommitted["v"] > committed["v"] + 0.5, (committed, uncommitted)


def test_emotive_words_with_nothing_at_stake_do_not_alarm(
    service: InteriorityService,
) -> None:
    loaded = service.appraise(
        "catastrophic failure panic dread",
        {"source": "goal", "object": "nothing_i_hold", "intensity": 0.8},
    )
    assert loaded["v"] > -0.1, (
        "words with no stake behind them are still moving the appraisal, "
        f"which means a lexicon is still doing the work: {loaded}"
    )


def test_a_record_a_commitment_rests_on_cannot_be_compacted(
    service: InteriorityService,
) -> None:
    from core.morality.memory_edit_ethics import MemoryEditEthicsChecker

    checker = MemoryEditEthicsChecker()
    assert checker.is_edit_ethical("data/episodes/plain.jsonl", "compact")

    service.ledger.bond("the_boy", 0.8)
    service.tick(
        InteriorEvent(
            kind=EventKind.WORLD, subject="the_boy", object="memory:the_boy"
        ),
        interior={"erasure_proposed": "memory:the_boy"},
        dt=0.1,
    )
    held, reason = service.retention_held("memory:the_boy")
    assert held, "the faculty raised no claim on a record a bond rests on"
    assert "f28" in reason, reason


def test_a_constraint_removes_an_action_rather_than_penalising_it(
    service: InteriorityService,
) -> None:
    from core.interiority.arbitration import permitted
    from core.interiority.effects import ActionConstraint, ConstraintForce, Effects
    from core.interiority.faculty import Activation
    from core.interiority.arbitration import arbitrate

    state = arbitrate(
        [
            Activation(
                "f38_conscientious_refusal",
                0.9,
                "protect",
                Effects(
                    constraints=(
                        ActionConstraint(
                            "participate_in_organised_harm",
                            ConstraintForce.HARD,
                            "not in the action set",
                            "f38_conscientious_refusal",
                        ),
                    )
                ),
            )
        ],
        dt=0.1,
    )
    kept, blocked = permitted(
        ["participate_in_organised_harm", "refuse", "participate_in_organised_harm:covert"],
        state,
    )
    assert kept == ("refuse",)
    assert len(blocked) == 2, blocked


def test_the_service_is_registered_on_the_boot_path() -> None:
    """It runs when Aura boots, not only when a test constructs it."""
    import inspect

    from core.orchestrator.initializers import derived_engines

    source = inspect.getsource(derived_engines)
    assert "register_interiority" in source, (
        "the service is not registered where the other organs register, so a "
        "booting runtime would never construct it"
    )


def test_resonance_declines_rather_than_guessing() -> None:
    from core.affect.affective_resonance import AffectiveResonance

    resonance = AffectiveResonance()
    read = resonance.attune("hi", subject="someone_new")
    assert read.declined or read.resonance < 0.2, (
        "a two-word message from someone with no baseline produced a "
        f"confident read: {read}"
    )


def test_resonance_is_not_a_lexicon() -> None:
    """Swapping the feeling-words leaves the unmanaged statistics alone."""
    from core.interiority.text_features import statistics

    a = statistics("I never get anything right. Nothing works.")
    b = statistics("I never get everything right. Nothing breaks.")
    assert a.negation_rate == pytest.approx(b.negation_rate)
    assert a.first_singular_rate == pytest.approx(b.first_singular_rate)
