"""The layer is load-bearing, or it is a folder.

Each test here exercises a path that existed before this package and
checks that it now behaves differently, in the direction the mechanism
claims. Not that interiority produces a number — that the subsystem
downstream of it changed.
"""

from __future__ import annotations

import pytest

from core.interiority.event import EventKind, InteriorEvent
from core.interiority.evidence import measured
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


def test_an_outcome_reaches_back_to_the_faculties_that_fired(
    service: InteriorityService,
) -> None:
    """Delayed credit assignment: O3 on the council docket.

    Without this the faculties are frozen at the values they were written
    with, and no amount of living moves them.
    """
    service.ledger.goal("g", 0.9)
    service.ledger.notes.note_goal_delta("g", -0.8)
    event = InteriorEvent(
        kind=EventKind.GOAL, object="g", source="goal",
        observations={"timing": measured(0.8)},
    )
    state = service.tick(event, dt=0.1)
    assert state.transmitted, "nothing fired, so there is nothing to credit"

    for faculty in state.transmitted:
        assert service.attribution.hit_rate(faculty) is None, (
            "an unmeasured faculty must report None, not a default; a caller "
            "that cannot tell them apart will treat a guess as a finding"
        )

    credited = service.record_outcome(event_id=event.event_id, claim_held=True)
    assert credited["faculties_credited"], "the outcome reached nobody"
    for faculty in credited["faculties_credited"]:
        assert service.attribution.hit_rate(faculty) is not None


def test_an_outcome_that_names_nothing_is_dropped(
    service: InteriorityService,
) -> None:
    """Spreading credit over whoever was active is how a learner acquires
    confident nonsense."""
    before = service.attribution.snapshot()["outcomes_with_no_trace"]
    result = service.record_outcome(event_id="no-such-event", claim_held=True)
    assert result["faculties_credited"] == {}
    after = service.attribution.snapshot()["outcomes_with_no_trace"]
    assert after == before + 1, "the drop was not counted"


def test_a_standard_that_never_serves_her_becomes_one_she_merely_obeys() -> None:
    """O4: endorsement was set once and never moved.

    The difference between guilt and resentment is whether she holds the
    standard, so a standard that cannot move is a system that cannot tell
    those apart.
    """
    from core.interiority.ledger import RelationalLedger

    ledger = RelationalLedger()
    ledger.standing.norm("held", weight=0.9, endorsement=0.5)
    ledger.standing.norm("obeyed", weight=0.9, endorsement=0.5)
    for _ in range(20):
        ledger.standing.reinforce_norm("held", served_her_own=True)
        ledger.standing.reinforce_norm("obeyed", served_her_own=False)

    held = ledger.standing.norm_for("held")
    obeyed = ledger.standing.norm_for("obeyed")
    assert held.endorsement > 0.7, held
    assert obeyed.endorsement < 0.3, obeyed
    assert held.evidence == obeyed.evidence == 20


def test_a_channel_can_learn_what_it_means_within_a_bound() -> None:
    """O2: the mapping from channel to readiness was fixed forever.

    The priors are measured asymmetries — a face is controllable, a pause
    is not — so learning has to be able to adjust them without
    overwriting them. The parameter claims the bound is wide enough to
    reverse a weak prior and not a strong one, and this measures both
    rather than taking the number on faith.
    """
    from core.interiority.other_minds import _LOADINGS, OtherMindsModel

    model = OtherMindsModel()

    def read(value: float):
        model.estimate(
            InteriorEvent(
                kind=EventKind.SOCIAL, subject="p",
                observations={"timing": measured(0.2)},
            )
        )
        return model.estimate(
            InteriorEvent(
                kind=EventKind.SOCIAL, subject="p",
                observations={"timing": measured(value)},
            )
        )

    weak_before = _LOADINGS["timing"]["attend"]
    strong_before = _LOADINGS["timing"]["inhibit"]
    assert weak_before < 0 < strong_before

    for _ in range(30):
        model.record_outcome(read(0.9), actual_tendency="attend")

    effective = model._effective_loadings("timing")
    assert effective["attend"] > 0, (
        "a weak prior did not reverse under thirty consistent outcomes, so "
        "nothing is being learned"
    )
    assert effective["inhibit"] == strong_before, (
        "a strong published asymmetry moved; the drift bound is supposed to "
        "keep the finding this started from"
    )
    assert model.status()["learned_loadings"] > 0


def test_a_read_uses_whatever_the_senses_are_carrying() -> None:
    """N1: the channels were declared and nothing was connected to them.

    core/senses/interaction_signals.py has been producing typing
    hesitation, voice steadiness and gaze direction the whole time. A read
    restricted to what a caller remembered to pass is a read on two text
    channels, which is why it kept refusing to be confident.
    """
    import time

    from core.container import ServiceContainer
    from core.interiority.senses import channels_from, live_channels

    now = time.time()
    status = {
        "typing": {
            "updated_at": now, "active": True, "hesitation": 0.7,
            "pause_before_submit_ms": 3200.0, "correction_rate": 0.4,
        },
        "voice": {
            "updated_at": now, "activation": 0.8, "stress_cue": 0.6,
            "steadiness": 0.2, "speech_ratio": 0.7,
        },
        "vision": {
            "updated_at": now, "sample_available": True, "face_present": True,
            "mouth_motion_score": 0.5, "face_area_ratio": 0.3,
            "gaze_direction": "away", "attention_available": 0.3,
        },
    }
    channels = channels_from(status, now=now)
    assert set(channels) == {"timing", "prosody", "face", "posture"}

    # The unmanaged channels enter as measurements; the rough one does not.
    assert channels["timing"].provenance.name == "MEASURED"
    assert channels["face"].provenance.name == "INFERRED"
    assert channels["face"].confidence < channels["timing"].confidence, (
        "a backend that calls its own output a rough attention indicator "
        "should not enter at the strength of an unmanaged measurement"
    )

    class _Engine:
        def get_status(self):
            return status

    ServiceContainer.register("interaction_signals", _Engine())
    try:
        assert set(live_channels(now=now)) == set(channels)
    finally:
        ServiceContainer.register("interaction_signals", None)


def test_a_silent_microphone_is_not_a_silent_person() -> None:
    """Stale is absent, not zero.

    A system that cannot tell those apart will describe a dead sensor as a
    calm human being.
    """
    import time

    from core.interiority.senses import channels_from

    now = time.time()
    status = {
        "typing": {"updated_at": now, "active": True, "hesitation": 0.5},
        "voice": {"updated_at": now - 600, "activation": 0.9, "stress_cue": 0.9},
    }
    channels = channels_from(status, now=now)
    assert "timing" in channels
    assert "prosody" not in channels, (
        "a ten-minute-old voice sample was read as the present"
    )


def test_the_census_can_answer_what_the_constructed_proofs_cannot(
    service: InteriorityService,
) -> None:
    """N2: how often anything fires in ordinary use.

    Every other measurement here runs in a world the harness builds. This
    is the instrument that answers the question the constructed proofs
    cannot, and it has to be running before the first real turn or the
    answer is lost.
    """
    from core.interiority.faculty import registry

    service.census.reset_for_test()
    assert service.census.report()["turns"] == 0

    service.ledger.goal("g", 0.9)
    service.ledger.notes.note_goal_delta("g", -0.8)
    for step in range(6):
        service.tick(
            InteriorEvent(
                kind=EventKind.GOAL, object="g", source="goal",
                observations={"timing": measured(0.4 + step * 0.05)},
            ),
            dt=0.1,
        )

    report = service.census.report()
    assert report["turns"] == 6
    assert report["firing_rate"], "no faculty was recorded as firing"
    assert report["decline_reasons"], "declining is normal and must be counted"
    assert report["channel_availability"].get("timing") == 1.0

    # Both halves of the judgement are reported: a rate alone cannot tell a
    # mechanism firing hard every turn from one emitting almost nothing.
    for faculty, rate, mean in service.census.always_fires():
        assert 0.0 <= rate <= 1.0
        assert 0.0 <= mean <= 1.0

    never = service.census.never_fired(registry().ids())
    assert len(never) < 43, "nothing fired at all"


def test_the_service_is_retrievable_after_the_boot_path_registers_it() -> None:
    """The registrar is handed the orchestrator, not a container.

    An earlier version called `orchestrator.register(...)`, so on a booting
    runtime the service went somewhere nothing reads and every lookup
    returned None — while the boot log still said the engine had
    registered. It passed a test that handed it None and was wrong in the
    only case that matters, which is why this test hands it an object that
    fails if it is used as a container.
    """
    from core.container import ServiceContainer
    from core.orchestrator.initializers.derived_engines import (
        register_derived_engines,
    )

    class _Orchestrator:
        def register(self, *args: object, **kwargs: object) -> None:
            raise AssertionError(
                "the registrar used the orchestrator as a service container"
            )

    register_derived_engines(_Orchestrator())
    service = ServiceContainer.get("interiority", default=None)
    assert service is not None, (
        "registered according to the boot log, and unreachable to every "
        "consumer"
    )
    assert service.snapshot()["faculties"] == 43
