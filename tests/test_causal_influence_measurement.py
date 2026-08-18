"""Contract tests for measured causal influence.

The property under test throughout is that a claim of influence cannot be made
without the measurement that supports it — not by accumulating treatment trials
without a null, not by asserting a boolean, and not by filling a dict with
constants until it stops being empty.
"""

from __future__ import annotations

import asyncio

import pytest

from core.verify.causal_influence import (
    InfluenceLedger,
    Verdict,
    text_divergence,
    vector_divergence,
)
from core.verify.influence_probe import measure_channel, measure_channels
from core.verify.lesion_registry import (
    LesionRegistry,
    LesionUnavailable,
    apply_channel,
    get_lesion_registry,
    lesioned,
    register_flag_lesion,
)


# ---------------------------------------------------------------------------
# Divergence metrics
# ---------------------------------------------------------------------------


def test_identical_text_has_zero_divergence():
    assert text_divergence("the same words", "the same words") == 0.0


def test_disjoint_text_has_maximal_divergence():
    assert text_divergence("alpha beta gamma", "one two three") == 1.0


def test_divergence_is_symmetric_and_bounded():
    a, b = "a shared opening then divergence", "a shared opening then something else"
    forward = text_divergence(a, b)
    assert forward == text_divergence(b, a)
    assert 0.0 < forward < 1.0


def test_one_empty_side_is_total_divergence():
    assert text_divergence("", "anything at all") == 1.0
    assert text_divergence("", "") == 0.0


def test_vector_divergence_uses_direction_not_magnitude():
    assert vector_divergence([1.0, 0.0], [2.0, 0.0]) == pytest.approx(0.0)
    assert vector_divergence([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0)


def test_zero_vector_does_not_score_a_free_null():
    """A dead channel must not look identical to itself in a way that flatters it."""

    assert vector_divergence([0.0, 0.0], [1.0, 0.0]) == 1.0


# ---------------------------------------------------------------------------
# The refusal that matters
# ---------------------------------------------------------------------------


def test_treatment_without_a_null_can_never_reach_a_verdict():
    """The d=17.3 lesson, encoded.

    An A/B in this repository once scored a large, significant effect on its own
    null arm. Any amount of treatment divergence is compatible with a channel
    that does nothing, because the decoder is not deterministic. Without a null
    there is no verdict, at any sample size.
    """

    ledger = InfluenceLedger()
    for i in range(500):
        ledger.record_treatment(
            "starved",
            intact="one two three four five",
            lesioned="wholly different words entirely here now",
            turn_id=str(i),
        )

    verdict = ledger.verdict("starved")
    assert verdict.verdict is Verdict.UNMEASURED
    assert verdict.n_treatment == 500
    assert verdict.n_null == 0
    assert not verdict.is_influential
    assert "no null pairs" in verdict.reason


def test_a_channel_that_moves_the_output_is_influential():
    ledger = InfluenceLedger()
    for i in range(40):
        ledger.record_treatment(
            "real",
            intact="the reply as the system ships it",
            lesioned="an entirely unrelated sentence produced without it",
            turn_id=str(i),
        )
        # The null jitters: two intact runs are not byte-identical.
        ledger.record_null(
            "real",
            first="the reply as the system ships it",
            second=(
                "the reply as the system ships it"
                if i % 3
                else "the answer as the system ships it"
            ),
            turn_id=str(i),
        )

    verdict = ledger.verdict("real")
    assert verdict.verdict is Verdict.INFLUENTIAL
    assert verdict.is_influential
    assert verdict.ci_low > 0.0
    assert verdict.effect > verdict.noise_floor


def test_a_decorative_channel_is_reported_inert_not_influential():
    """Lesioning changes nothing beyond noise — and the receipt says so."""

    ledger = InfluenceLedger()
    for i in range(40):
        jittered = "steady output" if i % 3 else "steady result"
        ledger.record_treatment(
            "decorative", intact="steady output", lesioned=jittered, turn_id=str(i)
        )
        ledger.record_null(
            "decorative", first="steady output", second=jittered, turn_id=str(i)
        )

    verdict = ledger.verdict("decorative")
    assert verdict.verdict is Verdict.INERT
    assert not verdict.is_influential


def test_too_few_observations_stay_unmeasured_rather_than_inert():
    """Absence of evidence is not evidence of absence, and INERT is a claim."""

    ledger = InfluenceLedger()
    for i in range(3):
        ledger.record_treatment("thin", intact="a b c", lesioned="a b d", turn_id=str(i))
        ledger.record_null("thin", first="a b c", second="a b e", turn_id=str(i))

    assert ledger.verdict("thin").verdict is Verdict.UNMEASURED


def test_verdicts_are_reproducible_from_the_same_observations():
    """An unseeded bootstrap turns a boundary case into a coin flip."""

    def build() -> InfluenceLedger:
        ledger = InfluenceLedger()
        for i in range(30):
            ledger.record_treatment("c", intact="x y z", lesioned="p q r", turn_id=str(i))
            ledger.record_null("c", first="x y z", second="x y z", turn_id=str(i))
        return ledger

    first = build().verdict("c")
    second = build().verdict("c")
    assert first.as_dict() == second.as_dict()


def test_observations_survive_a_restart():
    ledger = InfluenceLedger()
    for i in range(30):
        ledger.record_treatment("c", intact="x y z", lesioned="p q r", turn_id=str(i))
        ledger.record_null("c", first="x y z", second="x y z", turn_id=str(i))
    before = ledger.verdict("c")

    revived = InfluenceLedger()
    revived.load(ledger.as_dict())
    assert revived.verdict("c").as_dict() == before.as_dict()


def test_an_unnamed_channel_is_refused():
    with pytest.raises(ValueError):
        InfluenceLedger().record_treatment("  ", intact="a", lesioned="b")


# ---------------------------------------------------------------------------
# Lesion registry
# ---------------------------------------------------------------------------


@pytest.fixture
def registry(monkeypatch) -> LesionRegistry:
    """A registry isolated from the process-wide one."""

    import core.verify.lesion_registry as module

    fresh = LesionRegistry()
    monkeypatch.setattr(module, "_REGISTRY", fresh)
    return fresh


def test_an_unregistered_channel_cannot_be_lesioned(registry):
    with pytest.raises(LesionUnavailable):
        with registry.lesion("nobody.registered.this"):
            pass


def test_flag_lesion_substitutes_the_neutral_only_while_lesioned(registry):
    register_flag_lesion(
        "test.channel",
        owner="tests",
        neutral="zero",
        direct_actuation=True,
    )
    assert apply_channel("test.channel", 0.42, neutral=0.0) == 0.42
    with registry.lesion("test.channel"):
        assert apply_channel("test.channel", 0.42, neutral=0.0) == 0.0
    assert apply_channel("test.channel", 0.42, neutral=0.0) == 0.42


def test_lesion_restores_when_the_body_raises(registry):
    register_flag_lesion("test.raise", owner="tests", neutral="n", direct_actuation=True)
    with pytest.raises(RuntimeError):
        with registry.lesion("test.raise"):
            raise RuntimeError("trial blew up")
    assert not registry.is_lesioned("test.raise")
    assert registry.active_lesions() == ()


def test_nested_lesions_restore_once_on_the_outermost_exit(registry):
    register_flag_lesion("test.nest", owner="tests", neutral="n", direct_actuation=True)
    with registry.lesion("test.nest"):
        with registry.lesion("test.nest"):
            assert registry.is_lesioned("test.nest")
        assert registry.is_lesioned("test.nest"), "inner exit must not restore"
    assert not registry.is_lesioned("test.nest")


def test_a_stateful_faculty_lesions_and_restores_exactly_once(registry):
    calls: list[str] = []
    register_flag_lesion("ignored", owner="tests", neutral="n", direct_actuation=True)
    from core.verify.lesion_registry import LesionHandle

    registry.register(
        LesionHandle(
            channel="test.stateful",
            lesion=lambda: calls.append("lesion"),
            restore=lambda: calls.append("restore"),
            owner="tests",
            neutral_description="flat",
            direct_actuation=True,
        )
    )
    with registry.lesion("test.stateful"):
        with registry.lesion("test.stateful"):
            pass
    assert calls == ["lesion", "restore"]


def test_a_lesion_is_invisible_to_a_turn_running_beside_it():
    """The safety property.

    A probe measuring the live runtime must not degrade a real person's reply.
    Lesions are scoped to the execution context, so a concurrent turn never
    sees one.
    """

    channel = "test.concurrency"
    register_flag_lesion(channel, owner="tests", neutral=0.0, direct_actuation=True)
    registry = get_lesion_registry()
    observed: dict[str, float] = {}

    async def probe() -> None:
        with lesioned(channel):
            await asyncio.sleep(0.05)
            observed["probe"] = apply_channel(channel, 1.0, neutral=0.0)

    async def user_turn() -> None:
        await asyncio.sleep(0.02)
        observed["user"] = apply_channel(channel, 1.0, neutral=0.0)

    async def main() -> None:
        await asyncio.gather(probe(), user_turn())

    try:
        asyncio.run(main())
        assert observed["probe"] == 0.0, "the probe must see its own lesion"
        assert observed["user"] == 1.0, "a concurrent turn must not be lesioned"
        assert registry.active_lesions() == ()
    finally:
        registry.unregister(channel)


# ---------------------------------------------------------------------------
# The probe
# ---------------------------------------------------------------------------


def _make_generator(channel: str, *, lesioned_output: str, intact_output: str):
    """A generator whose output genuinely depends on the channel."""

    counter = {"n": 0}

    async def generate() -> str:
        counter["n"] += 1
        # A little jitter so the null arm is not degenerate, exactly as a real
        # decoder above temperature zero behaves.
        suffix = " tail" if counter["n"] % 4 == 0 else ""
        return apply_channel(channel, intact_output, neutral=lesioned_output) + suffix

    return generate


def test_probe_runs_three_arms_per_trial_and_reaches_a_verdict(registry):
    channel = "test.probe.real"
    register_flag_lesion(channel, owner="tests", neutral="off", direct_actuation=True)
    ledger = InfluenceLedger()

    report = asyncio.run(
        measure_channel(
            channel,
            generate=_make_generator(
                channel,
                intact_output="the shaped reply carries these particular words",
                lesioned_output="unshaped output nothing like it at all here",
            ),
            trials=12,
            per_generation_timeout_s=5.0,
            deadline_s=30.0,
            ledger=ledger,
        )
    )

    assert report.trials_completed == 12
    assert report.generations == 36, "three arms per trial: intact, lesioned, intact"
    assert report.generation_failures == 0
    assert report.verdict is not None
    assert report.verdict.verdict is Verdict.INFLUENTIAL
    assert report.verdict.n_null == 12, "the null arm must be recorded, not skipped"


def test_probe_reports_inert_for_a_channel_that_changes_nothing(registry):
    channel = "test.probe.decorative"
    register_flag_lesion(channel, owner="tests", neutral="same", direct_actuation=False)
    ledger = InfluenceLedger()

    report = asyncio.run(
        measure_channel(
            channel,
            generate=_make_generator(
                channel,
                intact_output="identical either way",
                lesioned_output="identical either way",
            ),
            trials=12,
            per_generation_timeout_s=5.0,
            deadline_s=30.0,
            ledger=ledger,
        )
    )

    assert report.verdict is not None
    assert report.verdict.verdict is Verdict.INERT


def test_probe_discards_a_treatment_pair_whose_null_arm_failed(registry):
    """Recording treatment without its null would tilt toward an unearned verdict."""

    channel = "test.probe.partial"
    register_flag_lesion(channel, owner="tests", neutral="off", direct_actuation=True)
    ledger = InfluenceLedger()
    calls = {"n": 0}

    async def flaky() -> str:
        calls["n"] += 1
        if calls["n"] % 3 == 0:  # every third call is the null arm
            raise RuntimeError("null arm unavailable")
        return apply_channel(channel, "shaped reply", neutral="unshaped")

    report = asyncio.run(
        measure_channel(
            channel,
            generate=flaky,
            trials=5,
            per_generation_timeout_s=5.0,
            deadline_s=30.0,
            ledger=ledger,
        )
    )

    assert report.trials_completed == 0
    assert report.generation_failures == 5
    assert ledger.verdict(channel).n_treatment == 0, "no half-pairs may be recorded"
    assert ledger.verdict(channel).verdict is Verdict.UNMEASURED


def test_probe_refuses_a_channel_with_no_registered_lesion(registry):
    async def generate() -> str:
        return "irrelevant"

    with pytest.raises(LesionUnavailable):
        asyncio.run(
            measure_channel(
                "test.probe.unregistered",
                generate=generate,
                trials=1,
                per_generation_timeout_s=1.0,
                deadline_s=1.0,
            )
        )


def test_probe_bounds_are_mandatory(registry):
    register_flag_lesion("test.bounds", owner="tests", neutral="n", direct_actuation=True)

    async def generate() -> str:
        return "x"

    for kwargs in (
        {"trials": 0, "per_generation_timeout_s": 1.0, "deadline_s": 1.0},
        {"trials": 1, "per_generation_timeout_s": 0.0, "deadline_s": 1.0},
        {"trials": 1, "per_generation_timeout_s": 1.0, "deadline_s": 0.0},
    ):
        with pytest.raises(ValueError):
            asyncio.run(measure_channel("test.bounds", generate=generate, **kwargs))


def test_probe_stops_at_its_deadline(registry):
    channel = "test.probe.slow"
    register_flag_lesion(channel, owner="tests", neutral="off", direct_actuation=True)

    async def slow() -> str:
        await asyncio.sleep(0.05)
        return apply_channel(channel, "shaped", neutral="unshaped")

    report = asyncio.run(
        measure_channel(
            channel,
            generate=slow,
            trials=1000,
            per_generation_timeout_s=1.0,
            deadline_s=0.4,
            ledger=InfluenceLedger(),
        )
    )

    assert report.stopped_early == "deadline"
    assert report.trials_completed < 1000
    assert report.elapsed_s < 5.0


def test_probe_survives_a_generator_that_times_out(registry):
    channel = "test.probe.hang"
    register_flag_lesion(channel, owner="tests", neutral="off", direct_actuation=True)

    async def hangs() -> str:
        await asyncio.sleep(10.0)
        return "never"

    report = asyncio.run(
        measure_channel(
            channel,
            generate=hangs,
            trials=2,
            per_generation_timeout_s=0.05,
            deadline_s=5.0,
            ledger=InfluenceLedger(),
        )
    )

    assert report.trials_completed == 0
    assert report.generation_failures == 2
    assert not get_lesion_registry().active_lesions()


# ---------------------------------------------------------------------------
# The structured floor must not invent its own evidence
# ---------------------------------------------------------------------------


def _floor(context: dict, *, stamp: bool = True) -> dict:
    """Run the floor over a context, stamped the way a real one would be.

    _live_mind_controls_bound requires is_stamped_runtime_payload: the
    snapshot has to carry THIS process's mark, because think() accepts an
    arbitrary context and an unstamped dict reaching it could otherwise take
    control of temperature, top_p and recurrent depth. That hardening landed
    after these tests were written, so they handed over a bare dict, could
    never bind, and failed on a clean tree — while asserting things about
    binding that the fixture made unreachable.

    Stamping here restores what the tests are actually about. The unstamped
    case is worth its own assertion rather than being every case by accident,
    so `stamp=False` keeps it reachable.
    """
    from core.brain.cognitive_engine import CognitiveEngine
    from core.utils.injected_blocks import stamp_runtime_payload

    prepared = dict(context)
    live_mind_context = prepared.get("live_mind_context")
    if stamp and isinstance(live_mind_context, dict):
        prepared["live_mind_context"] = stamp_runtime_payload(dict(live_mind_context))

    return CognitiveEngine._live_mind_structured_floor_metadata(
        prepared, source="regression"
    )


def test_an_unstamped_snapshot_can_never_bind_controls():
    """think() accepts any context; only this process's own may steer it."""
    metadata = _floor(
        {
            "desktop_cognitive_engine_required": True,
            "live_mind_controls_bound": True,
            "live_mind_generation_controls": {"temperature": 0.61, "top_p": 0.9},
            "live_mind_context": {
                "mind_snapshot_quality": {"ready": True},
                "mind_snapshot": {"mood": "settled"},
                "required_subsystems_ok": True,
            },
        },
        stamp=False,
    )

    assert metadata["live_mind_controls_bound"] is False


def test_structured_floor_does_not_fabricate_controls_when_the_mind_is_absent():
    """The defect this file exists to keep shut.

    The floor used to fill live_mind_generation_controls with four constants
    when the real derivation came back empty, then set controls_bound=True
    because the dict was no longer empty — without ever checking that a
    mind_snapshot existed. The receipt for a turn the mind never touched was
    byte-identical to the receipt for a turn it shaped.
    """

    metadata = _floor(
        {
            "desktop_cognitive_engine_required": True,
            # Ready and healthy, but with no snapshot to derive anything from.
            "live_mind_snapshot_ready": True,
            "live_mind_required_subsystems_ok": True,
            "live_mind_context": {
                "mind_snapshot_quality": {"ready": True},
                "required_subsystems_ok": True,
            },
        }
    )

    assert metadata["live_mind_controls_bound"] is False
    assert metadata["live_mind_generation_controls"] == {}


def test_structured_floor_binds_when_a_real_snapshot_is_present():
    """The fix must not have simply nailed the field to False."""

    metadata = _floor(
        {
            "desktop_cognitive_engine_required": True,
            "live_mind_controls_bound": True,
            "live_mind_generation_controls": {
                "temperature": 0.61,
                "top_p": 0.9,
                "clean_user_surface_recurrent_loops": 2,
                "clean_user_surface_steering_alpha": 0.28,
            },
            "live_mind_context": {
                "mind_snapshot_quality": {"ready": True},
                "mind_snapshot": {"mood": "settled"},
                "required_subsystems_ok": True,
            },
        }
    )

    assert metadata["live_mind_controls_bound"] is True
    assert metadata["live_mind_generation_controls"]["temperature"] == 0.61


def test_structured_floor_separates_provenance_from_causality():
    """`bound` says the controls came from the mind. It never said they mattered."""

    metadata = _floor(
        {
            "desktop_cognitive_engine_required": True,
            "live_mind_controls_bound": True,
            "live_mind_generation_controls": {
                "temperature": 0.61,
                "top_p": 0.9,
                "clean_user_surface_recurrent_loops": 2,
                "clean_user_surface_steering_alpha": 0.28,
            },
            "live_mind_context": {
                "mind_snapshot_quality": {"ready": True},
                "mind_snapshot": {"mood": "settled"},
                "required_subsystems_ok": True,
            },
        }
    )

    influence = metadata["live_mind_influence"]
    assert metadata["live_mind_controls_bound"] is True
    assert influence["bound"] is False, (
        "provenance must never be reported as measured causal influence"
    )
    assert influence["status"] == "unmeasured"
    assert set(influence["unmeasured"]) >= {
        "live_mind.generation_controls",
        "live_mind.steering_alpha",
        "live_mind.context_block",
    }


def test_every_channel_the_floor_claims_is_measurable():
    """A claim nothing can falsify is not a claim. Each needs a registered lesion."""

    from core.verify.influence_receipt import unfalsifiable_channels

    metadata = _floor(
        {
            "desktop_cognitive_engine_required": True,
            "live_mind_context": {"mind_snapshot_quality": {"ready": True}},
        }
    )
    claimed = list(metadata["live_mind_influence"]["channels"])
    assert claimed, "the floor must name the channels it claims"
    assert unfalsifiable_channels(claimed) == ()


def test_measure_channels_shares_one_deadline(registry):
    channels = ["test.sweep.a", "test.sweep.b", "test.sweep.c"]
    for name in channels:
        register_flag_lesion(name, owner="tests", neutral="off", direct_actuation=True)

    async def slow() -> str:
        await asyncio.sleep(0.05)
        return "output"

    reports = asyncio.run(
        measure_channels(
            channels,
            generate=slow,
            trials=100,
            per_generation_timeout_s=1.0,
            deadline_s=0.3,
            ledger=InfluenceLedger(),
        )
    )

    assert len(reports) == 3
    assert any(r.stopped_early for r in reports)
