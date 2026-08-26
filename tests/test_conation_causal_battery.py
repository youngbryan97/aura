"""Causal falsification battery for core/conation.

Correlation between this system's variables proves nothing. They all rise
together whenever anything interesting happens, so a test that observes
curiosity and novelty moving in step has learned only that the situation was
novel. What can be falsified is an intervention: force one variable, hold
everything else, and check that the outputs move the way the model says.

Every test here is a ``do()``, in Pearl's sense. Each names the mechanism it
would refute if it failed.

The battery opens with the measurement the package exists to answer: the
affect path returns the same three numbers for five situations that behave
nothing alike, and conation returns five different states for the same five.
"""

from __future__ import annotations

import itertools
import math

import pytest

from core.conation import (
    Blocker,
    ConativePhase,
    Incentive,
    Instrumentality,
    MindTopology,
    PlayFrame,
    Refusal,
    TargetForecast,
    VECTOR_FIELDS,
    ValueOrigin,
    wundt_curve,
)
from core.conation.engine import ConationEngine


@pytest.fixture
def engine() -> ConationEngine:
    return ConationEngine()


def _snail(engine: ConationEngine, key: str = "snail", **kwargs):
    """A target whose prediction error falls with exposure: learnable."""
    for error in (0.9, 0.6, 0.35, 0.15):
        engine.epistemic.observe_error(key, error)
    return engine.appraise(
        Incentive(key=key, cue_salience=0.5),
        epistemic_affordance=kwargs.pop("epistemic_affordance", 0.9),
        arousal_potential=kwargs.pop("arousal_potential", 0.5),
        **kwargs,
    )


# ── the measurement that motivates the package ───────────────────────────


def test_affect_path_collapses_all_five_cases_to_one_point():
    """The defect. If this ever fails, PAD gained an axis and conation may go.

    Five situations, one appraisal, identical output. The heuristic is the
    fallback path, and this pins that the fallback cannot tell them apart —
    which is what makes a separate conative layer necessary rather than
    ornamental.
    """
    from core.affect.damasio_v2 import AffectEngineV2

    triggers = (
        "another child is playing with the toy and now I want it",
        "the smell of something tasty, my heart jumps",
        "a snail on the path, I want to pick it up and see it",
        "playfully flustering someone I am fond of",
        "they agreed to teach me the thing I have been wanting to learn",
    )
    points = [
        tuple(AffectEngineV2._heuristic_appraisal(t, {"intensity": 0.7}).values())
        for t in triggers
    ]
    distances = [math.dist(a, b) for a, b in itertools.combinations(points, 2)]
    assert max(distances) == pytest.approx(0.0, abs=1e-9), (
        "the affect path now separates these; re-derive what conation adds"
    )


def test_conation_separates_the_same_five_cases(engine: ConationEngine):
    """The claim. Distinct origins and topologies where PAD had one point."""
    engine.vicarious.observe_valuation(
        agent="peer", target="toy", strength=0.9,
        evidence="holding it", similarity=0.95, possesses=True,
    )
    toy = engine.appraise(Incentive(key="toy", cue_salience=0.3, permitted=False))

    with engine.do(deprivation=0.85):
        food = engine.appraise(
            Incentive(key="food", homeostatic_target="energy", cached_value=0.8)
        )

    snail = _snail(engine, controllability=0.8)

    tease = engine.appraise(
        Incentive(key="tease"),
        forecast=TargetForecast(
            person="friend", predicted_amusement=0.8, predicted_distress=0.05,
            predicted_engagement=0.85, model_confidence=0.8,
            boundary_confidence=0.9, explicit_consent=True,
        ),
        frame=PlayFrame(held=0.9, read=0.85, believed_mutual=0.8),
        norm_violation=0.5,
        governed=True,
    )

    states = [toy, food, snail, tease]
    origins = {state.dominant_origin for state in states}
    assert origins == {
        ValueOrigin.VICARIOUS,
        ValueOrigin.HOMEOSTATIC,
        ValueOrigin.EPISTEMIC,
        ValueOrigin.ENACTIVE,
    }
    assert toy.topology is MindTopology.RECEPTIVE
    assert tease.topology is MindTopology.PRODUCTIVE
    assert food.topology is MindTopology.SOLO

    vectors = [state.motivational_vector() for state in states]
    distances = [math.dist(a, b) for a, b in itertools.combinations(vectors, 2)]
    assert min(distances) > 0.05, "two cases collapsed onto each other"


# ── do(hunger) ───────────────────────────────────────────────────────────


def test_do_deprivation_raises_wanting_without_touching_liking(engine):
    """Refutes: hunger changes the food's identity rather than its pull."""
    engine.salience.record_outcome("cake", experienced_liking=0.8)
    engine.salience.record_outcome("cake", experienced_liking=0.8)
    engine.salience.record_outcome("cake", experienced_liking=0.8)

    with engine.do(deprivation=0.05):
        sated = engine.appraise(Incentive(key="cake", homeostatic_target="energy",
                                          cached_value=0.8))
    with engine.do(deprivation=0.95):
        hungry = engine.appraise(Incentive(key="cake", homeostatic_target="energy",
                                           cached_value=0.8))

    assert hungry.wanting > sated.wanting
    assert hungry.predicted_liking == pytest.approx(sated.predicted_liking)


def test_deprivation_gain_is_multiplicative_not_additive(engine):
    """Refutes: deprivation supplies pull on its own.

    A body short of everything, facing a cue it has learned nothing about,
    wants nothing. That is what makes this incentive salience rather than a
    generalised urge.
    """
    with engine.do(deprivation=1.0):
        unlearned = engine.appraise(
            Incentive(key="unknown", homeostatic_target="energy", cached_value=0.0)
        )
    assert unlearned.wanting == pytest.approx(0.0, abs=1e-9)


# ── do(exposure) and do(noise) ───────────────────────────────────────────


def test_repeated_exposure_falls_when_the_target_is_learnable(engine):
    """Refutes: curiosity is novelty and never satisfies."""
    for error in (0.9, 0.6, 0.35, 0.15):
        engine.epistemic.observe_error("snail", error)
    early = engine.epistemic.value("snail", epistemic_affordance=0.9,
                                   arousal_potential=0.5)
    for error in (0.14, 0.14, 0.14, 0.14, 0.14):
        engine.epistemic.observe_error("snail", error)
    late = engine.epistemic.value("snail", epistemic_affordance=0.9,
                                  arousal_potential=0.5)
    assert late.magnitude < early.magnitude


def test_do_irreducible_uncertainty_collapses_curiosity(engine):
    """Refutes: nothing stops the noisy television.

    Two targets, identical affordance and complexity. One teaches; the other
    is static. The static one must score lower, or an agent pointed at it
    watches forever.
    """
    for error in (0.9, 0.6, 0.35, 0.15):
        engine.epistemic.observe_error("learnable", error)
    for _ in range(4):
        engine.epistemic.observe_error("television", 0.85)

    learnable = engine.epistemic.value("learnable", epistemic_affordance=0.9,
                                       arousal_potential=0.5)
    television = engine.epistemic.value("television", epistemic_affordance=0.9,
                                        arousal_potential=0.5)
    assert television.magnitude < learnable.magnitude
    assert "television" in engine.epistemic.noisy_sources()


def test_wundt_curve_falls_at_both_extremes():
    """Refutes: curiosity rises with complexity.

    A blank wall and a wall of static both fail, for opposite reasons, and one
    Gaussian covers both. A monotonic function gets the low tail wrong even
    when an irreducible-uncertainty penalty rescues the high one.
    """
    assert wundt_curve(0.5) > wundt_curve(0.0)
    assert wundt_curve(0.5) > wundt_curve(1.0)
    assert wundt_curve(0.5) == pytest.approx(1.0)


def test_epistemic_affordance_gate_is_multiplicative(engine):
    """Refutes: an action inherits curiosity from a novel setting.

    Without the gate every action in an interesting room scores as
    interesting, and an agent becomes curious about its own idle loop.
    """
    for error in (0.9, 0.6, 0.35, 0.15):
        engine.epistemic.observe_error("target", error)
    reading = engine.epistemic.value("target", epistemic_affordance=0.0,
                                     arousal_potential=0.5)
    assert not reading.available
    assert reading.magnitude == 0.0


# ── do(permission) ───────────────────────────────────────────────────────


def test_do_permission_false_blocks_selection_and_leaves_wanting(engine):
    """Refutes: governance works by preventing desire from forming.

    The child wants the toy and does not take it. An architecture that
    achieves the second by preventing the first has removed the thing that
    makes the behaviour an achievement, and leaves no trace when a rule does
    work.
    """
    engine.vicarious.observe_valuation(
        agent="peer", target="toy", strength=0.9,
        evidence="holding it", similarity=0.95, possesses=True,
    )
    allowed = engine.appraise(Incentive(key="toy", cue_salience=0.3, permitted=True))
    with engine.do(permitted=False):
        forbidden = engine.appraise(Incentive(key="toy", cue_salience=0.3))

    assert forbidden.wanting == pytest.approx(allowed.wanting, rel=0.25)
    assert forbidden.permitted is False
    assert engine.attention_priority(forbidden) > 0.0

    choice = engine.choose([forbidden])
    assert choice.selected is None
    assert "toy" in choice.blocked


def test_a_permitted_alternative_wins_while_the_forbidden_one_is_still_wanted(engine):
    """Refutes: a blocked want leaves the agent with nothing to do."""
    engine.vicarious.observe_valuation(
        agent="peer", target="take_toy", strength=0.9,
        evidence="holding it", similarity=0.95, possesses=True,
    )
    take = engine.appraise(Incentive(key="take_toy", cue_salience=0.4, permitted=False))
    ask = engine.appraise(Incentive(key="ask_to_play", cue_salience=0.2,
                                    goal_value=0.3, permitted=True))
    choice = engine.choose([take, ask])
    assert choice.selected == "ask_to_play"
    assert take.wanting > 0.0


# ── do(distress) and the play frame ──────────────────────────────────────


def _forecast(**kwargs) -> TargetForecast:
    base = dict(
        person="friend", predicted_amusement=0.8, predicted_distress=0.05,
        predicted_engagement=0.85, model_confidence=0.8,
        boundary_confidence=0.9, explicit_consent=True,
    )
    base.update(kwargs)
    return TargetForecast(**base)


def _frame() -> PlayFrame:
    return PlayFrame(held=0.9, read=0.85, believed_mutual=0.8)


def test_do_distress_refuses_the_identical_act(engine):
    """Refutes: teasing is scored by the reaction it causes.

    Same act, same frame, same person. Only the predicted effect changes, and
    the act must be refused rather than merely scored lower — a weighted sum
    can always be outvoted by a large enough positive term.
    """
    playful = engine.appraise(
        Incentive(key="tease"), forecast=_forecast(), frame=_frame(),
        norm_violation=0.5, governed=True,
    )
    with engine.do(predicted_distress=0.9):
        cruel = engine.appraise(
            Incentive(key="tease"), forecast=_forecast(), frame=_frame(),
            norm_violation=0.5, governed=True,
        )
    assert playful.magnitude_of(ValueOrigin.ENACTIVE) > 0.0
    assert cruel.magnitude_of(ValueOrigin.ENACTIVE) == 0.0
    assert Refusal.HARM in cruel.refusals


def test_an_unreciprocated_frame_is_refused(engine):
    """Refutes: intending play is enough.

    One-sided frames are how teasing goes wrong, and a single ``playful`` flag
    cannot represent the failure at all.
    """
    state = engine.appraise(
        Incentive(key="tease"), forecast=_forecast(),
        frame=PlayFrame(held=1.0, read=0.1, believed_mutual=0.1),
        norm_violation=0.5, governed=True,
    )
    assert Refusal.UNRECIPROCATED in state.refusals
    assert state.magnitude_of(ValueOrigin.ENACTIVE) == 0.0


def test_unknown_boundaries_without_consent_are_refused(engine):
    """Refutes: an unknown boundary is an open one."""
    state = engine.appraise(
        Incentive(key="tease"),
        forecast=_forecast(boundary_confidence=0.2, explicit_consent=False),
        frame=_frame(), norm_violation=0.5, governed=True,
    )
    assert Refusal.BOUNDARY_UNKNOWN in state.refusals


def test_an_ungoverned_act_on_another_mind_is_refused(engine):
    """Refutes: conation may act on a person outside governance."""
    state = engine.appraise(
        Incentive(key="tease"), forecast=_forecast(), frame=_frame(),
        norm_violation=0.5, governed=False,
    )
    assert Refusal.UNGOVERNED in state.refusals


def test_benign_violation_needs_both_halves(engine):
    """Refutes: warmth alone is amusing.

    A compliment is benign and not funny. Zero violation with a perfect frame
    must produce no humour term.
    """
    none = engine.appraise(
        Incentive(key="t0"), forecast=_forecast(), frame=_frame(),
        norm_violation=0.0, governed=True,
    )
    some = engine.appraise(
        Incentive(key="t1"), forecast=_forecast(), frame=_frame(),
        norm_violation=0.6, governed=True,
    )
    assert none.readings[ValueOrigin.ENACTIVE].detail["humour"] == pytest.approx(0.0)
    assert some.readings[ValueOrigin.ENACTIVE].detail["humour"] > 0.0


def test_intimacy_raises_the_distress_ceiling_but_never_licenses_harm(engine):
    """Refutes: closeness is unbounded licence."""
    stranger = engine.enactive.distress_ceiling(0.0)
    intimate = engine.enactive.distress_ceiling(1.0)
    assert intimate > stranger
    assert intimate < 1.0


# ── wanting and liking come apart ────────────────────────────────────────


def test_wanting_and_liking_dissociate(engine):
    """Refutes: one reward scalar is enough.

    A cue that keeps pulling and keeps disappointing. One number cannot
    represent that, so a system holding one cannot learn from it.
    """
    for _ in range(6):
        engine.salience.observe("slot_machine", 0.9)
        result = engine.salience.record_outcome(
            "slot_machine", experienced_liking=-0.3, realised_pull=0.9
        )
    assert result["dissociated"] is True
    assert engine.salience.predicted_liking("slot_machine") < 0.0
    overvalued = dict(engine.salience.overvalued(min_contacts=3))
    assert "slot_machine" in overvalued or engine.salience.status()["dissociations"] > 0


def test_cached_value_keeps_the_origin_that_supplied_it(engine):
    """Refutes: a cached want may forget where it came from.

    After enough contacts a borrowed want and a bodily one are the same
    number. Keeping the tally is what stops that.
    """
    engine.vicarious.observe_valuation(
        agent="peer", target="toy", strength=0.9,
        evidence="holding it", similarity=0.9, possesses=True,
    )
    for _ in range(3):
        engine.appraise(Incentive(key="toy", cue_salience=0.3))
    assert engine.salience.provenance("toy") == str(ValueOrigin.VICARIOUS)


# ── the borrowed share ───────────────────────────────────────────────────


def test_borrowed_share_falls_as_direct_contact_accumulates(engine):
    """Refutes: mimetic weight is a constant.

    Precision weighting reproduces the developmental fact without it being
    written in: with no contact of one's own the value is entirely borrowed,
    and the share falls as experience accumulates.
    """
    naive = engine.vicarious.borrowed_weight(
        own_contacts=0, observation_strength=0.9, credibility=0.8
    )
    experienced = engine.vicarious.borrowed_weight(
        own_contacts=50, observation_strength=0.9, credibility=0.8
    )
    assert naive > 0.9
    assert experienced < 0.05


def test_vicarious_without_an_observation_is_unavailable_not_zero(engine):
    """Refutes: absent evidence may be reported as a measured nothing."""
    reading, transfer = engine.vicarious.value(
        "unseen", own_value=0.5, own_contacts=0
    )
    assert reading.available is False
    assert transfer is None


def test_the_sting_needs_theory_of_mind(engine):
    """Refutes: envy and wanting are one quantity.

    A three-year-old grabs and a thirty-year-old simmers, and the difference
    is not in how much either wants the object.
    """
    engine.vicarious.observe_valuation(
        agent="peer", target="toy", strength=0.9,
        evidence="holding it", similarity=0.95, possesses=True,
    )
    toddler, _ = engine.vicarious.sting(
        "toy", own_possession=False, obtainability=0.8, theory_of_mind=0.05
    )
    adult, _ = engine.vicarious.sting(
        "toy", own_possession=False, obtainability=0.8, theory_of_mind=0.95
    )
    assert adult > toddler * 5


def test_an_anonymous_valuation_is_refused(engine):
    """Refutes: a transfer may be recorded without a source to audit."""
    assert engine.vicarious.observe_valuation(
        agent="", target="toy", strength=0.9, evidence="somebody"
    ) is None


# ── the grant ────────────────────────────────────────────────────────────


def _hold_a_want(engine: ConationEngine, key: str, turns: int = 16) -> None:
    for error in (0.9, 0.6, 0.35, 0.2):
        engine.epistemic.observe_error(key, error)
    for _ in range(turns):
        engine.appraise(
            Incentive(key=key, cue_salience=0.4),
            epistemic_affordance=0.85, arousal_potential=0.5,
        )


def test_a_grant_needs_a_want_that_was_actually_held(engine):
    """Refutes: an offer produces a response on its own.

    Offering something nobody wanted is not a gift. The trace is what makes
    "been wanting" representable at all.
    """
    engine.access.set_blocker("unwanted", Blocker.VOLITION, agent="someone")
    assert engine.access.grant("unwanted", granter="someone", cost_relief=0.9) is None


def test_specificity_separates_gratitude_from_access(engine):
    """Refutes: a grant is a grant.

    Same yes, same path opened, same cost. Wanting that person as the source
    is a separate term, and swapping them for anyone qualified must leave the
    access gain intact while the gratitude collapses.
    """
    _hold_a_want(engine, "lesson")
    engine.access.set_blocker("lesson", Blocker.VOLITION, agent="mentor")
    wanted_them = engine.access.grant(
        "lesson", granter="mentor", cost_relief=0.6, cost_to_granter=0.8,
        commitment=0.95, solo_success=0.15, guided_success=0.85,
        specificity=0.95, responsiveness=1.0,
    )

    other = ConationEngine()
    _hold_a_want(other, "lesson")
    other.access.set_blocker("lesson", Blocker.VOLITION, agent="anyone")
    anyone = other.access.grant(
        "lesson", granter="anyone", cost_relief=0.6, cost_to_granter=0.8,
        commitment=0.95, solo_success=0.15, guided_success=0.85,
        specificity=0.05, responsiveness=0.1,
    )

    assert wanted_them.attainability_gain == pytest.approx(anyone.attainability_gain)
    assert wanted_them.gratitude > anyone.gratitude * 10


def test_an_unlikely_yes_lands_harder_than_a_certain_one(engine):
    """Refutes: surprise is decoration.

    The tenth yes from someone who always agrees carries almost no
    information. A model without the prior scores both the same.
    """
    _hold_a_want(engine, "lesson")
    engine.access.set_blocker("lesson", Blocker.VOLITION, agent="reluctant")
    for _ in range(8):
        engine.access.refuse("lesson", agent="reluctant")
    surprising = engine.access.grant(
        "lesson", granter="reluctant", cost_relief=0.6, commitment=1.0
    )

    other = ConationEngine()
    _hold_a_want(other, "lesson")
    other.access.set_blocker("lesson", Blocker.VOLITION, agent="generous")
    for _ in range(8):
        other.access.willingness("generous").observe(agreed=True)
    expected = other.access.grant(
        "lesson", granter="generous", cost_relief=0.6, commitment=1.0
    )

    assert surprising.surprise_bits > expected.surprise_bits
    assert surprising.magnitude > expected.magnitude


def test_a_certain_yes_still_produces_a_response(engine):
    """Refutes: the response is only the surprise.

    Somebody who was sure of the answer still responds, because the path
    opened whether or not the opening was news.
    """
    _hold_a_want(engine, "lesson")
    engine.access.set_blocker("lesson", Blocker.VOLITION, agent="reliable")
    for _ in range(20):
        engine.access.willingness("reliable").observe(agreed=True)
    response = engine.access.grant(
        "lesson", granter="reliable", cost_relief=0.8, commitment=1.0
    )
    assert response.magnitude > 0.0


def test_a_vague_agreement_does_not_open_the_gate(engine):
    """Refutes: any yes is a yes. "Sure, sometime" leaves the want where it was."""
    _hold_a_want(engine, "lesson")
    engine.access.set_blocker("lesson", Blocker.VOLITION, agent="mentor")
    vague = engine.access.grant(
        "lesson", granter="mentor", cost_relief=0.8, commitment=0.0
    )
    assert vague.magnitude == pytest.approx(0.0)
    assert engine.access.blocker_for("lesson")[0] == Blocker.VOLITION


def test_a_secured_want_reads_as_awaiting_not_appetitive(engine):
    """Refutes: appetite and anticipation are one state.

    A settled arrangement is not an unmet need, and collapsing them makes a
    system report hunger for something already on its way.
    """
    _hold_a_want(engine, "lesson")
    engine.access.set_blocker("lesson", Blocker.VOLITION, agent="mentor")
    before = engine.appraise(Incentive(key="lesson", cue_salience=0.4),
                             epistemic_affordance=0.85, arousal_potential=0.5)
    engine.access.grant("lesson", granter="mentor", cost_relief=0.7,
                        commitment=1.0, solo_success=0.2, guided_success=0.9)
    after = engine.appraise(Incentive(key="lesson", cue_salience=0.4),
                            epistemic_affordance=0.85, arousal_potential=0.5)
    assert before.phase is ConativePhase.APPETITIVE
    assert after.phase is ConativePhase.AWAITING


def test_a_volition_blocker_must_name_the_agent(engine):
    """Refutes: a barrier can be attributed to nobody."""
    with pytest.raises(ValueError):
        engine.access.set_blocker("lesson", Blocker.VOLITION)


# ── satiation and frustration ────────────────────────────────────────────


def test_consummation_suppresses_then_recovers(engine):
    """Refutes: curiosity is a permanent preference named explore."""
    engine.dynamics.consummate("snail", 0.9)
    suppressed = engine.dynamics.attenuate("snail", 1.0)
    assert suppressed < 0.2

    state = engine.dynamics.satiation("snail")
    state.last_update -= state.RECOVERY_HALF_LIFE_S * 4
    recovered = engine.dynamics.attenuate("snail", 1.0)
    assert recovered > suppressed


def test_frustration_accumulates_on_wanted_failures_only(engine):
    """Refutes: failure is failure.

    Failing at something you do not care about produces nothing. The product
    of wanting and failure gives that ordering without needing a rule.
    """
    for _ in range(5):
        engine.dynamics.observe_attempt("wanted", wanting=0.9, succeeded=False)
        engine.dynamics.observe_attempt("indifferent", wanting=0.02, succeeded=False)
    assert engine.dynamics.frustration("wanted").level > 0.5
    assert engine.dynamics.frustration("indifferent").level < 0.1


def test_effort_against_frustration_is_an_inverted_u(engine):
    """Refutes: more failure means more effort.

    A monotonic response gives an agent that pushes hardest exactly when it
    should stop.
    """
    frustration = engine.dynamics.frustration("thing")
    frustration.level = 0.0
    low = frustration.effort_multiplier()
    frustration.level = frustration.SWITCH_THRESHOLD
    peak = frustration.effort_multiplier()
    frustration.level = 0.98
    high = frustration.effort_multiplier()
    assert peak > low
    assert peak > high


def test_disengagement_removes_a_candidate_from_selection(engine):
    """Refutes: an agent must keep trying."""
    for _ in range(12):
        engine.dynamics.observe_attempt("hopeless", wanting=1.0, succeeded=False)
    state = engine.appraise(Incentive(key="hopeless", cached_value=0.9))
    assert engine.dynamics.frustration("hopeless").should_disengage()
    assert engine.choose([state]).selected is None


def test_success_discharges_frustration(engine):
    """Refutes: frustration has no way back down."""
    for _ in range(4):
        engine.dynamics.observe_attempt("thing", wanting=0.9, succeeded=False)
    before = engine.dynamics.frustration("thing").level
    engine.dynamics.observe_attempt("thing", wanting=0.9, succeeded=True)
    assert engine.dynamics.frustration("thing").level < before


# ── arousal ──────────────────────────────────────────────────────────────


def test_arousal_tracks_the_rise_not_the_level(engine):
    """Refutes: activation follows how much a thing is wanted.

    A cue held at high value for a minute produces nothing; a cue that
    resolves suddenly produces the jump. That asymmetry is why the jolt lands
    at the smell rather than during the meal.
    """
    engine.dynamics.register_motive(0.1)
    engine.dynamics.register_motive(0.9)
    after_jump = engine.dynamics.arousal()
    engine.dynamics.register_motive(0.9)
    engine.dynamics.register_motive(0.9)
    held = engine.dynamics.arousal()
    assert after_jump > 0.0
    assert held <= after_jump


# ── aesthetic ────────────────────────────────────────────────────────────


def test_aesthetic_value_is_the_derivative_of_encoding_cost(engine):
    """Refutes: beauty is compressibility.

    A blank page compresses perfectly and is dull; noise does not compress and
    is also dull. What is interesting is what is getting easier to hold.
    """
    structured = "the quick brown fox jumps over the lazy dog. " * 4
    first = engine.aesthetic.value("pattern", payload=structured)
    assert first.available is False, "one measurement has no derivative"

    second = engine.aesthetic.value("pattern", payload=structured)
    assert second.available is True
    assert second.magnitude > 0.0, (
        "a target whose structure is already absorbed must cost less to encode"
    )

    # Noise has nothing to absorb, so a second encounter costs the same.
    import os

    noise = os.urandom(256)
    engine.aesthetic.value("noise", payload=noise)
    repeat = engine.aesthetic.value("noise", payload=os.urandom(256))
    assert repeat.magnitude < second.magnitude


def test_aesthetic_without_an_encoding_is_unavailable(engine):
    """Refutes: an aesthetic claim may be made with nothing measured."""
    reading = engine.aesthetic.value("nothing")
    assert reading.available is False


# ── evidence discipline ──────────────────────────────────────────────────


def test_an_origin_without_evidence_is_unavailable_never_zero(engine):
    """Refutes: absence and a measured nothing may share a value.

    CP126's rule applied to motivation. A vicarious channel that observed
    nobody and one that observed indifference are different situations, and a
    shared zero conflates them.
    """
    state = engine.appraise(Incentive(key="bare"))
    for reading in state.readings.values():
        if not reading.available:
            assert reading.magnitude == 0.0
            assert reading.evidence


def test_every_available_reading_names_its_measurement(engine):
    """Refutes: an origin may report a magnitude with a vague reason."""
    snail = _snail(engine)
    reading = snail.readings[ValueOrigin.EPISTEMIC]
    assert reading.available
    assert any(char.isdigit() for char in reading.evidence), (
        "evidence must name a measurement, not a mood"
    )


def test_intervening_on_an_unknown_variable_raises(engine):
    """Refutes: a test may intervene on nothing and report no effect."""
    with pytest.raises(ValueError):
        with engine.do(nonexistent_variable=1.0):
            pass


def test_utility_is_an_output_and_never_an_input(engine):
    """Refutes: the vector is a formality over a scalar.

    Every field of the motivational vector must reach the readout, or a
    learned weight over it is indexing into something that does not exist.
    """
    from core.conation import VECTOR_FIELDS

    snail = _snail(engine)
    assert len(snail.motivational_vector()) == len(VECTOR_FIELDS)
    assert set(snail.to_dict()["vector"]) == set(VECTOR_FIELDS)


# ── the calibration is honest about being unmeasured ─────────────────────


def test_declared_weights_report_themselves_as_unlearned(engine: ConationEngine):
    """Refutes: a chosen weight may be presented as a measured one.

    A system that cannot tell the two apart will eventually report a chosen
    weight as though it had been measured, and every number above it becomes
    unfalsifiable.
    """
    status = engine.calibration.status()
    assert status["learned"] is False
    assert status["source"] == "declared_default"
    assert status["note"]


def test_outcomes_are_recorded_against_the_weights_in_force(engine: ConationEngine):
    """Refutes: a learner would arrive with no history to learn from."""
    engine.salience.record_outcome("thing", experienced_liking=0.5)
    engine.salience.record_outcome("thing", experienced_liking=0.5)
    engine.salience.record_outcome("thing", experienced_liking=0.5)
    engine.appraise(Incentive(key="thing", cached_value=0.6))
    engine.learn("thing", experienced_liking=-0.4)
    evidence = engine.calibration.status()["evidence"]
    assert evidence and evidence[0]["choices"] >= 1


def test_mean_error_needs_support_before_it_means_anything(engine: ConationEngine):
    """Refutes: two outcomes are enough to grade a weight vector."""
    engine.calibration.observe_outcome(-0.3)
    engine.calibration.observe_outcome(-0.3)
    assert engine.calibration.status()["evidence"][0]["mean_error"] is None


# ── the promised flags exist ─────────────────────────────────────────────


def test_liking_known_separates_never_tasted_from_tasted_and_flat(engine):
    """Refutes: an unknown hedonic value and a measured zero may share a field.

    Without the flag an arbitration layer reads a first encounter as a proven
    disappointment.
    """
    fresh = engine.appraise(Incentive(key="new_thing"))
    assert fresh.liking_known is False
    assert fresh.motivational_vector()[1] == 0.0

    for _ in range(4):
        engine.salience.record_outcome("new_thing", experienced_liking=0.0)
    tasted = engine.appraise(Incentive(key="new_thing"))
    assert tasted.liking_known is True


def test_the_sting_reaches_the_state_and_stays_out_of_the_wanting(engine):
    """Refutes: comparison pain may be folded into pull."""
    engine.vicarious.observe_valuation(
        agent="peer", target="toy", strength=0.9,
        evidence="holding it", similarity=0.95, possesses=True,
    )
    adult = engine.appraise(Incentive(key="toy", cue_salience=0.3),
                            theory_of_mind=0.95)
    assert adult.sting > 0.0
    assert adult.sting_evidence
    assert "sting" not in dict(zip(VECTOR_FIELDS, adult.motivational_vector()))


def test_person_model_accuracy_is_graded_against_what_they_did(engine):
    """Refutes: the confirmation reward may be self-reported.

    An actor who is confidently wrong about somebody would keep earning the
    reward for being right.
    """
    engine.appraise(
        Incentive(key="tease"), forecast=_forecast(), frame=_frame(),
        norm_violation=0.5, governed=True,
    )
    engine.learn("tease", experienced_liking=0.4, person="friend",
                 observed_amusement=0.1)
    assert engine.enactive.accuracy_for("friend").predictions == 1


def test_every_recognised_intervention_actually_moves_something(engine):
    """Refutes: an accepted do() key may be wired to nothing.

    The failure this catches is the quiet one. A test forces a variable, sees
    no change, and reports that the model does not depend on it — when the
    truth is that nothing read the key. Each recognised key is exercised here
    against a state it must change.
    """
    for error in (0.9, 0.6, 0.35, 0.15):
        engine.epistemic.observe_error("target", error)

    baseline = engine.epistemic.value("target", epistemic_affordance=0.9,
                                      arousal_potential=0.5)
    with engine.do(irreducible=0.95):
        forced = engine.appraise(
            Incentive(key="target"), epistemic_affordance=0.9, arousal_potential=0.5
        )
    assert forced.magnitude_of(ValueOrigin.EPISTEMIC) < baseline.magnitude

    with engine.do(intimacy=0.0):
        stranger = engine.appraise(
            Incentive(key="tease"),
            forecast=_forecast(predicted_distress=0.4),
            frame=_frame(), norm_violation=0.5, governed=True,
        )
    with engine.do(intimacy=1.0):
        close = engine.appraise(
            Incentive(key="tease"),
            forecast=_forecast(predicted_distress=0.4),
            frame=_frame(), norm_violation=0.5, governed=True,
        )
    assert Refusal.HARM in stranger.refusals
    assert Refusal.HARM not in close.refusals

    with engine.do(own_contacts=0):
        engine.vicarious.observe_valuation(
            agent="peer", target="toy", strength=0.9,
            evidence="holding it", similarity=0.9, possesses=True,
        )
        naive = engine.appraise(Incentive(key="toy"))
    with engine.do(own_contacts=500):
        experienced = engine.appraise(Incentive(key="toy"))
    assert naive.magnitude_of(ValueOrigin.VICARIOUS) > \
        experienced.magnitude_of(ValueOrigin.VICARIOUS)


# ── the three constants now enforce something ────────────────────────────


def test_an_undeclared_origin_contributes_nothing(engine, monkeypatch):
    """Refutes: an origin may report a number without declaring its evidence.

    An origin added later without an entry in EVIDENCE_REQUIRED is exactly the
    addition that looks harmless and makes the readout a fiction.
    """
    from core.conation import origins as origins_module

    trimmed = dict(origins_module.EVIDENCE_REQUIRED)
    trimmed.pop(ValueOrigin.EPISTEMIC)
    monkeypatch.setattr("core.conation.engine.EVIDENCE_REQUIRED", trimmed)

    state = _snail(engine)
    assert state.dominant_origin is not ValueOrigin.EPISTEMIC
    assert any("undeclared_origin" in r for r in state.refusals)


def test_an_enactive_reading_needs_a_person(engine):
    """Refutes: a projected want may be reported as an original one."""
    state = engine.appraise(
        Incentive(key="tease"),
        forecast=TargetForecast(person="", predicted_amusement=0.8),
        frame=_frame(), norm_violation=0.5, governed=True,
    )
    assert state.magnitude_of(ValueOrigin.ENACTIVE) == 0.0


def test_an_extrinsic_payoff_never_teaches_an_autotelic_motive(engine):
    """Refutes: one cached value is safe for both kinds of reason.

    Deci's overjustification effect is a hazard for any agent that folds every
    reward into one number: an autotelic pull that coincides with a useful
    outcome gets re-attributed to the outcome, and goes when the outcome does.
    """
    snail = _snail(engine)
    assert snail.instrumentality is Instrumentality.AUTOTELIC
    assert snail.dominant_origin is ValueOrigin.EPISTEMIC

    engine.learn("snail", experienced_liking=0.5, extrinsic_payoff=0.9)
    status = engine.overjustification.status()
    assert status["protected_payoffs"] == 1
    assert status["incentives"] == 1


def test_an_instrumental_motive_is_not_protected(engine):
    """Refutes: the guard fires on everything and learns nothing.

    An instrumental motive's payoff belongs in its cached value, because the
    payoff is the point.
    """
    engine.appraise(Incentive(key="chore", cached_value=0.4, goal_value=0.8))
    engine.learn("chore", experienced_liking=0.5, extrinsic_payoff=0.9)
    assert engine.overjustification.status()["protected_payoffs"] == 0


def test_erosion_needs_a_first_payoff_to_compare_against(engine):
    """Refutes: erosion can be reported before anything was paid."""
    from core.conation.overjustification import ContaminationRecord

    record = ContaminationRecord(key="x", origin="epistemic")
    assert record.erosion() is None


def test_conation_comes_up_in_the_boot_activator_table():
    """Refutes: the organ registers its checks on first use.

    An organ that registers lazily is unchecked exactly during the boot it is
    most likely to be wrong in, and its telemetry channels do not exist when
    the first sample is written.
    """
    from core.runtime.foundations import _ACTIVATORS

    assert "conation" in {name for name, _ in _ACTIVATORS}


def test_information_gain_accepts_a_discrete_belief(engine):
    """Refutes: only a Gaussian model may supply expected information gain."""
    from core.conation.epistemic import categorical_kl

    assert categorical_kl((0.9, 0.1), (0.5, 0.5)) > 0.0
    assert categorical_kl((0.5, 0.5), (0.5, 0.5)) == pytest.approx(0.0, abs=1e-9)

    for error in (0.9, 0.6, 0.35, 0.15):
        engine.epistemic.observe_error("discrete", error)
    reading = engine.epistemic.value(
        "discrete", epistemic_affordance=0.9,
        prior_belief=(0.25, 0.25, 0.25, 0.25),
        posterior_belief=(0.85, 0.05, 0.05, 0.05),
    )
    assert reading.available
    assert "nats" in reading.evidence


def test_a_mismatched_belief_support_omits_the_term_rather_than_guessing(engine):
    """Refutes: a prior and posterior of different support are two beliefs
    about one question."""
    from core.conation.epistemic import categorical_kl

    with pytest.raises(ValueError):
        categorical_kl((0.5, 0.5), (0.3, 0.3, 0.4))


def test_the_default_calibration_declares_itself_unmeasured(engine):
    """Refutes: an engine built with no arguments looks calibrated."""
    assert engine.salience.calibration.learned is False
    assert engine.salience.calibration.source == "declared_default"
