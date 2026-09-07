"""Fourteen dispositions, each held to the claim its module makes.

The tests are arranged around one question per module: what would be true if
the mechanism were doing nothing, and does the code distinguish that case? A
module whose removal changes no measurement is a file, and CONTRIBUTING.md is
explicit that finding that out is the point of the exercise rather than an
embarrassment. Where the answer is an ablation, it is run here rather than
described.

Nothing in this file exercises the live runtime. The wiring, the container
registrations and the snapshot are covered separately in
``tests/test_phenomena_wiring.py``.
"""

from __future__ import annotations

import math
import random

import pytest

# ---------------------------------------------------------------------------
# 1. An identity that is its practices


def test_practices_that_travel_together_cohere_and_scattered_ones_do_not():
    """The same practices, the same number of enactments, different coupling."""
    from core.identity.constitutive_identity import ConstitutiveIdentity

    day = 86400.0
    names = ["tend", "make", "adorn", "gather", "sing"]
    rng = random.Random(7)

    together = ConstitutiveIdentity("together")
    for week in range(40):
        when = week * day
        for name in names:
            together.enact(name, at=when + rng.uniform(0, 600))

    apart = ConstitutiveIdentity("apart")
    for week in range(40):
        for index, name in enumerate(names):
            apart.enact(name, at=week * day + index * 4 * 3600 + rng.uniform(0, 600))

    now = 40 * day + 10
    coherent = together.coherence(at=now, record=False)
    scattered = apart.coherence(at=now, record=False)

    assert coherent.coherent, "practices enacted together did not cohere"
    assert not scattered.coherent, "practices never enacted together cohered anyway"
    assert scattered.r < scattered.incoherent_floor


def test_a_coherence_below_its_own_null_is_reported_as_absent():
    """Unrelated phases give a nonzero order parameter. That is the null."""
    from core.identity.constitutive_identity import incoherent_floor

    # Expected order parameter of a two-dimensional random walk of n unit steps.
    for n in (4, 9, 16, 64):
        assert incoherent_floor(n) == pytest.approx(math.sqrt(math.pi / n) / 2)
    assert incoherent_floor(4) > 0.4, "four practices already give a large null"


def test_declaring_an_identity_does_not_establish_it():
    """The load-bearing claim. A label may be recorded and never counted."""
    from core.identity.constitutive_identity import ConstitutiveIdentity

    identity = ConstitutiveIdentity("claimed")
    identity.enact("make", at=0.0)
    identity.enact("make", at=100.0)
    before = identity.coherence(at=200.0, record=False)
    for _ in range(50):
        identity.declare("a maker", source="self-report", at=150.0)
    after = identity.coherence(at=200.0, record=False)
    assert after.r == pytest.approx(before.r)
    assert after.n_active == before.n_active


def test_a_label_with_no_practice_behind_it_is_named():
    from core.identity.constitutive_identity import ConstitutiveIdentity

    identity = ConstitutiveIdentity("claimed")
    identity.declare("a maker")
    assert identity.unsupported_declarations(at=0.0) == ["a maker"]
    identity.enact("make", at=0.0)
    assert identity.unsupported_declarations(at=10.0) == []


# ---------------------------------------------------------------------------
# 2. Doing that has no finish line


def test_expression_forgets_where_it_started():
    """A limit cycle restores its own amplitude. A wobble keeps what it was given."""
    from core.embodiment.expressive_dynamics import is_limit_cycle

    converged, amplitudes = is_limit_cycle(
        starts=((0.05, 0.0), (3.0, 0.0), (0.5, 2.0)), cycles=60
    )
    assert converged, f"orbits did not converge: {amplitudes}"
    assert max(amplitudes) - min(amplitudes) < 0.1


def test_the_locking_band_widens_with_the_drive_and_is_empty_without_one():
    """The Arnold tongue, measured rather than asserted."""
    from core.embodiment.expressive_dynamics import entrainment_band

    widths = []
    for amplitude in (0.0, 0.15, 0.4, 1.0):
        low, high, _ = entrainment_band(
            natural_period_s=1.0, drive_amplitude=amplitude, samples=25, cycles=60
        )
        widths.append(high - low)
    assert widths[0] == 0.0, "locked to a driver that was not there"
    assert widths == sorted(widths), f"band did not widen with drive: {widths}"
    assert widths[-1] > 0.3


def test_locking_value_alone_would_report_a_lock_that_is_not_there():
    """Why the criterion is locking value together with bounded slip.

    Two rhythms a hair apart hold a nearly constant phase difference over any
    window short against the beat between them. Without the slip term the
    undriven case scores as locked.
    """
    from core.embodiment.expressive_dynamics import (
        ExpressiveOscillator,
        sinusoidal_driver,
    )

    force, phase = sinusoidal_driver(1.0 / 0.97, 0.0)
    oscillator = ExpressiveOscillator(period_s=1.0)
    oscillator.run(20.0, driver=force, driver_phase=phase)
    orbit = oscillator.run(120.0, driver=force, driver_phase=phase)
    assert not orbit.locked
    assert orbit.drift_cycles >= 1.0, "a free oscillator did not drift against a phantom"


def test_an_orbit_has_no_completion():
    from core.embodiment.expressive_dynamics import ExpressiveLedger, ExpressiveOscillator

    ledger = ExpressiveLedger()
    ledger.record(ExpressiveOscillator(period_s=0.5).run(4.0))
    assert ledger.completion() is None
    assert ledger.status()["completion"] is None
    assert ledger.time_on_cycle_s > 0


# ---------------------------------------------------------------------------
# 3. Care under a floor


def test_the_floor_cannot_be_bought_at_any_need():
    """A weighted floor fails at some finite need. A constraint fails at none."""
    from core.ethics.care_allocation import CareAllocator

    allocator = CareAllocator(priority=1.0, self_floor=3.0)
    for need in (10.0, 1e3, 1e6, 1e12):
        allocation = allocator.allocate(10.0, needs={"one": need}, record=False)
        assert allocation.spent <= 7.0 + 1e-9, f"floor sold at need {need}"


def test_a_budget_below_the_floor_refuses_and_says_what_for():
    from core.ethics.care_allocation import CareAllocator

    allocator = CareAllocator(self_floor=5.0)
    allocation = allocator.allocate(4.0, needs={"someone": 100.0}, record=False)
    assert allocation.given == {}
    assert allocation.refused_for_floor == pytest.approx(100.0)


def test_care_that_never_lands_stops_being_poured_and_is_named():
    from core.ethics.care_allocation import CareAllocator

    allocator = CareAllocator(priority=1.0)
    allocator.set_need("reached", 5.0)
    allocator.set_need("unreached", 5.0)
    for _ in range(20):
        allocator.observe_reception("reached", True)
        allocator.observe_reception("unreached", False)
    assert allocator.recipient("unreached").unreachable
    assert not allocator.recipient("reached").unreachable


def test_priority_changes_the_shape_and_not_only_the_total():
    from core.ethics.care_allocation import CareAllocator

    needs = {"a": 10.0, "b": 6.0, "c": 3.0, "d": 1.0}
    ginis = []
    for priority in (0.0, 1.0, 3.0):
        allocator = CareAllocator(priority=priority, self_floor=2.0)
        allocation = allocator.allocate(12.0, needs=needs, record=False)
        assert allocation.spent == pytest.approx(10.0, abs=1e-6)
        ginis.append(allocation.gini)
    assert ginis == sorted(ginis), f"priority did not concentrate care: {ginis}"


def test_inequality_counts_the_people_who_got_nothing():
    """Measured over the funded only, denying someone reads as more equality."""
    from core.ethics.care_allocation import gini

    assert gini([4.0, 3.0, 2.0, 1.0]) < gini([4.0, 3.0, 2.0, 0.0])
    assert gini([1.0, 1.0, 1.0]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 4. Accepting what is offered


def test_trust_breaks_faster_than_it_builds_without_that_being_set():
    """The asymmetry comes out of the two likelihoods, not out of a constant."""
    from core.social.receptivity import Receptivity

    receptivity = Receptivity()
    for _ in range(6):
        receptivity.observe("them", True)
        built = receptivity.regard("them").posterior()
    after_one_lapse = receptivity.observe("them", False)
    # One unkindness undoes several kindnesses. The exact number follows from
    # the likelihood ratio and is checked as a range rather than a value so
    # the test survives a principled revision of either likelihood.
    assert after_one_lapse < built
    assert 2.0 < (built - 0.5) / max(after_one_lapse - 0.5, 1e-9) < 8.0


def test_a_costly_offer_moves_the_posterior_further_than_a_free_one():
    from core.social.receptivity import Receptivity

    free = Receptivity()
    costly = Receptivity()
    for _ in range(3):
        free.observe("a", True, cost_to_source=0.0)
        costly.observe("b", True, cost_to_source=1.5)
    assert costly.regard("b").posterior() > free.regard("a").posterior()


def test_something_is_taken_only_for_what_it_would_settle():
    """The term a myopic rule does not have, and the horizon that turns it off."""
    from core.social.receptivity import Offer, Receptivity

    receptivity = Receptivity(horizon=5)
    offer = Offer(source="stranger", value=4.0, exposure=5.0)
    considered = receptivity.consider(offer)
    assert considered.expected_value < 0
    assert considered.accepted and considered.learning_led
    myopic = receptivity.consider(offer, horizon=0)
    assert not myopic.accepted


def test_the_value_of_learning_falls_away_as_the_question_settles():
    from core.social.receptivity import Offer, Receptivity

    receptivity = Receptivity()
    values = []
    for index in range(6):
        decision = receptivity.receive(
            Offer(source="t", value=4.0, exposure=5.0, label=f"o{index}")
        )
        values.append(decision.value_of_learning)
        receptivity.observe("t", True)
    assert values[-1] < values[0], f"learning value did not decay: {values}"


def test_refusing_everything_is_only_a_fault_when_it_is_why_nothing_is_known():
    from core.social.receptivity import Offer, Receptivity

    closed = Receptivity()
    for index in range(6):
        closed.receive(Offer(source="u", value=1.0, exposure=500.0, label=f"o{index}"))
    assert closed.isolation()["closed"]

    open_enough = Receptivity()
    for index in range(6):
        open_enough.receive(Offer(source="v", value=10.0, exposure=1.0, label=f"o{index}"))
    assert not open_enough.isolation()["closed"]


# ---------------------------------------------------------------------------
# 5. Markers that mean something only because we agree


def test_the_same_marker_settles_either_way_from_where_it_started():
    from core.social.conventions import ConventionRegistry

    now = ConventionRegistry()
    now.declare("pink", ("for girls", "for boys"), frequency=0.85)
    then = ConventionRegistry()
    then.declare("pink", ("for girls", "for boys"), frequency=0.20)
    assert now.get("pink").current_meaning() == "for girls"
    assert then.get("pink").current_meaning() == "for boys"
    assert now.settles_at("pink") == pytest.approx(1.0, abs=1e-3)
    assert then.settles_at("pink") == pytest.approx(0.0, abs=1e-3)


def test_the_alternative_meaning_is_a_value_and_not_a_caveat():
    from core.social.conventions import ConventionRegistry

    registry = ConventionRegistry()
    registry.declare("pink", ("for girls", "for boys"), frequency=0.85)
    marker = registry.get("pink")
    assert marker.arbitrary
    assert marker.counterfactual_meaning() == "for boys"


def test_a_minority_has_a_size_it_has_to_reach():
    from core.social.conventions import tipping_point

    assert tipping_point(coordination=1.0) == pytest.approx(0.5)
    # A marker the population has its own reason to prefer needs a smaller
    # minority to flip toward it, and past a large enough reason the interior
    # rest leaves the interval and no minority can win.
    assert tipping_point(coordination=1.0, bias=0.6) == pytest.approx(0.2)
    assert tipping_point(coordination=1.0, bias=2.5) is None


def test_a_marker_can_be_worth_using_against_how_it_is_read():
    from core.social.conventions import ConventionRegistry

    registry = ConventionRegistry()
    registry.declare("pink", ("for girls", "for boys"), frequency=0.2)
    against = registry.adopt("pink", expressive_value=0.9, cost=0.1)
    assert against.use and against.expressive_value > against.coordination_value
    hollow = registry.adopt("pink", expressive_value=0.0, cost=0.5)
    assert not hollow.use


# ---------------------------------------------------------------------------
# 6. A feeling against an argument


def _train_two_domains(arbiter, rounds: int = 60, seed: int = 11):
    """One domain turning on a hidden variable, one on a checkable fact."""
    rng = random.Random(seed)
    for index in range(rounds):
        hidden = rng.random()
        arbiter.arbitrate(
            "social",
            min(max(hidden + rng.gauss(0, 0.12), 0), 1),
            min(max(0.5 + rng.gauss(0, 0.30), 0), 1),
            key=f"s{index}",
        )
        arbiter.resolve(f"s{index}", hidden > 0.5)

        fact = rng.random() > 0.5
        arbiter.arbitrate(
            "checkable",
            min(max(0.5 + rng.gauss(0, 0.28), 0), 1),
            0.93 if fact else 0.07,
            key=f"c{index}",
        )
        arbiter.resolve(f"c{index}", fact)


def test_with_no_resolved_outcomes_it_abstains_rather_than_defaulting():
    from core.affect.dual_process_arbiter import DualProcessArbiter

    result = DualProcessArbiter().arbitrate("untested", 0.9, 0.1, record=False)
    assert result.abstained and result.probability is None
    assert result.weight_affective == result.weight_deliberate


def test_which_channel_leads_is_learned_and_differs_by_domain():
    from core.affect.dual_process_arbiter import AFFECTIVE, DELIBERATE, DualProcessArbiter

    arbiter = DualProcessArbiter()
    _train_two_domains(arbiter)
    profile = arbiter.profile()
    assert profile["social"]["leads"] == AFFECTIVE
    assert profile["checkable"]["leads"] == DELIBERATE

    # The same disagreement, resolved opposite ways.
    social = arbiter.arbitrate("social", 0.85, 0.20, record=False)
    checkable = arbiter.arbitrate("checkable", 0.85, 0.20, record=False)
    assert social.probability > checkable.probability


def test_forcing_a_channel_moves_the_answer_and_nothing_else_does():
    from core.affect.dual_process_arbiter import AFFECTIVE, DELIBERATE, DualProcessArbiter

    arbiter = DualProcessArbiter()
    _train_two_domains(arbiter)
    arbiter.force(DELIBERATE)
    forced = arbiter.arbitrate("social", 0.85, 0.20, record=False)
    assert forced.probability == pytest.approx(0.20)
    arbiter.force(AFFECTIVE)
    assert arbiter.arbitrate("social", 0.85, 0.20, record=False).probability == pytest.approx(0.85)
    arbiter.force(None)
    assert arbiter.arbitrate("social", 0.85, 0.20, record=False).probability > 0.5


def test_the_outweighed_channel_is_still_scored():
    """Otherwise an early bad stretch is unrecoverable and self-confirming."""
    from core.affect.dual_process_arbiter import AFFECTIVE, DELIBERATE, DualProcessArbiter

    arbiter = DualProcessArbiter()
    _train_two_domains(arbiter, rounds=20)
    assert arbiter.calibration(AFFECTIVE, "checkable").n == 20
    assert arbiter.calibration(DELIBERATE, "social").n == 20


# ---------------------------------------------------------------------------
# 7. Where to be


def test_refuge_cannot_come_from_the_sightline_graph():
    """The correction the module is built around, held in place by a test."""
    from core.environment.prospect_refuge import grid_field

    bare = grid_field(7, 5, walls=[(2, 0), (2, 1), (0, 2), (1, 2), (2, 2)])
    assert all(p.asymmetry == pytest.approx(0.0) for p in bare.score())
    assert not bare.status()["refuge_modelled"]

    covered = grid_field(
        7, 5, walls=[(2, 0), (2, 1), (0, 2), (1, 2), (2, 2)],
        cover={(3, 0): 0.8, (3, 1): 0.7},
    )
    scored = {p.key: p for p in covered.score()}
    assert scored["3,0"].asymmetry > 0.1
    assert covered.status()["refuge_modelled"]


def test_the_two_terms_are_never_weighted_for_the_caller():
    from core.environment.prospect_refuge import grid_field

    # Walls are what make prospect vary. An open room gives every cell full
    # prospect, correctly, so a ranking over one is a tie broken arbitrarily.
    field_obj = grid_field(
        9, 7, walls=[(4, y) for y in range(5)], cover={(0, 0): 0.95, (8, 6): 0.95}
    )
    with pytest.raises(TypeError):
        field_obj.rank()  # type: ignore[call-arg]
    by_prospect = [p.key for p, _ in field_obj.rank(prospect_weight=1.0, refuge_weight=0.0)]
    by_refuge = [p.key for p, _ in field_obj.rank(prospect_weight=0.0, refuge_weight=1.0)]
    assert by_prospect != by_refuge, "the weights made no difference to the order"
    assert by_refuge[0] in {"0,0", "8,6"}


def test_fit_counts_what_is_met_and_not_what_is_spare():
    from core.environment.prospect_refuge import graph_field

    field_obj = graph_field(
        [("a", "b"), ("b", "c")],
        demands={"b": {"patience": 1.0, "reach": 1.0}},
    )
    scored = {p.key: p for p in field_obj.score({"patience": 1.0, "reach": 1.0})}
    assert scored["b"].fit == pytest.approx(1.0)
    spare = {p.key: p for p in field_obj.score({"patience": 9.0, "reach": 9.0})}
    assert spare["b"].fit == pytest.approx(1.0)
    short = {p.key: p for p in field_obj.score({"patience": 1.0})}
    assert short["b"].fit == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 8. Getting better at something that pushes back


def _bowl(optimum, resistance_scale, rng, noise=0.01):
    def attempt(parameters):
        squared = sum((p - o) ** 2 for p, o in zip(parameters, optimum, strict=True))
        return math.exp(-squared) + rng.gauss(0, noise), resistance_scale * math.sqrt(squared)
    return attempt


def test_practice_finds_the_shape_it_cannot_see():
    from core.learning.craft_practice import CraftPractice

    rng = random.Random(5)
    practice = CraftPractice(seed=3)
    practice.add_skill("throwing", [0.0, 0.0, 0.0], step_size=0.6, perturbation=0.25)
    material = _bowl([0.8, -0.5, 0.3], 1.2, rng)
    for _ in range(60):
        practice.practise("throwing", material)
    found = practice._skills["throwing"].parameters
    for got, want in zip(found, [0.8, -0.5, 0.3], strict=True):
        assert abs(got - want) < 0.1, f"did not converge: {found}"


def test_the_estimate_of_the_material_follows_the_material():
    from core.learning.craft_practice import CraftPractice

    rng = random.Random(5)
    practice = CraftPractice(seed=3)
    practice.add_skill("stubborn", [0.0, 0.0], step_size=0.4, perturbation=0.2)
    practice.add_skill("forgiving", [0.0, 0.0], step_size=0.4, perturbation=0.2)
    for _ in range(30):
        practice.practise("stubborn", _bowl([1.0, 1.0], 2.0, rng))
        practice.practise("forgiving", _bowl([1.0, 1.0], 0.05, rng))
    assert (
        practice._skills["stubborn"].resistance_estimate
        > practice._skills["forgiving"].resistance_estimate * 3
    )


def test_practice_carries_on_after_the_task_stops_asking():
    """The ablation between the two schedulers, run rather than described."""
    from core.learning.craft_practice import CraftPractice

    rng = random.Random(5)
    practice = CraftPractice(seed=3)
    practice.add_skill("throwing", [0.0, 0.0, 0.0], step_size=0.35, perturbation=0.2)
    practice.set_sufficiency("throwing", 0.35)
    material = _bowl([1.2, -0.9, 0.6], 0.6, rng)

    quality_when_sufficient = None
    for _ in range(30):
        practice.practise("throwing", material)
        if practice.required_target() is None and quality_when_sufficient is None:
            quality_when_sufficient = practice._skills["throwing"].competence()

    assert quality_when_sufficient is not None, "the bar was never met"
    assert practice.required_target() is None
    assert practice.practice_target() == "throwing"
    assert "throwing" in practice.past_sufficiency()
    final = practice._skills["throwing"].competence()
    assert final > quality_when_sufficient + 0.2, (
        f"nothing was gained past sufficiency: {quality_when_sufficient} -> {final}"
    )


# ---------------------------------------------------------------------------
# 9. New and still legible


def _tradition():
    motifs = [b"CDEFG ", b"GFEDC ", b"CEGEC ", b"DFAFD "]
    def piece(order, length=8):
        return b"".join(motifs[i % len(motifs)] for i in order[:length])
    return piece


def test_a_copy_and_a_noise_both_lose_to_a_recombination():
    from core.creativity.novelty_value import NoveltyValuer

    piece = _tradition()
    valuer = NoveltyValuer()
    corpus = [
        piece([0, 1, 0, 1, 2, 0, 1, 2]),
        piece([2, 3, 2, 3, 0, 2, 3, 0]),
        piece([1, 2, 1, 2, 3, 1, 2, 3]),
    ]
    for index, payload in enumerate(corpus):
        valuer.absorb(f"t{index}", payload)

    rng = random.Random(4)
    # Drawn from one generator. Constructing Random(seed) inside the
    # comprehension rebuilds it per byte and produces the same byte every
    # time, which is a constant rather than a noise control.
    noise_bytes = bytes([rng.getrandbits(8) for _ in range(len(corpus[0]))])
    copy = valuer.value("copy", corpus[0])
    recombination = valuer.value("recombination", piece([3, 0, 2, 1, 0, 3, 1, 2]))
    noise = valuer.value("noise", noise_bytes)

    assert recombination.value > copy.value
    assert recombination.value > noise.value
    assert copy.novelty < 0.2 and copy.intelligibility > 0.7
    assert noise.novelty > 0.7 and noise.intelligibility < 0.3


def test_the_same_move_is_worth_less_the_second_time():
    from core.creativity.novelty_value import NoveltyValuer

    piece = _tradition()
    valuer = NoveltyValuer()
    for index in range(3):
        valuer.absorb(f"t{index}", piece([index, index + 1, index, index + 1]))
    again = piece([3, 0, 2, 1, 0, 3, 1, 2])
    first = valuer.offer("a", again).value
    second = valuer.offer("b", again).value
    assert second < first * 0.6, f"repeating was not punished: {first} -> {second}"


def test_distance_to_itself_is_nothing():
    from core.creativity.novelty_value import normalised_compression_distance as ncd

    payload = b"ABCABDABCABE" * 16
    rng = random.Random(1)
    noise = bytes([rng.getrandbits(8) for _ in range(192)])
    assert ncd(payload, payload) < 0.1
    assert ncd(payload, noise) > 0.7


# ---------------------------------------------------------------------------
# 10. Paying to keep an option open


def _spider_options():
    from core.morality.reversible_alternative import Option

    return [
        Option("kill", cost_to_actor=0.1, harm_to_subject=10.0, reversibility=0.0),
        Option("relocate", cost_to_actor=2.0, harm_to_subject=1.0, reversibility=0.9),
        Option("ignore", cost_to_actor=0.0, harm_to_subject=0.0, reversibility=1.0,
               effectiveness=0.0),
    ]


def test_the_gentle_option_wins_when_it_is_worth_what_it_costs():
    from core.morality.reversible_alternative import Situation, choose

    choice = choose(
        _spider_options(),
        Situation(subject="spider", patienthood=0.3, revision_probability=0.3),
    )
    assert choice.chosen is not None and choice.chosen.option.name == "relocate"
    assert choice.premium_paid <= choice.premium_justified


def test_the_gentle_option_loses_when_it_is_not():
    from core.morality.reversible_alternative import Situation, choose

    choice = choose(
        _spider_options(),
        Situation(subject="spider", patienthood=0.01, revision_probability=0.1),
    )
    assert choice.chosen is not None and choice.chosen.option.name == "kill"


def test_an_option_that_solves_nothing_is_excluded_rather_than_penalised():
    from core.morality.reversible_alternative import Situation, choose

    choice = choose(
        _spider_options()[2:],
        Situation(subject="spider", patienthood=0.9, revision_probability=0.9),
    )
    assert choice.chosen is None
    assert "solves enough" in choice.reason


def test_reversibility_is_worth_nothing_when_nothing_will_be_revised():
    from core.morality.reversible_alternative import Option, Situation, choose

    options = [
        Option("drop", cost_to_actor=0.05, harm_to_subject=100.0, reversibility=0.0),
        Option("rename", cost_to_actor=0.5, harm_to_subject=100.0, reversibility=0.95),
    ]
    might = choose(options, Situation("table", patienthood=1.0, revision_probability=0.15))
    never = choose(options, Situation("table", patienthood=1.0, revision_probability=0.0))
    assert might.chosen.option.name == "rename"
    assert never.chosen.option.name == "drop"


def test_the_answer_says_how_far_the_estimate_can_move():
    from core.morality.reversible_alternative import Situation, choose, sensitivity

    options = _spider_options()
    situation = Situation(subject="spider", patienthood=0.3, revision_probability=0.3)
    report = sensitivity(options, situation)
    assert report["chosen"] == choose(options, situation).chosen.option.name
    assert report["switches_at"] is not None
    low, high = report["holds_over"]
    assert low > 0.0 and high == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 11. Effort that is informative because it is wasted


def test_effort_rises_with_type_and_the_receiver_can_invert_it():
    from core.social.costly_signaling import SignalChannel

    channel = SignalChannel(benefit=2.0, min_type=1.0, cost_slope=1.0)
    efforts = []
    for quality in (1.0, 1.5, 2.0, 3.0):
        signal = channel.send("giver", quality)
        efforts.append(signal.effort)
        assert channel.receive(signal).implied_type == pytest.approx(quality)
    assert efforts == sorted(efforts)


def test_a_free_signal_carries_nothing_and_says_so():
    """The ablation. Take the cost away and the channel dies."""
    from core.social.costly_signaling import SignalChannel, pooling_check

    free = SignalChannel(benefit=2.0, cost_slope=0.0)
    check = pooling_check(free.schedule, [1.0, 1.5, 2.0, 3.0])
    assert not check["separating"] and check["effort_spread"] == 0.0
    reading = free.receive(free.send("giver", 3.0))
    assert reading.implied_type is None and not reading.informative
    assert free.status()["channel_dead"]


def test_effort_that_tracks_the_contents_is_a_description():
    from core.social.costly_signaling import Signal, SignalChannel

    channel = SignalChannel(benefit=2.0, cost_slope=1.0)
    reading = channel.receive(
        Signal(sender="g", effort=5.0, content_value=5.0, content_independence=False)
    )
    assert reading.implied_type is None and not reading.informative


def test_someone_who_cannot_afford_the_effort_is_read_as_less():
    from core.social.costly_signaling import SignalChannel

    channel = SignalChannel(benefit=2.0, cost_slope=1.0)
    constrained = channel.worth_sending(3.0, budget=2.0)
    assert constrained["understated"]
    assert constrained["read_as"] < 3.0


# ---------------------------------------------------------------------------
# 12. Returning in kind


def test_generous_repayment_beats_strict_repayment_once_mistakes_exist():
    from core.social.reciprocity_engine import compare

    clean = compare(error_rate=0.0, rounds=2000, seed=1)
    assert clean["gain"] == pytest.approx(0.0, abs=1e-9)
    for error_rate in (0.01, 0.05, 0.15):
        noisy = compare(error_rate=error_rate, rounds=4000, seed=1)
        assert noisy["gain"] > 0.3, f"no gain at error rate {error_rate}: {noisy}"
        assert noisy["strict"]["mutual_defections"] > noisy["generous"]["mutual_defections"]


def test_the_forgiveness_rate_is_the_largest_that_is_not_exploitable():
    from core.social.reciprocity_engine import generosity

    assert generosity(3.0, 1.0) == pytest.approx(0.5)
    assert generosity(1.0, 1.0) == pytest.approx(0.0)
    assert generosity(0.0, 1.0) == pytest.approx(0.0)


def test_the_rule_holds_or_does_not_by_the_relationship_and_not_the_person():
    from core.social.reciprocity_engine import (
        ALWAYS_DEFECT,
        GENEROUS,
        TIT_FOR_TAT,
        ReciprocityEngine,
    )

    day = 86400.0

    cheap = ReciprocityEngine()
    for index in range(12):
        cheap.record_exchange("them", they_cooperated=True, we_cooperated=True,
                              benefit_received=3.0, cost_borne=1.0, at=index * day)
    assert cheap.stance("them", at=12 * day).strategy == TIT_FOR_TAT

    costly = ReciprocityEngine()
    for index in range(12):
        costly.record_exchange("them", they_cooperated=True, we_cooperated=True,
                               benefit_received=1.0, cost_borne=4.0, at=index * day)
    assert costly.stance("them", at=12 * day).strategy == ALWAYS_DEFECT

    lossy = ReciprocityEngine()
    for index in range(12):
        lossy.record_exchange("them", they_cooperated=True, we_cooperated=True,
                              benefit_received=3.0, cost_borne=1.0,
                              intended_cooperation=(index != 4), at=index * day)
    assert lossy.stance("them", at=12 * day).strategy == GENEROUS


def test_too_little_history_gets_no_verdict():
    from core.social.reciprocity_engine import ReciprocityEngine

    stance = ReciprocityEngine().stance("stranger")
    assert stance.cooperation_stable is None and stance.threshold is None


# ---------------------------------------------------------------------------
# 13. Being moved without becoming them


def _carer_field(anchor: float):
    from core.affect.empathic_coupling import EmpathicField

    field_obj = EmpathicField()
    field_obj.add_person("her", setpoint=0.0, anchor=anchor)
    for who, state in (("a", -3.0), ("b", -2.5), ("c", -4.0)):
        field_obj.add_person(who, setpoint=state, anchor=2.0)
        field_obj.couple("her", who, 1.0)
        field_obj.couple(who, "her", 0.2)
    return field_obj


def test_the_anchor_decides_how_much_of_her_is_left():
    autonomies = [_carer_field(anchor).autonomy("her") for anchor in (0.05, 0.3, 1.0, 4.0)]
    assert autonomies == sorted(autonomies), f"anchor did not hold: {autonomies}"
    assert autonomies[0] < 0.05 and autonomies[-1] > 0.5


def test_the_account_of_a_rest_state_adds_up():
    shares = _carer_field(0.5).attribution("her")
    assert sum(shares.values()) == pytest.approx(1.0)
    assert 0.0 < shares["her"] < 1.0


def test_a_field_nobody_is_anchored_in_has_no_rest_to_report():
    from core.affect.empathic_coupling import EmpathicField

    field_obj = EmpathicField()
    for who in ("a", "b", "c"):
        field_obj.add_person(who, setpoint=1.0, anchor=0.0)
    for one in ("a", "b", "c"):
        for other in ("a", "b", "c"):
            if one != other:
                field_obj.couple(one, other, 1.0)
    assert field_obj.rest() is None
    assert field_obj.status()["no_rest"]


def test_coupling_is_allowed_to_be_one_sided():
    from core.affect.empathic_coupling import EmpathicField

    field_obj = EmpathicField()
    field_obj.add_person("x", setpoint=0.0, anchor=1.0)
    field_obj.add_person("y", setpoint=5.0, anchor=1.0)
    field_obj.couple("x", "y", 3.0)
    field_obj.couple("y", "x", 0.1)
    rest = field_obj.rest()
    assert rest["x"] > 3.0, "the strongly coupled one was not moved"
    assert rest["y"] > 4.5, "the weakly coupled one was moved as much"


# ---------------------------------------------------------------------------
# 14. Finding something beautiful, and stopping


def test_the_hedonic_curve_is_a_hump():
    from core.perception.aesthetic_response import berlyne_curve

    values = [p for _, p in berlyne_curve([i / 20 for i in range(21)])]
    peak = values.index(max(values))
    assert 0 < peak < len(values) - 1, "the curve is monotone, so the midpoints are wrong"
    assert values[0] < values[peak] and values[-1] < values[peak]


def test_the_two_measures_disagree_hardest_on_a_blank_page():
    from core.perception.aesthetic_response import AestheticObserver

    observer = AestheticObserver()
    blank = observer.consider("blank", b" " * 256)
    patterned = observer.consider(
        "patterned",
        b"".join(bytes([65 + (i * i) % 7, 65 + (i % 5), 32]) for i in range(85)),
    )
    assert blank.birkhoff > patterned.birkhoff
    assert blank.eysenck < patterned.eysenck


def test_noise_scores_nothing_on_either_measure():
    from core.perception.aesthetic_response import AestheticObserver

    rng = random.Random(3)
    noise = bytes([rng.getrandbits(8) for _ in range(256)])
    response = AestheticObserver().consider("noise", noise)
    assert response.order < 0.1
    assert response.birkhoff < 0.2 and response.eysenck < 0.2


def test_habituation_is_the_history_and_nothing_else():
    """Freeze the observer and the decay disappears. The ablation."""
    from core.perception.aesthetic_response import AestheticObserver

    stimulus = b"".join(bytes([65 + (i * i) % 7, 65 + (i % 5), 32]) for i in range(64))

    live = AestheticObserver()
    live.look("context", b"ABCABDABCABF" * 8)
    moving = [r.pleasure for r in live.curve("thing", stimulus, times=8)]

    frozen = AestheticObserver()
    frozen.look("context", b"ABCABDABCABF" * 8)
    frozen.frozen = True
    held = [r.pleasure for r in frozen.curve("thing", stimulus, times=8)]

    assert len(set(round(p, 6) for p in held)) == 1, "a frozen observer habituated"
    assert moving[-1] != pytest.approx(moving[0]), "a live observer did not"


def test_the_same_object_scores_differently_to_two_observers():
    from core.perception.aesthetic_response import AestheticObserver

    stimulus = b"ABCABDABCABE" * 16
    fresh = AestheticObserver()
    fresh.look("unrelated", b"zzzzzzzzzzzz" * 8)
    steeped = AestheticObserver()
    for _ in range(4):
        steeped.look("family", b"ABCABDABCABF" * 16)
    assert fresh.consider("x", stimulus).novelty > steeped.consider("x", stimulus).novelty


# ---------------------------------------------------------------------------
# 15. Wanting company against being able to sustain it


def test_the_duty_cycle_is_derived_from_the_two_rates():
    from core.social.social_stamina import SocialStamina, sustainable_hours_per_week

    assert sustainable_hours_per_week(1 / 6, 1 / 12) == pytest.approx(56.0)
    assert sustainable_hours_per_week(1 / 12, 1 / 6) == pytest.approx(112.0)
    stamina = SocialStamina(drain_per_s=2.0, recovery_per_s=1.0)
    assert stamina.sustainable_share() == pytest.approx(1 / 3)


def test_where_belonging_settles_follows_the_share_of_time_in_company():
    """One time constant in both directions, or more company means lonelier."""
    from core.social.social_stamina import SocialStamina

    hour = 3600.0
    settled = []
    for hours in (1, 4, 8, 12, 18):
        stamina = SocialStamina(
            drain_per_s=1 / (40 * hour), recovery_per_s=1 / (8 * hour),
            belonging_per_s=1 / (2 * 86400.0),
        )
        moment = 0.0
        for _ in range(30):
            stamina.spend(hours * hour, at=moment)
            moment += hours * hour
            stamina.rest((24 - hours) * hour, at=moment)
            moment += (24 - hours) * hour
        settled.append(stamina.belonging)
    assert settled == sorted(settled, reverse=True), f"more company, not less need: {settled}"


def test_exhaustion_costs_more_than_the_deficit_suggests():
    from core.social.social_stamina import SocialStamina

    hour = 3600.0
    emptied = SocialStamina(drain_per_s=1 / (2 * hour), recovery_per_s=1 / (8 * hour))
    emptied.spend(2 * hour)
    nearly = SocialStamina(drain_per_s=1 / (2 * hour), recovery_per_s=1 / (8 * hour))
    nearly.spend(1.9 * hour)
    assert emptied.stamina == pytest.approx(0.0)
    # Twenty times the remaining deficit, and more than twice the recovery.
    assert emptied.recovery_time() > 2.0 * nearly.recovery_time()


def test_wanting_company_and_having_nothing_left_is_a_readable_state():
    """The case a single sociability number cannot hold."""
    from core.social.social_stamina import SocialStamina

    hour = 3600.0
    stamina = SocialStamina(
        drain_per_s=1 / (6 * hour), recovery_per_s=1 / (40 * hour),
        belonging_per_s=1 / (4 * 86400.0),
    )
    moment = 0.0
    fired = False
    for _ in range(9):
        stamina.rest(16 * hour, at=moment)
        moment += 16 * hour
        stamina.spend(8 * hour, with_person="the office", at=moment)
        moment += 8 * hour
        fired = fired or stamina.read(at=moment).wants_but_cannot
    reading = stamina.read(at=moment)
    assert fired, "the state never became readable"
    assert reading.overdrawn and reading.exhausted
    assert stamina.overdrawn_for(at=moment) > 0


def test_what_company_costs_is_measured_rather_than_declared():
    from core.social.social_stamina import SocialStamina

    hour = 3600.0
    stamina = SocialStamina(drain_per_s=1 / (4 * hour), recovery_per_s=1 / (8 * hour))
    for _ in range(5):
        stamina.company("a quiet friend").observe(2 * hour, 0.02)
        stamina.company("a loud room").observe(2 * hour, 0.60)
    assert (
        stamina.sustainable_share(with_person="a quiet friend")
        > stamina.sustainable_share(with_person="a loud room")
    )
