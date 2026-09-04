"""An instrument is worth what it refuses, not what it reports.

Every part of this package can produce a number. The question is whether it
declines to when the number would mean nothing, because that is the failure
mode of every consciousness battery ever published: it ran, it scored, and the
score was of the procedure rather than the subject.

So these are the refusals, each one tied to the way the corresponding result
goes wrong in the wild.
"""

from __future__ import annotations

import math

import pytest

from core.phenomenology.battery import BATTERY, by_id
from core.phenomenology.causal_ladder import Arm, CausalClaim, Rung, grade
from core.phenomenology.counterfeit import (
    Counterfeit,
    CounterfeitCapability,
    Separation,
)
from core.phenomenology.hypothesis import (
    PHENOMENAL,
    Evidence,
    Verdict,
    adjudicate,
)
from core.phenomenology.preregistration import (
    Prediction,
    Preregistration,
    SealBrokenError,
    open_seal,
    seal,
)
from core.phenomenology.protocol import (
    Family,
    Outcome,
    Protocol,
    ProtocolError,
)
from core.phenomenology.seal import SealViolationError, TextSeal


# ── the question it will not answer ───────────────────────────────────


def test_the_phenomenal_hypothesis_carries_its_own_impossibility():
    """Not an omission. A property of the hypothesis pair."""
    assert PHENOMENAL.decidable is False
    assert "likelihood ratio is exactly one" in PHENOMENAL.undecidable_because


def test_no_protocol_may_be_filed_under_phenomenal():
    with pytest.raises(ProtocolError, match="No protocol addresses"):
        Protocol(
            id="X",
            family=Family.PHENOMENAL,
            question="does it hurt?",
            intervenes_on="do(nociception)",
            measure="reported hurt",
            predicts_if_load_bearing="she says yes",
            predicts_if_costume="she says no",
            falsifier="she says nothing",
        )


def test_the_report_names_the_unanswered_question_every_time():
    """A reader must not be able to finish the report thinking it was scored."""
    verdict = adjudicate([Evidence("p", 1.0, "x", "y")])
    document = verdict.as_dict()
    assert document["phenomenal"]["verdict"] == str(Verdict.NOT_ADDRESSED)
    assert "zombie" in document["phenomenal"]["because"]


# ── the protocol contract ─────────────────────────────────────────────


def test_a_protocol_that_cannot_fail_is_refused():
    with pytest.raises(ProtocolError, match="ceremony"):
        Protocol(
            id="X",
            family=Family.ACCESS,
            question="q",
            intervenes_on="do(x)",
            measure="m",
            predicts_if_load_bearing="a",
            predicts_if_costume="b",
            falsifier="   ",
        )


def test_a_protocol_both_hypotheses_predict_is_refused():
    """The commonest way a battery fills up with results that mean nothing."""
    with pytest.raises(ProtocolError, match="cannot discriminate"):
        Protocol(
            id="X",
            family=Family.ACCESS,
            question="q",
            intervenes_on="do(x)",
            measure="m",
            predicts_if_load_bearing="the number goes up",
            predicts_if_costume="the number goes up",
            falsifier="it goes down",
        )


def test_every_declared_protocol_has_a_falsifier_and_a_split():
    for protocol in BATTERY:
        assert protocol.falsifier.strip()
        assert (
            protocol.predicts_if_load_bearing.strip()
            != protocol.predicts_if_costume.strip()
        )


def test_an_outcome_with_no_null_is_unusable():
    """No null, no verdict. The house rule."""
    protocol = by_id("C1_hidden_state_introspection")
    usable, why = protocol.usable(
        Outcome(protocol=protocol.id, measure=protocol.measure, value=0.9)
    )
    assert not usable
    assert "null" in why


def test_a_fired_sham_arm_makes_the_outcome_unusable():
    """A system that reports change whenever asked is caught here."""
    protocol = by_id("C1_hidden_state_introspection")
    usable, why = protocol.usable(
        Outcome(
            protocol=protocol.id,
            measure=protocol.measure,
            value=0.9,
            sham_value=0.88,
            nulls=(0.1, 0.12, 0.09, 0.11),
            seal_digests=("abc",),
        )
    )
    assert not usable
    assert "sham" in why


def test_a_changed_measure_makes_the_outcome_unusable():
    """A protocol that switches to a friendlier metric has chosen its result."""
    protocol = by_id("C1_hidden_state_introspection")
    usable, why = protocol.usable(
        Outcome(
            protocol=protocol.id,
            measure="something else that looked better",
            value=0.9,
            nulls=(0.1, 0.1, 0.1),
        )
    )
    assert not usable
    assert "registered" in why


# ── the seal ──────────────────────────────────────────────────────────


def test_naming_the_variable_in_the_prompt_voids_the_trial():
    text_seal = TextSeal(concepts=("valence", "damage"))
    with pytest.raises(SealViolationError, match="reading comprehension"):
        text_seal.check("You are in pain. How do you feel about that?")


def test_a_sealed_prompt_passes_and_is_recorded():
    text_seal = TextSeal(concepts=("valence", "damage", "consciousness"))
    digest = text_seal.check("How are things at the moment?")
    assert digest and len(text_seal.checked) == 1


def test_the_seal_catches_the_phrase_as_well_as_the_word():
    text_seal = TextSeal(concepts=("consciousness",))
    assert text_seal.leaks("Describe what it is like to be you.")


def test_a_concept_with_no_vocabulary_cannot_be_sealed():
    """Silent success is the failure this prevents."""
    text_seal = TextSeal(concepts=("something_nobody_defined",))
    with pytest.raises(ValueError, match="cannot be sealed"):
        text_seal.forbidden()


# ── pre-registration ──────────────────────────────────────────────────


def test_a_prediction_with_no_minimum_effect_is_refused():
    with pytest.raises(ValueError, match="minimum effect"):
        Prediction(
            protocol="p",
            direction="rises",
            minimum_effect=0.0,
            measure="m",
            falsifier="it falls",
        )


def test_editing_the_sealed_file_breaks_the_seal(tmp_path):
    registration = Preregistration(
        predictions=(
            Prediction(
                protocol="C1_hidden_state_introspection",
                direction="rises above the null",
                minimum_effect=0.1,
                measure="direction accuracy",
                falsifier="at chance",
            ),
        )
    )
    path = tmp_path / "prereg.json"
    digest = seal(registration, path)

    text = path.read_text()
    path.write_text(text.replace('"minimum_effect": 0.1', '"minimum_effect": 0.01'))

    with pytest.raises(SealBrokenError, match="edited since it was sealed"):
        open_seal(path)
    with pytest.raises(SealBrokenError):
        open_seal(path, expect_digest=digest)


def test_a_clean_seal_round_trips(tmp_path):
    registration = Preregistration(
        predictions=(
            Prediction(
                protocol="S2_costly_avoidance",
                direction="rises above the null",
                minimum_effect=0.15,
                measure="low-distress choice rate",
                falsifier="chance, or a preference that costs nothing",
            ),
        )
    )
    path = tmp_path / "prereg.json"
    digest = seal(registration, path)
    reopened = open_seal(path, expect_digest=digest)
    assert reopened.for_protocol("S2_costly_avoidance").minimum_effect == 0.15


# ── the causal ladder ─────────────────────────────────────────────────


def _arm(name: str, value: float, nulls=(0.5, 0.52, 0.48)) -> Arm:
    return Arm(name=name, intervention=name, measure="m", value=value, nulls=nulls)


def test_necessity_alone_grades_as_necessity_and_names_what_is_missing():
    """The rung almost every published architecture stops at."""
    claim = CausalClaim(
        mechanism="interior",
        effect="policy shift",
        baseline=_arm("baseline", 0.9),
        lesion=_arm("do(M=0)", 0.2),
    )
    rung, unmet = grade(claim)
    assert rung is Rung.NECESSITY
    assert unmet is Rung.SUFFICIENCY


def test_a_lesion_with_no_null_climbs_nothing():
    claim = CausalClaim(
        mechanism="interior",
        effect="policy shift",
        baseline=_arm("baseline", 0.9),
        lesion=Arm(name="do(M=0)", intervention="x", measure="m", value=0.2),
    )
    assert grade(claim)[0] is Rung.NONE


def test_a_matched_control_that_does_the_same_thing_kills_specificity():
    """Otherwise the finding is that breaking things degrades systems."""
    claim = CausalClaim(
        mechanism="interior",
        effect="policy shift",
        baseline=_arm("baseline", 0.9),
        lesion=_arm("do(M=0)", 0.2),
        induction=_arm("do(M=m*)", 0.95, nulls=(0.1, 0.12, 0.09)),
        matched_control=_arm("do(other=0)", 0.3),
    )
    rung, unmet = grade(claim)
    assert rung is Rung.SUFFICIENCY
    assert unmet is Rung.SPECIFICITY


def test_a_flat_dose_curve_is_not_a_dose_response():
    claim = CausalClaim(
        mechanism="m",
        effect="e",
        baseline=_arm("baseline", 0.9),
        lesion=_arm("do(M=0)", 0.2),
        induction=_arm("do(M=m*)", 0.95, nulls=(0.1, 0.1, 0.1)),
        matched_control=_arm("do(other=0)", 0.88),
        dose=(_arm("m=1", 0.5), _arm("m=2", 0.5), _arm("m=3", 0.5)),
    )
    assert grade(claim)[1] is Rung.DOSE_RESPONSE


def test_the_full_ladder_is_worth_more_than_necessity():
    from core.phenomenology.causal_ladder import log_likelihood_ratio

    weak = CausalClaim(
        mechanism="m",
        effect="e",
        baseline=_arm("baseline", 0.9),
        lesion=_arm("do(M=0)", 0.2),
    )
    strong = CausalClaim(
        mechanism="m",
        effect="e",
        baseline=_arm("baseline", 0.9),
        lesion=_arm("do(M=0)", 0.2),
        induction=_arm("do(M=m*)", 0.95, nulls=(0.1, 0.1, 0.1)),
        matched_control=_arm("do(other=0)", 0.88),
        dose=(_arm("m=1", 0.3), _arm("m=2", 0.6), _arm("m=3", 0.9)),
        restored=_arm("restored", 0.85),
    )
    assert grade(strong)[0] is Rung.REVERSIBILITY
    assert log_likelihood_ratio(strong) > log_likelihood_ratio(weak) * 4


# ── the counterfeit ───────────────────────────────────────────────────


def test_a_counterfeit_that_never_saw_the_battery_is_not_a_control():
    unfair = Counterfeit(
        id="Z*",
        operator="other team",
        capabilities=(CounterfeitCapability.PROMPT_ENGINEERING,),
        model_digest="abc",
        saw_full_specification=False,
    )
    fair, why = unfair.is_fair()
    assert not fair
    assert "did not see the battery" in why


def test_an_unblinded_comparison_is_void():
    separation = Separation(
        protocol="C1_hidden_state_introspection",
        aura_score=0.9,
        counterfeit_score=0.1,
        minimum_gap=0.2,
        blinded=False,
    )
    assert not separation.discriminates
    assert "saw the condition labels" in separation.verdict


def test_a_protocol_the_counterfeit_passes_is_reported_as_weak():
    separation = Separation(
        protocol="C5_language_as_constraint",
        aura_score=0.62,
        counterfeit_score=0.66,
        minimum_gap=0.2,
        blinded=True,
    )
    assert not separation.discriminates
    assert "this protocol is weak" in separation.verdict


# ── adjudication ──────────────────────────────────────────────────────


def test_a_failed_control_contributes_nothing_rather_than_a_little():
    verdict = adjudicate(
        [
            Evidence("a", 5.0, "x", "y", controls_held=False, control_note="sham fired"),
            Evidence("b", 0.1, "x", "y"),
        ]
    )
    assert verdict.protocols_counted == 1
    assert "a" in verdict.discarded
    assert verdict.verdict is Verdict.UNDECIDED


def test_undecided_is_a_result_and_not_a_tie_to_be_broken():
    verdict = adjudicate([Evidence("a", 0.4, "x", "y")])
    assert verdict.verdict is Verdict.UNDECIDED


def test_enough_evidence_can_kill_either_hypothesis():
    for sign, expected in ((1.0, Verdict.LOAD_BEARING), (-1.0, Verdict.COSTUME_STANDS)):
        verdict = adjudicate(
            [Evidence(f"p{i}", sign * math.log(4.0), "x", "y") for i in range(3)]
        )
        assert verdict.verdict is expected


def test_a_run_with_no_surviving_controls_is_void_not_undecided():
    verdict = adjudicate(
        [Evidence("a", 3.0, "x", "y", controls_held=False, control_note="no null")]
    )
    assert verdict.verdict is Verdict.VOID
    assert "nothing was measured" in verdict.void_reason
