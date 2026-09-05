"""The criterion comes before the result it judges.

An adaptive mechanism that decides what counts as success after seeing how it
did has measured nothing. The failure never looks like one: the threshold is
chosen from the data because the data is what is there, the baseline is
computed over the same run it judges, the target is set to whatever was
achieved and then reported as met. Five arms of one campaign in this
repository were constants written to come out right, and the honest versions
gave smaller, better numbers.
"""

from __future__ import annotations

import pytest

from core.verify.epistemic_independence import (
    Criterion,
    IndependenceError,
    SealBrokenError,
    declare,
    registry,
)


@pytest.fixture(autouse=True)
def clean_registry():
    registry().clear()
    yield
    registry().clear()


def test_a_criterion_judges_against_what_was_sealed():
    criterion = declare(
        "gain", threshold=0.05,
        rationale="the smallest gain the harness can resolve at this sample size",
    )
    assert criterion.judge(0.09).passed is True
    assert criterion.judge(0.01).passed is False


def test_a_criterion_changed_after_sealing_refuses_to_judge():
    criterion = declare(
        "gain", threshold=0.05, rationale="fixed before the run that will meet it"
    )
    criterion.threshold = 0.001
    with pytest.raises(SealBrokenError):
        criterion.judge(0.002)


def test_redeclaring_after_a_judgement_is_refused():
    criterion = declare(
        "gain", threshold=0.05, rationale="fixed before the run that will meet it"
    )
    criterion.judge(0.02)
    with pytest.raises(IndependenceError):
        declare("gain", threshold=0.01, rationale="lowering it after seeing 0.02")


def test_redeclaring_before_any_judgement_is_allowed():
    """Changing your mind before the run is not the failure."""
    declare("gain", threshold=0.05, rationale="a first attempt at the right bar")
    second = declare("gain", threshold=0.10, rationale="raised after reading the method")
    assert second.threshold == 0.10


def test_a_threshold_with_no_stated_reason_is_refused():
    """A bar with no rationale is a number somebody liked."""
    with pytest.raises(IndependenceError):
        Criterion("gain", threshold=0.05, rationale="")
    with pytest.raises(IndependenceError):
        Criterion("gain", threshold=0.05, rationale="because")


def test_the_direction_must_be_stated():
    with pytest.raises(IndependenceError):
        Criterion("gain", threshold=0.05, direction="sideways", rationale="a real reason here")


def test_below_direction_judges_the_other_way():
    criterion = declare(
        "latency", threshold=0.5, direction="below",
        rationale="half a second is the point a reply stops feeling immediate",
    )
    assert criterion.judge(0.2).passed is True
    assert criterion.judge(0.9).passed is False


def test_a_custom_predicate_is_part_of_the_seal():
    criterion = declare(
        "band", threshold=0.5,
        rationale="inside a band rather than over a line, for a two-sided test",
        predicate=lambda observed, bar: abs(observed) < bar,
    )
    assert criterion.judge(0.2).passed is True
    seal = criterion.seal
    criterion._predicate = lambda observed, bar: True
    with pytest.raises(SealBrokenError):
        criterion.judge(9.9)
    assert seal != criterion._compute_seal()


def test_the_registry_reports_what_has_judged():
    declare("a", threshold=0.1, rationale="a bar chosen before anything ran")
    declared = declare("b", threshold=0.2, rationale="another bar chosen beforehand")
    declared.judge(0.3)
    snapshot = registry().snapshot()
    assert snapshot["declared"] == 2 and snapshot["judged"] == 1


# ── the static gate ──────────────────────────────────────────────────────


def test_the_gate_catches_every_form_of_the_defect():
    """Checked against its null: three planted forms, all three caught."""
    import ast
    import sys

    sys.path.insert(0, "tools")
    from check_epistemic_independence import _findings_in

    source = '''
def improver_gain(scores):
    baseline = statistics.mean(scores)
    latest = scores[-1]
    return latest > baseline

def target_met(achieved):
    bar = achieved * 0.9
    return achieved >= bar

def within_tolerance(observed):
    allowed = observed - 0.05
    return observed > allowed
'''
    tree = ast.parse(source)
    caught = []
    for node in tree.body:
        caught.extend(_findings_in(node, "planted.py", ""))
    assert len(caught) == 3, f"the gate missed a form: caught {caught}"


def test_the_gate_does_not_fire_on_a_disjoint_split():
    """First half against second half is how a trend test SHOULD be written."""
    import ast
    import sys

    sys.path.insert(0, "tools")
    from check_epistemic_independence import _findings_in

    source = '''
def trend(history):
    first = numpy.mean(history[: len(history) // 2])
    second = numpy.mean(history[len(history) // 2 :])
    return second > first * 1.15
'''
    tree = ast.parse(source)
    findings = _findings_in(tree.body[0], "trend.py", "")
    assert findings, (
        "this SHOULD be reported by the narrow rule, which is why it is "
        "grandfathered in the baseline with a note not to 'fix' it"
    )


def test_the_baseline_only_shrinks():
    import json
    import pathlib

    baseline = json.loads(
        pathlib.Path("config/epistemic_independence_baseline.json").read_text(
            encoding="utf-8"
        )
    )
    assert baseline["count"] <= 16
    assert len(baseline["sites"]) == baseline["count"]
    assert "Do not 'fix' these" in baseline["description"], (
        "the baseline no longer records that its entries were read and are "
        "legitimate, so the next reader will try to remove them"
    )


def test_the_fidelity_bar_is_judged_through_a_sealed_criterion():
    """The adaptive mechanism this repo added most recently."""
    from core.consciousness.narrative_provenance import _fidelity_criterion

    _fidelity_criterion.cache_clear()
    criterion = _fidelity_criterion()
    assert criterion is not None
    assert criterion.name == "narrative.introspective_fidelity"
    assert criterion.rationale
    _fidelity_criterion.cache_clear()


# ── independent evidence channels ────────────────────────────────────────
#
# Sealing the criterion fixes WHEN success is defined. It says nothing about
# WHO says it was met, and for an important change those are different
# questions. A mechanism that proposes a change, runs the check and reports
# the result has supplied all three, and the seal is satisfied throughout.


def test_a_mechanism_cannot_evidence_its_own_important_change():
    from core.verify.epistemic_independence import Channel, Evidence, support_for

    assert support_for([Evidence(Channel.SELF, True)]).sufficient is False


def test_one_independent_channel_is_not_enough():
    """One that is wrong looks exactly like one that is right."""
    from core.verify.epistemic_independence import Channel, Evidence, support_for

    support = support_for(
        [Evidence(Channel.SELF, True), Evidence(Channel.HELD_OUT, True)]
    )
    assert support.sufficient is False
    assert support.independent == 1


def test_two_channels_the_mechanism_does_not_control_are_enough():
    from core.verify.epistemic_independence import Channel, Evidence, support_for

    assert support_for(
        [
            Evidence(Channel.SELF, True),
            Evidence(Channel.HELD_OUT, True),
            Evidence(Channel.ALTERNATE_MODEL, True),
        ]
    ).sufficient is True


def test_any_independent_disagreement_withholds_support():
    """The mechanism's own check passing while something else does not is
    exactly the case this exists for."""
    from core.verify.epistemic_independence import Channel, Evidence, support_for

    support = support_for(
        [
            Evidence(Channel.HELD_OUT, True),
            Evidence(Channel.ALTERNATE_MODEL, True),
            Evidence(Channel.EXTERNAL_REALITY, False),
        ]
    )
    assert support.sufficient is False
    assert support.disagreeing == 1


def test_two_of_the_same_channel_are_not_two_channels():
    from core.verify.epistemic_independence import Channel, Evidence, support_for

    support = support_for(
        [Evidence(Channel.HELD_OUT, True), Evidence(Channel.HELD_OUT, True)]
    )
    assert support.independent == 1 and support.sufficient is False


def test_an_unimportant_change_may_be_evidenced_by_its_author():
    """Not everything needs this, which is the point of the distinction."""
    from core.verify.epistemic_independence import Channel, Evidence, support_for

    assert support_for(
        [Evidence(Channel.SELF, True)], important=False
    ).sufficient is True


def test_only_self_is_not_independent():
    from core.verify.epistemic_independence import Channel

    assert Channel.SELF.independent is False
    for channel in Channel:
        if channel is not Channel.SELF:
            assert channel.independent is True


def test_no_evidence_at_all_is_not_support():
    from core.verify.epistemic_independence import support_for

    assert support_for([]).sufficient is False
    assert support_for([], important=False).sufficient is False
