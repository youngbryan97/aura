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
