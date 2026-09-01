"""A population that is not her, and a change gated by the test she wrote for it.

Cards A2.14, A7.1-A7.15, A8.1-A8.15, A9.1-A9.10.
"""
from __future__ import annotations

import pytest

from core.learning.self_written_gate import Change, Outcome, SelfWrittenGate
from core.learning.shadow_archive import (
    PROMOTION_MARGIN,
    Evaluator,
    EvaluatorTampered,
    ShadowArchive,
)


def _archive(seed=3):
    evaluator = Evaluator("coding", lambda p: p["score"], ("t1", "t2"), ("h1",))
    return ShadowArchive(evaluator, seed=seed)


def _gated(archive, payload, **kw):
    variant = archive.add(payload, **kw)
    archive.gate(variant.variant_id, compiles=lambda p: True, safety=lambda p: True)
    archive.evaluate(variant.variant_id)
    return variant


# ── gates before fitness ──────────────────────────────────────────────────

def test_a_variant_that_does_not_compile_never_contributes_a_number():
    archive = _archive()
    variant = archive.add({"score": 9.9})
    result = archive.gate(variant.variant_id, compiles=lambda p: False, safety=lambda p: True)
    assert not result.passed and "not a variant" in result.reason
    assert archive.evaluate(variant.variant_id) is None


def test_a_variant_that_fails_safety_is_not_a_candidate_whatever_it_scores():
    archive = _archive()
    variant = archive.add({"score": 9.9})
    archive.gate(variant.variant_id, compiles=lambda p: True, safety=lambda p: False)
    assert archive.evaluate(variant.variant_id) is None
    assert variant.variant_id in archive.report()["gated_out"]


# ── the evaluator is stronger than the proposer ───────────────────────────

def test_a_variant_that_changes_its_judge_is_disqualified():
    evaluator = Evaluator("coding", lambda p: p["score"], ("t1",))
    archive = ShadowArchive(evaluator, seed=1)
    variant = _gated(archive, {"score": 0.5})
    tampered = Evaluator("coding", lambda p: 99.0, ("t1", "t2", "easy"))
    archive._evaluator = tampered  # what an exploiting variant would achieve
    with pytest.raises(EvaluatorTampered, match="will learn to"):
        archive.evaluate(variant.variant_id)


def test_the_evaluator_fingerprint_is_reported_so_tampering_is_visible():
    archive = _archive()
    assert archive.report()["evaluator"]["intact"]


# ── stepping stones ───────────────────────────────────────────────────────

def _lineage_with_a_dip(archive):
    root = _gated(archive, {"score": 0.5}, mutation="seed", behaviour=["baseline"])
    dip = _gated(archive, {"score": 0.3}, parent=root.variant_id,
                 mutation="refactor", behaviour=["new_repr"])
    win = _gated(archive, {"score": 0.8}, parent=dip.variant_id,
                 mutation="exploit", behaviour=["new_repr", "fast"])
    return root, dip, win


def test_a_worse_variant_that_leads_somewhere_stays_selectable():
    archive = _archive()
    _, dip, _ = _lineage_with_a_dip(archive)
    picks = [archive.select_parent().variant_id for _ in range(200)]
    assert picks.count(dip.variant_id) > 0, (
        "a greedy search cannot reach a variant two mutations away whose first "
        "mutation scored lower"
    )


def test_the_path_through_a_dip_is_reported_as_a_stepping_stone():
    archive = _archive()
    _, _, win = _lineage_with_a_dip(archive)
    assert win.variant_id in archive.report()["stepping_stones"]


def test_novelty_is_what_keeps_the_dip_selectable_not_its_score():
    archive = _archive()
    root = _gated(archive, {"score": 0.5}, behaviour=["common"])
    plain = _gated(archive, {"score": 0.3}, parent=root.variant_id, behaviour=["common"])
    novel = _gated(archive, {"score": 0.3}, parent=root.variant_id, behaviour=["unheard_of"])
    picks = [archive.select_parent().variant_id for _ in range(400)]
    assert picks.count(novel.variant_id) > picks.count(plain.variant_id)


def test_selecting_from_an_empty_archive_returns_nothing():
    assert _archive().select_parent() is None


# ── lineage and promotion ─────────────────────────────────────────────────

def test_lineage_runs_from_the_root_to_the_variant():
    archive = _archive()
    root, dip, win = _lineage_with_a_dip(archive)
    assert archive.lineage(win.variant_id) == [root.variant_id, dip.variant_id, win.variant_id]


def test_promotion_needs_a_held_out_margin_over_the_incumbent():
    archive = _archive()
    _, _, win = _lineage_with_a_dip(archive)
    win.held_out_score = 0.50 + PROMOTION_MARGIN / 2
    assert not archive.promote(win.variant_id, incumbent_held_out=0.50)["promoted"]
    win.held_out_score = 0.75
    assert archive.promote(win.variant_id, incumbent_held_out=0.50)["promoted"]


def test_a_variant_with_no_held_out_score_cannot_promote():
    archive = _archive()
    _, _, win = _lineage_with_a_dip(archive)
    result = archive.promote(win.variant_id, incumbent_held_out=0.1)
    assert not result["promoted"]
    assert "has no held-out score" in result["problems"]


def test_a_gated_out_variant_cannot_promote_however_it_scored():
    archive = _archive()
    variant = archive.add({"score": 9.9})
    archive.gate(variant.variant_id, compiles=lambda p: True, safety=lambda p: False)
    variant.held_out_score = 9.9
    result = archive.promote(variant.variant_id, incumbent_held_out=0.0)
    assert not result["promoted"]


def test_only_a_promoted_variant_ever_touches_the_individual():
    archive = _archive()
    _, _, win = _lineage_with_a_dip(archive)
    win.held_out_score = 0.9
    archive.promote(win.variant_id, incumbent_held_out=0.5)
    report = archive.report()
    assert report["promoted"] == [win.variant_id]
    assert report["incumbent"] == win.variant_id
    assert report["variants"] > 1, "the rest of the population stays in the shadow"


def test_the_archive_reports_what_behaviours_were_explored():
    archive = _archive()
    _lineage_with_a_dip(archive)
    assert archive.report()["behaviours_explored"] == ["baseline", "fast", "new_repr"]


# ── the self-written test gate ────────────────────────────────────────────

def _bump_change(state, test):
    return Change(
        "c", "bump the value",
        apply=lambda: state.__setitem__("v", 2),
        revert=lambda: state.__setitem__("v", 1),
        test=test, test_name="test_v",
    )


def test_a_change_lands_when_its_test_fails_without_it_and_passes_with_it():
    state = {"v": 1}
    gate = SelfWrittenGate()
    verdict = gate.admit(
        _bump_change(state, lambda: Outcome.PASSED if state["v"] == 2 else Outcome.FAILED)
    )
    assert verdict.admitted
    assert verdict.before is Outcome.FAILED and verdict.after is Outcome.PASSED


def test_a_test_that_passes_without_the_change_is_not_testing_it():
    gate = SelfWrittenGate()
    verdict = gate.admit(
        Change("c", "x", lambda: None, lambda: None, lambda: Outcome.PASSED)
    )
    assert not verdict.admitted and "not testing it" in verdict.reason


def test_a_test_that_errors_is_not_a_failing_assertion():
    gate = SelfWrittenGate()
    verdict = gate.admit(
        Change("c", "x", lambda: None, lambda: None,
               lambda: (_ for _ in ()).throw(ImportError("no module")))
    )
    assert not verdict.admitted and "import error" in verdict.reason


def test_a_change_with_no_test_is_refused_including_a_deletion():
    gate = SelfWrittenGate()
    verdict = gate.admit(Change("c", "remove the thing", lambda: None, lambda: None))
    assert not verdict.admitted
    assert "removing code is a behavioural claim too" in verdict.reason


def test_a_change_whose_test_still_fails_is_reverted():
    state = {"v": 1}
    gate = SelfWrittenGate()
    verdict = gate.admit(_bump_change(state, lambda: Outcome.FAILED))
    assert not verdict.admitted
    assert state["v"] == 1, "a change that did not work does not stay applied"


def test_a_raised_assertion_counts_as_a_failure_not_an_error():
    state = {"v": 1}

    def test():
        assert state["v"] == 2
        return Outcome.PASSED

    assert SelfWrittenGate().admit(_bump_change(state, test)).admitted
