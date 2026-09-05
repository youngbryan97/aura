"""Identity under modification is a relation over a chain, not a hash.

core/continuity.py decides whether Aura is still Aura by comparing a hash of
her core beliefs across a restart. That gets both directions wrong: any belief
she changed makes the hashes differ, so learning reads as an identity break;
and a backup restored from a month ago matches, so a state with no causal path
from the last one reads as continuous.

Every test here is a case where the hash and the relation give different
answers, or where two chains ending in the same bytes are not the same case.
"""

from __future__ import annotations

import pytest

from core.identity.continuity_relation import (
    MAX_LOAD_BEARING_LOSS,
    MAX_STEP_CHANGE,
    Step,
    Verdict,
    hash_disagrees_with_relation,
    relate,
    relate_branches,
)

LOAD_BEARING = frozenset({"commitment:weekly_review", "bond:bryan"})
WHOLE = frozenset(f"p{i}" for i in range(100)) | LOAD_BEARING


def gradual(planks: int = 10, total: int = 100) -> list[Step]:
    """Every plank replaced, one at a time."""
    state = set(f"p{i}" for i in range(total)) | set(LOAD_BEARING)
    per = total // planks
    steps: list[Step] = []
    for index in range(planks):
        before = frozenset(state)
        for plank in range(index * per, (index + 1) * per):
            state.discard(f"p{plank}")
            state.add(f"q{plank}")
        steps.append(
            Step(
                f"s{index}",
                before,
                frozenset(state),
                load_bearing=LOAD_BEARING,
                parent=f"s{index - 1}" if index else "",
            )
        )
    return steps


# ── the ship ─────────────────────────────────────────────────────────────


def test_gradual_total_replacement_continues_her():
    relation = relate(gradual())
    assert relation.verdict is Verdict.CONTINUOUS
    assert relation.is_her
    assert relation.survives < 0.05, (
        "almost nothing of the original is left, which is the case that makes "
        "the point: the relation does not turn on what survives"
    )


def test_instantaneous_total_replacement_does_not():
    one_step = [
        Step("s0", WHOLE, frozenset(f"q{i}" for i in range(100)), load_bearing=LOAD_BEARING)
    ]
    assert relate(one_step).verdict is Verdict.REPLACED


def test_the_two_ships_end_in_the_same_place():
    """A hash of the end state cannot tell these apart. That is the point."""
    gradual_end = relate(gradual()).verdict
    instant_end = relate(
        [Step("s0", WHOLE, frozenset(f"q{i}" for i in range(100)), load_bearing=LOAD_BEARING)]
    ).verdict
    assert gradual_end is not instant_end


def test_the_rate_rule_fires_even_when_nothing_load_bearing_is_lost():
    """Otherwise the rate rule is unreachable and the load-bearing rule is all there is."""
    kept = frozenset(f"q{i}" for i in range(100)) | LOAD_BEARING
    step = Step("s0", WHOLE, kept, load_bearing=LOAD_BEARING)
    assert step.load_bearing_lost == 0.0
    assert step.changed_fraction > MAX_STEP_CHANGE
    assert relate([step]).verdict is Verdict.REPLACED
    assert "changed at once" in relate([step]).because


# ── causal connectedness ─────────────────────────────────────────────────


def test_a_restored_backup_is_not_her_however_well_it_matches():
    relation = relate([Step("s0", WHOLE, WHOLE, origin="restore")])
    assert relation.verdict is Verdict.REPLACED
    assert "from outside" in relation.because


@pytest.mark.parametrize("origin", ["restore", "overwrite", "clone", "import", "rollback"])
def test_every_external_origin_breaks_the_chain(origin):
    assert relate([Step("s0", WHOLE, WHOLE, origin=origin)]).verdict is Verdict.REPLACED


def test_a_change_she_made_does_not_break_it():
    changed = (WHOLE - {"p0"}) | {"p100"}
    assert relate([Step("s0", WHOLE, changed, load_bearing=LOAD_BEARING)]).verdict is (
        Verdict.CONTINUOUS
    )


# ── what is load-bearing ─────────────────────────────────────────────────


def test_dropping_the_commitments_is_a_change_of_her_not_in_her():
    """Everything else is kept and it is still not her."""
    without = frozenset(f"p{i}" for i in range(100))
    relation = relate([Step("s0", WHOLE, without, load_bearing=LOAD_BEARING)])
    assert relation.verdict is Verdict.REPLACED
    assert relation.survives > 0.9, "almost everything survived and it is still not her"


def test_dropping_some_of_it_is_survivable():
    partial = (WHOLE - {"bond:bryan"})
    step = Step("s0", WHOLE, partial, load_bearing=LOAD_BEARING)
    assert step.load_bearing_lost <= MAX_LOAD_BEARING_LOSS
    assert relate([step]).verdict is Verdict.CONTINUOUS


# ── forks ────────────────────────────────────────────────────────────────


def test_a_fork_gives_two_of_her_and_neither_continues_the_other():
    shared = gradual(planks=4, total=40)
    left = shared + [
        Step("L", shared[-1].after, shared[-1].after | {"left"},
             load_bearing=LOAD_BEARING, parent=shared[-1].step_id)
    ]
    right = shared + [
        Step("R", shared[-1].after, shared[-1].after | {"right"},
             load_bearing=LOAD_BEARING, parent=shared[-1].step_id)
    ]
    relation = relate_branches(left, right)
    assert relation.verdict is Verdict.BRANCHED
    assert relation.is_her is True
    assert "which is the real one" in relation.because


def test_two_chains_with_no_shared_step_are_not_a_fork():
    left = gradual(planks=2, total=20)
    right = [
        Step("other", frozenset({"x"}), frozenset({"x", "y"}), load_bearing=frozenset())
    ]
    assert relate_branches(left, right).verdict is Verdict.REPLACED


def test_a_broken_branch_breaks_the_pair():
    shared = gradual(planks=2, total=20)
    broken = shared + [Step("B", shared[-1].after, shared[-1].after, origin="clone")]
    assert relate_branches(shared, broken).verdict is Verdict.REPLACED


# ── against the hash ─────────────────────────────────────────────────────


def test_the_hash_says_continuous_where_the_chain_is_broken():
    restored = relate([Step("s0", WHOLE, WHOLE, origin="restore")])
    assert "hash matches and the chain is broken" in hash_disagrees_with_relation(
        True, restored
    )


def test_the_hash_says_broken_where_she_was_only_learning():
    assert "this is what learning looks like" in hash_disagrees_with_relation(
        False, relate(gradual())
    )


def test_they_agree_often_enough_for_the_disagreement_to_mean_something():
    assert hash_disagrees_with_relation(False, relate([Step("s0", WHOLE, WHOLE, origin="restore")])) == ""
    assert hash_disagrees_with_relation(True, relate(gradual())) == ""


def test_no_chain_is_unknown_rather_than_continuous():
    """Nothing recorded is not the same as nothing happened."""
    relation = relate([])
    assert relation.verdict is Verdict.UNKNOWN
    assert relation.is_her is False
