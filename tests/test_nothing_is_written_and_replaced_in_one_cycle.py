"""A field written and then written again, without reading what was there.

This defect has turned up four times in the subject core alone. A phase
computes something, writes it onto the state, and a later step in the same turn
recomputes the field from scratch. Nothing fails: the field holds a plausible
number, the tests that read it pass, and whatever fed the first write is dead.

`affect.curiosity` was the worst of them. The lifetime state blended it toward
how unprecedented the moment was; discourse depth nudged it; continuity
pressure nudged it; and `_derive_metrics` recomputed it from two emotion
channels, three steps later, every turn.
"""

from __future__ import annotations

import ast
from pathlib import Path

from tools.audit_overwritten_in_the_same_cycle import (
    ENTRY_POINTS,
    ROOTS,
    _called_in_order,
    _writes_of,
    findings,
)

REPO = Path(__file__).resolve().parents[1]

EXAMPLE = '''
class ExamplePhase:
    async def execute(self, state, objective=None):
        affect = state.affect
        self._nudge(affect)
        self._derive(affect)
        self._carry(affect)
        return state

    def _nudge(self, affect):
        affect.curiosity = min(1.0, affect.curiosity + 0.02)

    def _derive(self, affect):
        affect.curiosity = max(affect.emotions["curiosity"], 0.5)

    def _carry(self, affect):
        blended = 0.8 * affect.curiosity + 0.2
        affect.curiosity = blended
'''


def _methods():
    tree = ast.parse(EXAMPLE)
    klass = tree.body[0]
    return {node.name: node for node in klass.body}


def test_nothing_in_the_tree_writes_a_field_and_then_replaces_it() -> None:
    loose = findings()
    assert not loose, "same-cycle overwrites: " + "; ".join(
        f"{item['file']} {item['function']}: {item['overwritten_by']} replaces {item['field']}"
        for item in loose[:6]
    )


def test_the_rule_can_fire() -> None:
    """A rule with no worked example reports green whatever happens."""
    methods = _methods()
    calls = [name for name, _ in _called_in_order(methods["execute"])]
    assert calls == ["_nudge", "_derive", "_carry"]

    assert _writes_of(methods["_derive"])["affect.curiosity"][1] is False, (
        "a helper that recomputes the field from something else reads as carrying it"
    )


def test_a_blend_through_a_local_counts_as_carrying() -> None:
    """`blended = 0.8 * affect.curiosity + 0.2` then `affect.curiosity =
    blended` carries what was there as much as a read-modify-write on one
    line, and a rule that only looks at the assignment's own right-hand side
    calls it a replacement."""
    methods = _methods()
    assert _writes_of(methods["_carry"])["affect.curiosity"][1] is True


def test_a_read_modify_write_counts_as_carrying() -> None:
    methods = _methods()
    assert _writes_of(methods["_nudge"])["affect.curiosity"][1] is True


def test_the_scan_is_aimed_at_the_state_a_turn_carries() -> None:
    """An object's own status flag set on two branches of a try/except is
    ordinary code; a field the next phase will read is not."""
    assert "affect" in ROOTS and "state" in ROOTS
    assert "self" not in ROOTS
    assert "execute" in ENTRY_POINTS
