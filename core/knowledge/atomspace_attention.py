"""core/knowledge/atomspace_attention.py — reading and resetting the economy.

Three operations on :class:`core.knowledge.atomspace.AtomSpace`'s attention
that a caller performs from outside rather than the store performing on
itself: clearing salience between tasks, and asking what spreading cost so a
cheaper policy can be given the same budget.

They live here because they are how attention is MEASURED and controlled, not
how it works, and because the store was thirty-seven methods with them inside.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from core.knowledge.atomspace import HEBBIAN, Atom, Link

if TYPE_CHECKING:
    from core.knowledge.atomspace import AtomSpace

__all__ = ["reset_attention", "spread_importance_touches", "neighbours"]


def spread_importance_touches(space: "AtomSpace") -> int:
    """Spread, and say how many atoms it cost to do it.

    Attention is only worth comparing against a cheaper policy at equal
    compute, and "equal compute" for a graph policy is atoms touched.
    :meth:`spread_importance` returns STI moved, which is the benefit
    side; this is the price.
    """
    # Counted and spread under ONE hold of the lock. Counting first and then
    # calling spread_importance() takes the lock twice, sorts the focus twice,
    # and lets the focus change in between - so the price reported would be
    # for a spread that did not happen.
    touched = 0
    with space._lock:
        for atom, sti in space.attentional_focus():
            if sti > 0 and atom in space._records:
                touched += 1 + len(space._neighbors_locked(space._records[atom].atom))
        space.spread_importance()
    return touched


def neighbours(space: "AtomSpace", atom: Atom) -> list[Atom]:
    """Everything one hop from ``atom`` through the metagraph.

    Public because a baseline that walks the graph has to be able to walk
    the same graph. A comparison where only one arm can see the structure
    is not a comparison.
    """
    with space._lock:
        return sorted(space._neighbors_locked(atom), key=str)


def reset_attention(space: "AtomSpace") -> float:
    """Return every atom's STI to the fund, keeping truth and structure.

    A second task that starts on the first task's salience is not a second
    task. Long-term importance is left alone: it is the record of what has
    repeatedly mattered, which is not a thing one task gets to clear.
    Returns the STI reclaimed.
    """
    with space._lock:
        reclaimed = 0.0
        for rec in space._records.values():
            reclaimed += rec.av.sti
            rec.av.sti = 0.0
        space._sti_fund = space._sti_fund_capacity
        return reclaimed

