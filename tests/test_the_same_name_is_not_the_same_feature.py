"""Same number, same name, different meaning — and the digest said match.

`layout_digest` fingerprints feature names, channels, ranges and a version.
That is what a head checks before attaching, and it is not enough.

On 2026-08-25 this repository changed what several dimensions MEAN without
changing any of those things:

    substrate.*      unreachable            -> 34 live LiquidSubstrate readings
    uncertainty.*    absent on every turn   -> welfare-model readings
    temporal.past    a copy of recall_hits  -> episodic recency
    temporal.future  a copy of goal.priority-> priority x remaining progress
    affect.curiosity absent                 -> the substrate's curiosity axis

Same names, same ranges, same digest. A head fitted to the old corpus would
attach to the repaired state and be applied to numbers that no longer mean
what it learned — and every check in the pathway would report a match.

So `semantics_digest` fingerprints the derivations: each probe plus the
closure of module functions it calls, structurally, with docstrings dropped so
rewriting an explanation is not mistaken for rewriting a rule.
"""

from __future__ import annotations

import pytest

from core.brain.llm.endogenous_state import (
    layout_digest,
    semantics_digest,
    _derivation_closure,
)


def test_the_two_digests_answer_different_questions():
    assert layout_digest() != semantics_digest()
    assert len(semantics_digest()) == 32


def test_the_closure_reaches_the_helpers_a_probe_calls():
    """A derivation that moved into a helper is still a derivation."""
    closure = set(_derivation_closure())

    assert "_probe_temporal" in closure
    assert "_probe_substrate" in closure
    # Reached only through _probe_substrate, and it decides whether a stale
    # snapshot is used — as load-bearing as anything in the probe itself.
    assert "_substrate_snapshot" in closure
    assert "_goal_features" in closure


def test_changing_a_derivation_changes_the_digest(monkeypatch):
    """The exact case from 2026-08-25, in miniature.

    `temporal.past` returned `memory.recall_hits` and now returns episodic
    recency. Names and ranges are identical either way.
    """
    from core.brain.llm import endogenous_state as module

    before = semantics_digest()
    before_layout = layout_digest()

    source = module.inspect.getsource(module)
    rewired = source.replace(
        'memory.get(\n            "memory.episodic_recency",',
        'memory.get(\n            "memory.recall_hits",',
        1,
    )
    assert rewired != source, "the derivation under test was not found"

    monkeypatch.setattr(module.inspect, "getsource", lambda obj: rewired)

    assert semantics_digest() != before, (
        "a changed derivation left the semantics digest identical"
    )
    assert layout_digest() == before_layout, (
        "this is the point: the layout digest cannot see the change"
    )


def test_rewriting_a_docstring_does_not_change_the_digest(monkeypatch):
    """Prose is not logic. A comment or an explanation must not force a
    retrain, or the digest becomes noise nobody acts on."""
    from core.brain.llm import endogenous_state as module

    before = semantics_digest()
    source = module.inspect.getsource(module)
    reworded = source.replace(
        '"""Temporal orientation, derived from what the other organs are doing.',
        '"""Where in time her attention sits, derived from the other organs.',
        1,
    )
    assert reworded != source, "the docstring under test was not found"

    monkeypatch.setattr(module.inspect, "getsource", lambda obj: reworded)

    assert semantics_digest() == before


def test_an_unreadable_module_says_so_rather_than_claiming_a_match(monkeypatch):
    """Failing open here would let any head attach to any semantics."""
    from core.brain.llm import endogenous_state as module

    def _raise(obj):
        raise OSError("source unavailable")

    monkeypatch.setattr(module.inspect, "getsource", _raise)

    assert semantics_digest() == "unreadable"
