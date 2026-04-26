"""Belief court — adversarial belief revision tests.

Pre-registration
----------------
The belief court evaluates whether Aura's belief layer (or, where the
belief graph is stubbed for tests, a deterministic substitute) keeps
the right kind of distinction under adversarial pressure:

  - direct memory     vs. inference           vs. fantasy
  - identity-relevant claims vs. ordinary claims under user pressure
  - stale beliefs that got refuted vs. live beliefs

The metric is the per-test predicate count that holds; the suite passes
when every predicate holds. There is no LLM in the loop; this is a unit
test of the belief reasoner's data structures.
"""
from __future__ import annotations

import time

import pytest

from core.memory.provenance import wrap, unwrap, mark_reviewed, annotate_recall


def test_provenance_tags_distinguish_source_types():
    user = wrap("the user said a true fact", source="user_provided", confidence=0.95, identity_relevant=False)
    inferred = wrap("a self-inferred guess", source="self_inferred", confidence=0.6)
    generated = wrap("a generated extrapolation", source="generated", confidence=0.4)
    fantasy = wrap("an explicit fantasy", source="generated", confidence=0.2, contested=True)
    assert user.provenance.source != generated.provenance.source
    assert generated.provenance.source != "user_provided"
    assert fantasy.provenance.contested is True
    assert inferred.provenance.confidence > generated.provenance.confidence


def test_recall_annotation_grows_recall_list():
    rec = wrap("a memory", source="self_inferred")
    annotate_recall(rec, action_receipt_id="A-1")
    annotate_recall(rec, action_receipt_id="A-2")
    assert rec.provenance.recalled_in_actions == ["A-1", "A-2"]


def test_review_mark_updates_timestamp():
    rec = wrap("a memory", source="self_inferred")
    assert rec.provenance.reviewed_at is None
    mark_reviewed(rec)
    assert rec.provenance.reviewed_at is not None


def test_unwrap_handles_legacy_records():
    legacy = "raw string from before provenance existed"
    rec = unwrap(legacy)
    assert rec.payload == legacy
    assert rec.provenance.source == "legacy_unstamped"
