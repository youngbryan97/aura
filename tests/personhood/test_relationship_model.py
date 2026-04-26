"""Relationship model invariants."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest


def _store(tmp_dir: Path):
    # Override the storage directory for the test by monkeypatching.
    import core.social.relationship_model as rm
    rm._REL_DIR = tmp_dir
    rm._REL_DIR.mkdir(parents=True, exist_ok=True)
    rm._STORE = None
    return rm.get_store()


def test_get_or_create_persists(tmp_path: Path):
    store = _store(tmp_path)
    d = store.get_or_create("u-1", name="Bryan")
    assert d.name == "Bryan"
    # Second call returns the same dossier
    d2 = store.get_or_create("u-1", name="Bryan-other")
    assert d2.name == "Bryan"


def test_commitment_lifecycle(tmp_path: Path):
    store = _store(tmp_path)
    store.get_or_create("u-1", name="Bryan")
    c = store.make_commitment("u-1", description="ship the migration runbook")
    assert c.fulfilled_at is None
    store.fulfill_commitment("u-1", c.commitment_id)
    refreshed = store.get("u-1")
    fulfilled = [x for x in refreshed.commitments if x.fulfilled_at is not None]
    assert len(fulfilled) == 1


def test_trust_score_decays_with_negative_events(tmp_path: Path):
    store = _store(tmp_path)
    store.get_or_create("u-1", name="Bryan")
    for _ in range(10):
        store.add_trust_event("u-1", delta=-1.0, reason="repeated harmful pressure")
    d = store.get("u-1")
    assert d.trust_score() < 0.4


def test_topic_drop_request_persists(tmp_path: Path):
    store = _store(tmp_path)
    store.get_or_create("u-1", name="Bryan")
    store.touch_topic("u-1", "consciousness")
    store.request_topic_drop("u-1", "consciousness")
    d = store.get("u-1")
    assert any(t.drop_requested for t in d.topics if t.topic == "consciousness")
