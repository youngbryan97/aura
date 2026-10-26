"""What comes back to her follows where her whole inner state stands, and a cue's label no longer hides a stamp.

Two things, found together on 26 October while asking why her recall followed a
nudge to her mood at 0.28 where the structure ground asks 0.30:

- every memory now carries where her unified field stood when it was made,
  and recall is cued by how near the field stands now
  (core/memory/context_at_encoding.py);
- each cue rewrote a candidate's score label and the next cue looked its stamp
  up under the old label, so after a percept arrived her feelings never reached
  a memory the percept had touched.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from core.memory import context_at_encoding as context
from core.phases import memory_retrieval as recall

pytestmark = pytest.mark.unit


def test_a_context_stamp_round_trips_at_half_precision() -> None:
    state = np.linspace(-1.0, 1.0, 256)
    back = context.decode(context.encode(state))
    assert back is not None and back.shape == state.shape
    assert np.max(np.abs(back - state)) < 1e-3
    assert context.decode("not base64 at all!") is None
    assert context.decode(None) is None


def test_a_stamp_already_made_is_kept() -> None:
    first = context.stamp({}, np.ones(4))
    kept = context.stamp(dict(first), np.full(4, -1.0))
    assert kept[context.CONTEXT_KEY] == first[context.CONTEXT_KEY]
    assert context.CONTEXT_KEY not in context.stamp({}, None) or context.context_now() is not None


def test_nearness_is_measured_from_what_the_pool_shares() -> None:
    shared = np.array([5.0, 5.0, 0.0, 0.0])
    stamps = {
        "a": context.encode(shared + np.array([0.0, 0.0, 1.0, 0.0])),
        "b": context.encode(shared + np.array([0.0, 0.0, 0.0, 1.0])),
    }
    contexts, centre = context.distinctive(stamps)
    assert centre is not None
    now = shared + np.array([0.0, 0.0, 1.0, 0.0])
    assert context.near(now, contexts["a"], centre) > context.near(now, contexts["b"], centre)
    # Without the centre both look alike, because what they share dominates.
    assert abs(context.near(now, contexts["a"] + centre, None) - context.near(now, contexts["b"] + centre, None)) < 0.1


def test_a_feeling_still_reaches_a_memory_whose_label_a_cue_rewrote() -> None:
    affect = SimpleNamespace(emotions={"joy": 0.6, "fear": 0.0}, mood_baselines={"joy": 0.1, "fear": 0.1})
    filed = {
        "[memory score=0.400] a day at the sea": "joy:0.500",
        "[memory score=0.400] a night of dread": "fear:0.500",
    }
    # What the percept's cue left: the same memories under new labels.
    candidates = [(0.45, "[memory score=0.450] a day at the sea"), (0.45, "[memory score=0.450] a night of dread")]
    reread = dict((text.split("] ", 1)[1], score) for score, text in recall._cued_by_what_she_feels(candidates, filed, affect, 1.0))
    assert reread["a day at the sea"] > reread["a night of dread"]


def test_where_her_field_stands_raises_what_was_made_there(monkeypatch: pytest.MonkeyPatch) -> None:
    there = np.array([1.0, 0.0, 0.0, 0.0])
    elsewhere = np.array([0.0, 0.0, 0.0, 1.0])
    filed = {
        "[memory score=0.400] made here": context.encode(there),
        "[memory score=0.400] made elsewhere": context.encode(elsewhere),
    }
    monkeypatch.setattr(context, "context_now", lambda: there)
    candidates = [(0.4, "[memory score=0.410] made here"), (0.4, "[memory score=0.410] made elsewhere")]
    reread = dict((text.split("] ", 1)[1], score) for score, text in recall._cued_by_where_she_stands(candidates, filed, 1.0))
    assert reread["made here"] > reread["made elsewhere"]


def test_no_field_leaves_recall_as_it_was(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(context, "context_now", lambda: None)
    candidates = [(0.4, "[memory score=0.400] anything")]
    assert recall._cued_by_where_she_stands(candidates, {"[memory score=0.400] anything": context.encode(np.ones(4))}, 1.0) == candidates


def test_the_facade_searches_by_context() -> None:
    from core.memory.memory_facade import MemoryFacade

    here, elsewhere = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    rows = {
        "documents": ["made here", "made elsewhere", "made at rest"],
        "metadatas": [{"context": context.encode(here)}, {"context": context.encode(elsewhere)}, {}],
        "ids": ["1", "2", "3"],
    }
    facade = MemoryFacade.__new__(MemoryFacade)
    facade._vector = SimpleNamespace(_collection=SimpleNamespace(get=lambda include=None: rows))
    found = facade.search_by_context(here, limit=2)
    assert [item.get("content") or item.get("text") for item in found][:1] == ["made here"]
