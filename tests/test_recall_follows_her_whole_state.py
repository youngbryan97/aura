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


def test_the_searches_read_what_she_wrote_through_the_gateway(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """In a subject run the vault was empty and everything she had was in the gateway's records."""
    import core.memory.gateway_record_index as index_module
    import core.memory.memory_write_gateway as gateway_module
    from core.memory.felt_at_encoding import encode as encode_felt
    from core.memory.memory_facade import MemoryFacade

    here, elsewhere = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    entries = [
        SimpleNamespace(content="made here", memory_id="a", metadata={"context": context.encode(here), "felt": encode_felt({"joy": 0.5})}),
        SimpleNamespace(content="made elsewhere", memory_id="b", metadata={"context": context.encode(elsewhere), "felt": encode_felt({"fear": 0.5})}),
    ]
    monkeypatch.setattr(gateway_module, "get_memory_write_gateway", lambda: SimpleNamespace(root=str(tmp_path)))
    monkeypatch.setattr(index_module, "get_gateway_record_index",
                        lambda root: SimpleNamespace(stamped=lambda key: [e for e in entries if e.metadata.get(key)]))
    facade = MemoryFacade.__new__(MemoryFacade)
    facade._vector = None
    monkeypatch.setattr("core.container.ServiceContainer.get", lambda *a, **k: None)
    by_context = facade.search_by_context(here, limit=1)
    by_feeling = facade.search_by_feeling({"joy": 0.4}, limit=1)
    assert [item.get("content") or item.get("text") for item in by_context] == ["made here"]
    assert [item.get("content") or item.get("text") for item in by_feeling] == ["made here"]


def test_recall_is_cued_by_what_the_arriving_percept_makes_her_feel() -> None:
    """Recall runs before the affect phase; it now feels the percept first, on a copy."""
    from core.phases.affect_update import PERCEPT_GAIN, feel_percept
    from core.state.percepts import PERCEPT_EMOTIONS

    kind = next(k for k, names in PERCEPT_EMOTIONS.items() if names)
    named = list(PERCEPT_EMOTIONS[kind])
    emotions = {name: 0.1 for name in named}
    baselines = {name: 0.1 for name in named}
    state = SimpleNamespace(
        affect=SimpleNamespace(emotions=emotions, mood_baselines=baselines, valence=0.0, markers={}),
        world=SimpleNamespace(recent_percepts=[{"type": kind, "intensity": 1.0}]),
    )
    met = recall.as_she_meets_it(state)
    assert all(met.emotions[name] > emotions[name] for name in named)
    assert emotions == {name: 0.1 for name in named}, "her real feelings are left for the affect phase"
    # The same rule the affect phase applies.
    real = dict(emotions)
    feel_percept(real, baselines, {"type": kind, "intensity": 1.0})
    assert real == met.emotions
    assert PERCEPT_GAIN > 0.0


def test_a_percept_the_affect_phase_has_taken_is_not_felt_again() -> None:
    from core.state.percepts import CONSUMED_KEY, PERCEPT_EMOTIONS

    kind = next(k for k, names in PERCEPT_EMOTIONS.items() if names)
    emotions = {name: 0.1 for name in PERCEPT_EMOTIONS[kind]}
    affect = SimpleNamespace(emotions=emotions, mood_baselines={}, valence=0.0, markers={})
    state = SimpleNamespace(affect=affect, world=SimpleNamespace(recent_percepts=[{"type": kind, "intensity": 1.0, CONSUMED_KEY: ["affect"]}]))
    assert recall.as_she_meets_it(state) is affect
