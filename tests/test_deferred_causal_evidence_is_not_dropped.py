"""A deferral was read as a refusal, and 108 pieces of learning data vanished.

Live 2026-08-17: 108 "🚫 ACG write blocked: sync_approved|ontogeny:deferred"
in one sampled window. The reason string is the whole story — the executive's
own rules APPROVED the write, and the learned ontogeny organ then overrode the
outcome to DEFERRED, producing that combined reason.

acg.record_outcome saw approved == False and returned, dropping the entry. But
DEFERRED is not REJECTED. The action-consequence graph is how she learns what
her own actions actually do, so every dropped entry is an action whose outcome
she never gets to generalise from — and because the organ deferring is a
POLICY, it defers again under the same conditions, making the loss systematic
rather than random.

Deferred writes are now held and retried. A genuine refusal is still refused:
the point is to tell "not now" from "no", not to get past the gate.
"""
from __future__ import annotations

import tempfile

import pytest

import core.constitution as constitution
import core.world_model.acg as acg_module
from core.world_model.acg import (
    _PENDING_LIMIT,
    _PENDING_REPLAY_LIMIT,
    _PENDING_RETRY_INTERVAL_S,
    ActionConsequenceGraph,
    _is_deferral,
)


@pytest.fixture
def graph(monkeypatch):
    verdict = {"approved": True, "reason": "sync_approved"}
    clock = {"now": 100.0}

    class _Core:
        def approve_memory_write_sync(self, **_kw):
            return verdict["approved"], verdict["reason"]

    monkeypatch.setattr(constitution, "get_constitutional_core", lambda: _Core())
    monkeypatch.setattr(acg_module.time, "monotonic", lambda: clock["now"])
    g = ActionConsequenceGraph(persist_path=tempfile.mktemp(suffix=".json"))
    g.verdict = verdict  # type: ignore[attr-defined]
    g.clock = clock  # type: ignore[attr-defined]
    return g


def _record(g, tool="click"):
    g.record_outcome({"tool": tool, "params": {}}, "ctx", "outcome", True)


@pytest.mark.parametrize(
    ("reason", "deferral"),
    [
        ("sync_approved|ontogeny:deferred", True),
        ("capacity_full_8/8", True),
        ("backpressure", True),
        ("identity_assertion_failed:tool", False),
        ("rejected_by_policy", False),
        ("", False),
    ],
)
def test_a_deferral_is_told_apart_from_a_refusal(reason, deferral):
    assert _is_deferral(reason) is deferral


def test_a_deferred_write_is_held_not_dropped(graph):
    graph.verdict.update(approved=False, reason="sync_approved|ontogeny:deferred")

    _record(graph)

    assert graph.links == []
    assert len(graph._pending) == 1
    assert graph._deferred_total == 1


def test_a_held_write_lands_once_the_organ_stops_deferring(graph):
    graph.verdict.update(approved=False, reason="sync_approved|ontogeny:deferred")
    _record(graph, "click")

    graph.verdict.update(approved=True, reason="sync_approved")
    graph.clock["now"] += _PENDING_RETRY_INTERVAL_S
    _record(graph, "type")

    assert len(graph.links) == 2, "the held entry must be replayed, not lost"
    assert graph._pending == graph._pending.__class__(maxlen=_PENDING_LIMIT)
    assert graph._replayed_total == 1


def test_a_genuine_refusal_is_still_refused(graph):
    graph.verdict.update(approved=False, reason="identity_assertion_failed:tool")

    _record(graph)

    assert graph.links == []
    assert len(graph._pending) == 0, "a refusal must not be queued for retry"


def test_a_write_deferred_again_stays_held(graph):
    graph.verdict.update(approved=False, reason="sync_approved|ontogeny:deferred")
    _record(graph, "one")
    graph.clock["now"] += _PENDING_RETRY_INTERVAL_S
    _record(graph, "two")

    assert graph.links == []
    assert len(graph._pending) == 2
    assert graph._replayed_total == 0


def test_the_queue_is_bounded(graph):
    """A queue that grows while the organ keeps deferring is a leak."""
    graph.verdict.update(approved=False, reason="sync_approved|ontogeny:deferred")

    for index in range(_PENDING_LIMIT + 40):
        _record(graph, f"tool{index}")

    assert len(graph._pending) <= _PENDING_LIMIT


def test_replay_cannot_recurse(graph):
    """Replay calls record_outcome, which triggers replay. Once only."""
    graph.verdict.update(approved=False, reason="sync_approved|ontogeny:deferred")
    _record(graph, "held")

    graph.verdict.update(approved=True, reason="sync_approved")
    graph.clock["now"] += _PENDING_RETRY_INTERVAL_S
    _record(graph, "new")

    assert graph._replaying is False
    assert len(graph.links) == 2


def test_unchanged_defer_policy_does_not_replay_the_whole_queue(graph):
    graph.verdict.update(approved=False, reason="sync_approved|ontogeny:deferred")
    for index in range(12):
        _record(graph, f"held-{index}")

    assert len(graph._pending) == 12
    deferred_before = graph._deferred_total

    # A new action inside the retry window is offered once; the twelve held
    # writes are not all re-submitted to an unchanged policy.
    _record(graph, "new-within-window")
    assert graph._deferred_total == deferred_before + 1
    assert len(graph._pending) == 13

    # Once the window expires, replay is still bounded per event.
    graph.clock["now"] += _PENDING_RETRY_INTERVAL_S
    _record(graph, "new-after-window")
    assert graph._deferred_total == deferred_before + 2 + _PENDING_REPLAY_LIMIT
