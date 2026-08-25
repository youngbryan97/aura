"""The corpus is what happened, or it measures nothing.

Two ways a training corpus quietly stops being evidence: an experimental state
gets folded in as though the runtime held it, and records written against an
older channel layout get fitted alongside current ones, so one matrix is fitted
to two different meanings of the same column. Both are refused here.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from core.brain.llm.endogenous_pair_recorder import (
    corpus_summary,
    iter_pairs,
    pending_depth,
    record_pair,
    record_response,
    recording_enabled,
    remember_pending,
    reset_pending,
    store_directory,
)
from core.brain.llm.endogenous_state import (
    assemble_state,
    empty_state,
    layout_digest,
)


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path, monkeypatch):
    monkeypatch.setenv("AURA_ENDOGENOUS_PAIR_DIR", str(tmp_path / "pairs"))
    monkeypatch.setenv("AURA_LOG_DIR", str(tmp_path / "logs"))
    reset_pending()
    yield
    reset_pending()


def _live_state():
    """A state read from probes, so it carries no intervention marks."""
    return assemble_state(
        probes={
            "uncertainty": lambda: {"uncertainty.confidence": 0.6},
            "goal": lambda: {"goal.active": 1.0, "goal.priority": 0.4},
        }
    )


def test_a_recorded_turn_reads_back():
    assert record_pair(_live_state(), "a real reply", lane="chat", model="/m/qwen") is True
    pairs = list(iter_pairs())
    assert len(pairs) == 1
    assert pairs[0].text == "a real reply"
    assert pairs[0].lane == "chat"
    assert pairs[0].model == "qwen"
    assert pairs[0].coverage > 0.0
    assert np.all(np.isfinite(pairs[0].values))


def test_an_intervened_state_is_never_recorded():
    """An experiment is a condition the runtime never actually held."""
    from core.brain.llm.endogenous_state import EndogenousState

    live = _live_state().do(**{"uncertainty.confidence": 0.95})
    restored = EndogenousState.from_payload(live.to_payload())
    assert restored is not None and restored.interventions
    assert record_pair(restored, "a reply under an intervention") is False
    assert list(iter_pairs()) == []


def test_a_state_nothing_answered_for_is_never_recorded():
    assert record_pair(empty_state(), "a reply") is False


def test_an_empty_reply_is_never_recorded():
    assert record_pair(_live_state(), "   ") is False


def test_records_from_another_layout_are_skipped():
    assert record_pair(_live_state(), "a reply") is True
    path = store_directory() / "pairs.jsonl"
    payload = json.loads(path.read_text().strip())
    payload["layout"] = "0" * 32
    path.write_text(json.dumps(payload) + "\n")
    assert list(iter_pairs()) == []
    summary = corpus_summary()
    assert summary["total_records"] == 1
    assert summary["usable_records"] == 0
    assert summary["layout"] == layout_digest()


def test_a_malformed_line_does_not_stop_the_read():
    assert record_pair(_live_state(), "first") is True
    path = store_directory() / "pairs.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write("{not json\n")
    assert record_pair(_live_state(), "second") is True
    assert [p.text for p in iter_pairs()] == ["first", "second"]


def test_the_request_and_its_reply_are_paired_by_id():
    state = _live_state()
    # The pending map holds the payload, which is what crosses to the worker.
    remember_pending("req-1", state.to_payload(), lane="chat", model="/m/qwen")
    assert pending_depth() == 1
    assert record_response("req-1", "the reply") is True
    assert pending_depth() == 0
    stored = list(iter_pairs())
    assert [p.text for p in stored] == ["the reply"]
    assert stored[0].model == "qwen"


def test_a_reply_to_an_experimental_request_is_not_paired_into_the_corpus():
    """The pairing mechanism does not get to launder an intervention."""
    experimental = _live_state().do(**{"uncertainty.confidence": 0.95})
    remember_pending("req-2", experimental.to_payload())
    assert record_response("req-2", "a reply under an intervention") is False
    assert list(iter_pairs()) == []


def test_an_unmatched_reply_writes_nothing():
    assert record_response("never-seen", "text") is False


def test_the_pending_map_is_bounded():
    for i in range(200):
        remember_pending(f"req-{i}", _live_state().to_payload())
    assert pending_depth() <= 64


def test_recording_can_be_turned_off(monkeypatch):
    monkeypatch.setenv("AURA_ENDOGENOUS_RECORD", "0")
    assert recording_enabled() is False
    assert record_pair(_live_state(), "a reply") is False


def test_the_text_is_bounded():
    from core.brain.llm.endogenous_pair_recorder import MAX_TEXT_CHARS

    assert record_pair(_live_state(), "x" * (MAX_TEXT_CHARS * 2)) is True
    assert len(list(iter_pairs())[0].text) == MAX_TEXT_CHARS
