from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

from core.brain.llm import endogenous_pair_recorder as recorder
from core.brain.llm.endogenous_state import STATE_DIM, EndogenousState


def _state() -> EndogenousState:
    return EndogenousState(
        values=np.linspace(-0.5, 0.5, STATE_DIM, dtype=np.float32),
        present=np.ones(STATE_DIM, dtype=bool),
        sources=("test",),
    )


def test_async_recording_uses_the_same_bounded_rotation_transaction(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("AURA_ENDOGENOUS_PAIR_DIR", str(tmp_path))
    monkeypatch.setattr(recorder, "MAX_FILE_BYTES", 1)
    monkeypatch.setattr(recorder, "MAX_ROLLED_FILES", 2)

    assert asyncio.run(recorder.record_pair_async(_state(), "first", lane="chat"))
    assert asyncio.run(recorder.record_pair_async(_state(), "second", lane="chat"))
    assert asyncio.run(recorder.record_pair_async(_state(), "third", lane="chat"))

    active = tmp_path / "pairs.jsonl"
    rolled = sorted(tmp_path.glob("pairs-*.jsonl"))
    assert active.exists()
    assert len(rolled) == 2
    assert "third" in active.read_text(encoding="utf-8")
    assert any("first" in path.read_text(encoding="utf-8") for path in rolled)
    assert any("second" in path.read_text(encoding="utf-8") for path in rolled)


def test_sync_recording_refuses_an_event_loop_before_touching_storage(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("AURA_ENDOGENOUS_PAIR_DIR", str(tmp_path))

    async def exercise() -> None:
        with pytest.raises(RuntimeError, match="use record_pair_async"):
            recorder.record_pair(_state(), "must not block the loop")

    asyncio.run(exercise())
    assert not (tmp_path / "pairs.jsonl").exists()


def test_async_response_claims_pending_state_and_writes_off_loop(tmp_path, monkeypatch):
    monkeypatch.setenv("AURA_ENDOGENOUS_PAIR_DIR", str(tmp_path))
    recorder.reset_pending()
    recorder.remember_pending("request-1", _state().to_payload(), lane="chat")

    assert asyncio.run(recorder.record_response_async("request-1", "the response"))

    assert recorder.pending_depth() == 0
    assert [pair.text for pair in recorder.iter_pairs()] == ["the response"]


def test_rotation_accounts_for_the_record_about_to_be_appended(tmp_path, monkeypatch):
    monkeypatch.setenv("AURA_ENDOGENOUS_PAIR_DIR", str(tmp_path))
    first = "a"
    assert recorder.record_pair(_state(), first)
    active = tmp_path / "pairs.jsonl"
    first_size = active.stat().st_size
    monkeypatch.setattr(recorder, "MAX_FILE_BYTES", first_size + 1)

    assert recorder.record_pair(_state(), "b")

    rolled = list(tmp_path.glob("pairs-*.jsonl"))
    assert len(rolled) == 1
    assert rolled[0].stat().st_size == first_size
    assert active.stat().st_size <= recorder.MAX_FILE_BYTES or active.stat().st_size == first_size


def test_rotation_generation_names_do_not_replace_each_other(tmp_path, monkeypatch):
    monkeypatch.setenv("AURA_ENDOGENOUS_PAIR_DIR", str(tmp_path))
    monkeypatch.setattr(recorder, "MAX_FILE_BYTES", 1)
    monkeypatch.setattr(recorder, "MAX_ROLLED_FILES", 8)
    stamps = iter(("20260824-010101-101", "20260824-010101-102"))
    monkeypatch.setattr(recorder, "_rotation_stamp", stamps.__next__)

    assert recorder.record_pair(_state(), "first")
    assert recorder.record_pair(_state(), "second")
    assert recorder.record_pair(_state(), "third")

    names = {path.name for path in Path(tmp_path).glob("pairs-*.jsonl")}
    assert len(names) == 2
    assert any(name.endswith("-101.jsonl") for name in names)
    assert any(name.endswith("-102.jsonl") for name in names)
