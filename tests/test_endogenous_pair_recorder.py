from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import numpy as np
import pytest

from core.brain.llm import endogenous_pair_recorder as recorder
from core.brain.llm.endogenous_state import STATE_DIM, EndogenousState
from tests.clock_patch import patch_module_clock


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
    patch_module_clock(monkeypatch, recorder, time=lambda: 1788590347.125)
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


def test_terminal_observers_run_off_loop_in_receipt_pair_absorption_order(monkeypatch):
    from core.brain.llm import endogenous_client_hooks as hooks

    loop_thread = threading.get_ident()
    calls = []

    def observe(_response):
        assert threading.get_ident() != loop_thread
        calls.append("receipt")

    async def pair(_response):
        calls.append("pair")

    def absorb(_response):
        assert threading.get_ident() != loop_thread
        calls.append("absorption")

    monkeypatch.setattr(hooks, "observe_endogenous_receipt", observe)
    monkeypatch.setattr(hooks, "record_endogenous_pair", pair)
    monkeypatch.setattr(hooks, "absorb_endogenous_outcome", absorb)
    asyncio.run(hooks.process_endogenous_terminal_response({"id": "one"}))
    assert calls == ["receipt", "pair", "absorption"]


def test_each_pair_records_the_prompt_it_answered(tmp_path, monkeypatch):
    # 18 October: the prompt was never held, so all 9,338 pairs of the persona 27B shared one hash.
    import json

    monkeypatch.setenv("AURA_ENDOGENOUS_PAIR_DIR", str(tmp_path))
    recorder.reset_pending()
    recorder.remember_pending("a", _state().to_payload(), lane="chat", prompt="what is the capital of Peru")
    recorder.remember_pending("b", _state().to_payload(), lane="chat", prompt="how are you today")
    assert asyncio.run(recorder.record_response_async("a", "Lima."))
    assert asyncio.run(recorder.record_response_async("b", "Well, thanks."))
    hashes = [json.loads(line)["prompt_sha"] for line in (tmp_path / "pairs.jsonl").read_text().splitlines()]
    assert len(set(hashes)) == 2


def test_each_pair_records_the_state_its_generation_was_steered_by(tmp_path, monkeypatch):
    # 18 October: "z" holds the affect engine's readings, not the slots a steering vector is scaled by,
    # and directions fitted on it came out one axis for four dimensions.
    import json

    monkeypatch.setenv("AURA_ENDOGENOUS_PAIR_DIR", str(tmp_path))
    recorder.reset_pending()
    recorder.remember_pending("a", _state().to_payload(), lane="chat", steering=[0.61, 0.42, 0.0, 0.07])
    recorder.remember_pending("b", _state().to_payload(), lane="chat")
    assert asyncio.run(recorder.record_response_async("a", "Steered."))
    assert asyncio.run(recorder.record_response_async("b", "No substrate to read."))
    rows = [json.loads(line) for line in (tmp_path / "pairs.jsonl").read_text().splitlines()]
    assert rows[0]["s"] == [0.61, 0.42, 0.0, 0.07]
    assert "s" not in rows[1]


def test_the_job_hook_holds_the_steering_state_beside_the_endogenous_state(monkeypatch):
    from core.brain.llm import endogenous_client_hooks as hooks

    held = {}
    monkeypatch.setattr(hooks, "_steering_now", lambda: [0.5, 0.25])
    monkeypatch.setattr(recorder, "remember_pending", lambda *a, **k: held.update(k))
    monkeypatch.setattr("core.brain.llm.endogenous_vocab_head.alpha_from_env", lambda: 0.5)
    hooks.attach_endogenous_state({"id": "r1", "messages": [{"role": "user", "content": "hi"}]},
                                  override=_state().to_payload())
    assert held["steering"] == [0.5, 0.25]


def test_each_pair_records_what_she_felt_beside_the_steering_state(tmp_path, monkeypatch):
    # 22 October: the steering state has no discrete emotion, so a capture could
    # only ever find directions for curiosity and frustration. See core/affect/appraisal.py.
    import json

    monkeypatch.setenv("AURA_ENDOGENOUS_PAIR_DIR", str(tmp_path))
    recorder.reset_pending()
    recorder.remember_pending("a", _state().to_payload(), lane="chat", felt={"anger": 0.41, "upset": 0.2})
    recorder.remember_pending("b", _state().to_payload(), lane="chat")
    assert asyncio.run(recorder.record_response_async("a", "That was not fair."))
    assert asyncio.run(recorder.record_response_async("b", "Nothing risen."))
    rows = [json.loads(line) for line in (tmp_path / "pairs.jsonl").read_text().splitlines()]
    assert rows[0]["felt"] == {"anger": 0.41, "upset": 0.2}
    assert "felt" not in rows[1]


def test_the_job_hook_reads_her_feelings_from_her_live_state(monkeypatch):
    from types import SimpleNamespace

    from core.brain.llm import endogenous_client_hooks as hooks

    affect = SimpleNamespace(emotions={"anger": 0.5, "joy": 0.1}, mood_baselines={"anger": 0.1, "joy": 0.2})
    repo = SimpleNamespace(_current=SimpleNamespace(affect=affect))
    monkeypatch.setattr("core.runtime.service_access.resolve_state_repository", lambda default=None: repo)
    assert hooks._felt_now() == {"anger": pytest.approx(0.4)}
