"""Only advancing counters owned by the request renew its progress clock."""

import pytest

from core.brain.llm import mlx_client
from core.brain.llm.mlx_client import MLXLocalClient
from core.runtime import turn_progress


@pytest.fixture
def progress_client(monkeypatch):
    client = MLXLocalClient(model_path="/models/test-small")
    client._current_request_id = "active"
    client._current_turn_progress = turn_progress.capture_progress()
    client._current_first_token_at = 90.0
    client._last_token_progress_at = 100.0
    client._tokens_this_request = 16
    client._tokens_since_spawn = 16
    client._mark_progress = lambda: None
    monkeypatch.setattr(mlx_client.time, "time", lambda: 102.0)
    monkeypatch.setitem(mlx_client._HOST_RATES, "decode", 0.0)
    renewals = []
    monkeypatch.setattr(turn_progress, "note_progress", lambda **kw: renewals.append(True))
    return client, renewals


def test_batched_progress_measures_tokens_not_messages(progress_client):
    client, renewals = progress_client
    client._record_worker_stream_progress(
        {"id": "active", "tokens_generated": 32}, status="progress", action="generate"
    )
    assert client._tokens_this_request == 32
    assert client._tokens_since_spawn == 32
    assert mlx_client._HOST_RATES["decode"] == 8.0
    assert client._last_token_progress_at == 102.0
    assert renewals == [True]


@pytest.mark.parametrize("count", [0, 8, 16, -1, True, "32", 32.5])
def test_non_advancing_or_invalid_counter_cannot_renew(progress_client, count):
    client, renewals = progress_client
    client._record_worker_stream_progress(
        {"id": "active", "tokens_generated": count}, status="progress", action="generate"
    )
    assert client._tokens_this_request == 16
    assert client._last_token_progress_at == 100.0
    assert not renewals


@pytest.mark.parametrize("request_id", [None, "retired"])
@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_unowned_frame_cannot_renew_the_turn(progress_client, request_id, phase):
    client, renewals = progress_client
    client._record_worker_stream_progress(
        {"id": request_id, "phase": phase, "tokens_generated": 32,
         "prompt_tokens_processed": 128, "prompt_tokens_total": 512},
        status="progress", action="generate",
    )
    assert client._last_token_progress_at == 100.0
    assert not renewals


def test_repeated_prefill_frame_is_not_progress(progress_client):
    client, renewals = progress_client
    frame = {"id": "active", "phase": "prefill",
             "prompt_tokens_processed": 128, "prompt_tokens_total": 512}
    for _ in range(2):
        client._record_worker_stream_progress(frame, status="progress", action="generate")
    assert renewals == [True]


def test_legacy_visible_token_still_advances(progress_client):
    client, renewals = progress_client
    client._record_worker_stream_progress(
        {"id": "active", "text": "hi"}, status="token", action="stream"
    )
    assert client._tokens_this_request == 17
    assert renewals == [True]
