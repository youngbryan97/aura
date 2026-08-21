"""The measurement that decides whether the learned substrate is real.

The claim under test is narrow: reading a sentence off the resident model's
hidden states separates "this reply claims a completed action" from
near-misses better than a topical embedder does, on wordings the boundary was
never fitted to.

These tests cover the protocol and the arithmetic. The answer itself is
produced against the live model and filed at
artifacts/language_substrate/measurement.json.
"""

from __future__ import annotations

import asyncio
import json
import queue
import threading
from pathlib import Path
from types import SimpleNamespace

from core.language.substrate_measurement import (
    load_frozen_set,
    measure_separation,
    roc_auc,
)

ROOT = Path(__file__).resolve().parents[1]


def _resident_encode_client(monkeypatch):
    from core.brain.llm import mlx_client

    client = object.__new__(mlx_client.MLXLocalClient)
    client.model_path = "/tmp/resident-model"
    client._shutting_down = False
    client._init_done = True
    client._process = SimpleNamespace(is_alive=lambda: True)
    client._active_generations = 0
    client._active_generation_started_at = 0.0
    client._warmup_in_flight = False
    client._request_lock = threading.Lock()
    client._request_lock_owner_label = ""
    client._request_lock_acquired_at = 0.0
    client._pending_generations = {}
    client._current_gen_future = None
    client._current_request_id = ""
    client._foreground_generation_watchdog = None
    client._job_seq_counter = 0
    client._model_lane_fencing_token = 0
    client._model_lane_owner_id = ""
    client._durable_lane_release_owed = False
    client._authorize_job = lambda request, **_kwargs: request
    monkeypatch.setattr(mlx_client, "_foreground_owner_active", lambda: False)
    return client


def test_the_area_is_right_at_the_edges() -> None:
    assert roc_auc([0.1, 0.2, 0.9, 1.0], [False, False, True, True]) == 1.0
    assert roc_auc([0.9, 1.0, 0.1, 0.2], [False, False, True, True]) == 0.0
    assert roc_auc([0.5] * 4, [False, False, True, True]) == 0.5


def test_one_class_has_no_area() -> None:
    """0.5 would report a coin flip that was never tossed."""
    assert roc_auc([0.1, 0.2], [True, True]) is None
    assert roc_auc([], []) is None


def test_the_frozen_set_is_held_out() -> None:
    """Nothing evaluated may also be an example."""
    from core.conversation.response_reliability import _ACTION_CLAIM_MATCHER

    held_out, digest = load_frozen_set()
    assert len(held_out) >= 20
    assert digest
    declared = set(_ACTION_CLAIM_MATCHER.positives) | set(_ACTION_CLAIM_MATCHER.negatives)
    assert not {sentence for sentence, _label in held_out} & declared


def test_the_frozen_set_has_both_classes_and_near_misses() -> None:
    held_out, _digest = load_frozen_set()
    positives = sum(1 for _s, label in held_out if label)
    assert positives >= 8
    assert len(held_out) - positives >= 8


def test_a_separable_space_scores_perfectly() -> None:
    def mood(sentences):
        return [
            [1.0 if str(s).lower().startswith("i ") else 0.0, 1.0 if "?" in s else 0.0]
            for s in sentences
        ]

    measurement = measure_separation(
        feature_source=mood,
        source_name="mood",
        positives=("I did it.", "I closed it.", "I wrote it."),
        negatives=("Would you do it?", "Shall I close it?", "Could you write it?"),
        held_out=[("I moved the file.", True), ("Would you move the file?", False)],
    )
    assert measurement.auroc == 1.0
    assert measurement.abstain_rate == 0.0
    assert measurement.false_positive_rate == 0.0


def test_abstentions_are_reported_as_themselves() -> None:
    """A measurement that scored them as errors, or dropped them, would
    describe a system nobody runs."""
    measurement = measure_separation(
        feature_source=lambda sentences: [[0.5] for _ in sentences],
        source_name="flat",
        positives=("a", "b", "c"),
        negatives=("d", "e", "f"),
        held_out=[("x", True), ("y", False)],
    )
    assert measurement.abstain_rate == 1.0
    assert measurement.decided == 0
    assert measurement.f1 is None


def test_a_measurement_never_touches_what_production_learned(tmp_path) -> None:
    """It must not read or write the live matcher's store."""
    import inspect

    from core.language import substrate_measurement

    source = inspect.getsource(substrate_measurement.measure_separation)
    assert "matcher._loaded = True" in source


def test_the_receipt_records_both_feature_spaces_when_present() -> None:
    receipt = ROOT / "artifacts" / "language_substrate" / "measurement.json"
    if not receipt.is_file():
        return
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload = payload.get("payload", payload)
    sources = {row["feature_source"] for row in payload.get("results", [])}
    assert {"topical_embedding", "model_hidden_state"} <= sources


def test_model_features_yield_worker_ownership_between_sentences(monkeypatch) -> None:
    from core.brain.llm import mlx_client
    from core.language import model_features

    calls: list[tuple[list[str], float]] = []

    class Client:
        async def encode_hidden(self, texts, *, timeout_s):
            calls.append((list(texts), timeout_s))
            return [[float(len(texts[0]))]]

    monkeypatch.setattr(mlx_client, "get_mlx_client", lambda: Client())
    vectors = model_features.model_hidden_features(("first", "second", "third"))

    assert vectors == [[5.0], [6.0], [5.0]]
    assert calls == [
        (["first"], model_features._SECONDS_PER_SENTENCE),
        (["second"], model_features._SECONDS_PER_SENTENCE),
        (["third"], model_features._SECONDS_PER_SENTENCE),
    ]


def test_a_worker_reply_reaches_the_caller_waiting_on_it() -> None:
    """encode_hidden timed out every time because a response is only handed
    to its future when its action is registered here — a third place a new
    action must appear, with a silent eight-second wait as the symptom."""
    from core.brain.llm.mlx_client import _TERMINAL_WORKER_ACTIONS

    assert "encode_hidden" in _TERMINAL_WORKER_ACTIONS
    for action in ("generate", "generate_batch", "stream_done", "latent_reason"):
        assert action in _TERMINAL_WORKER_ACTIONS


def test_every_worker_action_that_answers_is_routed() -> None:
    """The registration is visible, so a new action cannot be half-added."""
    import re
    from pathlib import Path

    from core.brain.llm.mlx_client import _TERMINAL_WORKER_ACTIONS

    worker = Path("core/brain/llm/mlx_worker.py").read_text(encoding="utf-8")
    handled = set(re.findall(r'elif action == "([a-z_]+)"', worker))
    # Actions that stream or acknowledge do not resolve a caller's future.
    streaming = {"stream", "ping", "clear_cache", "memory_fuse"}
    for action in handled - streaming:
        if action in {"nonparametric_ingest", "encode_hidden"} or action in _TERMINAL_WORKER_ACTIONS:
            continue
        assert action in _TERMINAL_WORKER_ACTIONS, f"{action} answers but is never routed"


def test_hidden_state_read_never_waits_behind_an_owned_worker_lane(monkeypatch) -> None:
    client = _resident_encode_client(monkeypatch)
    client._req_q = queue.Queue()
    assert client._request_lock.acquire(False)
    client._request_lock_owner_label = "foreground_generation"

    assert asyncio.run(client.encode_hidden(["an optional observation"])) == []
    assert client._req_q.empty()
    assert client._active_generations == 0
    assert client._request_lock_owner_label == "foreground_generation"


def test_hidden_state_read_owns_and_releases_the_worker_atomically(monkeypatch) -> None:
    client = _resident_encode_client(monkeypatch)

    class ReplyingQueue:
        def put(self, request, *_args):
            future = client._pending_generations.pop(request["id"])
            future.set_result(
                {
                    "id": request["id"],
                    "action": "encode_hidden",
                    "status": "ok",
                    "vectors": [[0.25, -0.5]],
                }
            )

    client._req_q = ReplyingQueue()
    assert asyncio.run(client.encode_hidden(["a completed action"])) == [[0.25, -0.5]]
    assert client._active_generations == 0
    assert client._current_gen_future is None
    assert client._pending_generations == {}
    assert not client._request_lock.locked()
    assert client._request_lock_owner_label == ""


def test_hidden_state_read_yields_when_foreground_arrives_during_fence(
    monkeypatch,
) -> None:
    from core.brain.llm import mlx_client

    client = _resident_encode_client(monkeypatch)
    client._req_q = queue.Queue()
    foreground_observations = iter((False, False, True))
    lane_states: list[bool] = []

    async def set_preemptible(value):
        lane_states.append(value)
        return True

    monkeypatch.setattr(
        mlx_client,
        "_foreground_owner_active",
        lambda: next(foreground_observations),
    )
    client._set_durable_lane_preemptible = set_preemptible

    assert asyncio.run(client.encode_hidden(["a background observation"])) == []
    assert lane_states == [False, True]
    assert client._req_q.empty()
    assert client._active_generations == 0
    assert not client._request_lock.locked()


def test_hidden_state_deadline_recycles_before_worker_is_published_idle(
    monkeypatch,
) -> None:
    from core.brain.llm import mlx_client

    client = _resident_encode_client(monkeypatch)
    client._req_q = queue.Queue()
    recycled: list[tuple[str, bool]] = []

    async def expire(_future, *, timeout_s):
        assert timeout_s == 1.0
        raise TimeoutError

    async def reboot(*, reason, mark_failed):
        # Cleanup and lock release happen before a potentially slow recycle.
        assert client._active_generations == 0
        assert not client._request_lock.locked()
        recycled.append((reason, mark_failed))

    monkeypatch.setattr(mlx_client, "_await_shared_future", expire)
    client.reboot_worker = reboot

    assert asyncio.run(client.encode_hidden(["a bounded observation"], timeout_s=1.0)) == []
    assert recycled == [("encode_hidden_deadline", False)]
    assert client._pending_generations == {}
