"""Three loop-thread stalls from one afternoon's stall dumps, each fixed at its cause.

data/error_logs/stalls on 2026-09-15 named the loop thread's frame at the
moment of each stall:

- 5.2s in ``wake_word._get_latest_transcript`` inside ``Path.resolve()`` — a
  realpath walk, five times a second, on the loop.
- 5.6s in ``liquid_substrate.encode_text_to_stimulus`` — the 512x260 random
  projection redrawn on every broadcast winner.
- 5.4s in ``earned_metric.recurrence_verdict`` — 1,300 small NumPy calls per
  verdict, each a GIL round trip against busy worker threads.
"""
from __future__ import annotations

import asyncio
import threading
import re
from pathlib import Path

import numpy as np

from core.consciousness import liquid_substrate
from core.verify import earned_metric

ROOT = Path(__file__).resolve().parent.parent


def test_the_wake_word_poll_resolves_its_sidecar_once_and_reads_it_off_the_loop():
    source = (ROOT / "core/voice/wake_word.py").read_text(encoding="utf-8")
    body = source[source.index("def _get_latest_transcript") :]
    body = body[: body.index("\n    def ", 10)]
    assert "resolve()" not in body
    assert "_AUDIO_SIDECAR" in body
    assert re.search(r"^_AUDIO_SIDECAR = Path\(__file__\)\.resolve\(\)", source, re.M)
    loop = source[source.index("async def _detection_loop") : source.index("def _get_latest_transcript")]
    assert "await asyncio.to_thread(self._get_latest_transcript)" in loop


def test_the_text_projection_is_drawn_once_per_substrate_size():
    liquid_substrate._TEXT_PROJECTIONS.clear()
    first = liquid_substrate._text_projection(64)
    second = liquid_substrate._text_projection(64)
    assert first is second
    assert first.shape == (64, 260)
    # Same draw as before the cache: seeded by the neuron count.
    expected = np.random.RandomState(64).randn(64, 260).astype(np.float32) * (1.0 / np.sqrt(260))
    np.testing.assert_array_equal(first, expected)


def test_the_vectorised_null_is_the_loop_null():
    rng = np.random.default_rng(7)
    states = rng.normal(size=(20, 32))
    unit = states / np.linalg.norm(states, axis=1)[:, None]
    gram = unit @ unit.T
    orders = np.stack([np.random.default_rng(i).permutation(20) for i in range(16)])

    by_loop = np.array(
        [
            float(np.max(earned_metric._lag_profile_from_gram(gram[np.ix_(o, o)])))
            for o in orders
        ]
    )
    np.testing.assert_allclose(earned_metric._null_maxima(gram, orders), by_loop, rtol=0, atol=1e-12)


def test_the_verdict_takes_few_array_calls(monkeypatch):
    calls = {"diagonal": 0}
    real = np.diagonal

    def counting(*args, **kwargs):
        calls["diagonal"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(earned_metric.np, "diagonal", counting)
    history = [np.random.default_rng(i).normal(size=16) for i in range(20)]
    verdict = earned_metric.recurrence_verdict(history, percentile=95.0, surrogates=128)
    assert verdict is not None
    # One observed profile (10 lags) plus one call per lag across all surrogates.
    assert calls["diagonal"] == 20


def test_the_process_tree_walk_never_runs_on_the_loop(monkeypatch):
    """7.3s stall, 2026-09-15 15:48: the lane reconciler asked for the host
    total and paid for a psutil walk of every pid on the host, on the loop."""
    from core.runtime import resource_observation as ro

    observer = ro.HostResourceObserver()
    walks: list[str] = []

    class _Process:
        def children(self, recursive=True):
            walks.append(threading.current_thread().name)
            return []

    async def on_loop():
        first = observer._children_rss_bytes(_Process(), 4242)
        # No reading yet: the loop gets 0 now and a thread does the walk.
        assert first == 0
        for _ in range(50):
            if 4242 in observer._tree_children_rss_cache:
                break
            await asyncio.sleep(0.02)
        assert 4242 in observer._tree_children_rss_cache

    asyncio.run(on_loop())
    assert walks and all(name != "MainThread" for name in walks)
    assert all(name.startswith("children-rss-") for name in walks)

    # Off the loop the walk is inline, as before.
    walks.clear()
    observer._tree_children_rss_cache.clear()
    observer._children_rss_bytes(_Process(), 4242)
    assert walks == [threading.current_thread().name]


def test_lane_admission_reads_only_the_host_total():
    source = (ROOT / "core/brain/lane_admission.py").read_text(encoding="utf-8")
    body = source[source.index("def _host_total_gb") :]
    body = body[: body.index("\ndef ", 10)]
    assert ".memory(include_process_tree=False)" in body


def test_a_receipt_opened_on_the_loop_is_written_by_the_writer_thread(tmp_path):
    """5.5s stall, 2026-09-15 15:42: initiative arbiter -> preference learner
    -> OutcomeLedger.open -> _persist -> sqlite INSERT on the loop thread."""
    from core.cognition.outcome_ledger import OutcomeLedger

    ledger = OutcomeLedger(db_path=str(tmp_path / "ledger.db"))
    writers: list[str] = []
    real = ledger._write_rows

    def spy(rows):
        writers.append(threading.current_thread().name)
        real(rows)

    ledger._write_rows = spy

    async def on_loop():
        rid = ledger.open("act", 0.7, category="deliberation", horizon_s=60.0)
        assert threading.current_thread().name not in writers
        return rid

    rid = asyncio.run(on_loop())
    assert ledger._flush_writes(timeout_s=5.0)
    assert writers == ["outcome-ledger-writer"]

    # A reader sees the row: the reader flushes before it reads.
    ledger.resolve(rid, 0.9)
    assert ledger.measured_action_stats()["act"]["n"] == 1.0
    assert writers[0] == "outcome-ledger-writer"
    # The resolve above ran off the loop: inline, on this thread, after the queue.
    assert writers[-1] == threading.current_thread().name


def test_the_reference_store_guard_is_an_existence_check():
    """5.7s stall, 2026-09-15 15:40: wire_default_stores counted the whole
    corpus to decide whether to wire it."""
    source = (ROOT / "core/memory/intentional_retrieval.py").read_text(encoding="utf-8")
    body = source[source.index("def wire_default_stores") :]
    assert "corpus.has_documents()" in body
    assert "corpus.document_count()" not in body


def test_the_mapped_files_snapshot_is_copied_once_per_published_map():
    """5.5s stall, 2026-09-15 15:50: the vault sync worker copied every
    module entry under the class lock while the MLX listener waited for it
    on the loop thread."""
    from core.mycelium import MycelialNetwork

    net = MycelialNetwork.__new__(MycelialNetwork)
    net.mapped_files = {f"m{i}": {"path": f"m{i}.py", "imports": ["a", "b"]} for i in range(50)}
    net._mapped_files_snapshot_cache = None

    first = net._mapped_files_canonical_locked()
    second = net._mapped_files_canonical_locked()
    assert first is second  # one copy per published map
    assert first["m1"] is not net.mapped_files["m1"]

    # What readers are handed is detached from the shared copy.
    handed = net._mapped_files_snapshot_locked()
    handed["m1"]["imports"].append("x")
    assert net._mapped_files_canonical_locked()["m1"]["imports"] == ["a", "b"]

    net.mapped_files = {"n": {"path": "n.py", "imports": []}}
    assert list(net._mapped_files_canonical_locked()) == ["n"]


def test_the_listener_pulses_the_mycelium_from_a_thread():
    source = (ROOT / "core/brain/llm/mlx_client.py").read_text(encoding="utf-8")
    assert "await run_io_bound(self._pulse_mycelial_worker, res)" in source
    assert "\n                        self._pulse_mycelial_worker(res)\n" not in source
