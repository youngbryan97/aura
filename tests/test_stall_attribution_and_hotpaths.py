"""Contracts for truthful stall attribution + the on-loop hot paths it exposed.

The old triage/narrator parsers guessed the culprit from thread order and
routinely blamed parked bystander threads: 19 stalls were fingerprinted to a
``time.sleep`` in flagship_doctor while the real culprit was synchronous
SQLite ON the event loop (local_corpus.document_count via the research
pipeline), and 9 more were pinned on an executor-thread JSONL read while the
loop ran goal_engine._fetch_records. Misattribution sends engineers — and
Aura's own narrator — chasing ghosts, so attribution now has one shared,
loop-thread-aware implementation, and the REAL culprits it exposed are fixed
and pinned here.
"""
from __future__ import annotations

import threading
import time

import pytest

from core.observability.stall_dump import parse_stall_dump_text

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Synthetic dump builders — replay the real July 8 anatomies.
# ---------------------------------------------------------------------------

_SLEEPING_MONITOR = '''
Thread ID: 111
  File "/x/lib/python3.12/threading.py", line 1032, in _bootstrap
    self._bootstrap_inner()
  File "/repo/core/runtime/runtime_hygiene.py", line 540, in _wrapped_run
    return original_run(*args, **kwargs)
  File "/repo/core/runtime/flagship_doctor.py", line 693, in _monitor_loop
    time.sleep(self.check_interval)
'''

_EXECUTOR_JSONL_READER = '''
Thread ID: 222
  File "/x/lib/python3.12/threading.py", line 1032, in _bootstrap
    self._bootstrap_inner()
  File "/x/lib/python3.12/concurrent/futures/thread.py", line 58, in run
    result = self.fn(*self.args, **self.kwargs)
  File "/repo/core/consciousness/crsm_loop_monitor.py", line 129, in _jsonl_file_state
    for raw in fh:
'''

_LOOP_RUNNING_CALLBACK = '''
Thread ID: 333
  File "/repo/aura_main.py", line 3702, in <module>
    main()
  File "/x/lib/python3.12/asyncio/base_events.py", line 645, in run_forever
    self._run_once()
  File "/x/lib/python3.12/asyncio/base_events.py", line 1999, in _run_once
    handle._run()
  File "/x/lib/python3.12/asyncio/events.py", line 88, in _run
    self._context.run(self._callback, *self._args)
  File "/repo/core/search/research_pipeline.py", line 1416, in _retain_artifact
    if _corpus.document_count() > 0:
  File "/repo/core/knowledge/local_corpus.py", line 299, in document_count
    row = conn.execute("SELECT COUNT(*) FROM docs").fetchone()
'''

_LOOP_PARKED_IN_SELECTOR = '''
Thread ID: 333
  File "/repo/aura_main.py", line 3702, in <module>
    main()
  File "/x/lib/python3.12/asyncio/base_events.py", line 645, in run_forever
    self._run_once()
  File "/x/lib/python3.12/asyncio/base_events.py", line 1961, in _run_once
    event_list = self._selector.select(timeout)
  File "/x/lib/python3.12/selectors.py", line 566, in select
    kev_list = self._selector.control(None, max_ev, timeout)
'''

_GIL_HUNGRY_GLOB = '''
Thread ID: 444
  File "/x/lib/python3.12/threading.py", line 1032, in _bootstrap
    self._bootstrap_inner()
  File "/repo/core/memory/gateway_record_index.py", line 126, in _do_refresh
    files = self._list_record_files()[: self.MAX_ENTRIES]
  File "/x/lib/python3.12/pathlib.py", line 1094, in glob
    for p in selector.select_from(self)
'''


def _dump(*sections: str, header: str = "STALL DETECTED: 6.2s\n", extra_header: str = "") -> str:
    return header + extra_header + "=" * 40 + "\n" + "".join(sections)


# ---------------------------------------------------------------------------
# Attribution: the loop thread's work wins; bystanders never do.
# ---------------------------------------------------------------------------

class TestStallAttribution:
    def test_replays_the_flagship_misattribution_correctly(self):
        """The July 8 anatomy: sleeping monitor listed FIRST, real on-loop
        SQLite last. The old parser blamed the sleeper 19 times."""
        verdict = parse_stall_dump_text(
            _dump(_SLEEPING_MONITOR, _EXECUTOR_JSONL_READER, _LOOP_RUNNING_CALLBACK)
        )
        assert verdict.fingerprint_frame() == "local_corpus.py:document_count"
        assert verdict.thread_kind == "event_loop"
        assert verdict.elapsed_s == pytest.approx(6.2)

    def test_stamped_loop_thread_beats_heuristics(self):
        text = _dump(
            _LOOP_RUNNING_CALLBACK.replace("Thread ID: 333", "Thread ID: 333  [EVENT LOOP]"),
            _GIL_HUNGRY_GLOB,
        )
        verdict = parse_stall_dump_text(text)
        assert verdict.thread_kind == "event_loop"
        assert verdict.fingerprint_frame() == "local_corpus.py:document_count"

    def test_loop_thread_header_stamp_is_honored(self):
        text = _dump(
            _SLEEPING_MONITOR,
            _LOOP_RUNNING_CALLBACK,
            extra_header="LOOP THREAD: 333\n",
        )
        verdict = parse_stall_dump_text(text)
        assert verdict.thread_kind == "event_loop"
        assert verdict.function == "document_count"
        assert verdict.line == 299

    def test_parked_loop_falls_through_to_gil_suspect(self):
        """A GIL-starved loop often snapshots inside its selector; the busy
        background thread holding the GIL is then the honest culprit."""
        verdict = parse_stall_dump_text(
            _dump(_LOOP_PARKED_IN_SELECTOR, _SLEEPING_MONITOR, _GIL_HUNGRY_GLOB)
        )
        assert verdict.thread_kind == "gil_suspect"
        assert verdict.fingerprint_frame() == "gateway_record_index.py:_do_refresh"

    def test_all_idle_dump_names_the_native_gil_anatomy(self):
        """Every Python thread idle during a stall is not 'unknown' — the
        GIL was held where tracebacks cannot see (MLX/Metal, C extensions).
        Name the anatomy so triage ranks it as its own class."""
        verdict = parse_stall_dump_text(_dump(_SLEEPING_MONITOR))
        assert verdict.thread_kind == "all_idle"
        assert not verdict.known
        assert verdict.fingerprint_frame() == "all_threads_idle:native_gil_suspect"
        assert "native code" in verdict.described()

    def test_busy_loop_in_foreign_code_names_the_foreign_frame(self):
        """A loop stalled inside exec'd <string> or third-party code carries
        that identity — 17 live dumps and every test-driver dump used to
        collapse into unknown_frame."""
        foreign_loop = (
            "Thread ID: 999 [EVENT LOOP]\n"
            '  File "/usr/lib/python3.12/asyncio/events.py", line 88, in _run\n'
            "    self._context.run(self._callback, *self._args)\n"
            '  File "<string>", line 13, in main\n'
            "    do_work()\n"
        )
        verdict = parse_stall_dump_text(
            "STALL DETECTED: 5.0s\n" + foreign_loop
        )
        assert verdict.thread_kind == "event_loop_foreign"
        assert verdict.fingerprint_frame() == "<string>:main"

    def test_sleeping_bystander_is_never_the_culprit(self):
        for text in (
            _dump(_SLEEPING_MONITOR, _LOOP_RUNNING_CALLBACK),
            _dump(_SLEEPING_MONITOR, _GIL_HUNGRY_GLOB),
            _dump(_SLEEPING_MONITOR),
        ):
            verdict = parse_stall_dump_text(text)
            assert "flagship_doctor" not in verdict.fingerprint_frame()

    def test_narrator_tells_the_same_story(self, tmp_path):
        from core.observability.incident_narrator import IncidentNarrator

        path = tmp_path / "stall_1783564466.txt"
        path.write_text(
            _dump(_SLEEPING_MONITOR, _LOOP_RUNNING_CALLBACK), encoding="utf-8"
        )
        elapsed, culprit = IncidentNarrator._parse_stall_dump(path)
        assert elapsed == pytest.approx(6.2)
        assert culprit == "local_corpus.py:299 (document_count)"

    def test_fingerprint_is_stable_for_one_anatomy(self):
        """Multiple busy background threads: the dominant frame wins so one
        anatomy maps to one incident class, whatever the thread order."""
        a = _dump(_GIL_HUNGRY_GLOB, _GIL_HUNGRY_GLOB.replace("444", "445"), _SLEEPING_MONITOR)
        b = _dump(_SLEEPING_MONITOR, _GIL_HUNGRY_GLOB.replace("444", "446"), _GIL_HUNGRY_GLOB)
        assert (
            parse_stall_dump_text(a).fingerprint_frame()
            == parse_stall_dump_text(b).fingerprint_frame()
            == "gateway_record_index.py:_do_refresh"
        )


# ---------------------------------------------------------------------------
# The watchdog stamps the loop thread — attribution's source of truth.
# ---------------------------------------------------------------------------

class TestWatchdogLoopStamp:
    def _watchdog(self):
        import asyncio

        from core.resilience.stall_watchdog import StallWatchdog

        loop = asyncio.new_event_loop()
        try:
            return StallWatchdog(loop, threshold=5.0)
        finally:
            loop.close()

    def test_heartbeat_learns_the_loop_thread(self):
        dog = self._watchdog()
        assert dog._loop_thread_id is None
        dog._heartbeat()  # runs on the loop thread in production
        assert dog._loop_thread_id == threading.get_ident()

    def test_dump_text_stamps_header_and_section(self):
        dog = self._watchdog()
        dog._heartbeat()
        text = dog._compose_dump_text(7.5)
        assert "STALL DETECTED: 7.5s" in text
        assert f"LOOP THREAD: {threading.get_ident()}" in text
        assert f"Thread ID: {threading.get_ident()}  [EVENT LOOP]" in text
        # And the shared parser closes the loop on a REAL dump of THIS
        # process: this test's own frame is the loop thread's deepest work.
        verdict = parse_stall_dump_text(text)
        assert verdict.thread_kind == "event_loop"

    def test_dump_text_degrades_gracefully_before_first_heartbeat(self):
        dog = self._watchdog()
        text = dog._compose_dump_text(5.0)
        assert "STALL DETECTED: 5.0s" in text
        assert "LOOP THREAD:" not in text
        assert "[EVENT LOOP]" not in text


# ---------------------------------------------------------------------------
# The real culprits the truthful parser exposed, fixed and pinned.
# ---------------------------------------------------------------------------

class TestGoalEngineHotPath:
    def _engine(self, tmp_path):
        from core.goals.goal_engine import GoalEngine

        return GoalEngine(db_path=str(tmp_path / "goals.db"))

    def test_get_active_goals_serves_the_cached_snapshot(self, tmp_path, monkeypatch):
        """EVERY tool authorization descends into get_active_goals on the
        event loop; it must never rebuild from SQLite when the cache is
        warm (26 stall dumps on Jul 8 alone came from this path)."""
        engine = self._engine(tmp_path)
        engine.get_active_goals(limit=6)  # cold call fills the cache

        def _forbidden(*args, **kwargs):  # pragma: no cover - guard
            raise AssertionError("hot path rebuilt the snapshot from SQLite")

        monkeypatch.setattr(engine, "build_snapshot", _forbidden)
        goals = engine.get_active_goals(limit=6, include_external=True)
        assert isinstance(goals, list)

    def test_bulk_readers_still_get_a_fresh_build(self, tmp_path, monkeypatch):
        engine = self._engine(tmp_path)
        engine.get_active_goals(limit=6)
        calls = {"n": 0}
        real_build = engine.build_snapshot

        def _counting(*args, **kwargs):
            calls["n"] += 1
            return real_build(*args, **kwargs)

        monkeypatch.setattr(engine, "build_snapshot", _counting)
        engine.get_active_goals(limit=100)  # beyond snapshot depth
        assert calls["n"] == 1


class TestLocalCorpusExistenceGuard:
    def test_has_documents_matches_count_semantics(self, tmp_path):
        from core.knowledge.local_corpus import LocalCorpusStore

        store = LocalCorpusStore(db_path=tmp_path / "corpus.db")
        assert store.has_documents() is False
        assert store.document_count() == 0
        store.add_documents([("t", "hello corpus", "test")])
        assert store.has_documents() is True
        assert store.document_count() > 0

    def test_missing_db_is_false_not_error(self, tmp_path):
        from core.knowledge.local_corpus import LocalCorpusStore

        store = LocalCorpusStore(db_path=tmp_path / "never_created.db")
        assert store.has_documents() is False

    @pytest.mark.asyncio
    async def test_web_search_local_corpus_runs_off_event_loop(self, monkeypatch):
        """The exact live stall anatomy: corpus FTS must execute in a worker."""
        from core.skills.web_search import EnhancedWebSearchSkill

        skill = EnhancedWebSearchSkill()
        loop_thread = threading.get_ident()
        observed_thread: list[int] = []

        def local_first(_query, _limit):
            observed_thread.append(threading.get_ident())
            return {
                "ok": True,
                "provenance": "local_corpus",
                "results": [{"title": "Solaris", "source": "wikipedia"}],
            }

        monkeypatch.setattr(
            "core.skills.web_search.query_requires_source_reading",
            lambda _query: False,
        )
        monkeypatch.setattr(
            EnhancedWebSearchSkill,
            "_local_corpus_first",
            staticmethod(local_first),
        )
        result = await skill.execute({"query": "Who wrote Solaris?"}, context={})

        assert result["ok"] is True
        assert observed_thread and observed_thread[0] != loop_thread

    def test_research_retention_guard_is_off_loop(self):
        """The retention path must do ALL its sqlite (guard included) inside
        the worker thread — the on-loop COUNT(*) guard was a fingerprinted
        5s stall class."""
        import inspect

        from core.search import research_pipeline

        source = inspect.getsource(
            research_pipeline.ResearchSearchPipeline._retain_artifact
        )
        assert "document_count()" not in source, (
            "retention guard must use has_documents() inside the off-loop worker"
        )
        assert "has_documents" in source and "to_thread(_corpus_writeback)" in source


class TestCrsmDigestCache:
    def _monitor(self, tmp_path):
        from core.consciousness.crsm_loop_monitor import CRSMLoopMonitor

        return CRSMLoopMonitor(
            dataset_path=tmp_path / "dataset.jsonl",
            fused_model_dir=tmp_path / "fused",
            marker_path=tmp_path / "marker.json",
            integration_manifest_path=tmp_path / "manifest.json",
            training_state_path=tmp_path / "training_state.json",
            training_data_dir=tmp_path / "data",
        )

    def test_unchanged_file_is_not_rehashed(self, tmp_path, monkeypatch):
        monitor = self._monitor(tmp_path)
        dataset = tmp_path / "dataset.jsonl"
        dataset.write_text('{"a": 1}\n{"b": 2}\n', encoding="utf-8")

        first = monitor.dataset_state()
        assert first["lines"] == 2 and first["sha256"]

        real_open = type(dataset).open

        def _no_reads(self, *args, **kwargs):  # pragma: no cover - guard
            raise AssertionError("unchanged file was re-read on a status poll")

        monkeypatch.setattr(type(dataset), "open", _no_reads)
        second = monitor.dataset_state()
        monkeypatch.setattr(type(dataset), "open", real_open)
        assert second["sha256"] == first["sha256"]
        assert second["lines"] == first["lines"]

    def test_changed_file_is_rehashed(self, tmp_path):
        monitor = self._monitor(tmp_path)
        dataset = tmp_path / "dataset.jsonl"
        dataset.write_text('{"a": 1}\n', encoding="utf-8")
        first = monitor.dataset_state()
        time.sleep(0.02)
        dataset.write_text('{"a": 1}\n{"b": 2}\n{"c": 3}\n', encoding="utf-8")
        second = monitor.dataset_state()
        assert second["lines"] == 3
        assert second["sha256"] != first["sha256"]

    def test_jsonl_file_state_uses_the_same_cache(self, tmp_path, monkeypatch):
        monitor = self._monitor(tmp_path)
        target = tmp_path / "train.jsonl"
        target.write_text('{"x": 1}\n', encoding="utf-8")
        state = monitor._jsonl_file_state(target, {"sha256": "", "lines": 1, "size": 1})
        assert state["exists"] and state["lines"] == 1

        def _no_reads(self, *args, **kwargs):  # pragma: no cover - guard
            raise AssertionError("unchanged file was re-read")

        monkeypatch.setattr(type(target), "open", _no_reads)
        again = monitor._jsonl_file_state(target)
        assert again["sha256"] == state["sha256"]


class TestIdentityChronicleHotPath:
    def _chronicle(self, tmp_path):
        from core.identity.id_rag import IdentityChronicle

        return IdentityChronicle(db_path=tmp_path / "identity.db")

    def test_retrieve_serves_from_memory_once_warm(self, tmp_path, monkeypatch):
        """relevance_score runs inside EVERY Will decision on the event
        loop; per-call sqlite full scans there were a fingerprinted stall
        class. Warm retrieval must not touch the database."""
        chronicle = self._chronicle(tmp_path)
        try:
            chronicle.upsert_fact("Aura", "value", "honest receipts", tags=("honesty",))
            assert chronicle.retrieve("honest receipts")  # warms the cache

            reader_thread = threading.get_ident()
            real_connect = chronicle._connect

            def _forbidden(*args, **kwargs):
                # The background access-count writer may legitimately touch
                # sqlite; only the READ path on this thread is under test.
                if threading.get_ident() == reader_thread:  # pragma: no cover
                    raise AssertionError("warm retrieve() touched sqlite")
                return real_connect(*args, **kwargs)

            monkeypatch.setattr(chronicle, "_connect", _forbidden)
            results = chronicle.retrieve("honest receipts")
            assert results and results[0].fact.object == "honest receipts"
            assert chronicle.relevance_score("honest receipts") > 0.0
        finally:
            chronicle.close()

    def test_upsert_invalidates_the_snapshot(self, tmp_path):
        chronicle = self._chronicle(tmp_path)
        try:
            chronicle.upsert_fact("Aura", "value", "first fact")
            assert len(chronicle.retrieve("fact", min_score=0.0)) == 1
            chronicle.upsert_fact("Aura", "trait", "second fact")
            objects = {
                item.fact.object
                for item in chronicle.retrieve("fact", min_score=0.0)
            }
            assert objects == {"first fact", "second fact"}
        finally:
            chronicle.close()

    def test_relation_filter_still_works_from_cache(self, tmp_path):
        chronicle = self._chronicle(tmp_path)
        try:
            chronicle.upsert_fact("Aura", "value", "a value fact")
            chronicle.upsert_fact("Aura", "trait", "a trait fact")
            values = chronicle.retrieve("fact", relation_filter="value", min_score=0.0)
            assert [item.fact.relation for item in values] == ["value"]
        finally:
            chronicle.close()
