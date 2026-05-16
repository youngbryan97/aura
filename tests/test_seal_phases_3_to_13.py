"""Tests for Seal Phases 3, 5, 7, 8, 9, 10, 13.

Covers:
  Phase 3: DatabaseMaintenance (checkpoint, retention, vacuum, integrity)
  Phase 5: ResourceGovernor (thermal, inference semaphore, eviction)
  Phase 7: InitiativeOverflowManager, UserResponseTracker
  Phase 8: STDP weight caps, PhiCore disconnected graph detection
  Phase 9: SemanticDedupGate
"""
import asyncio
import sqlite3
import tempfile
import time
import numpy as np
import pytest


# ── Phase 3: DB Maintenance ──────────────────────────────────────────────

class TestDatabaseMaintenance:
    def _make_db(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("CREATE TABLE receipts (id INTEGER PRIMARY KEY, created_at REAL, data TEXT)")
        conn.execute("CREATE TABLE degraded_events (id INTEGER PRIMARY KEY, timestamp REAL, msg TEXT)")
        old_ts = time.time() - 86400 * 60  # 60 days ago
        for i in range(50):
            conn.execute("INSERT INTO receipts (created_at, data) VALUES (?, ?)", (old_ts + i, f"old_{i}"))
        recent_ts = time.time() - 3600
        for i in range(10):
            conn.execute("INSERT INTO receipts (created_at, data) VALUES (?, ?)", (recent_ts + i, f"new_{i}"))
        for i in range(20):
            conn.execute("INSERT INTO degraded_events (timestamp, msg) VALUES (?, ?)", (old_ts + i, f"evt_{i}"))
        conn.commit()
        conn.close()
        return db_path

    def test_retention_deletes_old_rows(self, tmp_path):
        from core.persistence.db_maintenance import DatabaseMaintenance
        db_path = self._make_db(tmp_path)
        maint = DatabaseMaintenance(db_path)
        result = maint.run_maintenance(force=True)
        assert result.rows_deleted.get("receipts", 0) > 0
        assert result.rows_deleted.get("degraded_events", 0) > 0
        conn = sqlite3.connect(db_path)
        remaining = conn.execute("SELECT COUNT(*) FROM receipts").fetchone()[0]
        assert remaining == 10  # Only recent rows remain
        conn.close()

    def test_checkpoint_runs(self, tmp_path):
        from core.persistence.db_maintenance import DatabaseMaintenance
        db_path = self._make_db(tmp_path)
        maint = DatabaseMaintenance(db_path)
        result = maint.run_maintenance(force=True)
        assert result.wal_checkpointed is True

    def test_integrity_check(self, tmp_path):
        from core.persistence.db_maintenance import DatabaseMaintenance
        db_path = self._make_db(tmp_path)
        maint = DatabaseMaintenance(db_path)
        result = maint.run_maintenance(force=True)
        assert result.integrity_ok is True

    def test_size_monitoring(self, tmp_path):
        from core.persistence.db_maintenance import DatabaseMaintenance
        db_path = self._make_db(tmp_path)
        maint = DatabaseMaintenance(db_path)
        result = maint.run_maintenance(force=True)
        assert result.db_size_bytes > 0

    def test_missing_db_no_crash(self, tmp_path):
        from core.persistence.db_maintenance import DatabaseMaintenance
        maint = DatabaseMaintenance(str(tmp_path / "nonexistent.db"))
        result = maint.run_maintenance(force=True)
        assert "no_connection" in result.errors

    def test_result_to_dict(self, tmp_path):
        from core.persistence.db_maintenance import DatabaseMaintenance
        db_path = self._make_db(tmp_path)
        maint = DatabaseMaintenance(db_path)
        result = maint.run_maintenance(force=True)
        d = result.to_dict()
        assert "duration_s" in d
        assert "total_rows_deleted" in d

    def test_status(self, tmp_path):
        from core.persistence.db_maintenance import DatabaseMaintenance
        db_path = self._make_db(tmp_path)
        maint = DatabaseMaintenance(db_path)
        maint.run_maintenance(force=True)
        status = maint.get_status()
        assert status["total_passes"] == 1


# ── Phase 5: Resource Governor ───────────────────────────────────────────

class TestResourceGovernor:
    def test_snapshot(self):
        from core.resource.resource_governor import ResourceGovernor
        gov = ResourceGovernor()
        snap = gov.sample()
        assert snap.memory_percent >= 0
        assert snap.thermal_state is not None
        assert snap.eviction_tier is not None

    def test_inference_semaphore_acquire_release(self):
        from core.resource.resource_governor import InferenceSemaphore
        sem = InferenceSemaphore(max_concurrent=1)
        assert not sem.is_active
        loop = asyncio.new_event_loop()
        acquired = loop.run_until_complete(sem.acquire(source="test", timeout=1.0))
        assert acquired
        assert sem.is_active
        sem.release()
        assert not sem.is_active
        loop.close()

    def test_inference_semaphore_timeout(self):
        from core.resource.resource_governor import InferenceSemaphore
        sem = InferenceSemaphore(max_concurrent=1)
        loop = asyncio.new_event_loop()
        loop.run_until_complete(sem.acquire(source="holder"))
        acquired = loop.run_until_complete(sem.acquire(source="waiter", timeout=0.1))
        assert not acquired
        sem.release()
        loop.close()

    def test_inference_stats(self):
        from core.resource.resource_governor import InferenceSemaphore
        sem = InferenceSemaphore(max_concurrent=1)
        stats = sem.get_stats()
        assert "total_acquired" in stats
        assert "total_timeouts" in stats

    def test_eviction_callback(self):
        from core.resource.resource_governor import ResourceGovernor, EvictionTier
        gov = ResourceGovernor()
        calls = []
        gov.register_eviction_callback(lambda tier: calls.append(tier))
        count = gov.execute_eviction(EvictionTier.SOFT)
        assert count == 1
        assert calls == [EvictionTier.SOFT]

    def test_eviction_none_noop(self):
        from core.resource.resource_governor import ResourceGovernor, EvictionTier
        gov = ResourceGovernor()
        count = gov.execute_eviction(EvictionTier.NONE)
        assert count == 0

    def test_status(self):
        from core.resource.resource_governor import ResourceGovernor
        gov = ResourceGovernor()
        status = gov.get_status()
        assert "inference" in status
        assert "throttle_active" in status


# ── Phase 7: Initiative Overflow ─────────────────────────────────────────

class TestInitiativeOverflow:
    def test_record_overflow(self):
        from core.autonomy.initiative_overflow import InitiativeOverflowManager
        mgr = InitiativeOverflowManager()
        mgr.record_overflow("test_goal", source="test", queue_depth=15)
        assert mgr._overflow_count == 1
        assert mgr._consecutive_overflows == 1

    def test_record_success_resets(self):
        from core.autonomy.initiative_overflow import InitiativeOverflowManager
        mgr = InitiativeOverflowManager()
        mgr.record_overflow("g1")
        mgr.record_overflow("g2")
        mgr.record_success()
        assert mgr._consecutive_overflows == 0

    def test_skill_gap_dedup(self):
        from core.autonomy.initiative_overflow import InitiativeOverflowManager
        mgr = InitiativeOverflowManager()
        mgr.record_skill_gap("web_scraping", "needed for research")
        mgr.record_skill_gap("web_scraping", "needed for deeper research")
        assert len(mgr._skill_gaps) == 1
        assert mgr._skill_gaps["web_scraping"].occurrences == 2

    def test_resolve_skill_gap(self):
        from core.autonomy.initiative_overflow import InitiativeOverflowManager
        mgr = InitiativeOverflowManager()
        mgr.record_skill_gap("coding", "fix bugs")
        assert mgr.resolve_skill_gap("coding") is True
        assert mgr._skill_gaps["coding"].resolved is True

    def test_top_skill_gaps(self):
        from core.autonomy.initiative_overflow import InitiativeOverflowManager
        mgr = InitiativeOverflowManager()
        mgr.record_skill_gap("a")
        mgr.record_skill_gap("b")
        mgr.record_skill_gap("a")
        gaps = mgr.get_top_skill_gaps(5)
        assert gaps[0]["skill_name"] == "a"
        assert gaps[0]["occurrences"] == 2

    def test_cap_defaults(self):
        from core.autonomy.initiative_overflow import InitiativeOverflowManager
        mgr = InitiativeOverflowManager()
        assert mgr.current_cap == 10

    def test_status(self):
        from core.autonomy.initiative_overflow import InitiativeOverflowManager
        mgr = InitiativeOverflowManager()
        s = mgr.get_status()
        assert "current_cap" in s
        assert "skill_gaps_total" in s


# ── Phase 7: User Response Tracker ───────────────────────────────────────

class TestUserResponseTracker:
    def test_initial_state(self):
        from core.autonomy.user_response_tracker import UserResponseTracker
        t = UserResponseTracker()
        assert t.response_rate == 1.0  # Assume engaged
        assert not t.should_backoff

    def test_track_send_no_response(self):
        from core.autonomy.user_response_tracker import UserResponseTracker
        t = UserResponseTracker()
        for _ in range(5):
            t.record_proactive_sent()
            # Simulate timeout
            evt = t._pending_event
            if evt:
                evt.sent_at -= 700  # Force past window
                t._events.append(evt)
                t._pending_event = None
        assert t.response_rate < 0.5

    def test_track_send_with_response(self):
        from core.autonomy.user_response_tracker import UserResponseTracker
        t = UserResponseTracker()
        for _ in range(5):
            t.record_proactive_sent()
            t.record_user_response()
        assert t.response_rate == 1.0

    def test_backoff_multiplier_increases(self):
        from core.autonomy.user_response_tracker import UserResponseTracker
        t = UserResponseTracker()
        for _ in range(10):
            t.record_proactive_sent()
            evt = t._pending_event
            if evt:
                evt.sent_at -= 700
                t._events.append(evt)
                t._pending_event = None
        m = t.get_backoff_multiplier()
        assert m > 1.0

    def test_engagement_score(self):
        from core.autonomy.user_response_tracker import UserResponseTracker
        t = UserResponseTracker()
        t.record_proactive_sent()
        t.record_user_response()
        assert t.engagement_score > 0.0


# ── Phase 8: STDP Hardening ──────────────────────────────────────────────

class TestSTDPHardening:
    def test_spectral_norm_cap(self):
        from core.consciousness.stdp_learning import STDPLearningEngine
        engine = STDPLearningEngine(n_neurons=8)
        W = np.random.randn(8, 8).astype(np.float32) * 10.0
        dw = np.zeros((8, 8), dtype=np.float32)
        W_new = engine.apply_to_connectivity(W, dw)
        s_max = np.linalg.norm(W_new, ord=2)
        assert s_max <= 3.01  # Allow tiny float tolerance

    def test_nan_guard(self):
        from core.consciousness.stdp_learning import STDPLearningEngine
        engine = STDPLearningEngine(n_neurons=4)
        W = np.array([[0, np.nan, 1, 0],
                       [np.inf, 0, 0, -np.inf],
                       [0, 0, 0, 1],
                       [1, 0, 0, 0]], dtype=np.float32)
        dw = np.zeros((4, 4), dtype=np.float32)
        W_new = engine.apply_to_connectivity(W, dw)
        assert np.isfinite(W_new).all()

    def test_homeostatic_scaling(self):
        from core.consciousness.stdp_learning import STDPLearningEngine
        engine = STDPLearningEngine(n_neurons=8)
        W = np.ones((8, 8), dtype=np.float32) * 1.5
        np.fill_diagonal(W, 0)
        dw = np.zeros((8, 8), dtype=np.float32)
        W_new = engine.apply_to_connectivity(W, dw)
        mean_new = float(np.mean(np.abs(W_new[W_new != 0])))
        assert mean_new < 1.5  # Should have corrected downward


# ── Phase 8: PhiCore Disconnected Graph ──────────────────────────────────

class TestPhiCoreDisconnectedGraph:
    def test_connected_graph(self):
        from core.consciousness.phi_core import PhiCore
        phi = PhiCore.__new__(PhiCore)
        graph = np.ones((4, 4), dtype=np.float64) * 0.1
        np.fill_diagonal(graph, 0)
        is_conn, n_comp, sizes = phi._detect_disconnected_graph(graph)
        assert is_conn is True
        assert n_comp == 1

    def test_disconnected_graph(self):
        from core.consciousness.phi_core import PhiCore
        phi = PhiCore.__new__(PhiCore)
        graph = np.zeros((4, 4), dtype=np.float64)
        graph[0, 1] = 0.5
        graph[1, 0] = 0.5
        graph[2, 3] = 0.5
        graph[3, 2] = 0.5
        is_conn, n_comp, sizes = phi._detect_disconnected_graph(graph)
        assert is_conn is False
        assert n_comp == 2
        assert sorted(sizes) == [2, 2]

    def test_empty_graph(self):
        from core.consciousness.phi_core import PhiCore
        phi = PhiCore.__new__(PhiCore)
        graph = np.zeros((4, 4), dtype=np.float64)
        is_conn, n_comp, sizes = phi._detect_disconnected_graph(graph)
        assert is_conn is False
        assert n_comp == 4


# ── Phase 9: Semantic Dedup ──────────────────────────────────────────────

class TestSemanticDedup:
    def test_exact_duplicate_rejected(self):
        from core.memory.semantic_dedup import SemanticDedupGate
        gate = SemanticDedupGate()
        assert gate.should_store("This is a test memory about something important")
        assert not gate.should_store("This is a test memory about something important")

    def test_near_duplicate_rejected(self):
        from core.memory.semantic_dedup import SemanticDedupGate
        gate = SemanticDedupGate()
        assert gate.should_store("The quick brown fox jumps over the lazy dog near the river")
        assert not gate.should_store("The quick brown fox jumps over the lazy dog near the river bank")

    def test_novel_content_passes(self):
        from core.memory.semantic_dedup import SemanticDedupGate
        gate = SemanticDedupGate()
        assert gate.should_store("Alpha beta gamma delta epsilon")
        assert gate.should_store("Completely different content about quantum mechanics")

    def test_high_importance_bypasses(self):
        from core.memory.semantic_dedup import SemanticDedupGate
        gate = SemanticDedupGate()
        gate.should_store("Important memory about identity")
        assert gate.should_store("Important memory about identity", importance=0.9)

    def test_trivial_text_rejected(self):
        from core.memory.semantic_dedup import SemanticDedupGate
        gate = SemanticDedupGate()
        assert not gate.should_store("hi")

    def test_status(self):
        from core.memory.semantic_dedup import SemanticDedupGate
        gate = SemanticDedupGate()
        gate.should_store("test content for status check")
        s = gate.get_status()
        assert s["total_checked"] == 1
        assert s["total_passed"] == 1
        assert "dedup_rate" in s
