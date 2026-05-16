"""tests/test_seal_infrastructure.py
=====================================
Comprehensive test suite for the Gold Master seal infrastructure.

Tests cover:
1. Will gate decorator behavior
2. WorldResult typed adapter results
3. Feature flag system
4. Incident manager lifecycle
5. Boring mode transitions
6. Process manager exponential backoff
7. Substrate NaN guard
8. Workspace jail path validation
9. Metrics collector
10. Health probes
"""
from __future__ import annotations

import asyncio
import math
import os
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np

# Ensure AURA_TEST_MODE is set before any imports that might check it
os.environ.setdefault("AURA_TEST_MODE", "1")


class TestWillGateDecorator(unittest.TestCase):
    """Tests for the @will_gated decorator."""

    def test_import_and_registry(self):
        from core.governance.will_gate import _GATED_METHODS, will_gated
        from core.will import ActionDomain

        @will_gated(domain=ActionDomain.TOOL_EXECUTION)
        def test_func():
            return "executed"

        self.assertTrue(any("test_func" in m for m in _GATED_METHODS))

    def test_audit_will_coverage(self):
        from core.governance.will_gate import audit_will_coverage

        report = audit_will_coverage(strict=False)
        self.assertIn("total_gated", report)
        self.assertIn("missing", report)
        self.assertIn("all_covered", report)
        self.assertIsInstance(report["total_gated"], int)

    def test_will_refused_exception(self):
        from core.governance.will_gate import WillRefused

        exc = WillRefused("receipt_123", "identity violation", "tool_execution")
        self.assertEqual(exc.receipt_id, "receipt_123")
        self.assertEqual(exc.reason, "identity violation")
        self.assertIn("REFUSED", str(exc))

    def test_will_deferred_exception(self):
        from core.governance.will_gate import WillDeferred

        exc = WillDeferred("receipt_456", "low priority", "initiative")
        self.assertEqual(exc.receipt_id, "receipt_456")
        self.assertIn("DEFERRED", str(exc))


class TestWorldResult(unittest.TestCase):
    """Tests for the typed adapter result."""

    def test_ok_result(self):
        from core.adapters.typed_result import WorldResult

        result = WorldResult.ok(["item1", "item2"])
        self.assertTrue(result.success)
        self.assertFalse(result.failed)
        self.assertEqual(result.data, ["item1", "item2"])
        self.assertIsNone(result.error_info)
        self.assertFalse(result.is_empty)

    def test_ok_empty_result(self):
        from core.adapters.typed_result import WorldResult

        result = WorldResult.ok([])
        self.assertTrue(result.success)
        self.assertTrue(result.is_empty)
        # CRITICAL: empty success is NOT a failure
        self.assertFalse(result.failed)

    def test_error_result(self):
        from core.adapters.typed_result import WorldResult

        result = WorldResult.fail("timeout", "connection timed out")
        self.assertFalse(result.success)
        self.assertTrue(result.failed)
        # CRITICAL: errors are NOT empty (they're errors)
        self.assertFalse(result.is_empty)
        self.assertIsNotNone(result.error_info)
        self.assertEqual(result.error_info.kind.value, "timeout")

    def test_unwrap_on_success(self):
        from core.adapters.typed_result import WorldResult

        result = WorldResult.ok(42)
        self.assertEqual(result.unwrap(), 42)

    def test_unwrap_on_failure(self):
        from core.adapters.typed_result import WorldResult

        result = WorldResult.fail("network_error", "unreachable")
        self.assertIsNone(result.unwrap())
        self.assertEqual(result.unwrap("fallback"), "fallback")

    def test_unwrap_or_raise_on_failure(self):
        from core.adapters.typed_result import WorldResult

        result = WorldResult.fail("internal_error", "something broke")
        with self.assertRaises(RuntimeError):
            result.unwrap_or_raise()

    def test_to_dict(self):
        from core.adapters.typed_result import WorldResult

        result = WorldResult.ok({"key": "value"}, adapter_name="test")
        d = result.to_dict()
        self.assertTrue(d["success"])
        self.assertEqual(d["adapter_name"], "test")
        self.assertIn("data_keys", d)

    def test_error_to_dict(self):
        from core.adapters.typed_result import WorldResult

        result = WorldResult.fail("timeout", "slow", adapter_name="web_search")
        d = result.to_dict()
        self.assertFalse(d["success"])
        self.assertIn("error", d)
        self.assertEqual(d["error"]["kind"], "timeout")

    def test_wrap_adapter_call_sync(self):
        from core.adapters.typed_result import WorldResult, wrap_adapter_call

        @wrap_adapter_call
        def good_adapter():
            return [1, 2, 3]

        result = good_adapter()
        self.assertIsInstance(result, WorldResult)
        self.assertTrue(result.success)
        self.assertEqual(result.data, [1, 2, 3])

    def test_wrap_adapter_call_failure(self):
        from core.adapters.typed_result import WorldResult, wrap_adapter_call

        @wrap_adapter_call
        def bad_adapter():
            raise TimeoutError("connection timeout")

        result = bad_adapter()
        self.assertIsInstance(result, WorldResult)
        self.assertFalse(result.success)
        self.assertEqual(result.error_info.kind.value, "timeout")


class TestFeatureFlags(unittest.TestCase):
    """Tests for the feature flag system."""

    def test_default_flags_loaded(self):
        from core.governance.feature_flags import FeatureFlags

        flags = FeatureFlags(config_path=Path("/tmp/test_flags_nonexistent.json"))
        all_flags = flags.get_all()
        self.assertIn("boring_mode_auto_enter", all_flags)
        self.assertIn("substrate_nan_guard", all_flags)
        self.assertTrue(all_flags["boring_mode_auto_enter"])

    def test_set_flag_runtime(self):
        from core.governance.feature_flags import FeatureFlags

        flags = FeatureFlags(config_path=Path("/tmp/test_flags_nonexistent.json"))
        flags.set_flag("boring_mode_auto_enter", False, reason="test")
        self.assertFalse(flags.is_enabled("boring_mode_auto_enter"))

    def test_change_log(self):
        from core.governance.feature_flags import FeatureFlags

        flags = FeatureFlags(config_path=Path("/tmp/test_flags_nonexistent.json"))
        flags.set_flag("substrate_nan_guard", False, reason="test_toggle")
        log = flags.get_change_log()
        self.assertEqual(len(log), 1)
        self.assertEqual(log[0]["flag"], "substrate_nan_guard")
        self.assertEqual(log[0]["new"], False)

    def test_env_override(self):
        from core.governance.feature_flags import FeatureFlags

        os.environ["AURA_FLAG_BORING_MODE_AUTO_ENTER"] = "0"
        try:
            flags = FeatureFlags(config_path=Path("/tmp/test_flags_nonexistent.json"))
            self.assertFalse(flags.is_enabled("boring_mode_auto_enter"))
        finally:
            del os.environ["AURA_FLAG_BORING_MODE_AUTO_ENTER"]

    def test_descriptions(self):
        from core.governance.feature_flags import FeatureFlags

        flags = FeatureFlags(config_path=Path("/tmp/test_flags_nonexistent.json"))
        descriptions = flags.get_descriptions()
        self.assertIn("boring_mode_auto_enter", descriptions)
        self.assertTrue(len(descriptions["boring_mode_auto_enter"]) > 10)


class TestIncidentManager(unittest.TestCase):
    """Tests for the incident manager."""

    def test_report_new_incident(self):
        from core.resilience.incident_manager import (
            IncidentManager,
            IncidentSeverity,
        )

        mgr = IncidentManager()
        incident = mgr.report(
            category="test_failure",
            description="Something broke",
            severity=IncidentSeverity.WARNING,
        )
        self.assertEqual(incident.category, "test_failure")
        self.assertEqual(incident.severity, IncidentSeverity.WARNING)
        self.assertEqual(incident.occurrence_count, 1)

    def test_deduplication(self):
        from core.resilience.incident_manager import IncidentManager

        mgr = IncidentManager()
        mgr.report(category="dup_test", description="first")
        mgr.report(category="dup_test", description="second")
        mgr.report(category="dup_test", description="third")

        active = mgr.get_active()
        dup_incidents = [i for i in active if i["category"] == "dup_test"]
        self.assertEqual(len(dup_incidents), 1)
        self.assertEqual(dup_incidents[0]["occurrence_count"], 3)

    def test_escalation(self):
        from core.resilience.incident_manager import (
            IncidentManager,
            IncidentSeverity,
        )

        mgr = IncidentManager()
        mgr.ESCALATION_THRESHOLD = 3  # Lower for test
        for i in range(5):
            incident = mgr.report(
                category="escalation_test",
                description=f"occurrence {i}",
                severity=IncidentSeverity.INFO,
            )

        active = mgr.get_active()
        test_incidents = [i for i in active if i["category"] == "escalation_test"]
        # Should have escalated from INFO
        self.assertNotEqual(test_incidents[0]["severity"], "info")

    def test_resolve(self):
        from core.resilience.incident_manager import IncidentManager

        mgr = IncidentManager()
        mgr.report(category="resolve_test", description="will resolve")
        resolved = mgr.resolve("resolve_test", "fixed it")
        self.assertIsNotNone(resolved)
        self.assertEqual(resolved.status.value, "recovered")

        # Should no longer be active
        active = mgr.get_active()
        self.assertFalse(any(i["category"] == "resolve_test" for i in active))

    def test_summary(self):
        from core.resilience.incident_manager import IncidentManager

        mgr = IncidentManager()
        mgr.report(category="summary_1", description="test")
        mgr.report(category="summary_2", description="test")
        summary = mgr.get_summary()
        self.assertEqual(summary["active_count"], 2)
        self.assertEqual(summary["total_incidents"], 2)


class TestBoringMode(unittest.TestCase):
    """Tests for Boring Mode."""

    def test_initial_state(self):
        from core.resilience.boring_mode import BoringMode

        bm = BoringMode()
        self.assertFalse(bm.is_active)

    def test_enter_exit(self):
        from core.resilience.boring_mode import BoringMode

        bm = BoringMode()
        bm.enter("test_substrate_nan")
        self.assertTrue(bm.is_active)
        bm.exit("test_recovered")
        self.assertFalse(bm.is_active)

    def test_action_blocking(self):
        from core.resilience.boring_mode import BoringMode

        bm = BoringMode()
        bm.enter("test")

        # Responses should always be allowed
        self.assertTrue(bm.should_allow_action("response"))
        # Tool execution should be blocked
        self.assertFalse(bm.should_allow_action("tool_execution"))
        # Exploration should be blocked
        self.assertFalse(bm.should_allow_action("exploration"))
        # Stabilization should be allowed
        self.assertTrue(bm.should_allow_action("stabilization"))

        bm.exit()

    def test_cooldown(self):
        from core.resilience.boring_mode import BoringMode

        bm = BoringMode()
        bm.ENTRY_COOLDOWN = 0.1  # Short for test
        bm.enter("test1")
        bm.exit("test1_done")

        # Immediate re-entry should be blocked by cooldown
        result = bm.enter("test2")
        self.assertFalse(result)

        time.sleep(0.15)
        # After cooldown, should succeed
        result = bm.enter("test3")
        self.assertTrue(result)
        bm.exit()

    def test_safe_response_prefix(self):
        from core.resilience.boring_mode import BoringMode

        bm = BoringMode()
        prefix = bm.get_safe_response_prefix()
        self.assertIn("safe mode", prefix)

    def test_status(self):
        from core.resilience.boring_mode import BoringMode

        bm = BoringMode()
        status = bm.get_status()
        self.assertFalse(status["active"])
        self.assertEqual(status["entry_count"], 0)


class TestWorkspaceJail(unittest.TestCase):
    """Tests for the workspace jail."""

    def test_allowed_path(self):
        from core.security.workspace_jail import WorkspaceJail

        jail = WorkspaceJail()
        allowed, resolved, reason = jail.validate_path(
            str(Path.home() / ".aura" / "test.txt")
        )
        self.assertTrue(allowed)
        self.assertEqual(reason, "allowed")

    def test_denied_path_traversal(self):
        from core.security.workspace_jail import WorkspaceJail

        jail = WorkspaceJail()
        allowed, _, reason = jail.validate_path("/etc/passwd")
        self.assertFalse(allowed)
        # On macOS, /etc resolves to /private/etc so it may be 'outside_jail'
        # rather than 'denied_path'. Both are correct denials.
        self.assertIn(reason, ("denied_path", "outside_jail"))

    def test_denied_ssh_keys(self):
        from core.security.workspace_jail import WorkspaceJail

        jail = WorkspaceJail()
        allowed, _, reason = jail.validate_path(
            str(Path.home() / ".ssh" / "id_rsa")
        )
        self.assertFalse(allowed)

    def test_denied_env_file(self):
        from core.security.workspace_jail import WorkspaceJail

        jail = WorkspaceJail()
        allowed, _, reason = jail.validate_path(
            str(Path.home() / ".aura" / ".env")
        )
        self.assertFalse(allowed)
        self.assertEqual(reason, "denied_filename")

    def test_outside_jail(self):
        from core.security.workspace_jail import WorkspaceJail

        jail = WorkspaceJail()
        allowed, _, reason = jail.validate_path("/usr/local/bin/python3")
        self.assertFalse(allowed)
        self.assertEqual(reason, "outside_jail")

    def test_sanitize_returns_none_on_denied(self):
        from core.security.workspace_jail import WorkspaceJail

        jail = WorkspaceJail()
        result = jail.sanitize_path("/etc/shadow")
        self.assertIsNone(result)

    def test_empty_path(self):
        from core.security.workspace_jail import WorkspaceJail

        jail = WorkspaceJail()
        allowed, _, reason = jail.validate_path("")
        self.assertFalse(allowed)
        self.assertEqual(reason, "empty_path")


class TestMetricsCollector(unittest.TestCase):
    """Tests for the metrics collector."""

    def test_record_tick(self):
        from core.observability.metrics import MetricsCollector

        mc = MetricsCollector()
        mc.record_tick(15.5)
        mc.record_tick(20.0)
        self.assertEqual(len(mc._tick_durations), 2)

    def test_record_will_decision(self):
        from core.observability.metrics import MetricsCollector

        mc = MetricsCollector()
        mc.record_will_decision("proceed")
        mc.record_will_decision("refuse")
        self.assertEqual(mc._will_decisions["proceed"], 1)
        self.assertEqual(mc._will_decisions["refuse"], 1)

    def test_collect(self):
        from core.observability.metrics import MetricsCollector

        mc = MetricsCollector()
        mc.record_tick(10.0)
        samples = mc.collect()
        self.assertTrue(len(samples) > 0)

        # Check that uptime metric exists
        names = [s.name for s in samples]
        self.assertIn("aura_uptime_seconds", names)
        self.assertIn("aura_ticks_total", names)

    def test_render_prometheus(self):
        from core.observability.metrics import MetricsCollector

        mc = MetricsCollector()
        mc.record_tick(10.0)
        mc.record_will_decision("proceed")
        text = mc.render_prometheus()

        self.assertIn("# HELP aura_uptime_seconds", text)
        self.assertIn("# TYPE aura_uptime_seconds gauge", text)
        self.assertIn("aura_ticks_total", text)

    def test_health_checks(self):
        from core.observability.metrics import check_liveness, check_readiness

        liveness = check_liveness()
        self.assertEqual(liveness["status"], "alive")
        self.assertIn("pid", liveness)

        readiness = check_readiness()
        self.assertIn("status", readiness)
        self.assertIn("ready", readiness)


class TestSubstrateNaNGuard(unittest.TestCase):
    """Tests for the substrate ODE NaN/Inf guard."""

    def test_normal_step(self):
        from core.brain.llm.continuous_substrate import ContinuousSubstrate

        substrate = ContinuousSubstrate()
        substrate._step_once()
        self.assertTrue(np.isfinite(substrate._state).all())

    def test_nan_injection_recovery(self):
        from core.brain.llm.continuous_substrate import ContinuousSubstrate

        substrate = ContinuousSubstrate()
        # Run a few normal steps to populate rollback ring
        for _ in range(25):
            substrate._step_once()

        # Inject NaN into state
        good_state = substrate._state.copy()
        substrate._state[0] = float("nan")
        substrate._state[1] = float("inf")

        # The ODE step should catch it... but NaN is already in _state
        # Inject NaN into input signal to trigger on next step
        substrate._state = good_state.copy()  # reset
        substrate._input_signal = np.full_like(substrate._input_signal, float("nan"))

        # Step should handle the NaN input gracefully via tanh saturation
        substrate._step_once()
        # State should still be finite (tanh of NaN might be NaN, so guard kicks in)
        self.assertTrue(np.isfinite(substrate._state).all())

    def test_rollback_ring_populated(self):
        from core.brain.llm.continuous_substrate import ContinuousSubstrate

        substrate = ContinuousSubstrate()
        for _ in range(25):
            substrate._step_once()

        # After 25 steps (> 20), rollback ring should have at least 1 entry
        self.assertTrue(len(substrate._rollback_ring) >= 1)


class TestProcessManagerBackoff(unittest.TestCase):
    """Tests for the process manager exponential backoff."""

    def test_permanently_failed_state_exists(self):
        from core.process_manager import ProcessState

        self.assertEqual(ProcessState.PERMANENTLY_FAILED.value, "permanently_failed")

    def test_compute_backoff(self):
        from core.process_manager import ManagedProcess, ProcessConfig

        config = ProcessConfig(
            name="test_process",
            target=lambda: None,
        )
        managed = ManagedProcess(config)

        # First restart: should be ~2s base
        managed.stats.restarts = 0
        backoff = managed._compute_backoff()
        self.assertGreater(backoff, 0.5)
        self.assertLess(backoff, 5.0)

        # Third restart: should be higher
        managed.stats.restarts = 3
        backoff3 = managed._compute_backoff()
        self.assertGreater(backoff3, backoff)

        # Cap at 120s
        managed.stats.restarts = 20
        backoff_max = managed._compute_backoff()
        self.assertLessEqual(backoff_max, 150.0)  # 120 + 25% jitter


class TestSingletonStaleLockReclamation(unittest.TestCase):
    """Tests for singleton stale lock reclamation."""

    def test_release_instance_lock(self):
        from core.utils.singleton import release_instance_lock

        # Should not raise even if no lock is held
        release_instance_lock()


if __name__ == "__main__":
    unittest.main()
