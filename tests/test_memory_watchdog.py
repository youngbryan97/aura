"""Tests for the out-of-band MemoryWatchdog escalation ladder.

The watchdog exists because every in-loop enforcement path goes blind when
swap pressure stalls the event loop. These tests drive the policy directly
(no thread, no psutil) through injected hooks.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.resilience.memory_watchdog import (  # noqa: E402
    MEMORY_ABORT_EXIT_CODE,
    MemorySample,
    MemoryWatchdog,
    _Thresholds,
)


def _sample(
    *,
    core_mb: float = 1000.0,
    child_mb: float = 0.0,
    swap_gb: float = 0.0,
    sys_pct: float = 50.0,
) -> MemorySample:
    return MemorySample(
        core_rss_mb=core_mb,
        child_rss_mb=child_mb,
        swap_used_gb=swap_gb,
        system_percent=sys_pct,
        total_ram_gb=64.0,
        sampled_at=0.0,
    )


class _Harness:
    """Builds a watchdog with all side effects captured, never started."""

    def __init__(self, *, lethal_action: str = "exit", boot_grace_s: float = 0.0):
        self.killed_calls = 0
        self.gc_calls = 0
        self.exits: list[int] = []
        self.thresholds = _Thresholds(
            soft_mb=10_000.0,
            hard_mb=20_000.0,
            lethal_mb=30_000.0,
            swap_hard_gb=16.0,
            soft_cooldown_s=30.0,
            hard_cooldown_s=60.0,
            lethal_confirmations=2,
            boot_grace_s=boot_grace_s,
        )
        self.dog = MemoryWatchdog(
            thresholds=self.thresholds,
            lethal_action=lethal_action,
            sample_interval_s=999.0,
            sampler=lambda: _sample(),
            worker_terminator=self._kill,
            gc_collect=self._gc,
            process_exit=self.exits.append,
        )
        # Tests control boot-grace via thresholds, not wall clock.
        self.dog._started_at = -10_000.0

    def _kill(self) -> int:
        self.killed_calls += 1
        return 1

    def _gc(self) -> int:
        self.gc_calls += 1
        return 42


class TestEscalationLadder(unittest.TestCase):
    def test_below_all_ceilings_takes_no_action(self):
        h = _Harness()
        tier = h.dog._evaluate(_sample(core_mb=5_000.0), now=100.0)
        self.assertEqual(tier, "none")
        self.assertEqual(h.killed_calls, 0)
        self.assertEqual(h.exits, [])

    def test_soft_ceiling_schedules_governor_and_respects_cooldown(self):
        h = _Harness()
        self.assertEqual(h.dog._evaluate(_sample(core_mb=12_000.0), now=100.0), "soft")
        self.assertEqual(
            h.dog._evaluate(_sample(core_mb=12_000.0), now=110.0), "soft_stable"
        )
        self.assertEqual(
            h.dog._evaluate(_sample(core_mb=13_100.0), now=120.0), "soft_cooldown"
        )
        self.assertEqual(
            h.dog._evaluate(_sample(core_mb=13_100.0), now=200.0), "soft"
        )
        self.assertEqual(h.killed_calls, 0)

    def test_soft_incident_rearms_after_real_recovery(self):
        h = _Harness()

        self.assertEqual(h.dog._evaluate(_sample(core_mb=12_000.0), now=100.0), "soft")
        self.assertEqual(h.dog._evaluate(_sample(core_mb=9_000.0), now=140.0), "none")
        self.assertFalse(h.dog.health_snapshot()["soft_incident"]["active"])
        self.assertEqual(h.dog._evaluate(_sample(core_mb=12_000.0), now=180.0), "soft")

    def test_soft_incident_rearms_when_host_pressure_materially_worsens(self):
        h = _Harness()

        self.assertEqual(
            h.dog._evaluate(_sample(core_mb=5_000.0, sys_pct=93.0), now=100.0),
            "soft",
        )
        self.assertEqual(
            h.dog._evaluate(_sample(core_mb=5_000.0, sys_pct=94.0), now=140.0),
            "soft_stable",
        )
        self.assertEqual(
            h.dog._evaluate(_sample(core_mb=5_000.0, sys_pct=96.0), now=180.0),
            "soft",
        )

    def test_high_system_percent_triggers_soft_even_with_low_rss(self):
        h = _Harness()
        tier = h.dog._evaluate(_sample(core_mb=5_000.0, sys_pct=95.0), now=100.0)
        self.assertEqual(tier, "soft")

    def test_hard_ceiling_kills_workers_and_forces_gc(self):
        h = _Harness()
        tier = h.dog._evaluate(_sample(core_mb=15_000.0, child_mb=7_000.0), now=100.0)
        self.assertEqual(tier, "hard")
        self.assertEqual(h.killed_calls, 1)
        self.assertEqual(h.gc_calls, 1)
        self.assertEqual(h.exits, [])

    def test_hard_ceiling_respects_cooldown(self):
        h = _Harness()
        h.dog._evaluate(_sample(core_mb=25_000.0), now=100.0)
        tier = h.dog._evaluate(_sample(core_mb=25_000.0), now=120.0)
        self.assertEqual(tier, "hard_cooldown")
        self.assertEqual(h.killed_calls, 1)

    def test_swap_exhaustion_escalates_to_hard_below_hard_rss(self):
        h = _Harness()
        tier = h.dog._evaluate(
            _sample(core_mb=12_000.0, swap_gb=20.0), now=100.0
        )
        self.assertEqual(tier, "hard")
        self.assertEqual(h.killed_calls, 1)

    def test_swap_alone_without_managed_rss_does_not_kill(self):
        # Another app may own the swap; only act when our tree is elevated.
        h = _Harness()
        tier = h.dog._evaluate(_sample(core_mb=2_000.0, swap_gb=40.0), now=100.0)
        self.assertEqual(tier, "none")
        self.assertEqual(h.killed_calls, 0)


class TestLethalPath(unittest.TestCase):
    def test_lethal_reclaims_first_then_exits_after_confirmations(self):
        h = _Harness()
        s = _sample(core_mb=35_000.0)
        self.assertEqual(h.dog._evaluate(s, now=100.0), "lethal_reclaim")
        self.assertEqual(h.killed_calls, 1)
        self.assertEqual(h.exits, [])
        self.assertEqual(h.dog._evaluate(s, now=103.0), "lethal_pending")
        self.assertEqual(h.exits, [])
        self.assertEqual(h.dog._evaluate(s, now=106.0), "lethal_exit")
        self.assertEqual(h.exits, [MEMORY_ABORT_EXIT_CODE])

    def test_recovery_below_lethal_resets_the_streak(self):
        h = _Harness()
        s = _sample(core_mb=35_000.0)
        h.dog._evaluate(s, now=100.0)
        # Reclaim worked: next sample is healthy again.
        self.assertEqual(h.dog._evaluate(_sample(core_mb=5_000.0), now=103.0), "none")
        # A later excursion starts over with a reclaim, not an exit.
        self.assertEqual(h.dog._evaluate(s, now=300.0), "lethal_reclaim")
        self.assertEqual(h.exits, [])

    def test_lethal_action_off_never_exits(self):
        h = _Harness(lethal_action="off")
        s = _sample(core_mb=35_000.0)
        h.dog._evaluate(s, now=100.0)
        h.dog._evaluate(s, now=103.0)
        tier = h.dog._evaluate(s, now=106.0)
        self.assertEqual(tier, "lethal_suppressed")
        self.assertEqual(h.exits, [])

    def test_lethal_action_shed_keeps_shedding_without_exit(self):
        h = _Harness(lethal_action="shed")
        s = _sample(core_mb=35_000.0)
        h.dog._evaluate(s, now=100.0)
        h.dog._evaluate(s, now=103.0)
        tier = h.dog._evaluate(s, now=200.0)
        self.assertEqual(tier, "lethal_shed")
        self.assertGreaterEqual(h.killed_calls, 2)
        self.assertEqual(h.exits, [])

    def test_boot_grace_suppresses_exit(self):
        h = _Harness(boot_grace_s=10_000.0)
        h.dog._started_at = 0.0
        s = _sample(core_mb=35_000.0)
        h.dog._evaluate(s, now=100.0)
        h.dog._evaluate(s, now=103.0)
        tier = h.dog._evaluate(s, now=106.0)
        self.assertEqual(tier, "lethal_suppressed")
        self.assertEqual(h.exits, [])

    def test_tombstone_written_before_exit(self):
        """The tombstone lands in the CONFIGURED forensic root, not beside the cwd.

        This used to rebind the module-level `_TOMBSTONE_DIR`, which meant it
        could pass while the real path was `Path("data/error_logs/memory")` —
        relative to whatever directory the launcher started in. That is the
        same split that had crash correlation reading an empty directory for
        weeks. Driving it through AURA_LOG_DIR proves the redirection works
        end to end, which is the property that actually matters.
        """
        import os
        import tempfile

        h = _Harness()
        s = _sample(core_mb=35_000.0)
        previous = os.environ.get("AURA_LOG_DIR")
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["AURA_LOG_DIR"] = tmp
            try:
                h.dog._evaluate(s, now=100.0)
                h.dog._evaluate(s, now=103.0)
                h.dog._evaluate(s, now=106.0)
                tombstones = list(
                    (Path(tmp) / "error_logs" / "memory").glob("oom_tombstone_*.json")
                )
            finally:
                if previous is None:
                    os.environ.pop("AURA_LOG_DIR", None)
                else:
                    os.environ["AURA_LOG_DIR"] = previous
        self.assertEqual(len(tombstones), 1)
        self.assertEqual(h.exits, [MEMORY_ABORT_EXIT_CODE])


class TestSpikeDumpBounding(unittest.TestCase):
    """Routine 20GB inference footprint jumps must not flood the crash dir.

    Live evidence: 1,568 identical 'footprint spike' stack dumps (55MB) in
    one afternoon, one per MLX generation.
    """

    def test_planned_boot_growth_below_soft_ceiling_is_not_an_incident(self):
        h = _Harness(boot_grace_s=300.0)
        h.dog._started_at = 0.0
        h.dog._sampler = iter(
            [_sample(core_mb=1_000.0), _sample(core_mb=9_500.0)]
        ).__next__

        with patch("core.resilience.memory_watchdog.time.monotonic", return_value=20.0):
            h.dog._tick()
            h.dog._tick()

        self.assertEqual(h.dog._spike_count, 0)

    def test_spike_dumps_are_throttled(self):
        h = _Harness()
        dumps: list[str] = []
        h.dog._dump_thread_stacks = dumps.append

        # Delta > 8192MB while staying below the soft ceiling, so only the
        # spike path (not the escalation ladder) can dump.
        samples = iter(
            [_sample(core_mb=100.0)]
            + [_sample(core_mb=9_000.0), _sample(core_mb=100.0)] * 6
        )
        h.dog._sampler = lambda: next(samples)
        for _ in range(13):
            h.dog._tick()

        self.assertEqual(len(dumps), 1, "spikes within the throttle window must not dump")
        self.assertIn("spike #1", dumps[0])
        self.assertEqual(h.dog._spike_count, 6, "every spike is still counted")

    def test_observer_provenance_upgrade_is_not_an_allocation_spike(self):
        h = _Harness()
        dumps: list[str] = []
        h.dog._dump_thread_stacks = dumps.append
        h.dog._last_sample = MemorySample(
            core_rss_mb=2_000.0,
            child_rss_mb=0.0,
            swap_used_gb=0.0,
            system_percent=30.0,
            total_ram_gb=64.0,
            sampled_at=1.0,
            observation_source="unavailable",
            observation_scenario_id="bootstrap",
        )
        h.dog._sampler = lambda: MemorySample(
            core_rss_mb=12_000.0,
            child_rss_mb=0.0,
            swap_used_gb=0.0,
            system_percent=40.0,
            total_ram_gb=64.0,
            sampled_at=2.0,
            observation_source="host",
            observation_scenario_id="live-host",
        )

        h.dog._tick()

        self.assertEqual(h.dog._spike_count, 0)
        self.assertEqual(dumps, [])

    def test_spike_dump_lifetime_cap(self):
        h = _Harness()
        dumps: list[str] = []
        h.dog._dump_thread_stacks = dumps.append
        h.dog._spike_dumps = h.dog.SPIKE_DUMP_LIFETIME_CAP

        h.dog._last_sample = _sample(core_mb=100.0)
        h.dog._sampler = lambda: _sample(core_mb=9_000.0)
        h.dog._tick()

        self.assertEqual(dumps, [], "past the lifetime cap no stack dump is written")
        self.assertEqual(h.dog._spike_count, 1)

    def test_spike_dumps_resume_after_throttle_window(self):
        h = _Harness()
        dumps: list[str] = []
        h.dog._dump_thread_stacks = dumps.append

        h.dog._record_footprint_spike(_sample(core_mb=1000.0), _sample(core_mb=21_000.0))
        # Pretend the last dump was long ago.
        h.dog._last_spike_dump_at -= h.dog.SPIKE_DUMP_MIN_INTERVAL_S + 1
        h.dog._record_footprint_spike(_sample(core_mb=1000.0), _sample(core_mb=21_000.0))

        self.assertEqual(len(dumps), 2)


class TestSpikeAttribution(unittest.TestCase):
    """A stack dump names where threads ARE; it cannot name what allocated.

    On 2026-07-29 this process gained 20.4GB in ten seconds. The only
    thread running at both stack samples was a MiniLM encode, measured
    afterwards at 3.7MB per two thousand calls — the stacks named a
    bystander, and the spike that mattered was throttled to a one-line
    'stack dump throttled' with no attribution at all.
    """

    def test_every_spike_is_attributed_even_when_the_stack_dump_is_throttled(self):
        h = _Harness()
        h.dog._dump_thread_stacks = lambda why: None
        attributed: list[str] = []
        h.dog._log_memory_attribution = attributed.append

        h.dog._record_footprint_spike(_sample(core_mb=1000.0), _sample(core_mb=21_000.0))
        # Second spike inside the throttle window: no stack dump, but the
        # attribution must still be recorded.
        h.dog._record_footprint_spike(_sample(core_mb=1000.0), _sample(core_mb=21_000.0))

        self.assertEqual(len(attributed), 2, "a throttled spike must still be attributed")
        self.assertIn("spike #2", attributed[1])

    def test_attribution_is_recorded_past_the_stack_dump_lifetime_cap(self):
        h = _Harness()
        h.dog._spike_dumps = h.dog.SPIKE_DUMP_LIFETIME_CAP + 1
        attributed: list[str] = []
        h.dog._log_memory_attribution = attributed.append

        h.dog._record_footprint_spike(_sample(core_mb=1000.0), _sample(core_mb=21_000.0))

        self.assertEqual(len(attributed), 1)

    def test_attribution_never_raises_when_the_infra_is_unavailable(self):
        h = _Harness()
        with patch.dict(sys.modules, {"core.runtime.memory_infra": None}):
            h.dog._log_memory_attribution("spike #1")  # must not raise


class TestLethalReclaimReporting(unittest.TestCase):
    def test_the_critical_line_reports_all_three_reclaim_levers(self):
        """'Reclaimed (killed=0)' hid whether shedding or gc found anything.

        It was the operator's last view of the runtime before it exited,
        and it described one of the three levers actually pulled.
        """
        h = _Harness()
        h.dog._ladder_shed = lambda: (3, 512 << 20)

        with self.assertLogs("Aura.Resilience.MemoryWatchdog", level="CRITICAL") as caught:
            h.dog._handle_lethal(_sample(core_mb=99_000.0), 0.0)

        line = "\n".join(caught.output)
        self.assertIn("shed=3 organs/512MB", line)
        self.assertIn("killed=1", line)
        self.assertIn("gc=42", line)


class TestRuntimeSurface(unittest.TestCase):
    def test_tick_survives_sampler_failure(self):
        import time

        h = _Harness()
        attempts: list[float] = []

        def _boom() -> MemorySample:
            attempts.append(time.monotonic())
            raise RuntimeError("psutil unavailable")

        h.dog._sampler = _boom
        with self.assertRaises(RuntimeError):
            h.dog._tick()
        self.assertEqual(len(attempts), 1)
        # The run loop catches this class of error; verify it is in the
        # recoverable set the loop guards against.
        from core.resilience.memory_watchdog import _WATCHDOG_RECOVERABLE_ERRORS

        self.assertTrue(issubclass(RuntimeError, _WATCHDOG_RECOVERABLE_ERRORS))

    def test_health_snapshot_shape(self):
        h = _Harness()
        h.dog._evaluate(_sample(core_mb=25_000.0), now=100.0)
        h.dog._last_sample = _sample(core_mb=25_000.0)
        snap = h.dog.health_snapshot()
        self.assertIn("thresholds", snap)
        self.assertIn("recent_actions", snap)
        self.assertEqual(snap["recent_actions"][-1]["tier"], "hard")
        self.assertAlmostEqual(snap["last_sample"]["core_rss_mb"], 25_000.0)

    def test_default_thresholds_scale_with_ram(self):
        with patch.dict(
            "os.environ",
            {
                "AURA_MEMWATCH_SOFT_MB": "",
                "AURA_MEMWATCH_HARD_MB": "",
                "AURA_MEMWATCH_LETHAL_MB": "",
                "AURA_MEMWATCH_SWAP_HARD_GB": "",
            },
            clear=False,
        ):
            full = _Thresholds.from_environment(64.0)
            half = _Thresholds.from_environment(32.0)
        self.assertAlmostEqual(full.soft_mb, 32768.0, delta=1.0)
        self.assertAlmostEqual(full.hard_mb, 64.0 * 1024.0 * 0.62, delta=1.0)
        self.assertAlmostEqual(full.lethal_mb, 64.0 * 1024.0 * 0.70, delta=1.0)
        self.assertAlmostEqual(full.swap_hard_gb, 7.68, delta=0.1)
        self.assertAlmostEqual(half.soft_mb, full.soft_mb / 2.0, delta=1.0)
        self.assertAlmostEqual(half.lethal_mb, full.lethal_mb / 2.0, delta=1.0)

    def test_memory_governor_daily_use_thresholds_align_with_watchdog(self):
        from types import SimpleNamespace

        from core.resilience.memory_governor import MemoryGovernor

        with patch.dict(
            "os.environ",
            {
                "AURA_GOVERNOR_PRUNE_MB": "",
                "AURA_GOVERNOR_UNLOAD_MB": "",
                "AURA_GOVERNOR_CRITICAL_MB": "",
            },
            clear=False,
        ):
            governor = MemoryGovernor(SimpleNamespace())

        self.assertLess(governor.threshold_prune, governor.threshold_unload)
        self.assertLess(governor.threshold_unload, governor.threshold_critical)
        self.assertLess(governor.threshold_critical, _Thresholds.from_environment(64.0).hard_mb)

    def test_singleton_start_stop(self):
        import core.resilience.memory_watchdog as mw

        try:
            dog = mw.start_memory_watchdog()
            self.assertTrue(dog.is_alive())
            self.assertIs(mw.start_memory_watchdog(), dog)
            self.assertIs(mw.get_memory_watchdog(), dog)
        finally:
            mw.stop_memory_watchdog()
        self.assertIsNone(mw.get_memory_watchdog())


if __name__ == "__main__":
    unittest.main()
