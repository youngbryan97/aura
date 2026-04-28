"""
core/consciousness/loop_monitor.py
====================================
Patch 3 of 3 — Consciousness Loop Health Monitor

THE GAP (confirmed by reading source):
  QualiaSynthesizer._push_bridges() calls:
    ServiceContainer.get("affect_engine").receive_qualia_echo(...)

  AffectEngineV2.receive_qualia_echo() is fully implemented.

  BUT: the Heartbeat's _qualia_synthesizer property caches on first access:
    if not hasattr(self, '_qualia_cache'):
        self._qualia_cache = ServiceContainer.get("qualia_synthesizer", default=None)

  If ServiceContainer hasn't registered "qualia_synthesizer" by the time
  the first Heartbeat tick fires, _qualia_cache is set to None — and it
  STAYS None for the rest of the session. The synthesizer starts, the
  bridge never fires, and the only signal is a DEBUG-level log line buried
  in _push_bridges. Nothing in the log tells you the loop is dead.

  Same risk exists for "affect_engine" — if it registers under a slightly
  different key or hasn't been created yet, the bridge silently no-ops.

WHAT THIS PATCH DOES:
  ConsciousnessLoopMonitor — a lightweight async monitor that:

  1. Runs a slow health-check loop (every HEALTH_CHECK_INTERVAL seconds)
     that verifies:
       a. ServiceContainer["qualia_synthesizer"] is present and has a
          non-zero _tick counter (i.e. synthesize() has been called)
       b. ServiceContainer["affect_engine"] is present and has
          receive_qualia_echo
       c. The stale cache problem — if the Heartbeat's _qualia_cache is
          None despite the synthesizer now being registered, it clears the
          cache so the next tick re-fetches

  2. Emits a WARNING (not DEBUG) when any link is broken, with a
     human-readable diagnosis and suggested fix

  3. Tracks when the loop last fired successfully and exposes
     get_status() for the dashboard / unified_audit

  4. Can trigger a single manual bridge fire to verify end-to-end
     connectivity without waiting for the next synthesizer tick

INSTALL:
  from core.consciousness.loop_monitor import ConsciousnessLoopMonitor
  monitor = ConsciousnessLoopMonitor(orchestrator)
  monitor.start()          # starts background task
  orchestrator.loop_monitor = monitor

  Or via apply_consciousness_patches() which handles all three patches.
"""
from __future__ import annotations
from core.runtime.errors import record_degradation



import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Aura.LoopMonitor")

# How often the health-check loop runs (seconds).
# Slow enough not to add noise; fast enough to catch startup races.
HEALTH_CHECK_INTERVAL = 45.0

# How many consecutive healthy checks before we log a positive confirmation.
_CONFIRM_AFTER = 4


class ConsciousnessLoopMonitor:
    """
    Watches the qualia→affect bridge and the Heartbeat's stale-cache risk.

    Runs as a background asyncio task. Emits WARNING on broken links,
    attempts self-healing where possible, tracks health history.
    """

    def __init__(self, orchestrator: Optional[Any] = None) -> None:
        self.orchestrator = orchestrator
        self._task:             Optional[asyncio.Task] = None
        self._running:          bool  = False
        self._last_healthy_at:  float = 0.0
        self._last_check_at:    float = 0.0
        self._consecutive_healthy: int = 0
        self._issue_log:        List[Dict[str, Any]] = []
        self._healed_count:     int = 0

    # ─────────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background monitoring task."""
        if self._running:
            return
        self._running = True
        from core.utils.task_tracker import get_task_tracker

        self._task = get_task_tracker().create_task(
            self._loop(),
            name="ConsciousnessLoopMonitor",
        )
        logger.info("🔍 ConsciousnessLoopMonitor started (interval=%.0fs)", HEALTH_CHECK_INTERVAL)

    def stop(self) -> None:
        """Cancel the monitoring task gracefully."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
        logger.info("🔍 ConsciousnessLoopMonitor stopped")

    # ─────────────────────────────────────────────────────────────────────────
    # Main loop
    # ─────────────────────────────────────────────────────────────────────────

    async def _loop(self) -> None:
        # Brief initial delay — let the orchestrator finish registering services
        await asyncio.sleep(12.0)

        while self._running:
            try:
                issues = await self._run_checks()
                self._last_check_at = time.time()

                if issues:
                    self._consecutive_healthy = 0
                    for issue in issues:
                        self._issue_log.append(issue)
                        logger.warning("⚠️  LoopMonitor: %s", issue["message"])
                        if issue.get("fix_attempted"):
                            logger.info("🔧 LoopMonitor: self-heal attempted — %s",
                                        issue["fix"])
                else:
                    self._consecutive_healthy += 1
                    self._last_healthy_at = time.time()
                    if self._consecutive_healthy == _CONFIRM_AFTER:
                        logger.info(
                            "✅ LoopMonitor: consciousness loop healthy "
                            "(%d consecutive clean checks)", _CONFIRM_AFTER
                        )

                # Cap issue log
                if len(self._issue_log) > 200:
                    self._issue_log = self._issue_log[-200:]

            except asyncio.CancelledError:
                break
            except Exception as exc:
                record_degradation('loop_monitor', exc)
                logger.debug("LoopMonitor._loop error: %s", exc)

            await asyncio.sleep(HEALTH_CHECK_INTERVAL)

    # ─────────────────────────────────────────────────────────────────────────
    # Health checks
    # ─────────────────────────────────────────────────────────────────────────

    async def _run_checks(self) -> List[Dict[str, Any]]:
        """Run all checks. Returns list of issue dicts (empty = healthy)."""
        issues: List[Dict[str, Any]] = []

        sc = self._get_service_container()
        if sc is None:
            # Can't check without ServiceContainer — not an error at startup
            return []

        # ── Check 1: qualia_synthesizer is registered ─────────────────────────
        synth = self._get(sc, "qualia_synthesizer")
        if synth is None:
            issues.append({
                "check": "qualia_synthesizer_registered",
                "status": "error",
                "message": "ServiceContainer['qualia_synthesizer'] is None — QualiaSynthesizer not yet registered. Bridge cannot fire.",
                "remedy": "Ensure _init_cognitive_architecture registers QualiaSynthesizer early.",
                "critical": True,
                "timestamp":     time.time(),
            })
        else:
            # ── Check 2: synthesizer has actually ticked ──────────────────────
            tick = getattr(synth, "_tick", None)
            if tick == 0:
                issues.append({
                    "check":         "qualia_synthesizer_ticking",
                    "message":       "QualiaSynthesizer registered but _tick=0 — "
                                     "synthesize() has never been called. "
                                     "LiquidSubstrate may not be providing substrate_metrics.",
                    "fix":           "Verify LiquidSubstrate is registered and the Heartbeat "
                                     "is passing qualia_metrics to the synthesizer.",
                    "fix_attempted": False,
                    "timestamp":     time.time(),
                })

        # ── Check 3: affect_engine is registered ──────────────────────────────
        affect = self._get(sc, "affect_engine")
        if affect is None:
            issues.append({
                "check":         "affect_engine_registered",
                "message":       "ServiceContainer['affect_engine'] is None — "
                                 "AffectEngineV2 not registered. receive_qualia_echo() "
                                 "can never be called.",
                "fix":           "Ensure AffectEngineV2 is registered as 'affect_engine' "
                                 "in the ServiceContainer before the qualia loop starts.",
                "fix_attempted": False,
                "timestamp":     time.time(),
            })
        elif not hasattr(affect, "receive_qualia_echo"):
            issues.append({
                "check":         "affect_engine_interface",
                "message":       "affect_engine is registered but lacks receive_qualia_echo(). "
                                 "Wrong engine type or incomplete initialisation.",
                "fix":           "Verify AffectEngineV2 (not V1) is registered.",
                "fix_attempted": False,
                "timestamp":     time.time(),
            })

        # ── Check 4: Stale-cache self-heal ────────────────────────────────────
        #
        # The Heartbeat caches _qualia_cache = None if ServiceContainer didn't
        # have the synthesizer when the first tick fired. If we now find both
        # the synthesizer IS present but the Heartbeat's cache is still None,
        # we clear the cache attr so the next tick re-fetches.
        if synth is not None:
            healed = self._try_heal_stale_cache(sc, synth)
            if healed:
                self._healed_count += 1
                issues.append({
                    "check":         "stale_cache_healed",
                    "message":       "Heartbeat had stale _qualia_cache=None despite "
                                     "qualia_synthesizer being registered. Cache cleared — "
                                     "bridge will reactivate on next tick.",
                    "fix":           "Cache attr deleted from Heartbeat instance.",
                    "fix_attempted": True,
                    "timestamp":     time.time(),
                })

        # ── Check 5: End-to-end connectivity probe (once per 4 checks) ────────
        if (
            synth is not None
            and affect is not None
            and hasattr(affect, "receive_qualia_echo")
            and self._consecutive_healthy % 4 == 0
            and self._consecutive_healthy > 0
        ):
            try:
                # Fire a negligibly small probe echo to verify the path
                affect.receive_qualia_echo(q_norm=0.001, pri=0.001, trend=0.0)
                logger.debug("LoopMonitor: end-to-end probe fired successfully")
            except Exception as exc:
                record_degradation('loop_monitor', exc)
                issues.append({
                    "check":         "end_to_end_probe",
                    "message":       f"receive_qualia_echo() raised an exception: {exc}",
                    "fix":           "Check AffectEngineV2.receive_qualia_echo implementation.",
                    "fix_attempted": False,
                    "timestamp":     time.time(),
                })

        return issues

    # ─────────────────────────────────────────────────────────────────────────
    # Self-healing: stale cache
    # ─────────────────────────────────────────────────────────────────────────

    def _try_heal_stale_cache(self, sc: Any, live_synth: Any) -> bool:
        """
        If the Heartbeat (ConsciousnessHeartbeat) has _qualia_cache=None
        while the ServiceContainer actually has a live synthesizer, delete
        the cache attr so the next tick does a fresh lookup.

        Returns True if healing was performed.
        """
        heartbeat = self._get(sc, "heartbeat")
        if heartbeat is None:
            # Try orchestrator attribute
            heartbeat = getattr(self.orchestrator, "heartbeat", None)
        if heartbeat is None:
            return False

        cached = getattr(heartbeat, "_qualia_cache", _SENTINEL)
        if cached is _SENTINEL:
            # attr doesn't exist on this heartbeat — no stale cache risk
            return False
        if cached is None:
            # Stale None cache — clear it
            try:
                delattr(heartbeat, "_qualia_cache")
                logger.debug("LoopMonitor: cleared stale _qualia_cache on Heartbeat")
                return True
            except Exception:
                return False
        # Cache is already live — no heal needed
        return False

    # ─────────────────────────────────────────────────────────────────────────
    # Manual probe
    # ─────────────────────────────────────────────────────────────────────────

    async def probe_now(self) -> Dict[str, Any]:
        """
        Run a health check immediately and return a summary dict.
        Useful for debugging or dashboard refresh.
        """
        issues = await self._run_checks()
        return {
            "healthy":       len(issues) == 0,
            "issues":        issues,
            "checked_at":    time.time(),
            "last_healthy":  self._last_healthy_at,
            "healed_count":  self._healed_count,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Status
    # ─────────────────────────────────────────────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        now = time.time()
        recent_issues = [
            i for i in self._issue_log
            if now - i.get("timestamp", 0) < 3600
        ]
        return {
            "running":              self._running,
            "last_check_ago_s":     round(now - self._last_check_at, 1) if self._last_check_at else None,
            "last_healthy_ago_s":   round(now - self._last_healthy_at, 1) if self._last_healthy_at else None,
            "consecutive_healthy":  self._consecutive_healthy,
            "healed_count":         self._healed_count,
            "issues_last_hour":     len(recent_issues),
            "recent_issues":        [i["check"] for i in recent_issues[-5:]],
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _get_service_container(self) -> Optional[Any]:
        try:
            from core.container import ServiceContainer
            return ServiceContainer
        except Exception:
            return None

    def _get(self, sc: Any, key: str) -> Optional[Any]:
        try:
            if sc is None:
                return None
            result = sc.get(key, default=None)
            return result
        except Exception as e:
            record_degradation('loop_monitor', e)
            # Degrade gracefully if ServiceContainer API changes
            logger.debug(f"LoopMonitor: _get({key}) failed: {e}")
            return None


# Sentinel for attr-existence check
_SENTINEL = object()


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

_monitor_instance: Optional[ConsciousnessLoopMonitor] = None


def get_loop_monitor(orchestrator: Optional[Any] = None) -> ConsciousnessLoopMonitor:
    """Return the singleton ConsciousnessLoopMonitor, creating if needed."""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = ConsciousnessLoopMonitor(orchestrator)
    elif orchestrator is not None and _monitor_instance.orchestrator is None:
        _monitor_instance.orchestrator = orchestrator
    return _monitor_instance
