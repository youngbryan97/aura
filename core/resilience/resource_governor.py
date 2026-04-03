"""core/resilience/resource_governor.py
Resource Governor — Long-Term Stability Enforcement
=====================================================
Runs periodically (every 60s from the kernel supervisor) and prevents
unbounded memory growth that would cause crashes during multi-day uptime.

Responsibilities:
  1. Cap unbounded in-memory collections across subsystems
  2. SQLite WAL compaction and old-transition pruning for CognitiveLedger
  3. System RSS memory monitoring with emergency cleanup
  4. Background asyncio task cleanup (remove completed/dead tasks)

Design: stateless per-call.  Each ``govern()`` invocation inspects live
objects, trims what exceeds safe thresholds, and returns a report dict.
All external lookups are wrapped in try/except so the governor itself
never crashes the kernel.
"""
from __future__ import annotations

import asyncio
import gc
import logging
import os
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Aura.ResourceGovernor")

# ---------------------------------------------------------------------------
# Thresholds (tunable)
# ---------------------------------------------------------------------------
_COMMITMENTS_MAX = 500          # Max entries in CommitmentEngine._commitments
_ACTIVE_TASKS_MAX = 200         # Max entries in TaskCommitmentVerifier._active_tasks
_COUNTERFACTUAL_MAX = 200       # Max entries in CounterfactualEngine._records
_CONFIDENCE_HISTORY_MAX = 500   # MetacognitiveCalibrator.confidence_history
_BELIEFS_MAX = 2000             # WorldModelEngine.beliefs dict
_NARRATIVE_CHAPTERS_MAX = 50    # NarrativeIdentityEngine.chapters
_LEDGER_PRUNE_DAYS = 7          # Prune transitions older than this (keep snapshots)

_RSS_WARN_PERCENT = 80          # Trigger cleanup at this % of system RAM
_RSS_EMERGENCY_PERCENT = 90     # Emit emergency event at this %


class ResourceGovernor:
    """Periodic resource governance for long-running Aura processes."""

    def __init__(self, kernel: Any = None):
        self._kernel = kernel
        self._last_run: float = 0.0
        self._run_count: int = 0
        self._total_freed: int = 0
        logger.info("ResourceGovernor initialized.")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def govern(self) -> Dict[str, Any]:
        """Run all governance checks.  Returns a report dict."""
        t0 = time.monotonic()
        report: Dict[str, Any] = {
            "run": self._run_count,
            "timestamp": time.time(),
            "items_freed": 0,
            "ledger_compacted": False,
            "memory": {},
            "bg_tasks_cleaned": 0,
        }

        try:
            freed = self._cap_collections()
            report["items_freed"] = freed
            self._total_freed += freed
        except Exception as e:
            logger.warning("ResourceGovernor: _cap_collections error: %s", e)

        try:
            report["ledger_compacted"] = self._compact_ledger()
        except Exception as e:
            logger.warning("ResourceGovernor: _compact_ledger error: %s", e)

        try:
            report["memory"] = self._check_memory_pressure()
        except Exception as e:
            logger.warning("ResourceGovernor: _check_memory_pressure error: %s", e)

        try:
            if self._kernel is not None:
                tasks_list = getattr(self._kernel, "_background_tasks", None)
                if tasks_list is not None:
                    report["bg_tasks_cleaned"] = self._cleanup_background_tasks(tasks_list)
        except Exception as e:
            logger.warning("ResourceGovernor: _cleanup_background_tasks error: %s", e)

        elapsed_ms = (time.monotonic() - t0) * 1000
        report["elapsed_ms"] = round(elapsed_ms, 2)
        self._last_run = time.time()
        self._run_count += 1

        if report["items_freed"] > 0 or report["bg_tasks_cleaned"] > 0:
            logger.info(
                "ResourceGovernor: freed %d items, cleaned %d tasks, "
                "ledger=%s, RSS=%s (%.1fms)",
                report["items_freed"],
                report["bg_tasks_cleaned"],
                "compacted" if report["ledger_compacted"] else "ok",
                report.get("memory", {}).get("status", "unknown"),
                elapsed_ms,
            )
        else:
            logger.debug("ResourceGovernor: all clear (%.1fms)", elapsed_ms)

        return report

    # ------------------------------------------------------------------
    # 1. Cap unbounded collections
    # ------------------------------------------------------------------

    def _cap_collections(self) -> int:
        """Trim unbounded collections across subsystems.  Returns count freed."""
        freed = 0

        # --- CommitmentEngine._commitments dict ---
        freed += self._trim_commitment_engine()

        # --- TaskCommitmentVerifier._active_tasks dict ---
        freed += self._trim_task_verifier()

        # --- CounterfactualEngine._records list ---
        freed += self._trim_counterfactual_engine()

        # --- MetacognitiveCalibrator.confidence_history list ---
        freed += self._trim_metacognitive_calibrator()

        # --- WorldModelEngine.beliefs dict ---
        freed += self._trim_world_model()

        # --- NarrativeIdentityEngine.chapters list ---
        freed += self._trim_narrative_identity()

        return freed

    def _trim_commitment_engine(self) -> int:
        """Evict completed/broken commitments beyond the cap."""
        try:
            from core.agency.commitment_engine import get_commitment_engine
            engine = get_commitment_engine()
            commits = engine._commitments
            if len(commits) <= _COMMITMENTS_MAX:
                return 0

            # Evict non-active (fulfilled/broken) first, oldest first
            non_active = sorted(
                [
                    (cid, c)
                    for cid, c in commits.items()
                    if c.status.value in ("fulfilled", "broken")
                ],
                key=lambda pair: pair[1].created_at,
            )
            to_remove = len(commits) - _COMMITMENTS_MAX
            removed = 0
            for cid, _ in non_active[:to_remove]:
                del commits[cid]
                removed += 1

            # If still over cap, evict oldest active
            if len(commits) > _COMMITMENTS_MAX:
                active_sorted = sorted(
                    list(commits.items()),
                    key=lambda pair: pair[1].created_at,
                )
                excess = len(commits) - _COMMITMENTS_MAX
                for cid, _ in active_sorted[:excess]:
                    del commits[cid]
                    removed += 1

            if removed > 0:
                logger.info("ResourceGovernor: trimmed %d commitments", removed)
            return removed
        except Exception as e:
            logger.debug("ResourceGovernor: commitment trim skipped: %s", e)
            return 0

    def _trim_task_verifier(self) -> int:
        """Evict completed/failed task records beyond the cap."""
        try:
            from core.agency.task_commitment_verifier import get_task_commitment_verifier
            verifier = get_task_commitment_verifier()
            tasks = verifier._active_tasks
            if len(tasks) <= _ACTIVE_TASKS_MAX:
                return 0

            # Remove terminal tasks first (completed, failed, capability_gap)
            terminal_statuses = {"completed", "failed", "capability_gap"}
            terminal_ids = [
                tid
                for tid, t in tasks.items()
                if t.get("status") in terminal_statuses
            ]
            # Sort by started_at so we evict oldest first
            terminal_ids.sort(key=lambda tid: tasks[tid].get("started_at", 0))

            to_remove = len(tasks) - _ACTIVE_TASKS_MAX
            removed = 0
            for tid in terminal_ids[:to_remove]:
                del tasks[tid]
                removed += 1

            # If still over, evict oldest regardless of status
            if len(tasks) > _ACTIVE_TASKS_MAX:
                all_sorted = sorted(tasks.keys(), key=lambda tid: tasks[tid].get("started_at", 0))
                excess = len(tasks) - _ACTIVE_TASKS_MAX
                for tid in all_sorted[:excess]:
                    del tasks[tid]
                    removed += 1

            if removed > 0:
                logger.info("ResourceGovernor: trimmed %d task records", removed)
            return removed
        except Exception as e:
            logger.debug("ResourceGovernor: task verifier trim skipped: %s", e)
            return 0

    def _trim_counterfactual_engine(self) -> int:
        """Trim _records list if above cap (already capped at 200 inline, but enforce here too)."""
        try:
            from core.consciousness.counterfactual_engine import get_counterfactual_engine
            engine = get_counterfactual_engine()
            records = engine._records
            if len(records) <= _COUNTERFACTUAL_MAX:
                return 0
            excess = len(records) - _COUNTERFACTUAL_MAX
            del records[:excess]
            logger.info("ResourceGovernor: trimmed %d counterfactual records", excess)
            return excess
        except Exception as e:
            logger.debug("ResourceGovernor: counterfactual trim skipped: %s", e)
            return 0

    def _trim_metacognitive_calibrator(self) -> int:
        """Trim confidence_history if above cap."""
        try:
            from core.container import ServiceContainer
            calibrator = ServiceContainer.get("metacognitive_calibrator", default=None)
            if calibrator is None:
                return 0
            history = calibrator.confidence_history
            if len(history) <= _CONFIDENCE_HISTORY_MAX:
                return 0
            excess = len(history) - _CONFIDENCE_HISTORY_MAX
            del history[:excess]
            logger.info("ResourceGovernor: trimmed %d confidence_history entries", excess)
            return excess
        except Exception as e:
            logger.debug("ResourceGovernor: metacognitive trim skipped: %s", e)
            return 0

    def _trim_world_model(self) -> int:
        """Evict lowest-confidence beliefs if above cap."""
        try:
            from core.container import ServiceContainer
            wm = ServiceContainer.get("world_model", default=None)
            if wm is None:
                return 0
            beliefs = wm.beliefs
            if len(beliefs) <= _BELIEFS_MAX:
                return 0

            # Sort by confidence ascending, evict lowest
            sorted_keys = sorted(beliefs.keys(), key=lambda k: beliefs[k].confidence)
            to_remove = len(beliefs) - _BELIEFS_MAX
            removed = 0
            for key in sorted_keys[:to_remove]:
                del beliefs[key]
                removed += 1

            if removed > 0:
                logger.info("ResourceGovernor: trimmed %d low-confidence beliefs", removed)
                # Persist the trimmed set
                if hasattr(wm, "_save_beliefs"):
                    wm._save_beliefs()
            return removed
        except Exception as e:
            logger.debug("ResourceGovernor: world model trim skipped: %s", e)
            return 0

    def _trim_narrative_identity(self) -> int:
        """Trim chapters if above cap."""
        try:
            from core.container import ServiceContainer
            ni = ServiceContainer.get("narrative_identity", default=None)
            if ni is None:
                return 0
            chapters = ni.chapters
            if len(chapters) <= _NARRATIVE_CHAPTERS_MAX:
                return 0
            excess = len(chapters) - _NARRATIVE_CHAPTERS_MAX
            ni.chapters = chapters[excess:]
            if hasattr(ni, "_save_narrative"):
                ni._save_narrative()
            logger.info("ResourceGovernor: trimmed %d narrative chapters", excess)
            return excess
        except Exception as e:
            logger.debug("ResourceGovernor: narrative trim skipped: %s", e)
            return 0

    # ------------------------------------------------------------------
    # 2. SQLite WAL compaction
    # ------------------------------------------------------------------

    def _compact_ledger(self) -> bool:
        """WAL checkpoint + prune old transitions.  Returns True if compaction ran."""
        try:
            from core.resilience.cognitive_ledger import get_cognitive_ledger
            ledger = get_cognitive_ledger()
            conn = ledger._conn
            if conn is None:
                return False

            compacted = False

            # WAL checkpoint (TRUNCATE reclaims disk space)
            with ledger._lock:
                try:
                    conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
                    compacted = True
                except Exception as e:
                    logger.debug("ResourceGovernor: WAL checkpoint failed: %s", e)

            # Prune transitions older than _LEDGER_PRUNE_DAYS (keep snapshots)
            cutoff_ts = time.time() - (_LEDGER_PRUNE_DAYS * 86400)
            with ledger._lock:
                try:
                    cursor = conn.execute(
                        "SELECT COUNT(*) FROM transitions WHERE ts < ?",
                        (cutoff_ts,),
                    )
                    old_count = cursor.fetchone()[0]
                    if old_count > 0:
                        conn.execute(
                            "DELETE FROM transitions WHERE ts < ?",
                            (cutoff_ts,),
                        )
                        conn.commit()
                        # Update the cached count
                        cursor = conn.execute("SELECT COUNT(*) FROM transitions")
                        ledger._transition_count = cursor.fetchone()[0]
                        logger.info(
                            "ResourceGovernor: pruned %d ledger transitions older than %d days",
                            old_count,
                            _LEDGER_PRUNE_DAYS,
                        )
                        compacted = True
                except Exception as e:
                    logger.debug("ResourceGovernor: ledger prune failed: %s", e)

            return compacted
        except Exception as e:
            logger.debug("ResourceGovernor: ledger compaction skipped: %s", e)
            return False

    # ------------------------------------------------------------------
    # 3. System memory monitoring
    # ------------------------------------------------------------------

    def _check_memory_pressure(self) -> Dict[str, Any]:
        """Check RSS vs system RAM.  Trigger cleanup if thresholds exceeded."""
        result: Dict[str, Any] = {"status": "unknown"}

        try:
            import psutil
        except ImportError:
            result["status"] = "psutil_unavailable"
            return result

        try:
            process = psutil.Process(os.getpid())
            rss_bytes = process.memory_info().rss
            total_bytes = psutil.virtual_memory().total
            rss_pct = (rss_bytes / total_bytes) * 100 if total_bytes > 0 else 0

            result.update({
                "rss_mb": round(rss_bytes / (1024 * 1024), 1),
                "total_mb": round(total_bytes / (1024 * 1024), 1),
                "rss_percent": round(rss_pct, 1),
            })

            if rss_pct >= _RSS_EMERGENCY_PERCENT:
                result["status"] = "emergency"
                logger.critical(
                    "ResourceGovernor: EMERGENCY RSS %.1f%% of system RAM "
                    "(%d MB / %d MB). Triggering emergency cleanup.",
                    rss_pct,
                    rss_bytes // (1024 * 1024),
                    total_bytes // (1024 * 1024),
                )
                self._emergency_cleanup()
                self._emit_emergency_event(rss_pct)

            elif rss_pct >= _RSS_WARN_PERCENT:
                result["status"] = "warning"
                logger.warning(
                    "ResourceGovernor: HIGH RSS %.1f%% of system RAM "
                    "(%d MB / %d MB). Running cache cleanup.",
                    rss_pct,
                    rss_bytes // (1024 * 1024),
                    total_bytes // (1024 * 1024),
                )
                self._emergency_cleanup()

            else:
                result["status"] = "healthy"

        except Exception as e:
            result["status"] = f"error: {e}"
            logger.debug("ResourceGovernor: memory check failed: %s", e)

        return result

    def _emergency_cleanup(self):
        """Clear non-essential caches and force garbage collection."""
        # 1. Force Python GC (all generations)
        collected = gc.collect(2)
        logger.info("ResourceGovernor: gc.collect() freed %d objects", collected)

        # 2. Clear any module-level caches we know about
        try:
            from core.container import ServiceContainer

            # Clear LLM router's reflex cache if present
            router = ServiceContainer.get("llm_router", default=None)
            if router and hasattr(router, "_cache"):
                cache = router._cache
                if hasattr(cache, "clear"):
                    cache.clear()
                    logger.info("ResourceGovernor: cleared LLM router cache")

            # Clear context assembler cache if present
            try:
                from core.brain.llm.context_assembler import ContextAssembler
                if hasattr(ContextAssembler, "_cache"):
                    ContextAssembler._cache.clear()
            except Exception as _exc:
                logger.debug("Suppressed Exception: %s", _exc)

        except Exception as e:
            logger.debug("ResourceGovernor: cache clearing error: %s", e)

    def _emit_emergency_event(self, rss_pct: float):
        """Emit an emergency event to the event bus."""
        try:
            from core.container import ServiceContainer
            bus = ServiceContainer.get("event_bus", default=None)
            if bus is None:
                return
            # Use publish_threadsafe if available (sync context), otherwise log
            payload = {
                "type": "resource_emergency",
                "rss_percent": round(rss_pct, 1),
                "timestamp": time.time(),
                "source": "ResourceGovernor",
            }
            if hasattr(bus, "publish_threadsafe"):
                bus.publish_threadsafe("aura.resource_emergency", payload)
            elif hasattr(bus, "publish_sync"):
                bus.publish_sync("aura.resource_emergency", payload)
            logger.critical(
                "ResourceGovernor: EMERGENCY EVENT emitted (RSS=%.1f%%)", rss_pct
            )
        except Exception as e:
            logger.debug("ResourceGovernor: emergency event emission failed: %s", e)

    # ------------------------------------------------------------------
    # 4. Background task cleanup
    # ------------------------------------------------------------------

    def _cleanup_background_tasks(self, tasks: List) -> int:
        """Remove completed/cancelled tasks from a task list.  Returns count removed."""
        if not tasks:
            return 0

        to_remove = []
        for task in tasks:
            if not isinstance(task, asyncio.Task):
                continue
            if task.done():
                # Log unexpected exceptions before removal
                try:
                    exc = task.exception()
                    if exc is not None:
                        logger.debug(
                            "ResourceGovernor: removing dead task '%s' (exception: %s)",
                            task.get_name(),
                            exc,
                        )
                except (asyncio.CancelledError, asyncio.InvalidStateError):
                    logger.debug("Suppressed bare exception")
                    pass
                to_remove.append(task)

        for task in to_remove:
            try:
                tasks.remove(task)
            except ValueError:
                pass  # Already removed by supervisor

        if to_remove:
            logger.debug("ResourceGovernor: cleaned %d completed background tasks", len(to_remove))

        return len(to_remove)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Return governor status for diagnostics."""
        return {
            "run_count": self._run_count,
            "total_freed": self._total_freed,
            "last_run": self._last_run,
            "last_run_ago_s": round(time.time() - self._last_run, 1) if self._last_run else None,
        }
