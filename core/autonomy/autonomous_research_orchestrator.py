"""core/autonomy/autonomous_research_orchestrator.py
─────────────────────────────────────────────────────
End-to-end orchestrator for autonomous content engagement. Wires together
all the autonomy modules: scheduler → router → fetcher → comprehension →
reflection → depth gate → memory persister → progress tracker.

Operating modes
---------------
- ``run_once()`` — engage with exactly one item end-to-end. Useful for
  testing or for serial-paced curiosity.
- ``run_loop()`` — long-running async background task. Picks one item
  at a time, sleeps between cycles, respects shutdown signals.

Session resume:
- After comprehension begins, the orchestrator writes a ``session.json``
  checkpoint to the runtime data directory. If interrupted, the next run can
  reload unfinished session evidence without mutating the source checkout.

Concurrency: one engagement at a time per orchestrator instance. Multiple
orchestrators can run in parallel if Bryan wants pipelined throughput.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import time
import traceback
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.autonomy.comprehension_loop import ComprehensionLoop, ComprehensionRecord
from core.autonomy.content_fetcher import ContentFetcher
from core.autonomy.content_method_router import MethodRouter
from core.autonomy.content_progress_tracker import (
    ProgressEntry,
    ProgressLog,
)
from core.autonomy.content_progress_tracker import (
    load as load_progress,
)
from core.autonomy.curiosity_scheduler import CuriosityScheduler, SchedulingDecision
from core.autonomy.depth_gate import DepthGate, DepthReport
from core.autonomy.memory_persister import (
    CommitReceipt,
    EpisodicEvent,
    MemoryPersister,
)
from core.autonomy.reflection_loop import ReflectionLoop, ReflectionRecord
from core.runtime.atomic_writer import atomic_write_text
from core.runtime.errors import Severity, record_degradation
from core.runtime.lockdep import checked_async_lock
from core.runtime.task_ownership import create_tracked_task

logger = logging.getLogger("Aura.AutonomousResearchOrchestrator")


def _quarantine_corrupt_progress() -> Path | None:
    """Move an unreadable progress log aside so it is not overwritten.

    Returns the quarantine path, or None if nothing could be moved. Never
    raises: this runs on a failure path and must not turn a recoverable
    corruption into a crash.
    """
    try:
        from core.autonomy.content_progress_tracker import _default_progress_path

        source = _default_progress_path()
        if not source.exists():
            return None
        target = source.with_suffix(f"{source.suffix}.corrupt.{int(time.time())}")
        source.replace(target)
        logger.warning("Quarantined unreadable progress log to %s", target)
        return target
    except (OSError, ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        logger.error("Could not quarantine the corrupt progress log: %s", exc)
        return None


def _default_sessions_dir() -> Path:
    override = os.environ.get("AURA_RESEARCH_SESSIONS_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".aura/data/autonomy/research-sessions"


SESSIONS_DIR = _default_sessions_dir()

#: The only phase meaning the engagement genuinely finished.
_TERMINAL_SESSION_PHASES = frozenset({"complete"})

#: Phases from which resuming is meaningful. persist_failed is deliberately
#: here: the research WAS done and only the memory commit failed, which is the
#: most worthwhile thing to retry and the easiest to lose.
_RESUMABLE_SESSION_PHASES = frozenset({
    "fetched", "comprehended", "reflected", "gated", "persisted",
    "persist_failed",
})
DEFAULT_LOOP_INTERVAL = 600.0   # 10 minutes between engagements
DEFAULT_MAX_CONSECUTIVE_FAILURES = 3
_ENGAGEMENT_RECOVERABLE_ERRORS = (
    sqlite3.Error,
    OSError,
    RuntimeError,
    AttributeError,
    TypeError,
    ValueError,
)


def _record_research_degradation(
    error: BaseException,
    *,
    action: str,
    severity: Severity = "degraded",
    extra: dict[str, Any] | None = None,
) -> None:
    record_degradation(
        "autonomous_research_orchestrator",
        error,
        severity=severity,
        action=action,
        extra=extra,
    )


@dataclass
class EngagementResult:
    """Single engagement's outcome."""
    item_title: str
    started_at: float
    completed_at: float | None = None
    decision: dict[str, Any] | None = None
    sources_engaged: list[str] = field(default_factory=list)
    priority_levels_engaged: list[int] = field(default_factory=list)
    depth_passed: bool = False
    depth_score: float = 0.0
    depth_failures: list[str] = field(default_factory=list)
    persist_receipt: dict[str, Any] | None = None
    #: Whether the memory commit was ACCEPTED. A rejected receipt used to be
    #: recorded and then ignored, so a completely failed commit still read as
    #: completed research.
    persisted: bool = True
    #: True when the depth gate failed and the model-generated facts and belief
    #: updates were therefore withheld from semantic memory.
    epistemic_content_withheld: bool = False
    inference_failures: int = 0
    error: str | None = None
    session_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_title": self.item_title,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "decision": self.decision,
            "sources_engaged": list(self.sources_engaged),
            "priority_levels_engaged": list(self.priority_levels_engaged),
            "depth_passed": self.depth_passed,
            "depth_score": round(self.depth_score, 3),
            "depth_failures": list(self.depth_failures),
            "persist_receipt": self.persist_receipt,
            "persisted": self.persisted,
            "epistemic_content_withheld": self.epistemic_content_withheld,
            "inference_failures": self.inference_failures,
            "error": self.error,
            "session_id": self.session_id,
        }


class AutonomousResearchOrchestrator:
    def __init__(
        self,
        scheduler: CuriosityScheduler | None = None,
        router: MethodRouter | None = None,
        fetcher: ContentFetcher | None = None,
        comprehension: ComprehensionLoop | None = None,
        reflection: ReflectionLoop | None = None,
        gate: DepthGate | None = None,
        persister: MemoryPersister | None = None,
        sessions_dir: Path | None = None,
        loop_interval: float = DEFAULT_LOOP_INTERVAL,
        max_consecutive_failures: int = DEFAULT_MAX_CONSECUTIVE_FAILURES,
        on_engagement_complete: Callable[[EngagementResult], None] | None = None,
    ) -> None:
        # The class documents a one-engagement-at-a-time contract and had
        # nothing enforcing it: concurrent run_once callers on one instance
        # could pick the SAME item and then race the shared scheduler, progress
        # log, cache, queue, and memory state. The lock is what makes the
        # documented guarantee true.
        self._engagement_lock = checked_async_lock("autonomous_research_orchestrator")
        self._scheduler = scheduler or CuriosityScheduler()
        self._router = router or MethodRouter()
        self._fetcher = fetcher or ContentFetcher()
        self._comprehension = comprehension or ComprehensionLoop()
        self._reflection = reflection or ReflectionLoop()
        self._gate = gate or DepthGate()
        self._persister = persister or MemoryPersister()
        self._sessions_dir = Path(sessions_dir).expanduser() if sessions_dir is not None else _default_sessions_dir()
        self._loop_interval = loop_interval
        self._max_failures = max_consecutive_failures
        self._on_complete = on_engagement_complete
        self._running = False
        self._task: asyncio.Task | None = None
        self._consecutive_failures = 0

    # ── Public API ────────────────────────────────────────────────────────

    async def run_once(self) -> EngagementResult | None:
        """Engage one item. Serialized: selection and engagement are one
        critical section, so two callers cannot select the same item."""
        async with self._engagement_lock:
            decision = self._scheduler.pick_next()
            if decision is None:
                logger.info("scheduler returned no candidate; nothing to do")
                return None
            return await self._engage(decision)

    async def start_loop(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = create_tracked_task(
            self._loop(),
            name="autonomous_research_orchestrator.loop",
        )

    async def stop_loop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except (TimeoutError, asyncio.CancelledError):
                pass  # no-op: intentional

    # ── Internal loop ─────────────────────────────────────────────────────

    async def _loop(self) -> None:
        try:
            while self._running:
                try:
                    result = await self.run_once()
                    if result is None:
                        await asyncio.sleep(self._loop_interval)
                        continue
                    if result.error:
                        self._consecutive_failures += 1
                    else:
                        self._consecutive_failures = 0
                    if self._consecutive_failures >= self._max_failures:
                        logger.warning(
                            "halting research loop after %d consecutive failures",
                            self._consecutive_failures,
                        )
                        self._running = False
                        return
                except asyncio.CancelledError:
                    raise
                except _ENGAGEMENT_RECOVERABLE_ERRORS as e:
                    _record_research_degradation(
                        e,
                        action=(
                            "counted the crashed research-loop iteration as a failed "
                            "engagement, preserved the loop failure budget, and will "
                            "back off or stop at the configured threshold"
                        ),
                        extra={
                            "consecutive_failures_after": self._consecutive_failures + 1,
                            "max_consecutive_failures": self._max_failures,
                            "phase": "loop_iteration",
                        },
                    )
                    logger.error("research loop iteration crashed: %s\n%s", e, traceback.format_exc())
                    self._consecutive_failures += 1
                    if self._consecutive_failures >= self._max_failures:
                        logger.warning(
                            "halting research loop after %d consecutive failures",
                            self._consecutive_failures,
                        )
                        self._running = False
                        return

                await asyncio.sleep(self._loop_interval)
        except asyncio.CancelledError:
            return

    # ── Single engagement ─────────────────────────────────────────────────

    async def _engage(self, decision: SchedulingDecision) -> EngagementResult:
        session_id = uuid.uuid4().hex[:12]
        result = EngagementResult(
            item_title=decision.item.title,
            started_at=time.time(),
            decision=decision.to_dict(),
            session_id=session_id,
        )
        session_path = self._sessions_dir / f"{session_id}.json"

        try:
            # 1. Plan fetches
            plan = self._router.plan(decision.item, decision.top_priority_level)
            self._save_session(session_path, {"phase": "planned", "result": result.to_dict(),
                                              "plan": {"attempts": [a.__dict__ for a in plan.attempts]}})

            # 2. Fetch
            execution = await self._fetcher.execute(plan)
            result.sources_engaged = execution.all_sources()
            result.priority_levels_engaged = execution.priority_levels_engaged()
            self._save_session(session_path, {"phase": "fetched", "result": result.to_dict(),
                                              "fetch": {
                                                  "successful_count": len(execution.successful),
                                                  "failed_count": len(execution.failed),
                                              }})

            if not execution.successful:
                result.error = "no fetch attempt succeeded"
                result.completed_at = time.time()
                self._record_scheduler_attempt(decision, outcome="abandoned", session_id=session_id)
                self._save_session(session_path, {
                    "phase": "abandoned",
                    "result": result.to_dict(),
                    "fetch": {
                        "successful_count": len(execution.successful),
                        "failed_count": len(execution.failed),
                    },
                })
                return result

            # 3. Comprehend
            comprehension: ComprehensionRecord = await self._comprehension.comprehend(
                decision.item, execution
            )
            result.inference_failures += comprehension.inference_failures
            self._save_session(session_path, {"phase": "comprehended", "result": result.to_dict(),
                                              "comprehension": {
                                                  "checkpoints": len(comprehension.checkpoints),
                                                  "shallow_read_flag": comprehension.shallow_read_flag,
                                                  "open_threads": len(comprehension.open_threads),
                                              }})

            if not comprehension.checkpoints:
                result.error = "comprehension produced no checkpoints"
                result.completed_at = time.time()
                self._record_scheduler_attempt(decision, outcome="abandoned", session_id=session_id)
                self._save_session(session_path, {"phase": "abandoned", "result": result.to_dict(),
                                                  "comprehension": {
                                                      "checkpoints": len(comprehension.checkpoints),
                                                      "shallow_read_flag": comprehension.shallow_read_flag,
                                                      "open_threads": len(comprehension.open_threads),
                                                  }})
                return result

            # 4. Reflect
            reflection: ReflectionRecord = await self._reflection.reflect(decision.item, comprehension)
            result.inference_failures += reflection.inference_failures
            self._save_session(session_path, {"phase": "reflected", "result": result.to_dict(),
                                              "reflection": {
                                                  "verification_keys": list(reflection.verification_answers.keys()),
                                                  "opinion_disagrees": reflection.opinion_disagrees,
                                                  "belief_updates": len(reflection.belief_updates),
                                                  "new_facts": len(reflection.new_facts),
                                                  "resolved_threads": len(reflection.resolved_threads),
                                                  "parked_threads": len(reflection.parked_threads),
                                              }})

            # 5. Depth-gate
            checkpoint_dicts = [c.to_dict() for c in comprehension.checkpoints]
            depth: DepthReport = self._gate.evaluate(
                item=decision.item,
                verification_answers=reflection.verification_answers,
                priority_levels_engaged=result.priority_levels_engaged,
                critical_view_engaged=reflection.critical_view_engaged,
                own_opinion=reflection.own_opinion,
                opinion_disagrees_somewhere=reflection.opinion_disagrees,
                comprehension_checkpoints=checkpoint_dicts,
                open_threads=comprehension.open_threads,
                parked_threads=reflection.parked_threads,
            )
            result.depth_passed = depth.passed
            result.depth_score = depth.score
            result.depth_failures = list(depth.failures)
            self._save_session(session_path, {"phase": "gated", "result": result.to_dict(),
                                              "depth": depth.to_dict()})

            # 6. Persist (regardless of depth pass — episodic always sticks; facts
            # & beliefs are provisional and may be revised on reconciliation)
            episodic = EpisodicEvent(
                summary=f"Engaged with `{decision.item.title}` via priority levels {result.priority_levels_engaged}",
                started_at=result.started_at,
                completed_at=time.time(),
                item_title=decision.item.title,
                method_priority_level=int(decision.top_priority_level),
                notes=f"depth_passed={depth.passed} score={depth.score:.2f}",
            )
            # Facts and belief updates used to be committed regardless of
            # depth.passed, so a shallow or FAILED engagement still wrote
            # model-generated claims into semantic memory — poisoning it while
            # only the progress outcome was labelled "shallow". The episodic
            # record always sticks (it happened, and that is true regardless of
            # quality); the epistemic content requires the engagement to have
            # actually met the depth bar.
            if depth.passed:
                committed_facts = reflection.new_facts
                committed_beliefs = reflection.belief_updates
            else:
                committed_facts = []
                committed_beliefs = []
                logger.warning(
                    "Depth gate failed for %r: withholding %d fact(s) and %d "
                    "belief update(s) from semantic memory; recording the "
                    "episode only.",
                    decision.item.title, len(reflection.new_facts),
                    len(reflection.belief_updates),
                )
            receipt: CommitReceipt = self._persister.commit_engagement(
                item_title=decision.item.title,
                episodic=episodic,
                facts=committed_facts,
                belief_updates=committed_beliefs,
            )
            result.persist_receipt = self._receipt_to_dict(receipt)
            result.epistemic_content_withheld = not depth.passed
            self._save_session(session_path, {"phase": "persisted", "result": result.to_dict()})

            # 7. Update progress tracker with the verification answers + content-type metadata
            result.completed_at = time.time()
            self._update_progress(decision, reflection, depth, result)

            # A rejected persistence receipt used to be copied into the result
            # and then ignored: progress, scheduler outcome, completion, and
            # session phase all said "completed" even when the memory commit had
            # entirely failed. Research that was never recorded is not research
            # that happened.
            persisted_ok = bool(getattr(receipt, "accepted", True))
            if not persisted_ok:
                outcome = "persist_failed"
                logger.error(
                    "Memory commit REJECTED for %r; recording the engagement as "
                    "failed rather than completed.", decision.item.title,
                )
            elif depth.passed:
                outcome = "completed"
            else:
                outcome = "shallow_engagement"
            result.persisted = persisted_ok
            self._record_scheduler_attempt(decision, outcome=outcome, session_id=session_id)

            self._save_session(
                session_path,
                {"phase": "complete" if persisted_ok else "persist_failed",
                 "result": result.to_dict()},
            )
        except asyncio.CancelledError:
            result.error = "cancelled"
            raise
        except _ENGAGEMENT_RECOVERABLE_ERRORS as e:
            _record_research_degradation(
                e,
                action=(
                    "marked the engagement as errored, preserved a recoverable session "
                    "checkpoint, and recorded a scheduler error outcome to prevent "
                    "immediate blind reselection"
                ),
                extra={
                    "item_title": decision.item.title,
                    "session_id": session_id,
                    "phase": "engagement",
                },
            )
            result.error = f"{type(e).__name__}: {e}"
            result.completed_at = time.time()
            logger.error("engagement crashed for %r: %s\n%s",
                         decision.item.title, e, traceback.format_exc())
            self._record_scheduler_attempt(decision, outcome="error", session_id=session_id)
            self._save_session(session_path, {"phase": "error", "result": result.to_dict()})
        finally:
            if self._on_complete:
                try:
                    self._on_complete(result)
                except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                    _record_research_degradation(
                        e,
                        severity="warning",
                        action=(
                            "isolated the completion callback failure after preserving "
                            "the engagement result"
                        ),
                        extra={
                            "item_title": decision.item.title,
                            "session_id": session_id,
                            "phase": "completion_callback",
                        },
                    )

        return result

    # ── Progress tracker integration ─────────────────────────────────────

    def _update_progress(
        self,
        decision: SchedulingDecision,
        reflection: ReflectionRecord,
        depth: DepthReport,
        result: EngagementResult,
    ) -> None:
        try:
            log = load_progress()
        except (OSError, RuntimeError, AttributeError, TypeError, ValueError) as e:
            # A failed LOAD used to produce a fresh empty ProgressLog, which the
            # save below then wrote to the canonical path — so one unreadable
            # byte silently replaced the entire durable research history with a
            # single entry. Recoverable data was destroyed by the recovery path.
            #
            # The corrupt file is quarantined first, so the history still exists
            # on disk and can be repaired, and the fresh log is only used to let
            # THIS engagement finish.
            quarantined = _quarantine_corrupt_progress()
            _record_research_degradation(
                e,
                severity="warning",
                action=(
                    "quarantined the unreadable progress log to "
                    f"{quarantined or '<quarantine failed>'} and used a fresh "
                    "in-memory log so the engagement could finish; prior history "
                    "is preserved for repair rather than overwritten"
                ),
                extra={
                    "item_title": decision.item.title,
                    "session_id": result.session_id,
                    "phase": "progress_load",
                    "quarantined_to": str(quarantined or ""),
                },
            )
            log = ProgressLog()

        existing = log.find(decision.item.title)
        verif = reflection.verification_answers
        method_level = int(decision.top_priority_level)

        recommend = "true" if depth.passed and reflection.opinion_disagrees else (
            "false" if not depth.passed else "true"
        )
        recommend_reason = (
            depth.notes[0] if depth.notes else
            ("strong engagement" if depth.passed else "; ".join(depth.failures))
        )

        if existing is None:
            entry = ProgressEntry(
                title=decision.item.title,
                started_at=_iso(result.started_at),
                method_priority_level=method_level,
                method_detail=f"levels={result.priority_levels_engaged}",
                completed_at=_iso(result.completed_at) if depth.passed else None,
                what_its_actually_about=verif.get("what_its_actually_about", ""),
                what_stayed_with_you=verif.get("what_stayed_with_you", ""),
                what_it_says_about_humans=verif.get("what_it_says_about_humans", ""),
                what_it_made_you_think_about_yourself=verif.get(
                    "what_it_made_you_think_about_yourself", ""
                ),
                open_threads=[t.get("thread", "") for t in reflection.parked_threads],
                would_recommend_to_bryan=f"{recommend} :: {recommend_reason}",
            )
            try:
                log.add_entry(entry)
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                _record_research_degradation(
                    e,
                    severity="warning",
                    action=(
                        "kept the engagement result and session checkpoint intact while "
                        "skipping the malformed progress entry"
                    ),
                    extra={
                        "item_title": decision.item.title,
                        "session_id": result.session_id,
                        "phase": "progress_add",
                    },
                )
                logger.warning("progress add_entry failed: %s", e)
        else:
            # Update existing — merge what was already there with new findings
            if not existing.completed_at and depth.passed:
                existing.completed_at = _iso(result.completed_at)
            existing.method_detail = f"levels={result.priority_levels_engaged}"
            for key in (
                "what_its_actually_about",
                "what_stayed_with_you",
                "what_it_says_about_humans",
                "what_it_made_you_think_about_yourself",
            ):
                v = verif.get(key)
                if v:
                    setattr(existing, key, v)
            existing.open_threads = [t.get("thread", "") for t in reflection.parked_threads]
            existing.would_recommend_to_bryan = f"{recommend} :: {recommend_reason}"

        try:
            log.save()
        except (OSError, RuntimeError, AttributeError, TypeError, ValueError) as e:
            _record_research_degradation(
                e,
                severity="warning",
                action=(
                    "preserved the session checkpoint and in-memory progress update; "
                    "durable progress index will be retried on a later engagement"
                ),
                extra={
                    "item_title": decision.item.title,
                    "session_id": result.session_id,
                    "phase": "progress_save",
                },
            )
            logger.warning("progress save failed: %s", e)

    # ── Helpers ──────────────────────────────────────────────────────────

    def _receipt_to_dict(self, receipt: CommitReceipt) -> dict[str, Any]:
        return {
            "accepted": receipt.accepted,
            "episodic_committed": receipt.episodic_committed,
            "facts_committed": receipt.facts_committed,
            "facts_total": receipt.facts_total,
            "beliefs_committed": receipt.beliefs_committed,
            "beliefs_total": receipt.beliefs_total,
            "queued_for_retry": receipt.queued_for_retry,
            "duplicates_skipped": receipt.duplicates_skipped,
            "failures": list(receipt.failures),
            "intent_ids": list(receipt.intent_ids),
        }

    def _record_scheduler_attempt(
        self,
        decision: SchedulingDecision,
        *,
        outcome: str,
        session_id: str,
    ) -> None:
        try:
            self._scheduler.record_attempt(decision, outcome=outcome)
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            _record_research_degradation(
                e,
                severity="warning",
                action=(
                    "preserved the engagement result while isolating scheduler attempt "
                    "bookkeeping failure"
                ),
                extra={
                    "item_title": decision.item.title,
                    "session_id": session_id,
                    "outcome": outcome,
                    "triggered_by": decision.triggered_by,
                    "phase": "scheduler_attempt",
                },
            )

    def unfinished_sessions(self) -> list[dict[str, Any]]:
        """Engagements that were interrupted rather than completed.

        The class documents session resume and writes checkpoints for it, but
        nothing ever SCANNED for unfinished work — so after a restart the
        checkpoints existed and no code could find them. Recovery has to start
        from "what was I doing?", because knowing the session id is exactly
        what a crash destroys.

        A corrupt checkpoint is skipped rather than aborting the scan: one bad
        file must not hide every other resumable session.
        """
        out: list[dict[str, Any]] = []
        try:
            paths = sorted(self._sessions_dir.glob("*.json"))
        except OSError as exc:
            _record_research_degradation(
                exc, severity="warning",
                action="could not scan for unfinished research sessions",
            )
            return out

        for path in paths:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            phase = str(payload.get("phase") or "")
            # "complete" is the only phase that means the engagement finished.
            # persist_failed is explicitly NOT terminal — it is exactly the
            # case worth resuming, because the research happened and only the
            # commit did not.
            if phase in _TERMINAL_SESSION_PHASES:
                continue
            result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
            out.append({
                "session_path": str(path),
                "session_id": path.stem,
                "phase": phase or "unknown",
                "item_title": str(result.get("item_title") or ""),
                "started_at": result.get("started_at"),
                "resumable": phase in _RESUMABLE_SESSION_PHASES,
            })
        return out

    def _save_session(self, path: Path, payload: dict[str, Any]) -> None:
        phase = str(payload.get("phase", "unknown"))
        try:
            atomic_write_text(path, json.dumps(payload, indent=2, default=str), encoding="utf-8")
        except (OSError, RuntimeError, TypeError, ValueError) as e:
            _record_research_degradation(
                e,
                severity="warning",
                action=(
                    "kept the engagement state in memory and will retry session "
                    "persistence on the next checkpoint"
                ),
                extra={
                    "session_path": str(path),
                    "phase": phase,
                },
            )


def _iso(ts: float | None) -> str:
    if ts is None:
        return ""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(float(ts)))
