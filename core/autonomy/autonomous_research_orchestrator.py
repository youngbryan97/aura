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
  checkpoint to ``aura/knowledge/research-sessions/``. If interrupted, the
  next run reloads any unfinished session and resumes from the last
  completed comprehension chunk rather than starting over.

Concurrency: one engagement at a time per orchestrator instance. Multiple
orchestrators can run in parallel if Bryan wants pipelined throughput.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import traceback
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from core.autonomy.curiosity_scheduler import CuriosityScheduler, SchedulingDecision
from core.autonomy.content_method_router import MethodRouter
from core.autonomy.content_fetcher import ContentFetcher, FetchExecution
from core.autonomy.comprehension_loop import ComprehensionLoop, ComprehensionRecord
from core.autonomy.reflection_loop import ReflectionLoop, ReflectionRecord
from core.autonomy.depth_gate import DepthGate, DepthReport
from core.autonomy.memory_persister import (
    MemoryPersister,
    EpisodicEvent,
    FactRecord,
    BeliefUpdate,
    CommitReceipt,
)
from core.autonomy.content_progress_tracker import (
    ProgressLog,
    ProgressEntry,
    load as load_progress,
    DEFAULT_PROGRESS_PATH,
)

logger = logging.getLogger("Aura.AutonomousResearchOrchestrator")

SESSIONS_DIR = Path.home() / ".aura/live-source/aura/knowledge/research-sessions"
DEFAULT_LOOP_INTERVAL = 600.0   # 10 minutes between engagements
DEFAULT_MAX_CONSECUTIVE_FAILURES = 3


@dataclass
class EngagementResult:
    """Single engagement's outcome."""
    item_title: str
    started_at: float
    completed_at: Optional[float] = None
    decision: Optional[Dict[str, Any]] = None
    sources_engaged: List[str] = field(default_factory=list)
    priority_levels_engaged: List[int] = field(default_factory=list)
    depth_passed: bool = False
    depth_score: float = 0.0
    depth_failures: List[str] = field(default_factory=list)
    persist_receipt: Optional[Dict[str, Any]] = None
    inference_failures: int = 0
    error: Optional[str] = None
    session_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
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
            "inference_failures": self.inference_failures,
            "error": self.error,
            "session_id": self.session_id,
        }


class AutonomousResearchOrchestrator:
    def __init__(
        self,
        scheduler: Optional[CuriosityScheduler] = None,
        router: Optional[MethodRouter] = None,
        fetcher: Optional[ContentFetcher] = None,
        comprehension: Optional[ComprehensionLoop] = None,
        reflection: Optional[ReflectionLoop] = None,
        gate: Optional[DepthGate] = None,
        persister: Optional[MemoryPersister] = None,
        sessions_dir: Path = SESSIONS_DIR,
        loop_interval: float = DEFAULT_LOOP_INTERVAL,
        max_consecutive_failures: int = DEFAULT_MAX_CONSECUTIVE_FAILURES,
        on_engagement_complete: Optional[Callable[[EngagementResult], None]] = None,
    ) -> None:
        self._scheduler = scheduler or CuriosityScheduler()
        self._router = router or MethodRouter()
        self._fetcher = fetcher or ContentFetcher()
        self._comprehension = comprehension or ComprehensionLoop()
        self._reflection = reflection or ReflectionLoop()
        self._gate = gate or DepthGate()
        self._persister = persister or MemoryPersister()
        self._sessions_dir = sessions_dir
        self._loop_interval = loop_interval
        self._max_failures = max_consecutive_failures
        self._on_complete = on_engagement_complete
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._consecutive_failures = 0

    # ── Public API ────────────────────────────────────────────────────────

    async def run_once(self) -> Optional[EngagementResult]:
        decision = self._scheduler.pick_next()
        if decision is None:
            logger.info("scheduler returned no candidate; nothing to do")
            return None
        return await self._engage(decision)

    async def start_loop(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())

    async def stop_loop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

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
                except Exception as e:
                    logger.error("research loop iteration crashed: %s\n%s", e, traceback.format_exc())
                    self._consecutive_failures += 1

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
        self._sessions_dir.mkdir(parents=True, exist_ok=True)

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
                self._scheduler.record_attempt(decision, outcome="abandoned")
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
                self._scheduler.record_attempt(decision, outcome="abandoned")
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
            receipt: CommitReceipt = self._persister.commit_engagement(
                item_title=decision.item.title,
                episodic=episodic,
                facts=reflection.new_facts,
                belief_updates=reflection.belief_updates,
            )
            result.persist_receipt = self._receipt_to_dict(receipt)
            self._save_session(session_path, {"phase": "persisted", "result": result.to_dict()})

            # 7. Update progress tracker with the verification answers + content-type metadata
            self._update_progress(decision, reflection, depth, result)

            outcome = "completed" if depth.passed else "shallow_engagement"
            self._scheduler.record_attempt(decision, outcome=outcome)

            result.completed_at = time.time()
            self._save_session(session_path, {"phase": "complete", "result": result.to_dict()})
        except asyncio.CancelledError:
            result.error = "cancelled"
            raise
        except Exception as e:
            result.error = f"{type(e).__name__}: {e}"
            logger.error("engagement crashed for %r: %s\n%s",
                         decision.item.title, e, traceback.format_exc())
        finally:
            if self._on_complete:
                try:
                    self._on_complete(result)
                except Exception:
                    pass

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
        except Exception:
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
            except Exception as e:
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
        except Exception as e:
            logger.warning("progress save failed: %s", e)

    # ── Helpers ──────────────────────────────────────────────────────────

    def _receipt_to_dict(self, receipt: CommitReceipt) -> Dict[str, Any]:
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

    def _save_session(self, path: Path, payload: Dict[str, Any]) -> None:
        try:
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
            os.replace(tmp, path)
        except Exception:
            pass


def _iso(ts: Optional[float]) -> str:
    if ts is None:
        return ""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(float(ts)))
