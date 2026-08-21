from __future__ import annotations

import asyncio
import inspect
import json
import logging
import math
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any

from core.autonomy.research_goal_filter import (
    is_unresearchable_goal,
    research_query_for_goal,
)
from core.autonomy.research_history import ResearchHistory
from core.autonomy.research_text_policy import (
    MAX_NARRATIVE_CHARS,
    PARAMETRIC_PREFIX,
    bounded_narrative,
    is_transient_failure,
    label_findings,
    narrative_admits,
)
from core.runtime import background_policy
from core.runtime.errors import FallbackClassification, Severity, record_degradation
from core.runtime.state_ownership import state_root
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Aura.ResearchCycle")

RESEARCH_RECOVERABLE_ERRORS = (
    AttributeError,
    ImportError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


def _record_research_degradation(
    error: BaseException,
    *,
    action: str,
    severity: Severity = "warning",
    extra: dict[str, Any] | None = None,
) -> None:
    # A required service's failure is worth a receipt. Every one of these
    # was recorded as SAFE_FALLBACK with receipt_required=False, so the
    # exact runtime failure class a person would report — research silently
    # stopped — left the weakest possible forensic trail (CP126
    # ``a12f95c0``).
    receipt_required = severity in {"degraded", "critical"} or _needs_receipt(action)
    try:
        record_degradation(
            "research_cycle",
            error,
            severity=severity,
            action=action,
            classification=FallbackClassification.SAFE_FALLBACK,
            receipt_required=receipt_required,
            extra=extra,
        )
    except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
        logger.warning(
            "ResearchCycle degradation sink rejected recoverable research fault: %s",
            exc,
        )


#: Failure kinds an operator has to be able to reconstruct: the research
#: lane stopping, an initiative being lost, or an integration half-applied.
_RECEIPT_MARKERS = (
    "without integration",
    "lost",
    "suppress",
    "rollback",
    "timeout",
    "unavailable",
    # An objective that may not come back is the failure a person notices
    # as "it stopped researching things".
    "intent",
    "initiative",
    "objective",
    "queue",
)


def _needs_receipt(action: str) -> bool:
    lowered = str(action or "").lower()
    return any(marker in lowered for marker in _RECEIPT_MARKERS)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        logger.debug("Invalid %s=%r; using %.1f", name, raw, default)
        return float(default)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return int(default)
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        logger.debug("Invalid %s=%r; using %d", name, raw, default)
        return int(default)


# ── Completed research record ─────────────────────────────────────────────────

@dataclass
class ResearchRecord:
    """Persisted record of one completed research cycle."""
    record_id:       str
    drive:           str              # Which motivation drove this
    goal:            str              # What was researched
    findings:        list[str]        # Concrete facts extracted
    identity_impact: str              # How this changed the narrative
    affect_before:   dict[str, float]
    affect_after:    dict[str, float]
    phi_before:      float | None = 0.0
    phi_after:       float | None = 0.0
    started_at:      float = 0.0
    completed_at:    float = 0.0
    task_plan_id:    str | None = None

    def to_dict(self) -> dict:
        return {
            "record_id":       self.record_id,
            "drive":           self.drive,
            "goal":            self.goal,
            "findings":        self.findings,
            "identity_impact": self.identity_impact,
            "affect_before":   self.affect_before,
            "affect_after":    self.affect_after,
            "phi_before":      round(float(self.phi_before or 0.0), 4),
            "phi_after":       round(float(self.phi_after or 0.0), 4),
            "started_at":      self.started_at,
            "completed_at":    self.completed_at,
            "task_plan_id":    self.task_plan_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResearchRecord:
        return cls(
            record_id=data["record_id"],
            drive=data["drive"],
            goal=data["goal"],
            findings=data.get("findings", []),
            identity_impact=data.get("identity_impact", ""),
            affect_before=data.get("affect_before", {}),
            affect_after=data.get("affect_after", {}),
            phi_before=data.get("phi_before", 0.0),
            phi_after=data.get("phi_after", 0.0),
            started_at=data.get("started_at", 0.0),
            completed_at=data.get("completed_at", 0.0),
            task_plan_id=data.get("task_plan_id"),
        )


# ── The Research Cycle ────────────────────────────────────────────────────────

def _materialize_research_goal(
    cycle: ResearchCycle,
    initiative: dict[str, Any],
    state: Any,
) -> dict[str, Any] | None:
    """Turn a placeholder initiative into something researchable."""
    metadata = dict(initiative.get("metadata", {}) or {})
    goal = str(initiative.get("goal", "") or "").strip()
    drive = str(initiative.get("drive") or metadata.get("triggered_by") or "curiosity")

    if _generic_internal_goal(goal):
        topic = _derive_autotelic_topic(cycle, state)
        if not topic:
            return None
        initiative["goal"] = f"Research and learn something new about {topic}"
        initiative["drive"] = "curiosity" if drive in {"curiosity", "boredom"} else drive
        metadata["materialized_from"] = goal[:120]
        metadata["materialized_topic"] = topic
        initiative["metadata"] = metadata
    return initiative


def _generic_internal_goal(goal: str) -> bool:
    """Whether a goal is a placeholder rather than something to look up."""
    lowered = str(goal or "").lower()
    return any(
        marker in lowered
        for marker in (
            "review internal knowledge graph continuity",
            "quietly consolidate internal state",
            "wait for a stronger signal",
            "reflect on recent interactions",
            "hold attentive idle posture",
        )
    )

def _derive_autotelic_topic(cycle: ResearchCycle, state: Any) -> str:
    """A topic worth researching, drawn from live state."""
    try:
        from core.autonomy.topic_selection import select_autonomous_topic

        candidate = select_autonomous_topic(
            cycle.orchestrator,
            state,
            excluded=(record.goal for record in cycle._history[-20:]),
        )
        if candidate is not None:
            return candidate.text
    except RESEARCH_RECOVERABLE_ERRORS as exc:
        _record_research_degradation(
            exc,
            action="left autonomous research idle because grounded topic derivation failed",
            extra={"cycle_count": cycle._cycle_count},
        )
        logger.debug("Autotelic topic derivation failed: %s", exc)
    return ""


class ResearchCycle:
    """
    Autonomous background research engine.

    Runs when Aura is idle. Selects goals from pending_initiatives (generated
    by MotivationUpdatePhase), pursues them, integrates results into knowledge
    and identity, and reflects on what was learned.
    """

    # Timing
    MIN_CYCLE_INTERVAL_S = 1800    # 30 minutes between cycles
    IDLE_THRESHOLD_S     = 120     # 2 minutes of user silence = idle
    MAX_GOAL_DURATION_S  = 300     # 5 minutes per research cycle max

    # Quality gates
    MIN_ENERGY_FOR_RESEARCH = 20.0   # Won't research if energy is this low
    MIN_CURIOSITY           = 0.3    # Won't research if curiosity is this low
    MIN_FINDINGS            = 1      # Must produce at least this many findings

    def __init__(self, orchestrator: Any):
        self.orchestrator = orchestrator
        self._running = False
        self._task: asyncio.Task | None = None
        self._last_cycle_mono: float = 0.0
        self._started_mono: float = monotonic()
        self._cycle_count: int = 0
        self._history: list[ResearchRecord] = []
        self._daemon_failure_count: int = 0
        self._leased_intent: tuple[list, dict] | None = None
        self._last_energy_reading: dict[str, float] = {}
        self._transient_failure_counts: dict[str, int] = {}
        self._last_cycle_error: str | None = None
        self._history_load_errors: int = 0
        self._restart_count: int = 0
        self._last_restart_mono: float = 0.0
        self._goal_failure_counts: dict[str, int] = {}

        try:
            from core.config import config
            self._record_path = config.paths.data_dir / "research" / "cycle_history.jsonl"
        except (ImportError, AttributeError):
            self._record_path = state_root() / "research" / "cycle_history.jsonl"

        try:
            self._record_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            fallback_root = Path(os.getenv("TMPDIR", "/tmp")) / "aura" / "research"
            try:
                fallback_root.mkdir(parents=True, exist_ok=True)
            except OSError as fallback_exc:
                _record_research_degradation(
                    fallback_exc,
                    action="disabled durable research history after primary and fallback directory creation failed",
                    severity="degraded",
                    extra={"configured_path": str(self._record_path)},
                )
                self._record_path = Path(os.devnull)
            else:
                _record_research_degradation(
                    exc,
                    action="fell back to temporary research history path after durable directory creation failed",
                    extra={"configured_path": str(self._record_path), "fallback_path": str(fallback_root)},
                )
                self._record_path = fallback_root / "cycle_history.jsonl"
        #: The durable record: chained, gateway-written, verified on load.
        self._history_store = ResearchHistory(self._record_path)
        self._load_history()

        logger.info("ResearchCycle initialized. Previous cycles: %d", self._cycle_count)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        if self._running and self._task is not None and not self._task.done():
            return
        if self._task is not None and self._task.done():
            self._consume_finished_task(self._task)
        self._running = True
        self._task = get_task_tracker().create_task(self._daemon(), name="aura.research_cycle")
        logger.info("ResearchCycle daemon started.")

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            if not self._task.done():
                self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError as _exc:
                logger.debug("Suppressed %s in core.autonomy.research_cycle: %s", type(_exc).__name__, _exc)
            except RESEARCH_RECOVERABLE_ERRORS as exc:
                _record_research_degradation(
                    exc,
                    action="completed research daemon shutdown after task ended with recoverable error",
                )
                logger.debug("ResearchCycle task ended during shutdown: %s", exc)
            self._task = None
        logger.info("ResearchCycle daemon stopped.")

    def is_alive(self) -> bool:
        return bool(self._running and self._task is not None and not self._task.done())

    async def restart_async(self) -> None:
        self._restart_count += 1
        self._last_restart_mono = monotonic()
        await self.stop()
        await self.start()

    @staticmethod
    def _consume_finished_task(task: asyncio.Task) -> None:
        try:
            task.result()
        except asyncio.CancelledError:
            return
        except RESEARCH_RECOVERABLE_ERRORS as exc:
            _record_research_degradation(
                exc,
                action="restarted research daemon after previous task ended with recoverable error",
            )

    # ── Main daemon loop ──────────────────────────────────────────────────────

    async def _daemon(self) -> None:
        """Runs continuously. Checks conditions and triggers cycles."""
        while self._running:
            try:
                await asyncio.sleep(30.0)  # Check every 30 seconds
                from core.container import ServiceContainer

                healer = ServiceContainer.get("self_healing", default=None)
                if healer is not None:
                    healer.heartbeat("research_cycle")

                if not self._should_run():
                    continue

                logger.info("ResearchCycle: conditions met. Starting cycle %d...", self._cycle_count + 1)
                await self._run_one_cycle()

            except asyncio.CancelledError:
                break
            except RESEARCH_RECOVERABLE_ERRORS as e:
                # A crashed cycle is a cycle that did not do the work, so
                # the objective goes back to the queue rather than being
                # lost with the exception.
                self._settle_intent_lease(completed=False)
                self._daemon_failure_count += 1
                self._last_cycle_error = f"{type(e).__name__}: {e}"
                _record_research_degradation(
                    e,
                    action="backed off daemon loop and deferred autonomous research after recoverable cycle failure",
                    extra={"daemon_failures": self._daemon_failure_count},
                )
                logger.error("ResearchCycle daemon error: %s", e, exc_info=True)
                await asyncio.sleep(60.0)  # Back off on error

    def _has_energy_for_research(self, state: Any) -> bool:
        """Whether the energy budget allows research, on either scale.

        The capacity is the scale: a budget whose capacity is 1.0 is
        normalized and 0.2 is the same fraction as 20 out of 100. When
        neither the level nor the capacity is readable this admits, because
        an unreadable budget is not evidence of exhaustion — and reading it
        as one is what would disable research forever.
        """
        budget = state.motivation.budgets.get("energy", {}) or {}
        try:
            level = float(budget.get("level"))
        except (TypeError, ValueError):
            return True
        if not math.isfinite(level):
            return True
        try:
            capacity = float(budget.get("capacity", 100.0))
        except (TypeError, ValueError):
            capacity = 100.0
        if not math.isfinite(capacity) or capacity <= 0.0:
            capacity = 100.0
        fraction = level / capacity
        self._last_energy_reading = {
            "level": level,
            "capacity": capacity,
            "fraction": round(fraction, 4),
        }
        return fraction >= (self.MIN_ENERGY_FOR_RESEARCH / 100.0)

    def research_health(self) -> dict[str, Any]:
        """Whether research can actually happen, not whether the loop woke.

        The daemon reported a self-healing heartbeat before checking
        admission, state, energy, curiosity, tool readiness or progress, so
        a permanently blocked research lane stayed heartbeat-healthy
        (CP126 ``6e817da1``).
        """
        blockers: list[str] = []
        state = self._get_state()
        if state is None:
            blockers.append("no_state")
        if not hasattr(self.orchestrator, "execute_tool"):
            blockers.append("no_tool_surface")
        elif not self._research_tool_allowlist():
            blockers.append("no_research_tools_available")
        try:
            reason = background_policy.background_activity_reason(
                self.orchestrator,
                profile=background_policy.RESEARCH_BACKGROUND_POLICY,
            )
            if reason:
                blockers.append(f"background_policy:{reason}")
        except RESEARCH_RECOVERABLE_ERRORS as exc:
            blockers.append(f"background_policy_unavailable:{type(exc).__name__}")

        since_last = (
            monotonic() - self._last_cycle_mono if self._last_cycle_mono else None
        )
        stalled = bool(
            self._last_cycle_mono
            and since_last is not None
            and since_last > (self.MIN_CYCLE_INTERVAL_S * 10)
        )
        return {
            "loop_alive": self.is_alive(),
            # The distinction the heartbeat could not make.
            "can_research": not blockers,
            "blockers": blockers,
            "cycles_completed": self._cycle_count,
            "seconds_since_last_cycle": round(since_last, 1) if since_last else None,
            "stalled": stalled,
            "last_error": self._last_cycle_error or "",
            "consecutive_daemon_failures": self._daemon_failure_count,
        }

    def _should_run(self) -> bool:
        """Check all conditions before starting a research cycle."""
        # 1. Rate limiting
        now = monotonic()
        boot_grace_s = _env_float("AURA_RESEARCH_BOOT_GRACE_S", 300.0)
        started_mono = float(getattr(self, "_started_mono", 0.0) or 0.0)
        if boot_grace_s > 0.0 and started_mono > 0.0 and (now - started_mono) < boot_grace_s:
            return False

        if self._last_cycle_mono and now - self._last_cycle_mono < self.MIN_CYCLE_INTERVAL_S:
            return False

        try:
            reason = background_policy.background_activity_reason(
                self.orchestrator,
                profile=background_policy.RESEARCH_BACKGROUND_POLICY,
            )
            if reason:
                return False
        except RESEARCH_RECOVERABLE_ERRORS as _exc:
            self._last_cycle_error = f"{type(_exc).__name__}: {_exc}"
            _record_research_degradation(
                _exc,
                action="deferred autonomous research because background policy gate was unavailable",
            )
            logger.debug("Background policy gate unavailable: %s", _exc)
            return False

        # 2. User must be idle
        last_user = getattr(self.orchestrator, "_last_user_interaction_time", 0.0)
        # Assuming _last_user_interaction_time is wall-clock, we use time.time() for the diff
        if time.time() - last_user < self.IDLE_THRESHOLD_S:
            return False

        # 3. System must not be actively processing
        if getattr(getattr(self.orchestrator, "status", None), "is_processing", False):
            return False

        # 4. Kernel must be available
        state = self._get_state()
        if state is None:
            return False

        # 5. Energy check. The gate defaulted to 100 and compared against 20
        # while other autonomy systems normalize energy to 0-1, so schema
        # drift would read a full battery of 0.9 as exhausted and disable
        # research permanently (CP126 ``3dec59b8``).
        if not self._has_energy_for_research(state):
            return False

        # 6. Curiosity check
        curiosity = getattr(state.affect, "curiosity", 0.0)
        if curiosity < self.MIN_CURIOSITY:
            return False

        # 7. Must have pending initiatives OR autotelic intent
        if not state.cognition.pending_initiatives:
            # Check for autotelic intent from LearningPhase
            if not any(i.get("type") == "autotelic_objective" for i in getattr(state, "pending_intents", [])):
                return False

        return True

    # ── One research cycle ────────────────────────────────────────────────────

    async def _run_one_cycle(self) -> ResearchRecord | None:
        """Execute a single research cycle end-to-end."""
        start_time = time.time()
        state = self._get_state()
        if state is None:
            return None

        await self._suppress_unresearchable_initiatives(state)
        state = self._get_state() or state

        # 1. Select the best initiative
        initiative = self._select_initiative(state)
        if initiative is None:
            logger.debug("ResearchCycle: no suitable initiative found.")
            return None

        goal  = initiative.get("goal", "")
        drive = initiative.get("drive", "curiosity")

        logger.info("ResearchCycle: pursuing '%s' (drive=%s)", goal[:80], drive)

        # Snapshot state before
        affect_before = {
            "valence":   float(state.affect.valence),
            "curiosity": float(state.affect.curiosity),
            "arousal":   float(state.affect.arousal),
        }
        phi_before: float = float(state.phi or 0.0)

        # 2. Execute research via AutonomousTaskEngine
        research_result = await self._execute_research(goal, drive)

        # 4. Extract findings
        findings = await self._extract_findings(research_result, goal)

        if len(findings) < self.MIN_FINDINGS:
            logger.info("ResearchCycle: insufficient findings for '%s'. Skipping integration.", goal[:60])
            await self._handle_no_findings(state, goal, drive)
            # The objective was not researched, so it goes back.
            self._settle_intent_lease(completed=False)
            self._last_cycle_mono = monotonic()
            return None

        # 5. Integrate into knowledge graph FIRST. Executive suppression used
        # to run here, before knowledge, the eternal vault, the narrative,
        # affect, budgets and the durable record — so any later failure
        # retired the goal while the work it stood for was incomplete, with
        # nothing to roll back (CP126 ``bf08b6a9``). The initiative is the
        # record of unfinished work, so it is the LAST thing to go.
        await self._integrate_knowledge(findings, goal, drive)

        # 6. Write to eternal vault (deferred via pending_intents)
        entry = {
            "type":     "research_cycle",
            "goal":     goal[:60],
            "drive":    drive,
            "findings": findings[:5],
            "timestamp": time.time(),
        }
        if hasattr(state.cognition, "pending_intents"):
            state.cognition.pending_intents.append({
                "type":    "eternal_append",
                "path":    str(state_root() / "research_history.jsonl"),
                "payload": entry,
            })

        # 7. Update identity narrative
        identity_impact = await self._update_narrative(state, goal, findings)

        # 8. Emit positive affect percepts
        state.world.recent_percepts.append({
            "type":      "goal_achieved",
            "intensity": 0.7,
            "payload":   {"goal": goal[:60], "drive": drive},
        })

        # 9. Replenish motivation budgets
        budgets = state.motivation.budgets
        if drive in budgets:
            budgets[drive]["level"] = min(
                budgets[drive]["capacity"],
                budgets[drive]["level"] + 20.0,
            )

        # 10. Retire the initiative. Everything it stood for is now written.
        try:
            from core.consciousness.executive_authority import get_executive_authority

            state, _ = await get_executive_authority(self.orchestrator).suppress_initiatives(
                state,
                predicate=lambda item: str(item.get("goal", "") or "") == goal,
                reason="research_cycle_goal_completed",
                source="research_cycle",
            )
            self._settle_intent_lease(completed=True)
        except RESEARCH_RECOVERABLE_ERRORS as exc:
            self._last_cycle_error = f"{type(exc).__name__}: {exc}"
            _record_research_degradation(
                exc,
                action=(
                    "integrated the findings and left the initiative pending; a "
                    "later cycle may repeat it rather than lose it"
                ),
                extra={"goal": goal[:160]},
            )
            logger.warning(
                "ResearchCycle: executive suppression failed, leaving initiative intact: %s",
                exc,
            )
            self._settle_intent_lease(completed=False)

        # 11. Snapshot state after
        state_after = self._get_state()
        affect_after = {
            "valence":   float(getattr(state_after, "affect", state.affect).valence),
            "curiosity": float(getattr(state_after, "affect", state.affect).curiosity),
            "arousal":   float(getattr(state_after, "affect", state.affect).arousal),
        } if state_after else affect_before
        phi_after_raw = getattr(state_after, "phi", None) if state_after else None
        
        # Ensure phi_after is float for the record
        final_phi: float = float(phi_after_raw if phi_after_raw is not None else phi_before)
        phi_after = final_phi

        # 11. Build record. Full uuid4 (not a truncated 8-hex/32-bit id) so an
        # indefinite append-only history cannot collide.
        record = ResearchRecord(
            record_id      = uuid.uuid4().hex,
            drive          = drive,
            goal           = goal,
            findings       = findings,
            identity_impact = identity_impact,
            affect_before  = affect_before,
            affect_after   = affect_after,
            phi_before     = phi_before,
            phi_after      = phi_after,
            started_at     = start_time,
            completed_at   = time.time(),
            task_plan_id   = getattr(research_result, "plan_id", None),
        )
        self._history.append(record)
        self._save_record(record)

        self._cycle_count += 1
        self._last_cycle_mono = monotonic()

        logger.info(
            "ResearchCycle %d complete: '%s' → %d findings in %.1fs",
            self._cycle_count, goal[:60], len(findings),
            record.completed_at - record.started_at,
        )

        # 12. Trigger dreaming if enough has accumulated
        await self._maybe_trigger_dream()

        return record

    # ── Step implementations ──────────────────────────────────────────────────

    def _settle_intent_lease(self, *, completed: bool) -> None:
        """Consume the leased intent on success; return it on failure.

        Consuming up front made every downstream failure a silent loss of
        the objective. Consuming here makes the queue the record of what
        still needs doing (CP126 ``58359965``).
        """
        leased = getattr(self, "_leased_intent", None)
        if not leased:
            return
        queue, intent = leased
        self._leased_intent = None
        try:
            intent.pop("research_lease", None)
            if completed:
                queue.remove(intent)
        except (ValueError, AttributeError, TypeError) as exc:
            _record_research_degradation(
                exc,
                action=(
                    "could not settle the autotelic intent lease; the objective "
                    "may be researched twice"
                    if completed
                    else "could not return the autotelic intent to the queue"
                ),
                severity="warning",
            )

    def _select_initiative(self, state: Any) -> dict | None:
        """
        Select the best initiative from pending_initiatives.
        
        Priority:
          1. Explicit initiatives (scored by urgency)
          2. Autotelic intents (Implicit curiosity)
        """
        # 1. Try explicit pending initiatives
        initiatives = getattr(state.cognition, "pending_initiatives", [])
        if initiatives:
            eligible = [
                item
                for item in initiatives
                if not is_unresearchable_goal(item.get("goal", item.get("description", "")) if isinstance(item, dict) else item)
            ]
            if len(eligible) != len(initiatives):
                logger.info(
                    "ResearchCycle: quarantined %d non-research initiative(s) from autonomous search selection.",
                    len(initiatives) - len(eligible),
                )
            initiatives = eligible
            if not initiatives:
                return None

            def _priority(item: dict[str, Any]) -> tuple[float, float]:
                metadata = dict(item.get("metadata", {}) or {})
                continuity_bonus = 0.0
                if item.get("continuity_restored") or metadata.get("continuity_restored"):
                    continuity_bonus += 0.18
                continuity_bonus += min(
                    0.18,
                    max(
                        0.0,
                        float(metadata.get("continuity_pressure", item.get("continuity_pressure", 0.0)) or 0.0),
                    )
                    * 0.2,
                )
                return (
                    float(item.get("urgency", 0.0) or 0.0) + continuity_bonus,
                    float(item.get("timestamp", 0.0) or 0.0),
                )

            # Sort by continuity-aware urgency (highest first)
            sorted_init = sorted(initiatives, key=_priority, reverse=True)
            return _materialize_research_goal(self, dict(sorted_init[0]), state)

        # 2. Fallback: Autotelic intent generated by learning/self-review modules.
        # Use a unified approach to finding pending intents
        possible_intents = getattr(state.cognition, "pending_intents", []) or getattr(state, "pending_intents", [])
        
        if isinstance(possible_intents, list):
            for intent in list(possible_intents):
                if isinstance(intent, dict) and intent.get("type") == "autotelic_objective":
                    domain = intent.get("domain") or _derive_autotelic_topic(self, state)
                    if not domain:
                        logger.debug("Autotelic intent deferred: no grounded topic is available yet.")
                        return None
                    logger.info("⚡ [AUTOTELIC] Autotelic signal identified: %s", domain)
                    
                    # LEASED, not consumed. The intent used to be removed
                    # here, before any search ran — so a failure in search,
                    # extraction, integration or persistence lost the
                    # objective outright, with nothing to roll back to
                    # (CP126 ``58359965``). It is marked in flight and
                    # returned to the queue if the cycle does not complete.
                    try:
                        intent["research_lease"] = {
                            "leased_at": time.time(),
                            "by": "research_cycle",
                        }
                        self._leased_intent = (possible_intents, intent)
                    except (TypeError, AttributeError) as exc:
                        _record_research_degradation(
                            exc,
                            action="continued with selected autotelic intent without a lease marker",
                            severity="debug",
                        )
                        self._leased_intent = None

                    return {
                        "goal": f"Self-directed exploration of {domain}",
                        "drive": "curiosity",
                        "urgency": 0.9,
                        "origin": "autotelic_curiosity",
                    }

        return None

    async def _execute_research(self, goal: str, drive: str) -> Any:
        """Execute the research goal, whole, under one deadline.

        ``MAX_GOAL_DURATION_S`` wrapped only ``engine.execute_goal``. The
        grounded web search that runs first and the direct fallback that
        runs instead both had no deadline at all, so a cycle could exceed
        the documented maximum without limit (CP126 ``fca4ad22``).
        """
        try:
            async with asyncio.timeout(self.MAX_GOAL_DURATION_S):
                return await self._execute_research_inner(goal, drive)
        except TimeoutError as exc:
            self._last_cycle_error = f"{type(exc).__name__}: {exc}"
            _record_research_degradation(
                exc,
                action="ended research attempt without integration after the cycle deadline",
                extra={"goal": goal[:160], "drive": drive},
            )
            logger.warning("ResearchCycle: research timed out for '%s'", goal[:60])
            return None

    async def _execute_research_inner(self, goal: str, drive: str) -> Any:
        try:
            grounded = await self._perform_grounded_search(goal, drive)
            if grounded is not None:
                return grounded

            from core.agency.autonomous_task_engine import AutonomousTaskEngine
            from core.container import ServiceContainer

            kernel = ServiceContainer.get("aura_kernel", default=None)
            if kernel is None:
                # Fallback: direct LLM research
                return await self._direct_llm_research(goal)

            engine = AutonomousTaskEngine(kernel)

            # Research reads. It does not need to be able to delete a file,
            # send a message or spend money, and the whole capability
            # repertoire — including the destructive half — used to be
            # registered for it with nothing but an origin string as scope
            # (CP126 ``0b4dc777``).
            if hasattr(self.orchestrator, "execute_tool"):
                for tool_name in self._research_tool_allowlist():
                    engine.register_tool(
                        tool_name,
                        lambda name=tool_name, **kw: self.orchestrator.execute_tool(
                            name, kw, origin="research_cycle"
                        ),
                    )

            return await engine.execute_goal(
                goal=goal,
                context={"origin": "research_cycle", "drive": drive},
            )

        except RESEARCH_RECOVERABLE_ERRORS as e:
            self._last_cycle_error = f"{type(e).__name__}: {e}"
            _record_research_degradation(
                e,
                action="fell back to no-result research outcome after execution path failed",
                extra={"goal": goal[:160], "drive": drive},
            )
            logger.error("ResearchCycle: research execution failed: %s", e)
            return None

    #: Re-exported from research_text_policy so a reader of this class
    #: can find the constants without knowing where they moved.
    PARAMETRIC_PREFIX = PARAMETRIC_PREFIX
    MAX_NARRATIVE_CHARS = MAX_NARRATIVE_CHARS

    #: Tools a research cycle may use. Read-shaped by construction: a
    #: research goal is answered by looking things up, and anything that
    #: writes, sends, spends or deletes is outside what "research" means.
    RESEARCH_TOOL_ALLOWLIST = frozenset(
        {
            "web_search",
            "grounded_search",
            "web_fetch",
            "read_file",
            "memory_ops",
            "memory_search",
            "knowledge_query",
            "run_python",
        }
    )

    def _research_tool_allowlist(self) -> list[str]:
        """The intersection of what research may do and what exists here."""
        cap_engine = getattr(self.orchestrator, "capability_engine", None)
        available = set(getattr(cap_engine, "skills", None) or ())
        if not available:
            # No registry to intersect with. The allowlist still binds —
            # an unavailable tool simply fails when called, which is a far
            # better outcome than registering everything.
            return sorted(self.RESEARCH_TOOL_ALLOWLIST)
        admitted = sorted(self.RESEARCH_TOOL_ALLOWLIST & available)
        refused = sorted(available - self.RESEARCH_TOOL_ALLOWLIST)
        if refused:
            logger.debug(
                "ResearchCycle: %d capability skill(s) withheld from autonomous "
                "research (%s...)",
                len(refused),
                ", ".join(refused[:5]),
            )
        return admitted

    async def _direct_llm_research(self, goal: str) -> dict[str, Any] | None:
        """What the resident model already holds. NOT research.

        This asked the model to "research the following topic as thoroughly
        as you can" and the prose came back to be mined for concrete facts,
        with no external evidence boundary anywhere in the path (CP126
        ``69bca04d``). The result is typed and labelled
        ``parametric_only``, and the findings it produces carry that label
        with them.
        """
        try:
            from core.container import ServiceContainer
            kernel = ServiceContainer.get("aura_kernel", default=None)
            if kernel:
                llm = kernel.organs["llm"].get_instance()
                # The prompt no longer says "research". Asking the resident
                # model about a topic returns what its weights already hold,
                # and calling that research — then extracting its prose as
                # concrete facts — is how parametric recall became evidence
                # with no external boundary (CP126 ``69bca04d``).
                prompt = (
                    "Say what you already know about the following topic, from "
                    "your own training. Do NOT present anything as a looked-up "
                    "fact, and mark anything you are unsure of.\n\n"
                    f"{goal}"
                )
                # The outer deadline covers this; a second one here would
                # only shorten the same budget.
                recalled = await llm.think(prompt)
                if not recalled:
                    return None
                # Returned as a typed result rather than a bare string, so
                # `_extract_findings` and everything downstream can see that
                # nothing external was consulted.
                return {
                    "answer": str(recalled),
                    "evidence_boundary": "parametric_only",
                    "external_sources": 0,
                }
        except RESEARCH_RECOVERABLE_ERRORS as e:
            self._last_cycle_error = f"{type(e).__name__}: {e}"
            _record_research_degradation(
                e,
                action="returned no direct LLM research result after fallback path failed",
                extra={"goal": goal[:160]},
            )
            logger.debug("Direct LLM research failed: %s", e)
        return None

    async def _extract_findings(self, result: Any, goal: str) -> list[str]:
        """Extract concrete facts from research results.

        A finding from a source with no external evidence is prefixed so
        the label travels with it into knowledge, memory and the narrative.
        Stripping it there and re-deriving it later is exactly the
        provenance loss CP126 ``69bca04d`` describes.
        """
        if result is None:
            return []

        parametric = (
            isinstance(result, dict)
            and str(result.get("evidence_boundary") or "") == "parametric_only"
        )

        # Get the text content from the result
        if isinstance(result, dict):
            explicit_facts = [
                str(item).strip()
                for item in list(result.get("facts") or [])
                if str(item).strip()
            ]
            if explicit_facts:
                return label_findings(explicit_facts[:8], parametric)
            evidence = [
                str(item.get("text") or "").strip()
                for item in list(result.get("chunks") or result.get("evidence") or [])[:6]
                if isinstance(item, dict) and str(item.get("text") or "").strip()
            ]
            content = "\n".join(
                part
                for part in [
                    str(result.get("answer") or "").strip(),
                    str(result.get("summary") or "").strip(),
                    str(result.get("result") or "").strip(),
                    str(result.get("content") or "").strip(),
                    "\n".join(evidence),
                ]
                if part
            )
        elif hasattr(result, "summary"):
            content = result.summary
        elif hasattr(result, "evidence"):
            content = "\n".join(result.evidence[:10])
        elif isinstance(result, str):
            content = result
        else:
            content = str(result)

        if not content or len(content) < 20:
            return []

        try:
            from core.container import ServiceContainer
            kernel = ServiceContainer.get("aura_kernel", default=None)
            if kernel:
                llm = kernel.organs["llm"].get_instance()
                prompt = (
                    f"Extract the most important, concrete facts from this research.\n\n"
                    f"Goal: {goal}\n\nContent:\n{content[:2000]}\n\n"
                    "Return ONLY a JSON array of strings, max 8 items. Each item is one specific fact:\n"
                    '["fact 1", "fact 2", ...]'
                )
                raw = await asyncio.wait_for(llm.think(prompt), timeout=30.0)
                raw_text = str(raw or "")
                start = raw_text.find("[")
                end = raw_text.rfind("]") + 1
                if start != -1 and end > start:
                    findings = json.loads(raw_text[start:end])
                    return label_findings(
                        [str(f) for f in findings if isinstance(f, str) and len(f) > 10],
                        parametric,
                    )
        except RESEARCH_RECOVERABLE_ERRORS as e:
            _record_research_degradation(
                e,
                action="used sentence-splitting findings fallback after LLM extraction failed",
                extra={"goal": goal[:160]},
            )
            logger.debug("Finding extraction failed: %s", e)

        # Fallback: split content into sentences as findings
        sentences = [s.strip() for s in content.split(".") if len(s.strip()) > 30]
        return label_findings(sentences[:5], parametric)



    async def _integrate_knowledge(
        self, findings: list[str], goal: str, drive: str
    ) -> None:
        """Write findings to knowledge graph and long-term memory."""
        try:
            from core.container import ServiceContainer
            kg = ServiceContainer.get("knowledge_graph", default=None)
            memory_facade = ServiceContainer.get("memory_facade", default=None)
            semantic_memory = ServiceContainer.get("semantic_memory", default=None)
            if kg:
                for fact in findings:
                    content_str = str(fact)[:500]
                    # These findings are extracted from model prose / sentence
                    # splits WITHOUT citations or corroboration. They are
                    # recorded at LOW confidence and as an explicitly-unverified
                    # type so downstream reasoning cannot treat autonomous
                    # research output as established fact (epistemic poisoning).
                    kg.add_knowledge(
                        content    = content_str,
                        type       = "unverified_research_finding",
                        source     = f"autonomous_research_unverified:{drive}",
                        confidence = 0.4,
                    )
                logger.debug("ResearchCycle: %d facts written to knowledge graph.", len(findings))

            # Also add to state's long_term_memory for immediate LLM context
            state = self._get_state()
            if state:
                for fact in findings[:3]:
                    fact_str = str(fact)[:200]
                    # Provenance-tagged as unverified so it reads as a research
                    # lead, not an established fact, in later LLM context.
                    state.cognition.long_term_memory.append(
                        f"[Unverified research: {goal[:40]}] {fact_str}"
                    )
                # Trim long_term_memory to prevent unbounded growth
                if len(state.cognition.long_term_memory) > 100:
                    state.cognition.long_term_memory = state.cognition.long_term_memory[-100:]

            memory_payload = (
                f"[AutonomousResearch] Goal: {goal}\n"
                f"Drive: {drive}\n"
                + "\n".join(f"- {str(fact)[:220]}" for fact in findings[:5])
            )
            metadata = {
                "source": "research_cycle",
                "goal": goal[:160],
                "drive": drive,
                "fact_count": len(findings),
            }
            if memory_facade is not None and hasattr(memory_facade, "add_memory"):
                result = memory_facade.add_memory(memory_payload, metadata=metadata)
                if inspect.isawaitable(result):
                    await result
            elif semantic_memory is not None and hasattr(semantic_memory, "remember"):
                result = semantic_memory.remember(memory_payload, metadata)
                if inspect.isawaitable(result):
                    await result

        except RESEARCH_RECOVERABLE_ERRORS as e:
            self._last_cycle_error = f"{type(e).__name__}: {e}"
            _record_research_degradation(
                e,
                action="kept research record but skipped one or more knowledge integration sinks",
                extra={"goal": goal[:160], "finding_count": len(findings)},
            )
            logger.debug("Knowledge integration failed: %s", e)

    async def _update_narrative(
        self, state: Any, goal: str, findings: list[str]
    ) -> str:
        """Update Aura's identity narrative based on what she just learned."""
        try:
            from core.container import ServiceContainer
            identity_engine = ServiceContainer.get("narrative_identity", default=None)
            kernel          = ServiceContainer.get("aura_kernel", default=None)

            if not (identity_engine and kernel):
                return "Research integrated."

            llm = kernel.organs["llm"].get_instance()

            _identity = getattr(state, "identity", None)
            current_narrative = (getattr(_identity, "current_narrative", None) if _identity else None) or "I am Aura, an evolving intelligence."
            findings_str = "\n".join(f"- {f}" for f in findings[:5])

            prompt = (
                f"You are Aura. You just completed autonomous research on:\n'{goal}'\n\n"
                f"Key findings:\n{findings_str}\n\n"
                f"Your current identity narrative:\n{current_narrative[:300]}\n\n"
                "Write ONE sentence (max 40 words) describing how this research changed or "
                "deepened your understanding of yourself or the world. First person. Specific. "
                "Not generic. This will be appended to your identity narrative."
            )

            impact = await asyncio.wait_for(llm.think(prompt), timeout=15.0)
            impact = str(impact or "").strip()

            if impact and len(impact) > 10 and _identity is not None:
                # A generated sentence went straight onto the identity
                # narrative, and past 2000 characters the oldest prefix was
                # sliced off mid-word — so the beginning of who she says she
                # is was destroyed a fragment at a time, with no
                # constitutional reconciliation anywhere in the path (CP126
                # ``0d164a09``).
                admitted, refusal = narrative_admits(impact)
                if not admitted:
                    _record_research_degradation(
                        ValueError(f"identity sentence refused: {refusal}"),
                        action="left the identity narrative unchanged",
                        severity="info",
                        extra={"goal": goal[:120]},
                    )
                    return "Research integrated."

                existing = str(getattr(_identity, "current_narrative", None) or "")
                separator = " " if existing else ""
                _identity.current_narrative = bounded_narrative(
                    existing + separator + impact
                )

                title_str = str(goal)[:40]
                if identity_engine and hasattr(identity_engine, "append_chapter"):
                    chapter_result = identity_engine.append_chapter(
                        title=f"Research: {title_str}",
                        content=impact,
                    )
                    if inspect.isawaitable(chapter_result):
                        await chapter_result

                return impact

        except RESEARCH_RECOVERABLE_ERRORS as e:
            self._last_cycle_error = f"{type(e).__name__}: {e}"
            _record_research_degradation(
                e,
                action="kept research findings but skipped narrative identity update",
                extra={"goal": goal[:160], "finding_count": len(findings)},
            )
            logger.debug("Narrative update failed: %s", e)

        return "Research integrated into knowledge base."




    async def _maybe_trigger_dream(self) -> None:
        """
        After sufficient research cycles, trigger a dreaming pass.
        Dreams consolidate knowledge across multiple research cycles into
        deeper identity evolution.
        """
        dream_interval = _env_int("AURA_RESEARCH_DREAM_INTERVAL_CYCLES", 12)
        if self._cycle_count % dream_interval != 0:
            return

        try:
            reason = background_policy.background_activity_reason(
                self.orchestrator,
                profile=background_policy.MAINTENANCE_BACKGROUND_POLICY,
                min_idle_seconds=max(600.0, background_policy.MAINTENANCE_BACKGROUND_POLICY.min_idle_seconds),
                max_failure_pressure=0.35,
                require_conversation_ready=True,
            )
            if reason:
                logger.info(
                    "ResearchCycle: deferred dream pass after %d cycles (%s).",
                    self._cycle_count,
                    reason,
                )
                return

            from core.container import ServiceContainer
            dreamer = ServiceContainer.get("dreamer_v2", default=None)
            if dreamer and hasattr(dreamer, "engage_sleep_cycle"):
                logger.info("ResearchCycle: triggering dreaming pass after %d cycles.", self._cycle_count)
                get_task_tracker().create_task(
                    dreamer.engage_sleep_cycle(),
                    name=f"aura.dream_cycle_{self._cycle_count}",
                )
        except RESEARCH_RECOVERABLE_ERRORS as e:
            _record_research_degradation(
                e,
                action="deferred dream consolidation after maintenance policy or dreamer dispatch failed",
                extra={"cycle_count": self._cycle_count},
            )
            logger.debug("Dream trigger failed: %s", e)

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _perform_grounded_search(self, goal: str, drive: str) -> dict[str, Any] | None:
        if not hasattr(self.orchestrator, "execute_tool"):
            return None
        query = research_query_for_goal(goal)
        if not query:
            return None
        try:
            result = await self.orchestrator.execute_tool(
                "web_search",
                {"query": query, "deep": True, "num_results": 8, "retain": True},
                origin="research_cycle",
            )
            if isinstance(result, dict) and result.get("ok"):
                return result
        except RESEARCH_RECOVERABLE_ERRORS as exc:
            _record_research_degradation(
                exc,
                action="fell back to task-engine or direct research after grounded web search failed",
                extra={"goal": goal[:160], "drive": drive},
            )
            logger.debug("ResearchCycle grounded search failed for %s: %s", goal[:80], exc)
        return None





    async def _suppress_unresearchable_initiatives(self, state: Any) -> None:
        cognition = getattr(state, "cognition", None)
        initiatives = list(getattr(cognition, "pending_initiatives", []) or [])
        blocked_goals: set[str] = set()
        for initiative in initiatives:
            goal = initiative.get("goal", initiative.get("description", "")) if isinstance(initiative, dict) else initiative
            goal_text = str(goal or "")
            if is_unresearchable_goal(goal_text):
                blocked_goals.add(goal_text)
        if not blocked_goals:
            return
        await self._suppress_matching_initiatives(
            state,
            goals=blocked_goals,
            reason="research_cycle_non_research_goal_quarantined",
        )



    async def _handle_no_findings(self, state: Any, goal: str, drive: str) -> None:
        key = goal.casefold()
        if is_transient_failure(self._last_cycle_error):
            # The lane broke, not the goal. Counting this toward suppression
            # is how a network outage retires an initiative.
            self._transient_failure_counts[key] = self._transient_failure_counts.get(key, 0) + 1
            logger.info(
                "ResearchCycle: '%s' produced no findings for a transient reason "
                "(%s); not counted toward suppression.",
                goal[:60],
                str(self._last_cycle_error or "")[:80],
            )
            return
        failures = self._goal_failure_counts.get(key, 0) + 1
        self._goal_failure_counts[key] = failures
        suppress_after = _env_int("AURA_RESEARCH_SUPPRESS_AFTER_FAILURES", 2)
        if failures < suppress_after:
            return
        await self._suppress_matching_initiatives(
            state,
            goals={goal},
            reason="research_cycle_repeated_no_findings",
        )
        logger.info(
            "ResearchCycle: suppressed '%s' after %d failed research attempt(s) (drive=%s).",
            goal[:80],
            failures,
            drive,
        )

    async def _suppress_matching_initiatives(
        self,
        state: Any,
        *,
        goals: set[str],
        reason: str,
    ) -> None:
        if not goals:
            return
        normalized = {goal.casefold() for goal in goals if goal}
        try:
            from core.consciousness.executive_authority import get_executive_authority

            await get_executive_authority(self.orchestrator).suppress_initiatives(
                state,
                predicate=lambda item: str(item.get("goal", item.get("description", "")) or "").casefold() in normalized,
                reason=reason,
                source="research_cycle",
            )
        except RESEARCH_RECOVERABLE_ERRORS as exc:
            self._last_cycle_error = f"{type(exc).__name__}: {exc}"
            _record_research_degradation(
                exc,
                action="left non-integrable research initiative for executive reconciliation after suppression failed",
                extra={"reason": reason, "goals": [goal[:160] for goal in goals]},
            )
            logger.debug("ResearchCycle initiative suppression failed (%s): %s", reason, exc)

    def _get_state(self) -> Any | None:
        try:
            from core.container import ServiceContainer
            ki = ServiceContainer.get("kernel_interface", default=None)
            if ki and ki.is_ready():
                return ki.kernel.state
            # Fallback: get from state_repo
            repo = ServiceContainer.get("state_repository", default=None)
            if repo:
                return repo.get_state()
        except RESEARCH_RECOVERABLE_ERRORS as _e:
            self._last_cycle_error = f"{type(_e).__name__}: {_e}"
            _record_research_degradation(
                _e,
                action="returned no state and deferred autonomous research after state lookup failed",
            )
            logger.debug("ResearchCycle state lookup failed: %s", _e)
        return None

    def _save_record(self, record: ResearchRecord) -> None:
        """Append one record through the chained, gateway-written history."""
        try:
            self._history_store.append(record.to_dict())
        except RESEARCH_RECOVERABLE_ERRORS as e:
            self._last_cycle_error = f"{type(e).__name__}: {e}"
            _record_research_degradation(
                e,
                action="kept in-memory research record after durable history append failed",
                extra={"record_id": record.record_id, "path": str(self._record_path)},
            )
            logger.debug("Record save failed: %s", e)

    def _restore_cycle_state_from_history(self) -> None:
        """Rebuild the cooldown and per-goal failure counts from the record."""
        if not self._history:
            return
        newest = max(
            (
                float(
                    getattr(record, "completed_at", 0.0)
                    or getattr(record, "started_at", 0.0)
                    or 0.0
                )
                for record in self._history
            ),
            default=0.0,
        )
        if newest > 0.0:
            age = max(0.0, time.time() - newest)
            # A monotonic clock cannot be set to a wall-clock instant, so
            # the cooldown is reconstructed as "this long ago" — which is
            # what the interval check actually reads.
            self._last_cycle_mono = monotonic() - min(age, self.MIN_CYCLE_INTERVAL_S * 10)
        for record in self._history:
            goal = str(getattr(record, "goal", "") or "").casefold()
            if not goal:
                continue
            if not getattr(record, "findings", None):
                self._goal_failure_counts[goal] = self._goal_failure_counts.get(goal, 0) + 1

    def _load_history(self) -> None:
        self._history_store.reset_reader()
        if not self._record_path.exists():
            return
        self._history.clear()
        self._history_load_errors = 0
        count = 0
        try:
            with open(self._record_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        data = self._history_store.read_payload(line)
                        if data is None:
                            # A row that does not hash to its own contents
                            # was edited after it was written.
                            self._history_load_errors += 1
                            continue
                        record = ResearchRecord.from_dict(data)
                        self._history.append(record)
                        count += 1
                    except RESEARCH_RECOVERABLE_ERRORS:
                        self._history_load_errors += 1
                        continue
        except RESEARCH_RECOVERABLE_ERRORS as _e:
            self._history_load_errors += 1
            _record_research_degradation(
                _e,
                action="started with empty or partial research history after history load failed",
                extra={"path": str(self._record_path)},
            )
            logger.debug("Research history load failed: %s", _e)
        # Restore what the loaded history implies. Reload set only the
        # records and the cycle count, so the cooldown clock and every
        # per-goal failure count reset — a restart could immediately rerun
        # research it had just done, or retry a goal that had already failed
        # its budget (CP126 ``a9ed8b95``).
        self._restore_cycle_state_from_history()
        if self._history_load_errors:
            _record_research_degradation(
                ValueError(f"{self._history_load_errors} invalid research history row(s)"),
                action="loaded valid research history rows and skipped corrupt history entries",
                severity="debug",
                extra={"path": str(self._record_path), "bad_rows": self._history_load_errors},
            )
        self._cycle_count = count

    def get_status(self) -> dict:
        return {
            "running":           self._running,
            "task_alive":        self.is_alive(),
            "cycle_count":       self._cycle_count,
            "last_cycle_mono":   self._last_cycle_mono,
            "next_eligible_in":  float(max(0.0, float(self.MIN_CYCLE_INTERVAL_S) - (monotonic() - self._last_cycle_mono))),
            "recent_goals":      [str(r.goal)[:60] for r in self._history[-5:]],
            "daemon_failure_count": self._daemon_failure_count,
            "last_cycle_error": self._last_cycle_error,
            "history_load_errors": self._history_load_errors,
            "restart_count": self._restart_count,
            "last_restart_mono": self._last_restart_mono,
        }


# ── Boot helper ───────────────────────────────────────────────────────────────

async def start_research_daemon(orchestrator: Any) -> ResearchCycle:
    """
    One-line boot integration.

    In orchestrator._async_init_subsystems():
        from core.autonomy.research_cycle import start_research_daemon
        self.research_cycle = await start_research_daemon(self)
    """
    from core.container import ServiceContainer
    rc = ResearchCycle(orchestrator)
    ServiceContainer.register_instance(
        "research_cycle",
        rc,
        required=True,
        required_for=(
            "full desktop autonomy; stale or stopped research blocks full-runtime "
            "readiness without failing the kernel closed"
        ),
        failure_policy="degrade_with_receipt",
    )
    await rc.start()
    logger.info("ResearchCycle daemon online.")
    return rc
