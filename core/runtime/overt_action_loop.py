"""Overt autonomous action executor.

The initiative funnel is only meaningful if selected initiatives become real,
measured work. This module owns the last mile:

    synthesize -> execute a governed skill -> verify -> receipt -> goal update

It is intentionally conservative. One cycle performs at most one concrete
tool action, records why it acted or skipped, and leaves enough evidence for a
human to reconstruct what happened later.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import sqlite3
import time
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from types import SimpleNamespace
from typing import Any

from core.container import ServiceContainer
from core.health.degraded_events import record_degraded_event
from core.runtime.background_policy import background_activity_reason
from core.runtime.errors import record_degradation
from core.runtime.service_access import resolve_orchestrator
from core.utils.task_tracker import get_task_tracker

SAFE_AUTONOMOUS_SKILLS = (
    "auto_refactor",
    "system_proprioception",
    "environment_info",
    "file_operation",
    "clock",
    "evolution_status",
)

_RETAINED_EVIDENCE_RE = re.compile(
    r"\[\s*retained\s+memory\s+evidence\s*\].*$",
    re.IGNORECASE | re.DOTALL,
)
_WEB_SEARCH_INTENT_PATTERNS = (
    re.compile(
        r"^\s*(?:please\s+)?(?:search|look\s+up|research|find)\s+"
        r"(?:the\s+)?(?:web|internet|online)\s+(?:for|about)\s+(?P<query>.+)$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?:please\s+)?(?:web|internet|online)\s+(?:search|research)\s+"
        r"(?:for|about)\s+(?P<query>.+)$",
        re.IGNORECASE,
    ),
)


@dataclass
class OvertActionResult:
    action_id: str
    status: str
    objective: str = ""
    source: str = ""
    skill: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    will_receipt_id: str = ""
    tool_receipt_id: str = ""
    autonomy_receipt_id: str = ""
    life_trace_id: str = ""
    verified: bool = False
    result_summary: str = ""
    error: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0
    duration_ms: float = 0.0
    goal_id: str = ""
    next_step_hint: str = ""
    action_expectation: dict[str, Any] = field(default_factory=dict)
    expectation_verdict: dict[str, Any] = field(default_factory=dict)
    expectation_receipt_id: str = ""
    selection_provenance: str = ""
    selection_reason: str = ""
    execution_mode: str = "skill"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ActionSelection:
    """Concrete, attributable action chosen from an initiative contract."""

    skill: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    provenance: str = ""
    reason: str = ""
    execution_mode: str = "skill"

    @property
    def actionable(self) -> bool:
        if self.reason:
            return False
        if self.execution_mode == "planned_goal":
            return bool(str(self.params.get("goal") or "").strip())
        return bool(self.skill)


def _json_digest(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str)[:20000]
    return hashlib.blake2b(raw.encode("utf-8"), digest_size=16).hexdigest()


def _short_text(value: Any, limit: int = 500) -> str:
    return " ".join(str(value or "").strip().split())[:limit]


class OvertActionLoop:
    """Executes one visible, governed autonomous action at a time."""

    def __init__(
        self,
        *,
        orchestrator: Any = None,
        capability_engine: Any = None,
        goal_engine: Any = None,
        synthesizer: Any = None,
        state_provider: Callable[[], Any] | None = None,
        receipt_store: Any = None,
        interval_s: float | None = None,
    ) -> None:
        self.orchestrator = orchestrator
        self.capability_engine = capability_engine
        self.goal_engine = goal_engine
        self.synthesizer = synthesizer
        self.state_provider = state_provider
        self.receipt_store = receipt_store
        self.interval_s = float(interval_s if interval_s is not None else os.getenv("AURA_OVERT_ACTION_INTERVAL_S", "120"))
        self._lock = asyncio.Lock()
        self._history: deque[OvertActionResult] = deque(maxlen=50)
        self._started_at = time.time()
        self._last_started_at = 0.0
        self._last_finished_at = 0.0
        self._consecutive_failures = 0
        self._actions_started = 0
        self._actions_verified = 0
        self._skips = 0

    @staticmethod
    def enabled() -> bool:
        return os.getenv("AURA_OVERT_ACTIONS", "1").strip().lower() not in {"0", "false", "off", "no"}

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled(),
            "uptime_s": round(time.time() - self._started_at, 1),
            "interval_s": self.interval_s,
            "actions_started": self._actions_started,
            "actions_verified": self._actions_verified,
            "skips": self._skips,
            "consecutive_failures": self._consecutive_failures,
            "last_started_at": self._last_started_at,
            "last_finished_at": self._last_finished_at,
            "last_action": self._history[-1].to_dict() if self._history else None,
            "recent": [item.to_dict() for item in list(self._history)[-5:]],
        }

    async def run_once(self, *, force: bool = False) -> dict[str, Any]:
        async with self._lock:
            if not self.enabled() and not force:
                return self._record_skip("disabled").to_dict()

            reason = self._background_reason()
            if reason and not force:
                return self._record_skip(f"background_policy:{reason}").to_dict()

            state = self._state()
            synth_result = await self._synthesize(state)
            initiative = dict(getattr(synth_result, "winner", None) or {})
            if not initiative:
                if os.getenv("AURA_OVERT_ACTION_FALLBACK", "0").strip().lower() in {"0", "false", "off", "no"}:
                    return self._record_skip("no_authorized_initiative").to_dict()
                initiative = self._fallback_initiative()
                will_receipt_id = self._authorize_fallback(initiative)
                if not will_receipt_id:
                    return self._record_skip("fallback_not_authorized").to_dict()
            else:
                will_receipt_id = str(getattr(synth_result, "will_receipt_id", "") or "")

            goal = await self._goal_for_initiative(initiative)
            selection = self._choose_skill_and_params(initiative, goal)
            if not selection.actionable:
                return self._record_initiative_skip(
                    initiative,
                    selection,
                    will_receipt_id=will_receipt_id,
                ).to_dict()

            action = await self._execute_initiative(
                initiative,
                goal=goal,
                selection=selection,
                will_receipt_id=will_receipt_id,
            )
            self._history.append(action)
            return action.to_dict()

    def _background_reason(self) -> str:
        allow_boot_anchor = os.getenv(
            "AURA_OVERT_ACTION_ALLOW_BOOT_ANCHOR",
            "1",
        ).strip().lower() not in {"0", "false", "off", "no"}
        reason = background_activity_reason(
            self._orchestrator(),
            min_idle_seconds=float(os.getenv("AURA_OVERT_ACTION_IDLE_S", "30")),
            max_memory_percent=float(os.getenv("AURA_OVERT_ACTION_MAX_MEMORY_PERCENT", "88")),
            max_failure_pressure=float(os.getenv("AURA_OVERT_ACTION_MAX_FAILURE_PRESSURE", "0.35")),
            require_conversation_ready=False,
            allow_no_user_anchor=allow_boot_anchor,
        )
        return str(reason or "")

    def _record_skip(self, reason: str) -> OvertActionResult:
        self._skips += 1
        now = time.time()
        result = OvertActionResult(
            action_id="skip_" + hashlib.sha256(f"{now}:{reason}".encode()).hexdigest()[:10],
            status="skipped",
            error=reason,
            started_at=now,
            finished_at=now,
            next_step_hint="wait_for_idle_window" if reason.startswith("background_policy") else "",
        )
        self._history.append(result)
        return result

    def _record_initiative_skip(
        self,
        initiative: dict[str, Any],
        selection: ActionSelection,
        *,
        will_receipt_id: str,
    ) -> OvertActionResult:
        reason = selection.reason or "missing_action_contract"
        result = self._record_skip(f"initiative_not_actionable:{reason}")
        result.objective = _short_text(
            initiative.get("goal") or initiative.get("objective"),
            1000,
        )
        result.source = str(initiative.get("source") or "")
        result.will_receipt_id = will_receipt_id
        result.selection_provenance = selection.provenance
        result.selection_reason = reason
        result.execution_mode = selection.execution_mode
        result.next_step_hint = (
            "retain_as_evidence_without_execution"
            if "evidence" in reason or "memory" in reason
            else "supply_an_actionable_objective"
        )
        self._emit_selection_skip_receipt(result)
        self._record_selection_skip_trace(result)
        return result

    def _orchestrator(self) -> Any:
        if self.orchestrator is not None:
            return self.orchestrator
        self.orchestrator = resolve_orchestrator()
        return self.orchestrator

    def _capability_engine(self) -> Any:
        if self.capability_engine is not None:
            return self.capability_engine
        self.capability_engine = ServiceContainer.get("capability_engine", default=None)
        return self.capability_engine

    def _goal_engine(self) -> Any:
        if self.goal_engine is not None:
            return self.goal_engine
        self.goal_engine = ServiceContainer.get("goal_engine", default=None)
        return self.goal_engine

    def _synthesizer(self) -> Any:
        if self.synthesizer is not None:
            return self.synthesizer
        from core.initiative_synthesis import get_initiative_synthesizer

        self.synthesizer = get_initiative_synthesizer()
        return self.synthesizer

    def _state(self) -> Any:
        if self.state_provider is not None:
            state = self.state_provider()
            if state is not None:
                return state
        repo = ServiceContainer.get("state_repo", default=None)
        state = getattr(repo, "_current", None) if repo is not None else None
        if state is not None:
            return state
        return SimpleNamespace(cognition=SimpleNamespace(pending_initiatives=[]))

    async def _synthesize(self, state: Any) -> Any:
        synth = self._synthesizer()
        if hasattr(synth, "start"):
            maybe = synth.start()
            if asyncio.iscoroutine(maybe):
                await maybe
        return await synth.synthesize(state)

    def _fallback_initiative(self) -> dict[str, Any]:
        return {
            "goal": "Run a light self-audit and record an overt action receipt.",
            "source": "overt_action_loop",
            "type": "fallback_maintenance",
            "urgency": 0.45,
            "triggered_by": "maintenance",
            "metadata": {"required_skills": ["system_proprioception"]},
        }

    def _authorize_fallback(self, initiative: dict[str, Any]) -> str:
        try:
            from core.will import ActionDomain, get_will

            decision = get_will().decide(
                content=str(initiative.get("goal", ""))[:240],
                source="overt_action_loop",
                domain=ActionDomain.INITIATIVE,
                priority=float(initiative.get("urgency", 0.45) or 0.45),
            )
            return decision.receipt_id if decision.is_approved() else ""
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation("overt_action_loop", exc)
            return ""

    async def _execute_initiative(
        self,
        initiative: dict[str, Any],
        *,
        goal: dict[str, Any],
        selection: ActionSelection,
        will_receipt_id: str,
    ) -> OvertActionResult:
        started = time.time()
        self._last_started_at = started
        self._actions_started += 1
        objective = _short_text(initiative.get("goal") or initiative.get("objective"), 1000)
        skill, params = selection.skill, dict(selection.params)
        action_id = hashlib.sha256(f"{started}:{objective}:{skill}".encode()).hexdigest()[:16]
        from core.actuators.actuator_registry import get_actuator_registry

        registry = get_actuator_registry()
        expectation = self._action_expectation_for(
            skill,
            params,
            initiative,
            goal,
            actuator_backed=registry.get_actuator(skill) is not None,
        )
        result = OvertActionResult(
            action_id=action_id,
            status="started",
            objective=objective,
            source=str(initiative.get("source") or ""),
            skill=skill,
            params=params,
            will_receipt_id=will_receipt_id,
            started_at=started,
            goal_id=str(goal.get("id") or initiative.get("metadata", {}).get("goal_id") or ""),
            action_expectation=expectation.to_dict() if expectation is not None else {},
            selection_provenance=selection.provenance,
            selection_reason=selection.reason,
            execution_mode=selection.execution_mode,
        )
        governance_context = {
            "origin": "overt_action_loop",
            "source": "overt_action_loop",
            "objective": objective,
            "will_receipt_id": will_receipt_id,
            "autonomous": True,
            "initiative": initiative,
            "priority": float(initiative.get("urgency", 0.7) or 0.7),
            "requested_authority_scope": f"overt_action_loop:{action_id}:{skill}",
            "authorization": "governed_autonomous_overt_action",
            "action_selection": {
                "provenance": selection.provenance,
                "reason": selection.reason,
                "execution_mode": selection.execution_mode,
            },
        }
        orchestrator = self._orchestrator()
        if orchestrator is not None:
            governance_context["orchestrator"] = orchestrator
        if expectation is not None:
            governance_context["action_expectation"] = expectation.to_dict()

        if selection.execution_mode == "planned_goal":
            raw = await self._execute_planned_goal(
                objective,
                governance_context=governance_context,
            )
        elif registry.get_actuator(skill) is not None:
            try:
                actuator_res = await registry.execute_action_async(
                    skill,
                    params,
                    context=governance_context,
                )
                raw = {
                    "ok": actuator_res.success,
                    "message": actuator_res.message,
                    "updates": actuator_res.updates,
                    "success": actuator_res.success,
                }
            except (ImportError, RuntimeError, OSError, AttributeError, TypeError, ValueError) as exc:
                result.status = "failed"
                result.error = f"Actuator {skill} failed: {exc}"
                raw = {"ok": False, "error": result.error}
        else:
            engine = self._capability_engine()
            if engine is None or not hasattr(engine, "execute"):
                result.status = "failed"
                result.error = "capability_engine_unavailable"
                return await self._finish(result, raw_result={})

            try:
                raw = await engine.execute(
                    skill,
                    params,
                    context=governance_context,
                )
            except (sqlite3.Error, OSError) as exc:
                result.status = "failed"
                result.error = f"{type(exc).__name__}: {exc}"
                record_degradation(
                    "overt_action_loop",
                    exc,
                    severity="warning",
                    action="failed overt initiative execution and preserved action receipt for review",
                    extra={
                        "action_id": action_id,
                        "skill": skill,
                        "objective": objective[:240],
                    },
                )
                raw = {"ok": False, "error": result.error}

        if isinstance(raw, dict) and raw.get("status") == "deferred":
            result.status = "deferred"
            result.error = ""
            result.result_summary = self._summarize_result(raw)
            result.next_step_hint = "retry_after_idle_and_resource_window"
            return await self._finish(result, raw_result=raw)

        if isinstance(raw, dict) and expectation is not None:
            from core.runtime.skill_contract import apply_action_expectation_payload

            raw = apply_action_expectation_payload(skill, raw, expectation)
        if isinstance(raw, dict):
            verdict = raw.get("expectation_verdict")
            if isinstance(verdict, dict):
                result.expectation_verdict = dict(verdict)
                if not bool(verdict.get("passed", False)):
                    result.next_step_hint = str(verdict.get("next_step") or "")
            result.expectation_receipt_id = str(
                raw.get("expectation_receipt_id") or ""
            )
        result.verified = self._verify(skill, params, raw)
        result.status = "verified" if result.verified else "failed"
        if not result.verified and result.expectation_verdict:
            result.status = str(
                result.expectation_verdict.get("status") or "failed_recoverable"
            )
        result.result_summary = self._summarize_result(raw)
        if not result.verified and not result.error:
            result.error = str(raw.get("error") or raw.get("status") or "verification_failed") if isinstance(raw, dict) else "verification_failed"
        try:
            from core.consciousness.closed_loop import (
                notify_closed_loop_action_outcome,
            )

            notify_closed_loop_action_outcome(
                skill,
                success=bool(isinstance(raw, dict) and raw.get("ok", False)),
                verified=result.verified,
                updates=(
                    dict(raw.get("updates") or {})
                    if isinstance(raw, dict) and isinstance(raw.get("updates"), dict)
                    else {}
                ),
                error=result.error,
            )
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "overt_action_loop",
                exc,
                severity="warning",
                action="preserved action result after closed-loop feedback failed",
                extra={"action_id": action_id, "skill": skill},
            )
        return await self._finish(result, raw_result=raw)

    async def _execute_planned_goal(
        self,
        objective: str,
        *,
        governance_context: dict[str, Any],
    ) -> dict[str, Any]:
        """Compile a self-chosen objective into governed, verified tool steps."""
        try:
            from core.runtime.service_access import resolve_task_engine

            task_engine = resolve_task_engine(default=None)
            if task_engine is None or not hasattr(task_engine, "execute_goal"):
                raise RuntimeError("autonomous_task_engine_unavailable")
            task_result = await task_engine.execute_goal(
                objective,
                context={
                    **governance_context,
                    "requested_by": "aura",
                    "requested_via": "overt_action_loop",
                    "foreground_request": False,
                },
            )
        except (ImportError, RuntimeError, OSError, AttributeError, TypeError, ValueError) as exc:
            record_degradation(
                "overt_action_loop.semantic_plan",
                exc,
                severity="warning",
                action="failed the selected autonomous objective without claiming execution",
                extra={"objective": objective[:240]},
            )
            return {
                "ok": False,
                "status": "semantic_plan_failed",
                "error": f"{type(exc).__name__}: {exc}",
            }

        task_payload = asdict(task_result)
        if task_result.deferred_reason:
            return {
                "ok": False,
                "status": "deferred",
                "summary": task_result.summary,
                "plan_id": task_result.plan_id,
                "trace_id": task_result.trace_id,
                "steps_completed": 0,
                "steps_total": 0,
                "evidence": [],
                "result": task_payload,
                "reason": task_result.deferred_reason,
                "error": "",
            }
        steps_total = int(task_result.steps_total or 0)
        steps_completed = int(task_result.steps_completed or 0)
        completed = bool(
            task_result.succeeded
            and steps_total > 0
            and steps_completed == steps_total
        )
        return {
            "ok": completed,
            "status": "completed_verified" if completed else "execution_not_completed",
            "summary": task_result.summary,
            "plan_id": task_result.plan_id,
            "trace_id": task_result.trace_id,
            "steps_completed": steps_completed,
            "steps_total": steps_total,
            "evidence": list(task_result.evidence or []),
            "result": task_payload,
            "error": "" if completed else task_result.summary,
        }

    @classmethod
    def _action_expectation_for(
        cls,
        skill: str,
        params: dict[str, Any],
        initiative: dict[str, Any],
        goal: dict[str, Any],
        *,
        actuator_backed: bool,
    ) -> Any | None:
        from core.runtime.skill_contract import ActionExpectation

        metadata = dict(initiative.get("metadata") or {})
        goal_metadata = dict(goal.get("metadata") or {})
        raw = (
            initiative.get("action_expectation")
            or metadata.get("action_expectation")
            or goal.get("action_expectation")
            or goal_metadata.get("action_expectation")
        )
        if isinstance(raw, ActionExpectation):
            return raw
        source = dict(raw) if isinstance(raw, dict) else {}
        for key in (
            "acceptance_criteria",
            "required_evidence",
            "required_evidence_present",
            "semantic_predicates",
            "user_visible_effect",
            "repair_hint",
            "rollback_hint",
            "allow_partial",
        ):
            for candidate in (initiative, metadata, goal_metadata):
                if key in candidate and key not in source:
                    source[key] = candidate[key]
        objective = _short_text(
            source.get("objective")
            or initiative.get("goal")
            or initiative.get("objective")
            or goal.get("objective")
            or skill,
            1000,
        )
        if source:
            source.setdefault("objective", objective)
            source.setdefault("repair_hint", f"repair_overt_{skill}_expectation")
            source.setdefault(
                "rollback_hint",
                "not_required_read_only" if not actuator_backed else "domain_specific_rollback_required",
            )
            source.setdefault("allow_partial", False)
            from core.capability_engine import CapabilityEngine

            explicit = CapabilityEngine.action_expectation_for(
                skill,
                params,
                {"objective": objective, "action_expectation": source},
            )
            if explicit is not None:
                return explicit

        if actuator_backed:
            if skill == "web_search":
                from core.capability_engine import CapabilityEngine

                deep_research = CapabilityEngine._web_query_requires_sources(
                    params,
                    {"objective": objective},
                )
                required = ["updates.search_results.summary"]
                if deep_research:
                    required.append("updates.search_results.sources")
                return ActionExpectation(
                    objective=objective,
                    required_evidence=required,
                    repair_hint="rerun_overt_web_search_with_source_evidence",
                    rollback_hint="not_required_read_only",
                    allow_partial=False,
                )
            return ActionExpectation(
                objective=objective,
                required_evidence=["updates"],
                repair_hint=f"verify_overt_{skill}_applied_updates",
                rollback_hint="domain_specific_rollback_required",
                allow_partial=False,
            )

        from core.capability_engine import CapabilityEngine

        default_expectation = CapabilityEngine.action_expectation_for(
            skill,
            params,
            {"objective": objective},
        )
        if default_expectation is not None:
            return default_expectation

        known_evidence: dict[str, tuple[list[str], list[str]]] = {
            "auto_refactor": ([], ["issues_found", "top_issues"]),
            "clock": (["time"], []),
            "environment_info": (["result"], []),
            "evolution_status": ([], ["status"]),
            "system_proprioception": (["system_map"], []),
        }
        if skill == "file_operation" and str(params.get("action") or "").lower() == "exists":
            return ActionExpectation(
                objective=objective,
                required_evidence_present=["exists"],
                repair_hint="repeat_path_observation",
                rollback_hint="not_required_read_only",
                allow_partial=False,
            )
        if skill not in known_evidence:
            return None
        required, present = known_evidence[skill]
        return ActionExpectation(
            objective=objective,
            required_evidence=required,
            required_evidence_present=present,
            repair_hint=f"repeat_overt_{skill}_with_evidence",
            rollback_hint="not_required_read_only",
            allow_partial=False,
        )

    async def _goal_for_initiative(self, initiative: dict[str, Any]) -> dict[str, Any]:
        metadata = dict(initiative.get("metadata", {}) or {})
        goal_id = str(metadata.get("goal_id") or initiative.get("goal_id") or "")
        goal_engine = self._goal_engine()
        if goal_id and goal_engine is not None and hasattr(goal_engine, "get_goal"):
            try:
                return dict(await asyncio.to_thread(goal_engine.get_goal, goal_id) or {})
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                record_degradation("overt_action_loop", exc)
        return {}

    def _choose_skill_and_params(
        self,
        initiative: dict[str, Any],
        goal: dict[str, Any],
    ) -> ActionSelection:
        metadata = dict(initiative.get("metadata", {}) or {})
        action_text = self._action_text(initiative, goal)
        params = self._action_params(initiative, goal, metadata)
        requested = self._explicit_action_names(initiative, goal, metadata)

        from core.actuators.actuator_registry import get_actuator_registry

        registry = get_actuator_registry()
        aliases = {
            "code_execution": {"code_execution", "run_code", "execute_code", "python"},
            "web_search": {"web_search", "search", "lookup"},
            "web_fetch": {"web_fetch", "fetch", "download"},
            "git_operation": {"git_operation", "git", "clone", "commit", "checkout"},
            "package_install": {"package_install", "pip", "install"},
            "process_supervisor": {"process_supervisor", "process", "spawn", "background"},
            "document_ingest": {"document_ingest", "doc_ingest", "ingest", "pdf", "html"},
        }

        invalid_reasons: list[str] = []
        for requested_name in requested:
            normalized = self._normalize_skill_name(requested_name)
            canonical = next(
                (
                    skill_name
                    for skill_name, skill_aliases in aliases.items()
                    if normalized in skill_aliases
                ),
                normalized,
            )
            prepared, invalid_reason = self._prepare_explicit_action(
                canonical,
                params,
                action_text=action_text,
                registry=registry,
            )
            if prepared is not None:
                return ActionSelection(
                    skill=canonical,
                    params=prepared,
                    provenance=f"structured:{requested_name}",
                )
            invalid_reasons.append(f"{canonical}:{invalid_reason}")

        if requested:
            return ActionSelection(
                provenance="structured",
                reason="invalid_explicit_action:" + ",".join(invalid_reasons)[:240],
            )

        evidence_reason = self._non_action_evidence_reason(initiative, goal, action_text)
        if evidence_reason:
            return ActionSelection(
                provenance="non_action_evidence",
                reason=evidence_reason,
            )

        for pattern in _WEB_SEARCH_INTENT_PATTERNS:
            match = pattern.fullmatch(action_text)
            if match:
                query = _short_text(match.group("query"), 500)
                if query:
                    return ActionSelection(
                        skill="web_search",
                        params={**params, "query": query},
                        provenance="natural_language:explicit_web_search",
                    )

        lowered = action_text.casefold()
        if re.fullmatch(
            r"(?:run|perform|conduct)\s+(?:a\s+)?(?:system|runtime)\s+"
            r"(?:self[- ]?)?(?:audit|inspection|health check)",
            lowered,
        ):
            return ActionSelection(
                skill="system_proprioception",
                params={"include_docstrings": False},
                provenance="natural_language:system_audit",
            )
        if re.fullmatch(
            r"(?:check|inspect|report)\s+(?:the\s+)?(?:system\s+)?environment(?:\s+info)?",
            lowered,
        ):
            return ActionSelection(
                skill="environment_info",
                params={"detail": "basic"},
                provenance="natural_language:environment_status",
            )
        if re.fullmatch(
            r"(?:check|report)\s+(?:the\s+)?(?:current\s+)?time",
            lowered,
        ):
            return ActionSelection(
                skill="clock",
                provenance="natural_language:clock",
            )

        if action_text:
            return ActionSelection(
                skill="autonomous_task_engine",
                params={"goal": action_text},
                provenance="semantic_plan:live_capability_catalog",
                execution_mode="planned_goal",
            )

        return ActionSelection(
            provenance="unstructured",
            reason="missing_action_objective",
        )

    @staticmethod
    def _non_action_evidence_reason(
        initiative: dict[str, Any],
        goal: dict[str, Any],
        action_text: str,
    ) -> str:
        """Keep recalled evidence and passive observations out of the action lane."""
        metadata = dict(initiative.get("metadata", {}) or {})
        source = " ".join(
            str(value or "").strip().casefold()
            for value in (
                initiative.get("source"),
                goal.get("source"),
                metadata.get("source"),
            )
            if value
        )
        raw = " ".join(
            str(value or "").casefold()
            for value in (
                initiative.get("goal"),
                initiative.get("objective"),
                goal.get("objective"),
            )
        )
        if (
            "retained memory evidence" in raw
            or "source=durable_memory" in raw
            or "durable_memory" in source
            or "conversation_memory" in source
        ):
            return "retained_memory_is_evidence_not_an_action"
        if not action_text.strip():
            return "passive_observation_has_no_action_objective"
        return ""

    @staticmethod
    def _action_text(initiative: dict[str, Any], goal: dict[str, Any]) -> str:
        values = list(
            dict.fromkeys(
                str(value or "").strip()
                for value in (
                    initiative.get("goal") or initiative.get("objective"),
                    goal.get("objective"),
                    goal.get("success_criteria"),
                )
                if value and str(value).strip()
            )
        )
        raw = " ".join(values)
        without_evidence = _RETAINED_EVIDENCE_RE.sub("", raw)
        return _short_text(without_evidence, 1200)

    @staticmethod
    def _action_params(
        initiative: dict[str, Any],
        goal: dict[str, Any],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        for candidate in (
            metadata.get("params"),
            goal.get("params"),
            initiative.get("params"),
        ):
            if isinstance(candidate, dict):
                return dict(candidate)
        return {}

    @staticmethod
    def _explicit_action_names(
        initiative: dict[str, Any],
        goal: dict[str, Any],
        metadata: dict[str, Any],
    ) -> list[str]:
        values: list[str] = []
        for owner in (metadata, goal, initiative):
            for key in ("required_skills", "required_tools"):
                raw = owner.get(key)
                if isinstance(raw, str):
                    values.append(raw)
                elif isinstance(raw, (list, tuple, set)):
                    values.extend(str(item) for item in raw if str(item).strip())
            for key in ("skill", "skill_name", "tool", "tool_name"):
                raw = owner.get(key)
                if isinstance(raw, str) and raw.strip():
                    values.append(raw)
        return list(dict.fromkeys(value.strip() for value in values if value.strip()))

    @staticmethod
    def _prepare_explicit_action(
        skill: str,
        params: dict[str, Any],
        *,
        action_text: str,
        registry: Any,
    ) -> tuple[dict[str, Any] | None, str]:
        prepared = dict(params)
        if skill == "code_execution":
            if not isinstance(prepared.get("code"), str) or not prepared["code"].strip():
                return None, "missing_code"
        elif skill == "web_search":
            prepared.setdefault("query", action_text)
            if not isinstance(prepared.get("query"), str) or not prepared["query"].strip():
                return None, "missing_query"
        elif skill == "web_fetch":
            if not isinstance(prepared.get("url"), str) or not prepared["url"].strip():
                return None, "missing_url"
        elif skill == "git_operation":
            prepared.setdefault("action", "status")
            if prepared.get("action") in {"branch", "commit", "checkout"} and not prepared.get(
                "allow_mutation"
            ):
                return None, "mutation_not_authorized"
            if prepared.get("action") == "clone" and not prepared.get("allow_external_clone"):
                return None, "external_clone_not_authorized"
        elif skill == "package_install":
            if not isinstance(prepared.get("package_name"), str) or not prepared[
                "package_name"
            ].strip():
                return None, "missing_package_name"
            if not prepared.get("allow_install"):
                return None, "install_not_authorized"
        elif skill == "process_supervisor":
            prepared.setdefault("action", "list")
            if prepared.get("action") == "spawn" and (
                not prepared.get("command") or not prepared.get("allow_spawn")
            ):
                return None, "spawn_not_authorized"
        elif skill == "document_ingest":
            if not isinstance(prepared.get("path"), str) or not prepared["path"].strip():
                return None, "missing_path"
        elif skill == "file_operation":
            if not isinstance(prepared.get("action"), str) or not prepared["action"].strip():
                return None, "missing_file_action"
            if not isinstance(prepared.get("path"), str) or not prepared["path"].strip():
                return None, "missing_path"
        elif skill == "auto_refactor":
            prepared = {"path": ".", "run_tests": False, **prepared}
        elif skill == "system_proprioception":
            prepared = {"include_docstrings": False, **prepared}
        elif skill == "environment_info":
            prepared = {"detail": "basic", **prepared}
        elif skill not in SAFE_AUTONOMOUS_SKILLS:
            return None, "unsupported_skill"

        actuator = registry.get_actuator(skill)
        if actuator is not None and not actuator.validate_params(prepared):
            return None, "actuator_validation_failed"
        return prepared, ""

    @staticmethod
    def _normalize_skill_name(value: str) -> str:
        text = str(value or "").strip().lower().replace("-", "_")
        aliases = {
            "shell": "auto_refactor",
            "pytest": "auto_refactor",
            "proof_bundle": "file_operation",
            "filesystem": "file_operation",
            "camera": "file_operation",
            "microphone": "file_operation",
            "screen": "file_operation",
            "coding": "auto_refactor",
        }
        return aliases.get(text, text)

    @staticmethod
    def _verify(skill: str, params: dict[str, Any], raw: Any) -> bool:
        if not isinstance(raw, dict) or not bool(raw.get("ok", False)):
            return False
        
        # Check if skill is registered in actuator registry
        from core.actuators.actuator_registry import get_actuator_registry
        if get_actuator_registry().get_actuator(skill) is not None:
            return bool(raw.get("ok", False) or raw.get("success", False))

        if skill == "file_operation" and params.get("action") == "exists":
            return "exists" in raw
        if skill == "auto_refactor":
            return "issues_found" in raw and "top_issues" in raw
        if skill == "system_proprioception":
            return bool(raw.get("system_map") is not None or raw.get("summary"))
        if skill == "environment_info":
            return bool(raw.get("result") or raw.get("summary"))
        if skill == "clock":
            return bool(raw.get("time") or raw.get("readable"))
        return True

    @staticmethod
    def _summarize_result(raw: Any) -> str:
        if isinstance(raw, dict):
            for key in ("summary", "message", "error", "status"):
                value = raw.get(key)
                if value:
                    return _short_text(value, 400)
            return _short_text(raw, 400)
        return _short_text(raw, 400)

    async def _finish(self, result: OvertActionResult, *, raw_result: Any) -> OvertActionResult:
        result.finished_at = time.time()
        result.duration_ms = round((result.finished_at - result.started_at) * 1000.0, 3)
        self._last_finished_at = result.finished_at

        if result.verified:
            self._actions_verified += 1
            self._consecutive_failures = 0
        elif result.status == "deferred":
            pass
        else:
            self._consecutive_failures += 1
            self._annotate_failure_learning(result, raw_result)

        # The receipt store fsyncs each receipt; LIVE 2026-09-19 the loop
        # report named it from this coroutine. The ids it writes back on the
        # result are read by the traces below, so it is awaited, off the loop.
        await asyncio.to_thread(self._emit_receipts, result, raw_result)
        self._record_life_trace(result, raw_result)
        self._update_goal(result)
        self._emit_visible_trace(result)
        return result

    @staticmethod
    def _failure_status(raw_result: Any) -> str:
        if isinstance(raw_result, dict):
            return str(raw_result.get("status") or raw_result.get("reason") or raw_result.get("error") or "").lower()
        return str(raw_result or "").lower()

    def _annotate_failure_learning(self, result: OvertActionResult, raw_result: Any) -> None:
        """Convert failed autonomous action into actionable self-learning.

        This does not retry blindly. It records the safer retry shape so the
        next initiative cycle can choose scan/propose/defer instead of repeating
        the same blocked act.
        """

        status = self._failure_status(raw_result)
        if "blocked_by_user_advocate" in status or "user advocate" in status:
            result.next_step_hint = (
                "retry_with_explicit_user_benefit_and_non_mutating_scope"
            )
        elif "deferred" in status or "background_policy" in status:
            result.next_step_hint = "retry_after_idle_and_resource_window"
        elif "self_preservation" in status or "memory" in status:
            result.next_step_hint = "retry_lower_cost_or_after_memory_pressure_drops"
        elif "verification" in status or "execution_not_completed" in status:
            result.next_step_hint = "retry_with_effect_verification_and_narrower_scope"
        elif not result.next_step_hint:
            result.next_step_hint = "classify_failure_before_retry"

        try:
            self_model = ServiceContainer.get("self_model", default=None)
            if self_model is None:
                orchestrator = self._orchestrator()
                self_model = getattr(orchestrator, "self_model", None) if orchestrator is not None else None
            update = getattr(self_model, "record_runtime_lesson", None)
            if callable(update):
                update(
                    source="overt_action_loop",
                    lesson=(
                        f"Autonomous {result.skill} failed with {status or 'unknown'}; "
                        f"next step: {result.next_step_hint}."
                    ),
                    confidence=0.74,
                    evidence={"action_id": result.action_id, "raw_result": raw_result},
                )
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("overt_action_loop", exc)

    def _emit_receipts(self, result: OvertActionResult, raw_result: Any) -> None:
        try:
            store = self.receipt_store
            if store is None:
                from core.runtime.receipts import get_receipt_store

                store = get_receipt_store()
                self.receipt_store = store
            from core.runtime.receipts import AutonomyReceipt, ToolExecutionReceipt

            tool_receipt = store.emit(
                ToolExecutionReceipt(
                    cause=result.objective,
                    tool=result.skill,
                    governance_receipt_id=result.will_receipt_id or None,
                    status=(
                        "success_verified"
                        if result.verified
                        else "deferred"
                        if result.status == "deferred"
                        else str(
                            result.expectation_verdict.get("status")
                            or "failed_unverified"
                        )
                    ),
                    output_digest=_json_digest(raw_result),
                    verification_evidence={
                        "verified": result.verified,
                        "duration_ms": result.duration_ms,
                        "summary": result.result_summary,
                        "action_expectation": dict(result.action_expectation),
                        "expectation_verdict": dict(result.expectation_verdict),
                        "upstream_expectation_receipt_id": result.expectation_receipt_id,
                    },
                    metadata={
                        "action_id": result.action_id,
                        "source": "overt_action_loop",
                        "execution_mode": result.execution_mode,
                        "selection_provenance": result.selection_provenance,
                        "expectation_next_step": result.next_step_hint,
                        "expectation_passed": bool(
                            result.expectation_verdict.get("passed", result.verified)
                        ),
                    },
                )
            )
            autonomy_receipt = store.emit(
                AutonomyReceipt(
                    cause=result.objective,
                    autonomy_level=3,
                    proposed_action=f"{result.skill}:{result.objective[:160]}",
                    governance_receipt_id=result.will_receipt_id or None,
                    budget_remaining=max(0.0, 1.0 - min(1.0, self._consecutive_failures / 5.0)),
                    metadata={
                        "action_id": result.action_id,
                        "tool_receipt_id": tool_receipt.receipt_id,
                        "execution_mode": result.execution_mode,
                        "selection_provenance": result.selection_provenance,
                        "expectation_receipt_id": result.expectation_receipt_id,
                        "expectation_passed": bool(
                            result.expectation_verdict.get("passed", result.verified)
                        ),
                    },
                )
            )
            result.tool_receipt_id = tool_receipt.receipt_id
            result.autonomy_receipt_id = autonomy_receipt.receipt_id
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation("overt_action_loop", exc)

    def _emit_selection_skip_receipt(self, result: OvertActionResult) -> None:
        try:
            store = self.receipt_store
            if store is None:
                from core.runtime.receipts import get_receipt_store

                store = get_receipt_store()
                self.receipt_store = store
            from core.runtime.receipts import AutonomyReceipt

            receipt = store.emit(
                AutonomyReceipt(
                    cause=result.objective,
                    autonomy_level=3,
                    proposed_action="not_executed:initiative_not_actionable",
                    governance_receipt_id=result.will_receipt_id or None,
                    budget_remaining=max(
                        0.0,
                        1.0 - min(1.0, self._consecutive_failures / 5.0),
                    ),
                    metadata={
                        "action_id": result.action_id,
                        "status": result.status,
                        "selection_provenance": result.selection_provenance,
                        "selection_reason": result.selection_reason,
                        "next_step_hint": result.next_step_hint,
                    },
                )
            )
            result.autonomy_receipt_id = receipt.receipt_id
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation("overt_action_loop", exc)

    @staticmethod
    def _record_selection_skip_trace(result: OvertActionResult) -> None:
        try:
            from core.runtime.life_trace import get_life_trace

            event = get_life_trace().record(
                "action_skipped",
                origin="overt_action_loop",
                user_requested=False,
                will_decision={"receipt_id": result.will_receipt_id},
                action_taken={
                    "action_id": result.action_id,
                    "executed": False,
                    "objective": result.objective,
                },
                result={
                    "status": result.status,
                    "reason": result.selection_reason,
                    "selection_provenance": result.selection_provenance,
                    "autonomy_receipt_id": result.autonomy_receipt_id,
                },
                future_policy_change={
                    "next_step_hint": result.next_step_hint,
                },
            )
            result.life_trace_id = event.event_id
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation("overt_action_loop", exc)

    def _record_life_trace(self, result: OvertActionResult, raw_result: Any) -> None:
        try:
            from core.runtime.life_trace import get_life_trace

            deferred = result.status == "deferred"
            event = get_life_trace().record(
                "action_deferred" if deferred else "action_executed",
                origin="overt_action_loop",
                user_requested=False,
                will_decision={"receipt_id": result.will_receipt_id},
                action_taken={
                    "action_id": result.action_id,
                    "skill": result.skill,
                    "params": result.params,
                    "objective": result.objective,
                    "executed": not deferred,
                },
                result={
                    "verified": result.verified,
                    "status": result.status,
                    "summary": result.result_summary,
                    "error": result.error,
                    "tool_receipt_id": result.tool_receipt_id,
                    "autonomy_receipt_id": result.autonomy_receipt_id,
                    "action_expectation": result.action_expectation,
                    "expectation_verdict": result.expectation_verdict,
                    "expectation_receipt_id": result.expectation_receipt_id,
                },
                memory_update={"goal_id": result.goal_id} if result.goal_id else {},
                future_policy_change={
                    "next_action_after_s": self.interval_s,
                    "next_step_hint": result.next_step_hint,
                },
            )
            result.life_trace_id = event.event_id
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation("overt_action_loop", exc)

    def _update_goal(self, result: OvertActionResult) -> None:
        if not result.goal_id or not result.verified:
            return
        goal_engine = self._goal_engine()
        if goal_engine is None or not hasattr(goal_engine, "get_goal") or not hasattr(goal_engine, "update_goal_status"):
            return

        async def _update() -> None:
            try:
                current = await asyncio.to_thread(goal_engine.get_goal, result.goal_id) or {}
                progress = min(0.95, max(float(current.get("progress", 0.0) or 0.0), 0.05) + 0.05)
                evidence = list(current.get("evidence") or [])
                evidence.append(result.tool_receipt_id or result.action_id)
                await goal_engine.update_goal_status(
                    result.goal_id,
                    status="in_progress",
                    progress=progress,
                    summary=result.result_summary,
                    evidence=evidence[-8:],
                    metadata={"last_overt_action_id": result.action_id, "last_overt_action_at": result.finished_at},
                )
            except (OSError, ConnectionError, TimeoutError) as exc:
                record_degradation("overt_action_loop", exc)

        try:
            get_task_tracker().create_task(
                _update(),
                name="overt_action_loop.goal_update",
            )
        except RuntimeError:
            asyncio.run(_update())

    def _emit_visible_trace(self, result: OvertActionResult) -> None:
        try:
            from core.thought_stream import get_emitter

            title = (
                "Overt Action Verified"
                if result.verified
                else "Overt Action Deferred"
                if result.status == "deferred"
                else "Overt Action Failed"
            )
            content = (
                f"{result.skill} -> {result.result_summary or result.error} "
                f"(receipt {result.tool_receipt_id or 'pending'})"
            )
            get_emitter().emit(
                title,
                content,
                level=(
                    "info"
                    if result.verified or result.status == "deferred"
                    else "warning"
                ),
                category="OvertAction",
            )
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation("overt_action_loop", exc)
        if not result.verified and result.status != "deferred":
            record_degraded_event(
                "overt_action_loop",
                "action_failed",
                detail=f"{result.skill}:{result.error}",
                severity="warning",
                classification="background_degraded",
                context={"action_id": result.action_id},
            )


_instance: OvertActionLoop | None = None


def get_overt_action_loop() -> OvertActionLoop:
    global _instance
    if _instance is None:
        _instance = OvertActionLoop()
    return _instance


__all__ = ["OvertActionLoop", "OvertActionResult", "get_overt_action_loop"]
