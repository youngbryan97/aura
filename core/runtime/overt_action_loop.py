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
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Optional

from core.container import ServiceContainer
from core.health.degraded_events import record_degraded_event
from core.runtime.background_policy import background_activity_reason
from core.runtime.errors import record_degradation


SAFE_AUTONOMOUS_SKILLS = (
    "auto_refactor",
    "system_proprioception",
    "environment_info",
    "file_operation",
    "clock",
    "evolution_status",
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
        state_provider: Optional[Callable[[], Any]] = None,
        receipt_store: Any = None,
        interval_s: Optional[float] = None,
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
                if os.getenv("AURA_OVERT_ACTION_FALLBACK", "1").strip().lower() in {"0", "false", "off", "no"}:
                    return self._record_skip("no_authorized_initiative").to_dict()
                initiative = self._fallback_initiative()
                will_receipt_id = self._authorize_fallback(initiative)
                if not will_receipt_id:
                    return self._record_skip("fallback_not_authorized").to_dict()
            else:
                will_receipt_id = str(getattr(synth_result, "will_receipt_id", "") or "")

            action = await self._execute_initiative(initiative, will_receipt_id=will_receipt_id)
            self._history.append(action)
            return action.to_dict()

    def _background_reason(self) -> str:
        reason = background_activity_reason(
            self._orchestrator(),
            min_idle_seconds=float(os.getenv("AURA_OVERT_ACTION_IDLE_S", "30")),
            max_memory_percent=float(os.getenv("AURA_OVERT_ACTION_MAX_MEMORY_PERCENT", "88")),
            max_failure_pressure=float(os.getenv("AURA_OVERT_ACTION_MAX_FAILURE_PRESSURE", "0.35")),
            require_conversation_ready=False,
        )
        if reason == "no_user_anchor" and os.getenv("AURA_OVERT_ACTION_ALLOW_BOOT_ANCHOR", "1").strip().lower() not in {"0", "false", "off", "no"}:
            return ""
        return reason

    def _record_skip(self, reason: str) -> OvertActionResult:
        self._skips += 1
        now = time.time()
        result = OvertActionResult(
            action_id="skip_" + hashlib.sha256(f"{now}:{reason}".encode("utf-8")).hexdigest()[:10],
            status="skipped",
            error=reason,
            started_at=now,
            finished_at=now,
            next_step_hint="wait_for_idle_window" if reason.startswith("background_policy") else "",
        )
        self._history.append(result)
        return result

    def _orchestrator(self) -> Any:
        if self.orchestrator is not None:
            return self.orchestrator
        self.orchestrator = ServiceContainer.get("orchestrator", default=None)
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
        except Exception as exc:
            record_degradation("overt_action_loop", exc)
            return ""

    async def _execute_initiative(self, initiative: dict[str, Any], *, will_receipt_id: str) -> OvertActionResult:
        started = time.time()
        self._last_started_at = started
        self._actions_started += 1
        objective = _short_text(initiative.get("goal") or initiative.get("objective"), 1000)
        goal = self._goal_for_initiative(initiative)
        skill, params = self._choose_skill_and_params(initiative, goal)
        action_id = hashlib.sha256(f"{started}:{objective}:{skill}".encode("utf-8")).hexdigest()[:16]
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
        )

        engine = self._capability_engine()
        if engine is None or not hasattr(engine, "execute"):
            result.status = "failed"
            result.error = "capability_engine_unavailable"
            return self._finish(result, raw_result={})

        try:
            raw = await engine.execute(
                skill,
                params,
                context={
                    "origin": "overt_action_loop",
                    "source": "overt_action_loop",
                    "objective": objective,
                    "will_receipt_id": will_receipt_id,
                    "autonomous": True,
                    "initiative": initiative,
                },
            )
        except Exception as exc:
            record_degradation("overt_action_loop", exc)
            result.status = "failed"
            result.error = f"{type(exc).__name__}: {exc}"
            raw = {"ok": False, "error": result.error}

        result.verified = self._verify(skill, params, raw)
        result.status = "verified" if result.verified else "failed"
        result.result_summary = self._summarize_result(raw)
        if not result.verified and not result.error:
            result.error = str(raw.get("error") or raw.get("status") or "verification_failed") if isinstance(raw, dict) else "verification_failed"
        return self._finish(result, raw_result=raw)

    def _goal_for_initiative(self, initiative: dict[str, Any]) -> dict[str, Any]:
        metadata = dict(initiative.get("metadata", {}) or {})
        goal_id = str(metadata.get("goal_id") or initiative.get("goal_id") or "")
        goal_engine = self._goal_engine()
        if goal_id and goal_engine is not None and hasattr(goal_engine, "get_goal"):
            try:
                return dict(goal_engine.get_goal(goal_id) or {})
            except Exception as exc:
                record_degradation("overt_action_loop", exc)
        return {}

    def _choose_skill_and_params(self, initiative: dict[str, Any], goal: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        text = " ".join(
            [
                str(initiative.get("goal") or ""),
                str(goal.get("objective") or ""),
                str(goal.get("success_criteria") or ""),
            ]
        ).lower()
        metadata = dict(initiative.get("metadata", {}) or {})
        required = [
            str(item)
            for item in (
                list(metadata.get("required_skills") or [])
                + list(goal.get("required_skills") or [])
                + list(goal.get("required_tools") or [])
            )
        ]
        normalized_required = {self._normalize_skill_name(item) for item in required}

        if "auto_refactor" in normalized_required or any(token in text for token in ("repair", "refactor", "architecture", "codebase", "bug")):
            return "auto_refactor", {"path": ".", "run_tests": False}
        if "proof" in text or "bundle" in text or "canonical" in text:
            return "file_operation", {"action": "exists", "path": "artifacts/proof_bundle/latest/CANONICAL_PROOF_BUNDLE.json"}
        if "sensor" in text or "camera" in text or "microphone" in text or "screen" in text:
            return "file_operation", {"action": "exists", "path": "sensory_vision.json"}
        if "evolution_status" in normalized_required or "evolution" in text:
            return "evolution_status", {}
        if "environment_info" in normalized_required or "environment" in text:
            return "environment_info", {"detail": "basic"}
        if "clock" in normalized_required or "time" in text:
            return "clock", {}
        return "system_proprioception", {"include_docstrings": False}

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

    def _finish(self, result: OvertActionResult, *, raw_result: Any) -> OvertActionResult:
        result.finished_at = time.time()
        result.duration_ms = round((result.finished_at - result.started_at) * 1000.0, 3)
        self._last_finished_at = result.finished_at

        if result.verified:
            self._actions_verified += 1
            self._consecutive_failures = 0
        else:
            self._consecutive_failures += 1

        self._emit_receipts(result, raw_result)
        self._record_life_trace(result, raw_result)
        self._update_goal(result)
        self._emit_visible_trace(result)
        return result

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
                    status="success_verified" if result.verified else "failed_unverified",
                    output_digest=_json_digest(raw_result),
                    verification_evidence={
                        "verified": result.verified,
                        "duration_ms": result.duration_ms,
                        "summary": result.result_summary,
                    },
                    metadata={"action_id": result.action_id, "source": "overt_action_loop"},
                )
            )
            autonomy_receipt = store.emit(
                AutonomyReceipt(
                    cause=result.objective,
                    autonomy_level=3,
                    proposed_action=f"{result.skill}:{result.objective[:160]}",
                    governance_receipt_id=result.will_receipt_id or None,
                    budget_remaining=max(0.0, 1.0 - min(1.0, self._consecutive_failures / 5.0)),
                    metadata={"action_id": result.action_id, "tool_receipt_id": tool_receipt.receipt_id},
                )
            )
            result.tool_receipt_id = tool_receipt.receipt_id
            result.autonomy_receipt_id = autonomy_receipt.receipt_id
        except Exception as exc:
            record_degradation("overt_action_loop", exc)

    def _record_life_trace(self, result: OvertActionResult, raw_result: Any) -> None:
        try:
            from core.runtime.life_trace import get_life_trace

            event = get_life_trace().record(
                "action_executed",
                origin="overt_action_loop",
                user_requested=False,
                will_decision={"receipt_id": result.will_receipt_id},
                action_taken={
                    "action_id": result.action_id,
                    "skill": result.skill,
                    "params": result.params,
                    "objective": result.objective,
                },
                result={
                    "verified": result.verified,
                    "status": result.status,
                    "summary": result.result_summary,
                    "error": result.error,
                    "tool_receipt_id": result.tool_receipt_id,
                    "autonomy_receipt_id": result.autonomy_receipt_id,
                },
                memory_update={"goal_id": result.goal_id} if result.goal_id else {},
                future_policy_change={"next_action_after_s": self.interval_s},
            )
            result.life_trace_id = event.event_id
        except Exception as exc:
            record_degradation("overt_action_loop", exc)

    def _update_goal(self, result: OvertActionResult) -> None:
        if not result.goal_id or not result.verified:
            return
        goal_engine = self._goal_engine()
        if goal_engine is None or not hasattr(goal_engine, "get_goal") or not hasattr(goal_engine, "update_goal_status"):
            return

        async def _update() -> None:
            try:
                current = goal_engine.get_goal(result.goal_id) or {}
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
            except Exception as exc:
                record_degradation("overt_action_loop", exc)

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_update(), name="overt_action_loop.goal_update")
        except RuntimeError:
            asyncio.run(_update())

    def _emit_visible_trace(self, result: OvertActionResult) -> None:
        try:
            from core.thought_stream import get_emitter

            title = "Overt Action Verified" if result.verified else "Overt Action Failed"
            content = (
                f"{result.skill} -> {result.result_summary or result.error} "
                f"(receipt {result.tool_receipt_id or 'pending'})"
            )
            get_emitter().emit(title, content, level="info" if result.verified else "warning", category="OvertAction")
        except Exception as exc:
            record_degradation("overt_action_loop", exc)
        if not result.verified:
            record_degraded_event(
                "overt_action_loop",
                "action_failed",
                detail=f"{result.skill}:{result.error}",
                severity="warning",
                classification="background_degraded",
                context={"action_id": result.action_id},
            )


_instance: Optional[OvertActionLoop] = None


def get_overt_action_loop() -> OvertActionLoop:
    global _instance
    if _instance is None:
        _instance = OvertActionLoop()
    return _instance


__all__ = ["OvertActionLoop", "OvertActionResult", "get_overt_action_loop"]
