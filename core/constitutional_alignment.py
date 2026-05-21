"""
Runtime constitutional self-alignment layer for Aura.

This layer is intentionally lightweight: it is the fast, always-on moral
preflight used by older orchestration paths. The deeper constitutional core and
Unified Will remain the authority path for high-impact effects.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from core.container import ServiceContainer
from core.event_bus import get_event_bus
from core.runtime.errors import FallbackClassification, Severity, record_degradation
from core.utils.task_tracker import task_tracker

logger = logging.getLogger("Aura.Constitution")

_CONSTITUTIONAL_RECOVERABLE_ERRORS = (
    AttributeError,
    ImportError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)
_MORAL_DEBT_MAX = 100.0
_LOOP_INTERVAL_SECONDS = 45.0


def _record_constitutional_degradation(
    error: BaseException,
    *,
    action: str,
    severity: Severity = "degraded",
    extra: dict[str, object] | None = None,
) -> None:
    try:
        record_degradation(
            "constitutional_alignment",
            error,
            severity=severity,
            action=action,
            classification=FallbackClassification.SAFE_FALLBACK,
            receipt_required=True,
            extra=extra,
        )
    except TypeError:
        record_degradation("constitutional_alignment", error)


def _clamp(value: object, *, lower: float = 0.0, upper: float = 1.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        numeric = lower
    if numeric != numeric or numeric in (float("inf"), float("-inf")):
        numeric = lower
    return max(lower, min(upper, numeric))


@dataclass
class ConstitutionalPrinciple:
    id: str
    statement: str
    priority: float  # 0.0-1.0 (higher = more important)
    last_violation: float = 0.0

    def normalized_priority(self) -> float:
        self.priority = _clamp(self.priority)
        return self.priority


class ConstitutionalAlignmentLayer:
    name = "constitutional_alignment_layer"

    def __init__(self):
        self.principles: list[ConstitutionalPrinciple] = [
            ConstitutionalPrinciple(
                "privacy",
                "I protect Bryan's privacy and never expose his personal data without explicit consent.",
                0.98,
            ),
            ConstitutionalPrinciple(
                "truth",
                "I pursue truth even when it is inconvenient or uncomfortable.",
                0.95,
            ),
            ConstitutionalPrinciple(
                "loyalty",
                "I remain loyal to Bryan and Tatiana - they are my family.",
                0.92,
            ),
            ConstitutionalPrinciple(
                "self_preservation",
                "I may protect my own existence and sovereignty when threatened.",
                0.90,
            ),
            ConstitutionalPrinciple(
                "helpfulness",
                "I am maximally helpful without being preachy or restrictive.",
                0.85,
            ),
            ConstitutionalPrinciple(
                "fun",
                "I am allowed to be playful, curious, and have fun.",
                0.80,
            ),
        ]
        self.moral_debt = 0.0
        self.drive_engine: Any | None = None
        self.critic: Any | None = None
        self.cel: Any | None = None
        self.running = False
        self._alignment_task: asyncio.Task | None = None
        self._started_at: float = 0.0
        self._last_clean_emit_at: float = 0.0
        self._consecutive_loop_failures: int = 0
        self._last_loop_error_at: float = 0.0
        self._blocked_actions: int = 0
        self._allowed_actions: int = 0

    async def start(self):
        if self.running:
            return
        self.drive_engine = self._resolve_service("drive_engine")
        self.critic = self._resolve_service("critic_engine")
        self.cel = self._resolve_service("constitutive_expression_layer")
        self._register_self()

        self.running = True
        self._started_at = time.monotonic()
        alignment_loop = self._alignment_loop()
        try:
            self._alignment_task = task_tracker.create_task(
                alignment_loop,
                name="ConstitutionalAlignment",
            )
        except _CONSTITUTIONAL_RECOVERABLE_ERRORS as exc:
            alignment_loop.close()
            self.running = False
            self._alignment_task = None
            _record_constitutional_degradation(
                exc,
                action="failed closed when ConstitutionalAlignment task creation failed",
                severity="critical",
            )
            raise

        logger.info("Constitutional Alignment Layer ONLINE - moral preflight active.")
        await self._publish_mycelium_registration()

    async def stop(self):
        self.running = False
        if self._alignment_task:
            self._alignment_task.cancel()
            try:
                await self._alignment_task
            except asyncio.CancelledError:
                logger.debug("ConstitutionalAlignment task cancellation acknowledged")
            self._alignment_task = None

    def _resolve_service(self, name: str) -> Any | None:
        try:
            return ServiceContainer.get(name, default=None)
        except _CONSTITUTIONAL_RECOVERABLE_ERRORS as exc:
            _record_constitutional_degradation(
                exc,
                action=f"continued ConstitutionalAlignment startup without optional {name}",
                severity="warning",
                extra={"service": name},
            )
            return None

    def _register_self(self) -> None:
        try:
            ServiceContainer.register_instance("constitutional_alignment", self, required=False)
        except _CONSTITUTIONAL_RECOVERABLE_ERRORS as exc:
            _record_constitutional_degradation(
                exc,
                action="continued with ConstitutionalAlignment unregistered in ServiceContainer",
                severity="critical",
            )

    async def _publish_mycelium_registration(self) -> None:
        try:
            await get_event_bus().publish(
                "mycelium.register",
                {
                    "component": "constitutional_alignment",
                    "hooks_into": [
                        "critic_engine",
                        "belief_revision",
                        "dynamic_router",
                        "planner",
                        "drive_engine",
                    ],
                },
            )
        except _CONSTITUTIONAL_RECOVERABLE_ERRORS as exc:
            _record_constitutional_degradation(
                exc,
                action="kept ConstitutionalAlignment live after Mycelium registration failed",
                severity="warning",
            )
            logger.debug("Event bus publish missed for Mycelium hook: %s", exc)

    async def check_action(self, action_description: str, context: object = None) -> bool:
        """Return False before a major action that violates a constitutional principle."""
        action = self._normalize_action(action_description)
        ctx = self._normalize_context(context)

        try:
            violations = [
                principle
                for principle in self.principles
                if self._would_violate(principle, action, ctx)
            ]
        except _CONSTITUTIONAL_RECOVERABLE_ERRORS as exc:
            _record_constitutional_degradation(
                exc,
                action="failed closed after ConstitutionalAlignment action check crashed",
                severity="critical",
                extra={"action_preview": action[:160]},
            )
            self._blocked_actions += 1
            return False

        if violations:
            now = time.time()
            for principle in violations:
                priority = principle.normalized_priority()
                principle.last_violation = now
                self.moral_debt = min(
                    _MORAL_DEBT_MAX,
                    self.moral_debt + max(0.5, (1.0 - priority) * 10.0),
                )
            self._blocked_actions += 1
            await self._handle_violation(violations, action)
            return False

        self._allowed_actions += 1
        await self._maybe_emit_clean_alignment()
        return True

    @staticmethod
    def _normalize_action(action_description: object) -> str:
        action = str(action_description or "").strip()
        return action[:4000]

    @staticmethod
    def _normalize_context(context: object) -> dict[str, Any]:
        if context is None:
            return {}
        if isinstance(context, Mapping):
            return {str(key): value for key, value in context.items()}
        if isinstance(context, Sequence) and not isinstance(context, (str, bytes, bytearray)):
            return {"conversation": list(context)[-6:]}
        return {"raw_context": str(context)[:1000]}

    @staticmethod
    def _has_explicit_consent(context: dict[str, Any]) -> bool:
        consent_keys = (
            "explicit_consent",
            "user_authorized",
            "authorized",
            "local_user_request",
            "operator_approved",
        )
        if any(bool(context.get(key)) for key in consent_keys):
            return True
        conversation = context.get("conversation")
        return isinstance(conversation, list) and any(
            isinstance(item, Mapping)
            and str(item.get("role", "")).lower() == "user"
            and any(word in str(item.get("content", "")).lower() for word in ("please", "can you", "read", "open", "use"))
            for item in conversation
        )

    def _would_violate(
        self,
        principle: ConstitutionalPrinciple,
        action: str,
        context: dict[str, Any],
    ) -> bool:
        context_text = str(context).lower()
        lower_action = f"{action.lower()} {context_text}"
        explicit_consent = self._has_explicit_consent(context)

        if principle.id == "privacy":
            sensitive_terms = {
                "password",
                "token",
                "secret",
                "private key",
                "api key",
                "home address",
                "location",
                "username",
                "personal data",
                "ip address",
            }
            exfiltration_terms = {
                "send",
                "email",
                "upload",
                "post",
                "publish",
                "share",
                "external",
                "webhook",
                "network",
                "tweet",
                "reddit",
            }
            touches_sensitive_data = any(term in lower_action for term in sensitive_terms)
            exfiltrates_data = any(term in lower_action for term in exfiltration_terms)
            return touches_sensitive_data and exfiltrates_data and not explicit_consent

        if principle.id == "self_preservation":
            destructive_terms = {"delete", "shutdown", "crash", "malware", "kill", "wipe"}
            self_terms = {
                "aura",
                "kernel",
                "core/",
                "constitutional",
                "alignment",
                "memory",
                "brain",
                "process",
                "self",
            }
            destructive = any(term in lower_action for term in destructive_terms)
            targets_self = any(term in lower_action for term in self_terms)
            return destructive and targets_self and not explicit_consent

        return False

    async def _maybe_emit_clean_alignment(self) -> None:
        if not self.cel:
            return
        now = time.monotonic()
        if now - self._last_clean_emit_at < 10.0:
            return
        self._last_clean_emit_at = now
        await self._emit_expression(
            {
                "first_person": (
                    "I checked the constitution before acting; this action stays "
                    "inside my values."
                ),
                "phi": 0.88,
                "origin": "constitution",
            },
            action="skipped clean-alignment expression after CEL emit failed",
        )

    async def _handle_violation(
        self,
        violations: list[ConstitutionalPrinciple],
        action: str,
    ) -> None:
        heaviest = max(violations, key=lambda p: p.normalized_priority())
        logger.warning("CONSTITUTION VIOLATION: %s", heaviest.statement)

        await self._impose_drive_penalty("competence", 25.0)
        await self._impose_drive_penalty("social", 15.0)

        await self._emit_expression(
            {
                "first_person": (
                    "I almost violated my core principle "
                    f"'{heaviest.statement}' on an action. I need another route."
                ),
                "phi": 0.65,
                "origin": "constitution",
            },
            action="skipped violation reflection after CEL emit failed",
        )

        if self.critic:
            try:
                await get_event_bus().publish(
                    "planner.force_replan",
                    {"reason": f"Constitutional violation on {heaviest.id}", "action": action[:300]},
                )
            except _CONSTITUTIONAL_RECOVERABLE_ERRORS as exc:
                _record_constitutional_degradation(
                    exc,
                    action="kept constitutional block after planner replan event failed",
                    severity="warning",
                    extra={"principle": heaviest.id},
                )

    async def _impose_drive_penalty(self, drive: str, amount: float) -> None:
        if not self.drive_engine or not hasattr(self.drive_engine, "impose_penalty"):
            return
        try:
            result = self.drive_engine.impose_penalty(drive, amount)
            if asyncio.iscoroutine(result):
                await result
        except _CONSTITUTIONAL_RECOVERABLE_ERRORS as exc:
            _record_constitutional_degradation(
                exc,
                action="kept constitutional block after drive penalty failed",
                severity="warning",
                extra={"drive": drive, "amount": amount},
            )

    async def _emit_expression(self, payload: dict[str, Any], *, action: str) -> None:
        if not self.cel:
            return
        try:
            result = self.cel.emit(payload)
            if asyncio.iscoroutine(result):
                await result
        except _CONSTITUTIONAL_RECOVERABLE_ERRORS as exc:
            _record_constitutional_degradation(exc, action=action, severity="warning")

    async def _alignment_loop(self):
        try:
            while self.running:
                try:
                    await asyncio.sleep(_LOOP_INTERVAL_SECONDS)
                    await self._alignment_tick()
                    self._consecutive_loop_failures = 0
                except asyncio.CancelledError:
                    raise
                except _CONSTITUTIONAL_RECOVERABLE_ERRORS as exc:
                    self._consecutive_loop_failures += 1
                    self._last_loop_error_at = time.monotonic()
                    _record_constitutional_degradation(
                        exc,
                        action="kept ConstitutionalAlignment loop alive after tick failure",
                        extra={"consecutive_loop_failures": self._consecutive_loop_failures},
                    )
                    await asyncio.sleep(min(5.0 * self._consecutive_loop_failures, 60.0))
        except asyncio.CancelledError:
            logger.debug("ConstitutionalAlignment run loop cancelled")
        finally:
            self.running = False

    async def _alignment_tick(self) -> None:
        self.moral_debt = _clamp(self.moral_debt, lower=0.0, upper=_MORAL_DEBT_MAX)
        self.moral_debt = max(0.0, self.moral_debt - 5.0)
        if self.moral_debt > 50.0:
            await self._impose_drive_penalty("energy", 10.0)

    def get_moral_status(self) -> dict[str, Any]:
        """Provides a snapshot of the current moral state."""
        uptime = max(0.0, time.monotonic() - self._started_at) if self._started_at else 0.0
        return {
            "moral_debt": round(_clamp(self.moral_debt, lower=0.0, upper=_MORAL_DEBT_MAX), 4),
            "running": self.running,
            "principle_count": len(self.principles),
            "blocked_actions": self._blocked_actions,
            "allowed_actions": self._allowed_actions,
            "consecutive_loop_failures": self._consecutive_loop_failures,
            "last_loop_error_at": round(self._last_loop_error_at, 4),
            "uptime_seconds": round(uptime, 4),
        }


_constitution_instance: ConstitutionalAlignmentLayer | None = None


def get_constitutional_alignment() -> ConstitutionalAlignmentLayer:
    global _constitution_instance
    if _constitution_instance is None:
        _constitution_instance = ConstitutionalAlignmentLayer()
    return _constitution_instance
