"""core/agency/agency_orchestrator.py

Canonical life-loop for every autonomous action. This is the single legal
path from drive to outcome:

    perceive -> update state -> generate drive -> propose initiative ->
    score -> simulate -> authorize -> execute -> observe outcome ->
    assess regret/lesson -> update memory/self-model

No autonomous action may execute without producing a complete ``ActionReceipt``
that records every stage. The static analyzer in ``tools/lint_governance.py``
fails CI if any code outside ``core/agency/agency_orchestrator.py`` directly
calls a consequential primitive (memory write, state mutation, tool execution,
external communication, code modification, social posting, file write,
shell execution, model fine-tuning, self-modification).

The orchestrator does NOT replace the existing UnifiedWill / AuthorityGateway
chain — it consumes them. Will is the policy engine; AgencyOrchestrator is
the runtime that drives the policy engine through the full life-loop and
produces forensic receipts for every decision.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import math
import subprocess
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

from core.runtime.errors import FallbackClassification, Severity, record_degradation
from core.runtime.state_ownership import state_root
from core.runtime.subprocess_gateway import get_subprocess_gateway

logger = logging.getLogger("Aura.AgencyOrchestrator")

AGENCY_RECOVERABLE_ERRORS = (
    ImportError,
    AttributeError,
    RuntimeError,
    TypeError,
    ValueError,
    OSError,
    subprocess.SubprocessError,
)


def _record_agency_degradation(
    error: BaseException,
    *,
    action: str,
    severity: Severity = "warning",
    extra: dict[str, Any] | None = None,
) -> None:
    record_degradation(
        "agency_orchestrator",
        error,
        severity=severity,
        action=action,
        classification=FallbackClassification.SAFE_FALLBACK,
        receipt_required=False,
        extra=extra,
    )


# ---------------------------------------------------------------------------
# Receipt dataclass — every autonomous action gets one of these end-to-end.
# ---------------------------------------------------------------------------


@dataclass
class ActionReceipt:
    """Complete forensic record of one autonomous action.

    Every field is required for the receipt to count as "complete" — partial
    receipts mark an aborted life-loop. ``record()`` writes this to the
    durable receipt log.
    """

    proposal_id: str
    drive: str
    state_snapshot: dict[str, Any]
    expected_outcome: str
    simulation_result: dict[str, Any]
    will_decision: str
    will_receipt_id: str | None
    authority_receipt: str | None
    capability_token: str | None
    execution_receipt: str | None
    outcome_assessment: dict[str, Any]
    started_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    blocked_at: str | None = None
    blocked_reason: str | None = None
    lesson: str | None = None
    regret: float | None = None  # 0.0 = no regret, 1.0 = high regret

    def is_complete(self) -> bool:
        return all(
            x is not None
            for x in (
                self.execution_receipt,
                self.outcome_assessment,
                self.completed_at,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return _receipt_json_safe(self)


def _receipt_json_safe(value: Any, *, _depth: int = 0) -> Any:
    """Serialize receipts without deepcopying coroutine or runtime objects."""
    if _depth > 8:
        return repr(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if inspect.iscoroutine(value):
        # A coroutine in a receipt is itself a receipt-chain bug. Close it so
        # serialization does not leak RuntimeWarning noise, then make the
        # failure visible in the stored evidence.
        try:
            value.close()
        except (RuntimeError, AttributeError, TypeError, ValueError) as _exc:
            logger.debug(
                "Suppressed %s in core.agency.agency_orchestrator: %s", type(_exc).__name__, _exc
            )
        return {
            "error": "coroutine_in_receipt",
            "repr": repr(value),
        }
    if inspect.isawaitable(value):
        return {
            "error": "awaitable_in_receipt",
            "repr": repr(value),
        }
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _receipt_json_safe(getattr(value, item.name), _depth=_depth + 1)
            for item in fields(value)
        }
    if isinstance(value, dict):
        return {
            str(_receipt_json_safe(key, _depth=_depth + 1)): _receipt_json_safe(
                item, _depth=_depth + 1
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_receipt_json_safe(item, _depth=_depth + 1) for item in value]
    try:
        json.dumps(value)
        return value
    except (json.JSONDecodeError, TypeError, ValueError):
        return repr(value)


# ---------------------------------------------------------------------------
# Durable receipt log
# ---------------------------------------------------------------------------


class _ReceiptLog:
    """JSONL receipt persistence.

    Keeps a 30-day rolling window in
    ``~/.aura/data/agency_receipts/agency_receipts.jsonl`` plus an in-memory
    deque so the dashboard can serve recent receipts without disk I/O.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or (
            state_root() / "data" / "agency_receipts" / "agency_receipts.jsonl"
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        from collections import deque

        self._recent: deque = deque(maxlen=512)
        self._lock = asyncio.Lock()

    async def append(self, receipt: ActionReceipt) -> None:
        async with self._lock:
            self._recent.append(receipt)
            try:
                await asyncio.to_thread(self._append_sync, receipt)
            except (json.JSONDecodeError, TypeError, ValueError, OSError) as exc:
                _record_agency_degradation(
                    exc,
                    action="Kept agency receipt in memory after durable receipt append failed",
                    severity="degraded",
                    extra={"path": str(self.path), "proposal_id": receipt.proposal_id},
                )
                logger.warning("Receipt log append failed: %s", exc)

    def _append_sync(self, receipt: ActionReceipt) -> None:
        from core.governance_context import local_internal_governed_scope
        from core.runtime.file_write_gateway import get_file_write_gateway

        with local_internal_governed_scope("agency_orchestrator.receipt_log", domain="file_write"):
            get_file_write_gateway().append_text(
                self.path,
                json.dumps(receipt.to_dict(), default=str) + "\n",
                encoding="utf-8",
                source="agency_orchestrator.receipt_log",
            )

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        return [r.to_dict() for r in list(self._recent)[-limit:]]


_RECEIPT_LOG = _ReceiptLog()


def get_receipt_log() -> _ReceiptLog:
    return _RECEIPT_LOG


# ---------------------------------------------------------------------------
# Primitive kinds — every consequential primitive is registered here.
# tools/lint_governance.py reads this list to enforce zero direct calls
# outside ``core/agency/agency_orchestrator.py``.
# ---------------------------------------------------------------------------


CONSEQUENTIAL_PRIMITIVES = (
    "memory_write",
    "state_mutation",
    "tool_execution",
    "external_communication",
    "code_modification",
    "persistent_belief_update",
    "initiative_release",
    "social_posting",
    "file_write",
    "shell_execution",
    "model_fine_tuning",
    "self_modification",
)


# ---------------------------------------------------------------------------
# Orchestrator — the canonical life-loop.
# ---------------------------------------------------------------------------


@dataclass
class Proposal:
    drive: str
    intent: str
    expected_outcome: str
    primitive: str
    payload: dict[str, Any] = field(default_factory=dict)
    priority: float = 0.5


class AgencyOrchestrator:
    """Single legal path from drive to outcome."""

    def __init__(self) -> None:
        self._receipt_log = _RECEIPT_LOG

    # --- top-level loop -----------------------------------------------------

    async def run(
        self,
        proposal: Proposal,
        *,
        perceive: Callable[[], Awaitable[dict[str, Any]]] | None = None,
        score: Callable[[Proposal, dict[str, Any]], Awaitable[float]] | None = None,
        simulate: Callable[[Proposal, dict[str, Any]], Awaitable[dict[str, Any]]] | None = None,
        execute: Callable[[Proposal, dict[str, Any], str], Awaitable[dict[str, Any]]] | None = None,
        assess: Callable[[Proposal, dict[str, Any], dict[str, Any]], Awaitable[dict[str, Any]]]
        | None = None,
    ) -> ActionReceipt:
        """Run the full life-loop for one proposal.

        Each callable is optional; the orchestrator supplies defaults that
        record the appropriate stage even when a subsystem is unavailable
        so receipts are never partial just because a hook is missing.
        """

        proposal_id = f"AO-{uuid.uuid4().hex[:12]}"
        receipt = ActionReceipt(
            proposal_id=proposal_id,
            drive=proposal.drive,
            state_snapshot={},
            expected_outcome=proposal.expected_outcome,
            simulation_result={},
            will_decision="",
            will_receipt_id=None,
            authority_receipt=None,
            capability_token=None,
            execution_receipt=None,
            outcome_assessment={},
        )

        # 1. perceive / 2. update state
        try:
            perceived = perceive() if perceive else self._default_perceive()
            state_snapshot = await perceived if inspect.isawaitable(perceived) else perceived
        except AGENCY_RECOVERABLE_ERRORS as exc:
            _record_agency_degradation(
                exc,
                action="Blocked agency life-loop at perception stage",
                severity="degraded",
                extra={"proposal_id": proposal_id, "stage": "perceive"},
            )
            return await self._block(receipt, "perceive", str(exc))
        if not isinstance(state_snapshot, dict):
            return await self._block(
                receipt,
                "perceive",
                f"perception returned {type(state_snapshot).__qualname__}, expected dict",
            )
        receipt.state_snapshot = state_snapshot

        # 3. proposal already given; 4. score
        try:
            score_value = await score(proposal, state_snapshot) if score else proposal.priority
            if not isinstance(score_value, (int, float)) or not math.isfinite(float(score_value)):
                raise ValueError(f"invalid score: {score_value!r}")
            score_value = max(0.0, min(1.0, float(score_value)))
        except AGENCY_RECOVERABLE_ERRORS as exc:
            _record_agency_degradation(
                exc,
                action="Blocked agency life-loop at scoring stage",
                severity="degraded",
                extra={"proposal_id": proposal_id, "stage": "score"},
            )
            return await self._block(receipt, "score", str(exc))

        # 5. simulate (counterfactual; must NOT mutate live state)
        try:
            simulation = (
                await simulate(proposal, state_snapshot) if simulate else {"score": score_value}
            )
            if not isinstance(simulation, dict):
                raise ValueError(
                    f"simulation returned {type(simulation).__qualname__}, expected dict"
                )
        except AGENCY_RECOVERABLE_ERRORS as exc:
            _record_agency_degradation(
                exc,
                action="Blocked agency life-loop at simulation stage",
                severity="degraded",
                extra={"proposal_id": proposal_id, "stage": "simulate"},
            )
            return await self._block(receipt, "simulate", str(exc))
        receipt.simulation_result = simulation

        # 6. authorize via UnifiedWill + AuthorityGateway
        will_outcome = await self._authorize(proposal, state_snapshot, simulation)
        receipt.will_decision = will_outcome["decision"]
        receipt.will_receipt_id = will_outcome.get("will_receipt_id")
        receipt.authority_receipt = will_outcome.get("authority_receipt")
        receipt.capability_token = will_outcome.get("capability_token")
        if will_outcome["decision"] != "approved":
            return await self._block(receipt, "authorize", will_outcome.get("reason", ""))

        # 7. execute
        try:
            exec_result = (
                await execute(proposal, state_snapshot, receipt.capability_token or "")
                if execute
                else await self._default_execute(proposal)
            )
        except AGENCY_RECOVERABLE_ERRORS as exc:
            _record_agency_degradation(
                exc,
                action="Blocked agency life-loop at execution stage",
                severity="degraded",
                extra={"proposal_id": proposal_id, "stage": "execute"},
            )
            return await self._block(receipt, "execute", str(exc))
        if not isinstance(exec_result, dict):
            return await self._block(
                receipt,
                "execute",
                f"execute returned {type(exec_result).__qualname__}, expected dict",
            )
        receipt.execution_receipt = str(exec_result.get("receipt") or exec_result)

        # 8. observe outcome / 9. assess regret / lesson
        try:
            outcome = (
                await assess(proposal, state_snapshot, exec_result)
                if assess
                else {"observed": exec_result}
            )
            if not isinstance(outcome, dict):
                raise ValueError(f"assess returned {type(outcome).__qualname__}, expected dict")
        except AGENCY_RECOVERABLE_ERRORS as exc:
            _record_agency_degradation(
                exc,
                action="Blocked agency life-loop at assessment stage",
                severity="degraded",
                extra={"proposal_id": proposal_id, "stage": "assess"},
            )
            return await self._block(receipt, "assess", str(exc))
        receipt.outcome_assessment = outcome
        receipt.regret = float(outcome.get("regret", 0.0) or 0.0)
        receipt.lesson = outcome.get("lesson")
        receipt.completed_at = time.time()

        # 10. update memory / self-model — handled by assess hook contract
        await self._receipt_log.append(receipt)
        return receipt

    # --- helpers ------------------------------------------------------------

    async def _default_perceive(self) -> dict[str, Any]:
        try:
            from core.container import ServiceContainer

            registry = ServiceContainer.get("unified_state_registry", default=None)
            if registry and hasattr(registry, "snapshot"):
                snap = registry.snapshot()
                return snap if isinstance(snap, dict) else {"raw": str(snap)[:1024]}
        except (ImportError, AttributeError, RuntimeError) as exc:
            _record_agency_degradation(
                exc,
                action="Returned empty perception snapshot after state registry snapshot failed",
                extra={"stage": "default_perceive"},
            )
            logger.debug("default perceive snapshot failed: %s", exc)
        return {}

    async def _default_execute(self, proposal: Proposal) -> dict[str, Any]:
        if proposal.primitive == "shell_execution":
            argv = proposal.payload.get("argv")
            if not isinstance(argv, list) or not all(isinstance(part, str) for part in argv):
                return {
                    "executed": False,
                    "receipt": "shell_execution:invalid_argv",
                    "error": "argv must be list[str]",
                }
            timeout = float(proposal.payload.get("timeout") or 30.0)
            cwd = proposal.payload.get("cwd")
            proc = await get_subprocess_gateway().run_async(
                argv,
                cwd=cwd if isinstance(cwd, str) else None,
                timeout=timeout,
                capture_output=True,
                check=False,
                source="core.agency.agency_orchestrator.default_shell_execution",
                accelerator_capability="auto",
            )
            return {
                "executed": True,
                "receipt": f"shell_execution:{proc.returncode}",
                "returncode": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            }
        return {"executed": False, "receipt": f"{proposal.primitive}:no_default_executor"}

    # Authority provenance a proposal is allowed to carry into the Will.
    #
    # LIVE DEFECT, 2026-07-25. Bryan asked Aura to build him a checkers game
    # and a 2048 clone and got back a governance sentence:
    #
    #   WILL REFUSED: agency_orchestrator/tool_execution --
    #   denied_by_default: tool_execution requires validated scoped authority
    #   (signed_standing_authority_lease_missing)
    #
    # The Will was right to refuse. The chat route had already established
    # every fact needed to authorize the turn — origin "user",
    # foreground_request, user_explicitly_authorized, user_requested_action,
    # with untrusted authority keys stripped — and put them in
    # proposal.payload["context"]. This method then built a Will context out
    # of drive, primitive, expected_outcome, state and simulation, and threw
    # the provenance away one frame later. Every proposal reaching the Will
    # therefore looked like an unattributed autonomous drive, so an owner
    # sitting at the keyboard asking for a game was indistinguishable from
    # the runtime deciding to execute a tool by itself.
    #
    # Only these keys cross. Anything that would ASSERT authority rather
    # than describe its origin is refused below, because payload is
    # attacker-reachable in a way the chat route's own dict is not.
    _AUTHORITY_CONTEXT_KEYS = (
        "origin",
        "source",
        "authority_origin",
        "route",
        "foreground_request",
        "user_explicit_action_request",
        "user_explicitly_authorized",
        "user_requested_action",
        "requested_authority_scope",
        "effect_scope",
        "risk_level",
        "tool",
        "skill",
        "skill_name",
    )

    # Keys that grant authority instead of describing it. A proposal payload
    # may never carry these: a forged token in a payload would otherwise
    # walk straight past the lease check it is supposed to satisfy.
    _FORGEABLE_AUTHORITY_KEYS = (
        "authority_args_digest",
        "capability_token",
        "capability_token_id",
        "scoped_authority",
        "standing_authority_grant_id",
        "standing_authority_receipt_id",
        "standing_authority_token",
    )

    @classmethod
    def _proposal_authority_context(cls, proposal: Proposal) -> dict[str, Any]:
        """Provenance the proposal's originating route established."""
        payload_context = proposal.payload.get("context")
        if not isinstance(payload_context, Mapping):
            return {}
        carried = {
            key: payload_context[key]
            for key in cls._AUTHORITY_CONTEXT_KEYS
            if key in payload_context
        }
        for forgeable in cls._FORGEABLE_AUTHORITY_KEYS:
            carried.pop(forgeable, None)
        return carried

    async def _mint_tool_authority(
        self,
        proposal: Proposal,
        authority_context: dict[str, Any],
    ) -> dict[str, Any]:
        """Ask standing authority for a lease before asking the Will.

        This is the same lease the tool-execution mixin issues; the agency
        loop simply never asked for one, which is why its context could
        never satisfy the Will's tool_execution gate.

        An autonomous drive carries no user-facing origin, so it lands on
        exactly the policy it lands on today — this restores the ability to
        tell the two apart, it does not lower the bar for either.
        """
        try:
            from core.executive.standing_authority import (
                context_has_user_authority,
                get_standing_authority_manager,
            )
        except (ImportError, AttributeError) as exc:
            _record_agency_degradation(
                exc,
                action="proceeded to the Will without a standing-authority lease",
                severity="warning",
                extra={"primitive": proposal.primitive},
            )
            return {}

        tool_name = (
            authority_context.get("skill_name")
            or authority_context.get("tool")
            or authority_context.get("skill")
            or proposal.payload.get("skill_name")
            or proposal.primitive
        )
        origin = (
            authority_context.get("authority_origin")
            or authority_context.get("origin")
            or authority_context.get("source")
            or ""
        )
        arguments = proposal.payload.get("params")
        try:
            decision = await get_standing_authority_manager().issue_child_lease(
                tool_name=tool_name,
                arguments=arguments if isinstance(arguments, Mapping) else {},
                origin=origin,
                context=authority_context,
                user_authorized=context_has_user_authority(origin, authority_context),
                effect_scope=authority_context.get("effect_scope", ""),
                risk_level=authority_context.get("risk_level", ""),
            )
        except (RuntimeError, TypeError, ValueError, OSError) as exc:
            _record_agency_degradation(
                exc,
                action="proceeded to the Will without a standing-authority lease",
                severity="warning",
                extra={"primitive": proposal.primitive, "tool": str(tool_name)[:80]},
            )
            return {}

        if not decision.approved:
            # Not an error: the Will is about to refuse with this reason, and
            # saying it here makes the refusal legible in the receipt.
            return {"standing_authority_denial_reason": decision.reason}
        return dict(decision.context or {})

    async def _authorize(
        self,
        proposal: Proposal,
        state: dict[str, Any],
        simulation: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            from core.governance.will_client import WillClient, WillRequest
            from core.will import ActionDomain

            domain = self._primitive_to_domain(proposal.primitive)
            will_context: dict[str, Any] = {
                "drive": proposal.drive,
                "primitive": proposal.primitive,
                "expected_outcome": proposal.expected_outcome,
                "state": state,
                "simulation": simulation,
            }
            authority_context = self._proposal_authority_context(proposal)
            will_context.update(authority_context)
            if domain is getattr(ActionDomain, "TOOL_EXECUTION", None):
                will_context.update(
                    await self._mint_tool_authority(proposal, authority_context)
                )
            decision = await WillClient().decide_async(
                WillRequest(
                    content=proposal.intent,
                    source="agency_orchestrator",
                    domain=domain,
                    priority=proposal.priority,
                    context=will_context,
                )
            )
            approved = WillClient.is_approved(decision)
            return {
                "decision": "approved" if approved else "blocked",
                "reason": getattr(decision, "reason", ""),
                "will_receipt_id": getattr(decision, "receipt_id", None),
                "authority_receipt": getattr(decision, "authority_receipt", None),
                "capability_token": getattr(decision, "capability_token", None),
            }
        except (ImportError, AttributeError, RuntimeError) as exc:
            _record_agency_degradation(
                exc,
                action="Blocked agency life-loop because Will authorization was unavailable",
                severity="degraded",
                extra={"primitive": proposal.primitive, "drive": proposal.drive},
            )
            return {"decision": "blocked", "reason": f"authorize_exception:{exc}"}

    @staticmethod
    def _primitive_to_domain(primitive: str) -> Any:
        try:
            from core.will import ActionDomain
        except (ImportError, AttributeError, RuntimeError):
            return primitive
        mapping = {
            "memory_write": getattr(ActionDomain, "MEMORY_WRITE", primitive),
            "state_mutation": getattr(ActionDomain, "STATE_MUTATION", primitive),
            "tool_execution": getattr(ActionDomain, "TOOL_EXECUTION", primitive),
            "external_communication": getattr(ActionDomain, "EXPRESSION", primitive),
            "code_modification": getattr(ActionDomain, "STATE_MUTATION", primitive),
            "persistent_belief_update": getattr(ActionDomain, "STATE_MUTATION", primitive),
            "initiative_release": getattr(ActionDomain, "INITIATIVE", primitive),
            "social_posting": getattr(ActionDomain, "EXPRESSION", primitive),
            "file_write": getattr(ActionDomain, "STATE_MUTATION", primitive),
            "shell_execution": getattr(ActionDomain, "TOOL_EXECUTION", primitive),
            "model_fine_tuning": getattr(ActionDomain, "STATE_MUTATION", primitive),
            "self_modification": getattr(ActionDomain, "STATE_MUTATION", primitive),
        }
        return mapping.get(primitive, primitive)

    async def _block(self, receipt: ActionReceipt, stage: str, reason: str) -> ActionReceipt:
        receipt.blocked_at = stage
        receipt.blocked_reason = reason
        receipt.completed_at = time.time()
        await self._receipt_log.append(receipt)
        return receipt


_ORCHESTRATOR: AgencyOrchestrator | None = None


def get_orchestrator() -> AgencyOrchestrator:
    global _ORCHESTRATOR
    if _ORCHESTRATOR is None:
        _ORCHESTRATOR = AgencyOrchestrator()
    return _ORCHESTRATOR
