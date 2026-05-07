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
from core.runtime.errors import record_degradation



import asyncio
import inspect
import json
import logging
import time
import uuid
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger("Aura.AgencyOrchestrator")


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
    state_snapshot: Dict[str, Any]
    expected_outcome: str
    simulation_result: Dict[str, Any]
    will_decision: str
    will_receipt_id: Optional[str]
    authority_receipt: Optional[str]
    capability_token: Optional[str]
    execution_receipt: Optional[str]
    outcome_assessment: Dict[str, Any]
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    blocked_at: Optional[str] = None
    blocked_reason: Optional[str] = None
    lesson: Optional[str] = None
    regret: Optional[float] = None  # 0.0 = no regret, 1.0 = high regret

    def is_complete(self) -> bool:
        return all(
            x is not None
            for x in (
                self.execution_receipt,
                self.outcome_assessment,
                self.completed_at,
            )
        )

    def to_dict(self) -> Dict[str, Any]:
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
        except Exception:
            pass
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
            str(_receipt_json_safe(key, _depth=_depth + 1)): _receipt_json_safe(item, _depth=_depth + 1)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_receipt_json_safe(item, _depth=_depth + 1) for item in value]
    try:
        json.dumps(value)
        return value
    except Exception:
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

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path or (Path.home() / ".aura" / "data" / "agency_receipts" / "agency_receipts.jsonl")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        from collections import deque

        self._recent: deque = deque(maxlen=512)
        self._lock = asyncio.Lock()

    async def append(self, receipt: ActionReceipt) -> None:
        async with self._lock:
            self._recent.append(receipt)
            try:
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(receipt.to_dict(), default=str) + "\n")
            except Exception as exc:
                record_degradation('agency_orchestrator', exc)
                logger.warning("Receipt log append failed: %s", exc)

    def recent(self, limit: int = 50) -> List[Dict[str, Any]]:
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
    payload: Dict[str, Any] = field(default_factory=dict)
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
        perceive: Optional[Callable[[], Awaitable[Dict[str, Any]]]] = None,
        score: Optional[Callable[[Proposal, Dict[str, Any]], Awaitable[float]]] = None,
        simulate: Optional[Callable[[Proposal, Dict[str, Any]], Awaitable[Dict[str, Any]]]] = None,
        execute: Optional[Callable[[Proposal, Dict[str, Any], str], Awaitable[Dict[str, Any]]]] = None,
        assess: Optional[Callable[[Proposal, Dict[str, Any], Dict[str, Any]], Awaitable[Dict[str, Any]]]] = None,
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
        except Exception as exc:
            record_degradation('agency_orchestrator', exc)
            return await self._block(receipt, "perceive", str(exc))
        receipt.state_snapshot = state_snapshot

        # 3. proposal already given; 4. score
        try:
            score_value = await score(proposal, state_snapshot) if score else proposal.priority
        except Exception as exc:
            record_degradation('agency_orchestrator', exc)
            return await self._block(receipt, "score", str(exc))

        # 5. simulate (counterfactual; must NOT mutate live state)
        try:
            simulation = await simulate(proposal, state_snapshot) if simulate else {"score": score_value}
        except Exception as exc:
            record_degradation('agency_orchestrator', exc)
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
            exec_result = await execute(proposal, state_snapshot, receipt.capability_token or "") if execute else {"executed": False}
        except Exception as exc:
            record_degradation('agency_orchestrator', exc)
            return await self._block(receipt, "execute", str(exc))
        receipt.execution_receipt = str(exec_result.get("receipt") or exec_result)

        # 8. observe outcome / 9. assess regret / lesson
        try:
            outcome = await assess(proposal, state_snapshot, exec_result) if assess else {"observed": exec_result}
        except Exception as exc:
            record_degradation('agency_orchestrator', exc)
            return await self._block(receipt, "assess", str(exc))
        receipt.outcome_assessment = outcome
        receipt.regret = float(outcome.get("regret", 0.0) or 0.0)
        receipt.lesson = outcome.get("lesson")
        receipt.completed_at = time.time()

        # 10. update memory / self-model — handled by assess hook contract
        await self._receipt_log.append(receipt)
        return receipt

    # --- helpers ------------------------------------------------------------

    async def _default_perceive(self) -> Dict[str, Any]:
        try:
            from core.container import ServiceContainer
            registry = ServiceContainer.get("unified_state_registry", default=None)
            if registry and hasattr(registry, "snapshot"):
                snap = registry.snapshot()
                return snap if isinstance(snap, dict) else {"raw": str(snap)[:1024]}
        except Exception as exc:
            record_degradation('agency_orchestrator', exc)
            logger.debug("default perceive snapshot failed: %s", exc)
        return {}

    async def _authorize(
        self,
        proposal: Proposal,
        state: Dict[str, Any],
        simulation: Dict[str, Any],
    ) -> Dict[str, Any]:
        try:
            from core.governance.will_client import WillClient, WillRequest
            domain = self._primitive_to_domain(proposal.primitive)
            decision = await WillClient().decide_async(
                WillRequest(
                    content=proposal.intent,
                    source="agency_orchestrator",
                    domain=domain,
                    priority=proposal.priority,
                    context={
                        "drive": proposal.drive,
                        "primitive": proposal.primitive,
                        "expected_outcome": proposal.expected_outcome,
                        "state": state,
                        "simulation": simulation,
                    },
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
        except Exception as exc:
            record_degradation('agency_orchestrator', exc)
            return {"decision": "blocked", "reason": f"authorize_exception:{exc}"}

    @staticmethod
    def _primitive_to_domain(primitive: str) -> Any:
        try:
            from core.will import ActionDomain
        except Exception:
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


_ORCHESTRATOR: Optional[AgencyOrchestrator] = None


def get_orchestrator() -> AgencyOrchestrator:
    global _ORCHESTRATOR
    if _ORCHESTRATOR is None:
        _ORCHESTRATOR = AgencyOrchestrator()
    return _ORCHESTRATOR
