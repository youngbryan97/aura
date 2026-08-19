"""Continuous pre-action cortex loop — deliberation wrapped AROUND action.

The architecture gap this closes: Aura's overt action loop already runs
choose → execute → verify → remember, but each stage thought alone. This
organ keeps ONE cognitive thread alive across the consequential-action
cycle:

    propose → REHEARSE (latent episode: predicted effect, preconditions,
    risks) → execute → observe → RECONCILE (discrepancy-driven replanning
    episode seeded with the rehearsal's own conclusion + the observed
    failure evidence)

Continuity is carried honestly: each phase's conclusion becomes a typed
cognitive-context item (source="action_thread") seeding identifiable
workspace slots in the NEXT phase's episode — same cognitive content,
receipted and individually ablatable — and reconciliation conclusions
return to the Global Workspace through the existing GWT coupling, where
replanning competes for broadcast like every other coalition. We do not
pretend to hold worker KV across generations; the thread is the workspace
content plus receipts, which is the part that must survive anyway.

Discrepancy is OBJECTIVE: reconciliation fires only on transport failure
or a failed effect verification — never on fuzzy text similarity. Matching
predictions are recorded, not re-deliberated.

Everything is defensive and bounded: no latent service, a busy generation
gate, or a kill switch (AURA_PREACTION_RLC=0) ⇒ the action proceeds
exactly as before, with a receipt saying deliberation was skipped and why.
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Any

from core.runtime.flags import FlagKind, declare

logger = logging.getLogger("Aura.PreActionCortex")

PREACTION_SCHEMA = "aura.preaction_cortex.v1"

# Domains whose side effects are consequential enough to buy deliberation.
CONSEQUENTIAL_DOMAINS = frozenset(
    {
        "external_action",
        "network_call",
        "cloud_call",
        "ci_cd",
        "self_modification",
        "environment_action",
    }
)

_MAX_THREAD_ITEMS = 3
_MAX_ITEM_CHARS = 400
_REHEARSAL_TIMEOUT_S = 45.0
_RECONCILE_TIMEOUT_S = 60.0


_PREACTION_RLC_FLAG = declare(
    "AURA_PREACTION_RLC",
    kind=FlagKind.BOOL,
    default=True,
    description="Pre-action cortex loop (RLC rehearsal before consequential actions)",
    owner="core.brain.preaction_cortex",
)


def _enabled() -> bool:
    return bool(_PREACTION_RLC_FLAG.value())


def _latent_service() -> Any:
    try:
        from core.runtime.service_registry import get_runtime_service
        from core.service_names import ServiceNames

        return get_runtime_service(ServiceNames.LATENT_CORTEX, default=None)
    except (ImportError, AttributeError):
        return None


def preaction_runtime_available() -> bool:
    """Return whether a live cortex can participate in this action."""

    return _enabled() and _latent_service() is not None


def build_rehearsal_objective(
    *,
    action_summary: str,
    expectation_objective: str,
) -> str:
    """Build the exact worker-visible objective bound into an action offer."""

    return (
        "Before I take this action, think it through.\n"
        f"Action: {str(action_summary)[:400]}\n"
        f"Intended outcome: {str(expectation_objective)[:400]}\n"
        "State: (1) the precise observable effect I predict, "
        "(2) the preconditions that must already hold, "
        "(3) the most likely failure mode and what it would look like. "
        "Return only one JSON object with exactly these fields: "
        '{"action_ready": boolean, "preconditions_met": boolean, '
        '"risk_acceptable": boolean, "expected_effect": string, '
        '"reason": string}. action_ready may be true only when every required '
        "precondition is met and the identified risk is acceptable."
    )


def _availability_failure(reason: str) -> str | None:
    normalized = str(reason or "").strip()
    if normalized in {
        "disabled:AURA_LATENT_CORTEX=0",
        "generation_gate_busy",
        "no_resident_model",
        # The rehearsal could not FIT, which is a scheduling fact about the
        # rehearsal and not a verdict about the action.
        #
        # `latent_cortex_service` refuses with this when
        # 65s + 0.26s/token exceeds the turn's remaining budget, and its own
        # comment says what it means by it: "No model owner has been acquired
        # yet, so ResponseGeneration can use the same resident checkpoint's
        # ordinary lane with the full answer surface immediately." It is a
        # hand-off, not a refusal of the work.
        #
        # Classified as integrity, it became one. An `episode_integrity_*`
        # reason may never bypass — correctly, since that is the case the
        # allowlist exists to stop from masquerading as a decision — so the
        # external-execution coordinator raised and the ACTION was refused
        # before dispatch. Live 2026-08-18 that is what stood between a
        # user-requested browser task and the browser, after every authority
        # gate ahead of it had been cleared.
        "answer_surface_unaffordable_before_execution",
    }:
        return normalized
    if normalized in {
        "client_unavailable:DependencyUnavailable",
        "client_unavailable:ImportError",
        "client_unavailable:ModelUnavailable",
        "client_unavailable:OSError",
        "client_unavailable:TimeoutError",
        "generation_lease_unavailable:DependencyUnavailable",
        "generation_lease_unavailable:ImportError",
        "generation_lease_unavailable:OSError",
        "generation_lease_unavailable:TimeoutError",
    }:
        return normalized
    if normalized in {
        "client_error:OSError",
        "client_error:TimeoutError",
    }:
        return normalized
    return None


class PreActionCortexThread:
    """One cognitive thread across a single consequential action."""

    def __init__(
        self,
        *,
        domain: str,
        action_name: str,
        request_digest: str = "",
        external_execution_offer: dict[str, Any] | None = None,
    ) -> None:
        self.domain = str(domain)
        self.action_name = str(action_name)[:120]
        self.request_digest = str(request_digest)[:80]
        if external_execution_offer is None:
            self.external_execution_offer = None
        else:
            from core.brain.llm.latent_cortex.external_execution import (
                validate_external_execution_offer,
            )

            self.external_execution_offer = validate_external_execution_offer(
                external_execution_offer
            )
        self.created_at = time.time()
        self.thread_items: list[dict[str, str]] = []
        self.rehearsal: dict[str, Any] = {}
        self.reconciliation: dict[str, Any] = {}
        self._external_execution_trace: list[dict[str, Any]] = []
        self._external_execution_readiness: dict[str, Any] = {}
        self._external_execution_model_output = ""
        self._external_action_policy_evidence: dict[str, Any] = {}
        self._external_action_policy_receipt: dict[str, Any] = {}
        self._external_action_executors: list[str] = []
        self._external_runtime_operation: dict[str, Any] = {}

    # ── Thread continuity ────────────────────────────────────────────────
    def _remember(self, phase: str, conclusion: str) -> None:
        text = f"[{phase}] {conclusion}".strip()[:_MAX_ITEM_CHARS]
        if not text:
            return
        self.thread_items.append({"source": "action_thread", "text": text})
        del self.thread_items[:-_MAX_THREAD_ITEMS]

    def _context(self) -> list[dict[str, str]] | None:
        return [dict(item) for item in self.thread_items] or None

    # ── Phase 1: rehearsal ───────────────────────────────────────────────
    async def rehearse(
        self,
        *,
        action_summary: str,
        expectation_objective: str,
        stakes: float = 0.7,
    ) -> dict[str, Any]:
        """Deliberate the proposed action BEFORE it runs.

        The episode's objective asks for exactly the trio the loop needs:
        predicted observable effect, preconditions that must already hold,
        and the failure modes worth watching. The conclusion seeds the
        reconciliation phase's slots.
        """
        receipt: dict[str, Any] = {
            "schema": PREACTION_SCHEMA,
            "phase": "rehearsal",
            "action_name": self.action_name,
            "domain": self.domain,
            "ran": False,
        }
        if not _enabled():
            receipt["skip_reason"] = "disabled:AURA_PREACTION_RLC=0"
            self.rehearsal = receipt
            return receipt
        service = _latent_service()
        if service is None:
            receipt["skip_reason"] = "latent_cortex_absent"
            self.rehearsal = receipt
            return receipt
        if self.external_execution_offer is not None:
            objective = build_rehearsal_objective(
                action_summary=action_summary,
                expectation_objective=expectation_objective,
            )
        else:
            objective = (
                "Before I take this action, think it through.\n"
                f"Action: {str(action_summary)[:400]}\n"
                f"Intended outcome: {str(expectation_objective)[:400]}\n"
                "State: (1) the precise observable effect I predict, "
                "(2) the preconditions that must already hold, "
                "(3) the most likely failure mode and what it would look like."
            )
        if (
            self.external_execution_offer is not None
            and self.external_execution_offer["objective_sha256"]
            != hashlib.sha256(objective.encode("utf-8")).hexdigest()
        ):
            receipt["skip_reason"] = "external_execution_objective_mismatch"
            self.rehearsal = receipt
            return receipt
        try:
            episode_kwargs = {
                "stakes": stakes,
                "uncertainty": 0.6,
                "domain": "action_rehearsal",
                "timeout_s": _REHEARSAL_TIMEOUT_S,
                "require_full_stack": False,
                "foreground_request": True,
                "cognitive_context": self._context(),
            }
            if self.external_execution_offer is not None:
                episode_kwargs["external_execution_offer"] = dict(
                    self.external_execution_offer
                )
            result = await service.deep_reason(objective, **episode_kwargs)
        except (
            AttributeError,
            RuntimeError,
            TypeError,
            ValueError,
            OSError,
            TimeoutError,
        ) as exc:
            if isinstance(exc, (OSError, TimeoutError)):
                receipt["skip_reason"] = (
                    f"availability_failure:{type(exc).__name__}"
                )
            else:
                receipt["skip_reason"] = (
                    f"episode_integrity_failure:{type(exc).__name__}"
                )
            self.rehearsal = receipt
            return receipt
        if not isinstance(result, dict) or not result.get("ok"):
            reason = ""
            if isinstance(result, dict):
                reason = str(result.get("reason") or "")
            availability = _availability_failure(reason)
            receipt["skip_reason"] = (
                f"availability_failure:{availability}"
                if availability is not None
                else f"episode_integrity_refusal:{reason or 'unknown'}"
            )
            self.rehearsal = receipt
            return receipt
        prediction = str(result.get("text") or "").strip()
        episode_receipt = result.get("receipt") or {}
        execution_handoff: dict[str, Any] = {}
        execution_readiness: dict[str, Any] = {}
        if self.external_execution_offer is not None:
            try:
                from core.brain.llm.latent_cortex.epistemic_runtime import (
                    validate_completed_runtime_operation_receipt,
                )
                from core.brain.llm.latent_cortex.epistemic_state import (
                    OperationKind,
                )
                from core.brain.llm.latent_cortex.external_execution import (
                    build_external_execution_readiness,
                    validate_external_execution_handoff,
                )
                from core.brain.llm.latent_cortex.value_of_computation import (
                    validate_action_trace,
                    validate_evidence_snapshot,
                )

                raw_trace = episode_receipt.get("cognitive_action_trace")
                policy_receipt = episode_receipt.get("value_of_computation")
                raw_evidence = episode_receipt.get("host_action_policy_evidence")
                if (
                    not isinstance(raw_trace, list)
                    or not isinstance(policy_receipt, dict)
                    or not isinstance(policy_receipt.get("executors"), list)
                ):
                    raise ValueError(
                        "external execution lacks host policy evidence"
                    )
                evidence = validate_evidence_snapshot(raw_evidence)
                policy_fields = {
                    "schema",
                    "bucket",
                    "snapshot_sha256",
                    "active",
                    "executors",
                    "actions_selected",
                    "checked_transitions",
                    "selected_actions",
                }
                if set(policy_receipt) != policy_fields:
                    raise ValueError(
                        "external execution policy summary fields differ"
                    )
                executors = tuple(
                    OperationKind(item)
                    for item in policy_receipt["executors"]
                )
                if not executors or len(set(executors)) != len(executors):
                    raise ValueError(
                        "external execution executor inventory is invalid"
                    )
                validated_trace = validate_action_trace(
                    raw_trace,
                    evidence_snapshot=evidence,
                    executors=executors,
                )
                selected_actions = validated_trace["selected_actions"]
                checked_transitions = sum(
                    int(row["transition"]["checked"])
                    for row in validated_trace["rows"]
                )
                if (
                    policy_receipt.get("schema") != evidence["schema"]
                    or policy_receipt.get("bucket") != evidence["bucket"]
                    or policy_receipt.get("snapshot_sha256")
                    != evidence["snapshot_sha256"]
                    or policy_receipt.get("active") is not True
                    or policy_receipt.get("actions_selected") != len(raw_trace)
                    or policy_receipt.get("selected_actions") != selected_actions
                    or policy_receipt.get("checked_transitions")
                    != checked_transitions
                ):
                    raise ValueError(
                        "external execution policy summary differs from trace"
                    )
                runtime_operation = validate_completed_runtime_operation_receipt(
                    episode_receipt.get("epistemic_operation"),
                    external_execution_offer=self.external_execution_offer,
                    action_policy_evidence=evidence,
                    action_policy_receipt=policy_receipt,
                    cognitive_action_trace=raw_trace,
                )
                execution_handoff = validate_external_execution_handoff(
                    episode_receipt.get("external_execution_handoff"),
                    offer=self.external_execution_offer,
                    cognitive_action_trace=raw_trace,
                )
                self._external_execution_trace = [
                    dict(row) for row in raw_trace
                    if isinstance(row, dict)
                ]
                execution_readiness = build_external_execution_readiness(
                    self.external_execution_offer,
                    prediction,
                )
                if execution_handoff["requested"] and not (
                    execution_readiness["action_ready"]
                    and execution_readiness["preconditions_met"]
                    and execution_readiness["risk_acceptable"]
                ):
                    raise ValueError(
                        "external execution requested without action readiness"
                    )
                self._external_execution_readiness = dict(execution_readiness)
                self._external_execution_model_output = prediction
                self._external_action_policy_evidence = dict(evidence)
                self._external_action_policy_receipt = dict(policy_receipt)
                self._external_action_executors = [
                    item.value for item in executors
                ]
                self._external_runtime_operation = dict(runtime_operation)
            except (ImportError, TypeError, ValueError) as exc:
                receipt["skip_reason"] = (
                    f"external_execution_handoff_invalid:{type(exc).__name__}"
                )
                self.rehearsal = receipt
                return receipt
        receipt.update(
            {
                "ran": True,
                "prediction": prediction[:800],
                "episode_id": str(episode_receipt.get("episode_id") or ""),
                "steps_taken": episode_receipt.get("steps_taken"),
                "honest_flags": list(episode_receipt.get("honest_flags") or []),
                "external_execution_handoff": execution_handoff,
                "external_execution_readiness": execution_readiness,
                "execution_requested": (
                    execution_handoff.get("requested") is True
                    if execution_handoff
                    else None
                ),
            }
        )
        self._remember("rehearsal", prediction)
        self.rehearsal = receipt
        return receipt

    def external_execution_trace(self) -> list[dict[str, Any]]:
        """Return an isolated trace copy for host-side handoff validation."""

        return [dict(row) for row in self._external_execution_trace]

    def external_execution_readiness(self) -> dict[str, Any]:
        return dict(self._external_execution_readiness)

    def external_execution_model_output(self) -> str:
        return self._external_execution_model_output

    def external_action_policy_evidence(self) -> dict[str, Any]:
        return dict(self._external_action_policy_evidence)

    def external_action_executors(self) -> list[str]:
        return list(self._external_action_executors)

    def external_action_policy_receipt(self) -> dict[str, Any]:
        return dict(self._external_action_policy_receipt)

    def external_runtime_operation(self) -> dict[str, Any]:
        return dict(self._external_runtime_operation)

    # ── Phase 2: reconciliation ──────────────────────────────────────────
    async def reconcile(self, action_result: dict[str, Any]) -> dict[str, Any]:
        """Compare prediction with reality; replan only on OBJECTIVE failure.

        Transport failure or unverified effect after transport success are
        the discrepancy triggers. The replanning episode is seeded with the
        rehearsal's own conclusion plus the observed evidence — the same
        cognitive thread, revised by reality — and its conclusion competes
        for Global Workspace broadcast via the standard RLC → GWT coupling.
        """
        transport = bool(action_result.get("transport_succeeded"))
        verified = action_result.get("effect_verified") is True
        receipt: dict[str, Any] = {
            "schema": PREACTION_SCHEMA,
            "phase": "reconciliation",
            "action_name": self.action_name,
            "domain": self.domain,
            "ran": False,
            "discrepancy": (not transport) or (transport and not verified),
            "transport_succeeded": transport,
            "effect_verified": verified,
        }
        if not receipt["discrepancy"]:
            receipt["skip_reason"] = "prediction_confirmed"
            self.reconciliation = receipt
            return receipt
        if not _enabled():
            receipt["skip_reason"] = "disabled:AURA_PREACTION_RLC=0"
            self.reconciliation = receipt
            return receipt
        service = _latent_service()
        if service is None:
            receipt["skip_reason"] = "latent_cortex_absent"
            self.reconciliation = receipt
            return receipt
        error = str(action_result.get("error") or "")[:300]
        status = str(action_result.get("status") or "")[:80]
        evidence = (
            f"status={status or 'unknown'}"
            + (f"; error={error}" if error else "")
            + (
                "; transport succeeded but the effect was never verified"
                if transport and not verified
                else ""
            )
        )
        self._remember("observed", evidence)
        objective = (
            f"My action did not go as predicted.\n"
            f"Action: {self.action_name}\n"
            f"Observed: {evidence}\n"
            "Diagnose the first divergence between my prediction and what "
            "happened, then state the revised plan: retry as-is, retry "
            "changed (say what changes), or abandon (say why)."
        )
        try:
            result = await service.deep_reason(
                objective,
                stakes=0.8,
                uncertainty=0.7,
                domain="action_reconciliation",
                timeout_s=_RECONCILE_TIMEOUT_S,
                require_full_stack=False,
                foreground_request=True,
                cognitive_context=self._context(),
            )
        except (
            AttributeError,
            RuntimeError,
            TypeError,
            ValueError,
            OSError,
            TimeoutError,
        ) as exc:
            receipt["skip_reason"] = f"episode_failed:{type(exc).__name__}"
            self.reconciliation = receipt
            return receipt
        if not isinstance(result, dict) or not result.get("ok"):
            reason = ""
            if isinstance(result, dict):
                reason = str(result.get("reason") or "")
            receipt["skip_reason"] = f"episode_refused:{reason or 'unknown'}"
            self.reconciliation = receipt
            return receipt
        replan = str(result.get("text") or "").strip()
        episode_receipt = result.get("receipt") or {}
        broadcast = episode_receipt.get("workspace_broadcast") or {}
        receipt.update(
            {
                "ran": True,
                "replan": replan[:800],
                "episode_id": str(episode_receipt.get("episode_id") or ""),
                "workspace_broadcast_submitted": bool(
                    broadcast.get("submitted")
                ),
                "honest_flags": list(episode_receipt.get("honest_flags") or []),
            }
        )
        self._remember("replan", replan)
        self.reconciliation = receipt
        return receipt

    def to_receipt(self) -> dict[str, Any]:
        return {
            "schema": PREACTION_SCHEMA,
            "action_name": self.action_name,
            "domain": self.domain,
            "request_digest": self.request_digest,
            "rehearsal": dict(self.rehearsal),
            "reconciliation": dict(self.reconciliation),
            "thread_length": len(self.thread_items),
        }


def deliberation_worthy(domain: str) -> bool:
    """Only consequential side-effect domains buy latent deliberation."""
    return str(domain).lower() in CONSEQUENTIAL_DOMAINS


# ── The runtime's single seam into cognition ────────────────────────────
#
# core/runtime is the foundation layer: "everything above depends on this;
# it depends on nothing above" (core/runtime/DEPS). An import from a
# cognitive module into the foundation makes the foundation un-bootable
# without the very thing it is supposed to be able to run without.
#
# core/runtime/action_executor.py has one grandfathered exception to that —
# this module — because its external-execution branch is gated on
# deliberation_worthy() and is meaningless without the pre-action cortex.
# Two further edges were then added straight into
# core.brain.external_execute_coordinator and
# core.brain.llm.latent_cortex.external_execution, which the layering gate
# refuses and the baseline may not grow to accommodate.
#
# Re-exporting them here narrows the architecture rather than widening it:
# the foundation reaches cognition through ONE named seam instead of three.
# The bodies stay lazy so importing this module still costs nothing — the
# latent-cortex stack is not pulled in until an action actually deliberates.


def get_external_execute_coordinator() -> Any:
    """The external-execution coordinator, imported on first use."""
    from core.brain.external_execute_coordinator import (
        get_external_execute_coordinator as _impl,
    )

    return _impl()


def build_external_execution_offer(*args: Any, **kwargs: Any) -> Any:
    """Build an external-execution offer, importing the builder on use."""
    from core.brain.llm.latent_cortex.external_execution import (
        build_external_execution_offer as _impl,
    )

    return _impl(*args, **kwargs)


__all__ = [
    "CONSEQUENTIAL_DOMAINS",
    "PREACTION_SCHEMA",
    "PreActionCortexThread",
    "build_external_execution_offer",
    "build_rehearsal_objective",
    "deliberation_worthy",
    "get_external_execute_coordinator",
    "preaction_runtime_available",
]
