"""Canonical governed transaction boundary for consequential actions."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import inspect
import json
import logging
import math
import re
import subprocess
import time
import urllib.parse
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.being.body_state_service import BodyStateService
from core.being.welfare_state import WelfareState
from core.being.welfare_transaction import WelfareTransaction
from core.governance.will import ActionDomain
from core.governance_context import (
    GovernanceViolation,
    get_active_governance,
    governed_scope,
    require_governance,
)
from core.memory.memory_write_gateway import get_memory_write_gateway
from core.runtime.action_verification import (
    EffectVerifier,
    capture_pre_action_state,
    default_action_expectation,
    observe_action_effect,
)
from core.runtime.desktop_action_gateway import get_desktop_action_gateway
from core.runtime.errors import (
    FallbackClassification,
    _raise_site,
    record_degradation,
)
from core.runtime.file_write_gateway import get_file_write_gateway
from core.runtime.flags import FlagKind, declare
from core.runtime.network_gateway import get_network_gateway
from core.runtime.post_action_receipt import (
    PostActionReceipt,
    get_post_action_receipt_store,
)
from core.runtime.skill_contract import (
    ActionExpectation,
    SkillStatus,
    apply_action_expectation_payload,
    semantic_predicate_from_mapping,
)
from core.runtime.state_ownership import state_root
from core.runtime.subprocess_gateway import get_subprocess_gateway
from core.state.state_gateway import get_state_gateway
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Aura.ActionExecutor")

_VERIFIER_TIMEOUT_FLAG = declare(
    "AURA_ACTION_VERIFIER_TIMEOUT_S",
    kind=FlagKind.FLOAT,
    default=5.0,
    description="Maximum time allowed for post-action observed-effect verification",
    owner="core.runtime.action_executor",
)
_ACTION_EXECUTOR_RECOVERABLE_ERRORS = (
    AttributeError,
    GovernanceViolation,
    LookupError,
    OSError,
    PermissionError,
    RuntimeError,
    subprocess.SubprocessError,
    TimeoutError,
    TypeError,
    ValueError,
)
_SENSITIVE_PARAM_MARKERS = (
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "credential",
    "password",
    "private_key",
    "secret",
    "session_id",
    "token",
)
_ACTION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,160}$")
_PRIVATE_MAINTENANCE_ACTIONS = {
    (
        ActionDomain.FILE_WRITE.value,
        "host_automation.screenshot_directory",
        "host_automation.ensure_screenshot_directory",
        "ensure_directory",
    ),
    (
        ActionDomain.FILE_WRITE.value,
        "host_automation.screenshot_retention",
        "host_automation.screenshot_retention_delete",
        "delete",
    ),
    (
        ActionDomain.FILE_WRITE.value,
        "host_automation.ephemeral_ocr_cleanup",
        "host_automation.ephemeral_ocr_cleanup",
        "delete",
    ),
}
EffectHandler = Callable[
    [Mapping[str, Any]],
    Mapping[str, Any] | Awaitable[Mapping[str, Any]],
]
_HANDLER_DOMAINS = frozenset(
    {
        ActionDomain.ENVIRONMENT_ACTION,
        ActionDomain.EXTERNAL_ACTION,
        ActionDomain.NETWORK_CALL,
    }
)


def get_will() -> Any:
    """Resolve Will dynamically so tests and isolated runtimes can inject it."""

    from core.will import get_will as resolve_will

    return resolve_will()


def _ambient_authority_context(
    domain: ActionDomain,
    *,
    source: str = "unknown",
    action_name: str = "",
    params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Authority provenance from the governed scope this call already runs in.

    LIVE DEFECT, 2026-08-10. Aura's screen perception was dead and nobody
    could see why. Every ambient perception tick emitted:

        WILL REFUSED: host_automation.screenshot_directory/file_write --
        denied_by_default: file_write requires validated scoped authority

    ``take_screenshot`` routes its directory step through ``execute``, and
    ``execute`` built the Will context out of action_id, request_digest,
    expectation objective and a rollback flag — no authority provenance at
    all, and no parameter through which a caller could supply any. Under
    strict default-deny every FILE_WRITE, NETWORK_CALL and TOOL_EXECUTION
    reaching this class was therefore refused *as a matter of structure*:
    the gate was wired so that passing it was impossible.

    A governed scope is provenance, not automatically action authority. This
    exposes a strict-mode grant only for the two exact, runtime-owned private
    maintenance contracts below. Other matching-domain scopes remain visible
    for diagnostics but cannot turn a domain label into blanket permission.

    It is deliberately NOT a bypass:

    * The token comes from a contextvar installed by governance code, never
      from caller-supplied params, so it is not forgeable the way a payload
      key is (see ``_FORGEABLE_AUTHORITY_KEYS`` in agency_orchestrator).
    * ``get_active_governance`` checks the shared lexical lease registry, so a
      copied child-task ContextVar becomes inert when the owner exits.
    * The operation, source and resolved private path must all equal the
      constraints installed by the runtime-owned scope.
    """
    token = get_active_governance()
    if token is None:
        return {}
    token_domain = str(getattr(token, "domain", "") or "").strip().lower()
    if token_domain != domain.value:
        # An honest narrowing: say the scope was seen and why it did not
        # apply, so a refusal never again looks like "no governance at all".
        return {
            "ambient_governed_scope": token_domain or "unknown",
            "ambient_scope_domain_mismatch": domain.value,
        }
    context = {
        "authority_origin": str(getattr(token, "source", "") or source)[:240],
        "ambient_governed_scope": token_domain,
    }
    maintenance_attested = _private_maintenance_attested(
        token,
        domain=domain,
        source=source,
        action_name=action_name,
        params=params,
    )
    if maintenance_attested:
        context.update(
            {
                "scoped_authority": "exact_private_runtime_maintenance",
                "capability_token_id": str(
                    getattr(token, "receipt_id", "") or ""
                )[:120],
                "internal_runtime_maintenance": True,
                "effect_scope": "private_runtime_maintenance",
                "no_external_effects": True,
                "maintenance_operation": str((params or {}).get("op") or ""),
            }
        )
    return context


def _private_maintenance_attested(
    token: Any,
    *,
    domain: ActionDomain,
    source: str,
    action_name: str,
    params: Mapping[str, Any] | None,
) -> bool:
    """Attest one private runtime prerequisite without widening file authority."""

    constraints = dict(getattr(token, "constraints", ()) or ())
    requested = dict(params or {})
    operation = str(requested.get("op") or "").strip().lower()
    source_name = str(source or "").strip()
    declared_source = str(getattr(token, "source", "") or "").strip()
    contract = (domain.value, source_name, str(action_name or "").strip(), operation)
    if contract not in _PRIVATE_MAINTENANCE_ACTIONS:
        return False
    if declared_source != source_name:
        return False
    if constraints.get("governance_origin") != "local_internal":
        return False
    if constraints.get("runtime_generated") is not True:
        return False
    if str(constraints.get("op") or "").strip().lower() != operation:
        return False

    declared_path = Path(str(constraints.get("path") or "")).expanduser().resolve()
    requested_path = Path(str(requested.get("path") or "")).expanduser().resolve()
    if declared_path != requested_path:
        return False
    capture_roots = {
        (state_root() / "data" / "screenshots").resolve(),
        (state_root() / "data" / "ephemeral").resolve(),
    }
    if operation == "ensure_directory":
        path_allowed = requested_path in capture_roots
    else:
        path_allowed = (
            requested_path.parent in capture_roots
            and requested_path.suffix.casefold() == ".png"
        )
    return path_allowed


@dataclass(frozen=True, slots=True)
class ActionAdmission:
    """One canonical Will admission and the authority that issued it."""

    approved: bool
    reason: str
    receipt_id: str
    decision: Any
    authority: Any


#: Domains that change the world outside this process, and therefore create
#: something to look at.
_WORLD_CHANGING_DOMAINS = frozenset(
    {ActionDomain.ENVIRONMENT_ACTION, ActionDomain.EXTERNAL_ACTION}
)

#: How long acting on the world keeps her watching it.
#:
#: The claim is deliberately NOT released when the action returns. Look-act-
#: look needs the look AFTER the act: a click has to be seen to land, a drag
#: has to be seen to move, a board has to be seen to change. Releasing on
#: return would drop perception back to one frame every ten seconds precisely
#: when the result appears. Letting it expire instead means acting on the
#: world raises perception for a window around the action and then decays on
#: its own, with no bookkeeping for a caller to forget.
ACTION_PERCEPTION_WINDOW_S = 8.0


def _hold_perception_for(domain: ActionDomain, action_name: str) -> None:
    """Keep her eyes open around an action that changes the world.

    Every governed action passes through ActionExecutor.execute, so this is
    the one place that catches skills which do not exist yet. Wrapping the
    individual host_automation methods instead would have missed the next one
    added, and there are five of them already.
    """
    if domain not in _WORLD_CHANGING_DOMAINS:
        return
    try:
        from core.runtime.perception_demand import claim_perception

        claim_perception(
            f"{domain.value}:{action_name}", ttl_s=ACTION_PERCEPTION_WINDOW_S
        )
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation(
            "action_executor",
            exc,
            severity="info",
            action="acted without raising perception cadence",
        )


#: Authority a caller may PRESENT. Every one of these is verified downstream —
#: the standing-authority token is signed and checked against the token store —
#: so presenting one proves nothing on its own and forging one fails.
_PRESENTABLE_AUTHORITY_KEYS = frozenset(
    {
        "standing_authority_token",
        "standing_authority_grant_id",
        "standing_authority_receipt_id",
        "capability_token_id",
        "executive_intent_id",
        "authority_origin",
        "tool",
        "skill",
    }
)


def _check_what_the_skill_gave_back(engine: Any, name: str, result: Any) -> None:
    """Compare a skill's result against the schema it declared.

    Records, never raises, and never changes the result. Every one of the 82
    tools now says what it gives back; a declaration nothing checks is a
    comment, and a check that could cost the turn would be worse than the
    comment.
    """
    try:
        from core.skills.what_every_skill_gives_back import check_a_result

        held = getattr(engine, "skills", None) or {}
        metadata = held.get(name) if isinstance(held, dict) else None
        declared = getattr(metadata, "result_schema_def", None)
        if not declared or not getattr(metadata, "declares_its_result", False):
            return
        check_a_result(str(name), declared, result)
    except Exception as exc:  # noqa: BLE001 — a contract check must cost nothing
        logger.debug("could not check what %s gave back: %s", name, exc)


class ActionExecutor:
    """Execute, observe, and receipt one consequential action."""

    @classmethod
    def authorize_action(
        cls,
        *,
        domain: ActionDomain | str,
        action_name: str,
        params: Mapping[str, Any] | None,
        source: str = "unknown",
        priority: float = 0.5,
        context: Mapping[str, Any] | None = None,
    ) -> ActionAdmission:
        """Own the single Unified Will decision for an action admission.

        Callers that already own dispatch but need the canonical decision use
        this method instead of importing Will. A policy owner that has already
        denied an action must not call this merely to produce a second denial.
        """

        resolved_domain = _coerce_domain(domain)
        resolved_name = _coerce_action_name(action_name)
        resolved_params = _coerce_params(dict(params or {}))
        try:
            resolved_priority = float(priority)
        except (TypeError, ValueError) as exc:
            raise ValueError("action priority must be numeric") from exc
        if not math.isfinite(resolved_priority) or not 0.0 <= resolved_priority <= 1.0:
            raise ValueError("action priority must be finite and between 0 and 1")
        authority = get_will()
        decision_context = dict(context or {})
        ambient_context = _ambient_authority_context(
            resolved_domain,
            source=source,
            action_name=resolved_name,
            params=resolved_params,
        )
        decision_context.update(ambient_context)
        # These bindings are written after caller context is merged. Policy
        # may therefore classify an exact runtime-owned read contract without
        # trusting payload fields that a generic caller can overwrite.
        decision_context["action_executor_action_name"] = resolved_name
        decision_context["action_executor_source"] = str(source or "unknown")[:240]
        decision = authority.decide(
            content=_safe_action_summary(resolved_name, resolved_params),
            source=str(source or "unknown")[:240],
            domain=resolved_domain,
            priority=resolved_priority,
            context=decision_context,
        )
        checker = getattr(decision, "is_approved", None)
        approved = bool(checker()) if callable(checker) else False
        return ActionAdmission(
            approved=approved,
            reason=str(getattr(decision, "reason", "") or ""),
            receipt_id=str(getattr(decision, "receipt_id", "") or ""),
            decision=decision,
            authority=authority,
        )

    @classmethod
    def request_desktop_transport(
        cls,
        *,
        script: str,
        source: str,
        timeout_s: float = 15.0,
    ) -> dict[str, Any]:
        """Run bounded desktop IO inside an existing action decision.

        CapabilityEngine or ``execute`` owns the Will decision, welfare
        accounting, and final effect verification. Internal desktop steps
        reuse that receipt instead of manufacturing authority per script.
        """
        source_text = str(source or "").strip()
        if not source_text.startswith(("computer_use", "web_interlocutor.")):
            raise ValueError(
                "desktop transport source must be owned by computer_use or web_interlocutor"
            )
        require_governance(
            "action_executor.request_desktop_transport",
            strict=True,
            allowed_domains=(
                ActionDomain.ENVIRONMENT_ACTION.value,
                ActionDomain.EXTERNAL_ACTION.value,
                ActionDomain.TOOL_EXECUTION.value,
            ),
        )
        result = get_desktop_action_gateway().run_applescript(
            script,
            source=source_text,
            timeout=timeout_s,
        )
        if not isinstance(result, Mapping):
            return {
                "ok": False,
                "stdout": "",
                "stderr": "desktop_gateway_returned_non_mapping_result",
                "exit_code": -2,
            }
        return dict(result)

    @classmethod
    async def request_network_transport(
        cls,
        *,
        method: str,
        url: str,
        headers: dict[str, str] | None = None,
        data: bytes | str | None = None,
        timeout_s: float = 30.0,
        source: str,
        read_only: bool = False,
    ) -> dict[str, Any]:
        """Run network IO inside an already-approved ActionExecutor transaction."""
        source_text = str(source or "").strip()
        if not source_text.startswith("world_bridge:"):
            raise ValueError("network transport source must be owned by world_bridge")
        require_governance(
            "action_executor.request_network_transport",
            strict=True,
            allowed_domains=(
                ActionDomain.ENVIRONMENT_ACTION.value,
                ActionDomain.EXTERNAL_ACTION.value,
                ActionDomain.NETWORK_CALL.value,
            ),
        )
        result = await get_network_gateway().request_async(
            method=method,
            url=url,
            headers=headers,
            data=data,
            timeout=timeout_s,
            source=source_text,
            read_only=read_only,
        )
        if not isinstance(result, Mapping):
            return {
                "ok": False,
                "status_code": 0,
                "error": "network_gateway_returned_non_mapping_result",
            }
        return dict(result)

    @staticmethod
    def _what_stops_this(stopping: Any) -> Any:
        """The token for this action: the caller's, or the ambient one."""

        if stopping is not None:
            return stopping
        try:
            from core.runtime.what_stops_it import current

            return current(whose="action_executor.execute").stopping
        except (ImportError, RuntimeError, TypeError, ValueError):
            return None

    @classmethod
    async def execute(
        cls,
        *,
        domain: ActionDomain | str,
        action_name: str,
        params: dict[str, Any],
        source: str = "unknown",
        predicted_welfare_delta: dict[str, float] | None = None,
        rollback_target: str | None = None,
        expectation: ActionExpectation | Mapping[str, Any] | None = None,
        effect_handler: EffectHandler | None = None,
        effect_verifier: EffectVerifier | None = None,
        execution_timeout_s: float | None = None,
        verification_timeout_s: float | None = None,
        action_id: str | None = None,
        authority_context: Mapping[str, Any] | None = None,
        stopping: Any = None,
    ) -> dict[str, Any]:
        domain = _coerce_domain(domain)
        action_name = _coerce_action_name(action_name)
        params = _coerce_params(params)
        # A caller that can say stop, saying it before anything happens.
        #
        # Two peer architectures thread a cancellation token through tool
        # calls so that cooperative cancellation composes. Aura had asyncio
        # cancellation, deadlines and a token inside the voice duplex, none of
        # which reach here: a turn abandoned mid-tool ran the tool to
        # completion and threw the answer away.
        #
        # Passed explicitly where the caller has one, read from the ambient
        # context where it does not — and reading it counts, so what has not
        # been threaded is a number rather than an impression.
        halt = cls._what_stops_this(stopping)
        if halt is not None and halt.stopped:
            return {
                "ok": False,
                "error": "stopped before it started",
                "why": halt.why,
                "domain": domain.value,
                "action_name": action_name,
            }
        _hold_perception_for(domain, action_name)
        handler_name = _validate_effect_handler(
            domain,
            effect_handler=effect_handler,
            effect_verifier=effect_verifier,
        )
        execution_timeout = _coerce_execution_timeout(execution_timeout_s)
        expectation_contract = _coerce_expectation(expectation) or default_action_expectation(
            domain,
            action_name,
        )
        action_id = _coerce_action_id(action_id)
        request_digest = _stable_digest(
            {
                "domain": domain.value,
                "action_name": action_name,
                "params": params,
                "source": source,
                "effect_handler": handler_name,
                "expectation": expectation_contract.to_dict(),
            }
        )

        # Terminal external-effect transactions are replayed before asking Will
        # for a new decision. The original authorization and effect receipt are
        # authoritative; minting a second Will receipt would split one action
        # across two governance lineages.
        preaction_thread = None
        external_execution_offer: dict[str, Any] | None = None
        external_execution_transaction: dict[str, Any] = {}
        external_execute_coordinator: Any = None
        deliberation_worthy_action = False
        build_rehearsal_objective: Any = None
        build_external_execution_offer: Any = None
        try:
            # core/runtime is the foundation layer and may not depend on
            # cognition (core/runtime/DEPS). This module is the one
            # grandfathered exception, because the branch below is gated on
            # deliberation_worthy() and means nothing without the pre-action
            # cortex. Everything cognitive this function needs comes through
            # that single seam — reaching past it into
            # external_execute_coordinator and latent_cortex.external_execution
            # added two more edges into the foundation, which the layering
            # gate refuses and the grandfathered baseline may not grow to
            # cover. Both are re-exported lazily by preaction_cortex, so the
            # latent-cortex stack is still not imported until an action
            # actually deliberates.
            from core.brain.preaction_cortex import (
                PreActionCortexThread,
                build_external_execution_offer,
                build_rehearsal_objective,
                deliberation_worthy,
                get_external_execute_coordinator,
            )

            deliberation_worthy_action = deliberation_worthy(domain.value)
            if deliberation_worthy_action:
                external_execute_coordinator = get_external_execute_coordinator()
                existing_transaction = await asyncio.to_thread(
                    external_execute_coordinator.lookup,
                    action_id=action_id,
                    request_digest=request_digest,
                )
                if existing_transaction is not None:
                    external_execution_offer = dict(
                        existing_transaction.get("offer") or {}
                    )
                    external_execution_transaction = await asyncio.to_thread(
                        external_execute_coordinator.prepare,
                        external_execution_offer,
                    )
                    replay = _external_transaction_result(
                        external_execution_transaction,
                        action_id=action_id,
                        request_digest=request_digest,
                        expectation=expectation_contract,
                    )
                    if replay is not None:
                        return await _heal_external_post_action_receipt(
                            coordinator=external_execute_coordinator,
                            offer=external_execution_offer,
                            transaction=external_execution_transaction,
                            replay=replay,
                            action_id=action_id,
                            request_digest=request_digest,
                            expectation=expectation_contract,
                        )
        except ImportError as exc:
            # `deliberation_worthy_action = False` is the SAME value this
            # function uses for "this action does not need deliberation", so a
            # cortex that failed to import was indistinguishable from a policy
            # decision that it was unnecessary — at logger.debug, which nobody
            # reads. Every consequential action would then proceed
            # undeliberated with no signal that the check had not run.
            #
            # Not deliberating because the cortex says so is a decision.
            # Not deliberating because the cortex could not be loaded is a lost
            # capability, and it is recorded as one.
            deliberation_worthy_action = False
            record_degradation(
                "action_executor.preaction_cortex",
                exc,
                action=(
                    f"proceeded with {action_name} WITHOUT pre-action "
                    "deliberation; the cortex could not be imported"
                ),
                severity="degraded",
                classification=FallbackClassification.SILENT_LOSS_OF_CAPABILITY,
                extra={"action": action_name, "domain": domain.value},
                enforce_failure_policy=False,
            )
        except _ACTION_EXECUTOR_RECOVERABLE_ERRORS as exc:
            record_degradation(
                "action_executor.external_execution",
                exc,
                action=(
                    f"refused {action_name} before authorization because prior "
                    "external execution state could not be proven"
                ),
                severity="degraded",
                enforce_failure_policy=False,
            )
            return {
                "ok": False,
                "status": SkillStatus.FAILED_RECOVERABLE.value,
                "error": (
                    "external_execution_preflight_failed:"
                    f"{type(exc).__name__}"
                ),
                "action_expectation": expectation_contract.to_dict(),
                "action_id": action_id,
                "request_digest": request_digest,
                "transport_succeeded": False,
                "effect_verified": False,
                "retry_safe": False,
                "manual_reconciliation_required": False,
                "external_execution_transaction": dict(
                    external_execution_transaction
                ),
            }

        # The authority the caller already holds, presented to the will.
        #
        # There was no channel for it. A request carrying a valid, signed
        # standing-authority grant had no way to show it here, so
        # `validate_context` looked for `standing_authority_token`, never found
        # one, and answered `signed_standing_authority_lease_missing` — for
        # every consequential action from every authorised origin. Measured
        # live 2026-08-18 on an owner foreground request that held
        # `owner.foreground-request` with allowed_tools ("*").
        #
        # Only these keys cross, and passing one forges nothing: the token is
        # signed and `validate_context` verifies it against the token store, so
        # a fabricated value fails exactly as it should. What this restores is
        # the ability to present a real one.
        authority_view = {
            key: value
            for key, value in dict(authority_context or {}).items()
            if key in _PRESENTABLE_AUTHORITY_KEYS
        }
        admission = cls.authorize_action(
            domain=domain,
            action_name=action_name,
            params=params,
            source=source,
            priority=0.5,
            context={
                "action_id": action_id,
                "request_digest": request_digest,
                "expectation_objective": expectation_contract.objective[:500],
                "rollback_target_declared": bool(rollback_target),
                # The arguments this action actually carries. A standing
                # lease is bound to an argument digest, and the will validated
                # with `arguments=None`, so a correctly-issued lease always
                # answered `standing_authority_arguments_mismatch`. The digest
                # exists to prove the lease is being used for the action it was
                # issued for; it can only do that if the action is shown.
                "authority_arguments": params,
                **authority_view,
            },
        )
        will = admission.authority
        decision = admission.decision
        if not admission.approved:
            logger.warning(
                "ActionExecutor refused %s in domain %s",
                action_name,
                domain.value,
            )
            return {
                "ok": False,
                "status": SkillStatus.BLOCKED_BY_POLICY.value,
                "error": f"Will refused action: {admission.reason}",
                "will_receipt_id": admission.receipt_id,
                "action_expectation": expectation_contract.to_dict(),
                "action_id": action_id,
                "request_digest": request_digest,
                "transport_succeeded": False,
                "effect_verified": False,
                "retry_safe": False,
                "manual_reconciliation_required": False,
            }

        will_receipt_id = admission.receipt_id
        # Pre-action cortex: consequential actions get ONE cognitive thread
        # across their whole cycle — a latent rehearsal now (predicted
        # effect, preconditions, failure mode) and a discrepancy-driven
        # reconciliation after observation. Fully defensive: no latent
        # service, busy gate, or kill switch ⇒ receipted skip, same action.
        try:
            if deliberation_worthy_action:
                action_summary = _safe_action_summary(action_name, params)
                if not external_execution_transaction:
                    rehearsal_objective = build_rehearsal_objective(
                        action_summary=action_summary,
                        expectation_objective=expectation_contract.objective,
                    )
                    external_execution_offer = build_external_execution_offer(
                        action_id=action_id,
                        domain=domain.value,
                        action_name=action_name,
                        request_digest=request_digest,
                        will_receipt_id=will_receipt_id,
                        objective=rehearsal_objective,
                        expectation=expectation_contract.to_dict(),
                    )
                    external_execution_transaction = await asyncio.to_thread(
                        external_execute_coordinator.prepare,
                        external_execution_offer,
                    )

                preaction_thread = PreActionCortexThread(
                    domain=domain.value,
                    action_name=action_name,
                    request_digest=request_digest,
                    external_execution_offer=external_execution_offer,
                )
                if (
                    external_execution_offer is not None
                    and external_execution_transaction.get("state") == "DECIDED"
                ):
                    preaction_thread.rehearsal = {
                        "schema": "aura.preaction_cortex.v1",
                        "phase": "rehearsal",
                        "action_name": action_name,
                        "domain": domain.value,
                        "ran": False,
                        "skip_reason": "durable_external_execution_decision_reused",
                    }
                else:
                    rehearsal = await preaction_thread.rehearse(
                        action_summary=action_summary,
                        expectation_objective=expectation_contract.objective,
                    )
                    if external_execution_offer is not None:
                        if rehearsal.get("ran") is True:
                            external_execution_transaction = await asyncio.to_thread(
                                external_execute_coordinator.record_handoff,
                                offer=external_execution_offer,
                                handoff=rehearsal.get("external_execution_handoff") or {},
                                cognitive_action_trace=(
                                    preaction_thread.external_execution_trace()
                                ),
                                readiness=(
                                    preaction_thread.external_execution_readiness()
                                ),
                                model_output=(
                                    preaction_thread.external_execution_model_output()
                                ),
                                action_policy_evidence=(
                                    preaction_thread.external_action_policy_evidence()
                                ),
                                executors=(
                                    preaction_thread.external_action_executors()
                                ),
                                action_policy_receipt=(
                                    preaction_thread.external_action_policy_receipt()
                                ),
                                runtime_operation=(
                                    preaction_thread.external_runtime_operation()
                                ),
                            )
                        else:
                            # Named as the class it belongs to. A rehearsal
                            # that recorded no reason did not run, which is an
                            # availability failure — and calling it anything
                            # else made the executor's own fallback ineligible
                            # for the bypass it was reaching for, refusing the
                            # action outright.
                            skip_reason = str(
                                rehearsal.get("skip_reason")
                                or "availability_failure:rehearsal_unavailable"
                            )
                            external_execution_transaction = await asyncio.to_thread(
                                external_execute_coordinator.record_bypass,
                                offer=external_execution_offer,
                                reason=skip_reason,
                            )

                replay = _external_transaction_result(
                    external_execution_transaction,
                    action_id=action_id,
                    request_digest=request_digest,
                    expectation=expectation_contract,
                    preaction_receipt=preaction_thread.to_receipt(),
                )
                if replay is not None:
                    if external_execution_transaction.get("state") == "ABSTAINED":
                        return await _finalize_approved_no_effect(
                            result=replay,
                            will=will,
                            will_receipt_id=will_receipt_id,
                            domain=domain,
                            action_name=action_name,
                            source=source,
                            expectation=expectation_contract,
                            action_id=action_id,
                            request_digest=request_digest,
                            rollback_target=rollback_target,
                            coordinator=external_execute_coordinator,
                            offer=external_execution_offer,
                        )
                    return await _heal_external_post_action_receipt(
                        coordinator=external_execute_coordinator,
                        offer=external_execution_offer,
                        transaction=external_execution_transaction,
                        replay=replay,
                        action_id=action_id,
                        request_digest=request_digest,
                        expectation=expectation_contract,
                    )
        except (ImportError, *_ACTION_EXECUTOR_RECOVERABLE_ERRORS) as exc:
            if (
                external_execution_offer is not None
                or external_execution_transaction
            ):
                record_degradation(
                    "action_executor.external_execution",
                    exc,
                    action=(
                        f"refused {action_name} before effect dispatch because "
                        "its external execution transaction could not be proven"
                    ),
                    severity="degraded",
                    enforce_failure_policy=False,
                )
                failure_result = {
                    "ok": False,
                    "status": SkillStatus.FAILED_RECOVERABLE.value,
                    # The type is not the reason.
                    #
                    # "external_execution_preparation_failed:ValueError" is
                    # what the person is shown when an action refuses before
                    # it starts, and it names none of the dozen things that
                    # raise ValueError in preparation. Live 2026-08-31 it was
                    # shown three times over a single afternoon for three
                    # different causes. What the exception says, and where it
                    # was raised, are both already known here.
                    "error": (
                        "external_execution_preparation_failed:"
                        f"{type(exc).__name__}: {exc}"
                        f" [raised at {_raise_site(exc)}]"
                    ),
                    "will_receipt_id": will_receipt_id,
                    "action_expectation": expectation_contract.to_dict(),
                    "action_id": action_id,
                    "request_digest": request_digest,
                    "transport_succeeded": False,
                    "effect_verified": False,
                    "retry_safe": not isinstance(exc, ValueError),
                    "manual_reconciliation_required": False,
                    "external_execution_transaction": dict(
                        external_execution_transaction
                    ),
                }
                return await _finalize_approved_no_effect(
                    result=failure_result,
                    will=will,
                    will_receipt_id=will_receipt_id,
                    domain=domain,
                    action_name=action_name,
                    source=source,
                    expectation=expectation_contract,
                    action_id=action_id,
                    request_digest=request_digest,
                    rollback_target=rollback_target,
                    coordinator=external_execute_coordinator,
                    offer=external_execution_offer,
                )
            logger.debug("Pre-action rehearsal unavailable: %s", exc)
            preaction_thread = None
        body_service = BodyStateService.get()
        welfare_service = WelfareState.get()
        tx = WelfareTransaction.begin(
            domain=domain.value,
            action=f"{action_name} ({source})",
            welfare_before=welfare_service.last_outputs,
            body_before=body_service.snapshot(),
            predicted_welfare_delta=predicted_welfare_delta,
            will_receipt_id=will_receipt_id,
        )

        result: dict[str, Any]
        pre_state: dict[str, Any] = {}
        dispatch_result: dict[str, Any] = {}
        external_dispatch_attempt_id = ""
        external_dispatch_heartbeat: asyncio.Task[None] | None = None
        try:
            verifier_budget_s = (
                float(verification_timeout_s)
                if verification_timeout_s is not None
                else float(_VERIFIER_TIMEOUT_FLAG.value())
            )
            # The governed scope must cover the operation it authorizes. The
            # default 30-second token otherwise expires inside legitimate
            # browser and desktop actions whose declared transaction budget is
            # longer, leaving a partially applied effect with no live receipt.
            # The scope is still revoked immediately when this lexical block
            # exits; this only prevents premature expiry while it is active.
            governance_ttl_s = execution_timeout + max(0.0, verifier_budget_s) + 15.0
            async with governed_scope(decision, ttl=governance_ttl_s):
                pre_state = await capture_pre_action_state(domain, params)
                if external_execution_offer is not None:
                    begin_task = get_task_tracker().create_task(
                        asyncio.to_thread(
                            external_execute_coordinator.begin_dispatch,
                            external_execution_offer,
                            authorization_receipt_id=will_receipt_id,
                            task_id=_external_dispatch_task_id(),
                        )
                    )
                    try:
                        external_execution_transaction = await asyncio.shield(
                            begin_task
                        )
                    except asyncio.CancelledError:
                        external_execution_transaction = await asyncio.shield(
                            begin_task
                        )
                        cancelled_owner = (
                            external_execution_transaction.get("dispatch_owner")
                            or {}
                        )
                        await _abandon_external_dispatch(
                            coordinator=external_execute_coordinator,
                            offer=external_execution_offer,
                            dispatch_attempt_id=str(
                                cancelled_owner.get("attempt_id") or ""
                            ),
                            effect_may_have_occurred=False,
                            reason="cancelled_before_effect_dispatch",
                        )
                        raise
                    external_dispatch_attempt_id = str(
                        (
                            external_execution_transaction.get("dispatch_owner")
                            or {}
                        ).get("attempt_id")
                        or ""
                    )
                    if not external_dispatch_attempt_id:
                        raise ValueError(
                            "external execution dispatch intent lacks an owner token"
                        )
                    external_dispatch_heartbeat = get_task_tracker().create_task(
                        _renew_external_dispatch_lease(
                            coordinator=external_execute_coordinator,
                            offer=external_execution_offer,
                            dispatch_attempt_id=external_dispatch_attempt_id,
                        )
                    )
                dispatch_result = await cls._dispatch(
                    domain=domain,
                    action_name=action_name,
                    params=params,
                    source=source,
                    will_receipt_id=will_receipt_id,
                    expectation=expectation_contract,
                    effect_handler=effect_handler,
                    execution_timeout_s=execution_timeout,
                )
                result = dict(dispatch_result)
                observation = await observe_action_effect(
                    domain,
                    params,
                    result,
                    pre_state=pre_state,
                    verifier=effect_verifier,
                    verifier_timeout_s=(
                        verifier_budget_s
                    ),
                )
                result.update(observation)
        except asyncio.CancelledError:
            if external_dispatch_attempt_id:
                await _abandon_external_dispatch(
                    coordinator=external_execute_coordinator,
                    offer=external_execution_offer,
                    dispatch_attempt_id=external_dispatch_attempt_id,
                    effect_may_have_occurred=True,
                    reason="cancelled_after_dispatch_intent",
                )
            await _stop_external_dispatch_heartbeat(
                external_dispatch_heartbeat
            )
            raise
        except _ACTION_EXECUTOR_RECOVERABLE_ERRORS as exc:
            record_degradation(
                "action_executor",
                exc,
                action=f"recorded failed action transaction for {action_name}",
            )
            logger.error("Error executing action %s: %s", action_name, exc, exc_info=True)
            transport_may_have_succeeded = dispatch_result.get("ok") is True
            result = {
                "ok": False,
                "status": SkillStatus.FAILED_RECOVERABLE.value,
                "error": str(exc),
                "effect_verified": False,
                "transport_succeeded": transport_may_have_succeeded,
                "verification_evidence": {
                    "observation": {
                        "effect_verified": False,
                        "reason": (
                            "verification_exception_after_transport"
                            if transport_may_have_succeeded
                            else "execution_exception"
                        ),
                        "error_type": type(exc).__qualname__,
                    }
                },
            }

        transport_succeeded = bool(
            result.get("transport_succeeded", result.get("ok", False))
        )
        result["action_id"] = action_id
        result["request_digest"] = request_digest
        result["transport_succeeded"] = transport_succeeded
        if result.get("ok", False):
            result["status"] = (
                SkillStatus.SUCCESS_VERIFIED.value
                if result.get("effect_verified") is True
                else SkillStatus.SUCCESS_UNVERIFIED.value
            )
            result = apply_action_expectation_payload(
                action_name,
                result,
                expectation_contract,
            )
        else:
            result.setdefault("status", SkillStatus.FAILED_RECOVERABLE.value)
            result["action_expectation"] = expectation_contract.to_dict()

        status = str(result.get("status") or SkillStatus.FAILED_RECOVERABLE.value)
        effect_verified = result.get("effect_verified") is True
        if transport_succeeded and not effect_verified and not result.get("ok", False):
            status = SkillStatus.PARTIAL_SUCCESS.value
            result["status"] = status
        result["retry_safe"] = bool(
            result.get("retry_safe") is True
            and not effect_verified
            and not transport_succeeded
        )
        result["manual_reconciliation_required"] = bool(
            transport_succeeded and not effect_verified
        )
        tx_outcome = _transaction_outcome(status, bool(result.get("ok", False)))
        error_msg = str(result.get("error") or "") if not result.get("ok", False) else ""
        tx_record = None
        welfare_transaction_completed = False
        try:
            tx_record = tx.complete(
                outcome=tx_outcome,
                welfare_after=welfare_service.last_outputs,
                body_after=body_service.snapshot(),
                error=error_msg,
            )
            welfare_transaction_completed = True
        except _ACTION_EXECUTOR_RECOVERABLE_ERRORS as exc:
            record_degradation(
                "action_executor",
                exc,
                action=f"effect lane completed but welfare transaction closure failed for {action_name}",
                enforce_failure_policy=False,
            )
            result["ok"] = False
            result["status"] = (
                SkillStatus.PARTIAL_SUCCESS.value
                if transport_succeeded
                else SkillStatus.FAILED_RECOVERABLE.value
            )
            result["error"] = _append_error(
                result.get("error"),
                f"welfare_transaction_completion_failed:{exc}",
            )
            result["manual_reconciliation_required"] = transport_succeeded
            result["retry_safe"] = False
            status = str(result["status"])
            error_msg = str(result["error"])

        if tx_record is not None:
            try:
                will.record_outcome(will_receipt_id, tx_record)
            except _ACTION_EXECUTOR_RECOVERABLE_ERRORS as exc:
                record_degradation(
                    "action_executor",
                    exc,
                    action=f"continued after Will outcome reinforcement failed for {action_name}",
                )

        if preaction_thread is not None:
            # Same cognitive thread, phase 2: reality arrived. Replanning
            # runs only on objective discrepancy (transport failure or an
            # unverified effect) and its conclusion competes for Global
            # Workspace broadcast — the loop closes through the mind.
            try:
                await _await_external_closure(
                    preaction_thread.reconcile(result),
                    coordinator=external_execute_coordinator,
                    offer=external_execution_offer,
                    dispatch_attempt_id=external_dispatch_attempt_id,
                    heartbeat=external_dispatch_heartbeat,
                    result=result,
                    cancellation_reason="cancelled_during_external_reconciliation",
                )
                result["preaction_cortex"] = preaction_thread.to_receipt()
            except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                logger.debug("Pre-action reconciliation unavailable: %s", exc)

        result["will_receipt_id"] = will_receipt_id
        result["welfare_transaction_id"] = tx.tx_id
        result["welfare_transaction_completed"] = welfare_transaction_completed
        output_hash = _stable_digest(result)
        actual_outcome = (
            tx_record.outcome
            if tx_record is not None
            else _transaction_outcome(status, bool(result.get("ok", False)))
        )
        body_delta = tx_record.body_delta if tx_record is not None else {}
        post_receipt_id = (
            _external_post_receipt_id(action_id, request_digest)
            if external_execution_offer is not None
            else f"post-{uuid.uuid4()}"
        )
        post_receipt = PostActionReceipt(
            receipt_id=post_receipt_id,
            will_receipt_id=will_receipt_id,
            executor_name=action_name,
            actual_outcome=actual_outcome,
            output_hash=output_hash,
            error_status=error_msg,
            welfare_transaction_id=tx.tx_id,
            body_delta=body_delta,
            memory_delta=(
                {"record_id": result.get("record_id")}
                if result.get("record_id")
                else {}
            ),
            rollback_target=rollback_target,
            status=status,
            effect_verified=effect_verified,
            action_expectation=_bounded_receipt_mapping(expectation_contract.to_dict()),
            verification_evidence=_bounded_receipt_mapping(
                result.get("verification_evidence") or {}
            ),
            action_id=action_id,
            domain=domain.value,
            source=str(source or "unknown")[:240],
            request_digest=request_digest,
            transport_succeeded=transport_succeeded,
            retry_safe=bool(result.get("retry_safe", False)),
            manual_reconciliation_required=bool(
                result.get("manual_reconciliation_required", False)
            ),
            welfare_transaction_completed=welfare_transaction_completed,
        )
        if external_execution_offer is not None:
            result.update(
                {
                    "receipt_persisted": False,
                    "post_action_receipt_pending": True,
                    "post_action_receipt_attempt_id": post_receipt.receipt_id,
                    "_post_action_recovery_contract": post_receipt.to_dict(),
                }
            )
        await _await_external_closure(
            _complete_external_execution_transaction(
                coordinator=external_execute_coordinator,
                offer=external_execution_offer,
                transaction=external_execution_transaction,
                dispatch_attempt_id=external_dispatch_attempt_id,
                result=result,
                action_name=action_name,
            ),
            coordinator=external_execute_coordinator,
            offer=external_execution_offer,
            dispatch_attempt_id=external_dispatch_attempt_id,
            heartbeat=external_dispatch_heartbeat,
            result=result,
            cancellation_reason="cancelled_during_external_completion",
        )
        await _stop_external_dispatch_heartbeat(external_dispatch_heartbeat)
        result.pop("_post_action_recovery_contract", None)
        final_transport_succeeded = result.get("transport_succeeded") is True
        final_status = str(
            result.get("status")
            or SkillStatus.FAILED_RECOVERABLE.value
        )
        final_effect_verified = result.get("effect_verified") is True
        final_error_msg = (
            str(result.get("error") or "")
            if not result.get("ok", False)
            else ""
        )
        if (
            final_transport_succeeded != post_receipt.transport_succeeded
            or final_status != post_receipt.status
            or final_effect_verified != post_receipt.effect_verified
            or final_error_msg != post_receipt.error_status
        ):
            post_receipt = PostActionReceipt(
                **{
                    **post_receipt.to_dict(),
                    "output_hash": _stable_digest(result),
                    "status": final_status,
                    "effect_verified": final_effect_verified,
                    "error_status": final_error_msg,
                    "transport_succeeded": final_transport_succeeded,
                    "retry_safe": bool(result.get("retry_safe", False)),
                    "manual_reconciliation_required": bool(
                        result.get(
                            "manual_reconciliation_required",
                            False,
                        )
                    ),
                }
            )
        transport_succeeded = final_transport_succeeded
        status = final_status
        effect_verified = final_effect_verified
        error_msg = final_error_msg
        if (
            external_execute_coordinator is not None
            and external_execution_offer is not None
        ):
            try:
                await asyncio.to_thread(
                    external_execute_coordinator.stage_post_action_receipt,
                    offer=external_execution_offer,
                    receipt_contract=post_receipt.to_dict(),
                )
            except _ACTION_EXECUTOR_RECOVERABLE_ERRORS as exc:
                record_degradation(
                    "action_executor.external_execution",
                    exc,
                    action=(
                        "preserved the terminal effect state while its final "
                        f"receipt recovery contract could not be staged for {action_name}"
                    ),
                    severity="degraded",
                    enforce_failure_policy=False,
                )
        try:
            receipt_store = get_post_action_receipt_store()
            await receipt_store.record_async(post_receipt)
            if external_execution_offer is not None:
                persisted_post_receipt = (
                    _validated_persisted_external_post_receipt(
                        receipt_store.get_receipt(post_receipt.receipt_id),
                        action_id=action_id,
                        request_digest=request_digest,
                        will_receipt_id=will_receipt_id,
                        expected_contract=post_receipt.to_dict(),
                    )
                )
            else:
                persisted_post_receipt = receipt_store.get_receipt(
                    post_receipt.receipt_id
                )
                if (
                    persisted_post_receipt is None
                    or persisted_post_receipt.to_dict() != post_receipt.to_dict()
                ):
                    raise ValueError(
                        "persisted post-action receipt content differs"
                    )
        except _ACTION_EXECUTOR_RECOVERABLE_ERRORS as exc:
            record_degradation(
                "action_executor",
                exc,
                severity="degraded",
                action=f"action effect occurred but post-action receipt failed for {action_name}",
                enforce_failure_policy=False,
            )
            result["ok"] = False
            result["status"] = (
                SkillStatus.PARTIAL_SUCCESS.value
                if transport_succeeded
                else SkillStatus.FAILED_RECOVERABLE.value
            )
            result["error"] = _append_error(
                result.get("error"),
                f"post_action_receipt_persistence_failed:{exc}",
            )
            result["receipt_persisted"] = False
            result["post_action_receipt_attempt_id"] = post_receipt.receipt_id
            result["manual_reconciliation_required"] = transport_succeeded
            result["retry_safe"] = False
            return result

        result["post_action_receipt_id"] = post_receipt.receipt_id
        result["post_action_output_hash"] = post_receipt.output_hash
        result["receipt_persisted"] = True
        result["post_action_receipt_pending"] = False
        if external_execute_coordinator is not None and external_execution_offer is not None:
            try:
                linked = await asyncio.to_thread(
                    external_execute_coordinator.link_post_action_receipt,
                    offer=external_execution_offer,
                    persisted_receipt=persisted_post_receipt.to_dict(),
                    receipt_store=receipt_store,
                )
                result["external_execution_transaction"] = linked
                result["external_execution_receipt_linked"] = True
            except _ACTION_EXECUTOR_RECOVERABLE_ERRORS as exc:
                record_degradation(
                    "action_executor.external_execution",
                    exc,
                    action=(
                        "preserved completed effect and durable post-action "
                        f"receipt after transaction-link failure for {action_name}"
                    ),
                    severity="degraded",
                    enforce_failure_policy=False,
                )
                result["external_execution_receipt_linked"] = False
        return result

    @staticmethod
    async def _dispatch(
        *,
        domain: ActionDomain,
        action_name: str,
        params: dict[str, Any],
        source: str,
        will_receipt_id: str,
        expectation: ActionExpectation,
        effect_handler: EffectHandler | None,
        execution_timeout_s: float,
    ) -> dict[str, Any]:
        if effect_handler is not None:
            return await _invoke_effect_handler(
                effect_handler,
                {
                    "domain": domain.value,
                    "action_name": action_name,
                    "params": dict(params),
                    "source": source,
                    "will_receipt_id": will_receipt_id,
                    "action_expectation": expectation.to_dict(),
                },
                timeout_s=execution_timeout_s,
            )
        if domain == ActionDomain.TOOL_EXECUTION:
            if "argv" in params:
                proc = await get_subprocess_gateway().run_async(
                    argv=params["argv"],
                    cwd=params.get("cwd"),
                    env=params.get("env"),
                    timeout=params.get("timeout", 30.0),
                    source=source,
                    accelerator_capability="auto",
                )
                return {
                    "ok": proc.returncode == 0,
                    "stdout": proc.stdout,
                    "stderr": proc.stderr,
                    "exit_code": proc.returncode,
                }
            from core.container import ServiceContainer

            engine = ServiceContainer.get("capability_engine", default=None)
            if engine is None or not hasattr(engine, "execute"):
                return {"ok": False, "error": "capability_engine_unavailable"}
            raw_result = await engine.execute(
                action_name,
                params,
                context={
                    "source": source,
                    "will_receipt_id": will_receipt_id,
                    "action_executor_managed_welfare_transaction": True,
                    "action_expectation": expectation.to_dict(),
                },
            )
            _check_what_the_skill_gave_back(engine, action_name, raw_result)
            return _coerce_result(raw_result)

        if domain == ActionDomain.FILE_WRITE:
            gateway = get_file_write_gateway()
            path = _coerce_path_param(params.get("path"), "path")
            operation = str(params.get("op") or "").strip().lower()
            if operation == "ensure_directory":
                directory = await gateway.ensure_directory_async(path, source=source)
                return {"ok": True, "path": directory, "directory_created": True}
            if operation == "delete":
                deleted = await gateway.delete_path_async(
                    path,
                    recursive=bool(params.get("recursive", False)),
                    source=source,
                )
                return {"ok": True, "path": str(path), "deleted": deleted}
            if operation == "move":
                destination = _coerce_path_param(
                    params.get("destination"),
                    "destination",
                )
                final = await gateway.move_path_async(
                    path,
                    destination,
                    source=source,
                )
                return {"ok": True, "path": str(path), "destination": final}
            if operation == "copy":
                destination = _coerce_path_param(
                    params.get("destination"),
                    "destination",
                )
                final = await gateway.copy_path_async(
                    path,
                    destination,
                    source=source,
                )
                return {"ok": True, "path": str(path), "destination": final}
            if "text" in params:
                await gateway.write_text_async(
                    path,
                    params["text"],
                    encoding=str(params.get("encoding") or "utf-8"),
                    source=source,
                )
                return {"ok": True, "path": str(path)}
            if "payload" in params:
                await gateway.write_bytes_async(path, params["payload"], source=source)
                return {"ok": True, "path": str(path)}
            if "obj" in params:
                await gateway.write_json_async(
                    path,
                    params["obj"],
                    schema_version=int(params.get("schema_version", 1)),
                    schema_name=params.get("schema_name"),
                    source=source,
                )
                return {"ok": True, "path": str(path)}
            return {"ok": False, "error": "invalid_file_write_params"}

        if domain in {
            ActionDomain.NETWORK_CALL,
            ActionDomain.CLOUD_CALL,
            ActionDomain.CLOUD_FALLBACK,
        }:
            network_result = await get_network_gateway().request_async(
                method=params.get("method", "GET"),
                url=params.get("url", ""),
                headers=params.get("headers"),
                data=params.get("data"),
                timeout=params.get("timeout", 30.0),
                source=source,
            )
            if not isinstance(network_result, Mapping):
                return {
                    "ok": False,
                    "error": "network_gateway_returned_non_mapping_result",
                }
            return dict(network_result)

        if domain in {ActionDomain.ENVIRONMENT_ACTION, ActionDomain.EXTERNAL_ACTION}:
            desktop_result = await get_desktop_action_gateway().run_applescript_async(
                params.get("script", ""),
                source=source,
                timeout=params.get("timeout", 15.0),
            )
            if not isinstance(desktop_result, Mapping):
                return {
                    "ok": False,
                    "error": "desktop_gateway_returned_non_mapping_result",
                }
            return dict(desktop_result)

        if domain == ActionDomain.MEMORY_WRITE:
            from core.runtime.gateways import MemoryWriteRequest

            memory_receipt = await get_memory_write_gateway().write(
                MemoryWriteRequest(
                    content=str(params.get("content", "")),
                    metadata=dict(params.get("metadata", {}) or {}),
                    receipt_id=will_receipt_id,
                    cause=source,
                )
            )
            return {
                "ok": True,
                "record_id": memory_receipt.record_id,
                "receipt_id": memory_receipt.receipt_id,
                "bytes_written": memory_receipt.bytes_written,
            }

        if domain == ActionDomain.STATE_MUTATION:
            from core.runtime.gateways import StateMutationRequest

            key = str(params.get("key") or "")
            new_value = params.get("new_value", params.get("value"))
            state_gateway = get_state_gateway()
            state_receipt = await state_gateway.mutate(
                StateMutationRequest(
                    key=key,
                    new_value=new_value,
                    receipt_id=will_receipt_id,
                    cause=source,
                    domain=str(params.get("state_domain") or "world_state"),
                )
            )
            state_domain = str(params.get("state_domain") or "world_state")
            readback = await state_gateway.read(
                key,
                default=object(),
                domain=state_domain,
                fresh=True,
            )
            return {
                "ok": True,
                "key": state_receipt.key,
                "old_value": state_receipt.old_value,
                "new_value": state_receipt.new_value,
                "receipt_id": state_receipt.receipt_id,
                "readback_verified": readback == new_value,
            }

        if domain == ActionDomain.SELF_MODIFICATION:
            from core.self_modification.safe_modification_harness import run_self_mod_test

            tested = await run_self_mod_test(
                params.get("patch_path"),
                params.get("test_command"),
            )
            return {
                "ok": bool(tested.get("passed", False)),
                "test_output": tested.get("output", ""),
                "canary_passed": bool(tested.get("passed", False)),
                "applied": bool(tested.get("applied", False)),
            }
        return {"ok": False, "error": f"unsupported_action_domain:{domain.value}"}


def _coerce_domain(domain: ActionDomain | str) -> ActionDomain:
    if isinstance(domain, ActionDomain):
        return domain
    try:
        return ActionDomain(str(domain))
    except ValueError as exc:
        raise ValueError(f"unsupported action domain: {domain}") from exc


def _coerce_action_name(action_name: str) -> str:
    if not isinstance(action_name, str):
        raise TypeError("action_name must be a string")
    text = action_name.strip()
    if not text:
        raise ValueError("action_name must not be empty")
    return text[:160]


def _coerce_params(params: dict[str, Any]) -> dict[str, Any]:
    if params is None:
        return {}
    if not isinstance(params, dict):
        raise TypeError("params must be a dict")
    return dict(params)


def _coerce_result(raw_result: Any) -> dict[str, Any]:
    if isinstance(raw_result, dict):
        result = dict(raw_result)
        result.setdefault("ok", bool(result.get("ok", False)))
        return result
    return {"ok": bool(raw_result), "result": raw_result}


def _validate_effect_handler(
    domain: ActionDomain,
    *,
    effect_handler: EffectHandler | None,
    effect_verifier: EffectVerifier | None,
) -> str:
    if effect_handler is None:
        return ""
    if domain not in _HANDLER_DOMAINS:
        raise ValueError(
            f"custom effect handlers are not permitted for action domain {domain.value}"
        )
    if effect_verifier is None:
        raise ValueError("custom effect handlers require an independent effect_verifier")
    return _callable_name(effect_handler)


#: How long an action may show no sign of life before it is treated as wedged.
#:
#: This used to be a ceiling on an action's TOTAL duration, which made it a
#: work budget for anything slow. It killed a sixty-question form at round
#: forty-one and threw away every answer, because a questionnaire's length is
#: not knowable in advance — page loads, question counts and a remote site's
#: latency are all outside the caller's control, and no constant here can know
#: them.
#:
#: Silence is a different claim from duration, and it is the one this number
#: can actually make: ten minutes with nothing happening is a wedge whatever
#: the work is. An action that keeps reporting progress is not wedged, and is
#: bounded instead by the total its caller declared — a figure derived from the
#: work requested rather than chosen here.
SILENCE_CEILING_S = 600.0


def _coerce_execution_timeout(value: float | None) -> float:
    if value is None:
        return 60.0
    try:
        timeout_s = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("execution_timeout_s must be numeric") from exc
    if timeout_s <= 0:
        raise ValueError("execution_timeout_s must be positive")
    return timeout_s


async def _invoke_effect_handler(
    handler: EffectHandler,
    context: Mapping[str, Any],
    *,
    timeout_s: float,
) -> dict[str, Any]:
    """Run one effect handler under a watchdog it can talk to.

    A handler is handed `report_progress` in its context. Calling it says "I am
    still working", and resets the silence clock. A handler that never calls it
    behaves exactly as before: it has `SILENCE_CEILING_S` to finish. A handler
    that does call it runs until the total its caller declared, which is the
    only figure in the system derived from the work actually requested.

    Ignoring the callback is fine. Handlers that finish quickly have nothing to
    report, and nothing about them changes.
    """

    last_sign_of_life = time.monotonic()

    def report_progress(_note: str = "") -> None:
        nonlocal last_sign_of_life
        last_sign_of_life = time.monotonic()

    handler_context = dict(context)
    handler_context["report_progress"] = report_progress

    async def invoke() -> Mapping[str, Any]:
        if inspect.iscoroutinefunction(handler):
            value = await handler(dict(handler_context))
        else:
            value = await asyncio.to_thread(handler, dict(handler_context))
            if inspect.isawaitable(value):
                value = await value
        if not isinstance(value, Mapping):
            raise TypeError("effect handler must return a mapping")
        return value

    started = time.monotonic()

    async def _gathered() -> list[Any]:
        return await asyncio.gather(invoke(), return_exceptions=True)

    # Tracked, not raw: a task nobody owns is invisible to shutdown and to the
    # runtime's task census. The tracker takes a COROUTINE, so the gather is
    # wrapped in one — handing it the gather future itself is not a coroutine
    # and the task never ran, which turned every governed effect into
    # failed_recoverable.
    from core.utils.task_tracker import get_task_tracker

    work = get_task_tracker().track(_gathered(), name="action_executor.effect")
    while True:
        silent_for = time.monotonic() - last_sign_of_life
        total_left = timeout_s - (time.monotonic() - started)
        window = min(SILENCE_CEILING_S - silent_for, total_left)
        if window <= 0:
            work.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await work
            raise TimeoutError(
                f"effect handler {_callable_name(handler)} exceeded its "
                + (
                    "declared total"
                    if total_left <= 0
                    else f"silence ceiling ({SILENCE_CEILING_S:.0f}s with no progress)"
                )
            )
        try:
            completed = await asyncio.wait_for(asyncio.shield(work), timeout=window)
            break
        except TimeoutError:
            # The window closed. Whether that is a wedge depends on whether
            # anything was reported while it was open, which the next pass of
            # the loop works out.
            continue
    raw_result = completed[0]
    if isinstance(raw_result, asyncio.CancelledError):
        raise raw_result
    if isinstance(raw_result, BaseException):
        raise RuntimeError(
            f"effect handler {_callable_name(handler)} failed: {raw_result}"
        ) from raw_result
    return _coerce_result(raw_result)


def _callable_name(value: Callable[..., Any]) -> str:
    module = str(getattr(value, "__module__", "") or "")
    qualname = str(
        getattr(value, "__qualname__", "")
        or getattr(value, "__name__", "")
        or type(value).__qualname__
    )
    return f"{module}.{qualname}".strip(".")[:240]


def _coerce_expectation(
    value: ActionExpectation | Mapping[str, Any] | None,
) -> ActionExpectation | None:
    if value is None:
        return value
    if isinstance(value, ActionExpectation):
        raw = value.to_dict()
    elif isinstance(value, Mapping):
        raw = dict(value)
    else:
        raise TypeError("expectation must be ActionExpectation, mapping, or None")
    return ActionExpectation(
        objective=str(raw.get("objective") or "")[:1000],
        acceptance_criteria=_string_list(raw.get("acceptance_criteria")),
        required_evidence=_string_list(raw.get("required_evidence")),
        required_evidence_present=_string_list(raw.get("required_evidence_present")),
        semantic_predicates=[
            semantic_predicate_from_mapping(item)
            for item in list(raw.get("semantic_predicates") or [])[:64]
            if isinstance(item, Mapping)
        ],
        user_visible_effect=(
            str(raw.get("user_visible_effect"))[:1000]
            if raw.get("user_visible_effect") is not None
            else None
        ),
        repair_hint=str(raw.get("repair_hint") or "")[:1000],
        rollback_hint=str(raw.get("rollback_hint") or "")[:1000],
        allow_partial=bool(raw.get("allow_partial", True)),
    )


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple, set)):
        return []
    return [str(item)[:500] for item in list(value)[:64] if str(item).strip()]


def _coerce_action_id(value: str | None) -> str:
    if value is None:
        return f"action-{uuid.uuid4()}"
    text = str(value).strip()
    if not _ACTION_ID_PATTERN.fullmatch(text):
        raise ValueError(
            "action_id must be 1-160 letters, digits, dot, colon, dash, or underscore"
        )
    return text


def _coerce_path_param(value: Any, label: str) -> str | Path:
    if not isinstance(value, (str, Path)):
        raise TypeError(f"file action {label} must be a string or Path")
    if not str(value).strip():
        raise ValueError(f"file action {label} must not be empty")
    return value


def _safe_action_summary(action_name: str, params: Mapping[str, Any]) -> str:
    summarized: dict[str, Any] = {}
    for key, value in params.items():
        key_text = str(key)
        if any(marker in key_text.casefold() for marker in _SENSITIVE_PARAM_MARKERS):
            summarized[key_text] = "[REDACTED]"
        elif key_text in {"content", "payload", "script", "text"}:
            length = len(value) if hasattr(value, "__len__") else 0
            summarized[key_text] = f"<{type(value).__name__}:{length}>"
        elif isinstance(value, Mapping):
            summarized[key_text] = f"<mapping:{len(value)}>"
        elif isinstance(value, (list, tuple, set)):
            summarized[key_text] = _safe_sequence_summary(value)
        elif key_text.casefold() in {"uri", "url"}:
            summarized[key_text] = _safe_url_summary(str(value))
        else:
            summarized[key_text] = str(value)[:160]
    encoded = json.dumps(summarized, sort_keys=True, default=str)
    return f"{action_name} params={encoded}"[:1000]


def _transaction_outcome(status: str, ok: bool) -> str:
    if ok and status == SkillStatus.SUCCESS_VERIFIED.value:
        return "success"
    if status in {
        SkillStatus.SUCCESS_UNVERIFIED.value,
        SkillStatus.PARTIAL_SUCCESS.value,
    }:
        return "partial"
    return "failure"


def _safe_sequence_summary(value: Any) -> list[Any]:
    summarized: list[Any] = []
    redact_next = False
    for raw_item in list(value)[:16]:
        if isinstance(raw_item, Mapping):
            summarized.append(_safe_nested_mapping_summary(raw_item))
            continue
        item = str(raw_item)
        lowered = item.casefold()
        if redact_next:
            summarized.append("[REDACTED]")
            redact_next = False
            continue
        if lowered.startswith(("http://", "https://")):
            summarized.append(_safe_url_summary(item))
            continue
        if any(marker in lowered for marker in _SENSITIVE_PARAM_MARKERS):
            if "=" in item:
                summarized.append(item.split("=", 1)[0][:80] + "=[REDACTED]")
            elif item.lstrip().startswith("-"):
                summarized.append(item[:80])
                redact_next = True
            else:
                summarized.append("[REDACTED]")
            continue
        summarized.append(item[:80])
    return summarized


def _safe_nested_mapping_summary(value: Mapping[Any, Any]) -> dict[str, Any]:
    summarized: dict[str, Any] = {}
    for raw_key, raw_value in list(value.items())[:32]:
        key = str(raw_key)[:80]
        lowered = key.casefold()
        if lowered == "value" or any(
            marker in lowered for marker in _SENSITIVE_PARAM_MARKERS
        ):
            summarized[key] = "[REDACTED]"
        elif lowered in {"content", "payload", "script", "text"}:
            length = len(raw_value) if hasattr(raw_value, "__len__") else 0
            summarized[key] = f"<{type(raw_value).__name__}:{length}>"
        elif lowered in {"uri", "url"}:
            summarized[key] = _safe_url_summary(str(raw_value))
        elif isinstance(raw_value, Mapping):
            summarized[key] = _safe_nested_mapping_summary(raw_value)
        elif isinstance(raw_value, (list, tuple, set)):
            summarized[key] = _safe_sequence_summary(raw_value)
        else:
            summarized[key] = str(raw_value)[:160]
    return summarized


def _safe_url_summary(value: str) -> str:
    try:
        parsed = urllib.parse.urlsplit(value)
        host = parsed.hostname or ""
        port = f":{parsed.port}" if parsed.port else ""
        return urllib.parse.urlunsplit(
            (parsed.scheme, host + port, parsed.path, "", "")
        )[:240]
    except ValueError:
        return "<invalid-url>"


def _append_error(existing: Any, new_error: str) -> str:
    current = str(existing or "").strip()
    addition = str(new_error or "").strip()
    if not current:
        return addition[:1000]
    if not addition or addition in current:
        return current[:1000]
    return f"{current}; {addition}"[:1000]


def _external_transaction_result(
    transaction: Mapping[str, Any] | None,
    *,
    action_id: str,
    request_digest: str,
    expectation: ActionExpectation,
    preaction_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    if not isinstance(transaction, Mapping):
        return None
    state = str(transaction.get("state") or "")
    if state not in {
        "SUCCEEDED",
        "FAILED",
        "FAILED_PRE_DISPATCH",
        "ABSTAINED",
        "UNKNOWN_EFFECT",
    }:
        return None
    stored = transaction.get("result")
    replay_payload = (
        stored.get("replay_payload")
        if isinstance(stored, Mapping)
        else None
    )
    result = dict(replay_payload) if isinstance(replay_payload, Mapping) else {}
    if state == "ABSTAINED":
        result.update(
            {
                "ok": False,
                "status": SkillStatus.BLOCKED_BY_POLICY.value,
                "error": "latent_cortex_declined_external_execution",
                "transport_succeeded": False,
                "effect_verified": False,
                "manual_reconciliation_required": False,
            }
        )
    elif state == "UNKNOWN_EFFECT":
        result.update(
            {
                "ok": False,
                "status": SkillStatus.FAILED_RECOVERABLE.value,
                "error": "external_execution_effect_unknown_requires_reconciliation",
                "effect_verified": False,
                "manual_reconciliation_required": True,
            }
        )
    result.update(
        {
            "action_id": action_id,
            "request_digest": request_digest,
            "will_receipt_id": str(
                transaction.get("dispatch_authorization_receipt_id")
                or (transaction.get("offer") or {}).get("will_receipt_id")
                or result.get("will_receipt_id")
                or ""
            ),
            "action_expectation": expectation.to_dict(),
            "external_execution_transaction": dict(transaction),
            "external_execution_replayed": True,
            "retry_safe": False,
        }
    )
    if preaction_receipt is not None:
        result["preaction_cortex"] = dict(preaction_receipt)
    return result


async def _heal_external_post_action_receipt(
    *,
    coordinator: Any,
    offer: Mapping[str, Any],
    transaction: Mapping[str, Any],
    replay: dict[str, Any],
    action_id: str,
    request_digest: str,
    expectation: ActionExpectation,
) -> dict[str, Any]:
    stored_result = transaction.get("result")
    authoritative_receipt_id = str(
        (
            stored_result.get("post_action_receipt_id")
            if isinstance(stored_result, Mapping)
            else ""
        )
        or replay.get("post_action_receipt_id")
        or ""
    )
    if authoritative_receipt_id:
        try:
            expected_receipt_id = _external_post_receipt_id(
                action_id,
                request_digest,
            )
            if authoritative_receipt_id != expected_receipt_id:
                raise ValueError(
                    "external transaction linked a noncanonical post-action receipt"
                )
            transaction_replay = (
                stored_result.get("replay_payload")
                if isinstance(stored_result, Mapping)
                else {}
            )
            expected_receipt_sha256 = (
                transaction_replay.get("post_action_receipt_sha256")
                if isinstance(transaction_replay, Mapping)
                else None
            )
            receipt = _validated_persisted_external_post_receipt(
                get_post_action_receipt_store().get_receipt(
                    authoritative_receipt_id
                ),
                action_id=action_id,
                request_digest=request_digest,
                will_receipt_id=str(
                    transaction.get("dispatch_authorization_receipt_id")
                    or (transaction.get("offer") or {}).get("will_receipt_id")
                    or ""
                ),
                expected_sha256=expected_receipt_sha256,
            )
            replay["post_action_receipt_id"] = receipt.receipt_id
            replay["post_action_output_hash"] = receipt.output_hash
            replay["receipt_persisted"] = True
            replay["post_action_receipt_pending"] = False
            replay.pop("_post_action_recovery_contract", None)
        except _ACTION_EXECUTOR_RECOVERABLE_ERRORS as exc:
            record_degradation(
                "action_executor.external_execution",
                exc,
                action=(
                    "refused to claim a linked post-action receipt without "
                    "matching durable store evidence"
                ),
                severity="degraded",
                enforce_failure_policy=False,
            )
            replay["receipt_persisted"] = False
            replay["post_action_receipt_pending"] = True
            replay["manual_reconciliation_required"] = True
            replay["error"] = _append_error(
                replay.get("error"),
                f"post_action_receipt_validation_failed:{type(exc).__name__}",
            )
        return replay

    contract = replay.get("_post_action_recovery_contract")
    if not isinstance(contract, Mapping):
        replay["receipt_persisted"] = False
        replay["post_action_receipt_pending"] = True
        replay["manual_reconciliation_required"] = True
        replay.pop("_post_action_recovery_contract", None)
        return replay
    try:
        receipt = PostActionReceipt(**dict(contract))
        expected_receipt_id = _external_post_receipt_id(
            action_id,
            request_digest,
        )
        expected_will_id = str(
            transaction.get("dispatch_authorization_receipt_id")
            or (transaction.get("offer") or {}).get("will_receipt_id")
            or ""
        )
        if (
            receipt.receipt_id != expected_receipt_id
            or receipt.action_id != action_id
            or receipt.request_digest != request_digest
            or receipt.will_receipt_id != expected_will_id
        ):
            raise ValueError(
                "external post-action recovery contract identity differs"
            )
        store = get_post_action_receipt_store()
        existing = store.get_receipt(receipt.receipt_id)
        if existing is None:
            await store.record_async(receipt)
        persisted = _validated_persisted_external_post_receipt(
            store.get_receipt(receipt.receipt_id),
            action_id=action_id,
            request_digest=request_digest,
            will_receipt_id=expected_will_id,
            expected_contract=receipt.to_dict(),
        )
        linked = await asyncio.to_thread(
            coordinator.link_post_action_receipt,
            offer=offer,
            persisted_receipt=persisted.to_dict(),
            receipt_store=store,
        )
        healed = _external_transaction_result(
            linked,
            action_id=action_id,
            request_digest=request_digest,
            expectation=expectation,
        )
        if healed is None:
            raise ValueError("external receipt healing lost terminal state")
        healed["post_action_output_hash"] = receipt.output_hash
        healed["external_execution_receipt_linked"] = True
        healed.pop("_post_action_recovery_contract", None)
        return healed
    except _ACTION_EXECUTOR_RECOVERABLE_ERRORS as exc:
        record_degradation(
            "action_executor.external_execution",
            exc,
            action=(
                "preserved terminal effect state while post-action receipt "
                "self-healing remained pending"
            ),
            severity="degraded",
            enforce_failure_policy=False,
        )
        replay["receipt_persisted"] = False
        replay["post_action_receipt_pending"] = True
        replay["manual_reconciliation_required"] = True
        replay["error"] = _append_error(
            replay.get("error"),
            f"post_action_receipt_recovery_failed:{type(exc).__name__}",
        )
        replay.pop("_post_action_recovery_contract", None)
        return replay


async def _finalize_approved_no_effect(
    **kwargs: Any,
) -> dict[str, Any]:
    result = kwargs["result"]
    action_name = str(kwargs["action_name"])
    try:
        return await _finalize_approved_no_effect_impl(**kwargs)
    except _ACTION_EXECUTOR_RECOVERABLE_ERRORS as exc:
        record_degradation(
            "action_executor.external_execution",
            exc,
            action=(
                "surfaced approved no-effect governance finalization failure "
                f"for {action_name}"
            ),
            severity="degraded",
            enforce_failure_policy=False,
        )
        result["ok"] = False
        result["error"] = _append_error(
            result.get("error"),
            f"no_effect_governance_finalization_failed:{type(exc).__name__}",
        )
        result["receipt_persisted"] = False
        result["retry_safe"] = False
        return result


async def _finalize_approved_no_effect_impl(
    *,
    result: dict[str, Any],
    will: Any,
    will_receipt_id: str,
    domain: ActionDomain,
    action_name: str,
    source: str,
    expectation: ActionExpectation,
    action_id: str,
    request_digest: str,
    rollback_target: str | None,
    coordinator: Any,
    offer: Mapping[str, Any] | None,
) -> dict[str, Any]:
    body_service = BodyStateService.get()
    welfare_service = WelfareState.get()
    transaction = WelfareTransaction.begin(
        domain=domain.value,
        action=f"{action_name} ({source})",
        welfare_before=welfare_service.last_outputs,
        body_before=body_service.snapshot(),
        predicted_welfare_delta=None,
        will_receipt_id=will_receipt_id,
    )
    tx_record = transaction.complete(
        outcome=_transaction_outcome(
            str(result.get("status") or SkillStatus.FAILED_RECOVERABLE.value),
            False,
        ),
        welfare_after=welfare_service.last_outputs,
        body_after=body_service.snapshot(),
        error=str(result.get("error") or ""),
    )
    will.record_outcome(will_receipt_id, tx_record)
    result.update(
        {
            "will_receipt_id": will_receipt_id,
            "welfare_transaction_id": transaction.tx_id,
            "welfare_transaction_completed": True,
            "transport_succeeded": False,
            "effect_verified": False,
            "manual_reconciliation_required": False,
        }
    )
    receipt_id = (
        _external_post_receipt_id(action_id, request_digest)
        if offer is not None
        else f"post-{uuid.uuid4()}"
    )
    receipt = PostActionReceipt(
        receipt_id=receipt_id,
        will_receipt_id=will_receipt_id,
        executor_name=action_name,
        actual_outcome=tx_record.outcome,
        output_hash=_stable_digest(result),
        error_status=str(result.get("error") or ""),
        welfare_transaction_id=transaction.tx_id,
        body_delta=tx_record.body_delta,
        memory_delta={},
        rollback_target=rollback_target,
        status=str(result.get("status") or SkillStatus.FAILED_RECOVERABLE.value),
        effect_verified=False,
        action_expectation=_bounded_receipt_mapping(expectation.to_dict()),
        verification_evidence={},
        action_id=action_id,
        domain=domain.value,
        source=str(source or "unknown")[:240],
        request_digest=request_digest,
        transport_succeeded=False,
        retry_safe=bool(result.get("retry_safe", False)),
        manual_reconciliation_required=False,
        welfare_transaction_completed=True,
    )
    if coordinator is not None and offer is not None:
        await asyncio.to_thread(
            coordinator.fail_preparation,
            offer=offer,
            result=result,
        )
        await asyncio.to_thread(
            coordinator.stage_post_action_receipt,
            offer=offer,
            receipt_contract=receipt.to_dict(),
        )
    store = get_post_action_receipt_store()
    if store.get_receipt(receipt.receipt_id) is None:
        await store.record_async(receipt)
    persisted = _validated_persisted_external_post_receipt(
        store.get_receipt(receipt.receipt_id),
        action_id=action_id,
        request_digest=request_digest,
        will_receipt_id=will_receipt_id,
        expected_contract=receipt.to_dict(),
    )
    result.update(
        {
            "post_action_receipt_id": receipt.receipt_id,
            "post_action_output_hash": receipt.output_hash,
            "receipt_persisted": True,
            "post_action_receipt_pending": False,
        }
    )
    if coordinator is not None and offer is not None:
        linked = await asyncio.to_thread(
            coordinator.link_post_action_receipt,
            offer=offer,
            persisted_receipt=persisted.to_dict(),
            receipt_store=store,
        )
        result["external_execution_transaction"] = linked
        result["external_execution_receipt_linked"] = True
    return result


def _external_dispatch_task_id() -> str:
    task = asyncio.current_task()
    if task is None:
        return f"task-unknown-{uuid.uuid4().hex}"
    name = str(task.get_name() or "unnamed")[:96]
    return f"task-{id(task):x}:{name}"


def _external_post_receipt_id(action_id: str, request_digest: str) -> str:
    identity = f"{action_id}\0{request_digest}".encode()
    return f"post-external-{hashlib.sha256(identity).hexdigest()[:32]}"


def _canonical_mapping_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validated_persisted_external_post_receipt(
    receipt: PostActionReceipt | None,
    *,
    action_id: str,
    request_digest: str,
    will_receipt_id: str,
    expected_contract: Mapping[str, Any] | None = None,
    expected_sha256: Any = None,
) -> PostActionReceipt:
    if receipt is None:
        raise ValueError("post-action receipt is absent from the durable store")
    contract = receipt.to_dict()
    expected_id = _external_post_receipt_id(action_id, request_digest)
    if (
        receipt.receipt_id != expected_id
        or receipt.action_id != action_id
        or receipt.request_digest != request_digest
        or receipt.will_receipt_id != will_receipt_id
    ):
        raise ValueError("persisted post-action receipt identity differs")
    if expected_contract is not None and contract != dict(expected_contract):
        raise ValueError("persisted post-action receipt content differs")
    actual_sha256 = _canonical_mapping_sha256(contract)
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        if expected_sha256 is not None:
            raise ValueError("transaction post-action receipt digest is invalid")
    elif actual_sha256 != expected_sha256:
        raise ValueError("persisted post-action receipt digest differs")
    return receipt


async def _await_external_closure(
    operation: Any,
    *,
    coordinator: Any,
    offer: Mapping[str, Any] | None,
    dispatch_attempt_id: str,
    heartbeat: asyncio.Task[None] | None,
    result: Mapping[str, Any],
    cancellation_reason: str,
) -> Any:
    try:
        return await operation
    except asyncio.CancelledError:
        await _abandon_external_dispatch(
            coordinator=coordinator,
            offer=offer,
            dispatch_attempt_id=dispatch_attempt_id,
            effect_may_have_occurred=True,
            reason=cancellation_reason,
            result=result,
        )
        await _stop_external_dispatch_heartbeat(heartbeat)
        raise


async def _renew_external_dispatch_lease(
    *,
    coordinator: Any,
    offer: Mapping[str, Any],
    dispatch_attempt_id: str,
) -> None:
    task = asyncio.current_task()
    while task is not None and not task.cancelled():
        await asyncio.sleep(20.0)
        try:
            await asyncio.to_thread(
                coordinator.renew_dispatch,
                offer=offer,
                dispatch_attempt_id=dispatch_attempt_id,
            )
        except _ACTION_EXECUTOR_RECOVERABLE_ERRORS as exc:
            record_degradation(
                "action_executor.external_execution",
                exc,
                action=(
                    "stopped renewing an external dispatch lease; expiry will "
                    "force reconciliation rather than duplicate execution"
                ),
                severity="degraded",
                enforce_failure_policy=False,
            )
            return


async def _stop_external_dispatch_heartbeat(
    task: asyncio.Task[None] | None,
) -> None:
    if task is None:
        return
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


async def _abandon_external_dispatch(
    *,
    coordinator: Any,
    offer: Mapping[str, Any] | None,
    dispatch_attempt_id: str,
    effect_may_have_occurred: bool,
    reason: str,
    result: Mapping[str, Any] | None = None,
) -> None:
    if coordinator is None or offer is None or not dispatch_attempt_id:
        return
    cleanup = get_task_tracker().create_task(
        asyncio.to_thread(
            coordinator.abandon_dispatch,
            offer=offer,
            dispatch_attempt_id=dispatch_attempt_id,
            effect_may_have_occurred=effect_may_have_occurred,
            reason=reason,
            result=result,
        )
    )
    try:
        await asyncio.shield(cleanup)
    except asyncio.CancelledError:
        await asyncio.shield(cleanup)
        raise
    except _ACTION_EXECUTOR_RECOVERABLE_ERRORS as exc:
        record_degradation(
            "action_executor.external_execution",
            exc,
            action=(
                "marked the task-scoped dispatch owner abandoned in memory; "
                "the next lookup will reconcile its durable state"
            ),
            severity="degraded",
            enforce_failure_policy=False,
        )


async def _complete_external_execution_transaction(
    *,
    coordinator: Any,
    offer: Mapping[str, Any] | None,
    transaction: Mapping[str, Any] | None,
    dispatch_attempt_id: str,
    result: dict[str, Any],
    action_name: str,
) -> None:
    if coordinator is None or offer is None:
        return
    if not dispatch_attempt_id:
        result["external_execution_transaction"] = dict(transaction or {})
        return
    try:
        completed = await asyncio.to_thread(
            coordinator.complete,
            offer=offer,
            result=result,
            dispatch_attempt_id=dispatch_attempt_id,
        )
        result["external_execution_transaction"] = completed
    except _ACTION_EXECUTOR_RECOVERABLE_ERRORS as exc:
        effect_may_have_occurred = (
            result.get("transport_succeeded") is True
            or result.get("retry_safe") is not True
        )
        record_degradation(
            "action_executor.external_execution",
            exc,
            action=(
                f"effect lane completed but external execution closure failed "
                f"for {action_name}"
            ),
            severity="degraded",
            enforce_failure_policy=False,
        )
        transport_succeeded = result.get("transport_succeeded") is True
        result["ok"] = False
        result["status"] = (
            SkillStatus.PARTIAL_SUCCESS.value
            if transport_succeeded
            else SkillStatus.FAILED_RECOVERABLE.value
        )
        result["error"] = _append_error(
            result.get("error"),
            f"external_execution_completion_failed:{type(exc).__name__}",
        )
        result["manual_reconciliation_required"] = transport_succeeded
        result["retry_safe"] = False
        result["external_execution_transaction"] = dict(transaction or {})
        await _abandon_external_dispatch(
            coordinator=coordinator,
            offer=offer,
            dispatch_attempt_id=dispatch_attempt_id,
            effect_may_have_occurred=effect_may_have_occurred,
            reason=(
                "external_execution_completion_failed:"
                f"{type(exc).__name__}"
            ),
            result=result,
        )


def _stable_digest(value: Any) -> str:
    digest = hashlib.sha256()
    _update_digest(digest, value, seen=set(), depth=0)
    return "sha256:" + digest.hexdigest()


def _update_digest(
    digest: Any,
    value: Any,
    *,
    seen: set[int],
    depth: int,
) -> None:
    if depth > 48:
        digest.update(b"<max-depth>")
        return
    if value is None or isinstance(value, (bool, int, float)):
        digest.update(json.dumps(value, sort_keys=True).encode("utf-8"))
        return
    if isinstance(value, str):
        encoded = value.encode("utf-8", errors="replace")
        digest.update(f"str:{len(encoded)}:".encode("ascii"))
        for offset in range(0, len(encoded), 1024 * 1024):
            digest.update(encoded[offset : offset + 1024 * 1024])
        return
    if isinstance(value, (bytes, bytearray, memoryview)):
        payload = bytes(value)
        digest.update(f"bytes:{len(payload)}:".encode("ascii"))
        digest.update(payload)
        return
    container_id = id(value)
    if container_id in seen:
        digest.update(b"<cycle>")
        return
    if isinstance(value, Mapping):
        seen.add(container_id)
        digest.update(b"{")
        for key in sorted(value, key=lambda item: (type(item).__qualname__, str(item))):
            _update_digest(digest, key, seen=seen, depth=depth + 1)
            _update_digest(digest, value[key], seen=seen, depth=depth + 1)
        digest.update(b"}")
        seen.remove(container_id)
        return
    if isinstance(value, (list, tuple)):
        seen.add(container_id)
        digest.update(b"[")
        for item in value:
            _update_digest(digest, item, seen=seen, depth=depth + 1)
        digest.update(b"]")
        seen.remove(container_id)
        return
    if isinstance(value, (set, frozenset)):
        seen.add(container_id)
        digest.update(b"<set>")
        for item in sorted(value, key=lambda entry: (type(entry).__qualname__, str(entry))):
            _update_digest(digest, item, seen=seen, depth=depth + 1)
        seen.remove(container_id)
        return
    _update_digest(
        digest,
        f"{type(value).__module__}.{type(value).__qualname__}:{value}",
        seen=seen,
        depth=depth + 1,
    )


def _bounded_receipt_mapping(value: Any) -> dict[str, Any]:
    state = {"items": 0, "truncated": False}
    bounded = _bounded_receipt_value(value, state=state, depth=0)
    if isinstance(bounded, dict):
        result = bounded
    else:
        result = {"value": bounded}
    if state["truncated"]:
        result = dict(result)
        result["_truncated"] = True
        result["_original_digest"] = _stable_digest(value)
    return result


def _bounded_receipt_value(
    value: Any,
    *,
    state: dict[str, Any],
    depth: int,
) -> Any:
    state["items"] = int(state["items"]) + 1
    if depth > 8 or int(state["items"]) > 512:
        state["truncated"] = True
        return "<truncated>"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if len(value) <= 2048:
            return value
        state["truncated"] = True
        return {
            "prefix": value[:2048],
            "characters": len(value),
            "digest": _stable_digest(value),
        }
    if isinstance(value, (bytes, bytearray, memoryview)):
        payload = bytes(value)
        return {
            "type": "bytes",
            "bytes": len(payload),
            "digest": _stable_digest(payload),
        }
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= 128 or int(state["items"]) > 512:
                state["truncated"] = True
                break
            result[str(key)[:240]] = _bounded_receipt_value(
                item,
                state=state,
                depth=depth + 1,
            )
        return result
    if isinstance(value, (list, tuple, set, frozenset)):
        items = list(value)
        if len(items) > 128:
            state["truncated"] = True
        return [
            _bounded_receipt_value(item, state=state, depth=depth + 1)
            for item in items[:128]
            if int(state["items"]) <= 512
        ]
    text = str(value)
    if len(text) > 2048:
        state["truncated"] = True
        text = text[:2048]
    return text


__all__ = ["ActionExecutor"]
