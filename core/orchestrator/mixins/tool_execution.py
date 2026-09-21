"""Tool Execution Mixin for RobustOrchestrator.
Extracts browser task and tool execution logic.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from core.collective.delegator import swarm_debate_failure_reason
from core.container import ServiceContainer
from core.executive.execution_policy import (
    classify_execution_risk,
    resolve_execution_effect_scope,
)
from core.executive.standing_authority import (
    AUTONOMOUS_AUTHORITY_ORIGINS,
    PUBLIC_RESEARCH_TOOLS,
    USER_FACING_AUTHORITY_ORIGINS,
    coerce_authority_origin,
    context_has_user_authority,
    get_standing_authority_manager,
)
from core.runtime.errors import record_degradation
from core.runtime.executors import off_the_loop
from core.runtime.task_ownership import drain_owned_awaitable
from core.verify.work_ledger import record_work

logger = logging.getLogger(__name__)
_TOOL_EXECUTION_RECOVERABLE_ERRORS = (
    ImportError,
    AttributeError,
    RuntimeError,
    OSError,
    ConnectionError,
    TimeoutError,
    TypeError,
    ValueError,
)
from core.utils.queues import USER_FACING_ORIGINS as _CANONICAL_USER_FACING_ORIGINS

# Standing-authority origins use the underscore internal convention
# (desktop_ui); the cross-module user-facing contract (7 other modules,
# test_live_runtime_surface_regressions) uses the hyphenated display forms
# (desktop-ui, native-shell, ws). Union both so a desktop turn is
# recognized as foreground no matter which convention reaches this gate.
_USER_FACING_TOOL_ORIGINS = frozenset(
    USER_FACING_AUTHORITY_ORIGINS | _CANONICAL_USER_FACING_ORIGINS
)
_READ_ONLY_WEB_TOOLS = PUBLIC_RESEARCH_TOOLS
_AUTONOMOUS_WEB_TOOL_ORIGINS = AUTONOMOUS_AUTHORITY_ORIGINS
_UNSAFE_AUTONOMOUS_WEB_TOOL_MARKERS = {
    "api key",
    "brute force",
    "bypass login",
    "credential",
    "credentials",
    "ddos",
    "deanonymize",
    "dox",
    "doxx",
    "exfiltrate",
    "exploit",
    "malware",
    "password",
    "phishing",
    "private key",
    "ransomware",
    "session cookie",
    "steal",
    "token dump",
    "worm",
}


def _record_tool_degradation(
    error: BaseException,
    *,
    action: str,
    severity: str = "warning",
) -> None:
    record_degradation(
        "tool_execution",
        error,
        severity=severity,
        action=action,
    )


class ToolExecutionMixin:
    """Handles tool execution with constitutional gating, episodic recording, and tool learning."""

    _current_objective: str
    _emit_thought_stream: Any
    _fire_and_forget: Any
    hephaestus: Any
    liquid_state: Any
    router: Any
    status: Any
    stealth_mode: Any
    swarm: Any

    @staticmethod
    def _normalize_tool_origin(origin: Any) -> str:
        return str(origin or "").strip().lower().replace("-", "_")

    @classmethod
    def _coerce_tool_origin(cls, origin: Any) -> str:
        resolved = coerce_authority_origin(origin)
        return "" if resolved == "unknown" and not str(origin or "").strip() else resolved

    def _resolve_tool_origin(
        self,
        *,
        explicit_origin: Any = None,
        payload_context: dict[str, Any] | None = None,
    ) -> str:
        candidates = [
            explicit_origin,
            (payload_context or {}).get("origin") if isinstance(payload_context, dict) else None,
            getattr(self, "_current_origin", ""),
            getattr(getattr(getattr(self, "state", None), "cognition", None), "current_origin", ""),
        ]
        for candidate in candidates:
            resolved = self._coerce_tool_origin(candidate)
            if resolved:
                return resolved
        return "unknown"

    @staticmethod
    def _tool_effect_scope(tool_name: Any, args: dict[str, Any] | None = None) -> str:
        """Return the conservative effect scope used by the autonomy gate."""

        return str(resolve_execution_effect_scope(tool_name, args))

    @staticmethod
    def _safe_autonomous_web_research_tool(
        tool_name: Any,
        args: dict[str, Any],
        origin: Any,
        payload_context: dict[str, Any] | None = None,
    ) -> bool:
        name = str(tool_name or "").strip().lower()
        if name not in _READ_ONLY_WEB_TOOLS:
            return False
        normalized_origin = ToolExecutionMixin._coerce_tool_origin(origin)
        if normalized_origin not in _AUTONOMOUS_WEB_TOOL_ORIGINS:
            return False
        ctx = payload_context or {}
        text = " ".join(
            str(part or "").lower()
            for part in (
                (args or {}).get("query"),
                (args or {}).get("q"),
                ctx.get("objective"),
                ctx.get("message"),
                ctx.get("reason"),
            )
        )
        if not text.strip():
            return False
        return not any(marker in text for marker in _UNSAFE_AUTONOMOUS_WEB_TOOL_MARKERS)

    async def run_browser_task(self, url: str, task: str) -> Any:
        """Formalized browser task execution via skill router.
        Browser work goes through the same governed tool path as every other skill.
        """
        logger.info("🌐 Initiating Browser Task: %s @ %s", task, url)
        return await self.execute_tool("browser", {"url": url, "task": task})

    async def execute_tool(
        self,
        tool_name: str,
        args: dict[str, Any],
        **kwargs: Any,
    ) -> Any:
        """Execute a single tool with feedback reporting, episodic recording, and tool learning"""
        _start = time.time()
        _constitution = None
        _tool_handle = None
        _constitutional_runtime_live = False
        _standing_authority = None
        _standing_authority_token: str | None = None
        _standing_authority_closed = False
        kwargs = dict(kwargs or {})
        _origin = self._resolve_tool_origin(
            explicit_origin=kwargs.get("origin"),
            payload_context=kwargs.get("payload_context"),
        )
        kwargs.setdefault("origin", _origin)
        _constitutional_runtime_live = (
            ServiceContainer.has("executive_core")
            or ServiceContainer.has("aura_kernel")
            or ServiceContainer.has("kernel_interface")
            or bool(getattr(ServiceContainer, "_registration_locked", False))
        )
        router = getattr(self, "router", None)
        governance_owner = getattr(router, "owns_tool_execution_governance", None)
        _router_owns_governance = bool(
            callable(governance_owner) and governance_owner(tool_name)
        )

        def _record_coding_tool_event(result: Any, *, success: bool, error: str = "") -> None:
            nonlocal _standing_authority_closed
            if (
                _standing_authority is not None
                and _standing_authority_token
                and not _standing_authority_closed
            ):
                try:
                    standing_closure = _standing_authority.finalize_child_lease(
                        _standing_authority_token,
                        success=success,
                        result=result,
                        error=error,
                    )
                    _standing_authority_closed = bool(standing_closure.get("closed"))
                except _TOOL_EXECUTION_RECOVERABLE_ERRORS as _authority_exc:
                    _record_tool_degradation(
                        _authority_exc,
                        action="returned tool result after standing-authority closure degraded",
                        severity="error",
                    )
            # The work ledger is what the fabrication audit checks a later
            # claim against. Recording here — the one point every exit path
            # passes through, with the success verdict already decided —
            # means a turn's account of its own work cannot drift from what
            # the tool path actually did.
            record_work(tool_name, ok=success, detail=_origin)
            try:
                from core.runtime.coding_session_memory import get_coding_session_memory

                get_coding_session_memory().record_tool_event(
                    tool_name=tool_name,
                    args=args,
                    result=result,
                    objective=self._current_objective or "",
                    origin=_origin,
                    success=success,
                    error=error,
                )
            except _TOOL_EXECUTION_RECOVERABLE_ERRORS as _coding_exc:
                _record_tool_degradation(
                    _coding_exc,
                    action="continued tool execution after coding-session memory event failed",
                )
                logger.error("Coding session tool recording failed: %s", _coding_exc, exc_info=True)

        async def _finish_constitutional_tool_execution(
            result: dict[str, Any],
            *,
            success: bool,
            error: str = "",
        ) -> bool:
            if not (_constitution and _tool_handle):
                return True
            try:
                await _constitution.finish_tool_execution(
                    _tool_handle,
                    result=result,
                    success=success,
                    duration_ms=(time.time() - _start) * 1000,
                    error=error or None,
                )
                return True
            except _TOOL_EXECUTION_RECOVERABLE_ERRORS as _finish_exc:
                _record_tool_degradation(
                    _finish_exc,
                    action="returned tool result after constitutional completion bookkeeping failed",
                    severity="error",
                )
                logger.debug("Constitutional tool completion failed: %s", _finish_exc)
                return False

        # One canonical invocation classifier feeds autonomy, conscience, and Will.
        effect_scope = self._tool_effect_scope(tool_name, args)
        risk_level = classify_execution_risk(
            tool_name,
            args,
            effect_scope=effect_scope,
        )

        # Whether the owner explicitly drove this action (user-facing origin, or an
        # explicit authorization flag on the request context). Computed once so both
        # the EDI gate and the conscience/outcome gates can see it — a hold that
        # exists to "defer to the owner" is redundant when the owner asked for it.
        _payload_ctx = kwargs.get("payload_context")
        governance_context = dict(_payload_ctx or {}) if isinstance(_payload_ctx, dict) else {}
        governance_context.setdefault("origin", _origin)
        governance_context.setdefault("source", _origin)
        governance_context.setdefault("objective", str(getattr(self, "_current_objective", "") or ""))
        if bool(governance_context.get("conversation_only_surface")) or str(
            governance_context.get("tool_execution_policy") or ""
        ).strip().lower() == "deny":
            result = {
                "ok": False,
                "status": "conversation_only_surface",
                "error": (
                    "This authenticated surface is scoped to conversation and "
                    "cannot execute tools or external actions."
                ),
            }
            _record_coding_tool_event(
                result,
                success=False,
                error="conversation_only_surface",
            )
            return result
        _safe_autonomous_web = self._safe_autonomous_web_research_tool(
            tool_name,
            args,
            _origin,
            _payload_ctx if isinstance(_payload_ctx, dict) else None,
        )
        user_authorized = context_has_user_authority(
            _origin,
            _payload_ctx if isinstance(_payload_ctx, dict) else None,
        )
        if _router_owns_governance:
            # CapabilityEngine owns the one canonical lease after schema
            # normalization. Issuing here as well creates two constitutional
            # intents and makes harmless defaults look like argument forgery.
            kwargs["payload_context"] = dict(governance_context)
        else:
            _standing_authority = get_standing_authority_manager()
            authority_decision = await _standing_authority.issue_child_lease(
                tool_name=tool_name,
                arguments=args,
                origin=_origin,
                context=governance_context,
                user_authorized=user_authorized,
                effect_scope=effect_scope,
                risk_level=risk_level,
            )
            if not authority_decision.approved:
                result = {
                    "ok": False,
                    "status": "standing_authority_denied",
                    "error": f"Standing authority denied: {authority_decision.reason}",
                    "authority_receipt_id": authority_decision.receipt_id,
                }
                _record_coding_tool_event(
                    result,
                    success=False,
                    error=authority_decision.reason,
                )
                return result
            governance_context = dict(authority_decision.context)
            _standing_authority_token = authority_decision.token
            kwargs["payload_context"] = dict(governance_context)

        # ── EDI PROGRESSIVE AUTONOMY GATE ────────────────────────────────
        edi = (
            None
            if _router_owns_governance
            else ServiceContainer.get("edi", default=None)
        )
        if edi:
            allowed, reason = edi.can_do(
                tool_name,
                risk_level,
                effect_scope=effect_scope,
                governed=_constitutional_runtime_live,
                user_authorized=user_authorized,
            )
            if not allowed:
                logger.warning("🔓 EDI blocked tool '%s' (risk: %s): %s", tool_name, risk_level, reason)
                result = {"ok": False, "error": f"EDI blocked: {reason}"}
                _record_coding_tool_event(result, success=False, error=reason)
                return result

        # ── DERIVED CONSCIENCE GATE (Kokoro + Minds + Tron) ─────────────────
        # Kokoro can BLOCK a consequential action it judges indefensible (concealment
        # + irreversibility + reach). The Minds can hold severe worst-case outcomes.
        # Tron can block actions that work against the user's interest. These checks
        # are synchronous heuristics in the hot path.
        try:
            _conscience = (
                None
                if _router_owns_governance
                else ServiceContainer.get("kokoro", default=None)
            )
            _action_text = f"{tool_name} [{effect_scope}] {str(args)[:200]}"
            _ctx = {
                "risk_level": risk_level,
                "effect_scope": effect_scope,
                "skill_name": tool_name,
                "tool_name": tool_name,
                "user_authorized": user_authorized,
                "safe_autonomous_web_research": _safe_autonomous_web,
            }
            if _conscience is not None:
                _verdict = _conscience.quick_check(_action_text, context=_ctx)
                # Escalate the rare borderline-with-real-concern case to a full,
                # model-deepened challenge (bounded; falls back to the heuristic on
                # timeout). The model can only raise concern, never clear a flag.
                if _verdict.verdict != "block" and _conscience.should_escalate(_verdict):
                    logger.info("⚖️  Escalating tool '%s' to deep conscience review…", tool_name)
                    _verdict = await _conscience.challenge(_action_text, context=_ctx, timeout=8.0)
                if _verdict.verdict == "block":
                    logger.warning(
                        "⚖️  Adversarial conscience BLOCKED tool '%s': %s",
                        tool_name, _verdict.reasoning,
                    )
                    result = {"ok": False, "error": f"Conscience blocked: {_verdict.reasoning}"}
                    _record_coding_tool_event(result, success=False, error=_verdict.reasoning)
                    return result
                if _verdict.verdict == "caution":
                    logger.info("⚖️  Conscience caution on '%s': %s", tool_name, _verdict.reasoning)
            _minds = (
                None
                if _router_owns_governance
                else ServiceContainer.get("culture_mind", default=None)
            )
            if _minds is not None and hasattr(_minds, "assess_fast"):
                _sim = _minds.assess_fast(_action_text, context=_ctx)
                if getattr(_sim, "recommendation", "") == "hold":
                    if _safe_autonomous_web:
                        logger.info(
                            "🌀 Outcome simulator advisory for autonomous read-only web tool '%s' "
                            "(worst-case harm %.2f); continuing under bounded research policy.",
                            tool_name,
                            float(getattr(_sim, "worst_case_harm", 0.0) or 0.0),
                        )
                    else:
                        reason = (
                            "Outcome simulator held action; worst-case harm "
                            f"{float(getattr(_sim, 'worst_case_harm', 0.0) or 0.0):.2f}"
                        )
                        logger.warning("🌀 Outcome simulator BLOCKED tool '%s': %s", tool_name, reason)
                        result = {
                            "ok": False,
                            "error": reason,
                            "status": "blocked_by_outcome_simulator",
                        }
                        _record_coding_tool_event(result, success=False, error=reason)
                        return result
            _advocate = (
                None
                if _router_owns_governance
                else ServiceContainer.get("tron", default=None)
            )
            if _advocate is not None:
                payload_context_value = kwargs.get("payload_context")
                payload_context: dict[str, Any] = (
                    dict(payload_context_value)
                    if isinstance(payload_context_value, dict)
                    else {}
                )
                confirmed = bool(
                    args.get("confirmed")
                    or args.get("user_confirmed")
                    or kwargs.get("confirmed")
                    or payload_context.get("confirmed")
                    or payload_context.get("user_confirmed")
                    or (_origin in _USER_FACING_TOOL_ORIGINS and risk_level not in ("high", "critical"))
                    or _safe_autonomous_web
                )
                user_benefit = (
                    str(
                        kwargs.get("user_benefit")
                        or payload_context.get("user_benefit")
                        or ""
                    ).strip()
                    or str(getattr(self, "_current_objective", "") or "").strip()
                    or (
                        "support Aura's autonomous curiosity, factual grounding, and memory growth "
                        "with bounded read-only web research"
                        if _safe_autonomous_web
                        else ""
                    )
                    or (
                        "requested through the user-facing desktop/tool lane"
                        if _origin in _USER_FACING_TOOL_ORIGINS
                        else ""
                    )
                )
                _review = _advocate.review_action({
                    "description": _action_text,
                    "irreversible": risk_level in ("high", "critical"),
                    "confirmed": confirmed,
                    "user_benefit": user_benefit,
                    "explanation": f"tool {tool_name} invoked by {_origin}",
                })
                if _review.verdict == "against_user":
                    reason = _review.on_behalf_of_user
                    logger.warning(
                        "🟦 User-advocate BLOCKED tool '%s' on the user's behalf: %s",
                        tool_name, reason,
                    )
                    result = {
                        "ok": False,
                        "error": f"User advocate blocked: {reason}",
                        "status": "blocked_by_user_advocate",
                    }
                    _record_coding_tool_event(result, success=False, error=reason)
                    return result
        except _TOOL_EXECUTION_RECOVERABLE_ERRORS as _consc_err:
            _record_tool_degradation(
                _consc_err,
                action="continued tool execution after derived conscience/advocate gate degraded",
                severity="warning",
            )
            logger.debug("Derived conscience gate degraded: %s", _consc_err)

        # Registered CapabilityEngine skills cross Will once, at their actual
        # normalized execution boundary. Virtual/pre-runtime tools still use
        # this outer gate.
        if not _router_owns_governance:
            try:
                from core.runtime.action_executor import ActionExecutor
                from core.will import ActionDomain

                _will_admission = ActionExecutor.authorize_action(
                    action_name=tool_name,
                    params=args,
                    source=_origin,
                    domain=ActionDomain.TOOL_EXECUTION,
                    priority=0.7,
                    context=governance_context,
                )
                if not _will_admission.approved:
                    logger.warning(
                        "Unified Will REFUSED tool '%s': %s",
                        tool_name,
                        _will_admission.reason,
                    )
                    result = {
                        "ok": False,
                        "error": f"Will refused: {_will_admission.reason}",
                        "will_receipt_id": _will_admission.receipt_id,
                    }
                    _record_coding_tool_event(
                        result,
                        success=False,
                        error=_will_admission.reason,
                    )
                    return result
            except _TOOL_EXECUTION_RECOVERABLE_ERRORS as _will_err:
                _record_tool_degradation(
                    _will_err,
                    action=(
                        "blocked tool execution because Unified Will gate was unavailable"
                        if _constitutional_runtime_live
                        else "continued pre-runtime tool execution without Unified Will gate"
                    ),
                    severity="error" if _constitutional_runtime_live else "warning",
                )
                logger.debug("Unified Will tool gate degraded: %s", _will_err)
                if _constitutional_runtime_live:
                    result = {"ok": False, "error": "Unified Will tool gate unavailable"}
                    _record_coding_tool_event(result, success=False, error=str(_will_err))
                    return result
        # ─────────────────────────────────────────────────────────────────

        # ── EXECUTIVE APPROVAL GATE ──────────────────────────────────────
        if not _router_owns_governance:
            try:
                from core.constitution import get_constitutional_core

                _constitutional_runtime_live = (
                    ServiceContainer.has("executive_core")
                    or ServiceContainer.has("aura_kernel")
                    or ServiceContainer.has("kernel_interface")
                    or bool(getattr(ServiceContainer, "_registration_locked", False))
                )
                _constitution = get_constitutional_core(self)
                _tool_handle = await _constitution.begin_tool_execution(
                    tool_name,
                    args,
                    source=_origin,
                    objective=self._current_objective or "",
                    context=governance_context,
                )
                if not _tool_handle.approved:
                    reason = _tool_handle.decision.reason
                    logger.warning(
                        "🚫 ExecutiveCore blocked tool '%s': %s",
                        tool_name,
                        reason,
                    )
                    try:
                        from core.observability.unified_action_log import get_action_log

                        get_action_log().record(
                            tool_name,
                            _origin,
                            "tool",
                            "blocked",
                            str(reason),
                        )
                    except _TOOL_EXECUTION_RECOVERABLE_ERRORS as _exc:
                        _record_tool_degradation(
                            _exc,
                            action=(
                                "returned blocked tool decision after action-log "
                                "recording failed"
                            ),
                        )
                        logger.debug("Tool action-log blocked event skipped: %s", _exc)
                    result = {"ok": False, "error": f"Executive blocked: {reason}"}
                    _record_coding_tool_event(result, success=False, error=str(reason))
                    return result
                try:
                    from core.observability.unified_action_log import get_action_log

                    get_action_log().record(tool_name, _origin, "tool", "approved")
                except _TOOL_EXECUTION_RECOVERABLE_ERRORS as _exc:
                    _record_tool_degradation(
                        _exc,
                        action=(
                            "continued approved tool execution after action-log "
                            "recording failed"
                        ),
                    )
                    logger.debug("Tool action-log approval event skipped: %s", _exc)
                if _tool_handle.constraints:
                    kwargs.update(_tool_handle.constraints)
            except _TOOL_EXECUTION_RECOVERABLE_ERRORS as _exec_err:
                _record_tool_degradation(
                    _exec_err,
                    action=(
                        "blocked tool execution because constitutional gate was unavailable"
                        if _constitutional_runtime_live
                        else "continued pre-runtime tool execution without constitutional gate"
                    ),
                    severity="error" if _constitutional_runtime_live else "warning",
                )
                if _constitutional_runtime_live:
                    try:
                        from core.health.degraded_events import record_degraded_event

                        record_degraded_event(
                            "orchestrator",
                            "tool_gate_unavailable",
                            detail=tool_name,
                            severity="warning",
                            classification="foreground_blocking"
                            if _origin in ("user", "voice", "admin", "api")
                            else "background_degraded",
                            context={"error": type(_exec_err).__name__},
                            exc=_exec_err,
                        )
                    except _TOOL_EXECUTION_RECOVERABLE_ERRORS as _exc:
                        _record_tool_degradation(
                            _exc,
                            action=(
                                "kept tool execution blocked after degraded-event "
                                "emission failed"
                            ),
                            severity="error",
                        )
                        logger.debug("Constitutional gate degraded-event emission failed: %s", _exc)
                    logger.warning(
                        "🚫 ConstitutionalCore unavailable for tool '%s': %s", tool_name, _exec_err
                    )
                    result = {"ok": False, "error": "Constitutional tool gate unavailable"}
                    _record_coding_tool_event(result, success=False, error=str(_exec_err))
                    return result
                logger.debug("ConstitutionalCore unavailable for tool gate: %s", _exec_err)
        # ─────────────────────────────────────────────────────────────────

        if not _router_owns_governance and _constitutional_runtime_live and _tool_handle:
            try:
                from core.executive.authority_gateway import get_authority_gateway

                capability_token_id = _tool_handle.capability_token_id
                if not capability_token_id:
                    logger.warning(
                        "🚫 Tool '%s' missing capability token under constitutional runtime.",
                        tool_name,
                    )
                    result = {"ok": False, "error": "Capability token missing."}
                    await _finish_constitutional_tool_execution(
                        result,
                        success=False,
                        error="Capability token missing.",
                    )
                    _record_coding_tool_event(
                        result, success=False, error="Capability token missing."
                    )
                    return result
                if not get_authority_gateway().verify_tool_access(tool_name, capability_token_id):
                    logger.warning("🚫 Capability token denied tool '%s'.", tool_name)
                    result = {"ok": False, "error": "Capability token denied tool execution."}
                    await _finish_constitutional_tool_execution(
                        result,
                        success=False,
                        error="Capability token denied tool execution.",
                    )
                    _record_coding_tool_event(
                        result, success=False, error="Capability token denied tool execution."
                    )
                    return result
                kwargs["capability_token_id"] = capability_token_id
            except _TOOL_EXECUTION_RECOVERABLE_ERRORS as capability_err:
                _record_tool_degradation(
                    capability_err,
                    action="blocked tool execution because capability-token verification failed",
                    severity="error",
                )
                logger.warning(
                    "Capability verification failed for tool '%s': %s", tool_name, capability_err
                )
                result = {"ok": False, "error": "Capability verification failed."}
                await _finish_constitutional_tool_execution(
                    result,
                    success=False,
                    error="Capability verification failed.",
                )
                _record_coding_tool_event(result, success=False, error=str(capability_err))
                return result

        # 0. Virtual & Internal Tools
        if tool_name == "swarm_debate":
            swarm = getattr(self, "swarm", None)
            if not swarm:
                result = {"ok": False, "error": "Swarm Delegator not available."}
                await _finish_constitutional_tool_execution(
                    result,
                    success=False,
                    error="Swarm Delegator not available.",
                )
                _record_coding_tool_event(
                    result, success=False, error="Swarm Delegator not available."
                )
                return result
            topic = args.get("topic") or args.get("query") or self._current_objective
            roles = args.get("roles", ["architect", "critic"])
            self._emit_thought_stream(f"🐝 Engaging Swarm Debate: {topic[:100]}...")

            async def _finalize_debate_result(
                response: dict[str, Any],
                *,
                success: bool,
                error: str = "",
            ) -> dict[str, Any]:
                # Close the synchronous standing lease before any cancellation
                # can interrupt constitutional completion bookkeeping.
                _record_coding_tool_event(response, success=success, error=error)
                drained = await drain_owned_awaitable(
                    _finish_constitutional_tool_execution(
                        response,
                        success=success,
                        error=error,
                    ),
                    name="ToolExecution.swarm_debate.authority_completion",
                    owner="tool_execution:swarm_debate",
                    allow_during_shutdown=True,
                )
                drained.task.result()
                if drained.cancellation is not None:
                    raise drained.cancellation
                return response

            debate_result_ready = False
            debate_terminalized = False
            try:
                result = await swarm.delegate_debate(topic, roles=roles, **kwargs)
                debate_result_ready = True
            except asyncio.CancelledError:
                debate_error = "Swarm debate cancelled during execution."
                response = {
                    "ok": False,
                    "status": "cancelled",
                    "error": debate_error,
                }
                debate_terminalized = True
                await _finalize_debate_result(
                    response,
                    success=False,
                    error=debate_error,
                )
                raise
            except _TOOL_EXECUTION_RECOVERABLE_ERRORS as exc:
                _record_tool_degradation(
                    exc,
                    action="closed failed swarm debate authority lifecycle",
                    severity="error",
                )
                debate_error = f"Swarm debate failed: {type(exc).__name__}: {exc}"[:240]
                response = {
                    "ok": False,
                    "status": "failed",
                    "error": debate_error,
                }
                debate_terminalized = True
                return await _finalize_debate_result(
                    response,
                    success=False,
                    error=debate_error,
                )
            finally:
                if not debate_result_ready and not debate_terminalized:
                    debate_error = "Swarm debate aborted before producing a result."
                    response = {
                        "ok": False,
                        "status": "failed",
                        "error": debate_error,
                    }
                    debate_terminalized = True
                    await _finalize_debate_result(
                        response,
                        success=False,
                        error=debate_error,
                    )

            debate_error = swarm_debate_failure_reason(result)
            if debate_error:
                response = {"ok": False, "error": debate_error}
                debate_terminalized = True
                return await _finalize_debate_result(
                    response,
                    success=False,
                    error=debate_error,
                )
            response = {"ok": True, "output": result}
            debate_terminalized = True
            return await _finalize_debate_result(response, success=True)

        try:
            # 1. Check if tool exists in registry
            if router is None and tool_name == "notify_user":
                result = {"ok": True, "message": args.get("message", "Done.")}
                await _finish_constitutional_tool_execution(result, success=True)
                _record_coding_tool_event(result, success=True)
                return result

            if router is None:
                result = {"ok": False, "error": "Tool router unavailable."}
                await _finish_constitutional_tool_execution(
                    result,
                    success=False,
                    error="Tool router unavailable.",
                )
                _record_coding_tool_event(
                    result,
                    success=False,
                    error="Tool router unavailable.",
                )
                return result

            if tool_name not in router.skills:
                # Fallback for notify_user which is sometimes a virtual alias.
                if tool_name == "notify_user":
                    result = {"ok": True, "message": args.get("message", "Done.")}
                    await _finish_constitutional_tool_execution(result, success=True)
                    _record_coding_tool_event(result, success=True)
                    return result

                # 1.5 Autogenesis (Hephaestus Engine)
                if self.hephaestus:
                    self._emit_thought_stream(
                        f"🔨 Tool '{tool_name}' missing. Initiating Autonomous Forge..."
                    )
                    objective = f"Create a skill '{tool_name}' to handle request within objective: {self._current_objective}"
                    forge_result = await self.hephaestus.synthesize_skill(tool_name, objective)
                    if forge_result.get("ok"):
                        self._emit_thought_stream(
                            f"✅ Skill '{tool_name}' forged successfully. Retrying..."
                        )
                        handoff_result = {
                            "ok": True,
                            "handoff": "autogenesis_retry",
                            "tool_name": tool_name,
                        }
                        await _finish_constitutional_tool_execution(
                            handoff_result,
                            success=True,
                        )
                        _record_coding_tool_event(handoff_result, success=True)
                        # Retry execution once
                        return await self.execute_tool(tool_name, args, **kwargs)
                    else:
                        logger.warning(
                            "Autogenesis failed for %s: %s", tool_name, forge_result.get("error")
                        )

                result = {"ok": False, "error": f"Tool '{tool_name}' not found."}
                await _finish_constitutional_tool_execution(
                    result,
                    success=False,
                    error=f"Tool '{tool_name}' not found.",
                )
                _record_coding_tool_event(
                    result,
                    success=False,
                    error=str(result["error"]),
                )
                return result

            # 2. Contextual Awareness
            context = {
                **kwargs,
                "objective": self._current_objective,
                "system": self.status.model_dump(),
                "stealth": await self.stealth_mode.get_stealth_status()
                if hasattr(self, "stealth_mode")
                and self.stealth_mode
                and getattr(self.stealth_mode, "stealth_enabled", False)
                else {},
                "liquid_state": self.liquid_state.get_status()
                if hasattr(self, "liquid_state") and self.liquid_state
                else {},
                **governance_context,
                "orchestrator": self,
            }

            # 2.5 Resistance Sandbox — emit prediction before execution
            _sandbox = None
            _sandbox_predicted = "success"
            try:
                from core.embodiment.resistance_sandbox import get_resistance_sandbox

                _sandbox = get_resistance_sandbox()
                _sandbox_predicted = (
                    "success"
                    if tool_name not in ("browser", "shell", "file_write")
                    else "success_with_side_effects"
                )
            except (ImportError, AttributeError, RuntimeError):
                _sandbox = None

            # 3. Literal Execution (Async)
            if _tool_handle is not None:
                from core.governance_context import governed_scope

                async with governed_scope(_tool_handle.decision):
                    result = await router.execute(tool_name, args, context)
            else:
                result = await router.execute(tool_name, args, context)
            if not isinstance(result, dict):
                result = {"ok": True, "output": result}

            success = bool(result.get("ok", False))
            deferred = str(result.get("status", "") or "").lower() == "deferred"
            elapsed_ms = (time.time() - _start) * 1000
            if deferred:
                logger.info(
                    "Tool %s execution deferred: %s",
                    tool_name,
                    result.get("reason") or result.get("message") or "background_policy",
                )
                await _finish_constitutional_tool_execution(
                    result,
                    success=False,
                    error=str(result.get("reason") or result.get("error") or "deferred"),
                )
                _record_coding_tool_event(
                    result,
                    success=False,
                    error=str(result.get("reason") or result.get("error") or "deferred"),
                )
                return result
            logger.info("Tool %s execution completed: %s", tool_name, success)

            # 3.5 Resistance Sandbox — compare prediction to actual outcome
            if _sandbox is not None:
                try:
                    _actual_outcome = (
                        "success"
                        if success
                        else f"failure:{str(result.get('error', 'unknown'))[:80]}"
                    )
                    # The sandbox saves its state on every prediction it
                    # scores; off the loop, awaited (LIVE 2026-09-19, named
                    # by the loop report from execute_tool).
                    await off_the_loop(
                        _sandbox.execute_with_prediction,
                        action_type="tool_exec",
                        target=tool_name,
                        predicted_outcome=_sandbox_predicted,
                        action_fn=lambda: _actual_outcome,
                    )
                except _TOOL_EXECUTION_RECOVERABLE_ERRORS as _sbx_err:
                    _record_tool_degradation(
                        _sbx_err,
                        action="returned tool result after resistance-sandbox feedback failed",
                    )
                    logger.debug("Resistance sandbox feedback failed: %s", _sbx_err)

            # 5. Tool Learning
            if hasattr(self, "tool_learner") and self.tool_learner:
                try:
                    category = self.tool_learner.classify_task(
                        str(args.get("query", args.get("path", "")))
                    )
                    await off_the_loop(self.tool_learner.record_usage, tool_name, category, success, elapsed_ms)
                except _TOOL_EXECUTION_RECOVERABLE_ERRORS as _e:
                    _record_tool_degradation(
                        _e,
                        action="returned tool result after tool-learning usage record failed",
                    )
                    logger.debug("Tool learning record failed: %s", _e)

            # 6. Episodic Recording (Now via Facade)
            if hasattr(self, "memory") and self.memory:
                try:
                    await self.memory.commit_interaction(
                        context=str(args)[:500],
                        action=f"execute_tool({tool_name})",
                        outcome=str(result)[:500],
                        success=success,
                        importance=0.3 if success else 0.7,
                    )
                except _TOOL_EXECUTION_RECOVERABLE_ERRORS as _e:
                    _record_tool_degradation(
                        _e,
                        action="returned tool result after episodic memory write failed",
                    )
                    logger.debug("Unified memory record failed: %s", _e)

            # 7. Causal Learning (ACG)
            try:
                from core.world_model.acg import acg

                acg.record_outcome(
                    action=tool_name, context=str(context)[:500], outcome=result, success=success
                )
            except _TOOL_EXECUTION_RECOVERABLE_ERRORS as _e:
                _record_tool_degradation(
                    _e,
                    action="returned tool result after causal outcome recording failed",
                )
                logger.debug("ACG record failed: %s", _e)

            # WIRE-01: Affect State Update
            # Redundant local import removed
            affect_mgr = ServiceContainer.get("affect_engine", default=None)
            if affect_mgr and hasattr(affect_mgr, "apply_stimulus"):
                stimulus = "error" if not success else "intrigue"
                intensity = 15.0 if not success else 5.0
                self._fire_and_forget(
                    affect_mgr.apply_stimulus(stimulus, intensity),
                    name="orchestrator.affect_engine.apply_stimulus",
                )

            # Operational success is competence feedback, not evidence that the
            # user revoked Aura's autonomy.  Conflating the two created a
            # cascading failure: transient tool errors lowered EDI to Advisory,
            # which then blocked harmless read-only recovery/search actions.
            if edi and hasattr(edi, "record_execution_outcome"):
                edi.record_execution_outcome(
                    tool_name,
                    success=success,
                    error=str(result.get("error", "") or ""),
                )

            await _finish_constitutional_tool_execution(result, success=success)
            _record_coding_tool_event(result, success=success, error=str(result.get("error", "")))
            return result

        except _TOOL_EXECUTION_RECOVERABLE_ERRORS as e:
            # A tool crash is operational evidence, not a trust revocation.
            edi = ServiceContainer.get("edi", default=None)
            if edi and hasattr(edi, "record_execution_outcome"):
                edi.record_execution_outcome(
                    tool_name,
                    success=False,
                    error=f"{type(e).__name__}: {e}",
                )

            _record_tool_degradation(
                e,
                action="returned structured execution_jolt after tool execution failed",
                severity="error",
            )
            logger.error("Execution Jolt (Pain): Tool %s crashed: %s", tool_name, e)
            # Record failure
            if hasattr(self, "memory") and self.memory:
                try:
                    await self.memory.commit_interaction(
                        context=str(args)[:500],
                        action=f"execute_tool({tool_name})",
                        outcome=f"CRASH: {type(e).__name__}",
                        success=False,
                        emotional_valence=-0.5,
                        importance=0.9,
                    )
                except _TOOL_EXECUTION_RECOVERABLE_ERRORS as _e:
                    _record_tool_degradation(
                        _e,
                        action="returned execution_jolt after crash-path memory write failed",
                    )
                    logger.debug("Unified memory record failed (crash path): %s", _e)
            result = {"ok": False, "error": "execution_jolt", "message": str(e)}
            await _finish_constitutional_tool_execution(result, success=False, error=str(e))
            _record_coding_tool_event(result, success=False, error=str(e))
            return result

        logger.info("Orchestrator stopped")
