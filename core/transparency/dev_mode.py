"""
Development Mode & Transparency Layer (v2026.5.1)

Provides Bryan with real-time visibility into:
- Aura's thought processes and reasoning
- Tool execution dispatch and results
- Memory operations and consolidation
- Constitutional approval workflows
- System state and diagnostics

This enables collaborative development and debugging while maintaining
security boundaries.
"""

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.turn_progress import ToolActivity

logger = logging.getLogger(__name__)
_DEV_MODE_CALLBACK_ERRORS = (OSError, RuntimeError, TypeError, ValueError)


class TransparencyLevel(StrEnum):
    """Verbosity levels for transparency output."""
    SILENT = "silent"          # No transparency
    MINIMAL = "minimal"        # Errors and major milestones only
    NORMAL = "normal"          # Standard logging + key operations
    VERBOSE = "verbose"        # Detailed operation traces
    DEBUG = "debug"             # Full internal state dumps


@dataclass
class ThoughtTrace:
    """Records a thought or reasoning step."""
    timestamp: float = field(default_factory=time.time)
    objective: str = ""
    reasoning: str = ""
    confidence: float = 0.0
    decision: str = ""
    alternatives: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ToolExecutionTrace:
    """Records a tool execution event."""
    timestamp: float = field(default_factory=time.time)
    tool_name: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    status: str = "pending"  # pending, running, succeeded, deferred, failed
    result: dict[str, Any] | None = None
    error: str | None = None
    execution_time_ms: float = 0.0
    origin: str = "unknown"
    _progress_activity: ToolActivity | None = field(default=None, repr=False, compare=False)
    
    def to_dict(self) -> dict[str, Any]:
        d = asdict(replace(self, _progress_activity=None))
        d.pop("_progress_activity")
        # Don't leak sensitive param details in logs
        d["params"] = {k: "***" if k in {"password", "token", "secret", "key"} else v 
                      for k, v in self.params.items()}
        return d


@dataclass
class ConsentRequest:
    """Records a consent/approval request."""
    timestamp: float = field(default_factory=time.time)
    request_type: str = ""  # tool_execution, memory_write, state_mutation, etc.
    description: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    requires_user_input: bool = False
    approved: bool = False
    approval_reason: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DevMode:
    """
    Development Mode: Real-time transparency into Aura's operations.
    
    This system allows Bryan to:
    - See Aura's reasoning in real-time
    - Watch tool execution and results
    - Monitor consent/approval workflows
    - Get system diagnostics
    - Trace memory operations
    - Understand decision-making
    """
    
    def __init__(self, level: TransparencyLevel = TransparencyLevel.NORMAL):
        self.level = level
        self.thought_traces: list[ThoughtTrace] = []
        self.tool_traces: list[ToolExecutionTrace] = []
        self.consent_requests: list[ConsentRequest] = []
        self.active_session_id = ""
        self._callbacks: list[Callable[[str, dict[str, Any]], None]] = []
        self._lock: asyncio.Lock | None = None  # Lazy-loaded in async context
    
    async def _get_lock(self) -> asyncio.Lock:
        """Get or create the async lock (lazy-loaded)."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock
        
    async def set_transparency_level(self, level: TransparencyLevel):
        """Change transparency level at runtime."""
        async with await self._get_lock():
            self.level = level
            logger.info("🔍 DevMode: Transparency level changed to %s", level)
    
    async def record_thought(self, objective: str, reasoning: str, 
                            confidence: float = 0.5, decision: str = "",
                            alternatives: list[str] | None = None) -> ThoughtTrace:
        """Record a reasoning step for visibility."""
        if self.level == TransparencyLevel.SILENT:
            return ThoughtTrace()
        
        trace = ThoughtTrace(
            objective=objective,
            reasoning=reasoning,
            confidence=confidence,
            decision=decision,
            alternatives=alternatives or []
        )
        
        async with await self._get_lock():
            self.thought_traces.append(trace)
            if len(self.thought_traces) > 100:  # Keep memory bounded
                self.thought_traces = self.thought_traces[-100:]
        
        if self.level in {TransparencyLevel.DEBUG, TransparencyLevel.VERBOSE}:
            logger.debug("💭 Thought: %s | Decision: %s | Conf: %.2f", 
                        objective[:60], decision[:40], confidence)
            await self._emit_event("thought_recorded", trace.to_dict())
        
        return trace
    
    async def record_tool_execution(self, tool_name: str, params: dict[str, Any],
                                   origin: str = "unknown") -> ToolExecutionTrace:
        """Record the start of tool execution."""
        trace = ToolExecutionTrace(
            tool_name=tool_name,
            params=params,
            status="running",
            origin=origin
        )
        
        # A turn running a tool is a turn that is working.
        #
        # Every deadline in a turn now defers to progress, and progress was
        # tokens arriving. A tool loop stops decoding while the tool runs, so
        # a turn reading three files went quiet for thirty-four seconds at a
        # time and its clocks concluded it had stopped. Live on 2026-08-28
        # that ended a ledgerkit turn with cognitive_engine_timeout after
        # three successful reads.
        try:
            from core.runtime.turn_progress import tool_started

            trace._progress_activity = tool_started()
        except ImportError:
            pass

        async with await self._get_lock():
            self.tool_traces.append(trace)
            if len(self.tool_traces) > 50:  # Keep memory bounded
                self.tool_traces = self.tool_traces[-50:]
        
        if self.level != TransparencyLevel.SILENT:
            logger.info("🔧 Tool Dispatch: %s (origin=%s)", tool_name, origin)
            if self.level in {TransparencyLevel.DEBUG, TransparencyLevel.VERBOSE}:
                logger.debug("   Params: %s", {k: "***" if k in {"password", "token"} else v 
                                              for k, v in params.items()})
        
        return trace
    
    async def complete_tool_execution(self, trace: ToolExecutionTrace,
                                     result: dict[str, Any],
                                     execution_time_ms: float = 0.0):
        """Record tool execution completion."""
        # And when it finishes, so the gap a long tool leaves behind is
        # bracketed by two signs of life rather than one.
        try:
            from core.runtime.turn_progress import tool_finished

            tool_finished(trace._progress_activity)
        except ImportError:
            pass

        result_status = str(result.get("status", "") or "").strip().lower()
        deferred = result_status == "deferred"
        trace.status = (
            "deferred"
            if deferred
            else ("succeeded" if result.get("ok", False) else "failed")
        )
        trace.result = result
        trace.execution_time_ms = execution_time_ms
        trace.error = (
            None
            if deferred or result.get("ok")
            else str(result.get("error") or result.get("reason") or "execution_failed")
        )

        # A reply may claim "I ran it" only if something did. This is the
        # receipt that entitles it; see the turn-scoped holder for why a
        # deferred tool does not count as one.
        if not deferred:
            try:
                from core.conversation.surface_disposition import record_tool_receipt

                params = trace.params if isinstance(trace.params, dict) else {}
                object_ref = next(
                    (
                        str(params[key])
                        for key in ("path", "app", "url", "query", "target", "name")
                        if params.get(key) not in (None, "")
                    ),
                    "",
                )
                effect_observed = bool(
                    result.get("effect_verified")
                    or result.get("postcondition_verified")
                    or result.get("verification") in {"observed", "verified", "postcondition_verified"}
                )
                record_tool_receipt(
                    trace.tool_name,
                    action=str(result.get("action") or trace.tool_name),
                    object_ref=object_ref,
                    ok=bool(result.get("ok", False)),
                    effect_observed=effect_observed,
                    verification=str(result.get("verification") or "tool_result"),
                    evidence=str(result.get("evidence") or result.get("error") or ""),
                )
            except Exception as exc:  # never let bookkeeping break a tool result
                # The tool result stands. A receipt recorder that throws on
                # every call is a real fault, and `pass` was hiding it.
                record_degradation(
                    "transparency_dev_mode",
                    exc,
                    severity="warning",
                    action="returned the tool result after receipt bookkeeping failed",
                    enforce_failure_policy=False,
                )
        
        if self.level != TransparencyLevel.SILENT:
            if deferred:
                logger.info(
                    "⏸️ Tool Deferred: %s in %.0fms (%s)",
                    trace.tool_name,
                    execution_time_ms,
                    str(result.get("reason") or "policy_deferred")[:120],
                )
            else:
                status_emoji = "✅" if result.get("ok") else "❌"
                logger.info("%s Tool Result: %s in %.0fms",
                           status_emoji, trace.tool_name, execution_time_ms)
            if self.level in {TransparencyLevel.DEBUG}:
                summary = result.get("summary") or result.get("result") or ""
                if isinstance(summary, str):
                    logger.debug("   Summary: %s", summary[:100])
        
        await self._emit_event("tool_completed", trace.to_dict())
    
    async def record_consent_request(self, request_type: str, description: str,
                                    details: dict[str, Any] | None = None,
                                    requires_user: bool = False) -> ConsentRequest:
        """Record a consent/approval request."""
        request = ConsentRequest(
            request_type=request_type,
            description=description,
            details=details or {},
            requires_user_input=requires_user
        )
        
        async with await self._get_lock():
            self.consent_requests.append(request)
            if len(self.consent_requests) > 25:
                self.consent_requests = self.consent_requests[-25:]
        
        if self.level != TransparencyLevel.SILENT:
            marker = "🔐" if requires_user else "✓"
            logger.info("%s Consent: %s — %s", marker, request_type, description)
        
        if requires_user and self.level != TransparencyLevel.SILENT:
            logger.warning("⚠️  User input required for: %s", description)
        
        await self._emit_event("consent_requested", request.to_dict())
        return request
    
    async def approve_consent(self, request: ConsentRequest, 
                             reason: str = "auto-approved"):
        """Mark a consent request as approved."""
        request.approved = True
        request.approval_reason = reason
        
        if self.level != TransparencyLevel.SILENT:
            logger.info("✅ Approved: %s (%s)", request.request_type, reason)
        
        await self._emit_event("consent_approved", request.to_dict())
    
    async def deny_consent(self, request: ConsentRequest,
                          reason: str = "user-denied"):
        """Mark a consent request as denied."""
        request.approved = False
        request.approval_reason = reason
        
        if self.level != TransparencyLevel.SILENT:
            logger.warning("❌ Denied: %s (%s)", request.request_type, reason)
        
        await self._emit_event("consent_denied", request.to_dict())
    
    def get_thought_history(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent thoughts for inspection."""
        return [t.to_dict() for t in self.thought_traces[-limit:]]
    
    def get_tool_history(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent tool executions for inspection."""
        return [t.to_dict() for t in self.tool_traces[-limit:]]
    
    def get_consent_history(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent consent requests for inspection."""
        return [c.to_dict() for c in self.consent_requests[-limit:]]
    
    async def get_session_summary(self) -> dict[str, Any]:
        """Get a summary of the current session."""
        async with await self._get_lock():
            return {
                "session_id": self.active_session_id,
                "transparency_level": self.level.value,
                "thoughts_recorded": len(self.thought_traces),
                "tools_executed": len(self.tool_traces),
                "consent_requests": len(self.consent_requests),
                "recent_thoughts": self.get_thought_history(3),
                "recent_tools": self.get_tool_history(3),
                "recent_consents": self.get_consent_history(3),
                "timestamp": time.time(),
            }
    
    async def register_callback(self, callback: Callable[[str, dict[str, Any]], None]):
        """Register a callback for transparency events."""
        async with await self._get_lock():
            self._callbacks.append(callback)
    
    async def _emit_event(self, event_type: str, data: dict[str, Any]):
        """Emit a transparency event to all registered callbacks."""
        async with await self._get_lock():
            callbacks = list(self._callbacks)
        
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event_type, data)
                else:
                    callback(event_type, data)
            except _DEV_MODE_CALLBACK_ERRORS as e:
                record_degradation("dev_mode.callback", e)
                logger.warning("DevMode callback failed: %s", e)


# Global dev mode instance
_dev_mode_instance: DevMode | None = None


def get_dev_mode(level: TransparencyLevel = TransparencyLevel.NORMAL) -> DevMode:
    """Get or create the global dev mode instance."""
    global _dev_mode_instance
    if _dev_mode_instance is None:
        _dev_mode_instance = DevMode(level)
    return _dev_mode_instance
