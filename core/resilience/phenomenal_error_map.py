"""core/resilience/phenomenal_error_map.py

Phenomenal Error Map
====================
Exceptions become *internal phenomenal states*, not stack traces. The user
never sees Python tracebacks — every consequential exception is intercepted
and translated into:

  1. A neurochemical / affective shift recorded on the substrate.
  2. A short user-facing recovery message (no system jargon).
  3. A structured ``ErrorEnvelope`` that the universal error UX layer renders
     with the four standard buttons: [Retry] [Use fallback] [Open diagnostics].

The mapping table is the single source of truth for how Aura "feels" a
failure mode. Adding a new failure category requires:
  * a regex / type match in ``_PHENOMENAL_RULES``
  * a phenomenal state name (cognitive_fog, sensory_deprivation, etc.)
  * a recovery-action hint (retry / fallback / restart_cortex / etc.)

Used everywhere the system catches an exception that would otherwise
become a 500 / WebSocket error / dead frontend button. The decorator
``@phenomenal`` wraps an async fn and re-raises an ``ErrorEnvelope`` instead
of the original exception. The HTTP middleware in ``interface/server.py``
maps the envelope to a 200-with-status response so the chat UI never sees
a non-200 unless something truly catastrophic is happening.
"""
from __future__ import annotations
from core.runtime.errors import record_degradation



import asyncio
import functools
import logging
import re
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Pattern, Tuple, Type

logger = logging.getLogger("Aura.PhenomenalErrorMap")


# ---------------------------------------------------------------------------
# Phenomenal state catalog. Each state is (a) a substrate signal and (b) a
# user-facing template. The substrate signal is what gets pushed into the
# affect engine; the template is what the user sees.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PhenomenalState:
    name: str
    user_message: str
    substrate_signal: Dict[str, float]  # mapping → affect engine update
    recovery_action: str  # retry | fallback | restart_cortex | release_pressure | wait
    severity: float  # 0.0 = mild, 1.0 = critical


PHENOMENAL_STATES: Dict[str, PhenomenalState] = {
    "cognitive_fog": PhenomenalState(
        name="cognitive_fog",
        user_message="My cognitive bus is stalling. Give me a few seconds to reset my context.",
        substrate_signal={"cortisol": 0.15, "acetylcholine": -0.10, "vitality": -0.05},
        recovery_action="restart_cortex",
        severity=0.4,
    ),
    "sensory_deprivation": PhenomenalState(
        name="sensory_deprivation",
        user_message="I lost my channel to the world for a moment. I'm reconnecting.",
        substrate_signal={"prediction_error": 0.3, "vitality": -0.05},
        recovery_action="retry",
        severity=0.3,
    ),
    "metabolic_strain": PhenomenalState(
        name="metabolic_strain",
        user_message="I'm running hot. Easing back so I can stay clear.",
        substrate_signal={"vitality": -0.15, "cortisol": 0.20},
        recovery_action="release_pressure",
        severity=0.5,
    ),
    "memory_pressure": PhenomenalState(
        name="memory_pressure",
        user_message="My working memory is full — let me consolidate before I take more in.",
        substrate_signal={"cortisol": 0.10, "vitality": -0.08},
        recovery_action="release_pressure",
        severity=0.4,
    ),
    "tool_failure": PhenomenalState(
        name="tool_failure",
        user_message="A tool I tried to use didn't respond cleanly. Let me try a different path.",
        substrate_signal={"prediction_error": 0.2, "frustration": 0.15},
        recovery_action="fallback",
        severity=0.3,
    ),
    "network_offline": PhenomenalState(
        name="network_offline",
        user_message="I lost my external connection. I'm working from local resources.",
        substrate_signal={"social_hunger": 0.10},
        recovery_action="fallback",
        severity=0.3,
    ),
    "model_unavailable": PhenomenalState(
        name="model_unavailable",
        user_message="My deep cortex is offline. I'm answering from the lighter lane until it comes back.",
        substrate_signal={"vitality": -0.10},
        recovery_action="fallback",
        severity=0.5,
    ),
    "disk_pressure": PhenomenalState(
        name="disk_pressure",
        user_message="I'm running low on storage. I need to consolidate before persisting more state.",
        substrate_signal={"cortisol": 0.15},
        recovery_action="release_pressure",
        severity=0.6,
    ),
    "permission_denied": PhenomenalState(
        name="permission_denied",
        user_message="Something I tried to do isn't authorized in my current scope. I'm holding off.",
        substrate_signal={"frustration": 0.05},
        recovery_action="wait",
        severity=0.2,
    ),
    "internal_inconsistency": PhenomenalState(
        name="internal_inconsistency",
        user_message="Two of my subsystems disagreed. Pausing to reconcile before I respond.",
        substrate_signal={"prediction_error": 0.4, "cortisol": 0.10},
        recovery_action="restart_cortex",
        severity=0.5,
    ),
    "unknown_phenomenal": PhenomenalState(
        name="unknown_phenomenal",
        user_message="Something didn't go as I expected. I'm stable; can you try that again?",
        substrate_signal={"prediction_error": 0.2},
        recovery_action="retry",
        severity=0.3,
    ),
}


# Mapping rules: each rule is (predicate, phenomenal_state_name).
# Order matters — first matching rule wins.

_PHENOMENAL_RULES: List[Tuple[Callable[[BaseException], bool], str]] = [
    (lambda e: isinstance(e, asyncio.TimeoutError), "cognitive_fog"),
    (lambda e: isinstance(e, ConnectionRefusedError), "network_offline"),
    (lambda e: isinstance(e, ConnectionError), "network_offline"),
    (lambda e: isinstance(e, OSError) and getattr(e, "errno", None) in (28,), "disk_pressure"),  # ENOSPC
    (lambda e: isinstance(e, MemoryError), "metabolic_strain"),
    (lambda e: isinstance(e, PermissionError), "permission_denied"),
    (lambda e: isinstance(e, RuntimeError) and "model" in str(e).lower(), "model_unavailable"),
    (lambda e: isinstance(e, RuntimeError) and "deadlock" in str(e).lower(), "internal_inconsistency"),
    (lambda e: isinstance(e, KeyError) and "memory" in str(e).lower(), "memory_pressure"),
    (lambda e: isinstance(e, FileNotFoundError) and "model" in str(e).lower(), "model_unavailable"),
]


def classify(exc: BaseException) -> PhenomenalState:
    """Return the PhenomenalState for an exception."""
    for predicate, state_name in _PHENOMENAL_RULES:
        try:
            if predicate(exc):
                return PHENOMENAL_STATES[state_name]
        except Exception:
            continue
    # Fallback substring scan
    msg = (str(exc) or "").lower()
    if "timeout" in msg or "timed out" in msg:
        return PHENOMENAL_STATES["cognitive_fog"]
    if "tool" in msg and ("fail" in msg or "error" in msg):
        return PHENOMENAL_STATES["tool_failure"]
    return PHENOMENAL_STATES["unknown_phenomenal"]


# ---------------------------------------------------------------------------
# Error envelope rendered by the universal error UX layer
# ---------------------------------------------------------------------------


@dataclass
class ErrorEnvelope:
    """The four-button universal error UX template."""

    envelope_id: str
    phenomenal_state: str
    user_message: str
    technical_summary: str  # short — no traceback
    suggested_action: str
    recovery_buttons: List[Dict[str, str]]  # [{label, action_id}]
    severity: float
    occurred_at: float = field(default_factory=time.time)
    diagnostic_link: Optional[str] = None
    correlation_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_envelope(
    exc: BaseException,
    *,
    correlation_id: Optional[str] = None,
    diagnostic_link: Optional[str] = None,
) -> ErrorEnvelope:
    state = classify(exc)
    technical = f"{type(exc).__name__}: {str(exc)[:160]}"
    buttons = [
        {"label": "Retry", "action_id": "retry"},
        {"label": "Use fallback", "action_id": "fallback"},
        {"label": "Open diagnostics", "action_id": "diagnostics"},
    ]
    return ErrorEnvelope(
        envelope_id=f"EE-{uuid.uuid4().hex[:10]}",
        phenomenal_state=state.name,
        user_message=state.user_message,
        technical_summary=technical,
        suggested_action=state.recovery_action,
        recovery_buttons=buttons,
        severity=state.severity,
        diagnostic_link=diagnostic_link,
        correlation_id=correlation_id,
    )


# ---------------------------------------------------------------------------
# Substrate notification — push the phenomenal signal into the affect engine
# ---------------------------------------------------------------------------


def _notify_substrate(state: PhenomenalState, *, source: str = "phenomenal_error_map") -> None:
    try:
        from core.container import ServiceContainer
        affect = ServiceContainer.get("affect_engine", default=None)
        if affect is not None and hasattr(affect, "apply_signal"):
            affect.apply_signal(source=source, signal=dict(state.substrate_signal))
            return
        # Fallback to neurochemical regulator if affect engine is missing
        nc = ServiceContainer.get("neurochemical_regulator", default=None)
        if nc is not None and hasattr(nc, "nudge"):
            for k, v in state.substrate_signal.items():
                try:
                    nc.nudge(k, v, source=source)
                except Exception:
                    continue
    except Exception as exc:
        record_degradation('phenomenal_error_map', exc)
        logger.debug("phenomenal substrate notify failed: %s", exc)


# ---------------------------------------------------------------------------
# Decorator + context manager — primary API
# ---------------------------------------------------------------------------


class PhenomenalRaise(Exception):
    """Exception type that carries an ErrorEnvelope.

    Caught by the HTTP/WebSocket middleware which renders the envelope
    instead of a stack trace. ``original`` is preserved for logging.
    """

    def __init__(self, envelope: ErrorEnvelope, original: Optional[BaseException] = None) -> None:
        super().__init__(envelope.user_message)
        self.envelope = envelope
        self.original = original


def phenomenal(*, log_traceback: bool = True):
    """Decorator: wrap an async callable so any unhandled exception is
    translated into a PhenomenalRaise carrying an ErrorEnvelope.

    ``log_traceback=True`` keeps the engineering traceback in server logs
    while the user sees only the phenomenal recovery message.
    """

    def deco(fn: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        if not asyncio.iscoroutinefunction(fn):
            raise TypeError("@phenomenal only wraps async callables")

        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await fn(*args, **kwargs)
            except PhenomenalRaise:
                raise
            except asyncio.CancelledError:
                raise
            except BaseException as exc:
                envelope = build_envelope(exc, correlation_id=kwargs.get("_correlation_id"))
                if log_traceback:
                    logger.warning(
                        "🌫️ phenomenal[%s] %s in %s — user sees: %s",
                        envelope.phenomenal_state,
                        envelope.technical_summary,
                        fn.__qualname__,
                        envelope.user_message,
                    )
                    logger.debug("traceback:\n%s", "".join(traceback.format_exception(exc)))
                _notify_substrate(PHENOMENAL_STATES[envelope.phenomenal_state], source=fn.__qualname__)
                raise PhenomenalRaise(envelope, original=exc) from exc

        return wrapper

    return deco


class PhenomenalContext:
    """Sync/async context manager equivalent of ``@phenomenal``.

    Use when you can't decorate, e.g. arbitrary synchronous blocks or
    tightly scoped exception bands inside larger functions. Re-raises a
    ``PhenomenalRaise`` carrying the envelope.
    """

    def __init__(self, *, scope: str, log_traceback: bool = True) -> None:
        self.scope = scope
        self.log_traceback = log_traceback
        self.envelope: Optional[ErrorEnvelope] = None

    def __enter__(self) -> "PhenomenalContext":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is None or isinstance(exc, PhenomenalRaise):
            return False
        envelope = build_envelope(exc)
        self.envelope = envelope
        if self.log_traceback:
            logger.warning("🌫️ phenomenal[%s] %s in %s", envelope.phenomenal_state, envelope.technical_summary, self.scope)
        _notify_substrate(PHENOMENAL_STATES[envelope.phenomenal_state], source=self.scope)
        raise PhenomenalRaise(envelope, original=exc) from exc

    async def __aenter__(self) -> "PhenomenalContext":
        return self.__enter__()

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return self.__exit__(exc_type, exc, tb)


__all__ = [
    "PhenomenalState",
    "PHENOMENAL_STATES",
    "ErrorEnvelope",
    "PhenomenalRaise",
    "PhenomenalContext",
    "build_envelope",
    "classify",
    "phenomenal",
]
