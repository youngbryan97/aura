# core/brain/persona_integration.py
"""Persona integration: apply a persona to the live cognitive engine.

The previous implementation prepended the persona's *system* instructions into
the user objective with textual markers, re-wrapped the target method on every
call to ``initialize_persona_integration``, hid an async method behind a ``def``
wrapper, and reported success when it had wrapped nothing at all.

The persona now travels as a structured context field consumed at system role
by ``CognitiveEngine`` (see the ``persona_system_prompt`` seam there), so it
cannot be overridden by later objective text or contaminate task semantics,
caching, memory or audit attribution.

CP126 ab3abbae / 51b7a4a1 / 481779b5 / 96ab9a40 / 71a42eba / 4de64c89.
"""
from __future__ import annotations

import functools
import inspect
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.service_registry import get_runtime_service

logger = logging.getLogger("Core.PersonaIntegration")

#: Structured context key the engine reads at system role.
PERSONA_CONTEXT_KEY = "persona_system_prompt"

#: Attributes stamped on an installed wrapper. CP126 51b7a4a1: without an
#: installation marker or an unwrap handle, every call captured the previously
#: wrapped method and installed another wrapper — duplicating the persona
#: prompt and growing the call depth for the lifetime of the process.
_WRAPPER_MARKER = "__aura_persona_wrapper__"
_WRAPPER_ORIGINAL = "__aura_persona_original__"
_WRAPPER_PERSONA = "__aura_persona_name__"

#: Bound so a pathological profile cannot crowd out the rest of the prompt.
MAX_PERSONA_PROMPT_CHARS = 2000

#: Failures the wrapper isolates instead of propagating into cognition.
#: CP126 4de64c89: only OSError/ConnectionError/TimeoutError were caught, yet
#: prompt building and argument manipulation fail with TypeError, ValueError,
#: AttributeError, KeyError and RuntimeError.
_WRAPPER_ERRORS = (
    AttributeError,
    IndexError,
    KeyError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


@dataclass
class PersonaIntegrationReceipt:
    """What actually happened, rather than a bare bool.

    CP126 96ab9a40: an unavailable cognitive engine returned True after saying
    the adapter was "ready for later use", while installing no later hook — so
    callers were told integration succeeded when nothing was wrapped.
    """

    installed: bool
    persona: str
    reason: str = ""
    target: str = ""
    replaced_existing: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.installed

    def to_dict(self) -> dict[str, Any]:
        return {
            "installed": self.installed,
            "persona": self.persona,
            "reason": self.reason,
            "target": self.target,
            "replaced_existing": self.replaced_existing,
            "details": dict(self.details),
        }


def _resolve_engine() -> Any:
    """The live cognitive engine, from the registry.

    The old code imported a module-level ``cognitive_engine`` symbol that does
    not exist, so this integration has never wrapped anything in production —
    it took the ImportError branch and returned True.
    """
    return get_runtime_service("cognitive_engine", default=None)


def _persona_prompt(adapter: Any, persona_name: str) -> str:
    prompts = adapter.build_prompts(
        persona_name, "Respond in-character to the user's request."
    )
    if not isinstance(prompts, dict):
        return ""
    return str(prompts.get("system") or "").strip()[:MAX_PERSONA_PROMPT_CHARS]


def _augment_call(
    adapter: Any,
    persona_name: str,
    args: tuple,
    kwargs: dict,
) -> tuple[tuple, dict]:
    """Attach the persona as structured context, leaving the objective alone.

    CP126 ab3abbae and 71a42eba: the objective is never rewritten, so a
    positional call and a keyword call now take the identical path and the
    persona keeps its system-role standing instead of becoming user data.
    """
    system_prompt = _persona_prompt(adapter, persona_name)
    if not system_prompt:
        return args, kwargs

    if "context" in kwargs:
        existing = kwargs.get("context")
        context = dict(existing) if isinstance(existing, dict) else {}
        context.setdefault(PERSONA_CONTEXT_KEY, system_prompt)
        kwargs = dict(kwargs)
        kwargs["context"] = context
        return args, kwargs

    # think(objective, context, ...) — context may also arrive positionally.
    if len(args) >= 2 and isinstance(args[1], dict):
        context = dict(args[1])
        context.setdefault(PERSONA_CONTEXT_KEY, system_prompt)
        return (args[0], context) + tuple(args[2:]), kwargs

    kwargs = dict(kwargs)
    kwargs["context"] = {PERSONA_CONTEXT_KEY: system_prompt}
    return args, kwargs


def _build_wrapper(original: Callable, adapter: Any, persona_name: str) -> Callable:
    """Wrap ``original`` preserving its call protocol.

    CP126 481779b5: the old wrapper was ``def`` around an ``async def`` method,
    so ``inspect.iscoroutinefunction`` reported it synchronous and callers that
    branch on that returned an un-awaited coroutine object.
    """
    if inspect.iscoroutinefunction(original):

        @functools.wraps(original)
        async def persona_think(*args: Any, **kwargs: Any) -> Any:
            try:
                call_args, call_kwargs = _augment_call(adapter, persona_name, args, kwargs)
            except _WRAPPER_ERRORS as exc:
                record_degradation(
                    "persona_integration",
                    exc,
                    action="ran the thought without persona conditioning",
                )
                logger.warning("persona wrapper skipped conditioning: %s", exc)
                call_args, call_kwargs = args, kwargs
            return await original(*call_args, **call_kwargs)

    else:

        @functools.wraps(original)
        def persona_think(*args: Any, **kwargs: Any) -> Any:
            try:
                call_args, call_kwargs = _augment_call(adapter, persona_name, args, kwargs)
            except _WRAPPER_ERRORS as exc:
                record_degradation(
                    "persona_integration",
                    exc,
                    action="ran the thought without persona conditioning",
                )
                logger.warning("persona wrapper skipped conditioning: %s", exc)
                call_args, call_kwargs = args, kwargs
            return original(*call_args, **call_kwargs)

    setattr(persona_think, _WRAPPER_MARKER, True)
    setattr(persona_think, _WRAPPER_ORIGINAL, original)
    setattr(persona_think, _WRAPPER_PERSONA, persona_name)
    return persona_think


def _restore(target: Any, original: Callable) -> None:
    """Put the original method back, whichever way the wrapper was attached."""
    try:
        del target.__dict__["think"]
        if getattr(getattr(target, "think", None), _WRAPPER_MARKER, False):
            # A class-level wrapper survived the instance-level delete.
            target.think = original
    except (AttributeError, KeyError, TypeError):
        target.think = original


def initialize_persona_integration(
    persona_name: str = "aura", *, engine: Any = None
) -> PersonaIntegrationReceipt:
    """Install persona conditioning on the cognitive engine. Idempotent."""
    try:
        from .persona_adapter import PersonaAdapter

        adapter = PersonaAdapter()
        available = list(adapter.list_personas() or ())
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation("persona_integration", exc)
        logger.error("Persona adapter unavailable: %s", exc)
        return PersonaIntegrationReceipt(
            False, persona_name, reason=f"persona_adapter_unavailable: {exc}"
        )

    if persona_name not in available:
        logger.warning("Persona '%s' not found; skipping integration", persona_name)
        return PersonaIntegrationReceipt(
            False, persona_name, reason="persona_not_found",
            details={"available": available},
        )
    adapter.set_persona(persona_name)

    target = engine if engine is not None else _resolve_engine()
    if target is None:
        # Honest: nothing was wrapped, so nothing is conditioned.
        logger.info(
            "Persona '%s' selected but cognitive_engine is not registered; "
            "no think() wrapper installed",
            persona_name,
        )
        return PersonaIntegrationReceipt(
            False, persona_name, reason="cognitive_engine_not_registered"
        )

    think = getattr(target, "think", None)
    if not callable(think):
        return PersonaIntegrationReceipt(
            False, persona_name, reason="engine_has_no_think",
            target=type(target).__name__,
        )

    replaced = False
    if getattr(think, _WRAPPER_MARKER, False):
        existing_persona = getattr(think, _WRAPPER_PERSONA, "")
        original = getattr(think, _WRAPPER_ORIGINAL, None)
        if existing_persona == persona_name:
            return PersonaIntegrationReceipt(
                True, persona_name, reason="already_installed",
                target=type(target).__name__,
            )
        if original is None:
            return PersonaIntegrationReceipt(
                False, persona_name, reason="wrapper_present_without_unwrap_handle",
                target=type(target).__name__,
            )
        # Re-point at the ORIGINAL, never at the previous wrapper.
        think = original
        replaced = True

    try:
        target.think = _build_wrapper(think, adapter, persona_name)
    except (AttributeError, TypeError) as exc:
        record_degradation("persona_integration", exc)
        return PersonaIntegrationReceipt(
            False, persona_name, reason=f"think_not_assignable: {exc}",
            target=type(target).__name__,
        )

    logger.info(
        "Persona integration: %s.think wrapped for persona '%s'%s",
        type(target).__name__,
        persona_name,
        " (replacing a previous persona)" if replaced else "",
    )
    return PersonaIntegrationReceipt(
        True, persona_name, target=type(target).__name__, replaced_existing=replaced
    )


def uninstall_persona_integration(*, engine: Any = None) -> bool:
    """Remove the wrapper and restore the engine's own think(). Idempotent."""
    target = engine if engine is not None else _resolve_engine()
    if target is None:
        return False
    think = getattr(target, "think", None)
    original = getattr(think, _WRAPPER_ORIGINAL, None)
    if original is None:
        return False
    _restore(target, original)
    logger.info("Persona integration uninstalled from %s", type(target).__name__)
    return True


def persona_integration_status(*, engine: Any = None) -> dict[str, Any]:
    """Whether a wrapper is installed, and for which persona."""
    target = engine if engine is not None else _resolve_engine()
    think = getattr(target, "think", None) if target is not None else None
    installed = bool(getattr(think, _WRAPPER_MARKER, False))
    return {
        "engine_available": target is not None,
        "installed": installed,
        "persona": getattr(think, _WRAPPER_PERSONA, "") if installed else "",
        "target": type(target).__name__ if target is not None else "",
        "context_key": PERSONA_CONTEXT_KEY,
    }


if __name__ == "__main__":
    print(initialize_persona_integration().to_dict())
