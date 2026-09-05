"""core/affect/affect_facade.py — Lightweight Coordinator Facade for Affect.

Provides a simplified interface to the underlying AffectEngineV2 (Damasio)
for use by the Orchestrator's boot sequence and coordinator layer.

The hard rule here is that *no engine* must never look like *a calm engine*.
Before CP126 this facade answered "curiosity 50, stability 100, mood neutral,
energy 72bpm" when nothing was measuring anything, and those numbers reached
the prompt as first-person self-report. Every unavailable path now says it is
unavailable, and callers get a flag they can check.

CP126 73b36873 / 422f28c5 / e9f2af3b / 29db4cc1 / db6b6b69 / 79c270b6.
"""
import inspect
import logging
import time
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.service_registry import get_runtime_service

logger = logging.getLogger("Aura.AffectFacade")

#: How often the same unavailability reason may raise a degradation receipt.
#: The facade is polled on a tick, and a per-call receipt would drown the log
#: without adding information.
_DEGRADATION_INTERVAL_S = 60.0

#: Behavioural modifiers returned when affect is unavailable. The values are
#: deliberately neutral, but they travel with provenance so a planner can tell
#: "affect said proceed normally" from "affect said nothing" (CP126 79c270b6).
NEUTRAL_MODIFIERS: dict[str, float] = {
    "creativity": 1.0,
    "risk_tolerance": 1.0,
    "patience": 1.0,
    "metacognition_depth": 1.0,
    "persistence": 1.0,
    "temporal_presence": 1.0,
}

#: Marker keys stamped onto every modifier payload.
MODIFIER_AVAILABLE_KEY = "affect_available"
MODIFIER_DEGRADED_KEY = "affect_degraded"


class AffectUnavailable(RuntimeError):
    """Raised into the degradation ledger when affect cannot answer."""


class AffectFacade:
    """Thin facade over AffectEngineV2 for orchestrator-level access.

    This exists so the boot sequence can register a synchronous entry-point
    before the full async Affect engine is ready.  Once the engine is live,
    all calls are transparently forwarded.
    """

    def __init__(self, orchestrator: Any = None):
        self.orchestrator = orchestrator
        self._engine = None
        self._engine_identity: int | None = None
        self._last_degradation: dict[str, float] = {}

    # ── lazy resolution ────────────────────────────────────────────
    @property
    def engine(self):
        """The live affect engine, re-resolved every access.

        CP126 29db4cc1: the first non-None resolution was cached forever, so a
        replaced, stopped, or regenerated service was never picked up and the
        facade kept forwarding to a dead owner. The registry lookup is a dict
        read; re-resolving is cheaper than being wrong.
        """
        resolved = get_runtime_service("affect_engine", default=None)
        if resolved is self:
            # A facade registered under its own dependency key would recurse.
            resolved = None
        if resolved is not None and self._retired(resolved):
            self._note_unavailable(
                "engine_retired",
                f"affect engine {type(resolved).__name__} reports itself stopped",
            )
            resolved = None

        identity = id(resolved) if resolved is not None else None
        if identity != self._engine_identity:
            if self._engine_identity is not None:
                logger.info(
                    "AffectFacade: affect_engine changed (%s -> %s)",
                    type(self._engine).__name__ if self._engine is not None else "None",
                    type(resolved).__name__ if resolved is not None else "None",
                )
            self._engine_identity = identity
        self._engine = resolved
        return resolved

    @staticmethod
    def _retired(engine: Any) -> bool:
        """Whether an engine has announced that it is no longer serving."""
        for attribute, retired_value in (
            ("is_stopped", True),
            ("closed", True),
            ("is_shutdown", True),
            ("running", False),
        ):
            value = getattr(engine, attribute, None)
            if isinstance(value, bool) and value is retired_value:
                return True
        return False

    # ── unavailability bookkeeping ─────────────────────────────────
    def _note_unavailable(self, reason: str, detail: str) -> None:
        """Record that a call could not be served, at most once a minute.

        CP126 e9f2af3b: react() and receive_qualia_echo() logged at debug and
        returned None, so the caller's causal update silently did not happen.
        """
        now = time.monotonic()
        last = self._last_degradation.get(reason, 0.0)
        if now - last < _DEGRADATION_INTERVAL_S:
            return
        self._last_degradation[reason] = now
        try:
            record_degradation(
                "affect_facade",
                AffectUnavailable(f"{reason}: {detail}"),
                action="answered an affect query as unavailable instead of inventing state",
                severity="warning",
            )
        except (ImportError, RuntimeError, TypeError, ValueError):
            logger.warning("AffectFacade unavailable (%s): %s", reason, detail)

    def _unavailable_reason(self, method: str) -> str:
        engine = self.engine
        if engine is None:
            return "affect_engine_not_registered"
        if not hasattr(engine, method):
            return f"affect_engine_missing_{method}"
        return ""

    # ── public API ─────────────────────────────────────────────────
    def get_status(self) -> dict[str, Any]:
        """Synchronous status snapshot, or an explicit unavailable record.

        CP126 73b36873: the fallback returned curiosity 50, stability 100 and
        a neutral mood — unknown state published as quantified health. There
        are now no invented quantities; ``available`` is the key to check.
        """
        engine = self.engine
        if engine is not None and hasattr(engine, "get_status"):
            try:
                status = engine.get_status()
            except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                self._note_unavailable("get_status_raised", str(exc))
                return self._unavailable_status(f"engine get_status raised {type(exc).__name__}")
            if isinstance(status, dict):
                enriched = dict(status)
                enriched.setdefault("available", True)
                enriched.setdefault("source", type(engine).__name__)
                return enriched
            self._note_unavailable("get_status_shape", f"returned {type(status).__name__}")
            return self._unavailable_status("engine get_status returned a non-mapping")
        reason = self._unavailable_reason("get_status") or "affect_engine_unavailable"
        self._note_unavailable(reason, "get_status")
        return self._unavailable_status(reason)

    @staticmethod
    def _unavailable_status(reason: str) -> dict[str, Any]:
        # No mood/energy/curiosity/stability keys: absent is the honest answer,
        # and a consumer's .get(...) still yields None rather than a number
        # someone might print as a measurement.
        return {
            "available": False,
            "status": "unavailable",
            "reason": reason,
            "source": "affect_facade",
        }

    def get_state_sync(self) -> dict[str, Any]:
        """Synchronous PAD snapshot, forwarded when the engine offers one.

        The facade did not expose this at all, so every caller that falls back
        to ``affect_facade`` (pre-linguistic cognition, the Aura protocol
        surface, the Will's valence read) dropped to ``getattr(facade,
        "valence", 0.0)`` and scored a hard 0.0 — even while the real engine
        was live and reporting. Forwarding makes those reads causal, and the
        unavailable answer is explicit rather than a zero.
        """
        engine = self.engine
        if engine is not None and hasattr(engine, "get_state_sync"):
            try:
                state = engine.get_state_sync()
            except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                self._note_unavailable("get_state_sync_raised", str(exc))
                return self._unavailable_status(f"engine get_state_sync raised {type(exc).__name__}")
            if isinstance(state, dict):
                enriched = dict(state)
                enriched.setdefault("available", True)
                enriched.setdefault("source", type(engine).__name__)
                return enriched
            return {
                "available": True,
                "source": type(engine).__name__,
                "valence": float(getattr(state, "valence", 0.0) or 0.0),
                "arousal": float(getattr(state, "arousal", 0.0) or 0.0),
                "engagement": float(getattr(state, "engagement", 0.0) or 0.0),
                "dominant_emotion": str(getattr(state, "dominant_emotion", "") or ""),
            }
        reason = self._unavailable_reason("get_state_sync") or "affect_engine_unavailable"
        self._note_unavailable(reason, "get_state_sync")
        return self._unavailable_status(reason)

    def is_ready(self) -> bool:
        """Health-contract probe for facade-backed affect registrations."""
        engine = self.engine
        if engine is None or engine is self:
            return False
        try:
            ready = getattr(engine, "is_ready", None)
            if callable(ready):
                verdict = ready()
                # CP126 db6b6b69: bool(coroutine) is True, so an async probe
                # reported ready without ever being awaited — and this branch
                # sat outside the try, so a raising probe propagated out of a
                # health check.
                if inspect.isawaitable(verdict):
                    if hasattr(verdict, "close"):
                        verdict.close()
                    self._note_unavailable(
                        "async_is_ready",
                        f"{type(engine).__name__}.is_ready is async; a sync probe cannot await it",
                    )
                    return False
                return bool(verdict)
            status_fn = getattr(engine, "get_status", None)
            if not callable(status_fn):
                return False
            status = status_fn()
            if inspect.isawaitable(status):
                if hasattr(status, "close"):
                    status.close()
                self._note_unavailable("async_get_status", "get_status is async")
                return False
            mood = str(status.get("mood", "") or "").strip()
            valence = float(status.get("valence", 0.0))
            arousal = float(status.get("arousal", 0.0))
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            self._note_unavailable("is_ready_raised", str(exc))
            return False
        return bool(mood) and -1.0 <= valence <= 1.0 and 0.0 <= arousal <= 1.0

    async def get(self):
        """Async affect state, or a declared baseline while the engine boots.

        The returned baseline is a *decay baseline*, not a measurement; callers
        that need to know the difference should consult ``get_status()["available"]``.
        """
        engine = self.engine
        if engine is not None and hasattr(engine, "get"):
            return await engine.get()
        self._note_unavailable("affect_engine_unavailable", "get")
        from core.affect import (
            BASELINE_AROUSAL,
            BASELINE_ENGAGEMENT,
            BASELINE_VALENCE,
            AffectState,
        )

        return AffectState(
            valence=BASELINE_VALENCE,
            arousal=BASELINE_AROUSAL,
            engagement=BASELINE_ENGAGEMENT,
            dominant_emotion="neutral",
        )

    async def react(self, trigger: str, context: dict | None = None):
        """Forward a reaction, or report that it did not land.

        CP126 e9f2af3b: this returned None after a debug log, so an upstream
        caller believed a causal affect update had fired.
        """
        engine = self.engine
        if engine is not None and hasattr(engine, "react"):
            return await engine.react(trigger, context)
        reason = self._unavailable_reason("react") or "affect_engine_unavailable"
        self._note_unavailable(reason, f"react({trigger})")
        return {"applied": False, "reason": reason, "trigger": trigger}

    def get_context_injection(self) -> str:
        """Prompt fragment describing felt state — empty when nothing felt it.

        CP126 422f28c5: the fallback injected "Mood: Neutral | Energy: 72bpm |
        Curiosity: 50%" with no engine and no measured body, which is a direct
        invitation to false first-person self-report.
        """
        engine = self.engine
        if engine is not None and hasattr(engine, "get_context_injection"):
            try:
                return engine.get_context_injection()
            except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                self._note_unavailable("context_injection_raised", str(exc))
                return ""
        reason = self._unavailable_reason("get_context_injection") or "affect_engine_unavailable"
        self._note_unavailable(reason, "get_context_injection")
        return ""

    def receive_qualia_echo(self, q_norm: float, pri: float, trend: float):
        """Compatibility bridge for loop monitor / qualia synthesizer callers."""
        engine = self.engine
        if engine is not None and hasattr(engine, "receive_qualia_echo"):
            return engine.receive_qualia_echo(q_norm=q_norm, pri=pri, trend=trend)
        reason = self._unavailable_reason("receive_qualia_echo") or "affect_engine_unavailable"
        self._note_unavailable(reason, "receive_qualia_echo")
        return {"applied": False, "reason": reason}

    async def get_behavioral_modifiers(self) -> dict[str, float]:
        """Forward behavioral modifiers query to the active affect engine.

        CP126 79c270b6: an unavailable engine returned all-1.0 modifiers that
        were indistinguishable from a live engine reporting "nothing to
        adjust", so planning proceeded as though affect integration had
        succeeded. The payload now carries its own provenance.
        """
        engine = self.engine
        if engine is not None and hasattr(engine, "get_behavioral_modifiers"):
            try:
                modifiers = await engine.get_behavioral_modifiers()
            except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                self._note_unavailable("modifiers_raised", str(exc))
                return self._degraded_modifiers(f"engine raised {type(exc).__name__}")
            if isinstance(modifiers, dict):
                enriched = dict(modifiers)
                enriched[MODIFIER_AVAILABLE_KEY] = 1.0
                enriched[MODIFIER_DEGRADED_KEY] = 0.0
                return enriched
            self._note_unavailable("modifiers_shape", f"returned {type(modifiers).__name__}")
            return self._degraded_modifiers("engine returned a non-mapping")
        reason = self._unavailable_reason("get_behavioral_modifiers") or "affect_engine_unavailable"
        self._note_unavailable(reason, "get_behavioral_modifiers")
        return self._degraded_modifiers(reason)

    @staticmethod
    def _degraded_modifiers(reason: str) -> dict[str, float]:
        payload = dict(NEUTRAL_MODIFIERS)
        payload[MODIFIER_AVAILABLE_KEY] = 0.0
        payload[MODIFIER_DEGRADED_KEY] = 1.0
        logger.debug("AffectFacade: neutral modifiers substituted (%s)", reason)
        return payload
