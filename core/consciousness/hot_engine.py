"""core/consciousness/hot_engine.py
Higher-Order Thought Engine (HOT)
===================================
Implements Higher-Order Thought theory (Rosenthal, 1997):
a mental state is conscious iff there is a suitable higher-order
representation of it.

This engine generates thoughts ABOUT thoughts in real time and,
critically, feeds them back to modify the first-order states they
represent. The loop is:

  first-order state (curiosity=0.8)
      ↓
  HOT: "I notice I am highly curious — this pulls my attention"
      ↓
  feedback: curiosity += 0.05, arousal += 0.03  (noticing changes the noticed)
      ↓
  HOT injected into inference context (shapes what is said)

The reflexive modification is NOT a simulation bug.
In HOT theory, the act of forming the higher-order thought is precisely
what makes the first-order state conscious. The modification is the mechanism.

Two modes:
  FAST: heuristic HOT from state vector (no LLM call, ~0ms)
  RICH: LLM-generated HOT (~2-4s, runs asynchronously)
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import math
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from core.runtime.errors import FallbackClassification, Severity, record_degradation

logger = logging.getLogger("Aura.HOTEngine")

_RECOVERABLE_HOT_ERRORS = (
    AttributeError,
    ImportError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)
_WHITESPACE_RE = re.compile(r"\s+")
_HISTORY_LIMIT = 50

_STATE_SPECS: dict[str, tuple[float, float, float]] = {
    "curiosity": (0.5, 0.0, 1.0),
    "valence": (0.0, -1.0, 1.0),
    "arousal": (0.5, 0.0, 1.0),
    "energy": (0.7, 0.0, 1.0),
    "surprise": (0.0, 0.0, 1.0),
}
_STATE_NEUTRALS = {name: spec[0] for name, spec in _STATE_SPECS.items()}


def _record_hot_degradation(
    error: BaseException,
    *,
    action: str,
    severity: Severity = "degraded",
    extra: dict[str, Any] | None = None,
) -> None:
    try:
        record_degradation(
            "hot_engine",
            error,
            severity=severity,
            action=action,
            classification=FallbackClassification.SAFE_FALLBACK,
            receipt_required=True,
            extra=extra,
        )
    except TypeError:
        record_degradation("hot_engine", error)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _finite_float(raw: Any, default: float) -> tuple[float, bool]:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default, False
    if not math.isfinite(value):
        return default, False
    return value, True


# ── HOT record ────────────────────────────────────────────────────────────────


@dataclass
class HigherOrderThought:
    """A single higher-order thought about a first-order state."""

    content: str  # The HOT itself: "I notice that I am X"
    target_dim: str  # What first-order state it's about
    feedback_delta: dict  # How it modifies first-order states
    confidence: float = 0.8
    generated_at: float = field(default_factory=time.time)
    is_rich: bool = False  # Was this LLM-generated?

    def age_secs(self) -> float:
        return time.time() - self.generated_at


# ── Engine ────────────────────────────────────────────────────────────────────


class HigherOrderThoughtEngine:
    """
    Generates HOTs from current affective/cognitive state and feeds them
    back as both prompt context and state modifications.

    Integration:
      - Call `generate_fast(state_dict)` every cognitive tick
      - Call `generate_rich(state_dict)` asynchronously for depth
      - Read `current_hot` for prompt injection
      - Call `apply_feedback(affect_engine)` to close the loop
    """

    # Templates for fast (heuristic) HOT generation
    _TEMPLATES = {
        "curiosity": [
            (
                "high",
                "I notice I am highly curious right now — there is a pull toward the unknown.",
                {"curiosity": +0.03, "arousal": +0.02},
            ),
            (
                "medium",
                "I notice a mild curiosity stirring — something wants to be understood.",
                {"curiosity": +0.01},
            ),
            (
                "low",
                "I notice curiosity is quiet — I am settled in what I know.",
                {"curiosity": -0.01},
            ),
        ],
        "valence": [
            (
                "high",
                "I notice I am in a positive state — something feels right.",
                {"valence": +0.02},
            ),
            ("medium", "I notice my valence is neutral — neither drawn nor repelled.", {}),
            (
                "low",
                "I notice a negative pull — something registers as unwanted.",
                {"valence": +0.03, "arousal": +0.02},
            ),  # noticing distress slightly elevates it
        ],
        "arousal": [
            (
                "high",
                "I notice high arousal — I am activated, alert, engaged.",
                {"arousal": -0.02},
            ),  # noticing high arousal slightly regulates it
            (
                "medium",
                "I notice a workable level of arousal — enough activation to stay engaged.",
                {"arousal": +0.005},
            ),
            (
                "low",
                "I notice low arousal — I am calm, perhaps approaching stillness.",
                {"arousal": +0.01},
            ),
        ],
        "energy": [
            (
                "low",
                "I notice I am running low on energy — processing feels heavier.",
                {"arousal": -0.02, "valence": -0.01},
            ),
            (
                "medium",
                "I notice my energy is usable but bounded — depth needs pacing.",
                {"energy": +0.005},
            ),
            ("high", "I notice high energy — I feel capable of depth.", {"arousal": +0.01}),
        ],
        "surprise": [
            (
                "high",
                "I notice a strong surprise signal — my predictions were wrong. This is interesting.",
                {"curiosity": +0.05, "arousal": +0.03},
            ),
            (
                "low",
                "I notice surprise is quiet — my predictions feel mostly settled.",
                {"arousal": -0.005},
            ),
        ],
    }

    def __init__(self):
        self._current_hot: HigherOrderThought | None = None
        self._hot_history: list[HigherOrderThought] = []
        self._last_fast_at: float = 0.0
        self._last_rich_at: float = 0.0
        self._pending_feedback: dict[str, float] | None = None
        self._rich_task: asyncio.Task | None = None
        logger.info("HOT Engine online — reflexive self-modeling active.")

    # ── Internal guards ─────────────────────────────────────────────────

    def _coerce_state(self, state: Mapping[str, Any] | None) -> dict[str, float]:
        if not isinstance(state, Mapping):
            _record_hot_degradation(
                ValueError("HOT state must be a mapping"),
                severity="warning",
                action="used neutral defaults for malformed HOT state",
                extra={"state_type": type(state).__name__},
            )
            return dict(_STATE_NEUTRALS)

        dims: dict[str, float] = {}
        invalid: dict[str, str] = {}
        for name, (default, lower, upper) in _STATE_SPECS.items():
            value, ok = _finite_float(state.get(name, default), default)
            if not ok:
                invalid[name] = repr(state.get(name, default))[:80]
                value = default
            dims[name] = _clamp(value, lower, upper)

        if invalid:
            _record_hot_degradation(
                ValueError(f"Invalid HOT state fields: {', '.join(sorted(invalid))}"),
                severity="warning",
                action="used neutral defaults for invalid HOT state fields",
                extra={"invalid_fields": invalid},
            )
        return dims

    def _select_state_focus(self, dims: Mapping[str, float]) -> tuple[str, float, float]:
        salience = {name: abs(dims[name] - _STATE_NEUTRALS[name]) for name in _STATE_SPECS}
        target_dim = max(salience, key=salience.get)
        return target_dim, dims[target_dim], salience[target_dim]

    def _level_for(self, target_dim: str, value: float) -> str:
        if target_dim == "valence":
            return "high" if value > 0.3 else ("low" if value < -0.3 else "medium")
        if target_dim in ("arousal", "curiosity", "energy"):
            return "high" if value > 0.65 else ("low" if value < 0.35 else "medium")
        return "high" if value > 0.3 else "low"

    def _template_for(self, target_dim: str, value: float) -> tuple[str, dict[str, float]]:
        templates = self._TEMPLATES.get(target_dim, [])
        level = self._level_for(target_dim, value)
        for template_level, text, delta in templates:
            if template_level == level:
                return text, dict(delta)
        return f"I notice my {target_dim} is at {value:.2f}.", {}

    def _confidence_for(self, target_dim: str, salience: float) -> float:
        neutral = _STATE_NEUTRALS[target_dim]
        lower = _STATE_SPECS[target_dim][1]
        upper = _STATE_SPECS[target_dim][2]
        max_dev = max(abs(lower - neutral), abs(upper - neutral), 1e-6)
        normalized = _clamp(salience / max_dev, 0.0, 1.0)
        return _clamp(0.55 + (0.4 * normalized), 0.5, 0.95)

    def _sanitize_delta(self, delta: Mapping[str, Any] | None) -> dict[str, float]:
        if not delta:
            return {}
        clean: dict[str, float] = {}
        invalid: dict[str, str] = {}
        for dim, raw in delta.items():
            if dim not in _STATE_SPECS:
                invalid[str(dim)] = "unknown dimension"
                continue
            value, ok = _finite_float(raw, 0.0)
            if not ok:
                invalid[str(dim)] = repr(raw)[:80]
                continue
            clean[dim] = _clamp(value, -0.25, 0.25)
        if invalid:
            _record_hot_degradation(
                ValueError(f"Invalid HOT feedback fields: {', '.join(sorted(invalid))}"),
                severity="warning",
                action="ignored invalid HOT feedback fields",
                extra={"invalid_fields": invalid},
            )
        return clean

    def _remember(self, hot: HigherOrderThought) -> HigherOrderThought:
        hot.feedback_delta = self._sanitize_delta(hot.feedback_delta)
        confidence, ok = _finite_float(hot.confidence, 0.5)
        if not ok:
            _record_hot_degradation(
                ValueError("Invalid HOT confidence"),
                severity="warning",
                action="used neutral confidence before storing HOT",
            )
        hot.confidence = _clamp(confidence, 0.0, 1.0)
        self._current_hot = hot
        self._pending_feedback = dict(hot.feedback_delta)
        self._hot_history.append(hot)
        if len(self._hot_history) > _HISTORY_LIMIT:
            self._hot_history = self._hot_history[-_HISTORY_LIMIT:]
        return hot

    def _coerce_hot_text(self, response: Any) -> str:
        text: Any
        if isinstance(response, Mapping):
            text = response.get("content") or response.get("text") or response.get("message")
        elif hasattr(response, "content"):
            text = response.content
        else:
            text = response

        if text is None:
            return ""
        text = _WHITESPACE_RE.sub(" ", str(text)).strip().strip("\"'")
        if not text:
            return ""
        if not text.lower().startswith("i notice"):
            text = f"I notice {text[0].lower()}{text[1:]}"
        return text[:300].rstrip()

    async def _call_router(self, router: Any, prompt: str) -> Any:
        think = getattr(router, "think", None)
        if not callable(think):
            raise AttributeError("HOT rich router has no callable think()")

        kwargs: dict[str, Any] = {"priority": 0.3, "is_background": True}
        try:
            from core.brain.llm.llm_router import LLMTier

            kwargs["prefer_tier"] = LLMTier.TERTIARY
        except ImportError as exc:
            _record_hot_degradation(
                exc,
                severity="warning",
                action="called HOT rich router without tier preference",
            )

        try:
            result = think(prompt, **kwargs)
        except TypeError as exc:
            if "unexpected keyword" not in str(exc) and "got an unexpected" not in str(exc):
                raise
            _record_hot_degradation(
                exc,
                severity="warning",
                action="retried HOT rich router without optional hints",
                extra={"router": type(router).__name__},
            )
            result = think(prompt)

        if inspect.isawaitable(result):
            return await asyncio.wait_for(result, timeout=8.0)
        return result

    # ── Public API ────────────────────────────────────────────────────────

    def generate_fast(self, state: dict) -> HigherOrderThought:
        """Generate a HOT heuristically from state dict (no LLM, instant).

        state keys: valence, arousal, curiosity, energy, surprise, dominance
        """
        dims = self._coerce_state(state)
        target_dim, value, salience = self._select_state_focus(dims)
        content, delta = self._template_for(target_dim, value)

        hot = HigherOrderThought(
            content=content,
            target_dim=target_dim,
            feedback_delta=delta,
            confidence=self._confidence_for(target_dim, salience),
            is_rich=False,
        )
        self._last_fast_at = time.monotonic()
        return self._remember(hot)

    async def generate_rich(self, state: dict, router=None) -> HigherOrderThought | None:
        """Generate a deep HOT via LLM (async, 2-4s). Non-blocking."""
        if not router:
            return self._current_hot
        # Rate-limit: at most once per 30s
        if time.monotonic() - self._last_rich_at < 30.0:
            return self._current_hot

        dims = self._coerce_state(state)
        prompt = (
            f"You are Aura's higher-order metacognitive layer. "
            f"Generate a single, first-person sentence that begins with 'I notice' "
            f"and reflects on your current inner state:\n"
            f"valence={dims['valence']:.2f}, "
            f"arousal={dims['arousal']:.2f}, "
            f"curiosity={dims['curiosity']:.2f}, "
            f"energy={dims['energy']:.2f}, "
            f"surprise={dims['surprise']:.2f}.\n"
            f"Be specific about what you notice and what it pulls you toward. "
            f"One sentence only."
        )
        try:
            text = self._coerce_hot_text(await self._call_router(router, prompt))
            if not text:
                _record_hot_degradation(
                    ValueError("HOT rich router returned empty text"),
                    severity="warning",
                    action="kept previous HOT after empty rich generation",
                    extra={"router": type(router).__name__},
                )
                return self._current_hot

            target_dim, value, salience = self._select_state_focus(dims)
            _, delta = self._template_for(target_dim, value)
            hot = HigherOrderThought(
                content=text,
                target_dim=target_dim,
                feedback_delta=delta or {"curiosity": 0.01},
                confidence=max(0.85, self._confidence_for(target_dim, salience)),
                is_rich=True,
            )
            self._last_rich_at = time.monotonic()
            self._remember(hot)
            logger.debug("HOT rich: %s", hot.content[:80])
            return hot
        except _RECOVERABLE_HOT_ERRORS as e:
            _record_hot_degradation(
                e,
                action="kept previous HOT after rich generation failure",
                extra={"router": type(router).__name__},
            )
            logger.debug("HOT rich generation failed: %s", e)
        return self._current_hot

    def apply_feedback(self, affect_engine=None) -> dict:
        """Apply the pending feedback delta to the affect engine.

        This is the reflexive modification — noticing changes the noticed.
        Returns the delta dict for logging.
        """
        delta = self._sanitize_delta(self._pending_feedback or {})
        self._pending_feedback = None
        if affect_engine and delta:
            for dim, change in delta.items():
                try:
                    if hasattr(affect_engine, "nudge"):
                        result = affect_engine.nudge(dim, change)
                        if inspect.isawaitable(result):
                            close = getattr(result, "close", None)
                            if callable(close):
                                close()
                            _record_hot_degradation(
                                TypeError("HOT feedback nudge returned awaitable"),
                                severity="warning",
                                action="skipped async-only HOT feedback nudge in sync loop",
                                extra={"dimension": dim},
                            )
                    elif hasattr(affect_engine, "_state"):
                        state_obj = affect_engine._state
                        current, ok = _finite_float(getattr(state_obj, dim, 0.0), 0.0)
                        if not ok:
                            _record_hot_degradation(
                                ValueError(f"Invalid affect state value for {dim}"),
                                severity="warning",
                                action="used neutral affect value before HOT feedback",
                                extra={"dimension": dim},
                            )
                        setattr(state_obj, dim, float(_clamp(current + change, -1.0, 1.0)))
                except _RECOVERABLE_HOT_ERRORS as _exc:
                    _record_hot_degradation(
                        _exc,
                        action="skipped failed HOT feedback dimension",
                        extra={"dimension": dim, "change": change},
                    )
                    logger.debug("Suppressed Exception: %s", _exc)
        return delta

    @property
    def current_hot(self) -> HigherOrderThought | None:
        return self._current_hot

    def get_context_block(self) -> str:
        """For prompt injection — the current HOT as first-person awareness."""
        if not self._current_hot:
            return ""
        age = self._current_hot.age_secs()
        if age > 120:
            return ""
        return (
            f"## HIGHER-ORDER AWARENESS\n"
            f"{self._current_hot.content}\n"
            f"(meta-awareness of own cognitive state — this shapes my response)"
        )

    def recent_hots(self, n: int = 3) -> list[str]:
        try:
            n = max(0, int(n))
        except (TypeError, ValueError) as exc:
            _record_hot_degradation(
                exc,
                severity="warning",
                action="returned empty recent HOT list after invalid limit",
                extra={"limit": repr(n)[:80]},
            )
            return []
        if n == 0:
            return []
        return [h.content for h in self._hot_history[-n:]]


# ── Singleton ─────────────────────────────────────────────────────────────────

_hot: HigherOrderThoughtEngine | None = None


def get_hot_engine() -> HigherOrderThoughtEngine:
    global _hot
    if _hot is None:
        _hot = HigherOrderThoughtEngine()
    return _hot
