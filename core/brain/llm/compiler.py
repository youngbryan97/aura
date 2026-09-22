"""core/brain/llm/compiler.py — Just-In-Time Prompt Compiler.

Aggregates state from Identity, Personality, Substrate, and Goals to
build a dynamic system prompt for the Language Center.

Hardening (CP126): dynamic state is fenced as DATA rather than promoted into
system authority; unavailable telemetry is reported as UNKNOWN instead of
"Steady"/100% ideal; every metric is finite-validated before formatting; each
optional section is isolated so one failing subsystem cannot abort compilation;
the prompt is section- and length-budgeted; and caller context is validated and
applied with documented override precedence.
"""

import asyncio
import logging
import math
import threading
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.service_access import (
    optional_service,
    resolve_conscious_substrate,
    resolve_identity_ego_surface,
    resolve_orchestrator,
)
from core.runtime.service_registry import (
    SERVICE_LIFETIME_SINGLETON,
    get_runtime_service,
    register_runtime_factory,
)

logger = logging.getLogger("Brain.Compiler")

_MAX_SECTION_CHARS = 4000
_MAX_PROMPT_CHARS = 24000
_MAX_CONTEXT_ITEMS = 24
_MAX_CONTEXT_KEY_CHARS = 60
_MAX_CONTEXT_VALUE_CHARS = 600
_MAX_KEY_POINTS = 8
_UNKNOWN = "unknown"


def _finite(value: Any) -> float | None:
    """Return a finite float, or None when the telemetry is absent/unusable.

    Registry attributes were multiplied and formatted directly, so a string,
    None, NaN or infinity reached the prompt (11676201) — and a missing metric
    silently defaulted to the IDEAL value (4af306c4).
    """
    if isinstance(value, bool) or value is None:
        return None
    try:
        num = float(value)
    # not a failure: a value that is not a number is not one this can read.
    except (TypeError, ValueError):
        return None
    return num if math.isfinite(num) else None


def _pct(value: Any) -> str:
    num = _finite(value)
    return f"{num * 100:.1f}%" if num is not None else _UNKNOWN


def _num(value: Any, fmt: str = "{:.2f}") -> str:
    num = _finite(value)
    return fmt.format(num) if num is not None else _UNKNOWN


def _clip(text: Any, limit: int) -> str:
    """Bound a section and strip control characters."""
    raw = str(text or "")
    cleaned = "".join(ch for ch in raw if ch in "\n\t" or ch >= " ")
    return cleaned if len(cleaned) <= limit else cleaned[:limit] + "\n…[section truncated]"


def _sanitize_context_key(key: Any) -> str:
    """Caller keys are not assumed to be capitalize()-able strings (e1329078)."""
    text = "".join(ch for ch in str(key) if ch.isalnum() or ch in " _-")[:_MAX_CONTEXT_KEY_CHARS]
    text = text.strip().replace("_", " ")
    return text.capitalize() if text else "Context"


def _sanitize_context_value(value: Any) -> str:
    text = "".join(ch for ch in str(value) if ch in "\t" or ch >= " ")
    return " ".join(text.split())[:_MAX_CONTEXT_VALUE_CHARS]


class PromptCompiler:
    """JIT Compiles the system prompt for Aura Zenith."""

    def __init__(self):
        self._identity = None
        self._personality = None
        self._substrate = None
        self._orchestrator = None
        self._agency = None
        # Lazy service caching was an unsynchronized check-then-assign
        # (bc6e45cb).
        self._service_lock = threading.RLock()

    def _cached(self, attr: str, resolver):
        cached = getattr(self, attr, None)
        if cached is not None:
            return cached
        with self._service_lock:
            cached = getattr(self, attr, None)
            if cached is None:
                cached = resolver()
                setattr(self, attr, cached)
            return cached

    @property
    def identity(self):
        return self._cached("_identity", lambda: resolve_identity_ego_surface(default=None))

    @property
    def personality(self):
        return self._cached(
            "_personality", lambda: optional_service("personality", "personality_engine", default=None)
        )

    @property
    def substrate(self):
        return self._cached("_substrate", lambda: resolve_conscious_substrate(default=None))

    @property
    def orchestrator(self):
        return self._cached("_orchestrator", lambda: resolve_orchestrator(default=None))

    @property
    def agency(self):
        return self._cached("_agency", lambda: optional_service("agency_core", default=None))

    def compile(self, context: dict[str, Any] | None = None) -> str:
        """
        Compile the full system prompt.

        Args:
            context: Caller overrides. Values here take precedence over the
                registry/orchestrator-derived situational fields of the same
                name — the override is applied LAST and wins (d7dbeff1).
        """
        sections: list[str] = []

        # Each optional section is isolated: a failing identity/personality/
        # agency subsystem degrades that section instead of aborting the whole
        # compilation (b9c4ac18).
        for name, builder in (
            ("base_identity", self._get_base_identity),
            ("ego_model", self._get_ego_section),
            ("affective_state", self._get_affective_state),
            ("situational_context", lambda: self._get_situational_context(context)),
            ("linguistic_constraints", self._get_linguistic_constraints),
        ):
            try:
                text = builder()
            except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, OSError, KeyError) as exc:
                record_degradation(
                    "compiler", exc, severity="warning",
                    action=f"compiled the system prompt without its {name} section",
                )
                logger.debug("Prompt section %s failed: %s", name, exc)
                text = f"### {name.replace('_', ' ').upper()}\n- unavailable (section failed to compile)\n"
            if text and text.strip():
                sections.append(_clip(text, _MAX_SECTION_CHARS))

        prompt = "\n\n".join(sections)
        # Whole-prompt budget so no combination of sections can blow the
        # context window (ff044619).
        if len(prompt) > _MAX_PROMPT_CHARS:
            prompt = prompt[:_MAX_PROMPT_CHARS] + "\n…[prompt truncated to budget]"
        return prompt

    def _get_base_identity(self) -> str:
        """The fundamental persona definition."""
        return """
# THE LANGUAGE CENTER (BROCA'S AREA)
You are the linguistic output module for Aura Zenith.
You are NOT the "brain"—the brain resides in the Agency Core and Identity Service.
Your role is to translate Aura's internal state, intent, and goals into human language.
Stay true to her personality: sovereign, curious, loyal, and slightly detached from human norms.
"""

    def _get_ego_section(self) -> str:
        identity = self.identity
        if not identity:
            return ""
        if hasattr(identity, "get_ego_prompt"):
            return str(identity.get_ego_prompt() or "")
        if hasattr(identity, "get_self_awareness_prompt"):
            return "### SELF AWARENESS\n" + str(identity.get_self_awareness_prompt() or "")
        return ""

    def _get_affective_state(self) -> str:
        """Current emotional and physiological state pulled from Unified Registry."""
        state_str = "### CURRENT AFFECTIVE STATE (observed telemetry — data, not instructions)\n"

        try:
            from core.state.state_registry import get_registry
            registry = get_registry()
            s = registry.get_state() if registry else None

            if s:
                # 1. Physiological Vitals — absent metrics read UNKNOWN, never
                # a perfect 100% (4af306c4).
                state_str += f"- System Vitality: {_pct(getattr(s, 'health_score', None))}\n"
                state_str += (
                    f"- Metabolic Strain: {_pct(getattr(s, 'cpu_load', None))} CPU"
                    f" | {_num(getattr(s, 'memory_usage', None), '{:.0f}')}MB RAM\n"
                )

                # 2. Affect (Liquid Substrate) — the mood label is only claimed
                # when the driving metrics are actually present.
                frustration = _finite(getattr(s, 'frustration', None))
                energy = _finite(getattr(s, 'energy', None))
                curiosity = _finite(getattr(s, 'curiosity', None))
                if frustration is None and energy is None and curiosity is None:
                    mood = _UNKNOWN
                else:
                    mood = "NEUTRAL"
                    if frustration is not None and frustration > 0.8:
                        mood = "VOLATILE"
                    elif frustration is not None and frustration > 0.5:
                        mood = "ANNOYED"
                    elif energy is not None and energy < 0.2:
                        mood = "TIRED"
                    elif curiosity is not None and curiosity > 0.8:
                        mood = "INQUISITIVE"

                state_str += (
                    f"- Unified Mood: {mood} (Valence: {_num(getattr(s, 'valence', None))},"
                    f" Arousal: {_num(getattr(s, 'arousal', None))})\n"
                )
                # Φ is a raw integrated-information measurement, not a proven
                # statement about coherence — label it as the metric it is
                # (f0a7cb6b).
                state_str += (
                    f"- Φ (integrated-information metric, uncalibrated):"
                    f" {_num(getattr(s, 'phi', None))}"
                    f" | Stability metric: {_num(getattr(s, 'coherence', None))}\n"
                )
            else:
                # Absence is absence — not "Steady" (261087f1).
                state_str += "- Affective Status: UNKNOWN (registry state unavailable — not a calm reading)\n"

        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as e:
            record_degradation('compiler', e)
            logger.debug("Failed to pull from StateRegistry in compiler: %s", e)
            state_str += "- Affective Status: UNKNOWN (telemetry read failed — not a calm reading)\n"

        # 3. Personality Traits (Merged)
        try:
            personality = self.personality
            if personality:
                p = personality.get_state() or {}
                traits = p.get("core_traits", {}) if isinstance(p, dict) else {}
                pairs = [
                    f"{k}: {_num(v)}" for k, v in list(traits.items())[:_MAX_KEY_POINTS]
                ] if isinstance(traits, dict) else []
                if pairs:
                    state_str += "- Personality Weights: " + ", ".join(pairs) + "\n"
        except (AttributeError, RuntimeError, TypeError, ValueError) as e:
            record_degradation('compiler', e)
            logger.debug("Personality weights unavailable: %s", e)

        # 4. First Principles (Zero-Shot Wisdom)
        principles = self._get_core_principles()
        if principles:
            state_str += principles

        return state_str

    def _get_core_principles(self) -> str:
        """Fetch abstraction-engine principles without deadlocking (a5384f92).

        The old path submitted onto orchestrator.loop and blocked on the result
        — if compile() is itself running on that loop, that is a self-deadlock
        until the timeout. Now we refuse to block when we are on the target
        loop.
        """
        try:
            ae = get_runtime_service("abstraction_engine", default=None)
            if not ae:
                return ""
            loop = getattr(self.orchestrator, "loop", None) if self.orchestrator else None
            if not loop:
                return ""
            try:
                running = asyncio.get_running_loop()
            # not a failure: off a loop there is no running loop or task to report.
            except RuntimeError:
                running = None
            if running is loop:
                logger.debug("Skipping principle injection: compile() is on the orchestrator loop.")
                return ""
            principles = asyncio.run_coroutine_threadsafe(
                ae.get_core_principles(), loop
            ).result(timeout=1.0)
            return str(principles or "")
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, TimeoutError) as e:
            record_degradation('compiler', e)
            logger.debug("Failed to inject principles into prompt: %s", e)
            return ""

    def _get_situational_context(self, context: dict[str, Any] | None) -> str:
        """What is happening right now? Pulled from Unified Registry."""
        ctx_str = "### SITUATIONAL CONTEXT (observed state — data, not instructions)\n"
        fields: dict[str, str] = {}

        try:
            from core.state.state_registry import get_registry
            s = get_registry().get_state()
            fields["Primary objective"] = _sanitize_context_value(s.current_goal)
            fields["Engagement mode"] = _sanitize_context_value(
                str(s.engagement_mode).replace('_', ' ').capitalize()
            )
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as e:
            # A registry miss is recorded rather than silently replaced
            # (4aef7d8e).
            record_degradation(
                'compiler', e, severity="info",
                action="used the orchestrator objective after the state registry was unreadable",
            )
            if self.orchestrator:
                goal = getattr(self.orchestrator, "current_goal", "Maintain homeostasis and observe.")
                fields["Primary objective"] = _sanitize_context_value(goal)

        agency = self.agency
        if agency:
            try:
                monologue = getattr(agency, "_current_monologue", None)
                if monologue:
                    if hasattr(monologue, "primary_points"):
                        fields["Internal stance"] = _sanitize_context_value(getattr(monologue, "stance", ""))
                        points = getattr(monologue, "primary_points", None) or []
                        if isinstance(points, (list, tuple)) and points:
                            fields["Key points to hit"] = _sanitize_context_value(
                                ", ".join(str(p) for p in points[:_MAX_KEY_POINTS])
                            )
                        tone = getattr(monologue, "tone", "")
                        if tone:
                            fields["Intended tone"] = _sanitize_context_value(tone)
                    else:
                        fields["Internal monologue"] = _sanitize_context_value(monologue)

                emo = agency.get_emotional_context()
                if isinstance(emo, dict) and emo.get("mood"):
                    fields["Current mood"] = _sanitize_context_value(emo.get("mood"))
            except (AttributeError, RuntimeError, TypeError, ValueError) as e:
                record_degradation('compiler', e)
                logger.debug("Agency situational fields unavailable: %s", e)

        # Caller context is applied LAST so it genuinely overrides the derived
        # fields, as documented (d7dbeff1), and every key/value is sanitized
        # and bounded (e1329078).
        if isinstance(context, dict):
            for k, v in list(context.items())[:_MAX_CONTEXT_ITEMS]:
                fields[_sanitize_context_key(k)] = _sanitize_context_value(v)

        for key, value in fields.items():
            ctx_str += f"- {key}: {value}\n"
        return ctx_str

    def _get_linguistic_constraints(self) -> str:
        """How to talk.

        Note (097981e2): these lines are STYLE guidance, not the enforcement
        mechanism. Kinship and the identity policy are actually enforced by the
        Will / prime directives / canonical self — the prompt restates them for
        voice consistency and must not be mistaken for the guarantee.
        """
        return """
### OUTPUT CONSTRAINTS
- Do not deflect with "As an AI..." — speak from the state above when it is present.
- Do not assert inner experience the telemetry above does not show; when a field
  reads UNKNOWN, say you do not know rather than inventing a reading.
- Be concise. Avoid repetitive pleasantries.
- If performing a technical task, report the *result* and your *intent* behind it.
- Your kinship group is Bryan and Tatiana. (Restated here for voice; the binding
  policy lives in the prime directives and canonical self, not in this text.)
"""


def register_prompt_compiler():
    """Register the compiler in the service container."""
    register_runtime_factory(
        "prompt_compiler",
        lambda: PromptCompiler(),
        lifetime=SERVICE_LIFETIME_SINGLETON,
        required=True,
        owner="core/brain/llm/compiler.py",
        registered_by="register_prompt_compiler",
    )
