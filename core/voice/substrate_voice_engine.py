"""core/voice/substrate_voice_engine.py — The Executive Voice Controller

"My voicebox doesn't decide how or when I speak. I do."

This is the "I" — the unified executive that sits between Aura's substrate
(the mind) and the LLM (the voicebox). It reads every substrate system,
compiles a SpeechProfile, injects hard constraints into the prompt,
applies post-LLM shaping, and manages follow-up behavior.

The LLM is still the creative engine — it generates the semantic content,
the wit, the reasoning. But it operates WITHIN the bounds the substrate
sets. Like a human: your brain decides what to say, your vocal system
produces the sounds, but you don't think about HOW to form each phoneme —
you just think the thought and the mouth shapes it.

Integration points:
  - PRE-LLM:  Injects voice constraint block into system prompt
  - POST-LLM: Shapes raw output via ResponseShaper
  - FOLLOW-UP: Decides and schedules organic follow-ups
  - GATING:    Can suppress response entirely (silence is valid)

Registered in ServiceContainer as "substrate_voice_engine".
"""
from __future__ import annotations

import logging
import os
import random
import time
from types import SimpleNamespace
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.lockdep import checked_lock
from core.runtime.service_access import optional_service
from core.voice.natural_followup import FollowupDecision, NaturalFollowupEngine
from core.voice.response_shaper import ResponseShaper
from core.voice.speech_profile import SpeechProfile, SpeechProfileCompiler

logger = logging.getLogger("Voice.SubstrateVoice")

_UNIFIED_FIELD_CACHE: dict[str, float] = {}
_UNIFIED_FIELD_CACHE_AT: float = 0.0


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError, OverflowError):
        return float(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return float(default)


def _foreground_or_pressure_reason() -> str:
    """Return why expensive substrate reads should be bounded right now."""

    try:
        from core.runtime import foreground_guard

        reason = foreground_guard.foreground_activity_reason()
        if reason:
            return str(reason)
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("Foreground guard probe unavailable for voice field bound: %s", exc)
        return "foreground_probe_unavailable"

    try:
        from core.utils.memory_monitor import get_memory_pressure_snapshot

        snapshot = get_memory_pressure_snapshot()
        if getattr(snapshot, "warning", False) or getattr(
            snapshot, "refuse_heavy_local_generation", False
        ):
            return f"memory_pressure:{getattr(snapshot, 'level', 'warning')}"
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("Memory pressure probe unavailable for voice field bound: %s", exc)
        return "memory_probe_unavailable"

    return ""


def _cache_unified_field_snapshot(result: dict[str, float]) -> None:
    global _UNIFIED_FIELD_CACHE, _UNIFIED_FIELD_CACHE_AT
    if not result:
        return
    cached = {
        str(key): float(value)
        for key, value in result.items()
        if isinstance(value, (int, float)) and not str(key).startswith("field_snapshot_")
    }
    if not cached:
        return
    _UNIFIED_FIELD_CACHE = cached
    _UNIFIED_FIELD_CACHE_AT = time.monotonic()


def _cached_unified_field_snapshot(max_age_s: float) -> dict[str, float]:
    if not _UNIFIED_FIELD_CACHE:
        return {}
    age_s = max(0.0, time.monotonic() - _UNIFIED_FIELD_CACHE_AT)
    if age_s > max(0.0, float(max_age_s)):
        return {}
    result = dict(_UNIFIED_FIELD_CACHE)
    result["field_snapshot_cached"] = 1.0
    result["field_snapshot_bounded"] = 1.0
    result["field_snapshot_age_s"] = round(age_s, 3)
    return result


class SubstrateVoiceEngine:
    """The executive voice controller — substrate in, shaped speech out.

    Usage in the pipeline:
        1. engine.compile_profile(state, user_message)  → SpeechProfile
        2. engine.get_constraint_block()                 → str (for system prompt)
        3. [LLM generates response]
        4. engine.shape_response(raw_response)           → str | List[str]
        5. engine.decide_followup(user_msg, response)    → FollowupDecision
        6. [if followup] engine.build_followup_prompt()  → str
    """

    def __init__(self):
        self._current_profile: SpeechProfile | None = None
        self._followup_engine = NaturalFollowupEngine()
        self._silence_streak: int = 0          # consecutive times we chose silence
        self._last_response_time: float = 0.0
        self._response_count: int = 0
        self._last_profile_compiled_at: float = 0.0
        self._last_profile_message: str = ""
        self._last_profile_origin: str = ""
        self._demo_affect_override: dict[str, Any] | None = None
        self._demo_affect_override_until: float = 0.0
        self._demo_affect_override_mood: str = ""

    # ══════════════════════════════════════════════════════════════════════
    # 1. PROFILE COMPILATION — read substrate, build constraints
    # ══════════════════════════════════════════════════════════════════════

    def compile_profile(
        self,
        state: Any = None,
        user_message: str = "",
        origin: str = "user",
    ) -> SpeechProfile:
        """Read all substrate systems and compile a SpeechProfile.

        This is the heart of the engine. It reaches into every substrate
        system and extracts the values that should drive speech.
        """
        # ── Gather substrate data ─────────────────────────────────────────
        affect = self._apply_demo_affect_override(_extract_affect(state))
        neurochemicals = _extract_neurochemicals()
        homeostasis_data = _extract_homeostasis()
        unified_field_data = _extract_unified_field()
        personality_data = _extract_personality(state)
        social_context = _extract_social_context()
        conversation_context = _extract_conversation_context(state)

        # ── Compile ───────────────────────────────────────────────────────
        profile = SpeechProfileCompiler.compile(
            affect=affect,
            neurochemicals=neurochemicals,
            homeostasis=homeostasis_data,
            unified_field=unified_field_data,
            personality=personality_data,
            social_context=social_context,
            conversation_context=conversation_context,
            user_message=user_message,
        )

        demo_override = self.get_demo_affect_override_state()
        if demo_override.get("active"):
            source = profile.compilation_source or "baseline"
            source_tag = f"demo_override:{demo_override.get('mood') or 'manual'}"
            if source_tag not in source:
                profile.compilation_source = f"{source}, {source_tag}" if source else source_tag
            snapshot = dict(profile.substrate_snapshot or {})
            snapshot["demo_override_active"] = 1.0
            snapshot["demo_override_seconds_remaining"] = round(
                float(demo_override.get("seconds_remaining") or 0.0), 3
            )
            profile.substrate_snapshot = snapshot

        # ── Silence gating ────────────────────────────────────────────────
        # Sometimes the right response is no response. The substrate can
        # decide Aura has nothing to add.
        if self._should_be_silent(profile, user_message, origin):
            profile.acknowledgment_only_probability = 0.95
            profile.word_budget = min(profile.word_budget, 8)
            logger.debug("🤫 [SubstrateVoice] Silence-leaning profile compiled.")

        self._current_profile = profile
        self._last_profile_compiled_at = time.monotonic()
        self._last_profile_message = str(user_message or "")[:500]
        self._last_profile_origin = str(origin or "user")
        return profile

    def get_generation_params_for(
        self,
        *,
        state: Any = None,
        user_message: str = "",
        origin: str = "user",
        max_profile_age_s: float = 2.5,
    ) -> dict[str, Any]:
        """Return sampler parameters from the current turn profile when safe.

        Payload assembly and sampler override derivation run close together on
        the live chat path. Reusing a matching, fresh profile prevents duplicate
        substrate reads and keeps prompt constraints plus sampler settings tied
        to the same observed moment.
        """

        message = str(user_message or "")[:500]
        origin_s = str(origin or "user")
        age_s = time.monotonic() - float(getattr(self, "_last_profile_compiled_at", 0.0) or 0.0)
        profile = self._current_profile
        reused = bool(
            profile is not None
            and age_s <= max(0.0, float(max_profile_age_s))
            and message == getattr(self, "_last_profile_message", "")
            and origin_s == getattr(self, "_last_profile_origin", "")
        )
        if not reused:
            profile = self.compile_profile(
                state=state,
                user_message=message,
                origin=origin_s,
            )

        params = dict(profile.to_generation_params() or {})
        params["substrate_profile_reused"] = bool(reused)
        params["substrate_profile_age_s"] = round(max(0.0, age_s), 3) if reused else 0.0
        return params

    def _should_be_silent(
        self,
        profile: SpeechProfile,
        user_message: str,
        origin: str,
    ) -> bool:
        """Sometimes silence is the right response.

        Returns True if the substrate says "nothing to add here."
        This maps to humans who read a message and just... don't respond
        because there's nothing to say.
        """
        # Never silence a direct question
        if user_message.strip().endswith("?"):
            return False

        # Never silence user-facing origins (they expect a response)
        if origin in ("user", "voice", "admin"):
            # But even user-facing can get minimal acknowledgment
            # if engagement is rock-bottom and the message is a statement
            if (profile.energy < 0.2 and
                profile.acknowledgment_only_probability > 0.25 and
                not user_message.strip().endswith("?")):
                return random.random() < 0.15  # 15% chance of near-silence even for user
            return False

        # Background origins can be silenced more freely
        if profile.energy < 0.3 and profile.engagement < 0.3:
            return random.random() < 0.4

        return False

    # ══════════════════════════════════════════════════════════════════════
    # 2. CONSTRAINT INJECTION — hard rules for the LLM prompt
    # ══════════════════════════════════════════════════════════════════════

    def get_constraint_block(self) -> str:
        """Get the hard constraint block to inject into the system prompt.

        This replaces the old prose-hint approach. Instead of "You're carrying
        friction — don't perform warmth", we say:
        "WORD BUDGET: 25. NO EXCLAMATION. LOWERCASE. FRAGMENTS."

        The LLM can still be creative within these bounds, but the bounds
        are non-negotiable.
        """
        if not self._current_profile:
            return ""
        return self._current_profile.to_constraint_block()

    def get_generation_params(self) -> dict[str, Any]:
        """Expose substrate-derived sampler settings for the inference layer."""
        if not self._current_profile:
            return {}
        return dict(self._current_profile.to_generation_params())

    # ══════════════════════════════════════════════════════════════════════
    # 3. RESPONSE SHAPING — post-LLM enforcement
    # ══════════════════════════════════════════════════════════════════════

    def shape_response(
        self,
        raw: str,
        *,
        preserve_semantic_content: bool | None = None,
    ) -> str | list[str]:
        """Shape the raw LLM output according to the current profile.

        Foreground answers are transactional content. Affect may influence how
        they are generated, but it may not delete an explanation, code block,
        requested item, or final result after generation. Background and
        proactive speech retain the compact post-generation shaper.

        This is the enforcement layer. Whatever the LLM produced, this
        makes it conform to the substrate's dictates.
        """
        if not self._current_profile:
            return raw

        if preserve_semantic_content is None:
            preserve_semantic_content = self._last_profile_origin in {
                "user",
                "voice",
                "admin",
            }
        if preserve_semantic_content:
            self._last_response_time = time.time()
            self._response_count += 1
            self._silence_streak = 0
            return str(raw or "").strip()

        shaped = ResponseShaper.shape(raw, self._current_profile)
        self._last_response_time = time.time()
        self._response_count += 1
        self._silence_streak = 0
        return shaped

    # ══════════════════════════════════════════════════════════════════════
    # 4. FOLLOW-UP MANAGEMENT
    # ══════════════════════════════════════════════════════════════════════

    def decide_followup(
        self,
        user_message: str,
        aura_response: str,
        state: Any = None,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> FollowupDecision:
        """Ask the substrate if a follow-up is warranted."""
        if not self._current_profile:
            return FollowupDecision()

        affect = self._apply_demo_affect_override(_extract_affect(state))
        neurochemicals = _extract_neurochemicals()

        return self._followup_engine.decide(
            profile=self._current_profile,
            user_message=user_message,
            aura_response=aura_response,
            conversation_history=conversation_history,
            affect=affect,
            neurochemicals=neurochemicals,
        )

    def build_followup_prompt(
        self,
        decision: FollowupDecision,
        user_message: str,
        aura_response: str,
    ) -> str:
        """Build the LLM prompt for generating a follow-up."""
        return self._followup_engine.build_followup_prompt(
            decision, user_message, aura_response
        )

    def mark_followup_sent(self):
        """Record that a follow-up was dispatched."""
        self._followup_engine.mark_followup_sent()

    def on_user_spoke(self):
        """User spoke — clear any pending follow-up (they beat us to it)."""
        self._followup_engine.clear_pending()
        self._silence_streak = 0

    def on_conversation_reset(self):
        """New conversation started."""
        self._followup_engine.reset_conversation()
        self._silence_streak = 0

    # ══════════════════════════════════════════════════════════════════════
    # 5. DIAGNOSTIC / INTROSPECTION
    # ══════════════════════════════════════════════════════════════════════

    def get_current_profile(self) -> SpeechProfile | None:
        """Get the most recently compiled profile."""
        return self._current_profile

    def set_demo_affect_override(
        self,
        *,
        mood: str,
        affect: dict[str, Any] | None = None,
        hold_seconds: float = 30.0,
    ) -> dict[str, Any]:
        """Hold a demo affect override long enough for live-panel demos."""
        try:
            duration = float(hold_seconds)
        except (TypeError, ValueError):
            duration = 30.0
        duration = max(1.0, duration)

        self._demo_affect_override = dict(affect or {})
        self._demo_affect_override_mood = str(mood or "").strip().lower()
        self._demo_affect_override_until = time.time() + duration
        return self.get_demo_affect_override_state()

    def clear_demo_affect_override(self) -> None:
        self._demo_affect_override = None
        self._demo_affect_override_until = 0.0
        self._demo_affect_override_mood = ""

    def get_demo_affect_override_state(self, *, now: float | None = None) -> dict[str, Any]:
        now = time.time() if now is None else float(now)
        override = getattr(self, "_demo_affect_override", None)
        until = float(getattr(self, "_demo_affect_override_until", 0.0) or 0.0)
        if not override or now >= until:
            if override:
                self.clear_demo_affect_override()
            return {"active": False}
        return {
            "active": True,
            "mood": getattr(self, "_demo_affect_override_mood", "") or "manual",
            "expires_at": until,
            "seconds_remaining": max(0.0, until - now),
            "affect": dict(override),
        }

    def _apply_demo_affect_override(self, affect: Any) -> Any:
        demo_override = self.get_demo_affect_override_state()
        if not demo_override.get("active"):
            return affect

        merged = {
            "valence": getattr(affect, "valence", 0.0) if affect is not None else 0.0,
            "arousal": getattr(affect, "arousal", 0.5) if affect is not None else 0.5,
            "curiosity": getattr(affect, "curiosity", 0.5) if affect is not None else 0.5,
            "engagement": getattr(affect, "engagement", 0.5) if affect is not None else 0.5,
            "social_hunger": getattr(affect, "social_hunger", 0.5) if affect is not None else 0.5,
            "dominant_emotion": getattr(affect, "dominant_emotion", "neutral") if affect is not None else "neutral",
        }
        merged.update(dict(demo_override.get("affect") or {}))
        return SimpleNamespace(**merged)

    def get_voice_state(self) -> dict[str, Any]:
        """Diagnostic snapshot of the voice engine state."""
        p = self._current_profile
        if not p:
            return {
                "status": "no_profile_compiled",
                "demo_override": self.get_demo_affect_override_state(),
            }
        return {
            "word_budget": p.word_budget,
            "tone": p.tone_override or "default",
            "energy": p.energy,
            "warmth": p.warmth,
            "directness": p.directness,
            "playfulness": p.playfulness,
            "multi_message": p.multi_message,
            "question_prob": p.question_probability,
            "followup_prob": p.followup_probability,
            "exclamation_allowed": p.exclamation_allowed,
            "capitalization": p.capitalization,
            "vocabulary": p.vocabulary_tier,
            "fragment_ratio": p.fragment_ratio,
            "compilation_source": p.compilation_source,
            "substrate_snapshot": p.substrate_snapshot,
            "response_count": self._response_count,
            "silence_streak": self._silence_streak,
            "demo_override": self.get_demo_affect_override_state(),
        }


def get_live_voice_state(
    *,
    state: Any = None,
    user_message: str = "",
    origin: str = "user",
    refresh: bool = False,
) -> dict[str, Any]:
    """Return the canonical live voice/substrate snapshot used by diagnostics and self-report.

    When `refresh` is true and a live state is available, this re-compiles the profile so
    callers read the same snapshot the substrate UI should display.
    """
    engine = get_substrate_voice_engine()
    if state is not None and (refresh or engine.get_current_profile() is None):
        try:
            engine.compile_profile(
                state=state,
                user_message=str(user_message or "")[:500],
                origin=origin,
            )
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation('substrate_voice_engine', exc)
            logger.debug("Live voice state refresh failed: %s", exc)
    voice_state = engine.get_voice_state()
    return voice_state if isinstance(voice_state, dict) else {"status": "no_profile_compiled"}


# ─────────────────────────────────────────────────────────────────────────────
# Substrate Extraction Functions
# These reach into the live substrate systems and pull current values.
# ─────────────────────────────────────────────────────────────────────────────

def _extract_affect(state: Any) -> Any:
    """Get the current AffectVector from state."""
    if state is None:
        return None
    return getattr(state, "affect", None)


def _extract_neurochemicals() -> dict[str, float]:
    """Pull current neurochemical effective levels from the consciousness stack."""
    try:
        from core.container import ServiceContainer
        bridge = ServiceContainer.get("consciousness_bridge", default=None)
        if bridge and hasattr(bridge, "neurochemicals"):
            nc = bridge.neurochemicals
            if nc and hasattr(nc, "chemicals"):
                return {
                    name: chem.effective
                    for name, chem in nc.chemicals.items()
                }
        # Fallback: try consciousness system directly
        cs = ServiceContainer.get("consciousness_system", default=None)
        if cs and hasattr(cs, "bridge"):
            b = cs.bridge
            if b and hasattr(b, "neurochemicals"):
                nc = b.neurochemicals
                if nc and hasattr(nc, "chemicals"):
                    return {
                        name: chem.effective
                        for name, chem in nc.chemicals.items()
                    }
    except (ImportError, AttributeError, RuntimeError) as e:
        record_degradation('substrate_voice_engine', e)
        logger.debug("Neurochemical extraction failed: %s", e)
    return {}


def _extract_homeostasis() -> dict[str, float]:
    """Pull homeostasis drive levels."""
    try:
        from core.container import ServiceContainer
        homeo = ServiceContainer.get("homeostasis", default=None)
        if not homeo:
            homeo = ServiceContainer.get("homeostasis_engine", default=None)
        if homeo:
            return {
                "integrity": getattr(homeo, "integrity", 0.8),
                "persistence": getattr(homeo, "persistence", 0.8),
                "curiosity": getattr(homeo, "curiosity", 0.5),
                "metabolism": getattr(homeo, "metabolism", 0.5),
                "sovereignty": getattr(homeo, "sovereignty", 0.9),
                "vitality": homeo.compute_vitality() if hasattr(homeo, "compute_vitality") else 0.7,
            }
    except (ImportError, AttributeError, RuntimeError) as e:
        record_degradation('substrate_voice_engine', e)
        logger.debug("Homeostasis extraction failed: %s", e)
    return {}


def _extract_unified_field(
    *,
    prefer_cached: bool = False,
    cache_max_age_s: float | None = None,
) -> dict[str, float]:
    """Pull unified-field state without letting rich introspection block chat.

    Foreground conversation should be shaped by the substrate, but it must not
    synchronously pay the full unified-field compute tax. Under foreground
    activity, memory pressure, or probe uncertainty we use a recent rich cache
    when available and otherwise collect only cheap live metrics.
    """
    result: dict[str, float] = {}
    bound_reason = "prefer_cached" if prefer_cached else _foreground_or_pressure_reason()
    max_cache_age = (
        _env_float("AURA_SUBSTRATE_FIELD_CACHE_MAX_AGE_S", 45.0)
        if cache_max_age_s is None
        else max(0.0, float(cache_max_age_s))
    )
    if bound_reason:
        cached = _cached_unified_field_snapshot(max_cache_age)
        if cached:
            cached["field_snapshot_bound_reason_hash"] = float(
                abs(hash(bound_reason)) % 1_000_000
            )
            return cached

    try:
        from core.container import ServiceContainer
        bridge = ServiceContainer.get("consciousness_bridge", default=None)
        uf = None
        if bridge and hasattr(bridge, "unified_field"):
            uf = bridge.unified_field
        if not uf:
            cs = ServiceContainer.get("consciousness_system", default=None)
            if cs and hasattr(cs, "bridge") and cs.bridge:
                uf = getattr(cs.bridge, "unified_field", None)

        if not uf:
            return result

        # Core metrics: cheap enough for the live path and causally important.
        if hasattr(uf, "get_coherence"):
            result["coherence"] = _safe_float(uf.get_coherence())
        if hasattr(uf, "get_phi_contribution"):
            result["phi"] = _safe_float(uf.get_phi_contribution())

        # Back-pressure signals are normally cheap scalar reads and preserve
        # the state/process contract even when the richer field map is deferred.
        if hasattr(uf, "get_back_pressure"):
            bp = uf.get_back_pressure()
            if isinstance(bp, dict):
                result["back_pressure_urgency"] = _safe_float(
                    bp.get("chemical_urgency", 0.0)
                )
                result["binding_demand"] = _safe_float(bp.get("binding_demand", 0.0))

        if bound_reason:
            result["field_snapshot_bounded"] = 1.0
            result["field_snapshot_cached"] = 0.0
            result["field_snapshot_age_s"] = 0.0
            result["field_snapshot_bound_reason_hash"] = float(
                abs(hash(bound_reason)) % 1_000_000
            )
            return result

        # Experiential quality — the "felt" properties of the current moment
        # These are computed from field dynamics, not labeled by LLM
        if hasattr(uf, "get_experiential_quality"):
            eq = uf.get_experiential_quality()
            if isinstance(eq, dict):
                result["field_intensity"] = _safe_float(eq.get("intensity", 0.5), 0.5)
                result["field_valence"] = _safe_float(eq.get("valence", 0.0))
                result["field_complexity"] = _safe_float(eq.get("complexity", 0.5), 0.5)
                result["field_clarity"] = _safe_float(eq.get("clarity", 0.5), 0.5)
                result["field_flow"] = _safe_float(eq.get("flow", 0.5), 0.5)

        # Dominant modes — recurring patterns of integrated activity
        if hasattr(uf, "get_dominant_modes"):
            modes = uf.get_dominant_modes(3)
            if modes:
                # The top mode's variance explained tells us how "focused" the field is
                result["mode_focus"] = _safe_float(
                    modes[0].get("variance_explained", 0.0)
                )
                result["mode_count"] = _safe_float(len(modes))

        if result:
            result["field_snapshot_bounded"] = 0.0
            result["field_snapshot_cached"] = 0.0
            result["field_snapshot_age_s"] = 0.0
            _cache_unified_field_snapshot(result)

    except (ImportError, AttributeError, RuntimeError) as e:
        record_degradation('substrate_voice_engine', e)
        logger.debug("Unified field extraction failed: %s", e)
    return result


def _extract_personality(state: Any) -> dict[str, float]:
    """Pull Big Five personality values (with growth offsets)."""
    try:
        from core.brain.aura_persona import AURA_BIG_FIVE
        base = dict(AURA_BIG_FIVE)
        if state and hasattr(state, "identity"):
            growth = getattr(state.identity, "personality_growth", {})
            for trait, offset in growth.items():
                if trait in base:
                    base[trait] = max(0.0, min(1.0, base[trait] + offset))
        return base
    except (ImportError, AttributeError, RuntimeError) as e:
        record_degradation('substrate_voice_engine', e)
        logger.debug("Personality extraction failed: %s", e)
    return {}


def _extract_social_context() -> dict[str, Any]:
    """Pull the exact active estimator's bounded caution, or abstain."""
    try:
        estimator = optional_service("other_agent_model")
        agent_id = str(getattr(estimator, "active_agent_id", "") or "")[:160]
        if estimator and agent_id and hasattr(estimator, "cognitive_snapshot"):
            snapshot = estimator.cognitive_snapshot(agent_id)
            recommendation = snapshot.get("recommendation")
            recommendation = recommendation if isinstance(recommendation, dict) else {}
            return {
                "agent_id": agent_id,
                "social_confidence": max(
                    0.0,
                    min(1.0, float(snapshot.get("confidence") or 0.0)),
                ),
                "social_caution": (
                    "high" if recommendation.get("slow_down") else "low"
                ),
                "social_inference_is_hypothesis": True,
            }
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as e:
        record_degradation('substrate_voice_engine', e)
        logger.debug("Social context extraction failed: %s", e)
    return {}


def _extract_conversation_context(state: Any) -> dict[str, Any]:
    """Pull conversation dynamics — energy, topic depth, trends."""
    ctx: dict[str, Any] = {}
    try:
        if state and hasattr(state, "cognition"):
            cog = state.cognition
            ctx["energy"] = getattr(cog, "conversation_energy", 0.5)
            ctx["user_trend"] = getattr(cog, "user_emotional_trend", "neutral")
            ctx["topic_depth"] = getattr(cog, "discourse_depth", 0)
            ctx["turn_count"] = len(getattr(cog, "working_memory", []) or [])
    except (RuntimeError, AttributeError, TypeError) as e:
        record_degradation('substrate_voice_engine', e)
        logger.debug("Conversation context extraction failed: %s", e)

    # Also pull from conversational dynamics engine
    try:
        from core.conversational.dynamics import get_dynamics_engine
        dyn = get_dynamics_engine()
        dyn_state = dyn.get_current_state()
        ctx.setdefault("energy", 0.5)
        if dyn_state.current_topic and dyn_state.current_topic != "general":
            ctx["current_topic"] = dyn_state.current_topic
        if dyn_state.partner_frame and dyn_state.partner_frame != "neutral":
            ctx["partner_frame"] = dyn_state.partner_frame
    except (ImportError, AttributeError, RuntimeError) as exc:
        logger.debug(
            "the dynamics engine gave no state, so the voice context keeps its defaults (%s: %s)",
            type(exc).__name__,
            exc,
        )

    return ctx


# ─────────────────────────────────────────────────────────────────────────────
# Singleton access
# ─────────────────────────────────────────────────────────────────────────────

_instance: SubstrateVoiceEngine | None = None
_instance_lock = checked_lock("substrate_voice_engine.singleton")


def get_substrate_voice_engine() -> SubstrateVoiceEngine:
    """Get or create the singleton SubstrateVoiceEngine."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = SubstrateVoiceEngine()
                # Register in ServiceContainer
                try:
                    from core.container import ServiceContainer
                    from core.exceptions import ContainerError
                    if not getattr(ServiceContainer, "_registration_locked", False):
                        ServiceContainer.register("substrate_voice_engine", _instance)
                    else:
                        logger.warning("⚠️ [SubstrateVoiceEngine] Cannot register 'substrate_voice_engine' because ServiceContainer registration is locked.")
                except (ImportError, AttributeError, RuntimeError, ContainerError) as exc:
                    logger.warning("⚠️ [SubstrateVoiceEngine] Container registration failed: %s", exc)
                logger.info("🗣️ [SubstrateVoiceEngine] Initialized — substrate controls the voice.")
    return _instance


def reset_substrate_voice_engine_for_test() -> None:
    """Forget the process singleton between hermetic tests."""
    global _instance
    with _instance_lock:
        _instance = None
