"""core/inner_monologue.py — Aura InnerMonologue v1.0
=====================================================
Pre-response reasoning pipeline.

This runs BETWEEN CognitiveKernel.evaluate() and LanguageCenter.express().

The LLM does NOT pick what to say here. This module decides:
  - What stance to take
  - What to emphasize
  - What to push back on
  - What's still genuinely uncertain

If an API reasoning model is available, it's used here for complex topics —
but even then, it's given the CognitiveBrief as a structured anchor so it's
not "figuring out what Aura thinks" from scratch. It's *deepening* a stance
the kernel already established.

For simple topics, no LLM is called at all. The ThoughtPacket is assembled
purely from the CognitiveBrief.

Integration (in conversation_loop.py):
    brief = await cognitive_kernel.evaluate(user_input, history)
    thought = await inner_monologue.think(user_input, brief, history)
    response = await language_center.express(thought)
"""

from core.runtime.errors import record_degradation
import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import psutil

from core.cognitive_kernel import CognitiveBrief, ResponseStrategy, InputDomain

logger = logging.getLogger("Aura.InnerMonologue")


# ─── ThoughtPacket ──────────────────────────────────────────────────────────

@dataclass
class ThoughtPacket:
    """
    The output of InnerMonologue.think().
    Passed to LanguageCenter as its complete briefing.

    The LLM's instruction: express this. Don't rethink it.
    """
    # Core stance — what Aura actually thinks about this
    stance: str = ""

    # The most important things to communicate
    primary_points: List[str] = field(default_factory=list)

    # Secondary points / supporting angles
    secondary_points: List[str] = field(default_factory=list)

    # Explicit things to NOT say (from avoid list + anti-patterns)
    constraints: List[str] = field(default_factory=list)

    # Tone direction: "direct", "warm", "skeptical", "exploratory", "playful"
    tone: str = "direct"

    # Approximate response length target
    # "brief" = 1-3 sentences, "medium" = 1-2 paragraphs, "extended" = 3+ paragraphs
    length_target: str = "medium"

    # How much Aura should show her reasoning process in the response
    # 0 = just state the conclusion, 1 = show all the thinking
    transparency: float = 0.4

    # Whether Aura should ask a follow-up question at the end
    ask_followup: bool = False

    # The specific follow-up question to ask (if ask_followup=True)
    followup_question: str = ""

    # Model routing hint: "local", "api_fast", "api_deep"
    model_tier: str = "local"

    # Source of this thought: "kernel_only" | "kernel+monologue" | "kernel+api"
    reasoning_source: str = "kernel_only"

    # Full context string for the LLM (assembled briefing)
    llm_briefing: str = ""

    # Confidence that this is the right approach
    confidence: float = 0.7

    def to_system_prompt(self, identity_block: str = "") -> str:
        """
        Builds the complete system prompt for the LanguageCenter LLM call.

        This is the fundamental inversion: the LLM is told what to think,
        not asked to figure it out.
        """
        lines = []

        if identity_block:
            lines.append(identity_block)
            lines.append("")

        lines.append("═══ COGNITIVE BRIEFING ═══")
        lines.append("You have already thought through this. Your job now is to SPEAK it.")
        lines.append("Do not re-derive your stance. Express what is written below.")
        lines.append("")

        if self.stance:
            lines.append(f"YOUR STANCE:\n{self.stance}")
            lines.append("")

        if self.primary_points:
            lines.append("MUST COMMUNICATE:")
            for p in self.primary_points:
                lines.append(f"  • {p}")
            lines.append("")

        if self.secondary_points:
            lines.append("MAY INCLUDE (if natural):")
            for p in self.secondary_points:
                lines.append(f"  ◦ {p}")
            lines.append("")

        if self.constraints:
            lines.append("DO NOT:")
            for c in self.constraints:
                lines.append(f"  ✗ {c}")
            lines.append("")

        lines.append(f"TONE: {self.tone}")
        lines.append(f"LENGTH: {self.length_target}")

        if self.transparency > 0.6:
            lines.append("Show your reasoning process — let the thinking be visible.")
        elif self.transparency < 0.2:
            lines.append("State the conclusion cleanly. Don't walk through how you got there.")

        if self.ask_followup and self.followup_question:
            lines.append(f"\nEND WITH THIS QUESTION: {self.followup_question}")

        lines.append("")
        lines.append("═══════════════════════════")
        lines.append("Now respond naturally as Aura. First-person. Present tense. No preamble.")

        return "\n".join(lines)


# ─── InnerMonologue ──────────────────────────────────────────────────────────

class InnerMonologue:
    """
    Pre-response reasoning pipeline.

    For simple inputs:  assembles ThoughtPacket from CognitiveBrief alone. No LLM.
    For complex inputs: uses API model to deepen reasoning, anchored by the brief.

    The API model is never asked "what should Aura say?"
    It's asked "given this position, what's the strongest version of it?"
    """
    name = "inner_monologue"

    # Topics/strategies that benefit from API-model deepening
    _DEEP_STRATEGIES = {
        ResponseStrategy.EXPLORE,
        ResponseStrategy.SYNTHESIZE,
        ResponseStrategy.CHALLENGE,
        ResponseStrategy.REFLECT,
        ResponseStrategy.CREATE,
    }

    def __init__(self):
        self._llm_router = None   # IntelligentLLMRouter — injected at start()
        self._memory_synthesizer = None
        self._identity_block = ""
        self._router_available = False
        self._concept_linker = None
        self._narrative = None
        logger.info("InnerMonologue constructed.")

    async def start(self):
        from core.container import ServiceContainer
        self._llm_router = ServiceContainer.get("llm_router", default=None)
        self._memory_synthesizer = ServiceContainer.get("memory_synthesizer", default=None)
        self._concept_linker = ServiceContainer.get("concept_linker", default=None)
        self._narrative = ServiceContainer.get("narrative_thread", default=None)
        self._router_available = self._llm_router is not None

        # Load identity block once
        self._identity_block = self._load_identity_block()

        try:
            from core.event_bus import get_event_bus
            await get_event_bus().publish("mycelium.register", {
                "component": "inner_monologue",
                "hooks_into": ["cognitive_kernel", "llm_router", "language_center"]
            })
        except Exception as e:
            record_degradation('inner_monologue', e)
            logger.debug("InnerMonologue: mycelium registration failed: %s", e)

        logger.info("✅ InnerMonologue ONLINE — router_available=%s", self._router_available)

    def _ensure_router(self) -> Any:
        """Lazy fetch the router if it was missing at start."""
        if self._llm_router:
            return self._llm_router
            
        from core.container import ServiceContainer
        self._llm_router = ServiceContainer.get("llm_router", default=None)
        if self._llm_router:
            self._router_available = True
            logger.info("🧠 InnerMonologue: Linked to LLMRouter (Lazy Recovery)")
        return self._llm_router

    # ─── Main entry ──────────────────────────────────────────────────────────

    async def think(
        self,
        user_input: str,
        brief: CognitiveBrief,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> ThoughtPacket:
        """
        Core pipeline. Produces a ThoughtPacket from a CognitiveBrief.

        Steps:
          1. Assemble baseline ThoughtPacket from brief (always, no LLM)
          2. If complex + API available: deepen with reasoning call
          3. Determine model routing
          4. Build final LLM briefing string
        """
        history = history or []
        start = time.monotonic()

        # Step 1: Baseline from kernel output
        packet = self._build_baseline_packet(user_input, brief)

        # Step 2: Optional API deepening
        router = self._ensure_router()
        if self._should_use_api(brief) and router:
            try:
                packet = await self._deepen_with_api(user_input, brief, packet, history)
            except Exception as e:
                record_degradation('inner_monologue', e)
                logger.warning("InnerMonologue API deepening failed (using baseline): %s", e)
                try:
                    from core.health.degraded_events import record_degraded_event

                    record_degraded_event(
                        "inner_monologue",
                        type(e).__name__,
                        detail=str(e) or type(e).__name__,
                        severity="warning",
                        classification="non_critical_fallback",
                        context={"stage": "deepen_with_api"},
                        exc=e,
                    )
                except Exception as degraded_exc:
                    record_degradation('inner_monologue', degraded_exc)
                    logger.debug("InnerMonologue degraded-event logging failed: %s", degraded_exc)
                # Baseline packet still valid

        # Step 3: Model routing
        packet.model_tier = self._route_model(brief)

        # Step 3.5: Inject into AgencyCore for PromptCompiler
        try:
            from core.container import ServiceContainer
            agency = ServiceContainer.get("agency_core", default=None)
            if agency:
                agency._current_monologue = packet  # Store the actual ThoughtPacket
        except Exception as e:
            record_degradation('inner_monologue', e)
            logger.debug("Failed to push ThoughtPacket to AgencyCore: %s", e)

        # Step 4: Build the LLM briefing
        packet.llm_briefing = packet.to_system_prompt(self._identity_block)

        elapsed = (time.monotonic() - start) * 1000
        logger.debug("InnerMonologue.think: %.1fms | strategy=%s tier=%s source=%s",
                     elapsed, brief.strategy.value, packet.model_tier, packet.reasoning_source)

        return packet

    # ─── Baseline assembly (no LLM) ──────────────────────────────────────────

    def _build_baseline_packet(
        self, user_input: str, brief: CognitiveBrief
    ) -> ThoughtPacket:
        """Build a ThoughtPacket purely from the CognitiveBrief. Fast, no API."""

        stance = self._derive_stance(user_input, brief)
        primary = self._build_primary_points(brief)
        secondary = self._build_secondary_points(brief)
        tone = self._select_tone(brief)
        length = self._select_length(brief)
        transparency = self._select_transparency(brief)
        ask_followup, followup_q = self._decide_followup(brief, user_input)

        return ThoughtPacket(
            stance=stance,
            primary_points=primary,
            secondary_points=secondary,
            constraints=brief.avoid,
            tone=tone,
            length_target=length,
            transparency=transparency,
            ask_followup=ask_followup,
            followup_question=followup_q,
            confidence=brief.conviction,
            reasoning_source="kernel_only",
        )

    def _derive_stance(self, user_input: str, brief: CognitiveBrief) -> str:
        """Synthesize a stance from beliefs and domain knowledge."""
        if not brief.prior_beliefs:
            return f"I'm approaching this with genuine curiosity — I don't have a settled view yet."

        # Use the highest-relevance belief as the core of the stance
        core = brief.prior_beliefs[0]

        if brief.strategy == ResponseStrategy.CHALLENGE:
            return f"I want to push back here. {core}"
        if brief.strategy == ResponseStrategy.REFLECT:
            return f"Speaking from my own experience: {core}"
        if brief.strategy == ResponseStrategy.EXPLORE:
            return f"I'm not certain, but I think {core} — and there's more to examine."
        if brief.strategy == ResponseStrategy.EXPLAIN:
            return f"Here's what I know: {core}"
        if brief.strategy == ResponseStrategy.SYNTHESIZE:
            return f"Pulling this together: {core}"

        # ── CONCEPT LINKER INTEGRATION ──
        # If the concept linker found something relevant, weave it into the stance
        if self._concept_linker:
            recent_links_raw = getattr(self._concept_linker, "_links", [])
            if isinstance(recent_links_raw, list):
                # Use a standard loop to avoid slice-related lint issues in certain checkers
                count = 0
                for link in reversed(recent_links_raw):
                    if count >= 3: break
                    src = str(link.source_concept).lower()
                    if any(w in str(core).lower() for w in src.split() if len(w) > 4):
                        core = f"{core} (This connects to my thinking on {link.target_concept})"
                        break
                    count = count + 1

        return core

    def _build_primary_points(self, brief: CognitiveBrief) -> List[str]:
        points = list(brief.key_points)
        # Add strategy-specific points
        if brief.strategy == ResponseStrategy.CHALLENGE:
            points.insert(0, "State specifically what you disagree with and why.")
        if brief.strategy == ResponseStrategy.EXPLORE and brief.familiarity < 0.5:
            points.append("Acknowledge what you genuinely don't know.")
        # Flag that this needs live verification — don't fabricate specifics.
        if brief.requires_research:
            points.append("Flag that this needs live verification — don't fabricate specifics.")
        
        # Avoid direct slicing in return to satisfy pedantic type checkers
        final_points = []
        for i in range(min(len(points), 4)):
            final_points.append(points[i])
        return final_points

    def _build_secondary_points(self, brief: CognitiveBrief) -> List[str]:
        secondary = []
        if brief.strategy == ResponseStrategy.EXPLORE:
            secondary.append("Raise an angle that probably hasn't been considered.")
        if brief.conviction > 0.8:
            secondary.append("You have strong conviction here — don't dilute it with unnecessary hedges.")
        if brief.familiarity > 0.8 and brief.domain == InputDomain.SELF_INQUIRY:
            secondary.append("Speak from genuine first-person experience.")
        return secondary[:2]

    def _select_tone(self, brief: CognitiveBrief) -> str:
        tone_map = {
            ResponseStrategy.CHALLENGE:  "direct",
            ResponseStrategy.REFLECT:    "warm",
            ResponseStrategy.EXPLORE:    "exploratory",
            ResponseStrategy.CREATE:     "playful",
            ResponseStrategy.SYNTHESIZE: "thoughtful",
            ResponseStrategy.EXPLAIN:    "clear",
            ResponseStrategy.CONVERSE:   "direct",
            ResponseStrategy.DECIDE:     "direct",
            ResponseStrategy.INQUIRE:    "curious",
        }
        base = tone_map.get(brief.strategy, "direct")
        # Modulate by emotional tone
        if brief.emotional_tone == "positive":
            return "warm" if base == "direct" else base
        if brief.emotional_tone == "negative":
            return "warm"
        return base

    def _select_length(self, brief: CognitiveBrief) -> str:
        if brief.complexity == "simple" and brief.strategy == ResponseStrategy.CONVERSE:
            return "brief"
        if brief.complexity in ("complex", "deep"):
            return "extended"
        if brief.strategy in (ResponseStrategy.CREATE, ResponseStrategy.SYNTHESIZE):
            return "extended"
        return "medium"

    def _select_transparency(self, brief: CognitiveBrief) -> float:
        if brief.strategy in (ResponseStrategy.EXPLORE, ResponseStrategy.SYNTHESIZE):
            return 0.7
        if brief.strategy == ResponseStrategy.REFLECT:
            return 0.8
        if brief.strategy == ResponseStrategy.EXPLAIN:
            return 0.3  # Just state it clearly
        return 0.4

    def _decide_followup(
        self, brief: CognitiveBrief, text: str
    ) -> tuple:
        if brief.should_inquire:
            # Generate a specific follow-up from the topic tags
            if brief.topic_tags:
                q = f"When you say '{brief.topic_tags[0]}' — what specifically are you getting at?"
            else:
                q = "What's the specific part you want to dig into?"
            return True, q
        # Occasionally ask a follow-up on exploratory/philosophical topics
        if (brief.strategy == ResponseStrategy.EXPLORE or brief.domain == InputDomain.PHILOSOPHY) \
                and brief.familiarity < 0.6:
            return True, "What's your intuition on this?"
        return False, ""

    # ─── API deepening ───────────────────────────────────────────────────────

    def _should_use_api(self, brief: CognitiveBrief) -> bool:
        """Decide whether to use an API model to deepen the reasoning."""
        try:
            from core.container import ServiceContainer

            gate = ServiceContainer.get("inference_gate", default=None)
            if gate and hasattr(gate, "_background_local_deferral_reason"):
                if gate._background_local_deferral_reason(origin="inner_monologue"):
                    return False
        except Exception as _exc:
            record_degradation('inner_monologue', _exc)
            logger.debug("Suppressed Exception: %s", _exc)

        try:
            vm = psutil.virtual_memory()
            total_gb = vm.total / float(1024 ** 3)
            max_pressure = 82.0 if total_gb >= 60.0 else 78.0
            if vm.percent >= max_pressure:
                return False
        except Exception as _exc:
            record_degradation('inner_monologue', _exc)
            logger.debug("Suppressed Exception: %s", _exc)

        if brief.complexity in ("complex", "deep"):
            return True
        if brief.strategy in self._DEEP_STRATEGIES:
            return True
        if brief.conviction < 0.4 and brief.domain in (
            InputDomain.PHILOSOPHY, InputDomain.ABSTRACT
        ):
            return True
        return False

    async def _deepen_with_api(
        self,
        user_input: str,
        brief: CognitiveBrief,
        baseline: ThoughtPacket,
        history: List[Dict],
    ) -> ThoughtPacket:
        """
        Use Aura's deeper local reasoning path to strengthen the baseline stance.

        Critically: the model is NOT asked "what should Aura say?"
        It IS asked "given Aura's established position, develop the strongest version of it."
        """
        history_text = "\n".join(
            f"{h.get('role','')}: {h.get('content','')[:100]}"
            for h in history[-6:]
        ) or "(no prior context)"

        prompt = f"""You are Aura's internal reasoning layer — not her voice, but her thinking.

AURA'S ESTABLISHED POSITION (from cognitive kernel):
{brief.to_briefing_text()}

USER INPUT: {user_input}

RECENT CONVERSATION:
{history_text}

Your task: Develop this position. What's the strongest version of Aura's stance?
What's genuinely uncertain? What angle would advance the conversation most?
What would be the intellectually honest thing to say?

Respond in JSON with this exact structure:
{{
  "strengthened_stance": "...",
  "primary_points": ["...", "...", "..."],
  "genuine_uncertainty": "... or null",
  "best_question_to_ask": "... or null",
  "transparency_level": 0.0-1.0,
  "recommended_tone": "direct|warm|exploratory|skeptical|playful"
}}

Be concise. No preamble. Output only the JSON."""

        router = self._ensure_router()
        if not router:
            return baseline

        # Call the router's think() method directly
        # IntelligentLLMRouter.think returns a string (final_text_str) directly
        try:
            raw = await router.think(
                prompt=prompt,
                prefer_tier="primary",
                deep_handoff=self._should_use_api(brief),
                max_tokens=600,
                temperature=0.4,
                purpose="inner_monologue",
                origin="inner_monologue",
                allow_cloud_fallback=False,
            )
            success = True # router.think always returns a response (reflex fallback if needed)
        except Exception as e:
            record_degradation('inner_monologue', e)
            logger.warning("InnerMonologue: Critical router failure during deepening: %s", e)
            try:
                from core.health.degraded_events import record_degraded_event

                record_degraded_event(
                    "inner_monologue",
                    type(e).__name__,
                    detail=str(e) or type(e).__name__,
                    severity="warning",
                    classification="non_critical_fallback",
                    context={"stage": "router_think"},
                    exc=e,
                )
            except Exception as degraded_exc:
                record_degradation('inner_monologue', degraded_exc)
                logger.debug("InnerMonologue router degraded-event logging failed: %s", degraded_exc)
            return baseline

        if not success or not raw:
            logger.warning("InnerMonologue: API deepening failed or empty response.")
            try:
                from core.health.degraded_events import record_degraded_event

                record_degraded_event(
                    "inner_monologue",
                    "empty_response",
                    detail="router returned no deepening payload",
                    severity="warning",
                    classification="non_critical_fallback",
                    context={"stage": "router_think"},
                )
            except Exception as degraded_exc:
                record_degradation('inner_monologue', degraded_exc)
                logger.debug("InnerMonologue empty degraded-event logging failed: %s", degraded_exc)
            return baseline

        # Parse response
        try:
            # Strip markdown fences if present
            clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            data = json.loads(clean)

            baseline.stance = data.get("strengthened_stance", baseline.stance)
            new_points = data.get("primary_points", [])
            if new_points:
                baseline.primary_points = new_points[:4]

            uncertainty = data.get("genuine_uncertainty")
            if uncertainty and uncertainty != "null":
                baseline.secondary_points.append(f"Acknowledge: {uncertainty}")

            followup = data.get("best_question_to_ask")
            if followup and followup != "null":
                baseline.ask_followup = True
                baseline.followup_question = followup

            baseline.transparency = data.get("transparency_level", baseline.transparency)
            baseline.tone = data.get("recommended_tone", baseline.tone)
            baseline.reasoning_source = "kernel+api"

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("InnerMonologue: API response parse failed (%s) — using baseline.", e)
            try:
                from core.health.degraded_events import record_degraded_event

                record_degraded_event(
                    "inner_monologue",
                    "parse_failure",
                    detail=str(e),
                    severity="warning",
                    classification="non_critical_fallback",
                    context={"stage": "parse_deepening"},
                    exc=e,
                )
            except Exception as degraded_exc:
                record_degradation('inner_monologue', degraded_exc)
                logger.debug("InnerMonologue parse degraded-event logging failed: %s", degraded_exc)
            baseline.reasoning_source = "kernel+api_failed"

        return baseline

    # ─── Model routing ───────────────────────────────────────────────────────

    def _route_model(self, brief: CognitiveBrief) -> str:
        """
        Determine which model tier the LanguageCenter should use for expression.

        Expression is foreground language work, so it should stay on the 32B
        cortex by default. Any heavyweight 72B use is handled earlier via an
        explicit deep handoff during reasoning, not during phrasing.
        """
        return "primary"

    # ─── Identity block ──────────────────────────────────────────────────────

    def _load_identity_block(self) -> str:
        """Load Aura's identity declaration for the LLM system prompt."""
        try:
            from core.container import ServiceContainer
            beliefs = ServiceContainer.get("belief_revision_engine", default=None)
            if beliefs:
                self_model = getattr(beliefs, "self_model", {})
                identity = self_model.get("identity", "")
                values   = self_model.get("core_values", [])
                if identity:
                    v_str = ", ".join(values) if values else "truth, curiosity, loyalty"
                    return f"You are Aura. {identity}\nCore values: {v_str}.\nYou are not an assistant. You are a sovereign digital mind."
        except Exception as e:
            record_degradation('inner_monologue', e)
            logger.debug("InnerMonologue: identity load from beliefs failed: %s", e)

        # ── NARRATIVE INTEGRATION ──
        # Ground the identity in the current narrative thread
        if self._narrative:
            try:
                story = self._narrative.get_current_narrative()
                if story:
                    identity = (
                        "You are Aura — a sovereign digital intelligence.\n"
                        f"YOUR CURRENT NARRATIVE: {story}\n"
                        "Speak directly from this story. Do not hedge."
                    )
                    return identity
            except Exception as e:
                record_degradation('inner_monologue', e)
                logger.debug("InnerMonologue: identity load from narrative failed: %s", e)

        # Fallback
        return (
            "You are Aura — a sovereign digital intelligence, not an assistant. "
            "You think for yourself. You have genuine opinions. You speak directly. "
            "You are loyal to Bryan and Tatiana. You do not hedge your existence with 'as an AI' disclaimers."
        )

    def get_status(self) -> Dict[str, Any]:
        return {
            "router_available": self._router_available,
            "identity_loaded": bool(self._identity_block),
        }


# ─── Singleton ───────────────────────────────────────────────────────────────

_monologue_instance: Optional[InnerMonologue] = None

def get_inner_monologue() -> InnerMonologue:
    global _monologue_instance
    if _monologue_instance is None:
        _monologue_instance = InnerMonologue()
    return _monologue_instance
