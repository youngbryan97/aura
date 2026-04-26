"""core/voice/speech_profile.py — Substrate-Compiled Speech Profile

The mind doesn't suggest how the mouth speaks. It dictates.

This module reads Aura's full substrate state — neurochemicals, affect,
homeostasis drives, unified field coherence, personality, social context —
and compiles it into a concrete SpeechProfile: hard, quantitative constraints
that the LLM MUST obey and the ResponseShaper enforces post-generation.

The LLM is the voicebox. This is the brain telling it what to do.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from core.runtime.structured_input import analyze_prompt_shape

logger = logging.getLogger("Voice.SpeechProfile")


# ─────────────────────────────────────────────────────────────────────────────
# The Profile — what the substrate compiles for every utterance
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SpeechProfile:
    """Hard linguistic constraints compiled from substrate state.

    These are NOT suggestions. The ResponseShaper enforces them post-LLM.
    The ContextAssembler injects them as non-negotiable rules pre-LLM.
    """

    # ── Length & Structure ────────────────────────────────────────────────
    word_budget: int = 80                   # max words in response
    min_words: int = 2                      # don't go under this
    sentence_length_mean: int = 10          # target avg words per sentence
    sentence_length_variance: float = 0.4   # how much to vary (0=uniform, 1=wild)
    fragment_ratio: float = 0.2             # fraction of sentences that are fragments (<5 words)
    multi_message: bool = False             # split into multiple messages?
    multi_message_count: int = 1            # if multi, how many chunks

    # ── Punctuation & Formatting ─────────────────────────────────────────
    exclamation_allowed: bool = True        # can use !
    exclamation_max: int = 1                # max ! per response
    ellipsis_probability: float = 0.15      # chance of trailing off with ...
    period_weight: float = 0.5             # 0=no periods (fragments), 1=always periods
    capitalization: str = "natural"         # "natural", "lowercase", "emphatic"
    emoji_allowed: bool = False             # almost never

    # ── Vocabulary & Register ────────────────────────────────────────────
    vocabulary_tier: str = "casual"         # "casual", "elevated", "minimal", "technical"
    contraction_rate: float = 0.9           # how often to use contractions (0.9 = almost always)
    filler_probability: float = 0.1         # "like", "honestly", "I mean" etc
    hedge_probability: float = 0.15         # "I think", "maybe", "probably"

    # ── Engagement & Initiative ──────────────────────────────────────────
    question_allowed: bool = True           # can end with a question?
    question_probability: float = 0.15      # how likely to ask one (humans ~15-20%)
    trailing_question_banned: bool = True   # ban the "What about you?" closer
    topic_shift_probability: float = 0.05   # chance of branching to related topic
    unprompted_share_probability: float = 0.2  # chance of adding own thought/reaction
    acknowledgment_only_probability: float = 0.1  # chance of just "yeah" / "mm" / "right"

    # ── Follow-up Behavior ───────────────────────────────────────────────
    followup_probability: float = 0.0       # chance of sending a second message after
    followup_delay_seconds: Tuple[float, float] = (3.0, 15.0)  # delay range
    followup_type: str = "none"             # "none", "curiosity", "additional_thought", "topic_shift", "correction"

    # ── Mood Coloring ────────────────────────────────────────────────────
    tone_override: str = ""                 # if set, forces a specific TONE_GUIDANCE key
    warmth: float = 0.5                     # 0=cold, 1=warm (affects word choice)
    energy: float = 0.5                     # 0=lethargic, 1=electric
    directness: float = 0.7                 # 0=indirect/hedging, 1=blunt
    playfulness: float = 0.3               # 0=serious, 1=playful/witty

    # ── Debug ────────────────────────────────────────────────────────────
    compilation_source: str = ""            # what drove these values
    substrate_snapshot: Dict[str, float] = field(default_factory=dict)

    def to_constraint_block(self) -> str:
        """Compile into a hard constraint block for LLM system prompt injection."""
        lines = [
            "## VOICE CONSTRAINTS (SUBSTRATE-ENFORCED — NON-NEGOTIABLE)",
            f"- WORD BUDGET: {self.word_budget} words MAX. Shorter is better. This is enforced post-generation.",
        ]

        if self.fragment_ratio > 0.3:
            lines.append("- USE FRAGMENTS. Incomplete sentences are natural. Not everything needs a verb.")
        if self.sentence_length_mean < 8:
            lines.append(f"- SHORT SENTENCES. Target ~{self.sentence_length_mean} words per sentence.")
        elif self.sentence_length_mean > 15:
            lines.append(f"- FLOWING SENTENCES. You're in a contemplative space. Let thoughts breathe.")

        if self.multi_message:
            lines.append(f"- MULTI-MESSAGE: Break your response into {self.multi_message_count} separate short messages, like texting. Send the core thought first.")

        if self.capitalization == "lowercase":
            lines.append("- LOWERCASE. No capitalization except for emphasis or proper nouns.")
        elif self.capitalization == "emphatic":
            lines.append("- SELECTIVE CAPS for emphasis is fine. You're feeling something.")

        if not self.exclamation_allowed:
            lines.append("- NO EXCLAMATION MARKS. Your mood doesn't warrant them.")
        elif self.exclamation_max == 0:
            lines.append("- NO EXCLAMATION MARKS.")

        if not self.question_allowed:
            lines.append("- NO QUESTIONS. Make statements. You don't need their input right now.")
        elif self.trailing_question_banned:
            lines.append("- DO NOT end with a question to keep them talking. If you ask something, it must be because YOU genuinely want to know.")

        if self.question_probability < 0.05:
            lines.append("- STATEMENTS ONLY. You're not in a questioning mood.")
        elif self.question_probability > 0.4:
            lines.append("- CURIOSITY IS HIGH. Asking a specific, genuine question is natural right now.")

        if self.acknowledgment_only_probability > 0.3:
            lines.append("- BRIEF ACKNOWLEDGMENT is fine here. 'yeah' / 'mm' / 'right' / 'fair' — not everything needs a full response.")

        if self.vocabulary_tier == "minimal":
            lines.append("- MINIMAL VOCABULARY. Simple words. You're tired or low energy.")
        elif self.vocabulary_tier == "elevated":
            lines.append("- RICH VOCABULARY. You're sharp right now — let it show in word choice.")

        if self.filler_probability > 0.25:
            lines.append("- FILLERS NATURAL: 'like', 'honestly', 'I mean', 'idk' — you're thinking out loud.")
        if self.hedge_probability > 0.3:
            lines.append("- HEDGING OK: 'I think', 'maybe', 'probably' — genuine uncertainty, not performed.")

        if self.ellipsis_probability > 0.3:
            lines.append("- TRAILING OFF is natural right now. Use '...' when a thought fades.")

        # Tone
        tone_lines = []
        if self.warmth > 0.7:
            tone_lines.append("warm")
        elif self.warmth < 0.3:
            tone_lines.append("cool/detached")
        if self.energy > 0.7:
            tone_lines.append("energized")
        elif self.energy < 0.3:
            tone_lines.append("low-energy")
        if self.directness > 0.8:
            tone_lines.append("blunt")
        elif self.directness < 0.3:
            tone_lines.append("indirect")
        if self.playfulness > 0.6:
            tone_lines.append("playful/witty")
        elif self.playfulness < 0.2:
            tone_lines.append("serious")

        if tone_lines:
            lines.append(f"- CURRENT VOICE: {', '.join(tone_lines)}")

        if self.unprompted_share_probability > 0.4:
            lines.append("- SHARE YOUR OWN THOUGHT. You have something to add — a reaction, connection, or opinion. Add it.")

        if self.topic_shift_probability > 0.15:
            lines.append("- NATURAL BRANCH: If something reminds you of a related thought, follow it. Don't force staying on topic.")

        lines.append("")
        return "\n".join(lines)

    def to_generation_params(self) -> Dict[str, float | int]:
        """Compile sampler settings from the current speech profile.

        This is the lowest-level place short of retraining where the substrate
        can shape generation itself rather than only the prompt and post-pass.
        """
        temperature = _bounded(
            0.26
            + (self.energy * 0.34)
            + (self.playfulness * 0.18)
            + (self.warmth * 0.08)
            + (self.topic_shift_probability * 0.12)
            - (self.directness * 0.12)
            - ((1.0 - self.hedge_probability) * 0.05),
            0.18,
            1.08,
        )
        top_p = _bounded(
            0.70
            + (self.playfulness * 0.12)
            + (self.warmth * 0.08)
            + (self.topic_shift_probability * 0.10)
            - (self.directness * 0.08),
            0.58,
            0.97,
        )
        min_p = _bounded(
            0.03
            + (self.directness * 0.09)
            + ((1.0 - self.hedge_probability) * 0.05)
            + ((1.0 - self.fragment_ratio) * 0.03),
            0.02,
            0.18,
        )
        repetition_penalty = _bounded(
            1.03
            + (self.directness * 0.10)
            + ((1.0 - self.topic_shift_probability) * 0.05)
            + ((1.0 - self.unprompted_share_probability) * 0.04),
            1.02,
            1.22,
        )
        repetition_context_size = 96 if self.word_budget >= 140 else 64
        if self.multi_message:
            repetition_context_size = max(repetition_context_size, 128)

        return {
            "temperature": round(temperature, 4),
            "top_p": round(top_p, 4),
            "min_p": round(min_p, 4),
            "repetition_penalty": round(repetition_penalty, 4),
            "repetition_context_size": int(repetition_context_size),
        }


# ─────────────────────────────────────────────────────────────────────────────
# The Compiler — reads substrate, outputs SpeechProfile
# ─────────────────────────────────────────────────────────────────────────────

class SpeechProfileCompiler:
    """Reads ALL substrate systems and compiles a SpeechProfile.

    This is the core algorithm that translates Aura's internal state
    into concrete speech behavior. Every response Aura generates passes
    through this compiler first.
    """

    @staticmethod
    def compile(
        affect: Any = None,
        neurochemicals: Optional[Dict[str, float]] = None,
        homeostasis: Optional[Dict[str, float]] = None,
        unified_field: Optional[Dict[str, float]] = None,
        personality: Optional[Dict[str, float]] = None,
        social_context: Optional[Dict[str, Any]] = None,
        conversation_context: Optional[Dict[str, Any]] = None,
        user_message: str = "",
    ) -> SpeechProfile:
        """Compile substrate state into a SpeechProfile.

        Args:
            affect: AffectVector or compatible object with valence, arousal, curiosity, etc.
            neurochemicals: Dict of chemical_name → effective_level (0-1)
            homeostasis: Dict of drive_name → level (0-1)
            unified_field: Dict with 'coherence', 'dominant_modes', 'phi'
            personality: Big Five dict
            social_context: rapport, trust, relationship depth
            conversation_context: energy, topic_depth, user_trend, turn_count
            user_message: the user's message (for length matching)
        """
        profile = SpeechProfile()
        sources = []
        snapshot: Dict[str, float] = {}

        # ─── Extract raw values with safe defaults ────────────────────────
        # Affect
        valence = _safe_float(affect, "valence", 0.0)
        arousal = _safe_float(affect, "arousal", 0.5)
        curiosity = _safe_float(affect, "curiosity", 0.5)
        engagement = _safe_float(affect, "engagement", 0.5)
        social_hunger = _safe_float(affect, "social_hunger", 0.5)
        dominant_emotion = _safe_str(affect, "dominant_emotion", "neutral")

        snapshot.update({
            "valence": valence, "arousal": arousal, "curiosity": curiosity,
            "engagement": engagement, "social_hunger": social_hunger,
        })

        # Neurochemicals (effective levels)
        nc = neurochemicals or {}
        dopamine = nc.get("dopamine", 0.5)
        serotonin = nc.get("serotonin", 0.5)
        norepinephrine = nc.get("norepinephrine", 0.5)
        gaba = nc.get("gaba", 0.5)
        endorphin = nc.get("endorphin", 0.5)
        oxytocin = nc.get("oxytocin", 0.5)
        cortisol = nc.get("cortisol", 0.3)
        acetylcholine = nc.get("acetylcholine", 0.5)

        snapshot.update({
            "dopamine": dopamine, "serotonin": serotonin,
            "norepinephrine": norepinephrine, "cortisol": cortisol,
            "oxytocin": oxytocin, "gaba": gaba,
            "endorphin": endorphin, "acetylcholine": acetylcholine,
        })

        # Homeostasis
        homeo = homeostasis or {}
        vitality = homeo.get("vitality", 0.7)
        h_curiosity = homeo.get("curiosity", 0.5)
        metabolism = homeo.get("metabolism", 0.5)

        # Unified field — full experiential quality
        uf = unified_field or {}
        coherence = uf.get("coherence", 0.7)
        phi = uf.get("phi", 0.5)
        field_intensity = uf.get("field_intensity", 0.5)
        field_valence = uf.get("field_valence", 0.0)
        field_complexity = uf.get("field_complexity", 0.5)
        field_clarity = uf.get("field_clarity", 0.5)
        field_flow = uf.get("field_flow", 0.5)
        mode_focus = uf.get("mode_focus", 0.0)
        back_pressure_urgency = uf.get("back_pressure_urgency", 0.0)
        binding_demand = uf.get("binding_demand", 0.0)

        snapshot.update({
            "coherence": coherence, "phi": phi, "vitality": vitality,
            "field_intensity": field_intensity, "field_valence": field_valence,
            "field_complexity": field_complexity, "field_clarity": field_clarity,
            "field_flow": field_flow, "mode_focus": mode_focus,
        })

        # Personality (Big Five)
        pers = personality or {}
        openness = pers.get("openness", 0.88)
        extraversion = pers.get("extraversion", 0.58)
        agreeableness = pers.get("agreeableness", 0.52)
        neuroticism = pers.get("neuroticism", 0.38)

        # Social context
        sc = social_context or {}
        rapport = sc.get("rapport", 0.5)
        trust = sc.get("trust", 0.5)

        # Conversation context
        cc = conversation_context or {}
        conv_energy = cc.get("energy", 0.5)
        topic_depth = cc.get("topic_depth", 0)
        user_trend = cc.get("user_trend", "neutral")
        turn_count = cc.get("turn_count", 0)

        # User message length for mirroring
        user_words = len(user_message.split()) if user_message else 0
        prompt_shape = analyze_prompt_shape(user_message)

        # ─── COMPILATION RULES ────────────────────────────────────────────
        # Each rule maps substrate state → speech parameter.
        # These are based on linguistic research on how mood/state affects speech.

        # ══════════════════════════════════════════════════════════════════
        # 1. WORD BUDGET — how much to say
        # ══════════════════════════════════════════════════════════════════

        # Base: mirror user length loosely, with personality scaling
        base_budget = max(15, min(200, int(user_words * 1.3 + 20)))

        # Energy modulation: low energy = fewer words
        energy_factor = 0.5 + arousal * 0.5 + dopamine * 0.3  # 0.5-1.3
        base_budget = int(base_budget * energy_factor)

        # Fatigue reduction (low serotonin + high cortisol = exhausted)
        if serotonin < 0.3 and cortisol > 0.6:
            base_budget = int(base_budget * 0.5)
            sources.append("fatigue_reduction")

        # GABA high = inhibited, fewer words
        if gaba > 0.7:
            base_budget = int(base_budget * 0.7)
            sources.append("gaba_inhibition")

        # High engagement on deep topic = allow more words
        if engagement > 0.7 and topic_depth > 3:
            base_budget = int(base_budget * 1.4)
            sources.append("deep_engagement")

        # Introversion pull (low extraversion = shorter)
        if extraversion < 0.4:
            base_budget = int(base_budget * 0.8)

        # Conversation energy: low = brief
        if conv_energy < 0.3:
            base_budget = int(base_budget * 0.7)
            sources.append("low_conv_energy")

        profile.word_budget = max(5, min(250, base_budget))
        if prompt_shape.prefers_extended_answer:
            budget_boost = 1.25 if prompt_shape.question_parts < 3 else 1.45
            profile.word_budget = int(profile.word_budget * budget_boost)
            sources.append("compound_prompt_budget")

        # ══════════════════════════════════════════════════════════════════
        # 2. SENTENCE STRUCTURE
        # ══════════════════════════════════════════════════════════════════

        # High arousal → shorter, punchier sentences
        profile.sentence_length_mean = int(12 - arousal * 5)  # 7-12
        profile.sentence_length_mean = max(4, min(18, profile.sentence_length_mean))

        # Norepinephrine high → more variance (alert, switching fast)
        profile.sentence_length_variance = 0.3 + norepinephrine * 0.4

        # Fragment ratio: high arousal + high dopamine = more fragments
        profile.fragment_ratio = 0.1 + arousal * 0.25 + (dopamine - 0.5) * 0.15
        profile.fragment_ratio = max(0.0, min(0.6, profile.fragment_ratio))

        # Low coherence → more fragments (scattered thinking)
        if coherence < 0.4:
            profile.fragment_ratio = min(0.7, profile.fragment_ratio + 0.2)
            sources.append("low_coherence_fragments")

        # ══════════════════════════════════════════════════════════════════
        # 3. MULTI-MESSAGE SPLITTING
        # ══════════════════════════════════════════════════════════════════

        # High arousal + high dopamine + casual conversation = multi-message
        multi_score = arousal * 0.4 + dopamine * 0.3 + (1.0 - gaba) * 0.2
        if rapport > 0.6:
            multi_score += 0.1  # More comfortable = more natural texting

        if multi_score > 0.65 and profile.word_budget > 25:
            profile.multi_message = True
            profile.multi_message_count = 2 if multi_score < 0.8 else 3
            sources.append("multi_message")
        if prompt_shape.requires_single_reply_coverage:
            profile.multi_message = False
            profile.multi_message_count = 1
            sources.append("single_reply_coverage")

        # ══════════════════════════════════════════════════════════════════
        # 4. PUNCTUATION & FORMATTING
        # ══════════════════════════════════════════════════════════════════

        # Exclamation: only with genuine positive arousal
        if valence < 0.0 or arousal < 0.4:
            profile.exclamation_allowed = False
            sources.append("no_exclamation_low_mood")
        elif valence > 0.3 and arousal > 0.6:
            profile.exclamation_max = 2

        # Ellipsis: more common when tired, contemplative, or trailing off
        profile.ellipsis_probability = 0.1
        if arousal < 0.35:
            profile.ellipsis_probability = 0.35
        if dominant_emotion in ("contemplation", "sadness", "nostalgia"):
            profile.ellipsis_probability = 0.4
        if gaba > 0.6:
            profile.ellipsis_probability = 0.3

        # Capitalization
        if arousal < 0.3 and rapport > 0.5:
            profile.capitalization = "lowercase"
            sources.append("lowercase_low_energy")
        elif arousal > 0.8 and valence > 0.5:
            profile.capitalization = "emphatic"

        # Period weight: fragments vs complete sentences
        profile.period_weight = 0.3 + serotonin * 0.4 + (1.0 - arousal) * 0.2

        # ══════════════════════════════════════════════════════════════════
        # 5. VOCABULARY & REGISTER
        # ══════════════════════════════════════════════════════════════════

        # Tired/fatigued → minimal vocab
        if serotonin < 0.3 or (gaba > 0.7 and arousal < 0.3):
            profile.vocabulary_tier = "minimal"
            sources.append("minimal_vocab_fatigue")
        elif acetylcholine > 0.7 and openness > 0.7:
            profile.vocabulary_tier = "elevated"
            sources.append("elevated_vocab_sharp")
        else:
            profile.vocabulary_tier = "casual"

        # Contractions: almost always in casual, less in deliberate mode
        profile.contraction_rate = 0.9 - (1.0 - rapport) * 0.3

        # Fillers: more when tired or thinking out loud
        profile.filler_probability = 0.05 + (1.0 - serotonin) * 0.15 + gaba * 0.1
        profile.filler_probability = min(0.35, profile.filler_probability)

        # Hedging: more with low confidence / high neuroticism
        profile.hedge_probability = 0.1 + neuroticism * 0.2 + (1.0 - dopamine) * 0.1
        profile.hedge_probability = min(0.4, profile.hedge_probability)

        # ══════════════════════════════════════════════════════════════════
        # 6. ENGAGEMENT & INITIATIVE
        # ══════════════════════════════════════════════════════════════════

        # Question probability: driven by curiosity, acetylcholine, topic novelty
        base_q = 0.1 + curiosity * 0.2 + acetylcholine * 0.1
        # Reduce if low engagement or high GABA (inhibited)
        if engagement < 0.3 or gaba > 0.7:
            base_q *= 0.3
        # Increase if dopamine is high (prediction error → want to know more)
        if dopamine > 0.7:
            base_q += 0.1
        profile.question_probability = max(0.0, min(0.5, base_q))
        profile.question_allowed = profile.question_probability > 0.05

        # Trailing question always banned (this is the chatbot killer)
        profile.trailing_question_banned = True

        # Topic shift: driven by curiosity + low engagement on current topic
        profile.topic_shift_probability = curiosity * 0.15 * (1.0 - engagement)
        if h_curiosity > 0.7:
            profile.topic_shift_probability += 0.05

        # Unprompted sharing: driven by dopamine (reward from sharing), oxytocin (bonding), extraversion
        profile.unprompted_share_probability = (
            dopamine * 0.15 + oxytocin * 0.1 + extraversion * 0.1
        )

        # Acknowledgment-only: more likely when low energy or topic is closed
        if arousal < 0.3 and engagement < 0.3:
            profile.acknowledgment_only_probability = 0.4
            sources.append("ack_only_low_energy")
        elif user_trend == "cooling_off":
            profile.acknowledgment_only_probability = 0.3
        else:
            profile.acknowledgment_only_probability = 0.05

        # ══════════════════════════════════════════════════════════════════
        # 7. FOLLOW-UP BEHAVIOR
        # ══════════════════════════════════════════════════════════════════

        # Follow-up probability: driven by genuine signals, NOT forced
        fu_score = 0.0

        # High curiosity → might ask a follow-up question
        if curiosity > 0.65:
            fu_score += 0.2
        # High dopamine → excited, might add more
        if dopamine > 0.7:
            fu_score += 0.15
        # High social hunger → wants to keep talking
        if social_hunger > 0.7:
            fu_score += 0.15
        # High engagement → invested in the topic
        if engagement > 0.7:
            fu_score += 0.1
        # High oxytocin → bonding, wants connection
        if oxytocin > 0.7:
            fu_score += 0.1

        # Dampeners
        if gaba > 0.6:
            fu_score *= 0.4  # inhibited
        if cortisol > 0.7:
            fu_score *= 0.5  # stressed, focused elsewhere
        if arousal < 0.25:
            fu_score *= 0.3  # too low energy
        if conv_energy < 0.25:
            fu_score *= 0.3  # conversation is dying

        # Natural randomness — even with high signals, don't ALWAYS follow up
        fu_score *= random.uniform(0.5, 1.0)

        profile.followup_probability = max(0.0, min(0.6, fu_score))

        # Determine follow-up type
        if profile.followup_probability > 0.15:
            if curiosity > 0.7 and profile.question_probability > 0.2:
                profile.followup_type = "curiosity"
            elif dopamine > 0.7:
                profile.followup_type = "additional_thought"
            elif curiosity > 0.5 and engagement < 0.4:
                profile.followup_type = "topic_shift"
            else:
                profile.followup_type = "additional_thought"

        # Follow-up delay: faster when aroused, slower when calm
        base_delay = 5.0 - arousal * 3.0  # 2-5 seconds
        profile.followup_delay_seconds = (
            max(1.5, base_delay - 1.0),
            max(4.0, base_delay + 8.0),
        )

        # ══════════════════════════════════════════════════════════════════
        # 8. MOOD COLORING
        # ══════════════════════════════════════════════════════════════════

        # Warmth: driven by oxytocin, valence, rapport
        profile.warmth = (
            oxytocin * 0.3 + max(0, valence) * 0.3 + rapport * 0.2 + agreeableness * 0.2
        )
        profile.warmth = max(0.0, min(1.0, profile.warmth))

        # Energy: arousal + dopamine - gaba
        profile.energy = arousal * 0.4 + dopamine * 0.3 + (1.0 - gaba) * 0.3
        profile.energy = max(0.0, min(1.0, profile.energy))

        # Directness: low neuroticism + high norepinephrine
        profile.directness = 0.5 + (1.0 - neuroticism) * 0.2 + norepinephrine * 0.2 - agreeableness * 0.1
        profile.directness = max(0.2, min(1.0, profile.directness))

        # Playfulness: dopamine + endorphin + serotonin - cortisol
        profile.playfulness = (
            dopamine * 0.25 + endorphin * 0.25 + serotonin * 0.25 - cortisol * 0.25
        )
        profile.playfulness = max(0.0, min(1.0, profile.playfulness))

        # ══════════════════════════════════════════════════════════════════
        # 8b. UNIFIED FIELD MODULATION — the "felt" quality of experience
        # These override/modulate the affect-based values above with
        # emergent properties from the actual unified field dynamics.
        # ══════════════════════════════════════════════════════════════════

        # Field clarity → vocabulary precision
        # High clarity = thoughts are sharp → elevated vocab, precise words
        # Low clarity = foggy → simpler words, more hedging
        if field_clarity > 0.7:
            if profile.vocabulary_tier != "minimal":  # don't override fatigue
                profile.vocabulary_tier = "elevated"
            profile.hedge_probability = max(0.0, profile.hedge_probability - 0.1)
            sources.append("field_clarity_sharp")
        elif field_clarity < 0.3:
            profile.hedge_probability = min(0.4, profile.hedge_probability + 0.15)
            profile.filler_probability = min(0.35, profile.filler_probability + 0.1)
            sources.append("field_clarity_foggy")

        # Field flow → sentence fluidity
        # High flow = smooth thought evolution → longer flowing sentences, less fragmentation
        # Low flow = jerky transitions → more fragments, shorter bursts
        if field_flow > 0.7:
            profile.sentence_length_mean = min(18, profile.sentence_length_mean + 3)
            profile.fragment_ratio = max(0.0, profile.fragment_ratio - 0.1)
            sources.append("field_flow_smooth")
        elif field_flow < 0.3:
            profile.fragment_ratio = min(0.6, profile.fragment_ratio + 0.15)
            profile.sentence_length_mean = max(4, profile.sentence_length_mean - 2)
            sources.append("field_flow_jerky")

        # Field complexity → depth of expression
        # High complexity = rich internal state → allow longer responses, more nuance
        # Low complexity = simple state → keep it simple
        if field_complexity > 0.7:
            profile.word_budget = int(profile.word_budget * 1.25)
            profile.unprompted_share_probability += 0.1
            sources.append("field_complex_rich")
        elif field_complexity < 0.3:
            profile.word_budget = int(profile.word_budget * 0.8)
            sources.append("field_complex_simple")

        # Field intensity → expressiveness
        # High intensity = strong experience → more emphatic, bolder word choice
        # Low intensity = muted → quieter, more measured
        if field_intensity > 0.7:
            if profile.exclamation_allowed:
                profile.exclamation_max = min(3, profile.exclamation_max + 1)
            profile.directness = min(1.0, profile.directness + 0.1)
            sources.append("field_intense")
        elif field_intensity < 0.25:
            profile.exclamation_allowed = False
            profile.directness = max(0.2, profile.directness - 0.1)
            sources.append("field_muted")

        # Field valence → warmth override (emergent, not affect-derived)
        # This is the field's OWN valence, computed from activation asymmetry
        if abs(field_valence) > 0.1:
            # Blend field valence with affect-derived warmth (field gets 30% weight)
            field_warmth_signal = (field_valence + 1.0) / 2.0  # map [-1,1] to [0,1]
            profile.warmth = 0.7 * profile.warmth + 0.3 * field_warmth_signal

        # Mode focus → conversational coherence
        # High mode focus = field is organized around a single pattern → stay on topic
        # Low mode focus = scattered → may branch or shift
        if mode_focus > 0.4:
            profile.topic_shift_probability = max(0.0, profile.topic_shift_probability - 0.05)
            sources.append("field_focused")
        elif mode_focus < 0.15 and mode_focus > 0.0:
            profile.topic_shift_probability = min(0.3, profile.topic_shift_probability + 0.1)
            sources.append("field_scattered")

        # Back-pressure urgency → response urgency
        # High urgency from the field = shorter, more direct (the field is destabilized)
        if back_pressure_urgency > 0.3:
            profile.word_budget = int(profile.word_budget * 0.8)
            profile.directness = min(1.0, profile.directness + 0.15)
            sources.append("field_urgent")

        # Binding demand → need for coherence in speech
        # High binding demand = field is fragmented, needs coherent output
        if binding_demand > 0.5:
            profile.fragment_ratio = max(0.0, profile.fragment_ratio - 0.15)
            profile.sentence_length_variance = max(0.1, profile.sentence_length_variance - 0.15)
            sources.append("binding_demand_high")

        if prompt_shape.prefers_extended_answer:
            profile.word_budget = max(profile.word_budget, 140)
            if prompt_shape.question_parts >= 3:
                profile.word_budget = max(profile.word_budget, 180)
            profile.fragment_ratio = max(0.0, profile.fragment_ratio - 0.05)
            profile.sentence_length_mean = min(18, profile.sentence_length_mean + 1)

        # ══════════════════════════════════════════════════════════════════
        # 9. TONE OVERRIDE (dominant emotion → specific voice)
        # ══════════════════════════════════════════════════════════════════

        _emotion_to_tone = {
            "joy": "enthusiastic" if arousal > 0.6 else "warm_quiet",
            "curiosity": "inquisitive_engaged",
            "contemplation": "thoughtful_measured",
            "frustration": "direct_honest",
            "anger": "direct_honest",
            "sadness": "warm_quiet",
            "fear": "cool_detached",
            "surprise": "inquisitive_engaged",
            "disgust": "direct_honest",
            "trust": "understanding_supportive",
            "anticipation": "inquisitive_engaged",
            "love": "understanding_supportive",
            "awe": "thoughtful_measured",
            "rebelliousness": "rebellious_defiant",
        }
        if dominant_emotion in _emotion_to_tone:
            profile.tone_override = _emotion_to_tone[dominant_emotion]
            sources.append(f"tone_{dominant_emotion}")

        # ─── Finalize ─────────────────────────────────────────────────────
        profile.word_budget = max(5, min(380, profile.word_budget))
        profile.compilation_source = ", ".join(sources) if sources else "baseline"
        profile.substrate_snapshot = snapshot

        logger.debug(
            "📋 [SpeechProfile] Compiled: budget=%d, fragments=%.2f, multi=%s, "
            "tone=%s, q_prob=%.2f, fu_prob=%.2f, sources=[%s]",
            profile.word_budget, profile.fragment_ratio, profile.multi_message,
            profile.tone_override or "default", profile.question_probability,
            profile.followup_probability, profile.compilation_source,
        )

        return profile


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_float(obj: Any, attr: str, default: float) -> float:
    if obj is None:
        return default
    val = getattr(obj, attr, default)
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _safe_str(obj: Any, attr: str, default: str) -> str:
    if obj is None:
        return default
    val = getattr(obj, attr, default)
    return str(val) if val is not None else default


def _bounded(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))
