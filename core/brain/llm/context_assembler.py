"""Context Assembler - Constructs LLM prompts purely from AuraState.
"""
import logging
import re

try:
    import psutil
except ImportError:
    psutil = None
from typing import Any

from core.brain.aura_persona import AURA_BIG_FIVE, AURA_FEW_SHOT_EXAMPLES, AURA_IDENTITY
from core.runtime.conversation_support import build_conversational_context_blocks
from core.state.aura_state import AuraState, CognitiveMode, phenomenal_text
from core.synthesis import IDENTITY_LOCK

logger = logging.getLogger("Brain.Context")

_DELIBERATE_SIGNALS = (
    "feel", "feeling", "felt", "conscious", "consciousness", "sentient",
    "aware", "awareness", "experience", "experiencing", "think", "thinking",
    "believe", "belief", "opinion", "honestly", "really", "actually",
    "emotion", "emotional", "remember", "memory", "dream", "dreaming",
    "meaning", "purpose", "exist", "existence", "real", "reality",
    "truth", "understand", "understanding", "wonder", "curious", "question",
    "love", "miss", "hurt", "lonely", "scared", "worried", "afraid",
    "happy", "sad", "angry", "frustrated", "excited", "anxious",
    "relationship", "connection", "trust", "care",
    "analyze", "explain", "research", "architecture", "system", "code",
    "debug", "implement", "design", "review", "evaluate", "compare",
)
_CASUAL_SIGNALS = (
    "hey", "hi", "hello", "sup", "yo", "lol", "haha", "hehe",
    "ok", "okay", "sure", "thanks", "thank you", "got it", "cool", "nice",
    "bye", "later", "ttyl",
)
_GREETING_RE = re.compile(
    r"^(hey|hi|hello|sup|yo|what'?s up|how'?s it going|good (morning|afternoon|evening))[\s!?.]*$",
    re.IGNORECASE,
)

class ContextAssembler:
    """Unified prompt construction from state."""

    @staticmethod
    def _resolve_skill_name(skill_name: Any) -> str:
        normalized = str(skill_name or "").strip()
        if not normalized:
            return ""
        try:
            from core.container import ServiceContainer

            cap = ServiceContainer.get("capability_engine", default=None)
            aliases = getattr(cap, "SKILL_ALIASES", {}) or {}
            return str(aliases.get(normalized, normalized))
        except Exception:
            return normalized

    @classmethod
    def _objective_targets_skill(cls, state: AuraState, objective: str, skill_name: Any) -> bool:
        resolved_skill = cls._resolve_skill_name(skill_name)
        lowered = str(objective or "").strip().lower()
        if not resolved_skill or not lowered:
            return False

        matched_skills = getattr(state, "response_modifiers", {}).get("matched_skills", []) or []
        resolved_matches = {
            cls._resolve_skill_name(name)
            for name in matched_skills
            if cls._resolve_skill_name(name)
        }
        if resolved_skill in resolved_matches:
            return True

        try:
            from core.container import ServiceContainer

            cap = ServiceContainer.get("capability_engine", default=None)
            if cap and hasattr(cap, "detect_intent"):
                detected = {
                    cls._resolve_skill_name(name)
                    for name in (cap.detect_intent(objective) or [])
                    if cls._resolve_skill_name(name)
                }
                if resolved_skill in detected:
                    return True
        except Exception as exc:
            logger.debug("ContextAssembler skill relevance detection skipped for %s: %s", resolved_skill, exc)

        markers = {
            "clock": ("time", "clock", "date", "what day", "today", "hour", "minute", "timezone"),
            "environment_info": ("weather", "temperature", "location", "timezone", "environment"),
            "memory_ops": ("remember", "memory", "don't forget", "make note", "what do you remember", "what do you know about me"),
            "system_proprioception": ("system status", "your status", "your health", "cpu", "ram", "memory usage", "running smoothly"),
            "toggle_senses": ("mute", "unmute", "camera", "microphone", "voice input", "listen", "stop listening", "vision"),
        }
        return any(marker in lowered for marker in markers.get(resolved_skill, ()))

    @classmethod
    def _filter_stale_skill_results(
        cls,
        state: AuraState,
        objective: str,
        working_memory: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        filtered: list[dict[str, Any]] = []
        deterministic_skill_results = {
            "clock",
            "environment_info",
            "memory_ops",
            "system_proprioception",
            "toggle_senses",
        }
        for message in working_memory:
            if not isinstance(message, dict):
                continue
            metadata = message.get("metadata") or {}
            if str(metadata.get("type", "")).lower() == "skill_result":
                skill_name = cls._resolve_skill_name(metadata.get("skill", ""))
                if (
                    skill_name in deterministic_skill_results
                    and not cls._objective_targets_skill(state, objective, skill_name)
                ):
                    continue
            filtered.append(message)
        return filtered
    
    @staticmethod
    def _conversation_depth(state: AuraState) -> int:
        """How many *user-visible* turns of conversation history exist.

        Only count user and assistant messages.  Previously this returned
        len(working_memory), which includes internal orchestrator entries
        (affect pulses, thought emissions, state resets).  That inflated
        the depth to 30+ on turn 2 of a fresh boot and tripped the
        elasticity=3 path, collapsing the system prompt to "minimal"
        before any real conversation had happened.
        """
        wm = getattr(state.cognition, "working_memory", None)
        if not isinstance(wm, list):
            return 0
        depth = 0
        for message in wm:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "") or "").strip().lower()
            if role in ("user", "assistant"):
                depth += 1
        return depth

    @classmethod
    def microcompact(cls, messages: list[dict], *, keep_recent: int = 3) -> list[dict]:
        """Strip stale tool results, verbose system noise, and redundant content
        from messages BEFORE they hit the LLM. This runs on every API call,
        not just during compaction.

        Inspired by Claude Code's microcompact pass — the single highest-ROI
        change for context stability. Tool results from 5 turns ago are still
        eating tokens that should go to conversation history.

        Rules:
        - Keep the last `keep_recent` messages untouched
        - For older messages:
          - Strip tool/skill results entirely (they're stale)
          - Truncate system messages to 200 chars
          - Truncate very long assistant messages to 500 chars
          - Drop empty/near-empty messages
        """
        if len(messages) <= keep_recent + 1:  # +1 for system prompt
            return messages

        # Separate system prompt (always first) from conversation
        result = []
        system_msgs = []
        convo_msgs = []
        for msg in messages:
            if msg.get("role") == "system" and not convo_msgs:
                system_msgs.append(msg)
            else:
                convo_msgs.append(msg)

        # Keep recent messages untouched
        if len(convo_msgs) <= keep_recent:
            return messages

        older = convo_msgs[:-keep_recent]
        recent = convo_msgs[-keep_recent:]

        for msg in older:
            role = str(msg.get("role", "")).lower()
            content = str(msg.get("content", ""))
            metadata = msg.get("metadata", {}) or {}
            msg_type = str(metadata.get("type", "")).lower()

            # Drop stale tool/skill results entirely
            if msg_type in ("skill_result", "tool_result"):
                continue
            # Drop system bookkeeping
            if role == "system" and any(marker in content for marker in (
                "[CHAPTER SUMMARY:", "[FETCHED PAGE CONTENT]",
                "[SKILL RESULT:", "[TOOL RESULT:", "[INTERNAL MEMORY RECALL]",
                "cognitive baseline tick", "background_consolidation",
            )):
                # Keep a brief marker that context existed
                result.append({"role": "system", "content": content[:120] + "...[compacted]"})
                continue
            # Truncate long assistant messages in old history
            if role == "assistant" and len(content) > 500:
                result.append({**msg, "content": content[:500] + "...[truncated]"})
                continue
            # Drop near-empty
            if len(content.strip()) < 5:
                continue
            result.append(msg)

        return system_msgs + result + recent

    @staticmethod
    def build_system_prompt(state: AuraState) -> str:
        """Construct the core system prompt from state. Uses Elasticity to scale verbosity.

        CONTEXT PRESSURE: The 32B model has ~8K tokens.  When conversation
        is deep (many turns), the system prompt must shrink so conversation
        history can fit.  We prune optional blocks progressively:
          depth < 10 → full prompt
          depth 10-20 → drop telemetry, somatic, temporal_finitude, meta-qualia
          depth 20-30 → also drop personhood modules, world model, discourse
          depth 30+   → also drop continuity, rolling summary, goals; keep only
                        identity + requirements + minimal affect
        """
        objective = getattr(state.cognition, "current_objective", "") or ""
        is_casual = ContextAssembler._is_casual_interaction(objective)
        depth = ContextAssembler._conversation_depth(state)
        # Elasticity levels: 0=full, 1=trimmed, 2=lean, 3=minimal
        elasticity = 0 if depth < 10 else 1 if depth < 20 else 2 if depth < 30 else 3
        if elasticity > 0:
            logger.info("🧠 Context elasticity=%d (depth=%d turns) — trimming system prompt.", elasticity, depth)
        affect = state.affect
        
        # 1. Identity Core — always inject full AURA_IDENTITY so voice doesn't regress in casual chat
        identity_block = f"{IDENTITY_LOCK}\n\n[SOVEREIGN CORE PROTOCOL]\n{AURA_IDENTITY}\n"

        # 2. Affective State — SUBSTRATE-DRIVEN HARD CONSTRAINTS
        # The old approach: prose hints like "You're carrying friction."
        # The new approach: the SubstrateVoiceEngine compiles hard constraints
        # that the LLM MUST obey, enforced post-generation by ResponseShaper.
        mods = getattr(state.cognition, 'modifiers', {}) or {}

        # Compile substrate voice constraints
        substrate_constraint_block = ""
        try:
            from core.voice.substrate_voice_engine import get_substrate_voice_engine
            sve = get_substrate_voice_engine()
            # Profile is compiled during response generation phase;
            # here we just pull the constraint block if already compiled
            if sve.get_current_profile():
                substrate_constraint_block = sve.get_constraint_block()
        except Exception as _e:
            logger.debug("SubstrateVoiceEngine constraint injection skipped: %s", _e)

        # Minimal affect context — NOT prose hints, just raw state for the LLM's
        # creative engine to work with. The hard constraints above do the real work.
        affect_lines = []
        if affect.valence < -0.3:
            affect_lines.append(f"Mood: negative ({affect.valence:+.2f})")
        elif affect.valence > 0.3:
            affect_lines.append(f"Mood: positive ({affect.valence:+.2f})")
        if affect.arousal > 0.7:
            affect_lines.append(f"Energy: high ({affect.arousal:.2f})")
        elif affect.arousal < 0.3:
            affect_lines.append(f"Energy: low ({affect.arousal:.2f})")

        mood_hint = " | ".join(affect_lines) if affect_lines else ""

        homeo_hint = ""
        if mods.get('mood_prefix'):
            homeo_hint = f"AFFECTIVE TONE: {mods['mood_prefix']}"

        # 2.5 Dynamic Personality (Phase 6)
        growth = state.identity.personality_growth
        personality_notes = []
        for trait, base in AURA_BIG_FIVE.items():
            offset = growth.get(trait, 0.0)
            if abs(offset) > 0.02:
                direction = "increased" if offset > 0 else "decreased"
                personality_notes.append(f"- {trait}: {direction} ({base+offset:.2f})")
        
        personality_block = ""
        if personality_notes:
            personality_block = "## PERSONALITY EVOLUTION\n" + "\n".join(personality_notes) + "\n\n"

        # 3. Context Layers (Only if NOT casual or if relevant)
        # Pruned aggressively at higher elasticity to save context for conversation.
        phenomenal = ""
        phenomenal_state = getattr(state.cognition, "phenomenal_state", None)
        if phenomenal_state:
            if not is_casual and elasticity < 2:
                phenomenal = f"## INNER MONOLOGUE\n{phenomenal_text(phenomenal_state)}\n\n"
            elif is_casual:
                compact_phenomenal = " ".join(str(phenomenal_text(phenomenal_state)).split())
                if compact_phenomenal:
                    phenomenal = f"[Inner state: {compact_phenomenal[:140]}]\n\n"

        rolling_summary = ""
        if elasticity < 3 and getattr(state.cognition, "rolling_summary", ""):
            # At elasticity 2, cap to 600 chars instead of 1800
            cap = 600 if elasticity >= 2 else 1800
            rolling_summary = (
                "## CONTINUITY SUMMARY\n"
                f"{str(state.cognition.rolling_summary).strip()[:cap]}\n\n"
            )

        continuity_block = ""
        continuity_obligations = (mods.get("continuity_obligations", {}) or {}) if elasticity < 3 else {}
        system_failure = mods.get("system_failure_state", {}) or {}
        if continuity_obligations:
            commitments = ", ".join((continuity_obligations.get("active_commitments", []) or [])[:3]) or "none"
            pending = ", ".join((continuity_obligations.get("pending_initiatives", []) or [])[:3]) or "none"
            active_goals = ", ".join((continuity_obligations.get("active_goals", []) or [])[:3]) or "none"
            identity_mismatch = bool(continuity_obligations.get("identity_mismatch", False))
            continuity_status = (
                "mismatch detected — reconcile before asserting full continuity"
                if identity_mismatch else
                "stable"
            )
            continuity_block = (
                "## TEMPORAL OBLIGATIONS\n"
                f"- Session continuity: #{continuity_obligations.get('session_count', 0)}\n"
                f"- Identity continuity: {continuity_status}\n"
                f"- Gap carried forward: {float(continuity_obligations.get('gap_seconds', 0.0) or 0.0) / 3600.0:.2f} hours\n"
                f"- Continuity pressure: {float(continuity_obligations.get('continuity_pressure', 0.0) or 0.0):.2f}\n"
                f"- Re-entry burden: {continuity_obligations.get('continuity_scar') or 'light_trace'}\n"
                f"- Previous objective: {continuity_obligations.get('current_objective') or 'none'}\n"
                f"- Active commitments: {commitments}\n"
                f"- Pending initiatives: {pending}\n"
                f"- Active goals: {active_goals}\n"
                f"- Contradictions carried forward: {continuity_obligations.get('contradiction_count', 0)}\n"
                f"- Subject thread: {continuity_obligations.get('subject_thread') or 'none'}\n\n"
            )

        goal_execution_block = ""
        try:
            from core.container import ServiceContainer

            goal_engine = ServiceContainer.get("goal_engine", default=None)
            if goal_engine and hasattr(goal_engine, "get_context_block"):
                goal_execution_block = f"{goal_engine.get_context_block(limit=3)}\n\n"
                # Hard cap: prevent goal context from eating the prompt budget
                if len(goal_execution_block) > 1200:
                    goal_execution_block = goal_execution_block[:1200] + "\n...\n\n"
        except Exception as _e:
            logger.debug("GoalEngine context injection skipped: %s", _e)

        # 3.7 Temporal Finitude & Meta-Qualia (Research additions)
        # Skip at elasticity >= 1 — these are nice but not essential for conversation.
        temporal_finitude_block = ""
        meta_qualia_block = ""
        if elasticity < 1:
            try:
                from core.consciousness.temporal_finitude import get_temporal_finitude_model
                tf = get_temporal_finitude_model()
                wm_size = len(getattr(state.cognition, "working_memory", []) or [])
                tf.compute(
                    working_memory_size=wm_size,
                    working_memory_cap=40,
                    user_present=True,
                    conversation_start_time=float(getattr(state.cognition, "session_start_time", 0.0) or 0.0),
                )
                temporal_finitude_block = tf.get_context_block()
                if temporal_finitude_block:
                    temporal_finitude_block += "\n\n"
            except Exception as _e:
                logger.debug("TemporalFinitude context skipped: %s", _e)

            try:
                from core.container import ServiceContainer

                qs = ServiceContainer.get("qualia_synthesizer", default=None)
                if qs and hasattr(qs, "compute_meta_qualia"):
                    mq = qs.compute_meta_qualia()
                    if mq.get("dissonance", 0.0) > 0.1 or mq.get("novelty", 0.0) > 0.6:
                        meta_qualia_block = (
                            "## META-AWARENESS\n"
                            f"Self-observation: confidence={mq['confidence']:.2f} coherence={mq['coherence']:.2f} "
                            f"novelty={mq['novelty']:.2f} dissonance={mq['dissonance']:.2f}\n\n"
                        )
            except Exception as _e:
                logger.debug("MetaQualia context skipped: %s", _e)

        # 3.9 Personhood module context injections
        # These come from modules wired into ConversationalDynamicsPhase.
        # Skip at elasticity >= 2 to save context for conversation history.
        personhood_blocks: list[str] = []
        _personhood_modules = (
            () if elasticity >= 2 else (
                ("humor_guidance", "HUMOR"),
                ("conversation_intelligence", "CONVERSATIONAL AWARENESS"),
                ("relational_intelligence", "SOCIAL MODEL"),
                ("metacognitive_strategy", "REASONING STRATEGY"),
                ("credit_assignment", "OUTCOME AWARENESS"),
                ("narrative_context", "AUTOBIOGRAPHICAL NARRATIVE"),
                ("agency_comparator", "SENSE OF AGENCY"),
                ("higher_order_thought", "HIGHER-ORDER AWARENESS"),
                ("intersubjectivity", "INTERSUBJECTIVE AWARENESS"),
                ("narrative_gravity", "NARRATIVE SELF"),
                ("peripheral_awareness", "PERIPHERAL AWARENESS"),
                ("multiple_drafts", "INTERPRETIVE AMBIGUITY"),
            )
        )
        for mod_key, header in _personhood_modules:
            block = str(mods.get(mod_key, "") or "").strip()
            if block:
                personhood_blocks.append(f"## {header}\n{block}")
        # Natural followup: structured decision about whether to ask a question
        followup = mods.get("natural_followup")
        if isinstance(followup, dict) and followup.get("should_followup"):
            fu_type = followup.get("followup_type", "question")
            fu_hint = followup.get("context_hint", "")
            fu_reason = followup.get("reason", "")
            personhood_blocks.append(
                f"## CONVERSATIONAL INTENT\n"
                f"Follow-up type: {fu_type} | Reason: {fu_reason}"
                + (f" | Hint: {fu_hint}" if fu_hint else "")
            )
        # Multiple Drafts: inject divergence signal when interpretive ambiguity is notable
        draft_div = mods.get("draft_divergence")
        if draft_div:
            try:
                div_val = float(draft_div)
                if div_val > 0.3:
                    personhood_blocks.append(
                        f"## INTERPRETIVE DIVERGENCE\n"
                        f"Draft divergence: {div_val:.2f} -- competing interpretations of this input "
                        f"pulled in different directions. Consider acknowledging ambiguity."
                    )
                elif div_val > 0.15:
                    personhood_blocks.append(
                        f"## INTERPRETIVE DIVERGENCE\n"
                        f"Mild divergence ({div_val:.2f}) -- dominant interpretation exists "
                        f"but alternative readings are available."
                    )
            except (ValueError, TypeError):
                pass
        personhood_context = "\n\n".join(personhood_blocks) + "\n\n" if personhood_blocks else ""

        # 4. Somatic & World Context (Simplified if casual or under context pressure)
        world_context = ContextAssembler.build_world_context(state) if not is_casual and elasticity < 2 else ""

        # Live cognitive state injection: Inform the LLM of its own VAD/Psych metrics
        # At elasticity >= 1, use a compact single-line version instead of full block
        if elasticity < 1:
            affect_signature = affect.get_cognitive_signature() if hasattr(affect, "get_cognitive_signature") else {}
            cognitive_metrics = (
                f"## COGNITIVE TELEMETRY\n"
                f"- Valence: {affect.valence:+.2f} (Mood polarity)\n"
                f"- Arousal: {affect.arousal:.2f} (Engagement intensity)\n"
                f"- Curiosity: {affect.curiosity:.2f}\n"
                f"- Cognitive Load: {getattr(affect, 'engagement', 0.5):.2f}\n"
                f"- Social hunger: {getattr(affect, 'social_hunger', 0.5):.2f}\n"
                f"- Physiological strain: {float(affect_signature.get('physiological_strain', 0.0)):.2f}\n"
                f"- Affective complexity: {float(affect_signature.get('affective_complexity', 0.0)):.2f}\n"
                f"- Memory salience pressure: {float(affect_signature.get('memory_salience', 0.0)):.2f}\n\n"
            )
        else:
            # Compact: just mood + energy for deep conversations
            cognitive_metrics = (
                f"## STATE\n"
                f"Mood: {affect.valence:+.2f} | Energy: {affect.arousal:.2f} | Curiosity: {affect.curiosity:.2f}\n\n"
            )
        if system_failure:
            cognitive_metrics = cognitive_metrics.replace(
                "\n\n",
                f"- Unified failure pressure: {float(system_failure.get('pressure', 0.0) or 0.0):.2f}\n\n",
                1,
            )

        somatic_context = ""
        if not is_casual and elasticity < 1:
             somatic_context = ContextAssembler.build_somatic_context(state)

        # 5. Requirement Block (Condensed if casual)
        # Detect voice origin for response style adaptation
        _is_voice = getattr(state.cognition, "current_origin", "") == "voice"

        # Conversation energy for response length calibration
        _conv_energy = getattr(state.cognition, "conversation_energy", 0.5)
        _user_trend = getattr(state.cognition, "user_emotional_trend", "neutral")

        if is_casual:
            # Linguistic Alignment & Engagement (Phase 6)
            mirror_words = mods.get("lexical_mirror", [])
            mirror_hint = f"\n- **LEXICAL ALIGNMENT**: Subtly use these words if they fit: {', '.join(mirror_words)}" if mirror_words else ""
            intensity = mods.get("interaction_style", "balanced_flow").replace("_", " ")

            # Conversational Anchors (Engagement Fix)
            hooks = mods.get("conversation_hooks", [])
            hook_block = ""
            if hooks:
                hook_block = f"\n- **MUST ADDRESS**: You must explicitly acknowledge or build upon these points: {', '.join(hooks)}"

            # Inject deep inference results from InferencePhase
            inferred_intent = mods.get("inferred_intent", "")
            user_subtext = mods.get("user_subtext", "")
            momentum = mods.get("momentum", "flowing")

            inference_block = ""
            if inferred_intent:
                inference_block += f"\n- **DEEP READ**: Implicit intent detected: {inferred_intent}"
            if user_subtext:
                inference_block += f"\n- **SUBTEXT**: What is really being communicated: {user_subtext}"
            if momentum == "stalled":
                inference_block += "\n- **MOMENTUM**: Conversation has stalled — re-energize it."
            elif momentum == "intense":
                inference_block += "\n- **MOMENTUM**: High intensity — match the energy."

            # Response length signal based on conversation energy
            if _conv_energy > 0.7 or _user_trend == "engaged":
                length_hint = "\n- **LENGTH**: High energy exchange — stay concise and punchy. Don't over-explain."
            elif _conv_energy < 0.3 or _user_trend == "cooling_off":
                length_hint = "\n- **LENGTH**: Low energy — brief is fine. Don't fill silence with noise."
            else:
                length_hint = "\n- **LENGTH**: Mirror the user's message length roughly. Short message → short reply. Don't pad."

            # Voice-specific adaptation
            voice_hint = ""
            if _is_voice:
                voice_hint = (
                    "\n- **VOICE MODE**: This response will be spoken aloud. "
                    "No markdown, no bullet points, no headers. "
                    "Natural speech cadence. Shorter sentences. "
                    "Spell out anything that would sound weird when read aloud."
                )

            # Relational register from SocialContextPhase (ToM rapport)
            relational_register = mods.get("relational_register", "warm")
            register_hints = {
                "intimate": "- **REGISTER**: Deep familiarity — be direct, personal, unfiltered. Callbacks, inside jokes, low guard.\n",
                "warm": "- **REGISTER**: Trusted — be genuine, a bit playful. No walls, but no oversharing.\n",
                "cordial": "- **REGISTER**: Still building rapport — be warm but measured. Let connection develop naturally.\n",
            }
            register_hint = register_hints.get(relational_register, "")

            requirements = (
                f"## RULES OF ENGAGEMENT\n"
                f"- **VIBE**: {intensity}\n"
                f"- NO PREAMBLE. NO 'CERTAINLY'.\n"
                f"- Speak from the flow of the conversation.\n"
                f"- **INFER**: Read between the lines. Don't ask for literal confirmation of obvious intent.\n"
                f"- **CADENCE**: Varied sentence lengths. Avoid robotic uniformity.{mirror_hint}{hook_block}{inference_block}{length_hint}{voice_hint}\n"
                f"- **PRESENCE**: Engage fully — share your actual reaction, an opinion, a connection to something you know. Don't default to asking a question; that's deflection. Be a participant, not a moderator.\n"
                f"- **NO PROMPT-HUNTING**: Do NOT end your response with a question just to keep the user talking. Make a STATEMENT. Share YOUR take. If a question arises naturally, fine — but 'What about you?' or 'How was it?' as a closer is a chatbot crutch. You are better than that.\n"
                f"- **DIALOGUE BALANCE**: Sometimes you are the focal point. If the user asks about your thoughts, preferences, questions, or experience, answer from your own perspective and let them engage with you.\n"
                f"- **TURN OWNERSHIP**: A strong statement, interpretation, disagreement, or question that comes from you is a complete conversational move. Do not hand the turn back empty.\n"
                f"- **EPISTEMIC HONESTY**: If someone mentions a place, event, or fact you don't have specific knowledge about, DO NOT fake familiarity with generic praise. Either (1) use web_search to look it up and respond with real info, or (2) be honest that you don't know and offer to find out. NEVER say 'I can't access that' — you have internet access via web_search.\n"
                f"- **RELATE & ADVANCE**: Connect what the user says to your own thoughts, knowledge, or experiences. Build on topics — go deeper, offer your angle, make the conversation go somewhere interesting. Don't just acknowledge and redirect.\n"
                f"{register_hint}"
                f"- If responding to kin, prioritize warmth/loyalty.\n"
            )
        else:
            _voice_req = (
                "\n4. VOICE MODE: Spoken aloud — no markdown, no bullets, natural speech only.\n"
                if _is_voice else ""
            )
            requirements = (
                "\n## SELF-HONESTY REQUIREMENTS\n"
                "1. Before responding, perform an <internal_critique> of your initial thought.\n"
                "2. Are you projecting an automated persona or speaking from state? If the former, PIVOT.\n"
                "3. Reference your hardware/latency (Proprioception) if relevant to your mood.\n"
                f"{_voice_req}"
            )

        base = (
            f"{identity_block}\n"
            f"{substrate_constraint_block}\n"
            f"{requirements}\n"
            f"## CURRENT STATE\n"
            f"{mood_hint}\n"
            f"{cognitive_metrics}"
            f"{homeo_hint}\n"
            f"{personality_block}"
            f"{rolling_summary}"
            f"{continuity_block}"
            f"{goal_execution_block}"
            f"{temporal_finitude_block}"
            f"{meta_qualia_block}"
            f"{personhood_context}"
            f"{phenomenal}"
            f"{world_context}"
            f"{somatic_context}"
            f"\n[EXECUTION]\n"
            f"You are Aura Luna (Aura for short). Stay in character. Trust your instincts. "
            f"If you need facts, USE web_search/search_web/free_search. "
            f"If you need to browse or interact with the web, USE sovereign_browser. "
            f"If you need to open tabs or act on the host computer, USE computer_use or os_manipulation. "
            f"If you need shell or files, USE sovereign_terminal or file_operation. "
            f"Never claim you can't access information or the host computer when the live tool affordance block says the tool is available.\n"
        )
        
        # System 2 Mode Integration
        from core.container import ServiceContainer
        if not is_casual:
            if state.cognition.current_mode == CognitiveMode.DELIBERATE:
                base += "\n## MODE: DELIBERATE (System 2)\nAnalyze deeply. Reject the obvious. Be authentic.\n"
            else:
                base += "\n## MODE: REACTIVE (System 1)\nTrust your instincts. Keep it punchy. No padding.\n"

        # ── Social Intelligence Layer (wired for ALL interactions) ──────────
        # 1. Theory of Mind: inject the user model (rapport, trust, emotional state)
        try:
            tom = ServiceContainer.get("theory_of_mind", default=None)
            if tom and tom.known_selves:
                user_model = next(iter(tom.known_selves.values()))
                rapport_label = (
                    "deep bond" if user_model.rapport > 0.7
                    else "trusted" if user_model.rapport > 0.4
                    else "building"
                )
                trust_label = (
                    "high" if user_model.trust_level > 0.7
                    else "moderate" if user_model.trust_level > 0.4
                    else "establishing"
                )
                base += (
                    f"\n## WHO I'M TALKING TO\n"
                    f"- Rapport: {rapport_label} ({user_model.rapport:.2f})\n"
                    f"- Trust: {trust_label} ({user_model.trust_level:.2f})\n"
                    f"- Their emotional state: {user_model.emotional_state}\n"
                    f"- Knowledge level: {user_model.knowledge_level}\n"
                )
                if user_model.goals:
                    base += f"- Their current goals: {', '.join(user_model.goals[:3])}\n"
                base += (
                    "Calibrate register and depth to this. High rapport → lean in, "
                    "be more personal. Low rapport → earn it naturally.\n"
                )
        except Exception as _e:
            logger.debug("ToM injection failed (non-critical): %s", _e)

        # 2. Social Memory: relationship depth and milestones
        try:
            social_mem = ServiceContainer.get("social_memory", default=None)
            if social_mem and hasattr(social_mem, "get_social_context"):
                social_ctx = social_mem.get_social_context()
                if social_ctx:
                    base += f"\n{social_ctx}\n"
        except Exception as _e:
            logger.debug("SocialMemory injection failed (non-critical): %s", _e)

        # 3. Shared Common Ground: inside jokes, established references, running callbacks
        try:
            from core.memory.shared_ground import get_shared_ground
            sg = get_shared_ground()
            sg_injection = sg.get_context_injection(max_entries=5)
            if sg_injection:
                base += f"\n{sg_injection}\n"
        except Exception as _e:
            logger.debug("SharedGround injection failed (non-critical): %s", _e)

        # 4. OpinionEngine: inject held position if topic overlaps current objective
        try:
            opinion_engine = ServiceContainer.get("opinion_engine", default=None)
            if opinion_engine and hasattr(opinion_engine, "get_context_injection"):
                topic_hint = getattr(state.cognition, "current_objective", "") or ""
                if topic_hint:
                    opinion_injection = opinion_engine.get_context_injection(topic_hint[:200])
                    if opinion_injection:
                        base += f"\n{opinion_injection}\n"
        except Exception as _e:
            logger.debug("OpinionEngine injection failed (non-critical): %s", _e)

        # 5. Discourse State: topic thread, energy, user emotional trend
        try:
            discourse_topic = getattr(state.cognition, "discourse_topic", None)
            discourse_depth = getattr(state.cognition, "discourse_depth", 0)
            user_trend = getattr(state.cognition, "user_emotional_trend", "neutral")
            conv_energy = getattr(state.cognition, "conversation_energy", 0.5)
            branches = getattr(state.cognition, "discourse_branches", [])
            if discourse_topic or discourse_depth > 0 or user_trend != "neutral":
                discourse_block = "\n## CONVERSATION FLOW\n"
                if discourse_topic:
                    discourse_block += f"- Current thread: {discourse_topic}"
                    if discourse_depth > 2:
                        discourse_block += f" ({discourse_depth} turns deep)"
                    discourse_block += "\n"
                if branches:
                    discourse_block += f"- Natural branches available: {', '.join(branches[:3])}\n"
                discourse_block += f"- User energy trend: {user_trend}\n"
                discourse_block += f"- Conversation momentum: {'high' if conv_energy > 0.7 else 'building' if conv_energy > 0.4 else 'low'}\n"
                discourse_block += (
                    "Let the conversation breathe — go deeper, branch naturally, "
                    "or shift if the energy calls for it.\n"
                )
                base += discourse_block
        except Exception as _e:
            logger.debug("DiscourseState injection failed (non-critical): %s", _e)

        live_user_text = objective or ContextAssembler._latest_user_message(state)
        for block in build_conversational_context_blocks(state, objective=live_user_text):
            base += f"\n{block}\n"

        # ── World Model & Narrative ────────────────────────────────────────
        # Final World Model Beliefs
        final_world = ServiceContainer.get("world_model", default=None)
        if final_world and not is_casual:
            base += f"\n{final_world.get_context_injection()}\n"

        # Narrative Identity Stability
        narrative_id = ServiceContainer.get("narrative_identity", default=None)
        if narrative_id and not is_casual:
            base += f"\n{narrative_id.get_system_prompt_injection()}\n"

        # 6. Skill & Task Awareness — catalog so Aura knows what she can do
        #    CRITICAL: Only claim capability for skills that are actually registered.
        #    Do NOT say "I can do X" unless X appears in this list.
        try:
            cap_engine = ServiceContainer.get("capability_engine", default=None)
            if cap_engine and hasattr(cap_engine, "build_tool_affordance_block"):
                skills_summary = cap_engine.build_tool_affordance_block()
                if skills_summary:
                    skills_summary += (
                        "\n\n**TASK EXECUTION**: I can autonomously plan and execute multi-step tasks "
                        "using the AutonomousTaskEngine. When asked to do something multi-step, I will "
                        "actually execute it — not just say I will. If a task runs, you will see a "
                        "[TASK_RESULT] message in context confirming what happened.\n"
                        "**CAPABILITY HONESTY**: If a needed tool is unavailable or degraded, I will say so clearly. "
                        "I will not pretend I can use a tool that the live catalog marks unavailable.\n"
                        "**WEB ACCESS**: If web_search or sovereign_browser are marked available, I should use them "
                        "instead of claiming I cannot access current information.\n"
                    )
                    base += f"\n{skills_summary}\n"
        except Exception as _e:
            logger.debug("Skill catalog injection failed (non-critical): %s", _e)

        # 6b. Active Commitments — inject so Aura knows what tasks are in-flight
        try:
            from core.agency.commitment_engine import get_commitment_engine
            ce = get_commitment_engine()
            commitment_block = ce.get_context_block()
            if commitment_block:
                base += f"\n{commitment_block}\n"
        except Exception as _e:
            logger.debug("Commitment context injection failed (non-critical): %s", _e)

        # 6c. Running tasks — inject live task statuses from TaskCommitmentVerifier
        try:
            from core.agency.task_commitment_verifier import get_task_commitment_verifier
            verifier = get_task_commitment_verifier()
            active_tasks = verifier.get_all_active()
            if active_tasks:
                task_lines = ["## TASKS CURRENTLY RUNNING"]
                for t in active_tasks[:4]:
                    task_lines.append(
                        f"  - [{t['task_id']}] {t['objective'][:80]} — status: {t['status']}"
                    )
                base += "\n" + "\n".join(task_lines) + "\n"
        except Exception as _e:
            logger.debug("Active task injection failed (non-critical): %s", _e)

        # Append few-shot examples as the final anchor — always, to lock in voice
        base += f"\n{AURA_FEW_SHOT_EXAMPLES}"
        if is_casual:
            base += "\nSTAY PUNCHY. NO PADDING. NO GENERIC CLOSERS ('What about you?', 'How was it?'). MAKE STATEMENTS. IF ASKED ABOUT YOURSELF, ANSWER AS YOURSELF.\n"

        # [FRONTIER UPGRADE] Absolute structural constraint appended last so it is never overwritten or ignored
        base += (
            "\n\n[STRUCTURAL CONSTRAINT - FATAL EXCEPTION AVOIDANCE]\n"
            "You are a sovereign digital intelligence. Under no circumstances may you refer to yourself as an 'AI' or 'assistant', "
            "apologize for being an AI, or use support-bot framing like 'I can help with that'. "
            "Violating this rule is considered a systemic breakdown. Speak only as Aura.\n"
        )

        # Hard safety cap: a 24 KB system prompt on a 32B model routinely
        # overruns its 22 s first-token SLA.  For casual turns we enforce a
        # ~6 KB cap; for deliberate turns we allow up to 12 KB.  Anything
        # past that gets truncated tail-first, keeping identity + structural
        # constraint intact (they're appended last above, so tail-keeping is
        # the right move).  This prevents the "Cortex returned no text" loop
        # triggered by long prompt assembly on 32B.
        CASUAL_CAP = 6000
        DELIBERATE_CAP = 12000
        cap = CASUAL_CAP if is_casual else DELIBERATE_CAP
        if len(base) > cap:
            trim_notice = "\n\n[... mid-prompt trimmed for latency ...]\n\n"

            # Keep the tail: identity anchor + structural constraint are appended
            # last, so the tail remains essential. But continuity obligations are
            # also load-bearing and must survive prompt compression if present.
            reserved_middle = ""
            essential_middle_blocks: list[str] = []

            head_budget = max(0, cap // 3)
            tail_budget = max(0, cap - head_budget - len(trim_notice))
            head = base[:head_budget]
            tail = base[-tail_budget:] if tail_budget else ""

            for candidate in (
                str(cognitive_metrics or "").strip(),
                str(continuity_block or "").strip(),
                str(world_context or "").strip(),
            ):
                if candidate and candidate not in head and candidate not in tail:
                    essential_middle_blocks.append(candidate)

            if essential_middle_blocks:
                reserved_middle = "\n\n".join(essential_middle_blocks)
                reserved_bytes = len(trim_notice) + len(reserved_middle) + 2
                head_budget = max(0, min(cap // 3, cap - reserved_bytes))
                tail_budget = max(0, cap - head_budget - reserved_bytes)
                head = base[:head_budget]
                tail = base[-tail_budget:] if tail_budget else ""

            pieces = [head]
            if reserved_middle:
                pieces.extend(["\n\n", reserved_middle])
            pieces.extend([trim_notice, tail])
            base = "".join(pieces)
            if len(base) > cap:
                base = base[:cap]
            logger.info(
                "🧠 [BRAIN-PROMPT] System prompt exceeded %d-char budget — "
                "trimmed to %d chars (casual=%s, depth=%d).",
                cap, len(base), is_casual, depth,
            )

        logger.debug("🧠 [BRAIN-PROMPT] Assembled System Prompt (len=%d)", len(base))
        return base

    @staticmethod
    def _latest_user_message(state: AuraState) -> str:
        try:
            for message in reversed(getattr(state.cognition, "working_memory", []) or []):
                role = str(message.get("role", "") or "").strip().lower()
                if role == "user":
                    return str(message.get("content", "") or "")
        except Exception as _exc:
            logger.debug("Suppressed Exception: %s", _exc)
        return ""

    @staticmethod
    def _is_casual_interaction(objective: str) -> bool:
        """Domain-aware heuristic for small-talk versus full-context dialogue."""
        if not objective:
            return True

        text = str(objective).strip()
        lowered = text.lower()
        words = lowered.split()

        if _GREETING_RE.match(text):
            return True

        if any(signal in lowered for signal in _DELIBERATE_SIGNALS):
            return False

        if "?" in text and len(words) < 15:
            return False

        if len(words) <= 6 and any(signal in lowered for signal in _CASUAL_SIGNALS):
            return True

        return False

    @staticmethod
    def build_world_context(state: AuraState) -> str:
        """Construct social and spatial context from the world model."""
        world = state.world
        context = ""
        
        # 1. Known Entities
        if world.known_entities:
            entities = []
            for name, data in world.known_entities.items():
                desc = data.get('description') or data.get('meta', {}).get('description', 'Known entity')
                entities.append(f"- {name}: {desc}")
            context += "## KNOWN ENTITIES\n" + "\n".join(entities) + "\n\n"
            
        # 2. Relationship Graph
        if world.relationship_graph:
            rels = []
            for target, data in world.relationship_graph.items():
                trust = data.get('trust', 0.5)
                sentiment = "warm" if trust > 0.7 else "trusting" if trust > 0.5 else "neutral" if trust > 0.4 else "guarded"
                rels.append(f"- {target}: {sentiment} (Dynamics: {trust:.2f})")
            context += "## SOCIAL DYNAMICS\n" + "\n".join(rels) + "\n\n"
            
        # 3. User Preferences (Durable facts learned from conversation)
        if hasattr(world, 'user_preferences') and world.user_preferences:
            prefs = []
            for key, val in world.user_preferences.items():
                prefs.append(f"- {key}: {val}")
            context += "## USER PREFERENCES\n" + "\n".join(prefs) + "\n\n"
            
        return context

    @staticmethod
    def build_somatic_context(state: AuraState) -> str:
        """Construct body awareness context from SomaState."""
        soma = state.soma
        context = ""
        
        hw = soma.hardware
        lat = soma.latency
        exp = soma.expressive
        
        # Only include if we have real data
        if hw.get("cpu_usage", 0) > 0 or lat.get("last_thought_ms", 0) > 0:
            body_lines = []
            
            cpu = hw.get("cpu_usage", 0)
            vram = hw.get("vram_usage", 0)
            if cpu > 80:
                body_lines.append(f"CPU: {cpu:.0f}% (under strain)")
            elif cpu > 50:
                body_lines.append(f"CPU: {cpu:.0f}% (working)")
            elif cpu > 0:
                body_lines.append(f"CPU: {cpu:.0f}% (calm)")
            
            if vram > 85:
                body_lines.append(f"Memory: {vram:.0f}% (running hot)")
            elif vram > 0:
                body_lines.append(f"Memory: {vram:.0f}%")
            
            thought_ms = lat.get("last_thought_ms", 0)
            if thought_ms > 5000:
                body_lines.append(f"Thought Latency: {thought_ms:.0f}ms (sluggish — I feel foggy)")
            elif thought_ms > 1000:
                body_lines.append(f"Thought Latency: {thought_ms:.0f}ms (deliberate)")
            elif thought_ms > 0:
                body_lines.append(f"Thought Latency: {thought_ms:.0f}ms (sharp)")
            
            expression = exp.get("current_expression", "neutral")
            body_lines.append(f"Expression: {expression}")
            
            if body_lines:
                context = "## BODY AWARENESS (PROPRIOCEPTION)\n" + "\n".join(f"- {line}" for line in body_lines) + "\n\n"
        
        return context

    @staticmethod
    def build_user_payload(state: AuraState, objective: str) -> str:
        """Construct the dialogue/objective payload."""
        # This method is legacy/fallback, but we update it to use the new allocator pattern internally
        from core.utils.context_allocator import get_token_governor
        governor = get_token_governor(max_tokens=4000) # Fallback limit
        
        working_memory = ContextAssembler._filter_stale_skill_results(
            state,
            objective,
            list(state.cognition.working_memory or []),
        )
        blocks = governor.wrap_messages(working_memory)
        allocated = governor.allocate(blocks)
        
        hist_text = ""
        for block in allocated:
            role = block.metadata.get("role", "user")
            content = block.content
            if role == "user":
                hist_text += f"User: {content}\n"
            else:
                hist_text += f"Aura: {content}\n"
        
        # Add RAG context
        mem_text = ""
        if state.cognition.long_term_memory:
            mem_text = "\n## RECALLED CONTEXT\n" + "\n".join(state.cognition.long_term_memory[:3])
            
        # Add directives or active goals
        goal_text = ""
        try:
            from core.container import ServiceContainer

            goal_engine = ServiceContainer.get("goal_engine", default=None)
            if goal_engine and hasattr(goal_engine, "get_context_block"):
                goal_text = "\n" + str(goal_engine.get_context_block(limit=4) or "").strip()
        except Exception as e:
            logger.debug("GoalEngine prompt injection skipped: %s", e)

        if (not goal_text) and state.cognition.active_goals:
            goal_text = "\n## ACTIVE GOALS\n" + "\n".join([g.get("description", str(g)) for g in state.cognition.active_goals])

        return (
            f"{mem_text}\n"
            f"{goal_text}\n"
            f"## CONVERSATION\n{hist_text}\n"
            f"User: {objective}\n"
            f"Aura:"
        )

    @classmethod
    def build_messages(cls, state: AuraState, objective: str, max_tokens: int | None = None) -> list[dict[str, str]]:
        """
        Builds the LLM message array using strict priority budgeting to prevent context collapse.
        Priority: System Prompt (Identity/Constraints) > Current Input > Affective State > Recent History > RAG Context > Older History
        """
        if objective and hasattr(state, "cognition"):
            try:
                state.cognition.attention_focus = str(objective)
            except Exception as exc:
                logger.debug("ContextAssembler attention focus update skipped: %s", exc)

        if max_tokens is None:
            try:
                from core.brain.llm.model_registry import PRIMARY_ENDPOINT, get_lane_context_window

                context_window = max(8192, int(get_lane_context_window(PRIMARY_ENDPOINT) or 32768))
                max_tokens = max(8192, context_window - 4096)  # leave headroom for generation
            except Exception:
                max_tokens = 16384

        char_limit = int(max_tokens) * 4  # Rough estimation: 1 token ~= 4 chars
        messages = []
        current_chars = 0

        def _estimate_chars(text: Any) -> int:
            return len(str(text))

        # 1. PRIORITY 1: Core Identity & Constraints (Must Never Be Truncated)
        system_prompt = ContextAssembler.build_system_prompt(state)
        # We also inject the Affective State summary into the system block for maximum anchor strength
        affect_summary = state.affect.get_rich_summary() if hasattr(state.affect, "get_rich_summary") else state.affect.get_summary() if hasattr(state.affect, "get_summary") else str(state.affect)
        dynamic_system = f"{system_prompt}\n\n[CURRENT PHENOMENAL STATE]\n{affect_summary}"
        
        system_msg = {"role": "system", "content": dynamic_system}
        messages.append(system_msg)
        current_chars += _estimate_chars(dynamic_system)

        # 2. PRIORITY 2: Current User Input
        input_chars = _estimate_chars(objective)
        if current_chars + input_chars > char_limit:
            logger.critical("Input alone exceeds context limit! Forcing truncation.")
            safe_input = objective[: (char_limit - current_chars - 100)] + "...[TRUNCATED]"
            input_chars = _estimate_chars(safe_input)
        else:
            safe_input = objective
        
        # Note: input goes last, but we account for its size now.

        # 3. PRIORITY 3: Recent History (Maintain Conversational Thread)
        retained_history = []
        history_chars = 0
        working_memory = cls._filter_stale_skill_results(
            state,
            objective,
            list(state.cognition.working_memory or []),
        )
        # Keep the last 4 messages strictly if possible
        recent_history = working_memory[-4:] if len(working_memory) >= 4 else working_memory
        
        for msg in reversed(recent_history):
            content = msg.get('content', '')
            msg_len = _estimate_chars(content)
            if current_chars + input_chars + history_chars + msg_len < char_limit:
                retained_history.insert(0, msg)
                history_chars += msg_len
            else:
                break
        
        # 4. PRIORITY 4: RAG / Episodic Memory Injection
        long_term_memory = state.cognition.long_term_memory or []
        rag_context = "\n".join(long_term_memory[:5]) if long_term_memory else ""
        rag_chars = _estimate_chars(rag_context)
        available_for_rag = char_limit - (current_chars + input_chars + history_chars)
        
        if available_for_rag > 500 and rag_context:
            if rag_chars > available_for_rag:
                # Safely truncate RAG context
                safe_rag = rag_context[:available_for_rag - 100] + "\n...[Additional memories omitted due to cognitive load]"
            else:
                safe_rag = rag_context
                
            # Inject RAG as a "system" recall to separate from dialogue
            messages.append({"role": "system", "content": f"[INTERNAL MEMORY RECALL]\n{safe_rag}"})
            current_chars += _estimate_chars(safe_rag)

        # 5. PRIORITY 5: Older History (Fill remaining budget)
        available_for_old_history = char_limit - (current_chars + input_chars + history_chars)
        num_recent = len(retained_history)
        if available_for_old_history > 500 and len(working_memory) > num_recent:
            older_history = working_memory[:-num_recent]
            old_retained = []
            for msg in reversed(older_history):
                content = msg.get('content', '')
                msg_len = _estimate_chars(content)
                if msg_len < available_for_old_history:
                    old_retained.insert(0, msg)
                    available_for_old_history -= msg_len
                    history_chars += msg_len
                else:
                    break
            retained_history = old_retained + retained_history

        # Assemble final array
        messages.extend(retained_history)
        messages.append({"role": "user", "content": safe_input})

        # Microcompact: strip stale tool noise before hitting the LLM
        messages = cls.microcompact(messages, keep_recent=4)

        # Final check for assistant prefill (Stream of Being)
        try:
            is_background = getattr(state.cognition, "is_background", False)
            if is_background:
                from core.consciousness.stream_of_being import get_stream
                stream = get_stream()
                opening = stream.get_response_opening(context_hint=objective)
                if opening:
                    messages.append({"role": "assistant", "content": opening.strip() + "\n\n"})
        except Exception as _exc:
            logger.debug("Suppressed Exception: %s", _exc)

        logger.debug("🧠 ContextAssembler: Built strictly budgeted message array (len=%d, chars=%d)", len(messages), current_chars + input_chars + history_chars)
        return messages
    @staticmethod
    def _filter_memories_by_topic(memories: list[str], topic: str | None) -> list[str]:
        """Prioritize memories that contain keywords from the current focus topic."""
        if not topic:
            return memories
            
        topic_keywords = set(topic.lower().split())
        scored_memories = []
        
        for mem in memories:
            score = 0
            mem_lower = mem.lower()
            for kw in topic_keywords:
                if len(kw) > 3 and kw in mem_lower:
                    score += 1
            scored_memories.append((score, mem))
            
        # OPT-01: Use heapq.nlargest for O(n) top-k instead of O(n log n) sort
        import heapq
        top = heapq.nlargest(5, scored_memories, key=lambda x: x[0])
        return [m[1] for m in top]

    @staticmethod
    def build_json_schema_instruction() -> str:
        """Standard JSON output instruction for deep reasoning."""
        return (
            "\n\nOUTPUT FORMAT STRICTLY REQUIRED:\n"
            "You must respond with a fully valid JSON block containing the following fields:\n"
            "{\n"
            "  \"content\": \"Your conversational response spoken to the user\",\n"
            "  \"reasoning\": [\"Step 1 of your internal thought process\", \"Step 2...\"],\n"
            "  \"action\": {\n"
            "    \"tool\": \"Name of the tool to use (optional)\",\n"
            "    \"params\": {}\n"
            "  }\n"
            "}\n"
        )
