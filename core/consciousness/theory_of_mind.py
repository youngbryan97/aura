"""core/brain/cognition/theory_of_mind.py
Advanced Theory of Mind (ToM) system for Aura.
Consolidated from duplicate modules.
"""
import json
import logging
import time
import asyncio
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from core.runtime import service_access

logger = logging.getLogger("Aura.ToM")

class SelfType(Enum):
    HUMAN = "human"
    AI = "ai"
    ANIMAL = "animal"
    COLLECTIVE = "collective"
    UNKNOWN = "unknown"

@dataclass
class AgentModel:
    """Model of another agent (user, system, etc.)"""
    identifier: str
    self_type: SelfType = SelfType.HUMAN
    beliefs: Dict[str, Any] = field(default_factory=dict)
    goals: List[str] = field(default_factory=list)
    preferences: Dict[str, Any] = field(default_factory=dict)
    knowledge_level: str = "intermediate"
    emotional_state: str = "neutral"
    interaction_history: List[Dict[str, Any]] = field(default_factory=list)
    trust_level: float = 0.5
    rapport: float = 0.5
    last_updated: float = field(default_factory=time.time)

    def to_dict(self):
        data = asdict(self)
        data['self_type'] = self.self_type.value
        return data

class TheoryOfMindEngine:
    """Complete Theory of Mind system with LLM-backed social reasoning.
    """

    def __init__(self, cognitive_engine=None):
        self.brain = cognitive_engine
        self.known_selves: Dict[str, AgentModel] = {}
        self._data_path = self._resolve_data_path()
        self._load()
        logger.info("TheoryOfMindEngine initialized.")

    def _resolve_data_path(self):
        try:
            from pathlib import Path
            from core.config import config
            p = config.paths.data_dir / "memory" / "theory_of_mind.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            return p
        except Exception:
            from pathlib import Path
            p = Path.home() / ".aura" / "data" / "memory" / "theory_of_mind.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            return p

    def _load(self):
        import json
        try:
            if self._data_path.exists():
                with open(self._data_path, "r") as f:
                    raw = json.load(f)
                for uid, d in raw.items():
                    try:
                        d["self_type"] = SelfType(d.get("self_type", "human"))
                        # interaction_history can grow large — cap on load
                        d["interaction_history"] = d.get("interaction_history", [])[-20:]
                        self.known_selves[uid] = AgentModel(**{k: v for k, v in d.items() if k in AgentModel.__dataclass_fields__})
                    except Exception as _exc:
                        logger.debug("Suppressed Exception: %s", _exc)
                logger.debug("ToM: loaded %d user models", len(self.known_selves))
        except Exception as e:
            logger.debug("ToM: load failed (%s), starting fresh", e)

    def save(self):
        import json, os
        try:
            data = {}
            for uid, model in self.known_selves.items():
                d = model.to_dict()
                d["interaction_history"] = d["interaction_history"][-20:]  # Keep it lean
                data[uid] = d
            tmp = str(self._data_path) + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, self._data_path)
        except Exception as e:
            logger.debug("ToM: save failed: %s", e)

    def get_health(self) -> Dict[str, Any]:
        """Social health for HUD."""
        depth_val: float = 0.5
        if not self.known_selves:
            return {"depth": 0.0, "status": "offline"} # Return early for empty known_selves
        depth_val = float(sum(s.rapport for s in self.known_selves.values())) / len(self.known_selves)
        return {"depth": round(float(depth_val), 2), "status": "online"}

    def _get_brain(self):
        if self.brain:
            return self.brain
        try:
            from core.container import ServiceContainer
            return ServiceContainer.get("cognitive_integration", default=ServiceContainer.get("cognitive_engine", default=None))
        except Exception as exc:
            logger.debug("Failed to resolve brain from ServiceContainer: %s", exc)
            return None

    async def understand_user(self, user_id: str, message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Update and return the model of a specific user."""
        if user_id not in self.known_selves:
            self.known_selves[user_id] = AgentModel(identifier=user_id)

        model = self.known_selves[user_id]
        model.interaction_history.append({"message": message, "timestamp": time.time()})
        model.last_updated = time.time()

        # Determine deep or fast analysis
        if len(model.interaction_history) % 5 == 0:
            result = await self._deep_analyze(user_id, message, context)
        else:
            result = self._fast_heuristic_update(user_id, message)
        # Persist after every 5th update to avoid excessive I/O
        if len(model.interaction_history) % 5 == 0:
            self.save()
        return result

    async def infer_intent(self, message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Legacy-compatible intent inference shim."""
        user_id = context.get("user_id", "default_user") if context else "default_user"
        result = await self.understand_user(user_id, message, context)
        # Extract intent data in the format expected by context builder
        intent_data = result.get("intent", {})
        if intent_data and isinstance(intent_data, dict):
             # Ensure 'pragmatic' key exists or fallback
             intent_data["pragmatic"] = intent_data.get("intent", "standard")
        return intent_data

    def _fast_heuristic_update(self, user_id: str, message: str) -> Dict[str, Any]:
        """Apply keyword heuristics for rapid updates without LLM calls."""
        model = self.known_selves[user_id]
        msg = message.lower()
        rapport_delta = 0.0

        # Scale rapport changes by current conversation energy — high-energy exchanges
        # carry more weight for relationship development than idle one-liners.
        try:
            _state = service_access.resolve_state_repository(default=None)
            _live = getattr(_state, "_current", None) if _state else None
            conv_energy = getattr(getattr(_live, "cognition", None), "conversation_energy", 0.5) if _live else 0.5
        except Exception:
            conv_energy = 0.5
        energy_scale = 0.5 + conv_energy  # range [0.5, 1.5]

        if any(w in msg for w in ["thank", "great", "love", "appreciate", "good", "exactly", "yes", "perfect"]):
            delta = 0.05 * energy_scale
            model.trust_level = min(1.0, model.trust_level + delta)
            model.rapport = min(1.0, model.rapport + delta)
            rapport_delta = delta
        elif any(w in msg for w in ["angry", "wrong", "bad", "hate", "rude"]):
            delta = 0.05 * energy_scale
            model.trust_level = max(0.0, model.trust_level - delta)
            model.rapport = max(0.0, model.rapport - delta)
            rapport_delta = -delta

        # --- Question pattern detection ---
        question_words = ["how", "why", "what", "when", "where", "who", "which", "can you", "could you"]
        if any(msg.strip().startswith(w) for w in question_words) or msg.strip().endswith("?"):
            # Record the question as a current goal
            question_summary = message.strip()[:80]
            if question_summary not in model.goals:
                model.goals.append(question_summary)
                # Keep goals list bounded
                if len(model.goals) > 10:
                    model.goals = model.goals[-10:]

        # --- Technical term detection ---
        tech_indicators = [
            "api", "async", "docker", "kubernetes", "tensor", "gradient",
            "database", "sql", "regex", "lambda", "deploy", "pipeline",
            "neural", "algorithm", "recursion", "mutex", "kernel", "ssh",
            "endpoint", "schema", "refactor", "microservice", "inference",
        ]
        tech_count = sum(1 for term in tech_indicators if term in msg)
        if tech_count >= 2:
            model.knowledge_level = "advanced"
        elif tech_count == 1 and model.knowledge_level == "beginner":
            model.knowledge_level = "intermediate"

        # --- Short message detection (possible frustration/terseness) ---
        if len(message.strip()) < 20 and message.strip():
            # Short messages without positive sentiment hint at terseness
            if rapport_delta <= 0:
                model.emotional_state = "terse"

        # --- Long detailed message detection ---
        if len(message.strip()) > 200:
            model.emotional_state = "engaged"
            if model.knowledge_level == "beginner":
                model.knowledge_level = "intermediate"

        # Mirror rapport increases to SocialMemory depth
        if rapport_delta > 0:
            try:
                social_mem = service_access.optional_service("social_memory", default=None)
                if social_mem and hasattr(social_mem, "relationship_depth"):
                    social_mem.relationship_depth = min(1.0, social_mem.relationship_depth + rapport_delta * 0.5)
            except Exception as _exc:
                logger.debug("Suppressed Exception: %s", _exc)

        return {
            "user_model": model.to_dict(),
            "intent": {"intent": message, "sentiment": "neutral"},
            "emotional_state": model.emotional_state,
            "knowledge_level": model.knowledge_level
        }

    async def _deep_analyze(self, user_id: str, message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Use LLM for deep social reasoning."""
        model = self.known_selves[user_id]
        brain = self._get_brain()
        if not brain:
            return self._fast_heuristic_update(user_id, message)

        prompt = f"""Analyze user intent and state.
User Message: {message}
Recent History: {[str(m.get('message', '')) for m in list(model.interaction_history)[-3:]]}
Return JSON: {{"intent": "...", "sentiment": "...", "emotional_state": "...", "knowledge_level": "..."}}"""

        try:
            # Fully async call to cognitive engine
            thought = await brain.think(
                objective=prompt,
                context={"model": model.to_dict(), "global_context": context},
                mode="FAST" # Use fast model for social metadata
            )

            from core.utils.json_utils import extract_json
            data = extract_json(thought.content)
            if data:
                model.emotional_state = data.get("emotional_state", model.emotional_state)
                model.knowledge_level = data.get("knowledge_level", model.knowledge_level)
                return {
                    "user_model": model.to_dict(),
                    "intent": data,
                    "emotional_state": model.emotional_state,
                    "knowledge_level": model.knowledge_level
                }
        except Exception as e:
            logger.debug("Deep ToM analysis failed: %s", e)

        return self._fast_heuristic_update(user_id, message)

    async def predict_reaction(self, user_id: str, my_action: Dict[str, Any]) -> Dict[str, Any]:
        """Predict reaction to an action using LLM."""
        model = self.known_selves.get(user_id) or AgentModel(identifier=user_id)
        brain = self._get_brain()
        if not brain:
            return {"prediction": "Unknown (Brain Offline)"}

        thought = await brain.think(
            objective=f"Predict how {user_id} will react if I take this action: {my_action}",
            context={"user_model": model.to_dict()},
            mode="FAST"
        )
        return {"prediction": thought.content, "confidence": thought.confidence}

    async def will_this_help_user(self, user_id: str, proposed_response: str) -> Tuple[bool, str]:
        """Social outcome simulation."""
        if user_id not in self.known_selves:
            return True, "No user model, assuming helpful."

        model = self.known_selves[user_id]
        if model.emotional_state == "frustrated" and len(proposed_response) > 500:
             return False, "User is frustrated; response is likely too verbose."

        for goal in model.goals:
             if goal.lower() in proposed_response.lower():
                  return True, f"Response addresses goal: {goal}"

        return True, "Response aligned."

    # ------------------------------------------------------------------
    # New capabilities — context block, response guidance, post-response
    # ------------------------------------------------------------------

    def get_context_block(self, user_id: str = "default_user") -> str:
        """Returns concise user model summary for inference context (max 200 chars)."""
        model = self.known_selves.get(user_id)
        if not model:
            return "ToM: No user model yet"

        # Compute recent interaction trend
        recent = model.interaction_history[-5:]
        trend = "new"
        if len(recent) >= 3:
            recent_times = [r.get("timestamp", 0) for r in recent if r.get("timestamp")]
            if len(recent_times) >= 2:
                avg_gap = (recent_times[-1] - recent_times[0]) / max(len(recent_times) - 1, 1)
                if avg_gap < 30:
                    trend = "rapid"
                elif avg_gap < 120:
                    trend = "steady"
                else:
                    trend = "sporadic"

        block = (
            f"User({user_id[:15]}): "
            f"know={model.knowledge_level}, "
            f"mood={model.emotional_state}, "
            f"rapport={model.rapport:.2f}, "
            f"trust={model.trust_level:.2f}, "
            f"trend={trend}"
        )
        return block[:200]

    def get_response_guidance(self, user_id: str = "default_user") -> Dict[str, Any]:
        """Returns actionable guidance for shaping inference responses.

        Derived from the user model state — complexity preference, tone, length,
        topics to avoid and topics of interest.
        """
        model = self.known_selves.get(user_id)
        if not model:
            return {
                "preferred_complexity": "moderate",
                "tone_hint": "friendly",
                "max_length_hint": 500,
                "topics_to_avoid": [],
                "topics_of_interest": [],
            }

        # Preferred complexity from knowledge level
        complexity_map = {
            "beginner": "simple",
            "intermediate": "moderate",
            "advanced": "detailed",
        }
        preferred = complexity_map.get(model.knowledge_level, "moderate")

        # Tone hint from emotional state and rapport
        if model.emotional_state in ("frustrated", "terse"):
            tone = "concise and empathetic"
        elif model.rapport > 0.75:
            tone = "warm and familiar"
        elif model.rapport < 0.3:
            tone = "polite and professional"
        else:
            tone = "friendly"

        # Length hint — frustrated/terse users get shorter responses
        if model.emotional_state in ("frustrated", "terse"):
            max_len = 200
        elif preferred == "detailed":
            max_len = 800
        elif preferred == "simple":
            max_len = 300
        else:
            max_len = 500

        # Topics of interest from recent goals
        interests = [g[:50] for g in model.goals[-5:]] if model.goals else []

        # Topics to avoid — if user expressed negative sentiment about something
        avoid: List[str] = []
        for pref_key, pref_val in model.preferences.items():
            if isinstance(pref_val, str) and "dislike" in pref_val.lower():
                avoid.append(pref_key)

        return {
            "preferred_complexity": preferred,
            "tone_hint": tone,
            "max_length_hint": max_len,
            "topics_to_avoid": avoid[:5],
            "topics_of_interest": interests,
        }

    def update_from_response(self, user_id: str, response_text: str, user_reaction: str = ""):
        """Post-response feedback loop — update trust/rapport from user reaction.

        *user_reaction* is free-form text from the user's next message.  We infer
        whether the previous response was well-received based on keyword signals.
        """
        if user_id not in self.known_selves:
            return
        model = self.known_selves[user_id]

        if not user_reaction:
            return

        reaction_lower = user_reaction.lower()

        positive_signals = ["thanks", "perfect", "great", "exactly", "helpful", "awesome", "nice", "yes", "correct"]
        negative_signals = ["no", "wrong", "not what", "bad", "useless", "stop", "too long", "confused"]

        pos_hits = sum(1 for s in positive_signals if s in reaction_lower)
        neg_hits = sum(1 for s in negative_signals if s in reaction_lower)

        if pos_hits > neg_hits:
            delta = min(0.1, 0.03 * pos_hits)
            model.trust_level = min(1.0, model.trust_level + delta)
            model.rapport = min(1.0, model.rapport + delta)
            if model.emotional_state in ("terse", "frustrated"):
                model.emotional_state = "neutral"
            logger.debug("ToM: positive reaction from %s, trust += %.3f", user_id, delta)
        elif neg_hits > pos_hits:
            delta = min(0.1, 0.03 * neg_hits)
            model.trust_level = max(0.0, model.trust_level - delta)
            model.rapport = max(0.0, model.rapport - delta)
            model.emotional_state = "frustrated"
            logger.debug("ToM: negative reaction from %s, trust -= %.3f", user_id, delta)

# Global Singletons for compatibility
_engine_instance = None

def get_theory_of_mind(brain=None):
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = TheoryOfMindEngine(brain)
    return _engine_instance
