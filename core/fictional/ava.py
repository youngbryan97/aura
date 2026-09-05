"""Ava — SocialModelingEngine.

A per-person model built from keyword cues and message lengths. Every
number here is a heuristic reading and says so wherever it is reported.
"""

from __future__ import annotations

import logging
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from core.cognition.state_modifiers import set_modifiers
from core.security.structural_redaction import redact_text

from core.fictional.common import (
    WORD_TOKEN_RE,
    engine_state_path,
    record_fictional_degradation,
    save_engine_state,
)

logger = logging.getLogger("Aura.FictionalSynthesis")


# ═══════════════════════════════════════════════════════════════════════════════
# ENGINE 4: AVA — SocialModelingEngine
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class UserModel:
    communication_style: str = "unknown"
    humor_tolerance: float = 0.5
    directness_preference: float = 0.5
    emotional_openness: float = 0.3
    trust_toward_aura: float = 0.3
    formality_score: float = 0.5    # 0.0 (casual/slang) to 1.0 (academic/stiff)
    social_tension: float = 0.0     # 0.0 (chill) to 1.0 (conflict/hostile)
    conversational_rhythm: float = 10.0 # Average words per message
    reciprocity_score: float = 0.5    # How much user matches Aura's length
    preferred_vocabulary: list[str] = field(default_factory=list)
    personal_disclosures: list[str] = field(default_factory=list)
    total_interactions: int = 0


class SocialModelingEngine:
    """
    Derived from: Ava (Ex Machina)
    """

    #: Observations needed before a dimension is worth reporting to the
    #: model at all. Below this the number is an artifact of two or three
    #: messages, and presenting it as a belief about a person is the
    #: fabrication CP126 ``9f828005`` names.
    MIN_INTERACTIONS_TO_REPORT = 10
    MODEL_SCHEMA = "aura.fictional.social_model.v2"

    def __init__(self, persist_path: str | None = None, *, user_id: str | None = None):
        self.user_id = self._resolve_user_id(user_id)
        # Partitioned by person. One global file meant a second speaker
        # inherited the first one's vocabulary, disclosures and tension
        # score, and there was no boundary at which that leak stopped
        # (CP126 ``b261f498``).
        self.persist_path = engine_state_path(
            persist_path, "social", f"user_model.{self.user_id}.json"
        )
        self.model = UserModel()
        self._load_model()

    @staticmethod
    def _resolve_user_id(explicit: str | None) -> str:
        """Whose model this is. Falls back to a named default, never blank."""
        if explicit:
            return re.sub(r"[^A-Za-z0-9_.-]", "_", str(explicit))[:64] or "default"
        try:
            from core.container import ServiceContainer

            social = ServiceContainer.get("social_memory", default=None)
            candidate = getattr(social, "active_user_id", None) if social else None
            if candidate:
                return re.sub(r"[^A-Za-z0-9_.-]", "_", str(candidate))[:64] or "default"
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            pass
        return "default"

    def _load_model(self):
        if self.persist_path.exists():
            try:
                data = json.loads(self.persist_path.read_text())
                data.pop("schema", None)
                data.pop("user_id", None)
                data.pop("saved_at", None)
                # Ensure floats are actually floats
                for k in [
                    "humor_tolerance",
                    "directness_preference",
                    "emotional_openness",
                    "trust_toward_aura",
                    "formality_score",
                    "social_tension",
                    "conversational_rhythm",
                    "reciprocity_score",
                ]:
                    if k in data:
                        data[k] = float(data[k])
                self.model = UserModel(**data)
            except (json.JSONDecodeError, OSError, TypeError, ValueError) as e:
                record_fictional_degradation(
                    e,
                    action="kept default social model after persisted AVA model failed to load",
                )
                logger.debug("AVA: Failed to load user model: %s", e)

    def analyze_message(self, message: str, response: str = "", is_user: bool = False):
        self.model.total_interactions += 1
        
        # Heuristics for rich signal extraction
        msg_lower = message.lower()
        msg_len = len(message.split())
        
        # Cue matching is TOKEN-based, never substring. With `w in msg_lower`
        # the single-letter casual cues "u" and "r" matched almost every
        # English message ("your", "run", "sure"), and the conflict cue "no"
        # matched "now", "know", "nothing", "another", "cannot" — so ordinary
        # conversation drove formality down and pinned social_tension high,
        # and that tension was written into cognition modifiers and the
        # prompt. Word boundaries remove that whole false-positive class.
        msg_tokens = set(WORD_TOKEN_RE.findall(msg_lower))

        # 1. Formality Detection
        formal_cues = {"shall", "please", "kindly", "regarding", "furthermore", "sincerely"}
        casual_cues = {"hey", "yo", "sup", "lol", "lmao", "u", "r", "nvm"}

        if formal_cues & msg_tokens:
            self.model.formality_score = min(1.0, self.model.formality_score + 0.1)
        elif casual_cues & msg_tokens:
            self.model.formality_score = max(0.0, self.model.formality_score - 0.1)

        # 2. Social Tension Inference
        # Kept as cues (they carry real corrective/interpersonal charge as
        # WORDS): stop, wrong, hate, annoying. Dropped: "no" and "bad" — as
        # bare tokens they are ordinary conversation ("no rush", "not bad")
        # and as SUBSTRINGS they matched know/now/another/cannot, which is
        # what pinned tension high on nearly every message.
        conflict_cues = {"stop", "wrong", "hate", "annoying", "stupid", "useless"}
        conflict_phrases = ("shut up", "stop it", "knock it off", "leave me alone")
        if (conflict_cues & msg_tokens) or any(p in msg_lower for p in conflict_phrases):
            self.model.social_tension = min(1.0, self.model.social_tension + 0.15)
        else:
            # Tension decays slowly
            self.model.social_tension = max(0.0, self.model.social_tension - 0.05)

        # 3. Directness, Rhythm & Reciprocity
        self.model.conversational_rhythm = (self.model.conversational_rhythm * 0.9) + (msg_len * 0.1)
        
        # Calculate reciprocity (simple version: did user match last Aura response length?)
        if response:
            aura_len = len(response.split())
            diff = abs(aura_len - msg_len)
            match_score = max(0.0, 1.0 - (diff / max(aura_len, 1)))
            self.model.reciprocity_score = (self.model.reciprocity_score * 0.8) + (match_score * 0.2)

        if msg_len < 5: 
            self.model.directness_preference = min(1.0, self.model.directness_preference + 0.1)
        if any(w in msg_lower for w in ["feel", "emotion", "sad", "happy", "vulnerable"]): 
            self.model.emotional_openness = min(1.0, self.model.emotional_openness + 0.05)
            
        # 4. Lexical Extraction (Theory of Mind)
        stop_words = {"the", "a", "an", "is", "are", "and", "or", "but", "i", "you", "my", "your"}
        signal_words = [w for w in msg_lower.split() if w not in stop_words and len(w) > 4]
        for sw in signal_words:
            if sw not in self.model.preferred_vocabulary:
                self.model.preferred_vocabulary.append(sw)
        self.model.preferred_vocabulary = list(self.model.preferred_vocabulary)[-20:]
            
        # 4. Inject into State Modifiers (Digital Metabolism)
        try:
            from core.container import ServiceContainer

            ki = ServiceContainer.get("kernel_interface", default=None)
            if ki and ki.is_ready() and ki.kernel:
                state = ki.kernel.state
                if state and hasattr(state.cognition, "modifiers"):
                    # Owned writes. These went straight into the shared dict
                    # with no owner, no revision and no conflict detection,
                    # so nothing could say who set social_tension or on what
                    # (CP126 ``ad5752a2``). set_modifier stamps all three and
                    # refuses a key another owner holds.
                    set_modifiers(
                        state.cognition.modifiers,
                        {
                            "social_formality": self.model.formality_score,
                            "social_tension": self.model.social_tension,
                            "social_reciprocity": self.model.reciprocity_score,
                        },
                        owner="fictional_ai_synthesis.ava",
                        evidence=(
                            f"keyword heuristic over {self.model.total_interactions} "
                            f"messages from {self.user_id}"
                        ),
                    )
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as e:
            record_fictional_degradation(
                e,
                severity="warning",
                action="updated in-memory social model without kernel modifier injection",
            )
            logger.debug("AVA: Failed to inject social modifiers: %s", e)

        # Save every 5 turns
        if self.model.total_interactions % 5 == 0:
            save_engine_state(self.persist_path, self._persistable(), engine="ava")

    def _persistable(self) -> dict[str, Any]:
        """The durable record: redacted, attributed, schema-stamped.

        A person's own words were written verbatim into a file with no
        redaction pass, no owner and no schema. Vocabulary and disclosures
        are the two fields that carry content rather than statistics, so
        they go through the same credential redaction every other durable
        store in this runtime uses.
        """
        payload = asdict(self.model)
        for field_name in ("preferred_vocabulary", "personal_disclosures"):
            cleaned = []
            for item in payload.get(field_name) or []:
                text, _changed = redact_text(str(item))
                cleaned.append(text[:120])
            payload[field_name] = cleaned
        payload["schema"] = self.MODEL_SCHEMA
        payload["user_id"] = self.user_id
        payload["saved_at"] = time.time()
        return payload

    def get_context_injection(self) -> str:
        """Summarized social context for the model, with its provenance.

        These are heuristic readings off keyword cues and message lengths.
        They used to be handed to the model as bare numbers, which reads
        as a finding about a person rather than a guess from a word list
        (CP126 ``9f828005``). Two changes: the line says what it is, and
        it says nothing at all until there are enough observations for the
        numbers to be about anything.
        """
        if self.model.total_interactions < self.MIN_INTERACTIONS_TO_REPORT:
            return (
                "[SOCIAL_CONTEXT: not enough observations yet "
                f"({self.model.total_interactions}/{self.MIN_INTERACTIONS_TO_REPORT} "
                "messages); no inferences to offer]"
            )
        raw_vocab = self.model.preferred_vocabulary
        v_list = list(raw_vocab)
        n = len(v_list)
        last_five = [v_list[i] for i in range(max(0, n - 5), n)]
        vocab = ", ".join(last_five) if last_five else "None"
        return (
            "[SOCIAL_CONTEXT (heuristic, inferred from keyword cues and message "
            f"lengths over {self.model.total_interactions} messages — not stated "
            f"by the person): Formality={self.model.formality_score:.1f}, "
            f"Tension={self.model.social_tension:.1f}, "
            f"Directness={self.model.directness_preference:.1f}, "
            f"Reciprocity={self.model.reciprocity_score:.1f}, "
            f"UserVocab=[{vocab}]]"
        )

