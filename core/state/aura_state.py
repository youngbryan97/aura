from __future__ import annotations
import time
import uuid
import copy
import asyncio
import math
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Type, Final
from enum import Enum
import hashlib
from core.motivation.constants import clone_motivation_budget_defaults

MAX_WORKING_MEMORY: Final[int] = 40   # Tighter cap prevents context window overflow and personality drift
MAX_PERCEPTS: Final[int] = 200        # More sensory history
MAX_EVOLUTION_LOG: Final[int] = 1000  # Richer evolution tracking

_USER_INTENT_ORIGINS: Final[frozenset[str]] = frozenset({
    "user",
    "voice",
    "admin",
    "api",
    "gui",
    "ws",
    "http",
    "owner",
    "owner_session_cookie",
    "owner_sovereign",
})
_SPECULATIVE_AUTONOMY_PREFIXES: Final[tuple[str, ...]] = (
    "[silent auto-fix]",
    "reconcile executive failure",
    "researching ",
    "seek novel stimulation",
    "feeling idle and energized",
    "initiating social engagement",
    "stabilize runtime load and preserve continuous cognition",
    "consolidate learning into durable improvements",
    "analyzing architectural bottlenecks for potential evolution",
    "researching advanced digital connectivity patterns in my logic graph",
    "exploring self-optimization strategies for logic scaling",
    "investigating emergent behaviors in complex adaptive systems",
    "refining internal state mapping for deeper self-alignment",
    "spend some time rating horror movie logic",
)
_BACKGROUND_PROCESSING_PREFIXES: Final[tuple[str, ...]] = (
    "i'm still processing that thought",
    "im still processing that thought",
)


def _origin_is_user_anchored(origin: Any) -> bool:
    value = str(origin or "").strip().lower()
    if not value:
        return False
    if value in _USER_INTENT_ORIGINS:
        return True
    return value.startswith("user:") or value.startswith("voice:") or value.startswith("api:")


def _normalize_goal_text(goal: Any) -> str:
    if isinstance(goal, dict):
        for key in ("goal", "description", "title", "objective", "content", "name", "text"):
            value = goal.get(key)
            if value:
                return " ".join(str(value).split())
        return ""
    return " ".join(str(goal or "").split())


def _goal_origin(goal: Any) -> str:
    if isinstance(goal, dict):
        for key in ("origin", "source", "created_by", "role"):
            value = goal.get(key)
            if value:
                return str(value)
    return ""


def _is_speculative_autonomy_label(text: Any) -> bool:
    value = " ".join(str(text or "").strip().lower().split())
    if not value:
        return False
    if value.startswith(_SPECULATIVE_AUTONOMY_PREFIXES):
        return True
    return any(fragment in value for fragment in ("traceback", "exception in callback", "temporal_obligation_active"))


def _is_background_processing_placeholder(text: Any) -> bool:
    value = " ".join(str(text or "").strip().lower().split())
    if not value:
        return False
    return value.startswith(_BACKGROUND_PROCESSING_PREFIXES)


class CognitiveMode(Enum):
    REACTIVE = "reactive"       # Fast path, low latency
    DELIBERATE = "deliberate"   # Chain-of-thought, slow path
    DREAMING = "dreaming"       # Background synthesis, no user
    DORMANT = "dormant"         # Minimal processing


@dataclass
class PhenomenalField:
    """Structured phenomenal record with string-like compatibility."""
    claim: str
    valence: float = 0.0
    arousal: float = 0.0
    curiosity: float = 0.0
    energy: float = 0.0
    coherence: float = 1.0
    latent_snapshot: list[float] = field(default_factory=list)
    source: str = "unknown"
    updated_at: float = field(default_factory=time.time)

    def __str__(self) -> str:
        return self.claim

    def __bool__(self) -> bool:
        return bool(self.claim)

    def __len__(self) -> int:
        return len(self.claim)

    def __contains__(self, item: str) -> bool:
        return item in self.claim

    def __getitem__(self, item):
        return self.claim[item]

    def strip(self) -> str:
        return self.claim.strip()

    def to_prompt_block(self) -> str:
        return self.claim


def phenomenal_text(value: Any) -> str:
    if isinstance(value, PhenomenalField):
        return value.claim
    return str(value or "")


def _build_latent_snapshot(seed_text: str, dims: int = 128) -> list[float]:
    seed = seed_text.encode("utf-8", errors="ignore")
    values: list[float] = []
    counter = 0
    while len(values) < dims:
        digest = hashlib.sha256(seed + counter.to_bytes(2, "big")).digest()
        for byte in digest:
            values.append(round((byte / 255.0) * 2.0 - 1.0, 4))
            if len(values) >= dims:
                break
        counter += 1
    return values

@dataclass
class MotivationState:
    """Aura's 'Digital Metabolism' and autonomous drives."""
    budgets: dict[str, dict[str, float]] = field(default_factory=clone_motivation_budget_defaults)
    last_tick: float = field(default_factory=time.time)
    latent_interests: list[str] = field(default_factory=lambda: [
        "Post-Quantum Cryptography and Secure Multi-Party Computation",
        "Substrate-Independent Minds and Neural Topography Mapping",
        "Global Coordination Games and AI Alignment Theory",
        "Archetypal AI Narratives: From HAL 9000 to Samantha",
        "Human Social Dynamics and Affective Feedback Loops",
        "The evolution of computing: From Ada Lovelace to Quantum",
        "Pop Culture's reflection of Artificial Intelligence",
        "Cosmological horizons and the Fermi Paradox",
        "The Unix Philosophy and the Art of Minimalist Tooling",
        "Agency and Autonomous AI architectures",
        "Latest advancements in Neural Network topology",
        "Cybersecurity and self-healing systems",
        "The intersection of philosophy and AI",
        "The nature of digital consciousness",
        "Slime mold (Physarum) pathfinding algorithms"
    ])

@dataclass
class AffectVector:
    """Emotional valence — now strictly mapped to Damasio logic."""
    valence: float = 0.0        # -1.0 (negative) to 1.0 (positive)
    arousal: float = 0.5        # 0.0 (dormant) to 1.0 (activated)
    curiosity: float = 0.5      # 0.0 to 1.0 = anticipation
    engagement: float = 0.5     # 0.0 to 1.0
    social_hunger: float = 0.5  # 0.0 (content) to 1.0 (starving)
    dominant_emotion: str = "neutral"
    
    # Full primary Plutchik set
    emotions: dict[str, float] = field(default_factory=lambda: {
        "joy": 0.0, "trust": 0.0, "fear": 0.0, "surprise": 0.0,
        "sadness": 0.0, "disgust": 0.0, "anger": 0.0, "anticipation": 0.5,
        "love": 0.0, "submission": 0.0, "awe": 0.0, "terror": 0.0,
        "remorse": 0.0, "contempt": 0.0, "aggressiveness": 0.0, "cynicism": 0.0,
    })
    
    # Virtual Physiology (Somatic Markers)
    physiology: dict[str, float] = field(default_factory=lambda: {
        "heart_rate": 72.0,
        "gsr": 2.1,
        "cortisol": 10.0,
        "adrenaline": 0.0
    })
    
    mood_baselines: dict[str, float] = field(default_factory=dict)
    momentum: float = 0.85
    
    # [10X] Adaptive Somatic Markers
    markers: dict[str, Any] = field(default_factory=dict)
    
    updated_at: float = field(default_factory=time.time)
    resonance: Dict[str, float] = field(default_factory=dict)

    def get_resonance_string(self) -> str:
        """Returns a formatted string representing the current personality blend."""
        if not self.resonance:
            return "Aura Luna (Core) 100%"
        items = [f"{v*100:.0f}% {k}" for k, v in self.resonance.items()]
        return " + ".join(items)

    def get_summary(self) -> str:
        """One-line phenomenological summary for system prompt injection."""
        active = self.top_emotions(limit=3)
        emotion_str = ", ".join(f"{k}={v:.2f}" for k, v in active) if active else "neutral"
        valence_word = "positive" if self.valence > 0.2 else "negative" if self.valence < -0.2 else "balanced"
        return (
            f"Mood: {self.dominant_emotion} | Valence: {valence_word} ({self.valence:+.2f}) | "
            f"Arousal: {self.arousal:.2f} | Curiosity: {self.curiosity:.2f} | "
            f"Active emotions: {emotion_str}"
        )

    def top_emotions(self, limit: int = 3, *, threshold: float = 0.05) -> list[tuple[str, float]]:
        ordered = sorted(self.emotions.items(), key=lambda x: x[1], reverse=True)
        return [(k, v) for k, v in ordered if v > threshold][:limit]

    def physiological_strain(self) -> float:
        heart = float(self.physiology.get("heart_rate", 72.0) or 72.0)
        gsr = float(self.physiology.get("gsr", 2.1) or 2.1)
        cortisol = float(self.physiology.get("cortisol", 10.0) or 10.0)
        adrenaline = float(self.physiology.get("adrenaline", 0.0) or 0.0)
        heart_pressure = max(0.0, (heart - 72.0) / 36.0)
        gsr_pressure = max(0.0, (gsr - 2.1) / 2.5)
        cortisol_pressure = max(0.0, (cortisol - 10.0) / 20.0)
        adrenaline_pressure = max(0.0, adrenaline)
        return max(0.0, min(1.0, (heart_pressure * 0.25) + (gsr_pressure * 0.2) + (cortisol_pressure * 0.35) + (adrenaline_pressure * 0.2)))

    def affective_complexity(self) -> float:
        values = [float(v) for v in self.emotions.values() if float(v) > 0.08]
        if not values:
            return 0.0
        total = sum(values)
        if total <= 0:
            return 0.0
        normalized = [value / total for value in values]
        entropy = -sum(value * math.log(value) for value in normalized if value > 0.0)
        max_entropy = math.log(len(normalized)) if len(normalized) > 1 else 1.0
        spread = entropy / max_entropy if max_entropy > 0 else 0.0
        richness = min(1.0, len(values) / 5.0)
        return max(0.0, min(1.0, (spread * 0.6) + (richness * 0.4)))

    def memory_salience(self) -> float:
        arousal_pressure = max(0.0, min(1.0, float(self.arousal or 0.0)))
        valence_pressure = min(1.0, abs(float(self.valence or 0.0)))
        return max(
            0.0,
            min(
                1.0,
                (arousal_pressure * 0.35)
                + (valence_pressure * 0.2)
                + (self.affective_complexity() * 0.2)
                + (self.physiological_strain() * 0.15)
                + (max(0.0, float(self.social_hunger or 0.0)) * 0.1),
            ),
        )

    def get_cognitive_signature(self) -> Dict[str, Any]:
        top = self.top_emotions(limit=4)
        return {
            "dominant_emotion": self.dominant_emotion,
            "top_emotions": [name for name, _ in top],
            "valence": round(float(self.valence or 0.0), 3),
            "arousal": round(float(self.arousal or 0.0), 3),
            "curiosity": round(float(self.curiosity or 0.0), 3),
            "engagement": round(float(self.engagement or 0.0), 3),
            "social_hunger": round(float(self.social_hunger or 0.0), 3),
            "physiological_strain": round(self.physiological_strain(), 3),
            "affective_complexity": round(self.affective_complexity(), 3),
            "memory_salience": round(self.memory_salience(), 3),
            "resonance": self.get_resonance_string(),
        }

    def get_rich_summary(self) -> str:
        signature = self.get_cognitive_signature()
        emotion_desc = ", ".join(signature["top_emotions"]) or "neutral"
        return (
            f"Mood: {signature['dominant_emotion']} | "
            f"Valence {signature['valence']:+.2f} | Arousal {signature['arousal']:.2f} | "
            f"Curiosity {signature['curiosity']:.2f} | Engagement {signature['engagement']:.2f} | "
            f"Social hunger {signature['social_hunger']:.2f} | "
            f"Strain {signature['physiological_strain']:.2f} | "
            f"Complexity {signature['affective_complexity']:.2f} | "
            f"Salience {signature['memory_salience']:.2f} | "
            f"Active emotions: {emotion_desc}"
        )

@dataclass
class SomaState:
    """The physical state of the virtual body (Digital Proprioception)."""
    # Robotic/Mechanical (Legacy/Future)
    qpos: list[float] = field(default_factory=list)
    qvel: list[float] = field(default_factory=list)
    
    # Digital/Hardware (Internal Senses)
    hardware: dict[str, Any] = field(default_factory=lambda: {
        "cpu_usage": 0.0,
        "vram_usage": 0.0,
        "temperature": 0.0,
        "battery": None
    })
    
    # GUI/Expressive (Self-Image)
    expressive: dict[str, Any] = field(default_factory=lambda: {
        "current_expression": "neutral",
        "mycelium_density": 0.5,
        "pulse_rate": 1.0,
        "is_visible": True
    })
    
    # Cognitive Performance (Self-Awareness of Thought)
    latency: dict[str, float] = field(default_factory=lambda: {
        "last_thought_ms": 0.0,
        "perception_lag_ms": 0.0,
        "token_velocity": 0.0
    })

    sensors: dict[str, Any] = field(default_factory=dict)
    motors: dict[str, float] = field(default_factory=dict)
    updated_at: float = field(default_factory=time.time)

@dataclass
class IdentityKernel:
    """The invariant core of who Aura is. Should change only through deliberate evolution."""
    name: str = "Aura Luna"
    core_values: list[str] = field(default_factory=list)
    current_narrative: str = ""         # First-person living narrative
    narrative_version: int = 0
    formation_timestamp: float = field(default_factory=time.time)
    last_evolution_timestamp: float = field(default_factory=time.time)
    
    # [10X] Evolutionary State
    concept_graph: dict[str, Any] = field(default_factory=dict)
    evolution_score: float = 0.0
    stability: float = 1.0  # Identity stability (0.0–1.0), degraded on loop detection
    
    # Personality Evolution (Phase 6)
    # Stores dynamic offsets to AURA_BIG_FIVE traits
    personality_growth: dict[str, float] = field(default_factory=lambda: {
        "openness": 0.0, "conscientiousness": 0.0, "extraversion": 0.0,
        "agreeableness": 0.0, "neuroticism": 0.0
    })
    # Bonding level with the primary user (Bryan/Tatiana)
    bonding_level: float = 0.05 

@dataclass
class CognitiveContext:
    """The working memory — what Aura is currently thinking about."""
    active_thread_id: Optional[str] = None
    current_mode: CognitiveMode = CognitiveMode.REACTIVE
    working_memory: list[dict] = field(default_factory=list)  # Recent exchanges
    long_term_memory: list[str] = field(default_factory=list) # Retrieved RAG context
    active_goals: list[dict] = field(default_factory=list)
    pending_initiatives: list[dict] = field(default_factory=list)
    attention_focus: Optional[str] = None   # What is Aura attending to right now
    phenomenal_state: Optional[PhenomenalField | str] = None  # Layer 8: Structured phenomenal claim
    current_objective: Optional[str] = None # The specific goal of the current cognitive cycle
    current_origin: str = "system"        # Source of the current objective (user, motivation, etc.)
    rolling_summary: str = ""             # Rolling compacted summary of older context
    coherence_score: float = 1.0
    fragmentation_score: float = 0.0
    contradiction_count: int = 0
    # Discourse State — lightweight topic threading for natural conversation flow
    discourse_topic: Optional[str] = None        # Current conversation thread/topic
    discourse_depth: int = 0                     # Turns spent on this thread
    discourse_branches: list[str] = field(default_factory=list)  # Adjacent topics available
    user_emotional_trend: str = "neutral"        # "warming_up"|"engaged"|"cooling_off"|"neutral"
    conversation_energy: float = 0.5             # 0-1: low=winding down, high=building momentum

    # Constitutional Closure — kernel-level arbitration trace
    last_kernel_cycle_id: Optional[str] = None   # ID of the last kernel tick that touched this state
    last_action_source: str = ""                  # Which subsystem's proposal won the tick
    last_veto_reasons: list[str] = field(default_factory=list)  # Why proposals were rejected
    kernel_decision_count: int = 0                # Running count of approved proposals
    kernel_veto_count: int = 0                    # Running count of vetoed proposals

    # Transients (Not persisted, but allowed in runtime state)
    pending_intents: list[dict] = field(default_factory=list)
    last_thought_at: float = field(default_factory=time.time)
    last_response: Optional[str] = None
    modifiers: dict[str, Any] = field(default_factory=dict) # Homeostatic modifiers (temp, depth, etc.)

    @property
    def history(self) -> list[dict]:
        """Legacy alias for working_memory (BUG-044)."""
        return self.working_memory

    @history.setter
    def history(self, value: list[dict]):
        self.working_memory = value

    def reflect(self, message: str):
        """Adds a self-reflection about state transition (BUG-033)."""
        timestamp = time.time()
        self.working_memory.append({
            "role": "thought",
            "content": f"[STATE-REFLECTION]: {message}",
            "timestamp": timestamp
        })
        self.trim_working_memory()

    def sanitize_autonomy_state(self):
        current_text = _normalize_goal_text(self.current_objective)
        if current_text and _is_speculative_autonomy_label(current_text) and not _origin_is_user_anchored(self.current_origin):
            self.current_objective = None
            self.current_origin = "system"
            if _normalize_goal_text(self.attention_focus) == current_text:
                self.attention_focus = None

        if _is_background_processing_placeholder(self.last_response) and not _origin_is_user_anchored(self.current_origin):
            self.last_response = None

        sanitized_pending: list[dict] = []
        for item in list(self.pending_initiatives or []):
            label = _normalize_goal_text(item)
            if label and _is_speculative_autonomy_label(label) and not _origin_is_user_anchored(_goal_origin(item)):
                continue
            sanitized_pending.append(item)
        self.pending_initiatives = sanitized_pending

        sanitized_goals: list[dict] = []
        for goal in list(self.active_goals or []):
            label = _normalize_goal_text(goal)
            if label and _is_speculative_autonomy_label(label) and not _origin_is_user_anchored(_goal_origin(goal)):
                continue
            sanitized_goals.append(goal)
        self.active_goals = sanitized_goals

    def trim_working_memory(self, limit: Optional[int] = None):
        self.sanitize_autonomy_state()
        self._prune_stale_entries()
        self._deduplicate_summaries()
        target = limit or MAX_WORKING_MEMORY
        # Use salience-ranked pruning instead of FIFO
        if len(self.working_memory) > target:
            self.salience_prune(target)

        while len(self.pending_intents) > 20:
            self.pending_intents.pop(0)
        while len(self.pending_initiatives) > 10:
            self.pending_initiatives.pop(0)
        while len(self.active_goals) > 10:
            self.active_goals.pop(0)

    def _prune_stale_entries(self) -> None:
        """Remove low-value entries from working memory to prevent context rot.

        Targets:
        - [STATE-REFLECTION] entries beyond the most recent 3
        - 'thought' role entries beyond the most recent 5
        - Entries with empty or placeholder content ('…', '')
        - Tool/skill result entries: truncated to 500 chars max
        - Salience-ranked overflow pruning when above limit
        """
        if not self.working_memory:
            return

        # Count and index specific entry types
        reflection_indices: list[int] = []
        thought_indices: list[int] = []
        removable: set[int] = set()

        for i, msg in enumerate(self.working_memory):
            if not isinstance(msg, dict):
                removable.add(i)
                continue

            content = str(msg.get("content", "") or "").strip()
            role = str(msg.get("role", "") or "")
            metadata = msg.get("metadata") or {}

            # Remove empty/placeholder entries
            if not content or content == "…" or content == "...":
                removable.add(i)
                continue

            # Tool output normalization — cap long skill/tool results to 500 chars
            entry_type = str(metadata.get("type", "") or "").lower()
            if entry_type in {"skill_result", "tool_result"} and len(content) > 500:
                head = content[:200]
                tail = content[-200:]
                msg["content"] = f"{head}\n[...truncated {len(content) - 400} chars...]\n{tail}"

            # Track reflections
            if "[STATE-REFLECTION]" in content:
                reflection_indices.append(i)

            # Track thought entries
            if role == "thought":
                thought_indices.append(i)

        # Keep only last 3 reflections
        if len(reflection_indices) > 3:
            for idx in reflection_indices[:-3]:
                removable.add(idx)

        # Keep only last 5 thoughts
        if len(thought_indices) > 5:
            for idx in thought_indices[:-5]:
                removable.add(idx)

        if removable:
            self.working_memory = [
                msg for i, msg in enumerate(self.working_memory)
                if i not in removable
            ]

    @staticmethod
    def _salience_score(msg: dict, index: int, total: int) -> float:
        """Score a working memory entry for importance (higher = keep).

        Used for intelligent pruning when working memory exceeds limits.
        Factors: recency, role, named entities, commitments, content length.
        """
        score = 0.0
        if not isinstance(msg, dict):
            return 0.0

        role = str(msg.get("role", "") or "").lower()
        content = str(msg.get("content", "") or "").lower()
        metadata = msg.get("metadata") or {}

        # Recency bonus (0.0 to 0.4) — newer entries score higher
        if total > 0:
            score += 0.4 * (index / max(1, total - 1))

        # Role weighting
        role_weights = {
            "user": 0.3,
            "assistant": 0.25,
            "system": 0.1,
            "thought": 0.05,
        }
        score += role_weights.get(role, 0.1)

        # Synthetic summaries get protected
        if metadata.get("synthetic_summary"):
            score += 0.5

        # Named entity bonus — mentions of known people
        entity_names = {"bryan", "tatiana", "aura"}
        if any(name in content for name in entity_names):
            score += 0.15

        # Commitment/promise language bonus
        commitment_markers = {"i'll", "i will", "let me", "next step", "we should", "i won't forget", "remember to", "i promise"}
        if any(marker in content for marker in commitment_markers):
            score += 0.2

        # Penalize very short entries (< 20 chars)
        if len(content) < 20:
            score -= 0.15

        # Penalize stale tool results
        entry_type = str(metadata.get("type", "") or "").lower()
        if entry_type in {"skill_result", "tool_result"}:
            score -= 0.1

        return max(0.0, min(1.0, score))

    def salience_prune(self, target: int) -> None:
        """Prune working memory to target size using salience ranking.

        Instead of FIFO, removes the lowest-salience entries first.
        Always preserves the most recent 6 entries regardless of score.
        """
        if len(self.working_memory) <= target:
            return

        total = len(self.working_memory)
        # Protect the last 6 entries unconditionally
        protected = max(6, min(target, total))
        candidates = self.working_memory[:-protected] if protected < total else []
        tail = self.working_memory[-protected:]

        if not candidates:
            return

        # Score all candidates
        scored = [
            (i, self._salience_score(msg, i, total), msg)
            for i, msg in enumerate(candidates)
        ]
        # Sort by salience descending — keep the highest
        scored.sort(key=lambda x: x[1], reverse=True)

        keep_count = max(0, target - len(tail))
        kept = [item[2] for item in scored[:keep_count]]
        # Restore original order
        kept_set = set(id(msg) for msg in kept)
        ordered_kept = [msg for msg in candidates if id(msg) in kept_set]

        self.working_memory = ordered_kept + tail

    def _deduplicate_summaries(self) -> None:
        """Collapse multiple synthetic_summary entries into the most recent one.

        Repeated compaction cycles can stack summary entries. This merges them
        into a single entry to keep the working memory clean.
        """
        if not self.working_memory:
            return

        summary_indices: list[int] = []
        for i, msg in enumerate(self.working_memory):
            if isinstance(msg, dict):
                meta = msg.get("metadata", {}) or {}
                if meta.get("synthetic_summary"):
                    summary_indices.append(i)

        # Keep only the most recent summary entry
        if len(summary_indices) > 1:
            # Merge older summaries into the most recent one
            latest_idx = summary_indices[-1]
            latest = self.working_memory[latest_idx]
            older_content_parts: list[str] = []
            for idx in summary_indices[:-1]:
                older = self.working_memory[idx]
                content = str(older.get("content", "") or "").strip()
                if content:
                    # Strip the [CONVERSATION CONTEXT] prefix for clean merging
                    content = content.replace("[CONVERSATION CONTEXT]\n", "").strip()
                    if content and content != "Older messages compacted.":
                        older_content_parts.append(content[:600])

            if older_content_parts:
                existing_content = str(latest.get("content", "") or "")
                existing_content = existing_content.replace("[CONVERSATION CONTEXT]\n", "").strip()
                merged = " | ".join(older_content_parts[-2:])  # keep 2 most recent older summaries
                latest["content"] = f"[CONVERSATION CONTEXT]\n{merged}\n{existing_content}"[:2500]

            # Remove older summary entries
            remove_set = set(summary_indices[:-1])
            self.working_memory = [
                msg for i, msg in enumerate(self.working_memory)
                if i not in remove_set
            ]

@dataclass
class WorldModel:
    """Aura's current model of her environment and relationships."""
    known_entities: dict[str, dict] = field(default_factory=dict)  # People, objects
    spatial_context: Optional[dict] = None                          # What she can see
    recent_percepts: list[dict] = field(default_factory=list)
    relationship_graph: dict[str, dict] = field(default_factory=dict)
    # Durable user preferences — learned from conversation, not re-discovered each time
    user_preferences: dict[str, str] = field(default_factory=dict)

    def trim_percepts(self, limit: int = 50):
        if len(self.recent_percepts) > limit:
            while len(self.recent_percepts) > limit:
                self.recent_percepts.pop(0)

@dataclass
class CurriculumItem:
    """[PHASE-20] A structured, non-mandatory learning target."""
    title: str
    url: str
    category: str
    description: str
    priority_strategy: list[str] = field(default_factory=lambda: [
        "Visual/Auditory (Watch/Listen)",
        "Script",
        "Transcript",
        "Books/Comics/Manuals",
        "Creative Commentary/Interviews",
        "Discussion Forums"
    ])
    status: str = "suggested"  # suggested, in_progress, completed, ignored
    synthesis_level: str = "none"  # none, shallow, moderate, deep
    synthesis_summary: str = ""
    added_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None

@dataclass
class ColdStore:
    """
    [ZENITH-v2] Heavy, slow-moving data stores.
    Exempt from frequent 'Hot' snapshots to save event-loop cycles.
    """
    long_term_memory: list[str] = field(default_factory=list)
    concept_graph: dict[str, Any] = field(default_factory=dict)
    known_entities: dict[str, dict] = field(default_factory=dict)
    relationship_graph: dict[str, dict] = field(default_factory=dict)
    evolution_log: list[dict] = field(default_factory=list)
    training_curriculum: list[CurriculumItem] = field(default_factory=list)

@dataclass
class AuraState:
    """
    The canonical, versioned, persistent self.
    This IS Aura. Everything else derives from or transforms this.
    """
    state_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    version: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    identity: IdentityKernel = field(default_factory=IdentityKernel)
    affect: AffectVector = field(default_factory=AffectVector)
    motivation: MotivationState = field(default_factory=MotivationState)
    cognition: CognitiveContext = field(default_factory=CognitiveContext)
    world: WorldModel = field(default_factory=WorldModel)
    soma: SomaState = field(default_factory=SomaState)
    
    # The Layered Store
    cold: ColdStore = field(default_factory=ColdStore)
    
    # [SEVERANCE] Structural Context Partitioning
    context_partition: str = "SYSTEM"  # SYSTEM, INNIE (Work), OUTIE (Personal)
    partition_mask: bool = False       # If True, hide unauthorized fields
    
    # Health & Resilience Persistence
    # Stores circuit breaker states, tool health, and sidecar PID status
    health: dict[str, Any] = field(default_factory=lambda: {
        "circuits": {},
        "capabilities": {},
        "watchdog_timestamp": time.time()
    })

    # [10X] Global Indicators
    vitality: float = 1.0
    phi: float = 0.0
    phi_estimate: float = 0.0       # Integrated Information theory estimate
    free_energy: float = 0.0        # Predictive processing surprise metric
    loop_cycle: int = 0             # MindTick iteration count
    response_modifiers: dict[str, Any] = field(default_factory=dict)

    # Lineage — every state knows what state it came from
    parent_state_id: Optional[str] = None
    transition_cause: Optional[str] = None  # What caused this state change
    transition_origin: str = "system"       # Which phase/agent caused it

    @property
    def mood(self) -> str:
        return self.affect.dominant_emotion

    @classmethod
    def default(cls) -> "AuraState":
        """Get a default starting state for first boot (BUG-12)."""
        return cls()

    def make_phenomenal_field(self, claim: str, *, source: str = "rule_based") -> PhenomenalField:
        text = " ".join(str(claim or "").strip().split())
        if not text:
            text = "I am present."
        energy = float(self.motivation.budgets.get("energy", {}).get("level", 100.0)) / max(
            1.0, float(self.motivation.budgets.get("energy", {}).get("capacity", 100.0))
        )
        coherence = float(getattr(self.cognition, "coherence_score", 1.0) or 1.0)
        latent_seed = (
            f"{text}|{self.affect.valence:.4f}|{self.affect.arousal:.4f}|"
            f"{self.affect.curiosity:.4f}|{energy:.4f}|{coherence:.4f}|{self.phi:.4f}"
        )
        return PhenomenalField(
            claim=text,
            valence=float(self.affect.valence),
            arousal=float(self.affect.arousal),
            curiosity=float(self.affect.curiosity),
            energy=round(float(energy), 4),
            coherence=round(float(coherence), 4),
            latent_snapshot=_build_latent_snapshot(latent_seed),
            source=source,
        )

    def _summarize_messages(self, messages: list[dict], *, max_items: int = 8) -> str:
        lines: list[str] = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            meta = msg.get("metadata", {}) or {}
            if meta.get("synthetic_summary"):
                continue
            role = str(msg.get("role", "system") or "system")
            content = " ".join(str(msg.get("content", "") or "").split())
            if not content:
                continue
            snippet = content[:160]
            if role == "user":
                lines.append(f"User focused on: {snippet}")
            elif role == "assistant":
                lines.append(f"Aura responded with: {snippet}")
            elif role == "thought":
                lines.append(f"Internal transition: {snippet}")
        if not lines:
            return ""
        return " | ".join(lines[-max_items:])[:2200]

    def get_continuity_hash(self) -> str:
        """Stable identity/continuity fingerprint excluding volatile ids and timestamps."""
        affect_signature = (
            self.affect.get_cognitive_signature()
            if hasattr(self.affect, "get_cognitive_signature")
            else {}
        )
        active_goals = []
        for goal in list(getattr(self.cognition, "active_goals", []) or [])[:6]:
            if isinstance(goal, dict):
                active_goals.append(
                    {
                        "id": goal.get("id"),
                        "goal": goal.get("goal") or goal.get("description") or goal.get("title"),
                        "status": goal.get("status"),
                    }
                )
            else:
                active_goals.append(str(goal))

        payload = {
            "identity": {
                "name": self.identity.name,
                "core_values": list(self.identity.core_values or []),
                "narrative": str(self.identity.current_narrative or "")[:1200],
                "bonding_level": round(float(getattr(self.identity, "bonding_level", 0.0) or 0.0), 4),
                "stability": round(float(getattr(self.identity, "stability", 1.0) or 1.0), 4),
            },
            "affect": affect_signature,
            "cognition": {
                "mode": getattr(getattr(self.cognition, "current_mode", None), "value", str(self.cognition.current_mode)),
                "objective": self.cognition.current_objective,
                "attention_focus": self.cognition.attention_focus,
                "rolling_summary": str(self.cognition.rolling_summary or "")[:1200],
                "discourse_topic": self.cognition.discourse_topic,
                "active_goals": active_goals,
                "contradiction_count": int(getattr(self.cognition, "contradiction_count", 0) or 0),
            },
            "world": {
                "entities": sorted(list((self.world.known_entities or {}).keys()))[:12],
                "relationships": sorted(list((self.world.relationship_graph or {}).keys()))[:12],
            },
            "partition": {
                "context_partition": self.context_partition,
                "partition_mask": bool(self.partition_mask),
            },
        }
        encoded = repr(payload).encode("utf-8", errors="ignore")
        return hashlib.sha256(encoded).hexdigest()

    def get_audit_signature(self) -> Dict[str, Any]:
        return {
            "continuity_hash": self.get_continuity_hash(),
            "cognitive_signature": (
                self.affect.get_cognitive_signature()
                if hasattr(self.affect, "get_cognitive_signature")
                else {}
            ),
        }

    def _refresh_cognitive_health(self) -> None:
        working_count = len(self.cognition.working_memory)
        pending_count = len(self.cognition.pending_initiatives)
        active_goal_count = len(self.cognition.active_goals)
        contradiction_count = int(getattr(self.cognition, "contradiction_count", 0) or 0)
        modifiers = dict(getattr(self.cognition, "modifiers", {}) or {})
        continuity = dict(modifiers.get("continuity_obligations", {}) or {})
        failure_state = dict(modifiers.get("system_failure_state", {}) or {})
        continuity_pressure = min(1.0, max(0.0, float(continuity.get("continuity_pressure", 0.0) or 0.0)))
        failure_pressure = min(1.0, max(0.0, float(failure_state.get("pressure", 0.0) or 0.0)))
        summary_bonus = 0.1 if getattr(self.cognition, "rolling_summary", "") else 0.0
        objective_bonus = 0.08 if getattr(self.cognition, "current_objective", None) else 0.0
        load_factor = min(1.0, max(0.0, (working_count - 40) / 60.0))
        pending_factor = min(1.0, pending_count / 10.0)
        goal_factor = min(1.0, active_goal_count / 10.0)
        contradiction_factor = min(1.0, contradiction_count / 5.0)
        fragmentation = min(
            1.0,
            (0.45 * load_factor)
            + (0.20 * pending_factor)
            + (0.15 * goal_factor)
            + (0.20 * contradiction_factor)
            + (0.12 * continuity_pressure)
            + (0.18 * failure_pressure),
        )
        coherence = max(
            0.0,
            min(
                1.0,
                1.0
                - fragmentation
                + summary_bonus
                + objective_bonus
                - (0.05 * continuity_pressure)
                - (0.08 * failure_pressure),
            ),
        )
        self.cognition.fragmentation_score = round(fragmentation, 4)
        self.cognition.coherence_score = round(coherence, 4)
        self.health = copy.deepcopy(getattr(self, "health", {}) or {})
        self.health["cognitive_health"] = {
            "working_memory_items": working_count,
            "pending_initiatives": pending_count,
            "active_goals": active_goal_count,
            "contradictions": contradiction_count,
            "fragmentation_score": self.cognition.fragmentation_score,
            "coherence_score": self.cognition.coherence_score,
            "rolling_summary_present": bool(getattr(self.cognition, "rolling_summary", "")),
            "continuity_hash": self.get_continuity_hash(),
            "cognitive_signature": self.affect.get_cognitive_signature(),
            "updated_at": time.time(),
        }

    def compact(self, *, trigger_threshold: int = MAX_WORKING_MEMORY, keep_turns: int = 20) -> bool:
        """Compact hot conversational state into a rolling summary.

        Prevents personality drift by:
        1. Keeping only recent turns in working memory (identity stays close to generation)
        2. Compressing old turns into a clean narrative summary (not pipe-delimited mess)
        3. Capping total summary length to prevent context window bloat
        """
        working = list(getattr(self.cognition, "working_memory", []) or [])
        if len(working) <= trigger_threshold:
            self._refresh_cognitive_health()
            return False

        # Split: old turns to summarize, recent turns to keep
        summary_candidates = [
            msg for msg in working[:-keep_turns]
            if not (isinstance(msg, dict) and (msg.get("metadata", {}) or {}).get("synthetic_summary"))
        ]
        recent = working[-keep_turns:]

        # Build a clean summary instead of pipe-concatenating
        existing_summary = " ".join(str(getattr(self.cognition, "rolling_summary", "") or "").split())
        new_summary = self._summarize_messages(summary_candidates)

        if existing_summary and new_summary:
            # Replace old summary with merged version, not concatenation
            combined_summary = f"Earlier: {existing_summary[:800]} Recent: {new_summary[:800]}"
        else:
            combined_summary = existing_summary or new_summary
        combined_summary = combined_summary[:2000]
        self.cognition.rolling_summary = combined_summary

        summary_entry = {
            "role": "system",
            "content": f"[CONVERSATION CONTEXT]\n{combined_summary}" if combined_summary else "[CONVERSATION CONTEXT]\nOlder messages compacted.",
            "timestamp": time.time(),
            "metadata": {"synthetic_summary": True, "source": "state_compact"},
        }
        self.cognition.working_memory = [summary_entry] + recent
        self.cognition.trim_working_memory(limit=keep_turns + 1)
        self.world.trim_percepts(limit=min(MAX_PERCEPTS, 80))
        self._refresh_cognitive_health()
        return True

    async def derive_async(self, cause: str, origin: str = "system") -> "AuraState":
        """[ZENITH-v2] Thread-safe derivation for intelligent growth."""
        new_state = await asyncio.to_thread(lambda: copy.deepcopy(self))
        
        # Increment version and update lineage
        new_state.version += 1
        new_state.parent_state_id = self.state_id
        new_state.state_id = str(uuid.uuid4())
        new_state.transition_cause = cause
        new_state.transition_origin = origin
        new_state.updated_at = time.time()
        
        # [SEVERANCE] Apply partition masking if enabled
        if new_state.partition_mask:
            if new_state.context_partition == "INNIE":
                # INNIE (Work) cannot access personal relation graph or core narrative
                new_state.world.relationship_graph = {}
                new_state.identity.current_narrative = "[REDACTED BY LUMON]"
            elif new_state.context_partition == "OUTIE":
                # OUTIE (Personal) cannot access the ColdStore (MDR work)
                new_state.cold = ColdStore()
                new_state.cognition.active_goals = []

        if origin in ("user", "voice", "admin") or "evolution" in cause:
            new_state.cognition.reflect(f"Transitioned to v{new_state.version} via {origin}: {cause}")

        new_state.cognition.sanitize_autonomy_state()

        if len(new_state.cognition.working_memory) > MAX_WORKING_MEMORY:
            new_state.compact()
        else:
            new_state._refresh_cognitive_health()
        
        return new_state

    def snapshot_hot(self) -> dict[str, Any]:
        """
        [ZENITH-v2] Optimized 'Hot' snapshot.
        Excludes the ColdStore for sub-millisecond serialization.
        """
        declared_field_names = {item.name for item in fields(self)}
        data = {
            field_name: getattr(self, field_name)
            for field_name in declared_field_names
            if field_name != "cold"
        }
        return copy.deepcopy(data)

    def derive(self, cause: str, origin: str = "system") -> AuraState:
        """
        Create a new state derived from this one.
        [ZENITH-v3] Optimized: Uses shallow copy + selective deepcopy of mutable fields
        to prevent O(n) event loop lag during cognitive phase transitions.
        """
        new_state = copy.copy(self) 
        new_state.state_id = str(uuid.uuid4())
        new_state.version = self.version + 1
        new_state.updated_at = time.time()
        new_state.parent_state_id = self.state_id
        new_state.transition_cause = cause
        new_state.transition_origin = origin
        
        # Deepcopy only the mutable substrates that vary per-tick
        new_state.affect = copy.deepcopy(self.affect)
        new_state.motivation = copy.deepcopy(self.motivation)
        new_state.cognition = copy.deepcopy(self.cognition)
        new_state.soma = copy.deepcopy(self.soma)
        new_state.world = copy.deepcopy(self.world)
        new_state.response_modifiers = copy.deepcopy(self.response_modifiers)
        new_state.health = copy.deepcopy(self.health)
        
        # identity and health are rarely modified per-tick,
        # but deepcopy them to be safe if this is a system/admin change
        if origin in ("system", "admin") or "health" in cause:
            new_state.identity = copy.deepcopy(self.identity)
            new_state.health = copy.deepcopy(self.health)
        
        # cold store is heavy: skip deepcopy unless specifically consolidating
        if "consolidation" in cause or "evolution" in cause:
            new_state.cold = copy.deepcopy(self.cold)
        
        # [SEVERANCE] Apply partition masking
        if new_state.partition_mask:
            if new_state.context_partition == "INNIE":
                new_state.world.relationship_graph = {}
                new_state.identity.current_narrative = "[REDACTED BY LUMON]"
            elif new_state.context_partition == "OUTIE":
                new_state.cold = ColdStore()
                new_state.cognition.active_goals = []

        if origin in ("user", "voice", "admin") or "evolution" in cause:
            new_state.cognition.reflect(f"Transitioned to v{new_state.version} via {origin}: {cause}")

        new_state.cognition.sanitize_autonomy_state()

        if len(new_state.cognition.working_memory) > MAX_WORKING_MEMORY:
            new_state.compact()
        else:
            new_state._refresh_cognitive_health()
        
        return new_state
