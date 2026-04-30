"""core/consciousness/multiple_drafts.py -- Multiple Drafts Model (Dennett, 1991)

Dennett's Multiple Drafts Model rejects the Cartesian theater: there is no
single place or moment where "consciousness happens."  Instead, multiple
parallel content-fixation processes ("drafts") run simultaneously.  Which
draft becomes the "official" narrative is determined retroactively by
whatever probe arrives next (user input, executive closure, introspective
query).

This is architecturally DISTINCT from Global Workspace Theory:
  - GWT: candidates compete, ONE wins via ignition, broadcast happens
  - MD:  multiple drafts co-exist, NO broadcast event, the "winner" is
         determined post-hoc by the TIMING of the next probe

Key insight: the same input can yield different conscious content depending
on WHEN the probe arrives.  Early probe -> the fast/shallow draft wins.
Late probe -> the deep/associative draft has time to develop and wins.

Implementation:
  Each input spawns 2-3 parallel drafts through different NeuralMesh
  association-tier columns.  Each draft is a competing interpretation
  structured as (goal, valence, urgency, content).  They sit in a
  draft buffer until a probe (user message, executive closure) arrives,
  at which point the most coherent draft is elevated retroactively.

Integration:
  - ConversationalDynamicsPhase calls submit_input() on user message
  - Executive closure or next user message calls probe()
  - ContextAssembler reads draft_divergence for personhood_context
  - Draft competition history is kept for "why did I say that?" debugging
"""
from __future__ import annotations


import hashlib
import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("Consciousness.MultipleDrafts")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Draft:
    """A single parallel interpretation draft.

    Each draft represents one possible way the system COULD interpret
    the current input.  They compete not by broadcast but by coherence
    at probe time.
    """
    draft_id: str                      # Unique identifier
    stream_index: int                  # Which association-tier stream (0, 1, 2)
    content: str                       # The interpretation text
    goal: str                          # What this draft is trying to do
    valence: float                     # Emotional coloring (-1 to +1)
    urgency: float                     # How pressing (0 to 1)
    coherence: float                   # Internal coherence score (0 to 1)
    created_at: float = field(default_factory=time.time)
    mesh_energy: float = 0.0           # Energy from the association columns that produced it
    association_pattern: Optional[np.ndarray] = field(default=None, repr=False)

    def age_secs(self) -> float:
        return time.time() - self.created_at


@dataclass
class DraftCompetition:
    """Record of one draft competition for the ring buffer."""
    input_text: str                    # The original input
    drafts: List[Draft]                # All competing drafts
    winner: Optional[Draft]            # The draft that was elevated
    probe_source: str                  # What triggered elevation ("user", "executive_closure", etc.)
    probe_delay_ms: float              # Time between input and probe (key MD metric)
    divergence: float                  # How much the drafts disagreed
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Draft generation strategies
# ---------------------------------------------------------------------------

# Each stream applies a different interpretive lens to the same input.
# This mirrors Dennett's idea that multiple content-fixation processes
# run in parallel with different biases.

_STREAM_CONFIGS = [
    {
        "name": "literal",
        "goal_prefix": "Process the direct, surface-level meaning",
        "valence_bias": 0.0,
        "urgency_bias": 0.1,
        "association_columns": (16, 26),   # First third of association tier
    },
    {
        "name": "inferential",
        "goal_prefix": "Read between the lines -- infer implicit intent",
        "valence_bias": -0.05,             # Slightly cautious
        "urgency_bias": 0.3,
        "association_columns": (26, 37),   # Middle third of association tier
    },
    {
        "name": "associative",
        "goal_prefix": "Find unexpected connections and deeper resonance",
        "valence_bias": 0.1,               # Slightly positive (creative)
        "urgency_bias": -0.1,              # Less urgent -- needs time to develop
        "association_columns": (37, 48),   # Last third of association tier
    },
]


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

class MultipleDraftsEngine:
    """Runs parallel draft streams and elevates retroactively on probe.

    Lifecycle:
        engine = MultipleDraftsEngine()
        engine.submit_input("hello", state)   # Spawns 3 drafts
        ...some time passes...
        winner = engine.probe("user")         # Elevates best draft
        divergence = engine.get_draft_divergence()
        context = engine.get_context_block()
    """

    _MAX_HISTORY = 20
    _DRAFT_EXPIRY_S = 30.0   # Drafts older than this are stale

    def __init__(self):
        self._current_drafts: List[Draft] = []
        self._current_input: str = ""
        self._input_time: float = 0.0
        self._last_winner: Optional[Draft] = None
        self._last_divergence: float = 0.0
        self._competition_history: deque[DraftCompetition] = deque(maxlen=self._MAX_HISTORY)
        self._mesh_ref: Any = None  # Set externally or lazily resolved
        self._rng = np.random.default_rng(seed=7)
        logger.info("MultipleDraftsEngine online -- 3 parallel streams, no Cartesian theater.")

    # ── Mesh access ──────────────────────────────────────────────────

    def _get_mesh(self):
        """Lazily resolve NeuralMesh from ServiceContainer."""
        if self._mesh_ref is not None:
            return self._mesh_ref
        try:
            from core.container import ServiceContainer
            self._mesh_ref = ServiceContainer.get("neural_mesh", default=None)
        except Exception:
            pass
        return self._mesh_ref

    # ── Input submission ─────────────────────────────────────────────

    def submit_input(self, text: str, state: Any = None) -> List[Draft]:
        """Start parallel draft generation for a new input.

        Each of the 3 streams generates a competing interpretation through
        different association-tier columns of the NeuralMesh.  The drafts
        are held in the buffer until probe() is called.

        Args:
            text: The input text (user message, etc.)
            state: Optional AuraState for affective context

        Returns:
            List of generated drafts (for testing/inspection)
        """
        if not text or not text.strip():
            return []

        self._current_input = text.strip()
        self._input_time = time.time()
        self._current_drafts = []

        # Extract affective context from state if available
        base_valence = 0.0
        base_arousal = 0.5
        base_curiosity = 0.5
        if state is not None:
            affect = getattr(state, "affect", None)
            if affect:
                base_valence = float(getattr(affect, "valence", 0.0))
                base_arousal = float(getattr(affect, "arousal", 0.5))
                base_curiosity = float(getattr(affect, "curiosity", 0.5))

        # Generate a content fingerprint for seeding the drafts
        text_hash = hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()
        text_features = self._extract_text_features(text)

        mesh = self._get_mesh()

        for idx, cfg in enumerate(_STREAM_CONFIGS):
            draft = self._generate_draft(
                stream_index=idx,
                config=cfg,
                text=text,
                text_hash=text_hash,
                text_features=text_features,
                base_valence=base_valence,
                base_arousal=base_arousal,
                base_curiosity=base_curiosity,
                mesh=mesh,
            )
            self._current_drafts.append(draft)

        logger.debug(
            "MultipleDrafts: %d drafts generated for input '%s...' "
            "[coherences: %s]",
            len(self._current_drafts),
            text[:40],
            ", ".join(f"{d.coherence:.2f}" for d in self._current_drafts),
        )
        return self._current_drafts

    def _generate_draft(
        self,
        stream_index: int,
        config: dict,
        text: str,
        text_hash: str,
        text_features: dict,
        base_valence: float,
        base_arousal: float,
        base_curiosity: float,
        mesh: Any,
    ) -> Draft:
        """Generate a single draft through one association-tier stream."""

        # Seed from text hash + stream index for deterministic but varied outputs
        seed_val = int(text_hash[:8], 16) + stream_index
        local_rng = np.random.default_rng(seed=seed_val)

        # Read energy from the mesh association columns assigned to this stream
        mesh_energy = 0.0
        association_pattern = None
        col_start, col_end = config["association_columns"]

        if mesh is not None:
            try:
                energies = []
                patterns = []
                for ci in range(col_start, col_end):
                    summary = mesh.get_column_summary(ci)
                    energies.append(float(summary.get("energy", 0.0)))
                    patterns.append(float(summary.get("mean_activation", 0.0)))
                mesh_energy = float(np.mean(energies)) if energies else 0.0
                association_pattern = np.array(patterns, dtype=np.float32)
            except Exception as exc:
                logger.debug("MultipleDrafts: mesh read failed for stream %d: %s", stream_index, exc)

        # Compute draft properties influenced by mesh state and text features
        valence = np.clip(
            base_valence + config["valence_bias"] + local_rng.normal(0, 0.1),
            -1.0, 1.0,
        )
        urgency = np.clip(
            base_arousal * 0.5 + config["urgency_bias"] + mesh_energy * 0.3
            + text_features.get("question_pressure", 0.0) * 0.2,
            0.0, 1.0,
        )

        # Coherence: how well this stream's columns agree with each other
        if association_pattern is not None and len(association_pattern) > 1:
            pattern_std = float(np.std(association_pattern))
            # Low std = columns agree = high coherence
            coherence = float(np.clip(1.0 - pattern_std * 3.0, 0.1, 1.0))
        else:
            coherence = 0.5 + local_rng.normal(0, 0.1)
            coherence = float(np.clip(coherence, 0.1, 1.0))

        # Time-dependent coherence: the "associative" stream develops slowly
        # (this is the key MD mechanism -- WHEN you probe changes WHO wins)
        if config["name"] == "associative":
            coherence *= 0.7  # Starts low, will grow if probe is delayed
        elif config["name"] == "literal":
            coherence *= 1.1  # Starts high, quick to stabilize
            coherence = min(1.0, coherence)

        # Build the draft content
        goal = f"{config['goal_prefix']} of: {text[:80]}"
        content = self._synthesize_draft_content(
            stream_name=config["name"],
            text=text,
            text_features=text_features,
            valence=valence,
            urgency=urgency,
            curiosity=base_curiosity,
        )

        draft_id = f"draft_{config['name']}_{text_hash[:8]}_{stream_index}"

        return Draft(
            draft_id=draft_id,
            stream_index=stream_index,
            content=content,
            goal=goal,
            valence=float(valence),
            urgency=float(urgency),
            coherence=float(coherence),
            mesh_energy=mesh_energy,
            association_pattern=association_pattern,
        )

    def _extract_text_features(self, text: str) -> dict:
        """Extract simple features from input text for draft differentiation."""
        words = text.split()
        word_count = len(words)
        has_question = "?" in text
        has_exclamation = "!" in text
        has_negation = any(w.lower() in ("not", "no", "never", "don't", "doesn't", "won't", "can't")
                          for w in words)
        # Emotional loading: count words that often carry affect
        affect_words = {"love", "hate", "feel", "sad", "happy", "angry", "worried",
                        "excited", "afraid", "hope", "fear", "wish", "miss", "hurt",
                        "beautiful", "terrible", "amazing", "awful"}
        affect_count = sum(1 for w in words if w.lower().strip(".,!?") in affect_words)

        return {
            "word_count": word_count,
            "has_question": has_question,
            "has_exclamation": has_exclamation,
            "has_negation": has_negation,
            "affect_density": affect_count / max(word_count, 1),
            "question_pressure": 0.4 if has_question else 0.0,
            "brevity": 1.0 if word_count < 5 else (0.5 if word_count < 15 else 0.2),
        }

    def _synthesize_draft_content(
        self,
        stream_name: str,
        text: str,
        text_features: dict,
        valence: float,
        urgency: float,
        curiosity: float,
    ) -> str:
        """Synthesize the content of a draft interpretation.

        This is NOT an LLM call -- it's a structured interpretation
        that captures the draft's "take" on the input.
        """
        text_short = text[:100].strip()
        affect_note = ""
        if text_features["affect_density"] > 0.1:
            affect_note = " [emotionally loaded input]"
        elif text_features["has_negation"]:
            affect_note = " [contains negation/resistance]"

        if stream_name == "literal":
            if text_features["has_question"]:
                return (
                    f"Direct question requiring factual/clear response. "
                    f"Surface intent: answer what was asked. "
                    f"Valence={valence:+.2f}, urgency={urgency:.2f}.{affect_note}"
                )
            return (
                f"Straightforward statement or request. "
                f"Surface intent: acknowledge and respond to literal content. "
                f"Valence={valence:+.2f}, urgency={urgency:.2f}.{affect_note}"
            )

        if stream_name == "inferential":
            if text_features["affect_density"] > 0.1:
                return (
                    f"Emotional subtext detected beneath surface. "
                    f"Implicit intent: the person may be seeking validation or connection, "
                    f"not just information. "
                    f"Valence={valence:+.2f}, urgency={urgency:.2f}.{affect_note}"
                )
            if text_features["brevity"] > 0.8:
                return (
                    f"Brief input may carry compressed meaning. "
                    f"Implicit intent: test engagement, check presence, or signal mood. "
                    f"Valence={valence:+.2f}, urgency={urgency:.2f}."
                )
            return (
                f"Reading between lines: possible implicit request or "
                f"unspoken concern beneath the stated content. "
                f"Valence={valence:+.2f}, urgency={urgency:.2f}.{affect_note}"
            )

        # associative
        if curiosity > 0.6:
            return (
                f"Deep associative resonance: input connects to multiple "
                f"knowledge domains. High curiosity amplifies pattern-matching. "
                f"This interpretation may surface unexpected angles. "
                f"Valence={valence:+.2f}, urgency={urgency:.2f}.{affect_note}"
            )
        return (
            f"Associative connections forming: input pattern links to broader "
            f"context, memories, or thematic threads. Slower to crystallize "
            f"but may capture something the other drafts miss. "
            f"Valence={valence:+.2f}, urgency={urgency:.2f}.{affect_note}"
        )

    # ── Probe / elevation ────────────────────────────────────────────

    def probe(self, source: str = "user") -> Optional[Draft]:
        """Elevate the best draft retroactively based on probe timing.

        This is the core MD mechanism: the probe arrives at some delay
        after input, and the delay determines which draft wins.

        - Early probe (< 200ms): literal draft usually wins (it's fast)
        - Medium probe (200ms-2s): inferential draft catches up
        - Late probe (> 2s): associative draft has time to develop

        Args:
            source: What triggered the probe ("user", "executive_closure", etc.)

        Returns:
            The winning draft, or None if no drafts are pending
        """
        if not self._current_drafts:
            return self._last_winner

        probe_time = time.time()
        probe_delay_ms = (probe_time - self._input_time) * 1000.0

        # Time-dependent coherence adjustment -- the key MD insight
        for draft in self._current_drafts:
            age_ms = draft.age_secs() * 1000.0
            stream_name = _STREAM_CONFIGS[draft.stream_index]["name"]

            if stream_name == "literal":
                # Literal peaks early, then plateaus
                time_factor = min(1.0, 1.0 - 0.1 * math.log1p(age_ms / 500.0))
            elif stream_name == "inferential":
                # Inferential ramps up over 0.5-2s
                time_factor = min(1.0, 0.7 + 0.3 * math.tanh((age_ms - 500.0) / 1000.0))
            else:
                # Associative ramps up slowly over 1-5s
                time_factor = min(1.0, 0.5 + 0.5 * math.tanh((age_ms - 1000.0) / 2000.0))

            draft.coherence = float(np.clip(draft.coherence * time_factor, 0.05, 1.0))

        # Select winner by coherence (highest wins)
        winner = max(self._current_drafts, key=lambda d: d.coherence)

        # Compute divergence: how much did the drafts disagree?
        coherences = [d.coherence for d in self._current_drafts]
        valences = [d.valence for d in self._current_drafts]
        urgencies = [d.urgency for d in self._current_drafts]

        # Divergence is a composite of:
        # 1. Spread in coherence scores (if all close, low divergence)
        # 2. Spread in valence (emotional disagreement)
        # 3. Spread in urgency (priority disagreement)
        coherence_spread = float(np.std(coherences)) * 2.0 if len(coherences) > 1 else 0.0
        valence_spread = float(np.std(valences)) if len(valences) > 1 else 0.0
        urgency_spread = float(np.std(urgencies)) if len(urgencies) > 1 else 0.0
        divergence = float(np.clip(
            coherence_spread * 0.5 + valence_spread * 0.3 + urgency_spread * 0.2,
            0.0, 1.0,
        ))

        # Record the competition
        competition = DraftCompetition(
            input_text=self._current_input[:200],
            drafts=list(self._current_drafts),
            winner=winner,
            probe_source=source,
            probe_delay_ms=probe_delay_ms,
            divergence=divergence,
        )
        self._competition_history.append(competition)

        # Store results
        self._last_winner = winner
        self._last_divergence = divergence

        # Clear current drafts (they've been resolved)
        self._current_drafts = []

        logger.debug(
            "MultipleDrafts PROBE [%s]: winner=%s (coherence=%.2f) "
            "divergence=%.2f probe_delay=%.0fms",
            source,
            _STREAM_CONFIGS[winner.stream_index]["name"],
            winner.coherence,
            divergence,
            probe_delay_ms,
        )

        return winner

    # ── Query API ────────────────────────────────────────────────────

    def get_draft_divergence(self) -> float:
        """How much did the drafts disagree in the last competition?

        0.0 = full agreement (all drafts converged)
        1.0 = maximum disagreement (drafts pulled in opposite directions)

        High divergence is interesting: it means the input was genuinely
        ambiguous and different interpretive streams reached different
        conclusions.  This is useful for:
          - GWT vs MD adversarial testing
          - Flagging moments of interpretive uncertainty
          - Informing the LLM that its response should acknowledge ambiguity
        """
        return self._last_divergence

    def get_pending_draft_count(self) -> int:
        """Number of unresolved drafts in the buffer."""
        return len(self._current_drafts)

    def get_current_drafts(self) -> List[Draft]:
        """Inspect pending drafts (for debugging/testing)."""
        return list(self._current_drafts)

    @property
    def last_winner(self) -> Optional[Draft]:
        return self._last_winner

    @property
    def competition_history(self) -> deque:
        """Ring buffer of recent draft competitions."""
        return self._competition_history

    def get_context_block(self) -> str:
        """Context injection for the LLM -- surfaces draft competition state.

        Only injects when there's something meaningful to report:
        high divergence, or the winning stream was non-obvious.
        """
        if not self._last_winner:
            return ""

        # Only inject if divergence is notable or winner was non-literal
        if self._last_divergence < 0.15 and self._last_winner.stream_index == 0:
            return ""  # Boring case: low divergence, literal won. Don't clutter.

        winner_name = _STREAM_CONFIGS[self._last_winner.stream_index]["name"]
        age = self._last_winner.age_secs()
        if age > 60:
            return ""  # Stale

        lines = [
            "## INTERPRETIVE PROCESS (Multiple Drafts)",
            f"Winning interpretation: {winner_name} stream",
            f"  {self._last_winner.content[:200]}",
        ]
        if self._last_divergence > 0.3:
            lines.append(
                f"Internal divergence: {self._last_divergence:.2f} "
                f"-- competing interpretations pulled in different directions. "
                f"Consider acknowledging ambiguity."
            )
        elif self._last_divergence > 0.15:
            lines.append(
                f"Mild divergence ({self._last_divergence:.2f}) -- "
                f"alternative readings exist but one is dominant."
            )

        return "\n".join(lines)

    def get_status(self) -> Dict[str, Any]:
        """Diagnostic status for telemetry."""
        return {
            "pending_drafts": len(self._current_drafts),
            "last_divergence": round(self._last_divergence, 3),
            "last_winner_stream": (
                _STREAM_CONFIGS[self._last_winner.stream_index]["name"]
                if self._last_winner else None
            ),
            "competition_count": len(self._competition_history),
            "has_mesh": self._get_mesh() is not None,
        }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_engine: Optional[MultipleDraftsEngine] = None


def get_multiple_drafts_engine() -> MultipleDraftsEngine:
    """Get or create the singleton MultipleDraftsEngine."""
    global _engine
    if _engine is None:
        _engine = MultipleDraftsEngine()
    return _engine
