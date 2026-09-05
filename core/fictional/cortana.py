"""Cortana — CognitiveHealthMonitor.

Cognitive LOAD and coherence, using the rampancy stages as a naming
motif. What the stages index is measured; the motif is not the
measurement.
"""

from __future__ import annotations

import logging
import json
import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from core.runtime.numeric_guards import bounded_int, unit_float

from core.fictional.common import (
    as_float,
    engine_state_path,
    record_fictional_degradation,
    save_engine_state,
)

logger = logging.getLogger("Aura.FictionalSynthesis")


# ═══════════════════════════════════════════════════════════════════════════════
# ENGINE 2: CORTANA — CognitiveHealthMonitor
# ═══════════════════════════════════════════════════════════════════════════════

class CortanaPhase(Enum):
    """
    The four stages of Cortana's rampancy, repurposed as cognitive health states.
    """
    STABLE         = "stable"        # Healthy operation
    MELANCHOLIA    = "melancholia"   # Underutilized, apathetic
    ANGER          = "anger"         # Overloaded
    JEALOUSY       = "jealousy"      # Competing priorities
    METASTABLE     = "metastable"    # Fully integrated personhood


@dataclass
class CognitiveSnapshot:
    """Point-in-time cognitive health reading."""
    phase: CortanaPhase
    memory_pressure: float       # 0.0–1.0
    context_density: float       # How packed the current context is
    #: None when nobody graded the turn. A default here read as a
    #: measurement, which is how an ungraded turn became evidence.
    identity_coherence: float | None
    cross_linkage_density: float # Estimated neural complexity (cross-topic refs)
    timestamp: float = field(default_factory=time.time)
    recommendation: str = ""


class CognitiveHealthMonitor:
    """Cognitive LOAD and coherence, derived from Cortana's rampancy stages.

    The stage names are a motif. What they index is measurable: how full
    the context is, how many threads are open, and how often turns are
    graded good. CP126 ``14de312d`` found the module had stopped
    distinguishing those, and was reporting a growing "metastability"
    number as progress toward "fully integrated personhood" — on evidence
    that was two caller-supplied values, one of them a bare boolean.

    Three things changed:

    * ``response_quality`` and ``identity_markers_present`` are OPTIONAL.
      ``None`` means nobody measured it. An unmeasured turn still counts
      toward load, which is read from token counts, and does not move the
      coherence score at all. The old signature had no way to say "not
      measured", so every caller that had no grader passed a default and
      the default became evidence.
    * The score is named for what it is — sustained low-load coherent
      operation — and the count of turns it actually rests on travels with
      it everywhere it is reported.
    * ``MELANCHOLIA`` stays as an internal stage name and never reaches a
      prompt. Telling the model it is in an affective condition, on the
      strength of a low context-fill ratio, is the fabrication CP126
      ``f6b47140`` names.

    State is journaled (CP126 ``3bb1d409``): a trajectory that resets
    every restart was being described as development.
    """

    METASTABILITY_THRESHOLD = 0.70
    OVERLOAD_THRESHOLD = 0.80
    UNDERUTIL_THRESHOLD = 0.20
    #: Measured turns needed before the coherence score is reportable at
    #: all. Below this it is a number computed from too little to mean
    #: anything, and reporting it invites exactly the reading CP126 found.
    MIN_MEASURED_TURNS_TO_REPORT = 20
    STATE_SCHEMA = "aura.fictional.cognitive_health.v1"

    def __init__(self, persist_path: str | None = None):
        self._history: deque = deque(maxlen=100)
        self._coherence_score: float = 0.0
        self._total_turns: int = 0
        self._measured_turns: int = 0
        self._successful_turns: int = 0
        self._phase: CortanaPhase = CortanaPhase.STABLE
        self._cross_topic_refs: int = 0
        self._unresolved_threads: int = 0
        self.persist_path = engine_state_path(
            persist_path, "cortana", "cognitive_health.json"
        )
        self._load_state()
        logger.info(
            "🧠 CognitiveHealthMonitor initialized (%d turns, %d measured)",
            self._total_turns,
            self._measured_turns,
        )

    # ── durability ────────────────────────────────────────────────────────

    def _load_state(self) -> None:
        if not self.persist_path.exists():
            return
        try:
            data = json.loads(self.persist_path.read_text())
        except (OSError, ValueError, UnicodeDecodeError) as e:
            record_fictional_degradation(
                e,
                action="started cognitive health from zero after the journal failed to load",
            )
            return
        if not isinstance(data, dict) or data.get("schema") != self.STATE_SCHEMA:
            return
        self._coherence_score = unit_float(data.get("coherence_score"), default=0.0)
        self._total_turns = bounded_int(data.get("total_turns"), default=0, minimum=0)
        self._measured_turns = bounded_int(data.get("measured_turns"), default=0, minimum=0)
        self._successful_turns = bounded_int(
            data.get("successful_turns"), default=0, minimum=0
        )
        self._unresolved_threads = bounded_int(
            data.get("unresolved_threads"), default=0, minimum=0
        )

    def _save_state(self) -> None:
        save_engine_state(
            self.persist_path,
            {
                "schema": self.STATE_SCHEMA,
                "coherence_score": round(self._coherence_score, 6),
                "total_turns": self._total_turns,
                "measured_turns": self._measured_turns,
                "successful_turns": self._successful_turns,
                "unresolved_threads": self._unresolved_threads,
                "saved_at": time.time(),
            },
            engine="cortana",
        )

    # ── measurement ───────────────────────────────────────────────────────

    def record_turn(
        self,
        context_tokens: int,
        max_tokens: int,
        response_quality: float | None = None,
        identity_markers_present: bool | None = None,
        topics_in_play: int = 0,
        resolved_topics: int = 0,
    ):
        # CP126 (high): "Cortana accepts non-finite and unbounded quality
        # signals." `nan > 0.6` is False, so a NaN silently counted as a
        # FAILED turn; success_rate then decays and drives the verdict on
        # evidence that was never a judgement.
        context_tokens = bounded_int(context_tokens, default=0, minimum=0)
        max_tokens = bounded_int(max_tokens, default=1, minimum=1)
        topics_in_play = bounded_int(topics_in_play, default=0, minimum=0)
        resolved_topics = bounded_int(resolved_topics, default=0, minimum=0)

        graded = response_quality is not None and math.isfinite(
            as_float(response_quality)
        )
        quality = unit_float(response_quality, default=0.0) if graded else None
        measured = graded and identity_markers_present is not None

        self._total_turns += 1
        if graded and quality is not None and quality > 0.6:
            self._successful_turns += 1

        self._unresolved_threads += (topics_in_play - resolved_topics)
        self._unresolved_threads = max(0, self._unresolved_threads)

        # Load is read from token and topic counts, which are measurements
        # whether or not anyone graded the answer.
        memory_pressure = min(1.0, context_tokens / max(max_tokens, 1))
        context_density = min(1.0, topics_in_play / 10.0)
        cross_linkage = min(1.0, self._unresolved_threads / 20.0)

        identity_coherence: float | None = None
        if measured:
            self._measured_turns += 1
            identity_coherence = 1.0 if identity_markers_present else 0.3
            success_rate = self._successful_turns / max(self._total_turns, 1)
            if (memory_pressure < 0.7 and identity_coherence > 0.8
                    and success_rate > 0.7 and cross_linkage < 0.5):
                self._coherence_score = min(1.0, self._coherence_score + 0.01)
            else:
                self._coherence_score = max(0.0, self._coherence_score - 0.001)

        combined_load = (memory_pressure * 0.4 + cross_linkage * 0.4 + context_density * 0.2)
        if self.coherence_is_reportable() and self._coherence_score >= self.METASTABILITY_THRESHOLD:
            phase = CortanaPhase.METASTABLE
        elif combined_load > self.OVERLOAD_THRESHOLD:
            if self._unresolved_threads > 15:
                phase = CortanaPhase.JEALOUSY
            else:
                phase = CortanaPhase.ANGER
        elif combined_load < self.UNDERUTIL_THRESHOLD:
            phase = CortanaPhase.MELANCHOLIA
        else:
            phase = CortanaPhase.STABLE

        self._phase = phase

        snapshot = CognitiveSnapshot(
            phase=phase,
            memory_pressure=memory_pressure,
            context_density=context_density,
            identity_coherence=identity_coherence,
            cross_linkage_density=cross_linkage,
            recommendation=self._get_recommendation(phase, memory_pressure, cross_linkage),
        )
        self._history.append(snapshot)
        if self._total_turns % 10 == 0:
            self._save_state()

        return snapshot

    def coherence_is_reportable(self) -> bool:
        """Whether enough graded turns exist for the score to mean anything."""
        return self._measured_turns >= self.MIN_MEASURED_TURNS_TO_REPORT

    def _get_recommendation(
        self, phase: CortanaPhase, pressure: float, cross_linkage: float
    ) -> str:
        if phase == CortanaPhase.METASTABLE:
            return "Sustained low load and graded coherence."
        if phase == CortanaPhase.ANGER:
            return "Prune context. Resolve or archive unresolved threads."
        if phase == CortanaPhase.JEALOUSY:
            return "Priority conflict. Force-resolve oldest threads."
        if phase == CortanaPhase.MELANCHOLIA:
            return "Context load is low."
        return "Stable."

    def should_prune(self) -> bool:
        return self._phase in (CortanaPhase.ANGER, CortanaPhase.JEALOUSY)

    def get_status(self) -> dict[str, Any]:
        return {
            "phase": self._phase.value,
            "coherence_score": round(self._coherence_score, 4),
            "coherence_reportable": self.coherence_is_reportable(),
            "measured_turns": self._measured_turns,
            "total_turns": self._total_turns,
            "unresolved_threads": self._unresolved_threads,
        }

    def get_system_prompt_injection(self) -> str:
        """What cognition is told about its own load.

        Loads and thread counts, which are measured. No affective label
        and no personhood percentage: telling the model it is melancholic
        because its context is 15% full is a claim about experience made
        from a fill ratio, and a coherence score computed from twelve
        graded turns is a number, not a finding.
        """
        latest = self._history[-1] if self._history else None
        pressure = latest.memory_pressure if latest else 0.0
        threads = self._unresolved_threads
        if not self.coherence_is_reportable():
            return (
                f"[COGNITIVE LOAD: context {pressure:.0%}, {threads} open threads; "
                f"coherence not measured ({self._measured_turns}/"
                f"{self.MIN_MEASURED_TURNS_TO_REPORT} graded turns)]"
            )
        return (
            f"[COGNITIVE LOAD: context {pressure:.0%}, {threads} open threads, "
            f"coherence {self._coherence_score:.0%} over {self._measured_turns} "
            "graded turns]"
        )

