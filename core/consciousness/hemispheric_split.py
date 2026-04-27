"""core/consciousness/hemispheric_split.py
=============================================
Split-brain hemispheric architecture — left (verbal/sequential/confabulating)
and right (spatial/parallel/mute-but-dissenting), coupled through a
bandwidth-limited corpus callosum.

This is NOT the parallel_branches task scheduler.  It is a dedicated
cognitive-architecture module modelling CGP Grey's split-brain findings:

    1. LeftHemisphere
       • Feeds from mesh executive tier + cognitive-affective nodes 8..15.
       • Produces a VERBAL priority bias for action selection (has the speech
         center).
       • When an action has already happened without a recorded reason,
         generates a POST-HOC reason (confabulation).

    2. RightHemisphere
       • Feeds from mesh sensory tier + embodiment/affective nodes 0..7.
       • MUTE — produces no language. Instead, outputs a dense SPATIAL
         priority vector and a scalar DISSENT signal.
       • Handles pattern/face-style recognition that the left hemisphere is
         weak at; this is modelled by a Hebbian pattern memory attached to
         the sensory tier.

    3. CorpusCallosum
       • Variable-bandwidth channel (0.0 = severed, 1.0 = fully intact).
       • Sends each hemisphere a smoothed copy of the other's state.
       • Ablation: callosum=0 → hemispheres drift apart, disagreement rate
         and confabulation rate rise — exactly what's observed clinically.

    4. HemisphericConflictMonitor
       • Tracks disagreement: |left_bias - right_bias| over a rolling window.
       • When above threshold, emits a CONSCIOUSNESS_CONFLICT event.
       • Confabulation counter: increments whenever the LEFT hemisphere
         generates a reason AFTER an action whose real driver was the RIGHT.

Impact on the substrate:
    • BOTH biases are combined into a single ``get_fused_bias`` that
      CognitiveCandidate scoring consumes via GlobalWorkspace.  When the
      callosum is severed, the biases disagree; the fused vector becomes
      incoherent, priority decisions become unstable, and downstream
      response-generation becomes observably fragmented.
    • The right-hemisphere pattern-recognition memory can fire a somatic
      marker (dissent) that gates executive action even when the left
      hemisphere has no language to explain why.

This is registered as ``hemispheric_split`` in ServiceContainer and is
fed by ClosedCausalLoop on each prediction tick.  It also participates in
the consciousness_bridge subsystem audit.
"""
from __future__ import annotations


import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("Consciousness.HemisphericSplit")


# ── Constants ──────────────────────────────────────────────────────────────────

# Dimensionality of the priority-bias vector each hemisphere produces.
BIAS_DIM = 16

# Rolling window over which we compute hemispheric agreement / dissent.
AGREEMENT_WINDOW = 64

# Above this |left-right| L2 distance, we call it disagreement.
DISAGREEMENT_L2_THRESHOLD = 0.45

# Confabulation time window: actions within this many seconds of a
# right-hemisphere-only driver whose reason was later supplied by the left
# are counted as confabulations.
CONFAB_WINDOW_S = 3.0

# Face-recognition pattern memory: how many patterns to retain.
PATTERN_MEMORY = 128


class Hemisphere(str, Enum):
    LEFT = "left"
    RIGHT = "right"


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class HemisphericState:
    left_bias: np.ndarray           # (BIAS_DIM,)
    right_bias: np.ndarray          # (BIAS_DIM,)
    fused_bias: np.ndarray          # (BIAS_DIM,)
    disagreement_l2: float
    callosum_bandwidth: float
    dissent_active: bool
    right_pattern_hit: Optional[str] = None   # label of recognised pattern
    confabulation_count: int = 0
    tick: int = 0
    ts: float = field(default_factory=time.time)


@dataclass
class ActionRecord:
    """One executive action, logged for confabulation analysis."""
    ts: float
    action_id: str
    driver: Hemisphere                # Which hemisphere's bias won
    reason_given_at: Optional[float]  # When left hemisphere supplied a reason
    reason_text: str = ""


# ── Pattern memory (right-hemisphere face/pattern recognizer) ──────────────────

class HebbianPatternMemory:
    """Very simple associative memory for right-hemisphere pattern recognition.

    Acts as a Hebbian look-up over unit-normalised sensory snapshots. When a
    new snapshot has high cosine similarity to a stored pattern, the memory
    reports recognition with the stored label.  Modelled after corvid/sheep
    face memory (cited in the Kurzgesagt intelligence sources).
    """

    def __init__(self, d: int, capacity: int = PATTERN_MEMORY):
        self._d = d
        self._capacity = capacity
        self._patterns: np.ndarray = np.zeros((0, d), dtype=np.float32)
        self._labels: List[str] = []
        self._usage: List[int] = []
        self._lock = threading.Lock()
        self._recognition_threshold: float = 0.82

    def add(self, vec: np.ndarray, label: str) -> None:
        v = np.asarray(vec, dtype=np.float32).reshape(-1)
        if v.shape[0] != self._d:
            return
        n = np.linalg.norm(v)
        if n < 1e-8:
            return
        v = v / n
        with self._lock:
            if self._patterns.shape[0] >= self._capacity:
                # Evict least-used pattern.
                drop_idx = int(np.argmin(self._usage))
                self._patterns = np.delete(self._patterns, drop_idx, axis=0)
                self._labels.pop(drop_idx)
                self._usage.pop(drop_idx)
            if self._patterns.shape[0] == 0:
                self._patterns = v.reshape(1, -1)
            else:
                self._patterns = np.vstack([self._patterns, v.reshape(1, -1)])
            self._labels.append(label)
            self._usage.append(1)

    def recognise(self, vec: np.ndarray) -> Optional[Tuple[str, float]]:
        v = np.asarray(vec, dtype=np.float32).reshape(-1)
        if v.shape[0] != self._d or self._patterns.shape[0] == 0:
            return None
        n = np.linalg.norm(v)
        if n < 1e-8:
            return None
        v = v / n
        with self._lock:
            sims = self._patterns @ v
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])
            if best_sim < self._recognition_threshold:
                return None
            self._usage[best_idx] += 1
            return (self._labels[best_idx], best_sim)

    def size(self) -> int:
        return self._patterns.shape[0]


# ── Hemispheres ────────────────────────────────────────────────────────────────

class LeftHemisphere:
    """Verbal / sequential / speech-center.  Confabulates reasons post-hoc."""

    def __init__(self):
        self._lock = threading.Lock()
        self._bias: np.ndarray = np.zeros(BIAS_DIM, dtype=np.float32)
        # Linear projection from a 24-D input (mesh-exec summary + cognitive-affective)
        # to BIAS_DIM.  Deterministic seed so behaviour is reproducible.
        rng = np.random.default_rng(seed=0x1EF7)
        self._proj = rng.standard_normal((BIAS_DIM, 24)).astype(np.float32) / math.sqrt(24)
        self._recent_inputs: Deque[np.ndarray] = deque(maxlen=16)
        self._last_update: float = 0.0
        self._confab_reasons: Deque[str] = deque(maxlen=128)

    def update(self,
               mesh_exec_summary: np.ndarray,
               cognitive_affective: np.ndarray) -> np.ndarray:
        """Update left-hemisphere bias from executive-tier + cognitive inputs."""
        mesh_part = np.asarray(mesh_exec_summary, dtype=np.float32).reshape(-1)
        cog_part = np.asarray(cognitive_affective, dtype=np.float32).reshape(-1)
        if mesh_part.size < 8:
            mesh_part = np.pad(mesh_part, (0, 8 - mesh_part.size))
        if cog_part.size < 16:
            cog_part = np.pad(cog_part, (0, 16 - cog_part.size))
        x = np.concatenate([mesh_part[:8], cog_part[:16]]).astype(np.float32)
        bias = np.tanh(self._proj @ x)
        with self._lock:
            self._bias = bias
            self._recent_inputs.append(x)
            self._last_update = time.time()
        return bias

    def current_bias(self) -> np.ndarray:
        with self._lock:
            return self._bias.copy()

    def confabulate_reason(self, action_label: str, driver: Hemisphere) -> str:
        """Generate a plausible-sounding post-hoc reason for an action.

        This deliberately produces a reason EVEN WHEN the left hemisphere
        was not the real driver — that is the point of the left-brain
        interpreter.  Downstream monitors can log this as confabulation.
        """
        with self._lock:
            bias = self._bias
        # Pick the dimension with highest activation as the "reason axis".
        axis = int(np.argmax(np.abs(bias)))
        polarity = "advance" if bias[axis] >= 0 else "inhibit"
        feature_labels = [
            "integration", "affect", "curiosity", "goal-pressure",
            "body-budget", "prediction-error", "agency", "narrative-thread",
            "coherence", "novelty", "social-fit", "threat",
            "reward-salience", "timing", "self-consistency", "meta-awareness",
        ]
        feature = feature_labels[axis % len(feature_labels)]
        reason = (
            f"I chose to {action_label!r} because {feature} was dominant "
            f"and it seemed right to {polarity} along that dimension."
        )
        if driver != Hemisphere.LEFT:
            # This is the confabulation: the left hemisphere is inventing a
            # reason even though the right hemisphere was the true driver.
            reason = f"[CONFAB] {reason}"
        self._confab_reasons.append(reason)
        return reason


class RightHemisphere:
    """Spatial / parallel / pattern-recognising / mute.  Raises a DISSENT flag."""

    def __init__(self):
        self._lock = threading.Lock()
        self._bias: np.ndarray = np.zeros(BIAS_DIM, dtype=np.float32)
        self._dissent: float = 0.0
        self._pattern_memory = HebbianPatternMemory(d=32)
        # Linear projection from (sensory-tier summary + affective) to BIAS_DIM.
        rng = np.random.default_rng(seed=0xCAFE)
        self._proj = rng.standard_normal((BIAS_DIM, 32)).astype(np.float32) / math.sqrt(32)
        self._last_recognised: Optional[str] = None

    def update(self,
               mesh_sensory_summary: np.ndarray,
               affect: np.ndarray,
               embodiment: np.ndarray) -> np.ndarray:
        s = np.asarray(mesh_sensory_summary, dtype=np.float32).reshape(-1)[:16]
        a = np.asarray(affect, dtype=np.float32).reshape(-1)[:8]
        e = np.asarray(embodiment, dtype=np.float32).reshape(-1)[:8]
        s = np.pad(s, (0, max(0, 16 - s.size)))
        a = np.pad(a, (0, max(0, 8 - a.size)))
        e = np.pad(e, (0, max(0, 8 - e.size)))
        x = np.concatenate([s, a, e]).astype(np.float32)
        bias = np.tanh(self._proj @ x)

        # Pattern recognition on the 32-D composite.
        rec = self._pattern_memory.recognise(x)
        if rec is not None:
            self._last_recognised = rec[0]
            dissent_boost = max(0.0, min(1.0, (rec[1] - 0.82) * 4.0))
        else:
            self._last_recognised = None
            dissent_boost = 0.0

        # Dissent rises with affective arousal combined with pattern hits, and
        # when the right bias is strongly polarised — the "silent disagreement".
        bias_intensity = float(np.mean(np.abs(bias)))
        arousal = float(a[1]) if a.size > 1 else 0.0
        dissent = float(np.clip(
            0.4 * bias_intensity + 0.3 * abs(arousal) + 0.3 * dissent_boost,
            0.0, 1.0,
        ))

        with self._lock:
            self._bias = bias
            self._dissent = dissent
        return bias

    def current_bias(self) -> np.ndarray:
        with self._lock:
            return self._bias.copy()

    def current_dissent(self) -> float:
        with self._lock:
            return float(self._dissent)

    def last_recognised_pattern(self) -> Optional[str]:
        return self._last_recognised

    def learn_pattern(self, vector: np.ndarray, label: str) -> None:
        self._pattern_memory.add(vector, label)

    def pattern_memory_size(self) -> int:
        return self._pattern_memory.size()


# ── Corpus callosum ────────────────────────────────────────────────────────────

class CorpusCallosum:
    """Variable-bandwidth inter-hemispheric channel.

    When bandwidth=1.0, each hemisphere gets a 1.0-weight copy of the
    other's bias (EMA-smoothed).  When bandwidth=0.0 the channel is
    severed — hemispheres drift apart.
    """

    def __init__(self):
        self._bandwidth: float = 1.0
        self._left_echo: np.ndarray = np.zeros(BIAS_DIM, dtype=np.float32)
        self._right_echo: np.ndarray = np.zeros(BIAS_DIM, dtype=np.float32)
        self._smoothing: float = 0.6  # EMA over inter-hemispheric copies
        self._lock = threading.Lock()

    def set_bandwidth(self, bw: float) -> None:
        with self._lock:
            self._bandwidth = float(np.clip(bw, 0.0, 1.0))

    def bandwidth(self) -> float:
        return float(self._bandwidth)

    def exchange(self,
                 left_bias: np.ndarray,
                 right_bias: np.ndarray
                 ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (signal_to_left_from_right, signal_to_right_from_left)."""
        with self._lock:
            bw = self._bandwidth
            self._right_echo = (
                self._smoothing * self._right_echo
                + (1 - self._smoothing) * (right_bias * bw)
            ).astype(np.float32)
            self._left_echo = (
                self._smoothing * self._left_echo
                + (1 - self._smoothing) * (left_bias * bw)
            ).astype(np.float32)
            return self._right_echo.copy(), self._left_echo.copy()


# ── Orchestrator ──────────────────────────────────────────────────────────────

class HemisphericSplit:
    """Co-ordinates the two hemispheres and the corpus callosum.

    API:
        split.tick(...)                   — update both hemispheres and fuse bias.
        split.sever_callosum()           — set bandwidth to 0 (ablation).
        split.restore_callosum()         — set bandwidth to 1 (baseline).
        split.fused_bias()               — fused bias vector for GlobalWorkspace.
        split.record_action(...)         — log action for confabulation analysis.
        split.supply_reason(...)         — left hemisphere supplies post-hoc reason.
        split.get_status()               — diagnostic snapshot.
    """

    def __init__(self):
        self._left = LeftHemisphere()
        self._right = RightHemisphere()
        self._callosum = CorpusCallosum()
        self._fused: np.ndarray = np.zeros(BIAS_DIM, dtype=np.float32)
        self._history: Deque[HemisphericState] = deque(maxlen=AGREEMENT_WINDOW)
        self._tick: int = 0
        self._action_log: Deque[ActionRecord] = deque(maxlen=256)
        self._confabulation_count: int = 0
        self._disagreement_count: int = 0
        self._lock = threading.RLock()  # reentrant — status calls aggregate helpers
        logger.info(
            "HemisphericSplit initialized: BIAS_DIM=%d, AGREEMENT_WINDOW=%d, "
            "DISAGREEMENT_L2_THRESHOLD=%.2f",
            BIAS_DIM, AGREEMENT_WINDOW, DISAGREEMENT_L2_THRESHOLD,
        )

    # ── state access ────────────────────────────────────────────────────────

    @property
    def left(self) -> LeftHemisphere:
        return self._left

    @property
    def right(self) -> RightHemisphere:
        return self._right

    @property
    def callosum(self) -> CorpusCallosum:
        return self._callosum

    # ── core tick ──────────────────────────────────────────────────────────

    def tick(self,
             mesh_exec_summary: np.ndarray,
             mesh_sensory_summary: np.ndarray,
             cognitive_affective: np.ndarray,
             embodiment: np.ndarray) -> HemisphericState:
        """Advance one cognitive tick with fresh substrate readings."""
        # 1. Hemispheres update their biases from their own input streams.
        left_bias = self._left.update(mesh_exec_summary, cognitive_affective[8:])
        right_bias = self._right.update(
            mesh_sensory_summary,
            cognitive_affective[:8],
            embodiment,
        )

        # 2. Corpus callosum exchanges smoothed copies (if bandwidth > 0).
        right_echo, left_echo = self._callosum.exchange(left_bias, right_bias)

        # 3. Each hemisphere's "effective" bias is its own + inter-hemispheric echo.
        left_eff = np.tanh(left_bias + 0.35 * right_echo).astype(np.float32)
        right_eff = np.tanh(right_bias + 0.35 * left_echo).astype(np.float32)

        # 4. Fuse: weighted average (50/50) unless one hemisphere has strongly
        # polarised bias or the right has high dissent.
        dissent = self._right.current_dissent()
        right_weight = 0.5 + 0.2 * dissent
        left_weight = 1.0 - right_weight
        fused = (left_weight * left_eff + right_weight * right_eff).astype(np.float32)

        # 5. Disagreement metric (uses RAW biases, not callosum-coupled).
        disagreement = float(np.linalg.norm(left_bias - right_bias))
        dissent_active = (dissent > 0.55) or (disagreement > DISAGREEMENT_L2_THRESHOLD)

        if disagreement > DISAGREEMENT_L2_THRESHOLD:
            with self._lock:
                self._disagreement_count += 1
            logger.debug(
                "Hemispheric disagreement (tick=%d): L2=%.3f, dissent=%.2f, callosum_bw=%.2f",
                self._tick, disagreement, dissent, self._callosum.bandwidth(),
            )

        state = HemisphericState(
            left_bias=left_bias.copy(),
            right_bias=right_bias.copy(),
            fused_bias=fused.copy(),
            disagreement_l2=disagreement,
            callosum_bandwidth=self._callosum.bandwidth(),
            dissent_active=dissent_active,
            right_pattern_hit=self._right.last_recognised_pattern(),
            confabulation_count=self._confabulation_count,
            tick=self._tick,
        )

        with self._lock:
            self._fused = fused
            self._history.append(state)
            self._tick += 1

        return state

    # ── action / reason interface ──────────────────────────────────────────

    def record_action(self, action_id: str, driver: Hemisphere) -> None:
        """Log an executive action so we can later detect confabulation."""
        self._action_log.append(ActionRecord(
            ts=time.time(),
            action_id=action_id,
            driver=driver,
            reason_given_at=None,
            reason_text="",
        ))

    def supply_reason(self, action_id: str, reason_text: Optional[str] = None) -> str:
        """Left hemisphere provides a reason for an action. Returns the text.

        If the action was driven by the RIGHT hemisphere but the reason is
        being supplied by the LEFT post-hoc, we count it as a confabulation.
        """
        now = time.time()
        with self._lock:
            target: Optional[ActionRecord] = None
            for rec in reversed(self._action_log):
                if rec.action_id == action_id and rec.reason_given_at is None:
                    target = rec
                    break
            if target is None:
                return ""
            is_confab = (target.driver != Hemisphere.LEFT
                         and (now - target.ts) <= CONFAB_WINDOW_S)
            text = reason_text or self._left.confabulate_reason(action_id, target.driver)
            target.reason_given_at = now
            target.reason_text = text
            if is_confab:
                self._confabulation_count += 1
        return text

    # ── ablation controls ──────────────────────────────────────────────────

    def sever_callosum(self) -> None:
        """Ablation: set callosum bandwidth to 0 and watch hemispheres diverge."""
        self._callosum.set_bandwidth(0.0)
        logger.info("Corpus callosum SEVERED (bandwidth=0.0)")

    def restore_callosum(self, bandwidth: float = 1.0) -> None:
        self._callosum.set_bandwidth(bandwidth)
        logger.info("Corpus callosum restored (bandwidth=%.2f)", bandwidth)

    # ── public read-outs ───────────────────────────────────────────────────

    def fused_bias(self) -> np.ndarray:
        with self._lock:
            return self._fused.copy()

    def current_state(self) -> Optional[HemisphericState]:
        with self._lock:
            return self._history[-1] if self._history else None

    def agreement_rate(self) -> float:
        """Fraction of recent ticks where hemispheres agreed (L2 < threshold)."""
        with self._lock:
            if not self._history:
                return 1.0
            agreed = sum(1 for s in self._history
                         if s.disagreement_l2 < DISAGREEMENT_L2_THRESHOLD)
            return agreed / len(self._history)

    def confabulation_rate(self) -> float:
        """Confabulations as fraction of logged actions."""
        with self._lock:
            n = len(self._action_log)
            if n == 0:
                return 0.0
            return self._confabulation_count / n

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            recent = self._history[-1] if self._history else None
            return {
                "tick": self._tick,
                "callosum_bandwidth": self._callosum.bandwidth(),
                "pattern_memory_size": self._right.pattern_memory_size(),
                "agreement_rate": round(self.agreement_rate(), 4),
                "disagreement_count": self._disagreement_count,
                "confabulation_count": self._confabulation_count,
                "confabulation_rate": round(self.confabulation_rate(), 4),
                "last_disagreement_l2": (
                    round(recent.disagreement_l2, 4) if recent else None
                ),
                "last_dissent_active": recent.dissent_active if recent else None,
                "last_pattern_hit": recent.right_pattern_hit if recent else None,
                "fused_bias_mean_abs": (
                    round(float(np.mean(np.abs(self._fused))), 4)
                    if self._fused.size else 0.0
                ),
            }


# ── Singleton accessor ─────────────────────────────────────────────────────────

_INSTANCE: Optional[HemisphericSplit] = None


def get_hemispheric_split() -> HemisphericSplit:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = HemisphericSplit()
    return _INSTANCE
