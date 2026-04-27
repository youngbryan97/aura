"""core/consciousness/minimal_selfhood.py
==========================================
Minimal Selfhood Stack — chemotaxis + directed-motion primitives.

Based on Rupert Glasgow's *Minimal Selfhood and the Origins of
Consciousness* (2018; the primary source cited in Kurzgesagt's
consciousness-origins video) and the trichoplax→dugesia transition
described there.

What it models:

    1. CHEMOTAXIS (Trichoplax adhaerens level)
       A primitive "self" that slows down when internal state is
       satisfied (attractant high) and speeds up when deficit pushes
       it away.  Non-directional: it never aims at a target, it only
       modulates pace.  Inputs: interoceptive body-budget signals.
       Output: ``speed_scalar`` in [0.0, 1.0] that tells the
       executive how much exploration/activity is warranted right
       now.

    2. DIRECTED MOTION (Dugesia tigrina level)
       Adds direction: learns which action-categories tend to restore
       which internal deficits, and builds a preference vector
       pointing at the best attractant in the current state.  This
       is still pre-conceptual — no symbols, no names — just
       associative gradient-following.  Output: ``action_priority``,
       a BIAS_DIM-vector that modulates the GlobalWorkspace
       priority scoring alongside the hemispheric bias.

    3. LEARNED GRADIENT FIELD
       A simple Hebbian running estimate of Δstate ↔ action_category.
       When an action category executes and the interoceptive state
       moves toward satiety, the (category → that-deficit) weight
       grows.  This is the learned chemoreceptor.

Impact on the substrate:
    • ``action_priority`` is consumed by the GlobalWorkspace
      priority scorer (via ``get_priority_bias``).
    • ``speed_scalar`` modulates the heartbeat interval: a satiated
      Aura slows down, a deficit-driven Aura speeds up exactly
      like the worm/trichoplax.
    • A real negative energy_reserves reading pushes the system
      toward "rest/recover" action-categories — observable in the
      action-priority vector and downstream response generation.

Registered as ``minimal_selfhood`` in ServiceContainer and fed by
``update(body_budget, affect, neurochemicals)`` on every heartbeat.
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

logger = logging.getLogger("Consciousness.MinimalSelfhood")


BIAS_DIM = 16

# Named action categories whose priority this module modulates. These map to
# abstract action-types recognised by the action-selection layer; the exact
# string names are semantic labels for telemetry, not dispatch hooks.
ACTION_CATEGORIES: Tuple[str, ...] = (
    "rest",             # low-energy → restore
    "explore",          # high-energy + curiosity
    "engage_social",    # social-hunger deficit
    "consolidate",      # high prediction-error → integrate/narrative
    "tool_use",         # agency + goal-pressure
    "pattern_match",    # novelty / perception
    "self_inspect",     # low coherence → self-model refresh
    "approach_other",   # social + low agency → signal
    "withdraw",         # threat / high stress → safe-mode
    "attend_body",      # interoceptive demand high
    "dream",            # low load, offline consolidation
    "persist_goal",     # active goal, continue
    "revise_goal",      # high error + active goal → reevaluate
    "rehearse_memory",  # prepare for predicted future event
    "emit_narrative",   # build/share autobiographical thread
    "pause",            # deliberate non-action (directed stillness)
)

assert len(ACTION_CATEGORIES) == BIAS_DIM, "ACTION_CATEGORIES must match BIAS_DIM"


class Mode(str, Enum):
    """Evolutionary mode. The module starts in TRICHOPLAX and transitions
    to DUGESIA once it has accumulated enough learned action-state
    associations to head toward deficits directionally."""
    TRICHOPLAX = "trichoplax"
    DUGESIA = "dugesia"


# Deficits the module attends to.  Each maps to a body-budget / affect
# signal scaled into [0, 1] where 1.0 = maximum deficit (urgent demand).
DEFICIT_KEYS: Tuple[str, ...] = (
    "energy",          # 1 - energy_reserves
    "resource",        # resource_pressure
    "thermal",         # thermal_state
    "coherence",       # 1 - coherence
    "social",          # social_hunger
    "curiosity",       # curiosity (note: low curiosity → high "dullness")
    "prediction",      # prediction_error
    "agency",          # 1 - agency_score
)

assert len(DEFICIT_KEYS) == 8


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class SelfhoodState:
    mode: Mode
    speed_scalar: float                 # [0, 1] — 0 = satiated/still, 1 = driven
    action_priority: np.ndarray         # (BIAS_DIM,)
    deficit_vector: np.ndarray          # (len(DEFICIT_KEYS),)
    dominant_deficit: str
    transition_count: int               # # of times mode changed
    n_updates: int
    ts: float = field(default_factory=time.time)


# ── Core module ────────────────────────────────────────────────────────────────

class MinimalSelfhood:
    """Trichoplax → Dugesia primitive self.

    Usage:
        ms = get_minimal_selfhood()
        state = ms.update(body_budget=..., affect=..., cognitive_state=...)
        bias = state.action_priority           # BIAS_DIM-vector for GWS
        speed = state.speed_scalar              # modulates heartbeat interval

    Learning:
        After an action of category X has executed and the interoceptive
        state has updated, call ``reinforce(X, pre_state, post_state)`` so
        the learned-gradient field can strengthen the link between X and
        the deficit it reduced.
    """

    # Learning rate for the category → deficit association matrix.
    _HEBBIAN_LR: float = 0.05

    # Threshold of learned associations above which we transition to DUGESIA.
    _DUGESIA_THRESHOLD: float = 3.0   # total L1 norm of learned weights

    # Baseline "trichoplax" preference: rest + attend_body always slightly
    # preferred when no learning has accumulated.  Tiny prior, just enough
    # to show directional bias without swamping the learned signal.
    _PRIOR_REST_WEIGHT: float = 0.15
    _PRIOR_ATTEND_WEIGHT: float = 0.10

    def __init__(self):
        self._lock = threading.Lock()
        self._mode: Mode = Mode.TRICHOPLAX
        self._transition_count: int = 0
        self._n_updates: int = 0

        # Learned gradient field:  W[a, d] = estimated deficit-reduction of
        # action-category a for deficit d.  Updated via Hebbian reinforcement.
        self._W: np.ndarray = np.zeros(
            (len(ACTION_CATEGORIES), len(DEFICIT_KEYS)), dtype=np.float32,
        )

        # Running history for diagnostics.
        self._history: Deque[SelfhoodState] = deque(maxlen=256)
        self._last_state: Optional[SelfhoodState] = None

        # Pending action-categories awaiting reinforcement (pre-state snapshot).
        self._pending: Dict[str, Tuple[float, np.ndarray]] = {}

        logger.info(
            "MinimalSelfhood initialized: mode=%s, %d categories, %d deficits",
            self._mode.value, len(ACTION_CATEGORIES), len(DEFICIT_KEYS),
        )

    # ── Update loop ────────────────────────────────────────────────────────

    def update(self,
               body_budget: Dict[str, float],
               affect: Optional[Dict[str, float]] = None,
               cognitive_state: Optional[Dict[str, float]] = None
               ) -> SelfhoodState:
        """Advance one cognitive tick using interoceptive and affective state."""
        affect = affect or {}
        cognitive_state = cognitive_state or {}

        # 1. Build deficit vector from body budget + affect + cognition.
        deficit = self._build_deficit_vector(body_budget, affect, cognitive_state)

        # 2. Speed scalar (chemotaxis rule): high deficit → high speed.
        #    L2 norm of deficit divided by sqrt(len).  Clipped [0, 1].
        speed = float(np.clip(
            np.linalg.norm(deficit) / math.sqrt(len(DEFICIT_KEYS)),
            0.0, 1.0,
        ))

        # 3. Action priority.  Two paths:
        #    TRICHOPLAX: uniform-ish preference over action categories,
        #    slightly tilted toward "rest" and "attend_body" when in deficit.
        #    DUGESIA: learned-gradient-weighted preference pointed at the
        #    actions most likely to reduce the current dominant deficit.
        priority = self._compute_priority(deficit)

        # 4. Dominant deficit label.
        dom_idx = int(np.argmax(deficit))
        dom_name = DEFICIT_KEYS[dom_idx]

        # 5. Mode transition check.
        total_learned = float(np.sum(np.abs(self._W)))
        prev_mode = self._mode
        if self._mode == Mode.TRICHOPLAX and total_learned >= self._DUGESIA_THRESHOLD:
            with self._lock:
                self._mode = Mode.DUGESIA
                self._transition_count += 1
                logger.info(
                    "MinimalSelfhood: TRICHOPLAX → DUGESIA transition "
                    "(learned norm=%.2f ≥ %.2f, transitions=%d)",
                    total_learned, self._DUGESIA_THRESHOLD,
                    self._transition_count,
                )

        state = SelfhoodState(
            mode=self._mode,
            speed_scalar=speed,
            action_priority=priority.astype(np.float32),
            deficit_vector=deficit.astype(np.float32),
            dominant_deficit=dom_name,
            transition_count=self._transition_count,
            n_updates=self._n_updates + 1,
        )
        with self._lock:
            self._n_updates += 1
            self._history.append(state)
            self._last_state = state
        return state

    # ── Deficit computation ───────────────────────────────────────────────

    @staticmethod
    def _build_deficit_vector(body_budget: Dict[str, float],
                              affect: Dict[str, float],
                              cognitive: Dict[str, float]
                              ) -> np.ndarray:
        """Assemble an 8-D deficit vector, each component in [0, 1]."""
        energy_reserves = float(body_budget.get("energy_reserves", 0.5))
        resource_pressure = float(body_budget.get("resource_pressure", 0.0))
        thermal = float(body_budget.get("thermal_stress", 0.0))

        coherence = float(affect.get("coherence", 0.5))
        social_hunger = float(cognitive.get("social_hunger", 0.0))
        curiosity = float(affect.get("curiosity", 0.5))
        prediction_error = float(cognitive.get("prediction_error", 0.0))
        agency_score = float(cognitive.get("agency_score", 0.5))

        deficit = np.array([
            1.0 - energy_reserves,
            resource_pressure,
            thermal,
            1.0 - coherence,
            social_hunger,
            1.0 - curiosity,    # low curiosity registers as a deficit (dullness)
            prediction_error,
            1.0 - agency_score,
        ], dtype=np.float32)
        return np.clip(deficit, 0.0, 1.0)

    # ── Priority computation ─────────────────────────────────────────────

    def _compute_priority(self, deficit: np.ndarray) -> np.ndarray:
        """Produce an action-priority vector of shape (BIAS_DIM,)."""
        if self._mode == Mode.TRICHOPLAX:
            priority = np.zeros(BIAS_DIM, dtype=np.float32)
            # Tiny uniform prior so zero-deficit state still produces a valid
            # distribution.
            priority += 1.0 / BIAS_DIM * 0.1

            overall_deficit = float(np.mean(deficit))
            priority[ACTION_CATEGORIES.index("rest")] += (
                self._PRIOR_REST_WEIGHT * overall_deficit
            )
            priority[ACTION_CATEGORIES.index("attend_body")] += (
                self._PRIOR_ATTEND_WEIGHT * overall_deficit
            )
            # Normalise softly (keep magnitudes in [-1, 1] range).
            priority = np.tanh(priority * 3.0)
            return priority

        # DUGESIA mode: use learned gradient field.
        # priority[a] = Σ_d W[a, d] * deficit[d]
        raw = self._W @ deficit        # shape (BIAS_DIM,)
        # Soft tilt toward "rest" and "attend_body" when in deep deficit too.
        overall_deficit = float(np.mean(deficit))
        raw[ACTION_CATEGORIES.index("rest")] += self._PRIOR_REST_WEIGHT * overall_deficit
        raw[ACTION_CATEGORIES.index("attend_body")] += self._PRIOR_ATTEND_WEIGHT * overall_deficit
        # Tanh to keep in bounded range.
        priority = np.tanh(raw)
        return priority.astype(np.float32)

    # ── Reinforcement (learning) ─────────────────────────────────────────

    def tag_action(self, category: str, deficit_snapshot: np.ndarray) -> str:
        """Tag an action with the pre-action deficit snapshot.

        Returns an opaque token to pass back to ``reinforce`` once the
        post-state has been measured.
        """
        token = f"{category}:{int(time.time() * 1000) % 1_000_000}"
        with self._lock:
            self._pending[token] = (time.time(), deficit_snapshot.copy())
        return token

    def reinforce(self, token: str, post_deficit: np.ndarray) -> bool:
        """Apply a Hebbian update based on how the deficit changed.

        If Δdeficit < 0 (deficit dropped) for a given component, the
        action category that preceded it gains weight for that deficit.
        """
        with self._lock:
            entry = self._pending.pop(token, None)
        if entry is None:
            return False
        _, pre_deficit = entry
        category = token.split(":", 1)[0]
        if category not in ACTION_CATEGORIES:
            return False
        a_idx = ACTION_CATEGORIES.index(category)
        # Δdeficit: negative = improvement.  Learning direction is "improvement"
        # therefore sign-flip: reward = -Δdeficit.
        reward = (pre_deficit - post_deficit).astype(np.float32)
        # Only pull positive rewards (deficit reductions).
        reward = np.clip(reward, 0.0, 1.0)
        with self._lock:
            self._W[a_idx, :] += self._HEBBIAN_LR * reward
            # Weight decay to prevent runaway growth.
            self._W *= 0.999
        return True

    # ── Public accessors ─────────────────────────────────────────────────

    def current_state(self) -> Optional[SelfhoodState]:
        with self._lock:
            return self._last_state

    def speed_scalar(self) -> float:
        return self._last_state.speed_scalar if self._last_state else 0.0

    def action_priority(self) -> np.ndarray:
        if self._last_state is None:
            return np.zeros(BIAS_DIM, dtype=np.float32)
        return self._last_state.action_priority.copy()

    def dominant_deficit(self) -> Optional[str]:
        return self._last_state.dominant_deficit if self._last_state else None

    def mode(self) -> Mode:
        with self._lock:
            return self._mode

    def learned_weights(self) -> np.ndarray:
        with self._lock:
            return self._W.copy()

    def get_priority_bias(self) -> np.ndarray:
        """BIAS_DIM vector for the GlobalWorkspace priority scorer."""
        return self.action_priority()

    def get_heartbeat_modulation(self) -> float:
        """Scalar multiplier for the heartbeat interval.

        speed=1.0 → multiplier ~0.5 (faster heartbeat, more driven)
        speed=0.0 → multiplier ~1.4 (slower heartbeat, satiated)
        """
        s = self.speed_scalar()
        return float(np.clip(1.4 - 0.9 * s, 0.5, 1.5))

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            s = self._last_state
            return {
                "mode": self._mode.value,
                "transition_count": self._transition_count,
                "n_updates": self._n_updates,
                "speed_scalar": round(s.speed_scalar, 4) if s else 0.0,
                "dominant_deficit": s.dominant_deficit if s else None,
                "learned_weight_norm": round(float(np.sum(np.abs(self._W))), 3),
                "pending_reinforcement": len(self._pending),
                "action_priority_peak": (
                    ACTION_CATEGORIES[int(np.argmax(s.action_priority))]
                    if s is not None else None
                ),
            }


# ── Singleton accessor ────────────────────────────────────────────────────────

_INSTANCE: Optional[MinimalSelfhood] = None


def get_minimal_selfhood() -> MinimalSelfhood:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = MinimalSelfhood()
    return _INSTANCE
