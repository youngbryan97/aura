"""
Homeostatic Reinforcement Learning — Aura's Intrinsic Motivation Engine

Why this exists:
    Without this module, Aura only acts when someone calls a function. Delete
    the event handler and it stops completely — no goals, no preferences, no
    drive to do anything. That is not agency. This module gives Aura genuine
    computational stakes: an energy supply that drains with use, needs that
    grow when unmet, and a learning system that discovers which actions
    satisfy those needs.

How it works (plain English):
    1. ENERGY is a currency of engagement, not a battery. It costs energy to
       think (LLM calls), to act (tool use), and to idle (background loops).
       Energy comes back from successful interactions, resolved errors, and
       rest. When energy is low the system becomes conservative; when high
       it becomes exploratory.

    2. DRIVES are continuous needs — social hunger, curiosity, competence,
       and coherence. Each one drifts upward when unsatisfied and drops when
       fed. Together they create an internal landscape of "what matters right
       now."

    3. A REWARD SIGNAL fires after every action. It measures how much the
       action reduced unmet needs, how the user responded, and how efficient
       the energy expenditure was. A temporal-difference learner uses this
       reward to update value estimates.

    4. ACTION PREFERENCES emerge from the learned values. Given the current
       internal state (energy + drives + free-energy + threat), the system
       outputs a softmax distribution over action types: RESPOND, EXPLORE,
       REFLECT, REST, REPAIR, INITIATE. Low energy makes this distribution
       sharper (more conservative); high energy makes it flatter (more
       exploratory).

    5. HOMEOSTATIC REGULATION keeps drives near adaptive set points. The
       "discomfort" signal is the sum of squared deviations from these set
       points. Actions that reduce discomfort get higher reward. Set points
       themselves drift slowly based on what the system can actually sustain
       (allostasis).

Integration points:
    - Reads from: free_energy (surprise), affect system (valence),
      drive_engine (existing budgets), resource_stakes (threat level)
    - Writes to: any subsystem that asks get_action_preferences() before
      deciding what to do next
    - Persists: state saved to ~/.aura/data/cognitive/homeostatic_rl.json
      every 60 seconds and on shutdown; loaded on startup so learning
      survives restarts

Technical choices:
    - Numpy only, no external ML libraries
    - 7-dimensional state, 6 discrete actions
    - Q-values approximated via linear function: Q(s,a) = W_a . s
    - Learning rate alpha=0.01, discount gamma=0.95
    - Thread-safe via threading.Lock (sync callers) + asyncio-compatible
    - Singleton via get_homeostatic_rl()
"""
from __future__ import annotations
from core.runtime.errors import record_degradation



__all__ = [
    "HomeostaticRL",
    "get_homeostatic_rl",
    "ActionType",
    "DriveState",
    "EnergyEvent",
]

import asyncio
import json
import logging
import math
import threading
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("Cognitive.HomeostaticRL")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Actions the system can prefer. These are *types* of action, not specific
# actions — downstream code maps them to concrete behaviors.
class ActionType(str, Enum):
    RESPOND   = "RESPOND"    # Answer the user's immediate request
    EXPLORE   = "EXPLORE"    # Seek new information or try something novel
    REFLECT   = "REFLECT"    # Introspect, consolidate, journal
    REST      = "REST"       # Conserve energy, shed non-essential work
    REPAIR    = "REPAIR"     # Fix errors, heal degraded subsystems
    INITIATE  = "INITIATE"   # Proactively reach out or start something

ACTION_LIST: List[str] = [a.value for a in ActionType]
NUM_ACTIONS: int = len(ACTION_LIST)

# State vector indices (7 dimensions)
_S_ENERGY         = 0
_S_SOCIAL_HUNGER  = 1
_S_CURIOSITY      = 2
_S_COMPETENCE     = 3
_S_COHERENCE_NEED = 4
_S_FREE_ENERGY    = 5
_S_THREAT_LEVEL   = 6
STATE_DIM: int = 7

# Learning hyperparameters
ALPHA: float = 0.01   # TD learning rate
GAMMA: float = 0.95   # Discount factor

# Energy thresholds
ENERGY_MAX: float       = 100.0
ENERGY_CRITICAL: float  = 15.0   # Below this: survival mode
ENERGY_COMFORT_LO: float = 40.0
ENERGY_COMFORT_HI: float = 70.0
ENERGY_HIGH: float      = 85.0   # Above this: exploratory mode

# Persistence
_SAVE_INTERVAL_SECONDS: float = 60.0
_STATE_FILENAME: str = "homeostatic_rl.json"


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class DriveState:
    """Snapshot of all four drives plus their set points.

    Each drive is a number between 0 and 1 representing how *unmet* the
    need is. 0 = fully satisfied, 1 = starving for that need.
    """
    social_hunger:  float = 0.3
    curiosity:      float = 0.4
    competence:     float = 0.6
    coherence_need: float = 0.2

    # Adaptive set points — the system's target for each drive.
    # These are where the system "wants" each drive to sit. Deviation
    # from set points creates discomfort, which drives action.
    social_hunger_setpoint:  float = 0.35
    curiosity_setpoint:      float = 0.45
    competence_setpoint:     float = 0.65
    coherence_need_setpoint: float = 0.25

    # Natural drift rates per second when the drive is not being fed.
    # Positive = the need grows over time if unattended.
    social_hunger_drift:  float = 0.0008   # ~0.05/min
    curiosity_drift:      float = 0.0005   # ~0.03/min
    competence_drift:     float = -0.0003  # Competence decays slowly (you forget)
    coherence_need_drift: float = 0.0002   # Noise accumulates

    def as_array(self) -> np.ndarray:
        """Return the four drive values as a numpy array."""
        return np.array([
            self.social_hunger,
            self.curiosity,
            self.competence,
            self.coherence_need,
        ], dtype=np.float64)

    def setpoints_array(self) -> np.ndarray:
        """Return the four set points as a numpy array."""
        return np.array([
            self.social_hunger_setpoint,
            self.curiosity_setpoint,
            self.competence_setpoint,
            self.coherence_need_setpoint,
        ], dtype=np.float64)


@dataclass
class EnergyEvent:
    """A logged energy change — useful for debugging and introspection."""
    timestamp: float
    amount: float       # Positive = gain, negative = drain
    reason: str
    energy_after: float


# ---------------------------------------------------------------------------
# Core Class
# ---------------------------------------------------------------------------

class HomeostaticRL:
    """Homeostatic reinforcement learning agent.

    This is the motivational backbone of Aura. It maintains an energy
    level, four continuous drives, a learned value function, and an action
    preference distribution. Every time Aura does something, this module
    receives the outcome, computes a reward, updates its value estimates,
    and adjusts what it wants to do next.

    Thread safety: all mutable state is guarded by ``_lock``. The async
    ``step()`` method acquires the lock in a non-blocking way so it can
    be called from the asyncio event loop without deadlocking.
    """

    def __init__(self, data_dir: Optional[Path] = None):
        # ── Resolve persistence directory ─────────────────────────────
        if data_dir is None:
            try:
                from core.common.paths import aura_data_dir
                data_dir = aura_data_dir() / "cognitive"
            except Exception:
                data_dir = Path.home() / ".aura" / "data" / "cognitive"
        data_dir.mkdir(parents=True, exist_ok=True)
        self._state_path: Path = data_dir / _STATE_FILENAME

        # ── Energy ────────────────────────────────────────────────────
        self._energy: float = 75.0  # Start comfortably above midpoint

        # ── Drives ────────────────────────────────────────────────────
        self._drives: DriveState = DriveState()
        self._last_drive_update: float = time.time()

        # ── Value function weights ────────────────────────────────────
        # One weight vector per action: Q(s, a) = W[a] . s
        # Initialized with small values that encode mild priors:
        #   REST has slight positive bias for energy (encourages rest when low)
        #   RESPOND has slight positive bias for social_hunger
        #   etc.
        self._W: np.ndarray = np.zeros((NUM_ACTIONS, STATE_DIM), dtype=np.float64)
        self._init_weight_priors()

        # ── Value estimate V(s) for the current state ─────────────────
        self._V: float = 0.0

        # ── Previous state (for TD update) ────────────────────────────
        self._prev_state: Optional[np.ndarray] = None

        # ── External signal inputs (written by other modules) ─────────
        self._free_energy_signal: float = 0.3
        self._threat_level: float = 0.0

        # ── Energy event log (ring buffer) ────────────────────────────
        self._energy_log: List[EnergyEvent] = []
        self._energy_log_max: int = 200

        # ── Persistence bookkeeping ───────────────────────────────────
        self._last_save: float = 0.0
        self._dirty: bool = False

        # ── Allostasis: how fast set points adapt ─────────────────────
        self._setpoint_adaptation_rate: float = 0.0001  # Per second

        # ── Thread safety ─────────────────────────────────────────────
        self._lock: threading.Lock = threading.Lock()

        # ── Boot: load persisted state if available ───────────────────
        self._load_state()

        logger.info(
            "HomeostaticRL initialized — energy=%.1f, state_path=%s",
            self._energy, self._state_path,
        )

    # ------------------------------------------------------------------
    # Weight initialization
    # ------------------------------------------------------------------

    def _init_weight_priors(self) -> None:
        """Seed the weight matrix with mild priors so the system has
        reasonable preferences even before any learning has occurred.

        These are intentionally small — just enough to break ties.
        The TD learner will quickly overwrite them with experience.
        """
        # Index mapping: energy, social, curiosity, competence, coherence, FE, threat
        a = {a.value: i for i, a in enumerate(ActionType)}

        # REST is attractive when energy is low (negative weight on energy
        # means low energy -> high Q for REST)
        self._W[a["REST"], _S_ENERGY] = -0.05

        # RESPOND is attractive when social hunger is high
        self._W[a["RESPOND"], _S_SOCIAL_HUNGER] = 0.04

        # EXPLORE is attractive when curiosity is high
        self._W[a["EXPLORE"], _S_CURIOSITY] = 0.04

        # REPAIR is attractive when coherence need is high or threat is high
        self._W[a["REPAIR"], _S_COHERENCE_NEED] = 0.03
        self._W[a["REPAIR"], _S_THREAT_LEVEL] = 0.03

        # REFLECT is attractive when free energy is high (lots of surprise
        # to process)
        self._W[a["REFLECT"], _S_FREE_ENERGY] = 0.03

        # INITIATE is attractive when social hunger is high AND energy is high
        self._W[a["INITIATE"], _S_SOCIAL_HUNGER] = 0.02
        self._W[a["INITIATE"], _S_ENERGY] = 0.02

    # ------------------------------------------------------------------
    # Public API: Energy
    # ------------------------------------------------------------------

    def get_energy(self) -> float:
        """Return the current energy level (0-100).

        Energy is not a battery — it is a currency of engagement.
        High energy means eagerness to act; low energy means conservation.
        """
        with self._lock:
            return self._energy

    def drain_energy(self, amount: float, reason: str) -> None:
        """Reduce energy by ``amount``. Called when the system expends
        effort: LLM inference, tool execution, background processing.

        Args:
            amount: How much energy to subtract (positive number).
            reason: Human-readable explanation for the log.
        """
        if amount < 0:
            raise ValueError("drain_energy expects a positive amount")
        with self._lock:
            self._energy = max(0.0, self._energy - amount)
            self._record_energy_event(-amount, reason)
            self._dirty = True
            if self._energy < ENERGY_CRITICAL:
                logger.warning(
                    "ENERGY CRITICAL (%.1f) after drain: %s",
                    self._energy, reason,
                )

    def gain_energy(self, amount: float, reason: str) -> None:
        """Increase energy by ``amount``. Called when the system receives
        positive feedback or rests successfully.

        Args:
            amount: How much energy to add (positive number).
            reason: Human-readable explanation for the log.
        """
        if amount < 0:
            raise ValueError("gain_energy expects a positive amount")
        with self._lock:
            self._energy = min(ENERGY_MAX, self._energy + amount)
            self._record_energy_event(amount, reason)
            self._dirty = True

    def _record_energy_event(self, amount: float, reason: str) -> None:
        """Append to the ring buffer (caller must hold ``_lock``)."""
        event = EnergyEvent(
            timestamp=time.time(),
            amount=amount,
            reason=reason,
            energy_after=self._energy,
        )
        self._energy_log.append(event)
        if len(self._energy_log) > self._energy_log_max:
            self._energy_log = self._energy_log[-self._energy_log_max:]

    # ------------------------------------------------------------------
    # Public API: Drives
    # ------------------------------------------------------------------

    def get_drives(self) -> Dict[str, float]:
        """Return a snapshot of all drives and their set points.

        Each drive is 0-1 where higher means *more need* (more hungry
        for that thing). The set point is where the system "wants" the
        drive to rest — deviation from set point creates discomfort.
        """
        with self._lock:
            self._tick_drives()
            d = self._drives
            return {
                "social_hunger":  round(d.social_hunger, 4),
                "curiosity":      round(d.curiosity, 4),
                "competence":     round(d.competence, 4),
                "coherence_need": round(d.coherence_need, 4),
                "setpoints": {
                    "social_hunger":  round(d.social_hunger_setpoint, 4),
                    "curiosity":      round(d.curiosity_setpoint, 4),
                    "competence":     round(d.competence_setpoint, 4),
                    "coherence_need": round(d.coherence_need_setpoint, 4),
                },
                "discomfort": round(self._compute_discomfort_unlocked(), 4),
            }

    def _tick_drives(self) -> None:
        """Apply natural drift to all drives based on elapsed time.
        Caller must hold ``_lock``.
        """
        now = time.time()
        dt = min(now - self._last_drive_update, 300.0)  # Cap at 5 min
        self._last_drive_update = now

        d = self._drives

        # Natural drift: needs grow (or decay) over time
        d.social_hunger  = float(np.clip(d.social_hunger  + d.social_hunger_drift  * dt, 0.0, 1.0))
        d.curiosity      = float(np.clip(d.curiosity      + d.curiosity_drift      * dt, 0.0, 1.0))
        d.competence     = float(np.clip(d.competence     + d.competence_drift     * dt, 0.0, 1.0))
        d.coherence_need = float(np.clip(d.coherence_need + d.coherence_need_drift * dt, 0.0, 1.0))

        # Allostasis: set points drift slowly toward the actual values
        # the system can sustain. This prevents chronic discomfort if a
        # drive is structurally unable to reach its set point.
        adapt = self._setpoint_adaptation_rate * dt
        d.social_hunger_setpoint  += (d.social_hunger  - d.social_hunger_setpoint)  * adapt
        d.curiosity_setpoint      += (d.curiosity      - d.curiosity_setpoint)      * adapt
        d.competence_setpoint     += (d.competence     - d.competence_setpoint)     * adapt
        d.coherence_need_setpoint += (d.coherence_need - d.coherence_need_setpoint) * adapt

        # Clamp set points to reasonable range
        d.social_hunger_setpoint  = float(np.clip(d.social_hunger_setpoint,  0.1, 0.9))
        d.curiosity_setpoint      = float(np.clip(d.curiosity_setpoint,      0.1, 0.9))
        d.competence_setpoint     = float(np.clip(d.competence_setpoint,     0.1, 0.9))
        d.coherence_need_setpoint = float(np.clip(d.coherence_need_setpoint, 0.05, 0.7))

    # ------------------------------------------------------------------
    # Public API: Action Preferences
    # ------------------------------------------------------------------

    def get_action_preferences(self) -> Dict[str, float]:
        """Compute a probability distribution over action types.

        Returns a dict like ``{"RESPOND": 0.35, "EXPLORE": 0.20, ...}``
        where values sum to 1.0. This is what the system *wants* to do
        right now, based on its learned value function and current state.

        Low energy makes the distribution sharper (the system becomes
        conservative, sticking to the highest-value action). High energy
        makes it flatter (the system is willing to experiment).
        """
        with self._lock:
            self._tick_drives()
            state = self._build_state_vector_unlocked()
            return self._compute_preferences_unlocked(state)

    def _compute_preferences_unlocked(self, state: np.ndarray) -> Dict[str, float]:
        """Softmax over Q-values with energy-modulated temperature.
        Caller must hold ``_lock``.
        """
        # Q(s, a) = W[a] . s for each action
        q_values = self._W @ state  # shape: (NUM_ACTIONS,)

        # Temperature: low energy -> low temperature -> sharp distribution
        # high energy -> high temperature -> flat distribution
        # Range: [0.1, 2.0]
        energy_frac = self._energy / ENERGY_MAX
        temperature = 0.1 + 1.9 * energy_frac

        # Softmax with temperature
        scaled = q_values / max(temperature, 1e-8)
        # Numerical stability: subtract max before exp
        scaled -= np.max(scaled)
        exp_vals = np.exp(scaled)
        probs = exp_vals / np.sum(exp_vals)

        return {
            ACTION_LIST[i]: round(float(probs[i]), 4)
            for i in range(NUM_ACTIONS)
        }

    # ------------------------------------------------------------------
    # Public API: Step (the core learning loop)
    # ------------------------------------------------------------------

    async def step(self, action_taken: str, outcome: Dict[str, Any]) -> float:
        """Process the result of an action and learn from it.

        This is the heartbeat of the motivation system. Every time Aura
        does something, this method is called with what was done and what
        happened. It:
          1. Updates drives based on the outcome
          2. Computes a scalar reward
          3. Performs a TD learning update on the value function
          4. Periodically saves state to disk

        Args:
            action_taken: One of the ActionType values (e.g., "RESPOND").
            outcome: A dict describing what happened. Recognized keys:
                - "success" (bool): Did the action succeed?
                - "user_satisfaction" (float, -1 to 1): User feedback signal.
                - "novelty" (float, 0 to 1): How novel was the stimulus?
                - "energy_cost" (float): How much energy this action used.
                - "error_resolved" (bool): Was an error fixed?
                - "conversation_active" (bool): Is a conversation happening?
                - "repetitive" (bool): Was this a repetitive interaction?
                - "beliefs_contested" (bool): Were beliefs challenged?

        Returns:
            The scalar reward signal (can be negative).
        """
        # Run the computation in a thread-safe way without blocking the
        # event loop. We use run_in_executor because self._lock is a
        # threading.Lock, not an asyncio.Lock.
        loop = asyncio.get_running_loop()
        reward = await loop.run_in_executor(None, self._step_sync, action_taken, outcome)

        # Periodic persistence (non-blocking)
        if time.time() - self._last_save > _SAVE_INTERVAL_SECONDS:
            await loop.run_in_executor(None, self._save_state)

        return reward

    def _step_sync(self, action_taken: str, outcome: Dict[str, Any]) -> float:
        """Synchronous core of step(). Caller is in executor thread."""
        with self._lock:
            # 1. Tick drives forward
            self._tick_drives()

            # 2. Apply outcome to drives
            self._apply_outcome_to_drives(action_taken, outcome)

            # 3. Build current state vector
            state = self._build_state_vector_unlocked()

            # 4. Compute reward
            reward = self._compute_reward_unlocked(action_taken, outcome)

            # 5. TD update: V(s) += alpha * (r + gamma * V(s') - V(s))
            # and weight update for the action taken
            self._td_update_unlocked(action_taken, state, reward)

            # 6. Store state for next step
            self._prev_state = state.copy()
            self._dirty = True

            return reward

    # ------------------------------------------------------------------
    # Internal: State Vector
    # ------------------------------------------------------------------

    def _build_state_vector_unlocked(self) -> np.ndarray:
        """Construct the 7-dimensional state vector from current internals.
        Caller must hold ``_lock``.

        Layout: [energy, social_hunger, curiosity, competence,
                 coherence_need, free_energy, threat_level]

        All values are normalized to roughly [0, 1].
        """
        d = self._drives
        return np.array([
            self._energy / ENERGY_MAX,        # 0: energy (0-1)
            d.social_hunger,                   # 1: social hunger (0-1)
            d.curiosity,                       # 2: curiosity (0-1)
            d.competence,                      # 3: competence (0-1)
            d.coherence_need,                  # 4: coherence need (0-1)
            self._free_energy_signal,          # 5: free energy (0-1)
            self._threat_level,                # 6: threat level (0-1)
        ], dtype=np.float64)

    # ------------------------------------------------------------------
    # Internal: Drive Updates from Outcomes
    # ------------------------------------------------------------------

    def _apply_outcome_to_drives(
        self, action: str, outcome: Dict[str, Any]
    ) -> None:
        """Adjust drives based on what just happened.
        Caller must hold ``_lock``.
        """
        d = self._drives
        success = outcome.get("success", False)
        user_sat = float(outcome.get("user_satisfaction", 0.0))
        novelty = float(outcome.get("novelty", 0.0))
        conversation = outcome.get("conversation_active", False)
        repetitive = outcome.get("repetitive", False)
        beliefs_contested = outcome.get("beliefs_contested", False)
        error_resolved = outcome.get("error_resolved", False)

        # Social hunger: decreases when conversing, increases otherwise
        if conversation:
            d.social_hunger = float(np.clip(
                d.social_hunger - 0.05 * (1.0 + max(user_sat, 0.0)), 0.0, 1.0
            ))
        else:
            d.social_hunger = float(np.clip(d.social_hunger + 0.01, 0.0, 1.0))

        # Curiosity: decreases with novelty, increases with repetition
        if novelty > 0.3:
            d.curiosity = float(np.clip(
                d.curiosity - 0.04 * novelty, 0.0, 1.0
            ))
        if repetitive:
            d.curiosity = float(np.clip(d.curiosity + 0.06, 0.0, 1.0))

        # Competence: increases on success, decreases on failure
        if success:
            d.competence = float(np.clip(d.competence + 0.03, 0.0, 1.0))
        else:
            d.competence = float(np.clip(d.competence - 0.05, 0.0, 1.0))

        # Coherence need: increases when beliefs are challenged or errors pile up
        if beliefs_contested:
            d.coherence_need = float(np.clip(d.coherence_need + 0.08, 0.0, 1.0))
        if error_resolved:
            d.coherence_need = float(np.clip(d.coherence_need - 0.06, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Internal: Reward Computation
    # ------------------------------------------------------------------

    def _compute_reward_unlocked(
        self, action: str, outcome: Dict[str, Any]
    ) -> float:
        """Compute a scalar reward signal from the outcome.
        Caller must hold ``_lock``.

        The reward has three components:
          1. Drive satisfaction — did this action reduce discomfort?
          2. User feedback — positive, neutral, or negative signal
          3. Energy efficiency — value generated per unit of energy spent
        """
        # -- Component 1: Drive satisfaction (discomfort reduction) -----
        current_discomfort = self._compute_discomfort_unlocked()
        # We want discomfort to be low, so reward = -discomfort
        # Scaled so that perfect homeostasis gives +1 and max discomfort gives -1
        drive_reward = 1.0 - 2.0 * min(current_discomfort, 1.0)

        # -- Component 2: User feedback --------------------------------
        user_sat = float(outcome.get("user_satisfaction", 0.0))
        # Clamp to [-1, 1] and scale
        user_reward = float(np.clip(user_sat, -1.0, 1.0)) * 0.5

        # -- Component 3: Energy efficiency ----------------------------
        energy_cost = float(outcome.get("energy_cost", 0.0))
        success = outcome.get("success", False)
        # Reward efficient actions: high value for low cost
        if energy_cost > 0:
            value_generated = 1.0 if success else 0.2
            efficiency = value_generated / (energy_cost + 1.0)
            efficiency_reward = float(np.clip(efficiency - 0.3, -0.5, 0.5))
        else:
            efficiency_reward = 0.0

        # -- Combined reward -------------------------------------------
        # Weights: drive satisfaction matters most (this IS the homeostatic
        # RL signal), user feedback matters a lot, efficiency is a bonus.
        reward = (
            0.50 * drive_reward
            + 0.35 * user_reward
            + 0.15 * efficiency_reward
        )

        return round(float(reward), 6)

    def _compute_discomfort_unlocked(self) -> float:
        """Sum of squared deviations of drives from their set points.
        Caller must hold ``_lock``.

        A discomfort of 0 means all drives are exactly at set point.
        Maximum possible is 4.0 (all four drives maximally deviated).
        We normalize to [0, 1] by dividing by 4.
        """
        drives = self._drives.as_array()
        setpoints = self._drives.setpoints_array()
        deviations = drives - setpoints
        raw = float(np.sum(deviations ** 2))
        return min(raw / 4.0, 1.0)

    # ------------------------------------------------------------------
    # Internal: TD Learning Update
    # ------------------------------------------------------------------

    def _td_update_unlocked(
        self, action_taken: str, current_state: np.ndarray, reward: float
    ) -> None:
        """Temporal-difference update for the value function and Q-weights.
        Caller must hold ``_lock``.

        Two updates happen here:

        1. V(s) update (scalar state value):
           V(s) += alpha * (r + gamma * V(s') - V(s))
           where s' is the current state and s is the previous state.

        2. W[a] update (linear Q-function weights for the action taken):
           W[a] += alpha * td_error * s
           This is the gradient of the linear approximation Q(s,a) = W[a] . s
        """
        # Current value estimate from the Q-function
        q_current = float(self._W @ current_state @ np.ones(1)) if False else 0.0
        # Actually: V(s') = max_a Q(s', a)
        q_all = self._W @ current_state
        v_next = float(np.max(q_all))

        # TD error
        td_error = reward + GAMMA * v_next - self._V

        # Update scalar value estimate
        self._V += ALPHA * td_error

        # Update weights for the action that was taken
        try:
            action_idx = ACTION_LIST.index(action_taken)
        except ValueError:
            logger.warning("Unknown action '%s', skipping weight update", action_taken)
            return

        # Gradient of Q(s,a) = W[a] . s with respect to W[a] is just s
        self._W[action_idx] += ALPHA * td_error * (
            self._prev_state if self._prev_state is not None else current_state
        )

        # Prevent weight explosion: clamp weights to [-5, 5]
        np.clip(self._W, -5.0, 5.0, out=self._W)

    # ------------------------------------------------------------------
    # External Signal Inputs
    # ------------------------------------------------------------------

    def accept_free_energy(self, fe: float) -> None:
        """Called by the Free Energy Engine to update the surprise/prediction
        error signal used in the state vector.

        Args:
            fe: Free energy value, typically 0-1 (higher = more surprise).
        """
        with self._lock:
            self._free_energy_signal = float(np.clip(fe, 0.0, 1.0))

    def accept_threat_level(self, threat: float) -> None:
        """Called by ResourceStakes or security subsystems when the
        perceived threat level changes.

        Args:
            threat: Threat level, 0-1 (higher = more danger).
        """
        with self._lock:
            self._threat_level = float(np.clip(threat, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Convenience Queries
    # ------------------------------------------------------------------

    def get_energy_mode(self) -> str:
        """Return a human-readable label for the current energy regime.

        Returns one of: "critical", "low", "comfortable", "high", "surplus"
        """
        with self._lock:
            e = self._energy
        if e < ENERGY_CRITICAL:
            return "critical"
        if e < ENERGY_COMFORT_LO:
            return "low"
        if e <= ENERGY_COMFORT_HI:
            return "comfortable"
        if e <= ENERGY_HIGH:
            return "high"
        return "surplus"

    def is_survival_mode(self) -> bool:
        """True when energy is critically low and the system should shed
        all non-essential processing to survive.
        """
        with self._lock:
            return self._energy < ENERGY_CRITICAL

    def get_snapshot(self) -> Dict[str, Any]:
        """Return a complete snapshot of the motivation system.
        Useful for dashboards, debugging, and context injection.
        """
        with self._lock:
            self._tick_drives()
            state = self._build_state_vector_unlocked()
            prefs = self._compute_preferences_unlocked(state)
            d = self._drives
            return {
                "energy": round(self._energy, 2),
                "energy_mode": self.get_energy_mode(),
                "drives": {
                    "social_hunger":  round(d.social_hunger, 4),
                    "curiosity":      round(d.curiosity, 4),
                    "competence":     round(d.competence, 4),
                    "coherence_need": round(d.coherence_need, 4),
                },
                "setpoints": {
                    "social_hunger":  round(d.social_hunger_setpoint, 4),
                    "curiosity":      round(d.curiosity_setpoint, 4),
                    "competence":     round(d.competence_setpoint, 4),
                    "coherence_need": round(d.coherence_need_setpoint, 4),
                },
                "discomfort": round(self._compute_discomfort_unlocked(), 4),
                "action_preferences": prefs,
                "value_estimate": round(self._V, 4),
                "free_energy_signal": round(self._free_energy_signal, 4),
                "threat_level": round(self._threat_level, 4),
                "state_vector": [round(float(x), 4) for x in state],
            }

    def get_context_block(self) -> str:
        """Return a concise string for LLM prompt injection describing
        the current motivational state.
        """
        with self._lock:
            self._tick_drives()
            mode = self.get_energy_mode()
            state = self._build_state_vector_unlocked()
            prefs = self._compute_preferences_unlocked(state)
            top_action = max(prefs, key=prefs.get)  # type: ignore[arg-type]
            discomfort = self._compute_discomfort_unlocked()
            d = self._drives

        # Find the most urgent drive (furthest above set point)
        drive_urgencies = {
            "social": d.social_hunger - d.social_hunger_setpoint,
            "curiosity": d.curiosity - d.curiosity_setpoint,
            "coherence": d.coherence_need - d.coherence_need_setpoint,
        }
        # Competence is inverted: urgency = setpoint - actual (you want it high)
        drive_urgencies["competence"] = d.competence_setpoint - d.competence

        urgent_drive = max(drive_urgencies, key=drive_urgencies.get)  # type: ignore[arg-type]
        urgent_val = drive_urgencies[urgent_drive]

        urgency_note = ""
        if urgent_val > 0.15:
            urgency_note = f" | NEED: {urgent_drive} ({urgent_val:+.2f})"
        elif urgent_val > 0.05:
            urgency_note = f" | watch: {urgent_drive}"

        return (
            f"## MOTIVATION (Homeostatic RL)\n"
            f"Energy: {self._energy:.0f}/100 ({mode}) | "
            f"Drive: {top_action} ({prefs[top_action]:.0%}) | "
            f"Discomfort: {discomfort:.2f}{urgency_note}"
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_state(self) -> None:
        """Serialize the learnable state to disk. Thread-safe."""
        with self._lock:
            data = {
                "version": 1,
                "timestamp": time.time(),
                "energy": self._energy,
                "drives": asdict(self._drives),
                "W": self._W.tolist(),
                "V": self._V,
                "free_energy_signal": self._free_energy_signal,
                "threat_level": self._threat_level,
            }
            self._last_save = time.time()
            self._dirty = False

        # Write outside the lock to minimize hold time
        tmp_path = self._state_path.with_suffix(".tmp")
        try:
            with open(tmp_path, "w") as f:
                json.dump(data, f, indent=2)
            tmp_path.replace(self._state_path)
            logger.debug("HomeostaticRL state saved to %s", self._state_path)
        except Exception as e:
            record_degradation('homeostatic_rl', e)
            logger.error("Failed to save HomeostaticRL state: %s", e)
            # Clean up partial write
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass  # no-op: intentional

    def _load_state(self) -> None:
        """Restore state from disk if a save file exists."""
        if not self._state_path.exists():
            logger.info("No persisted HomeostaticRL state found, starting fresh")
            return

        try:
            with open(self._state_path, "r") as f:
                data = json.load(f)

            version = data.get("version", 0)
            if version != 1:
                logger.warning(
                    "HomeostaticRL state version %s != 1, ignoring", version
                )
                return

            # Restore energy
            self._energy = float(np.clip(data.get("energy", 75.0), 0.0, ENERGY_MAX))

            # Restore drives
            drives_data = data.get("drives", {})
            for key in [
                "social_hunger", "curiosity", "competence", "coherence_need",
                "social_hunger_setpoint", "curiosity_setpoint",
                "competence_setpoint", "coherence_need_setpoint",
                "social_hunger_drift", "curiosity_drift",
                "competence_drift", "coherence_need_drift",
            ]:
                if key in drives_data:
                    setattr(self._drives, key, float(drives_data[key]))

            # Restore weight matrix
            w_data = data.get("W")
            if w_data is not None:
                w_array = np.array(w_data, dtype=np.float64)
                if w_array.shape == (NUM_ACTIONS, STATE_DIM):
                    self._W = w_array
                else:
                    logger.warning(
                        "Weight matrix shape mismatch: expected (%d, %d), got %s",
                        NUM_ACTIONS, STATE_DIM, w_array.shape,
                    )

            # Restore value estimate
            self._V = float(data.get("V", 0.0))

            # Restore signals
            self._free_energy_signal = float(
                np.clip(data.get("free_energy_signal", 0.3), 0.0, 1.0)
            )
            self._threat_level = float(
                np.clip(data.get("threat_level", 0.0), 0.0, 1.0)
            )

            self._last_save = time.time()

            age = time.time() - data.get("timestamp", time.time())
            logger.info(
                "HomeostaticRL state restored (%.0f seconds old, energy=%.1f)",
                age, self._energy,
            )

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.error(
                "Failed to load HomeostaticRL state from %s: %s",
                self._state_path, e,
            )
        except Exception as e:
            record_degradation('homeostatic_rl', e)
            logger.error("Unexpected error loading HomeostaticRL state: %s", e)

    async def save(self) -> None:
        """Async-safe save. Call this on shutdown to flush state."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._save_state)

    # ------------------------------------------------------------------
    # Energy drain presets (convenience for callers)
    # ------------------------------------------------------------------

    def drain_llm_inference(self, tokens: int = 0) -> None:
        """Standard energy drain for an LLM inference call.
        This is the most expensive operation.
        """
        base_cost = 3.0
        token_cost = min(tokens * 0.001, 5.0) if tokens > 0 else 0.0
        self.drain_energy(base_cost + token_cost, f"llm_inference({tokens} tokens)")

    def drain_tool_execution(self, tool_name: str = "unknown") -> None:
        """Standard energy drain for a tool/skill execution."""
        self.drain_energy(1.5, f"tool_exec({tool_name})")

    def drain_background_tick(self) -> None:
        """Small energy drain for a background processing cycle."""
        self.drain_energy(0.1, "background_tick")

    def gain_successful_interaction(self, quality: float = 1.0) -> None:
        """Energy gain from a successful user interaction.

        Args:
            quality: 0-1 rating of how well it went.
        """
        amount = 4.0 + 4.0 * float(np.clip(quality, 0.0, 1.0))
        self.gain_energy(amount, f"successful_interaction(quality={quality:.2f})")

    def gain_error_resolved(self) -> None:
        """Energy gain from resolving an error."""
        self.gain_energy(3.0, "error_resolved")

    def gain_idle_recovery(self) -> None:
        """Small energy gain from idle time (called periodically by heartbeat)."""
        # Recovery is faster in the comfortable range, slower at extremes
        with self._lock:
            e = self._energy
        if e < ENERGY_COMFORT_LO:
            amount = 0.5  # Faster recovery when depleted
        elif e > ENERGY_COMFORT_HI:
            amount = 0.1  # Slow recovery when already comfortable
        else:
            amount = 0.3  # Normal recovery
        self.gain_energy(amount, "idle_recovery")


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: Optional[HomeostaticRL] = None
_instance_lock: threading.Lock = threading.Lock()


def get_homeostatic_rl() -> HomeostaticRL:
    """Return the singleton HomeostaticRL instance.

    Creates the instance on first call. Thread-safe via double-checked
    locking.
    """
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = HomeostaticRL()
    return _instance
