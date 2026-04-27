"""core/consciousness/octopus_arms.py
=========================================
Octopus-arm federation — 8 semi-autonomous arm-agents with local
chemoreception + central arbitration.

Modelled on the octopus nervous-system architecture described in
Kurzgesagt's "Most Secret Place" video and supporting papers
(Carls-Diamante 2022; Olson et al. 2025; Rosania 2014): 60% of the
octopus's neurons live in its arms; each arm has local sensors, local
decision circuits, and can act autonomously even when separated from
the central brain.

Each arm holds:
  • A local "chemoreceptor" state vector (3 scalar channels)
    representing what that arm is currently sensing.
  • A local policy that converts the sensor vector into a candidate
    action + strength.
  • An autonomy level [0, 1]: 0 = fully centralised, 1 = fully
    autonomous (acts regardless of central verdict).
  • A queue of proposed actions awaiting arbitration.

CentralArbiter:
  • Collects each arm's proposed action every tick.
  • When ``link`` is intact, combines proposals with a weighted
    priority vote and publishes the winning action.
  • When ``link`` is severed, the arbiter stops publishing; each
    arm's autonomy rises to 1.0 and arms act independently — their
    local decisions still execute, but without coordination.
  • When ``link`` is restored, the arbiter re-integrates the arm
    states and resumes publishing.  Integration-latency metric
    measures how many ticks are needed before arm agreement
    returns to baseline.

Impact on substrate:
  • The arbiter's winning action (or individual arm actions when
    severed) is published on the event bus as
    ``octopus_arm.action`` — downstream layers can consume it.
  • When the link is severed, the ``arm_decision_variance`` metric
    rises — observable in UnifiedField and get_status.
  • Arms contribute to the GlobalWorkspace candidate pool as
    per-arm CognitiveCandidate objects (priority = arm confidence
    × autonomy level).

Registered as ``octopus_federation`` in ServiceContainer.  Fed by
the heartbeat via ``tick(environment)``.
"""
from __future__ import annotations


import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("Consciousness.OctopusArms")


N_ARMS = 8
SENSOR_CHANNELS = 3            # (chemical, mechanical, visual)
ACTION_DIM = 8                 # Number of distinct action primitives
INTEGRATION_WINDOW = 32        # Ticks used to compute arm-decision variance


class ArmState(str, Enum):
    LINKED = "linked"            # Central arbiter is active and publishing
    SEVERED = "severed"          # No central coordination; each arm autonomous
    RECOVERING = "recovering"    # Link just restored, re-integrating


@dataclass
class ArmAction:
    arm_id: int
    action_idx: int
    confidence: float             # [0, 1]
    local_sensors: np.ndarray     # (SENSOR_CHANNELS,)
    ts: float = field(default_factory=time.time)


@dataclass
class ArbitrationResult:
    winning_action: Optional[int]
    winning_confidence: float
    tally: np.ndarray                # (ACTION_DIM,) per-action vote mass
    participating_arms: List[int]
    link_state: ArmState
    integration_latency: int         # Ticks since last stable-agreement
    decision_variance: float         # Across-arm variance of action choices


# ── Arm ────────────────────────────────────────────────────────────────────────

class OctopusArm:
    """One autonomous arm with local chemoreception + policy."""

    def __init__(self, arm_id: int, seed: int):
        self.arm_id = arm_id
        self.rng = np.random.default_rng(seed=seed)

        # Each arm has a distinct receptive field.  Chemoreceptors respond
        # to a particular subset of the environmental sensor pattern.
        self._receptive_field = self.rng.standard_normal(SENSOR_CHANNELS).astype(np.float32)
        self._receptive_field /= np.linalg.norm(self._receptive_field) + 1e-8

        # Local policy matrix: SENSOR_CHANNELS → ACTION_DIM preference.
        self._policy = self.rng.standard_normal((ACTION_DIM, SENSOR_CHANNELS)).astype(np.float32)
        self._policy /= np.linalg.norm(self._policy) + 1e-8

        self.autonomy: float = 0.1           # Base autonomy (rises when severed)
        self._last_action: Optional[ArmAction] = None
        self._action_history: Deque[ArmAction] = deque(maxlen=64)
        self._lock = threading.Lock()

    def sense(self, environment: np.ndarray) -> np.ndarray:
        """Apply this arm's receptive field to the environment vector."""
        env = np.asarray(environment, dtype=np.float32).reshape(-1)[:SENSOR_CHANNELS]
        env = np.pad(env, (0, max(0, SENSOR_CHANNELS - env.size)))
        # Arm-local chemoreception: scale by receptive field then tanh.
        local = np.tanh(env * self._receptive_field).astype(np.float32)
        return local

    def decide(self, environment: np.ndarray) -> ArmAction:
        local = self.sense(environment)
        # Softmax over action preferences.
        logits = self._policy @ local
        exp = np.exp(logits - logits.max())
        probs = exp / exp.sum()
        action = int(np.argmax(probs))
        confidence = float(probs[action])
        rec = ArmAction(
            arm_id=self.arm_id, action_idx=action, confidence=confidence,
            local_sensors=local,
        )
        with self._lock:
            self._last_action = rec
            self._action_history.append(rec)
        return rec

    def last_action(self) -> Optional[ArmAction]:
        with self._lock:
            return self._last_action

    def set_autonomy(self, val: float) -> None:
        self.autonomy = float(np.clip(val, 0.0, 1.0))


# ── Central arbiter ────────────────────────────────────────────────────────────

class CentralArbiter:
    """Weighted-vote arbitration across arm proposals.

    Under link=LINKED each arm contributes ``(1 - autonomy)`` × confidence
    vote mass to its chosen action.  Under link=SEVERED the arbiter
    stops publishing any decision — each arm must act alone with
    autonomy=1.
    """

    def __init__(self, arms: List[OctopusArm]):
        self._arms = arms
        self._link_state: ArmState = ArmState.LINKED
        self._link_changed_tick: int = 0
        self._tick: int = 0
        self._variance_history: Deque[float] = deque(maxlen=INTEGRATION_WINDOW)
        self._last_result: Optional[ArbitrationResult] = None
        self._lock = threading.Lock()
        self._observers: List[Callable[[ArbitrationResult], None]] = []

    # ── link controls ───────────────────────────────────────────────────

    def sever(self) -> None:
        with self._lock:
            if self._link_state == ArmState.SEVERED:
                return
            self._link_state = ArmState.SEVERED
            self._link_changed_tick = self._tick
            for a in self._arms:
                a.set_autonomy(1.0)
        logger.info("CentralArbiter: link SEVERED at tick=%d (arms→autonomy=1.0)",
                    self._link_changed_tick)

    def restore(self) -> None:
        with self._lock:
            if self._link_state == ArmState.LINKED:
                return
            self._link_state = ArmState.RECOVERING
            self._link_changed_tick = self._tick
            for a in self._arms:
                a.set_autonomy(0.1)
        logger.info("CentralArbiter: link restoring at tick=%d", self._link_changed_tick)

    def link_state(self) -> ArmState:
        return self._link_state

    # ── tick ────────────────────────────────────────────────────────────

    def tick(self, environment: np.ndarray) -> ArbitrationResult:
        self._tick += 1

        # Gather proposals.
        proposals: List[ArmAction] = []
        for a in self._arms:
            proposals.append(a.decide(environment))

        # Variance metric (0=all arms agree, 1=maximum disagreement).
        chosen = np.array([p.action_idx for p in proposals], dtype=np.int32)
        n = len(proposals)
        # Shannon entropy of action choices normalised to [0, 1].
        counts = np.bincount(chosen, minlength=ACTION_DIM).astype(np.float32)
        probs = counts / n
        probs = probs[probs > 0]
        entropy = float(-np.sum(probs * np.log2(probs)))
        max_entropy = math.log2(min(ACTION_DIM, n))
        variance = entropy / max_entropy if max_entropy > 0 else 0.0
        with self._lock:
            self._variance_history.append(variance)

        # Tally per-action vote mass (weighted by (1 - autonomy) × confidence).
        tally = np.zeros(ACTION_DIM, dtype=np.float32)
        participating: List[int] = []
        for a, p in zip(self._arms, proposals):
            weight = (1.0 - a.autonomy) * p.confidence
            if weight > 0:
                tally[p.action_idx] += weight
                participating.append(a.arm_id)

        # If linked, winner = argmax of tally.  If severed, no winner.
        with self._lock:
            if self._link_state == ArmState.SEVERED or not participating:
                winner = None
                winner_conf = 0.0
            else:
                winner = int(np.argmax(tally))
                winner_conf = float(tally[winner] / max(1e-8, tally.sum()))

            # Integration latency: ticks since variance first became low.
            integration_latency = 0
            if self._link_state == ArmState.RECOVERING:
                integration_latency = self._tick - self._link_changed_tick
                # Check if variance has stabilised at a low level for the last
                # 4 ticks; if yes, mark fully integrated.
                recent = list(self._variance_history)[-4:]
                if len(recent) >= 4 and max(recent) < 0.25:
                    self._link_state = ArmState.LINKED
                    logger.info(
                        "CentralArbiter: link fully re-integrated after %d ticks",
                        integration_latency,
                    )

            result = ArbitrationResult(
                winning_action=winner,
                winning_confidence=winner_conf,
                tally=tally.copy(),
                participating_arms=participating,
                link_state=self._link_state,
                integration_latency=integration_latency,
                decision_variance=variance,
            )
            self._last_result = result

        for cb in list(self._observers):
            try:
                cb(result)
            except Exception as exc:  # pragma: no cover
                logger.debug("arbiter observer failed: %s", exc)

        return result

    def subscribe(self, cb: Callable[[ArbitrationResult], None]) -> None:
        self._observers.append(cb)

    def current_state(self) -> Optional[ArbitrationResult]:
        with self._lock:
            return self._last_result


# ── Federation orchestrator ───────────────────────────────────────────────────

class OctopusFederation:
    """Public facade: 8 arms + 1 arbiter.

    Public API:
        fed.tick(env)                     → ArbitrationResult
        fed.sever_link()                  → stops central arbitration
        fed.restore_link()                → resumes central arbitration
        fed.get_status()                  → full diagnostic dict
        fed.arm_action(i)                 → last action from arm i
    """

    def __init__(self):
        self._arms = [OctopusArm(arm_id=i, seed=0xA000 + i) for i in range(N_ARMS)]
        self._arbiter = CentralArbiter(self._arms)
        logger.info(
            "OctopusFederation initialized: %d arms, %d sensor channels, %d actions",
            N_ARMS, SENSOR_CHANNELS, ACTION_DIM,
        )

    @property
    def arms(self) -> List[OctopusArm]:
        return self._arms

    @property
    def arbiter(self) -> CentralArbiter:
        return self._arbiter

    def tick(self, environment: np.ndarray) -> ArbitrationResult:
        return self._arbiter.tick(environment)

    def sever_link(self) -> None:
        self._arbiter.sever()

    def restore_link(self) -> None:
        self._arbiter.restore()

    def arm_action(self, arm_id: int) -> Optional[ArmAction]:
        if not (0 <= arm_id < N_ARMS):
            return None
        return self._arms[arm_id].last_action()

    def get_status(self) -> Dict[str, Any]:
        r = self._arbiter.current_state()
        return {
            "n_arms": N_ARMS,
            "link_state": self._arbiter.link_state().value,
            "last_winning_action": r.winning_action if r else None,
            "last_winning_confidence": round(r.winning_confidence, 4) if r else None,
            "last_variance": round(r.decision_variance, 4) if r else None,
            "integration_latency": r.integration_latency if r else 0,
            "arms_autonomy": [round(a.autonomy, 3) for a in self._arms],
            "arms_last_action": [
                a.last_action().action_idx if a.last_action() else None
                for a in self._arms
            ],
        }


# ── Singleton accessor ─────────────────────────────────────────────────────────

_INSTANCE: Optional[OctopusFederation] = None


def get_octopus_federation() -> OctopusFederation:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = OctopusFederation()
    return _INSTANCE
