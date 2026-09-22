"""Bounded spiking active-inference advisor for live cognition.

This module turns the "Aura brain" chalkboard ideas into runtime machinery:
Poisson-style spike traces, dendritic gating, eligibility traces, Bayesian
belief over cognitive state, and active-inference action tendencies.

It is intentionally advisory. It never executes tools, writes memory, or
mutates policy directly. Instead it produces auditable pressure signals that
the governed CognitiveEngine, AuthorityGateway, routing phases, and inference
sampler can consume.
"""

from __future__ import annotations

import hashlib
import math
import re
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from core.exceptions import ContainerError

EPS = 1e-12
_ADVISOR_RECOVERABLE_ERRORS = (
    ImportError,
    AttributeError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
    ContainerError,
)

STATE_NAMES: tuple[str, ...] = (
    "overloaded",
    "confused",
    "ready",
    "tool_needed",
    "research_needed",
    "repair_needed",
)

FEATURE_NAMES: tuple[str, ...] = (
    "clarity",
    "energy",
    "urgency",
    "novelty",
    "tool_pressure",
    "error_pressure",
    "social_pressure",
    "memory_pressure",
)

ACTION_NAMES: tuple[str, ...] = (
    "answer_directly",
    "plan_deeply",
    "seek_information",
    "use_governed_tools",
    "ask_clarification",
    "reduce_load",
    "repair_first",
)

_STATE_MEANS = np.array(
    [
        [0.20, 0.15, 0.85, 0.30, 0.25, 0.85, 0.20, 0.30],
        [0.25, 0.62, 0.45, 0.60, 0.25, 0.35, 0.45, 0.35],
        [0.88, 0.78, 0.35, 0.35, 0.20, 0.08, 0.40, 0.25],
        [0.65, 0.72, 0.60, 0.50, 0.92, 0.20, 0.30, 0.30],
        [0.55, 0.72, 0.45, 0.88, 0.55, 0.12, 0.35, 0.25],
        [0.40, 0.48, 0.70, 0.40, 0.45, 0.92, 0.25, 0.45],
    ],
    dtype=np.float64,
)

_REWARD_BY_STATE = np.array(
    [
        [-0.65, -0.15, 1.10, 0.10, 0.05, -0.35],
        [0.20, 0.75, 0.55, 0.40, 0.35, 0.25],
        [-0.25, 0.45, 0.25, 0.20, 1.10, 0.25],
        [-0.55, 0.10, 0.20, 1.25, 0.50, 0.55],
        [0.55, 1.15, -0.15, -0.10, 0.10, 0.20],
        [1.15, 0.35, -0.25, -0.30, -0.15, 0.55],
        [0.35, 0.25, -0.20, 0.15, 0.10, 1.30],
    ],
    dtype=np.float64,
)

_ACTION_COST = np.array([0.08, 0.20, 0.30, 0.34, 0.22, 0.16, 0.28], dtype=np.float64)
_ACTION_RISK = np.array([0.05, 0.08, 0.10, 0.22, 0.04, 0.03, 0.12], dtype=np.float64)
_ACTION_CONTROLLABILITY = np.array([0.10, 0.45, 0.80, 0.75, 0.90, 0.65, 0.85], dtype=np.float64)

_UNCERTAIN_RE = re.compile(
    r"\b(maybe|perhaps|not sure|confused|unclear|unknown|ambiguous|what do you mean|"
    r"which one|either|or|hypothetical|suppose|could|would)\b",
    re.IGNORECASE,
)
_TOOL_RE = re.compile(
    r"\b(open|create|save|export|download|search|browse|click|type|run|execute|"
    r"file|folder|desktop|notes?|chrome|docs?|pdf|app|terminal|website|url|tool)\b",
    re.IGNORECASE,
)
_RESEARCH_RE = re.compile(
    r"\b(research|look up|verify|latest|current|article|source|news|compare|find)\b",
    re.IGNORECASE,
)
_ERROR_RE = re.compile(
    r"\b(error|failed|failure|crash|broken|traceback|exception|stuck|loop|lag|"
    r"memory spike|ram|timeout|cortex warming|unavailable|degraded)\b",
    re.IGNORECASE,
)
_MEMORY_RE = re.compile(
    r"\b(remember|recall|earlier|last time|previous|across sessions|memory|forget)\b",
    re.IGNORECASE,
)
_SOCIAL_RE = re.compile(
    r"\b(hey|hi|hello|thanks|thank you|feel|feeling|lonely|miss|care|relationship|"
    r"who are you|what are you|conscious|sentient|self-aware)\b",
    re.IGNORECASE,
)


#: Matched as words. As substrings, "now" was found in "acknowledged", "fix"
#: in "prefix", "new" in "renew" and "idea" in "euclidean".
_URGENCY_WORDS_RE = re.compile(r"\b(?:now|urgent|immediately|top priority|fix)\b")
_NOVELTY_WORDS_RE = re.compile(r"\b(?:new|novel|unknown|idea|explore)\b")


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    if not math.isfinite(float(value)):
        return low
    return max(low, min(high, float(value)))


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    total = float(np.sum(values))
    if not math.isfinite(total) or total <= EPS:
        return np.ones_like(values, dtype=np.float64) / max(1, values.size)
    return values / total


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    temperature = max(EPS, float(temperature))
    z = np.asarray(logits, dtype=np.float64) / temperature
    z = z - float(np.max(z))
    return _normalize(np.exp(np.clip(z, -60.0, 60.0)))


def _entropy01(probabilities: np.ndarray) -> float:
    p = _normalize(np.asarray(probabilities, dtype=np.float64))
    if p.size <= 1:
        return 0.0
    entropy = float(-np.sum(p * np.log(p + EPS)))
    return _clamp(entropy / math.log(p.size))


def _safe_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    return parsed


def _status_unit(value: Any) -> float:
    """Normalize affect/status values that may arrive as 0..1 or 0..100."""

    parsed = _safe_float(value, 0.0)
    if abs(parsed) > 1.0:
        parsed = parsed / 100.0
    return _clamp(parsed)


def _unified_memory_pressure_features() -> dict[str, float]:
    """Read the canonical memory pressure gate without making it a hard dependency."""

    try:
        from core.utils.memory_monitor import get_memory_pressure_snapshot

        snapshot = get_memory_pressure_snapshot()
    except _ADVISOR_RECOVERABLE_ERRORS:
        return {}

    process_limit = max(0.1, _safe_float(getattr(snapshot, "process_rss_limit_gb", 0.0), 0.0))
    process_ratio = _clamp(_safe_float(getattr(snapshot, "process_rss_gb", 0.0), 0.0) / process_limit)
    pressure_pct = _clamp(_safe_float(getattr(snapshot, "pressure_pct", 0.0), 0.0) / 100.0)
    level = str(getattr(snapshot, "level", "normal") or "normal").lower()
    level_pressure = {
        "normal": 0.0,
        "warning": 0.35,
        "high": 0.62,
        "critical": 0.86,
        "emergency": 1.0,
    }.get(level, 0.0)
    return {
        "memory_pressure": max(pressure_pct, process_ratio, level_pressure),
        "process_pressure": process_ratio,
        "system_pressure": pressure_pct,
        "level_pressure": level_pressure,
    }


def _affective_driver_features() -> dict[str, float]:
    """Read operational affect drivers from the canonical affect service."""

    try:
        from core.container import ServiceContainer

        affect = ServiceContainer.get("affect_engine", default=None)
        if affect is None:
            return {}
        if hasattr(affect, "get_status"):
            status = affect.get_status() or {}
        elif hasattr(affect, "get_state_sync"):
            status = affect.get_state_sync() or {}
        else:
            return {}
    except _ADVISOR_RECOVERABLE_ERRORS:
        return {}

    if not isinstance(status, Mapping):
        return {}
    experiential = status.get("experiential")
    if not isinstance(experiential, Mapping):
        experiential = {}

    def get(name: str) -> float:
        if name in status:
            return _status_unit(status.get(name))
        return _status_unit(experiential.get(name))

    return {
        "confused": get("confused"),
        "curiosity": get("curiosity"),
        "frustration": get("frustration"),
        "upset": get("upset"),
        "longing": get("longing"),
        "loneliness": get("loneliness"),
        "pride": get("pride"),
        "empathy": get("empathy"),
    }


@dataclass(frozen=True)
class SpikingActiveInferenceConfig:
    dt: float = 0.01
    tau_psp: float = 0.12
    tau_eligibility: float = 0.65
    learning_rate: float = 0.045
    weight_decay: float = 0.002
    beta_0: float = 4.5
    threshold: float = 0.72
    seed: int = 2719


@dataclass(frozen=True)
class NeurodynamicAdvice:
    advice_id: str
    timestamp: float
    action: str
    belief: dict[str, float]
    probabilities: dict[str, float]
    features: dict[str, float]
    uncertainty: float
    confidence: float
    routing_bias: dict[str, Any]
    sampling_bias: dict[str, float]
    governance: dict[str, Any]
    neural_state: dict[str, float]
    stability: dict[str, float]
    working_memory: dict[str, Any]
    rationale: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MultiCompartmentSpikeResponseModel:
    """Small bounded SRM block used as a live cognitive pressure sensor."""

    def __init__(
        self,
        *,
        n_neurons: int = 16,
        compartments: int = 3,
        n_inputs: int = len(FEATURE_NAMES),
        config: SpikingActiveInferenceConfig | None = None,
    ) -> None:
        self.config = config or SpikingActiveInferenceConfig()
        self.n_neurons = int(max(4, min(64, n_neurons)))
        self.compartments = int(max(1, min(8, compartments)))
        self.n_inputs = int(max(1, min(32, n_inputs)))
        self._rng = np.random.default_rng(self.config.seed)
        self.weights = self._rng.normal(
            loc=0.25,
            scale=0.04,
            size=(self.n_neurons, self.compartments, self.n_inputs),
        ).astype(np.float64)
        self.weights = np.clip(self.weights, 0.0, 2.0)
        self.psp = np.zeros(self.n_inputs, dtype=np.float64)
        self.eligibility = np.zeros((self.n_neurons, self.compartments), dtype=np.float64)
        self.dendritic_trace = np.zeros((self.n_neurons, self.compartments), dtype=np.float64)
        self.post_trace = np.zeros(self.n_neurons, dtype=np.float64)
        self.threshold = np.full(self.n_neurons, self.config.threshold, dtype=np.float64)
        self.last_spike_rate = 0.0
        self.last_plateau_rate = 0.0

    def tick(self, input_vector: Sequence[float], *, modulation: float = 1.0) -> dict[str, float]:
        x = np.asarray(input_vector, dtype=np.float64).ravel()
        if x.size != self.n_inputs:
            padded = np.zeros(self.n_inputs, dtype=np.float64)
            n = min(self.n_inputs, x.size)
            padded[:n] = x[:n]
            x = padded
        x = np.clip(np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        modulation = _clamp(modulation, 0.1, 2.0)

        decay_psp = math.exp(-self.config.dt / self.config.tau_psp)
        decay_elig = math.exp(-self.config.dt / self.config.tau_eligibility)
        self.psp = self.psp * decay_psp + x

        dendritic_input = np.einsum("nci,i->nc", self.weights, self.psp)
        plateau = (dendritic_input >= self.threshold[:, None]).astype(np.float64)
        self.last_plateau_rate = float(np.mean(plateau))

        self.eligibility = self.eligibility * decay_elig + plateau
        self.dendritic_trace = (
            self.dendritic_trace * 0.94
            + 0.06 * np.tanh(dendritic_input + self.eligibility)
        )
        membrane = np.sum(dendritic_input + 0.35 * self.dendritic_trace, axis=1)
        logits = np.clip((membrane - self.threshold) * self.config.beta_0, -50.0, 50.0)
        spike_prob = np.clip(1.0 / (1.0 + np.exp(-logits)), 0.0, 1.0)
        spikes = self._rng.random(self.n_neurons) < spike_prob * self.config.dt
        self.last_spike_rate = float(np.mean(spikes))
        self.post_trace = self.post_trace * 0.90 + spikes.astype(np.float64)

        pre = self.psp[None, None, :]
        post = (self.post_trace[:, None, None] + 0.05)
        delta_w = (
            self.config.learning_rate
            * modulation
            * post
            * pre
            * self.config.dt
            - self.config.weight_decay * self.weights * self.config.dt
        )
        self.weights = np.clip(self.weights + delta_w, 0.0, 2.0)
        self.threshold = np.clip(
            self.threshold
            + self.config.dt * (0.02 * (self.post_trace - 0.10) - 0.01 * modulation),
            0.35,
            1.45,
        )

        return self.summary()

    def summary(self) -> dict[str, float]:
        return {
            "spike_rate": _clamp(self.last_spike_rate),
            "plateau_rate": _clamp(self.last_plateau_rate),
            "eligibility_mean": _clamp(float(np.mean(self.eligibility)), 0.0, 3.0),
            "weight_mean": _clamp(float(np.mean(self.weights)), 0.0, 2.0),
            "threshold_mean": _clamp(float(np.mean(self.threshold)), 0.0, 2.0),
        }


class BoundedWorkingMemoryQueueModel:
    """Deterministic M/M/1-inspired pressure model for attention admission.

    The model does not drop foreground user turns. It estimates how much work a
    new cognitive event adds, how quickly live cognition can service that work,
    and whether background work should be deferred while foreground pressure is
    high. The outputs are advisory pressure signals consumed by routing and
    sampling, not a separate execution path.
    """

    def __init__(
        self,
        *,
        arrival_rate: float = 7.0,
        service_rate: float = 10.0,
        max_queue: float = 5.0,
    ) -> None:
        self.arrival_rate = max(0.1, float(arrival_rate))
        self.service_rate = max(0.1, float(service_rate))
        self.max_queue = max(1.0, float(max_queue))
        self._load = 0.0
        self._last_update = time.monotonic()
        self._accepted = 0
        self._deferred = 0
        self._compressed = 0

    def reset(self) -> None:
        """Drop accumulated queue load and counters.

        The queue model is a genuine accumulator hanging off a process-wide
        advisor, which is correct for a running mind and wrong for a process
        that runs many independent scenarios in sequence: load left by one
        scenario turns the next one's admission from "accept" into
        "compress_foreground". That was the difference between an imagination
        test passing alone and failing in a long run.
        """
        self._load = 0.0
        self._last_update = time.monotonic()
        self._accepted = 0
        self._deferred = 0
        self._compressed = 0

    def observe(
        self,
        features: Mapping[str, float],
        *,
        is_background: bool,
    ) -> dict[str, Any]:
        now = time.monotonic()
        elapsed = min(5.0, max(0.0, now - self._last_update))
        self._last_update = now

        energy = _clamp(_safe_float(features.get("energy"), 0.7))
        existing_memory_pressure = _clamp(_safe_float(features.get("memory_pressure"), 0.0))
        effective_service = max(
            0.5,
            self.service_rate * max(0.25, energy) * (1.0 - 0.45 * existing_memory_pressure),
        )
        self._load = max(0.0, self._load - effective_service * elapsed)

        incoming_pressure = _clamp(
            0.30 * _safe_float(features.get("urgency"), 0.0)
            + 0.24 * _safe_float(features.get("tool_pressure"), 0.0)
            + 0.18 * _safe_float(features.get("error_pressure"), 0.0)
            + 0.12 * _safe_float(features.get("novelty"), 0.0)
            + 0.10 * existing_memory_pressure
            + 0.06 * (1.0 - _safe_float(features.get("clarity"), 1.0))
        )
        effective_arrival = max(0.1, self.arrival_rate * (0.35 + 1.15 * incoming_pressure))
        utilization = _clamp(effective_arrival / effective_service, 0.0, 2.5)
        work_units = 1.0 + (self.max_queue - 1.0) * incoming_pressure
        projected = self._load + work_units
        overflow = max(0.0, projected - self.max_queue)
        overload_pressure = _clamp(overflow / self.max_queue)

        if is_background and overflow > 0.0:
            self._deferred += 1
            admitted = False
            admission = "defer_background"
            self._load = min(self.max_queue, self._load)
        else:
            admitted = True
            if overflow > 0.0:
                self._compressed += 1
                admission = "compress_foreground"
            else:
                admission = "accept"
            self._accepted += 1
            self._load = min(self.max_queue, projected)

        queue_load = _clamp(self._load / self.max_queue)
        rho = min(0.99, max(0.0, utilization))
        wait_s = (rho / max(EPS, 1.0 - rho)) * (1.0 / effective_service)

        return {
            "admitted": admitted,
            "admission": admission,
            "queue_load": round(queue_load, 4),
            "overload_pressure": round(max(overload_pressure, _clamp(utilization - 1.0)), 4),
            "incoming_pressure": round(incoming_pressure, 4),
            "arrival_rate": round(effective_arrival, 4),
            "service_rate": round(effective_service, 4),
            "utilization": round(utilization, 4),
            "expected_wait_s": round(wait_s, 4),
            "accepted": self._accepted,
            "deferred": self._deferred,
            "compressed": self._compressed,
            "max_queue": self.max_queue,
        }

    def snapshot(self) -> dict[str, Any]:
        queue_load = _clamp(self._load / self.max_queue)
        return {
            "admitted": True,
            "admission": "idle",
            "queue_load": round(queue_load, 4),
            "overload_pressure": 0.0,
            "incoming_pressure": 0.0,
            "arrival_rate": round(self.arrival_rate, 4),
            "service_rate": round(self.service_rate, 4),
            "utilization": round(self.arrival_rate / max(EPS, self.service_rate), 4),
            "expected_wait_s": round(
                (
                    min(0.99, self.arrival_rate / max(EPS, self.service_rate))
                    / max(EPS, 1.0 - min(0.99, self.arrival_rate / max(EPS, self.service_rate)))
                )
                * (1.0 / self.service_rate),
                4,
            ),
            "accepted": self._accepted,
            "deferred": self._deferred,
            "compressed": self._compressed,
            "max_queue": self.max_queue,
        }


class SoftmaxCompetitionStabilityProbe:
    """Local Jacobian probe for softmax action competition stability.

    This is the runtime-safe version of bifurcation/Jacobian analysis: no SciPy,
    no symbolic algebra, no plotting stack, and no long sweeps in the live path.
    It analyzes the current action logits/probabilities and returns bounded
    instability signals that can steer metacognition and sampling.
    """

    @staticmethod
    def analyze(
        scores: Sequence[float],
        probabilities: Sequence[float],
        *,
        temperature: float,
        features: Mapping[str, float] | None = None,
    ) -> dict[str, float]:
        logits = np.asarray(scores, dtype=np.float64).ravel()
        p = _normalize(np.asarray(probabilities, dtype=np.float64).ravel())
        if logits.size == 0 or p.size == 0 or logits.size != p.size:
            return {
                "spectral_radius": 0.0,
                "entropy": 0.0,
                "winner_margin": 1.0,
                "decision_instability": 0.0,
                "ode_spectral_abscissa": 0.0,
                "fixed_point_residual": 0.0,
                "bifurcation_pressure": 0.0,
            }
        temp = max(EPS, float(temperature))
        jacobian = (np.diag(p) - np.outer(p, p)) / temp
        try:
            eigenvalues = np.linalg.eigvals(jacobian)
            spectral_radius = float(np.max(np.abs(eigenvalues))) if eigenvalues.size else 0.0
        # not a failure: a value that is not a number is not one this can read.
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            spectral_radius = 0.0
        ordered = np.sort(p)
        if ordered.size >= 2:
            winner_margin = float(ordered[-1] - ordered[-2])
        else:
            winner_margin = 1.0
        entropy = _entropy01(p)
        decision_instability = _clamp(
            0.52 * entropy
            + 0.33 * (1.0 - _clamp(winner_margin))
            + 0.15 * _clamp(spectral_radius / 1.25)
        )
        ode = SoftmaxODEStabilityProbe.analyze(
            logits,
            p,
            temperature=temp,
            features=features,
        )
        decision_instability = _clamp(
            decision_instability + 0.18 * ode["bifurcation_pressure"]
        )
        return {
            "spectral_radius": round(_clamp(spectral_radius, 0.0, 5.0), 4),
            "entropy": round(entropy, 4),
            "winner_margin": round(_clamp(winner_margin), 4),
            "decision_instability": round(decision_instability, 4),
            **ode,
        }


class SoftmaxODEStabilityProbe:
    """Bounded local ODE/Jacobian probe for live action competition.

    The attachment's full analysis script is useful offline, but symbolic
    Jacobians, root finding, and bifurcation sweeps do not belong in foreground
    chat. This probe keeps the causal part: it evaluates one local nonlinear
    competition model around the current action distribution and reports whether
    the live action landscape is drifting toward an unstable attractor.
    """

    @staticmethod
    def analyze(
        scores: Sequence[float],
        probabilities: Sequence[float],
        *,
        temperature: float,
        features: Mapping[str, float] | None = None,
    ) -> dict[str, float]:
        logits = np.asarray(scores, dtype=np.float64).ravel()
        p0 = _normalize(np.asarray(probabilities, dtype=np.float64).ravel())
        if logits.size == 0 or p0.size == 0 or logits.size != p0.size:
            return {
                "ode_spectral_abscissa": 0.0,
                "fixed_point_residual": 0.0,
                "bifurcation_pressure": 0.0,
            }

        features = features or {}
        urgency = _clamp(_safe_float(features.get("urgency"), 0.3))
        novelty = _clamp(_safe_float(features.get("novelty"), 0.0))
        error_pressure = _clamp(_safe_float(features.get("error_pressure"), 0.0))
        memory_pressure = _clamp(_safe_float(features.get("memory_pressure"), 0.0))
        clarity = _clamp(_safe_float(features.get("clarity"), 0.7))

        lambda_ = 0.10 + 0.32 * urgency + 0.18 * novelty
        damping = 0.92 + 0.35 * error_pressure + 0.30 * memory_pressure
        px = 1.0 + 0.45 * (1.0 - clarity) + 0.25 * error_pressure
        k = 0.05 + 0.10 * memory_pressure
        coupling = 0.05 + 0.22 * novelty + 0.16 * (1.0 - clarity)
        temp = max(EPS, float(temperature))

        def rhs(b: np.ndarray) -> np.ndarray:
            b = np.clip(np.nan_to_num(b, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
            local_logits = logits + coupling * (b - p0)
            local_p = _softmax(local_logits, temp)
            growth = lambda_ * b * (b + k)
            inhibition = b * damping * (local_p ** px)
            return growth - inhibition

        b = np.clip(p0, 0.0, 1.0)
        f0 = rhs(b)
        residual = float(np.linalg.norm(f0) / max(1.0, math.sqrt(float(b.size))))

        jacobian = np.zeros((b.size, b.size), dtype=np.float64)
        step = 1e-5
        for idx in range(b.size):
            shifted = b.copy()
            shifted[idx] = _clamp(shifted[idx] + step)
            jacobian[:, idx] = (rhs(shifted) - f0) / step
        try:
            eig = np.linalg.eigvals(jacobian)
            spectral_abscissa = float(np.max(np.real(eig))) if eig.size else 0.0
        # not a failure: a value that is not a number is not one this can read.
        except (np.linalg.LinAlgError, FloatingPointError, ValueError):
            spectral_abscissa = 0.0

        stress = _clamp(
            0.34 * urgency
            + 0.26 * novelty
            + 0.22 * error_pressure
            + 0.18 * memory_pressure
            + 0.16 * (1.0 - clarity)
        )
        positive_drift = _clamp((spectral_abscissa + 0.25) / 1.25)
        fixed_residual = _clamp(residual / 0.35)
        bifurcation_pressure = _clamp(
            0.46 * positive_drift + 0.24 * fixed_residual + 0.30 * stress
        )
        return {
            "ode_spectral_abscissa": round(_clamp(spectral_abscissa, -5.0, 5.0), 4),
            "fixed_point_residual": round(fixed_residual, 4),
            "bifurcation_pressure": round(bifurcation_pressure, 4),
        }


class SpikingActiveInferenceAdvisor:
    """General cognitive advisor combining SRM dynamics and active inference."""

    def __init__(self, config: SpikingActiveInferenceConfig | None = None) -> None:
        self.config = config or SpikingActiveInferenceConfig()
        self._lock = threading.Lock()
        self._rng = np.random.default_rng(self.config.seed + 1)
        self._belief = np.ones(len(STATE_NAMES), dtype=np.float64) / len(STATE_NAMES)
        self._weights = self._rng.normal(0.0, 0.025, size=(len(ACTION_NAMES), len(FEATURE_NAMES)))
        self._bias = np.zeros(len(ACTION_NAMES), dtype=np.float64)
        self._counts = np.zeros(len(ACTION_NAMES), dtype=np.int64)
        self._srm = MultiCompartmentSpikeResponseModel(config=self.config)
        self._working_memory = BoundedWorkingMemoryQueueModel()
        self._stability_probe = SoftmaxCompetitionStabilityProbe()
        self._last_advice: NeurodynamicAdvice | None = None

    def advise(
        self,
        objective: str,
        *,
        context: Mapping[str, Any] | None = None,
        state: Any = None,
        origin: str = "system",
        is_background: bool = False,
    ) -> NeurodynamicAdvice:
        with self._lock:
            features = self._features_from_runtime(objective, context=context, state=state)
            working_memory = self._working_memory.observe(
                features,
                is_background=is_background,
            )
            features = self._apply_working_memory_pressure(features, working_memory)
            x = np.array([features[name] for name in FEATURE_NAMES], dtype=np.float64)
            likelihood = self._state_likelihood(x)
            self._belief = _normalize(0.72 * self._belief + 0.28 * likelihood)
            uncertainty = _entropy01(self._belief)
            neural_state = self._srm.tick(
                x,
                modulation=0.65
                + 0.50 * features["error_pressure"]
                + 0.35 * features["novelty"]
                + 0.25 * features["tool_pressure"],
            )
            scores = self._score_actions(x, uncertainty, is_background=is_background)
            temperature = self._temperature(features, uncertainty)
            probabilities = _softmax(scores, temperature)
            stability = self._stability_probe.analyze(
                scores,
                probabilities,
                temperature=temperature,
                features=features,
            )
            action_index = int(np.argmax(probabilities))
            self._counts[action_index] += 1
            action = ACTION_NAMES[action_index]
            routing_bias = self._routing_bias(action, features, uncertainty, is_background, stability)
            sampling_bias = self._sampling_bias(action, features, uncertainty, stability)
            advice = NeurodynamicAdvice(
                advice_id=self._advice_id(objective, origin),
                timestamp=time.time(),
                action=action,
                belief={
                    name: _clamp(float(value))
                    for name, value in zip(STATE_NAMES, self._belief, strict=True)
                },
                probabilities={
                    name: _clamp(float(value))
                    for name, value in zip(ACTION_NAMES, probabilities, strict=True)
                },
                features={name: _clamp(value) for name, value in features.items()},
                uncertainty=uncertainty,
                confidence=1.0 - uncertainty,
                routing_bias=routing_bias,
                sampling_bias=sampling_bias,
                governance={
                    "consequential_action": False,
                    "executes_tools": False,
                    "writes_memory": False,
                    "authority_gateway_required_for_effects": True,
                    "advisory_only": True,
                    "origin": str(origin or "system")[:64],
                },
                neural_state=neural_state,
                stability=stability,
                working_memory=working_memory,
                rationale=self._rationale(
                    action,
                    features,
                    uncertainty,
                    routing_bias,
                    working_memory,
                    stability,
                ),
            )
            self._last_advice = advice
            return advice

    def learn_from_feedback(
        self,
        action: str,
        reward: float,
        features: Mapping[str, float] | Sequence[float],
    ) -> dict[str, float]:
        with self._lock:
            if action not in ACTION_NAMES:
                raise ValueError(f"unknown active-inference action: {action!r}")
            idx = ACTION_NAMES.index(action)
            if isinstance(features, Mapping):
                x = np.array([_safe_float(features.get(name), 0.0) for name in FEATURE_NAMES])
            else:
                x = np.asarray(features, dtype=np.float64).ravel()
            if x.size != len(FEATURE_NAMES):
                raise ValueError(f"features must have {len(FEATURE_NAMES)} values")
            x = np.clip(np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
            predicted = float(self._weights[idx] @ x + self._bias[idx])
            error = float(np.clip(float(reward) - predicted, -3.0, 3.0))
            self._weights[idx] += self.config.learning_rate * error * x
            self._weights[idx] -= self.config.weight_decay * self._weights[idx]
            self._bias[idx] += self.config.learning_rate * error
            return {
                "action": action,
                "predicted": predicted,
                "reward": float(reward),
                "prediction_error": error,
                "weight_l2": float(np.linalg.norm(self._weights[idx])),
            }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            if self._last_advice is None:
                return {
                    "status": "idle",
                    "belief": {
                        name: float(p)
                        for name, p in zip(STATE_NAMES, self._belief, strict=True)
                    },
                    "neural_state": self._srm.summary(),
                    "stability": self._stability_probe.analyze(
                        np.zeros(len(ACTION_NAMES)),
                        np.ones(len(ACTION_NAMES)) / len(ACTION_NAMES),
                        temperature=1.0,
                    ),
                    "working_memory": self._working_memory.snapshot(),
                }
            data = self._last_advice.to_dict()
            data["status"] = "active"
            return data

    def _state_likelihood(self, x: np.ndarray) -> np.ndarray:
        std = np.array([0.20, 0.22, 0.24, 0.24, 0.22, 0.20, 0.24, 0.22], dtype=np.float64)
        z = (x[None, :] - _STATE_MEANS) / std[None, :]
        log_likelihood = -0.5 * np.sum(z * z, axis=1)
        log_likelihood -= float(np.max(log_likelihood))
        return _normalize(np.exp(np.clip(log_likelihood, -60.0, 60.0)))

    def _score_actions(self, x: np.ndarray, uncertainty: float, *, is_background: bool) -> np.ndarray:
        learned = self._weights @ x + self._bias
        expected = _REWARD_BY_STATE @ self._belief
        epistemic = _ACTION_CONTROLLABILITY * uncertainty
        optimism = np.array([0.06 / math.sqrt(float(c + 1)) for c in self._counts])
        cost = _ACTION_COST + (0.18 if is_background else 0.0)
        risk = _ACTION_RISK
        scores = learned + expected + 0.38 * epistemic + optimism - cost - 0.85 * risk
        if x[4] > 0.60:
            scores[ACTION_NAMES.index("use_governed_tools")] += 0.40 * x[4]
        if x[5] > 0.55:
            scores[ACTION_NAMES.index("repair_first")] += 0.45 * x[5]
            scores[ACTION_NAMES.index("reduce_load")] += 0.25 * x[5]
        if x[7] > 0.55:
            scores[ACTION_NAMES.index("reduce_load")] += 0.55 * x[7]
        if x[0] < 0.35:
            scores[ACTION_NAMES.index("ask_clarification")] += 0.35 * (1.0 - x[0])
        if x[3] > 0.60 and x[4] < 0.50:
            scores[ACTION_NAMES.index("seek_information")] += 0.30 * x[3]
        return scores

    def _apply_working_memory_pressure(
        self,
        features: Mapping[str, float],
        working_memory: Mapping[str, Any],
    ) -> dict[str, float]:
        updated = {name: _clamp(_safe_float(features.get(name), 0.0)) for name in FEATURE_NAMES}
        queue_load = _clamp(_safe_float(working_memory.get("queue_load"), 0.0))
        overload = _clamp(_safe_float(working_memory.get("overload_pressure"), 0.0))
        pressure = max(queue_load, overload)
        if pressure <= 0.0:
            return updated
        updated["memory_pressure"] = max(updated["memory_pressure"], pressure)
        updated["error_pressure"] = max(updated["error_pressure"], 0.55 * overload)
        updated["urgency"] = max(updated["urgency"], 0.35 + 0.40 * pressure)
        updated["energy"] = min(updated["energy"], 1.0 - 0.42 * pressure)
        updated["clarity"] = _clamp(updated["clarity"] - 0.22 * pressure)
        return updated

    def _temperature(self, features: Mapping[str, float], uncertainty: float) -> float:
        base = 0.58 + 0.20 * features["novelty"] - 0.18 * features["error_pressure"]
        base += 0.12 * uncertainty
        return max(0.30, min(0.92, base))

    def _routing_bias(
        self,
        action: str,
        features: Mapping[str, float],
        uncertainty: float,
        is_background: bool,
        stability: Mapping[str, float],
    ) -> dict[str, Any]:
        decision_instability = _clamp(_safe_float(stability.get("decision_instability"), 0.0))
        return {
            "prefer_direct_answer": action == "answer_directly" and uncertainty < 0.55,
            "prefer_deep_reasoning": action == "plan_deeply",
            "seek_information": action == "seek_information",
            "use_tool_gateway": action == "use_governed_tools" or features["tool_pressure"] >= 0.58,
            "ask_clarification": (
                action == "ask_clarification"
                or (features["clarity"] < 0.34 and features["tool_pressure"] < 0.45)
                or (uncertainty >= 0.62 and features["tool_pressure"] < 0.45)
                or (decision_instability >= 0.78 and features["tool_pressure"] < 0.45)
            ),
            "reduce_load": (
                action == "reduce_load"
                or features["energy"] < 0.26
                or features["memory_pressure"] >= 0.62
            ),
            "repair_first": action == "repair_first" or features["error_pressure"] >= 0.68,
            "metacognition_depth": round(
                _clamp(
                    0.35
                    + 0.45 * uncertainty
                    + 0.35 * features["error_pressure"]
                    + 0.18 * features["memory_pressure"]
                    + 0.18 * decision_instability,
                    0.0,
                    1.0,
                ),
                4,
            ),
            "tool_pressure": round(features["tool_pressure"], 4),
            "background_request": bool(is_background),
        }

    def _sampling_bias(
        self,
        action: str,
        features: Mapping[str, float],
        uncertainty: float,
        stability: Mapping[str, float],
    ) -> dict[str, float]:
        reduce_load = (
            action == "reduce_load"
            or features["energy"] < 0.26
            or features["memory_pressure"] >= 0.62
        )
        concise_clarification = action == "ask_clarification"
        memory_budget = max(0.34, 1.0 - 0.55 * features["memory_pressure"])
        decision_instability = _clamp(_safe_float(stability.get("decision_instability"), 0.0))
        return {
            "temperature_delta": round(
                _clamp(
                    -0.10 * uncertainty
                    - 0.10 * features["error_pressure"]
                    - 0.06 * decision_instability
                    + 0.06 * features["novelty"],
                    -0.20,
                    0.12,
                ),
                4,
            ),
            "top_p_delta": round(
                _clamp(
                    -0.10 * uncertainty
                    - 0.04 * features["error_pressure"]
                    - 0.04 * decision_instability,
                    -0.20,
                    0.04,
                ),
                4,
            ),
            "max_tokens_factor": round(
                min(memory_budget, 0.62)
                if reduce_load
                else (0.75 if concise_clarification else memory_budget),
                4,
            ),
            "repetition_penalty_delta": round(
                _clamp(0.04 * features["error_pressure"] + 0.03 * uncertainty, 0.0, 0.08),
                4,
            ),
            "presence_penalty_delta": round(
                _clamp(0.08 * features["novelty"] + 0.04 * features["tool_pressure"], 0.0, 0.12),
                4,
            ),
        }

    def _features_from_runtime(
        self,
        objective: str,
        *,
        context: Mapping[str, Any] | None,
        state: Any,
    ) -> dict[str, float]:
        text = str(objective or "")
        lowered = text.lower()
        length_pressure = _clamp(len(text) / 2400.0)
        question_pressure = _clamp((text.count("?") + lowered.count(" or ")) / 5.0)
        uncertainty = _clamp(
            (0.30 if _UNCERTAIN_RE.search(text) else 0.0)
            + 0.25 * question_pressure
            + 0.18 * length_pressure
        )
        tool_pressure = _clamp(
            (0.62 if _TOOL_RE.search(text) else 0.0)
            + (0.20 if _RESEARCH_RE.search(text) else 0.0)
        )
        research_pressure = 0.38 if _RESEARCH_RE.search(text) else 0.0
        error_pressure = 0.70 if _ERROR_RE.search(text) else 0.0
        memory_pressure = 0.62 if _MEMORY_RE.search(text) else 0.0
        social_pressure = 0.55 if _SOCIAL_RE.search(text) else 0.0
        urgency = 0.72 if _URGENCY_WORDS_RE.search(lowered) else 0.30
        memory_features = _unified_memory_pressure_features()
        affect_features = _affective_driver_features()

        modifiers = getattr(state, "response_modifiers", {}) if state is not None else {}
        if not isinstance(modifiers, Mapping):
            modifiers = {}
        context = context or {}
        energy = _safe_float(modifiers.get("homeostatic_energy"), 75.0) / 100.0
        energy = _safe_float(context.get("homeostatic_energy"), energy)
        anomaly = _safe_float(modifiers.get("anomaly_threat_level"), 0.0)
        free_energy = _safe_float(modifiers.get("fe", modifiers.get("free_energy", 0.0)), 0.0)
        runtime_memory_pressure = _clamp(memory_features.get("memory_pressure", 0.0))
        if runtime_memory_pressure > 0.0:
            memory_pressure = max(memory_pressure, runtime_memory_pressure)
            error_pressure = max(error_pressure, 0.72 * runtime_memory_pressure)
            urgency = max(urgency, 0.38 + 0.42 * runtime_memory_pressure)
            energy = min(energy, 1.0 - 0.90 * runtime_memory_pressure)
        affect_confused = _clamp(affect_features.get("confused", 0.0))
        affect_frustration = _clamp(
            max(affect_features.get("frustration", 0.0), affect_features.get("upset", 0.0))
        )
        affect_curiosity = _clamp(affect_features.get("curiosity", 0.0))
        affect_social_need = _clamp(
            max(
                affect_features.get("longing", 0.0),
                affect_features.get("loneliness", 0.0),
                affect_features.get("empathy", 0.0) * 0.75,
            )
        )
        if affect_confused:
            uncertainty = _clamp(uncertainty + 0.26 * affect_confused)
            error_pressure = max(error_pressure, 0.30 * affect_confused)
        if affect_frustration:
            error_pressure = max(error_pressure, 0.55 * affect_frustration)
            urgency = max(urgency, 0.36 + 0.32 * affect_frustration)
        if affect_curiosity:
            research_pressure = max(research_pressure, 0.30 * affect_curiosity)
        if affect_social_need:
            social_pressure = max(social_pressure, 0.45 * affect_social_need)
        if bool(context.get("desktop_cognitive_engine_required")):
            tool_pressure = max(tool_pressure, 0.08)
        if modifiers.get("intent_type") == "SKILL" or modifiers.get("matched_skills"):
            tool_pressure = max(tool_pressure, 0.72)
        error_pressure = max(error_pressure, _clamp(anomaly), _clamp(free_energy))
        novelty = _clamp(
            research_pressure
            + (0.28 if _NOVELTY_WORDS_RE.search(lowered) else 0.0)
            + 0.24 * affect_curiosity
            + 0.20 * uncertainty
        )
        clarity = _clamp(
            1.0
            - uncertainty
            - 0.35 * error_pressure
            - 0.22 * affect_confused
            + 0.10 * social_pressure
        )
        return {
            "clarity": clarity,
            "energy": _clamp(energy),
            "urgency": _clamp(urgency + 0.20 * error_pressure),
            "novelty": novelty,
            "tool_pressure": tool_pressure,
            "error_pressure": error_pressure,
            "social_pressure": _clamp(social_pressure),
            "memory_pressure": _clamp(memory_pressure),
        }

    def _rationale(
        self,
        action: str,
        features: Mapping[str, float],
        uncertainty: float,
        routing_bias: Mapping[str, Any],
        working_memory: Mapping[str, Any],
        stability: Mapping[str, float],
    ) -> list[str]:
        rationale = [
            f"selected={action}",
            f"uncertainty={uncertainty:.3f}",
            f"clarity={features['clarity']:.3f}",
            f"tool_pressure={features['tool_pressure']:.3f}",
            f"error_pressure={features['error_pressure']:.3f}",
        ]
        if routing_bias.get("use_tool_gateway"):
            rationale.append("tool effects must go through governed capability path")
        if routing_bias.get("reduce_load"):
            rationale.append("load reduction requested by low energy or overload belief")
        if working_memory.get("admission") in {"compress_foreground", "defer_background"}:
            rationale.append(
                "working memory admission requested "
                f"{working_memory.get('admission')} at load={working_memory.get('queue_load')}"
            )
        if _safe_float(stability.get("decision_instability"), 0.0) >= 0.70:
            rationale.append(
                "softmax Jacobian indicated unstable action competition; metacognition increased"
            )
        return rationale

    @staticmethod
    def _advice_id(objective: str, origin: str) -> str:
        payload = f"{time.time_ns()}:{origin}:{objective[:240]}".encode("utf-8", errors="ignore")
        return hashlib.sha256(payload).hexdigest()[:16]


_ADVISOR: SpikingActiveInferenceAdvisor | None = None
_ADVISOR_LOCK = threading.Lock()


def _register_advisor(advisor: SpikingActiveInferenceAdvisor) -> None:
    try:
        from core.container import ServiceContainer

        current = ServiceContainer.get("spiking_active_inference", default=None)
        if current is advisor:
            return
        ServiceContainer.register_instance(
            "spiking_active_inference",
            advisor,
            required=False,
            owner="core/cognitive/spiking_active_inference.py",
            required_for="cognitive advisory routing and substrate sampling",
            failure_policy="degrade_without_effects",
        )
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        return


def get_spiking_active_inference_advisor() -> SpikingActiveInferenceAdvisor:
    global _ADVISOR
    try:
        from core.container import ServiceContainer
        from core.runtime.shutdown_coordinator import is_shutdown_requested

        current = ServiceContainer.peek("spiking_active_inference", default=None)
        if isinstance(current, SpikingActiveInferenceAdvisor):
            _ADVISOR = current
            return current
        if is_shutdown_requested():
            raise RuntimeError("runtime_shutdown")
        current = ServiceContainer.get("spiking_active_inference", default=None)
        if isinstance(current, SpikingActiveInferenceAdvisor):
            _ADVISOR = current
            return current
    except _ADVISOR_RECOVERABLE_ERRORS:
        from core.runtime.shutdown_coordinator import is_shutdown_requested

        if is_shutdown_requested():
            raise RuntimeError("runtime_shutdown") from None
    with _ADVISOR_LOCK:
        if _ADVISOR is None:
            _ADVISOR = SpikingActiveInferenceAdvisor()
        _register_advisor(_ADVISOR)
        return _ADVISOR


__all__ = [
    "ACTION_NAMES",
    "FEATURE_NAMES",
    "STATE_NAMES",
    "BoundedWorkingMemoryQueueModel",
    "MultiCompartmentSpikeResponseModel",
    "NeurodynamicAdvice",
    "SoftmaxCompetitionStabilityProbe",
    "SoftmaxODEStabilityProbe",
    "SpikingActiveInferenceAdvisor",
    "SpikingActiveInferenceConfig",
    "get_spiking_active_inference_advisor",
]
