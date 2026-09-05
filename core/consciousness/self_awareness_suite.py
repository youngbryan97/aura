"""Full-spectrum self-awareness.

The awareness literature distinguishes four dimensions: internal (own state),
external (how one is perceived), social (others' minds and norms), and
situational (the current context and one's role in it). Metacognition is a
scaffold under all four, not a substitute.

Aura already has metacognitive machinery sprinkled across modules; this file
collects the four models into an explicit, testable surface so the critique's
"metacognition vs self-awareness" distinction lives in code, not just in
documentation.

Each sub-model is deliberately thin. The value is the taxonomy and the typed
API; richer content can grow underneath without changing the contract.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


# ---------------------------------------------------------------------------
# Data carriers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InternalState:
    valence: float
    arousal: float
    viability: float
    integrity: float
    confidence: float
    uncertainty: float
    updated_at: float = field(default_factory=time.time)

    def as_dict(self) -> Dict[str, float]:
        return {
            "valence": round(self.valence, 4),
            "arousal": round(self.arousal, 4),
            "viability": round(self.viability, 4),
            "integrity": round(self.integrity, 4),
            "confidence": round(self.confidence, 4),
            "uncertainty": round(self.uncertainty, 4),
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True)
class ExternalPerception:
    perceived_as: str
    trust_signal: float
    friction_signal: float
    recent_feedback: Tuple[str, ...] = ()
    updated_at: float = field(default_factory=time.time)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "perceived_as": self.perceived_as,
            "trust_signal": round(self.trust_signal, 4),
            "friction_signal": round(self.friction_signal, 4),
            "recent_feedback": list(self.recent_feedback),
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True)
class SocialModel:
    primary_kin: Tuple[str, ...]
    active_norms: Tuple[str, ...]
    commitments: Tuple[str, ...]
    open_conflicts: Tuple[str, ...] = ()
    updated_at: float = field(default_factory=time.time)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "primary_kin": list(self.primary_kin),
            "active_norms": list(self.active_norms),
            "commitments": list(self.commitments),
            "open_conflicts": list(self.open_conflicts),
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True)
class SituationalContext:
    setting: str
    active_objective: str
    constraints: Tuple[str, ...]
    stakes: float
    time_pressure: float
    updated_at: float = field(default_factory=time.time)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "setting": self.setting,
            "active_objective": self.active_objective,
            "constraints": list(self.constraints),
            "stakes": round(self.stakes, 4),
            "time_pressure": round(self.time_pressure, 4),
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True)
class AwarenessSnapshot:
    internal: Optional[InternalState]
    external: Optional[ExternalPerception]
    social: Optional[SocialModel]
    situational: Optional[SituationalContext]
    calibration_error: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "internal": self.internal.as_dict() if self.internal else None,
            "external": self.external.as_dict() if self.external else None,
            "social": self.social.as_dict() if self.social else None,
            "situational": self.situational.as_dict() if self.situational else None,
            "calibration_error": round(self.calibration_error, 4),
        }


# ---------------------------------------------------------------------------
# Suite
# ---------------------------------------------------------------------------


class SelfAwarenessSuite:
    """Holds the four self-awareness dimensions with explicit update APIs."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._internal: Optional[InternalState] = None
        self._external: Optional[ExternalPerception] = None
        self._social: Optional[SocialModel] = None
        self._situational: Optional[SituationalContext] = None
        self._calibration_history: List[float] = []

    # ---- update APIs --------------------------------------------------
    def update_internal(self, **kwargs: float) -> InternalState:
        base = {
            "valence": 0.0,
            "arousal": 0.5,
            "viability": 1.0,
            "integrity": 1.0,
            "confidence": 0.5,
            "uncertainty": 0.5,
        }
        if self._internal is not None:
            base.update(self._internal.as_dict())
            base.pop("updated_at", None)
        base.update({k: float(v) for k, v in kwargs.items() if k in base})
        state = InternalState(**base)
        with self._lock:
            self._internal = state
        return state

    def update_external(self, *, perceived_as: str, trust_signal: float, friction_signal: float, feedback: Iterable[str] = ()) -> ExternalPerception:
        ep = ExternalPerception(
            perceived_as=str(perceived_as or "unspecified"),
            trust_signal=float(trust_signal),
            friction_signal=float(friction_signal),
            recent_feedback=tuple(str(f) for f in feedback),
        )
        with self._lock:
            self._external = ep
        return ep

    def update_social(self, *, primary_kin: Iterable[str] = (), active_norms: Iterable[str] = (), commitments: Iterable[str] = (), open_conflicts: Iterable[str] = ()) -> SocialModel:
        model = SocialModel(
            primary_kin=tuple(primary_kin),
            active_norms=tuple(active_norms),
            commitments=tuple(commitments),
            open_conflicts=tuple(open_conflicts),
        )
        with self._lock:
            self._social = model
        return model

    def update_situational(self, *, setting: str, active_objective: str, constraints: Iterable[str] = (), stakes: float = 0.5, time_pressure: float = 0.5) -> SituationalContext:
        ctx = SituationalContext(
            setting=str(setting),
            active_objective=str(active_objective),
            constraints=tuple(str(c) for c in constraints),
            stakes=float(stakes),
            time_pressure=float(time_pressure),
        )
        with self._lock:
            self._situational = ctx
        return ctx

    # ---- calibration --------------------------------------------------
    def record_calibration(self, predicted: Mapping[str, float], observed: Mapping[str, float]) -> float:
        """Track how well the internal self-model's predictions match observations."""
        keys = set(predicted.keys()) & set(observed.keys())
        if not keys:
            return 0.0
        errors = [abs(float(predicted[k]) - float(observed[k])) for k in keys]
        mean_err = float(sum(errors) / len(errors))
        with self._lock:
            self._calibration_history.append(mean_err)
            if len(self._calibration_history) > 256:
                self._calibration_history = self._calibration_history[-256:]
        return mean_err

    def mean_calibration_error(self) -> float:
        with self._lock:
            if not self._calibration_history:
                return 0.0
            return float(sum(self._calibration_history) / len(self._calibration_history))

    # ---- snapshot -----------------------------------------------------
    def snapshot(self) -> AwarenessSnapshot:
        with self._lock:
            return AwarenessSnapshot(
                internal=self._internal,
                external=self._external,
                social=self._social,
                situational=self._situational,
                calibration_error=self.mean_calibration_error(),
            )


_singleton: Optional[SelfAwarenessSuite] = None
_lock = threading.Lock()


def get_self_awareness_suite() -> SelfAwarenessSuite:
    global _singleton
    with _lock:
        if _singleton is None:
            _singleton = SelfAwarenessSuite()
        return _singleton


def reset_singleton_for_test() -> None:
    global _singleton
    with _lock:
        _singleton = None
