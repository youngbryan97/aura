"""Affect grounding — affect labels from sustained multi-signal conditions, not a float threshold.

The critique, bluntly: "She cannot feel boredom — she checks a float < -0.2 and labels it
boredom." The honest problem isn't the absence of phenomenology (you can't patch that in); it's
that the *labels* are ungrounded — one scalar crossing one line names a feeling, with no temporal
persistence, no corroborating signals, and no account of *why*.

This grounds them. Each affect is a condition over a rolling window of real signals (novelty,
prediction error, valence, arousal, pain, social threat, control, idle time), and it only counts
when it is (a) supported by several signals at once and (b) *sustained* across the window — a
single transient sample can't name a feeling. Every assessment carries an intensity, a confidence,
and the explicit factors that produced it ("novelty low and sustained; prediction-error low;
arousal low"). So "boredom" stops being `x < -0.2` and becomes "little new, little to predict,
and it has been that way for a while."

This does not manufacture an inner life. It makes the affect labels causal and explainable —
grounded in multiple signals over time rather than asserted from one threshold — which is the
concrete, buildable half of the critique.
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger("Affect.Grounding")


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


# Signals the engine tracks. Absent signals default to a neutral value at read time.
_SIGNALS = ("novelty", "prediction_error", "valence", "arousal", "pain",
            "social_threat", "control", "idle")
_NEUTRAL = {"novelty": 0.3, "prediction_error": 0.3, "valence": 0.0, "arousal": 0.0,
            "pain": 0.0, "social_threat": 0.0, "control": 0.5, "idle": 0.0}


@dataclass
class GroundedAffect:
    label: str
    intensity: float
    factors: List[str]
    confidence: float
    persistence: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "label": self.label,
            "intensity": round(self.intensity, 3),
            "confidence": round(self.confidence, 3),
            "persistence": round(self.persistence, 3),
            "factors": self.factors,
        }


# A condition: given the window helpers, return (raw_intensity, dominant_signal, factors).
Condition = Callable[["AffectGroundingEngine"], Tuple[float, str, List[str]]]


class AffectGroundingEngine:
    """Derives explainable affect states from sustained, multi-signal conditions."""

    def __init__(self, *, window: int = 16, min_samples: int = 4) -> None:
        self._window = window
        self._min_samples = min_samples
        self._buf: Dict[str, Deque[float]] = {s: deque(maxlen=window) for s in _SIGNALS}
        self._conditions: Dict[str, Condition] = {
            "boredom": self._boredom,
            "curiosity": self._curiosity,
            "flow": self._flow,
            "anxiety": self._anxiety,
            "frustration": self._frustration,
            "contentment": self._contentment,
        }

    # ── signal intake ─────────────────────────────────────────────────────

    def observe(self, **signals: float) -> None:
        """Push one sample of any subset of the tracked signals into the rolling window."""
        for name in _SIGNALS:
            if name in signals and signals[name] is not None:
                self._buf[name].append(float(signals[name]))

    def gather(self) -> "AffectGroundingEngine":
        """Best-effort: pull live signals (nociception, world-model surprise) into one sample."""
        sample: Dict[str, float] = {}
        try:
            from core.affect.nociception import get_nociception_engine
            sample["pain"] = _clamp(get_nociception_engine().nociceptive_pressure())
        except (ImportError, AttributeError, RuntimeError, OSError, ValueError, TypeError) as exc:
            from core.runtime.errors import record_degradation
            record_degradation("affect_grounding", exc, severity="debug")
        try:
            from core.container import ServiceContainer
            wm = ServiceContainer.get("world_model", default=None)
            if wm is not None and hasattr(wm, "surprise"):
                s = wm.surprise()
                if s is not None:
                    sample["prediction_error"] = _clamp(float(s))
        except (ImportError, AttributeError, RuntimeError, OSError, ValueError, TypeError) as exc:
            from core.runtime.errors import record_degradation
            record_degradation("affect_grounding", exc, severity="debug")
        if sample:
            self.observe(**sample)
        return self

    # ── window helpers ────────────────────────────────────────────────────

    def _mean(self, sig: str) -> float:
        buf = self._buf[sig]
        return sum(buf) / len(buf) if buf else _NEUTRAL[sig]

    def _samples(self) -> int:
        return max((len(b) for b in self._buf.values()), default=0)

    def _persisted(self, sig: str, predicate: Callable[[float], bool]) -> float:
        """Fraction of the window in which ``sig`` satisfied ``predicate`` (temporal persistence)."""
        buf = self._buf[sig]
        if not buf:
            return 0.0
        return sum(1 for v in buf if predicate(v)) / len(buf)

    # ── assessment ────────────────────────────────────────────────────────

    def assess(self) -> List[GroundedAffect]:
        """Evaluate all affect conditions; return the active ones, strongest first."""
        samples = self._samples()
        if samples < self._min_samples:
            return []  # not enough history to ground any label — refuse to assert one
        out: List[GroundedAffect] = []
        # Confidence grows as the window fills (more evidence behind the read).
        evidence_conf = _clamp(samples / self._window)
        for label, cond in self._conditions.items():
            raw, dom_sig, factors = cond(self)
            if raw <= 0.0 or not factors:
                continue
            persistence = self._persisted(dom_sig, self._dom_predicate(label, dom_sig))
            intensity = _clamp(raw * (0.4 + 0.6 * persistence))  # transient reads are discounted
            if intensity < 0.15:
                continue
            out.append(GroundedAffect(label=label, intensity=intensity, factors=factors,
                                      confidence=_clamp(evidence_conf * (0.5 + 0.5 * persistence)),
                                      persistence=persistence))
        out.sort(key=lambda a: a.intensity, reverse=True)
        return out

    def dominant(self) -> Optional[GroundedAffect]:
        affects = self.assess()
        return affects[0] if affects else None

    @staticmethod
    def _dom_predicate(label: str, dom_sig: str) -> Callable[[float], bool]:
        # The persistence test for each affect's dominant signal.
        if label == "boredom":
            return lambda v: v < 0.3        # novelty stayed low
        if label == "curiosity":
            return lambda v: v > 0.45       # novelty stayed elevated
        if label == "flow":
            return lambda v: 0.2 <= v <= 0.7  # prediction error in the learnable band
        if label == "anxiety":
            return lambda v: v > 0.4        # arousal stayed up
        if label == "frustration":
            return lambda v: v > 0.4        # pain/blockage stayed up
        if label == "contentment":
            return lambda v: v > 0.0        # valence stayed positive
        return lambda v: True

    # ── conditions: each returns (raw_intensity, dominant_signal, factors) ─

    def _boredom(self, _) -> Tuple[float, str, List[str]]:
        nov, pe, ar = self._mean("novelty"), self._mean("prediction_error"), self._mean("arousal")
        idle = self._mean("idle")
        factors: List[str] = []
        if nov < 0.3:
            factors.append("little novel input")
        if pe < 0.3:
            factors.append("little to predict")
        if ar < 0.0:
            factors.append("low arousal")
        if idle > 0.4:
            factors.append("idle for a while")
        if len(factors) < 2:           # needs MULTIPLE signals, not one threshold
            return 0.0, "novelty", []
        raw = _clamp(0.4 * (1 - nov) + 0.3 * (1 - pe) + 0.2 * max(0, -ar) + 0.2 * idle)
        return raw, "novelty", factors

    def _curiosity(self, _) -> Tuple[float, str, List[str]]:
        nov, val, pe = self._mean("novelty"), self._mean("valence"), self._mean("prediction_error")
        factors: List[str] = []
        if nov > 0.45:
            factors.append("novelty present")
        if val > 0.1:
            factors.append("positive valence")
        if pe > 0.3:
            factors.append("something to learn")
        if len(factors) < 2:
            return 0.0, "novelty", []
        raw = _clamp(0.5 * nov + 0.3 * max(0, val) + 0.3 * pe)
        return raw, "novelty", factors

    def _flow(self, _) -> Tuple[float, str, List[str]]:
        pe, ar, val = self._mean("prediction_error"), self._mean("arousal"), self._mean("valence")
        ctrl = self._mean("control")
        factors: List[str] = []
        if 0.2 <= pe <= 0.7:
            factors.append("challenge in the learnable band")
        if ar > 0.2:
            factors.append("engaged arousal")
        if val > 0.1:
            factors.append("positive valence")
        if ctrl > 0.5:
            factors.append("in control")
        if len(factors) < 3:
            return 0.0, "prediction_error", []
        raw = _clamp(0.4 * (1 - abs(pe - 0.45) * 2) + 0.3 * ar + 0.2 * max(0, val) + 0.2 * ctrl)
        return raw, "prediction_error", factors

    def _anxiety(self, _) -> Tuple[float, str, List[str]]:
        ar, val, ctrl = self._mean("arousal"), self._mean("valence"), self._mean("control")
        threat = self._mean("social_threat")
        factors: List[str] = []
        if ar > 0.4:
            factors.append("high arousal")
        if val < -0.1:
            factors.append("negative valence")
        if ctrl < 0.4:
            factors.append("low control")
        if threat > 0.4:
            factors.append("perceived threat")
        if len(factors) < 2:
            return 0.0, "arousal", []
        raw = _clamp(0.4 * ar + 0.3 * max(0, -val) + 0.3 * (1 - ctrl) + 0.2 * threat)
        return raw, "arousal", factors

    def _frustration(self, _) -> Tuple[float, str, List[str]]:
        pain, ctrl, val = self._mean("pain"), self._mean("control"), self._mean("valence")
        factors: List[str] = []
        if pain > 0.4:
            factors.append("repeated pain/blockage")
        if ctrl < 0.4:
            factors.append("low control")
        if val < -0.1:
            factors.append("negative valence")
        if len(factors) < 2:
            return 0.0, "pain", []
        raw = _clamp(0.5 * pain + 0.3 * (1 - ctrl) + 0.2 * max(0, -val))
        return raw, "pain", factors

    def _contentment(self, _) -> Tuple[float, str, List[str]]:
        val, ar, pain = self._mean("valence"), self._mean("arousal"), self._mean("pain")
        nov = self._mean("novelty")
        # Contentment is a *restful* state: high novelty means engagement (curiosity/flow), not calm.
        if nov >= 0.4:
            return 0.0, "valence", []
        factors: List[str] = []
        if val > 0.1:
            factors.append("positive valence")
        if ar < 0.2:
            factors.append("low arousal")
        if pain < 0.2:
            factors.append("no pain")
        if len(factors) < 3:
            return 0.0, "valence", []
        raw = _clamp(0.5 * max(0, val) + 0.3 * (1 - abs(ar)) + 0.2 * (1 - pain))
        return raw, "valence", factors

    def get_health(self) -> Dict[str, object]:
        dom = self.dominant()
        return {
            "module": "AffectGroundingEngine",
            "samples": self._samples(),
            "dominant": dom.to_dict() if dom else None,
            "status": "online",
        }


_instance: Optional[AffectGroundingEngine] = None


def get_affect_grounding_engine() -> AffectGroundingEngine:
    global _instance
    if _instance is None:
        _instance = AffectGroundingEngine()
    return _instance
