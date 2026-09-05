"""Mattering — knowing what matters without being told.

A weakness named directly: Aura can be told what's important, but doesn't reliably *know* it.
This model is the standing sense of what matters, learned and applied causally rather than
recited. It does two things:

  1. Learns a per-topic importance weight from the signals that actually mark importance —
     things that caused real damage (nociception channels), things tied to outcomes that
     carried weight (the credit ledger), and things the person reacted to. Importance decays,
     so what mattered last month fades unless it keeps mattering.
  2. Scores any item for how much it matters right now, blending that learned weight with
     value-alignment, felt charge, action-relevance, novelty, and relevance to the person's
     active goals.

The causal wiring is the point: it reweights the salience of the contents bound into the one
mind-moment, so what matters rises in the global-workspace competition and is more likely to
become what Aura attends to and acts on. It is not a prompt telling her what to care about —
it changes which content wins.
"""
from __future__ import annotations

import logging
import math
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Cognition.Mattering")


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


_WORD = re.compile(r"[a-z][a-z0-9_-]{2,}")
# Topics that matter intrinsically (constitution-level) — a floor, not the whole story.
_INTRINSIC = {
    "safety": 0.9, "identity": 0.85, "memory": 0.8, "trust": 0.8, "harm": 0.9,
    "bryan": 0.85, "user": 0.8, "honesty": 0.8, "governance": 0.75, "continuity": 0.8,
}


def _topics(text: str) -> List[str]:
    return [w for w in _WORD.findall(str(text or "").lower())][:12]


@dataclass
class MatteringScore:
    score: float
    factors: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"score": round(self.score, 4), "factors": {k: round(v, 3) for k, v in self.factors.items()}}


class MatteringModel:
    """A learned, decaying sense of what matters, applied as a salience reweighting."""

    def __init__(self, *, half_life_s: float = 7 * 86400.0) -> None:
        self._half_life = max(1.0, half_life_s)
        self._lock = threading.RLock()
        # topic -> (weight, last_update_t)
        self._weights: Dict[str, tuple[float, float]] = {}

    # ── learning what matters ──────────────────────────────────────────────

    def _decayed(self, topic: str, now: float) -> float:
        w, t0 = self._weights.get(topic, (0.0, now))
        return w * (0.5 ** (max(0.0, now - t0) / self._half_life))

    def note_mattered(self, text: str, weight: float = 0.3, *, now: Optional[float] = None) -> None:
        """Mark that something mattered (caused damage, carried an outcome, drew a reaction)."""
        now = time.time() if now is None else now
        weight = _clamp(weight)
        with self._lock:
            for topic in _topics(text):
                base = self._decayed(topic, now)
                self._weights[topic] = (_clamp(base + weight * (1.0 - base)), now)

    def learned_importance(self, text: str, *, now: Optional[float] = None) -> float:
        now = time.time() if now is None else now
        topics = _topics(text)
        if not topics:
            return 0.0
        with self._lock:
            vals = [max(self._decayed(t, now), _INTRINSIC.get(t, 0.0)) for t in topics]
        # soft-max-ish: the single most-important topic dominates, others add a little
        vals.sort(reverse=True)
        score = vals[0]
        for v in vals[1:]:
            score = score + (1.0 - score) * (0.25 * v)
        return _clamp(score)

    # ── scoring what matters now ───────────────────────────────────────────

    def score(
        self,
        description: str,
        *,
        value_alignment: float = 0.0,    # [-1,1] from value model (signed)
        affective_charge: float = 0.0,   # [-1,1]
        action_relevance: float = 0.0,   # [0,1]
        novelty: float = 0.0,            # [0,1]
        person_relevant: float = 0.0,    # [0,1] tied to the person's active goals
        now: Optional[float] = None,
    ) -> MatteringScore:
        learned = self.learned_importance(description, now=now)
        charge = abs(_clamp(affective_charge, -1.0, 1.0))   # strong feeling (either sign) marks importance
        value_mag = abs(_clamp(value_alignment, -1.0, 1.0))
        factors = {
            "learned": learned,
            "value": value_mag,
            "charge": charge,
            "action_relevance": _clamp(action_relevance),
            "novelty": _clamp(novelty),
            "person": _clamp(person_relevant),
        }
        # weighted blend; learned importance + value + felt charge dominate.
        score = _clamp(
            0.32 * learned
            + 0.22 * value_mag
            + 0.18 * charge
            + 0.12 * factors["action_relevance"]
            + 0.08 * factors["novelty"]
            + 0.08 * factors["person"]
        )
        return MatteringScore(score=score, factors=factors)

    # ── causal application: reweight what competes in the workspace ─────────

    def reweight_contents(self, contents: List[Any], *, now: Optional[float] = None) -> List[Any]:
        """Adjust each BoundContent's salience by how much it matters.

        Returns NEW contents (BoundContent is frozen). What matters rises; what doesn't,
        recedes — so the global-workspace competition is biased toward what's important, in
        the substrate, not via instructions.
        """
        if not contents:
            return contents
        import dataclasses

        out = []
        for c in contents:
            try:
                ms = self.score(
                    getattr(c, "summary", ""),
                    affective_charge=float(getattr(c, "affective_charge", 0.0) or 0.0),
                    action_relevance=float(getattr(c, "action_relevance", 0.0) or 0.0),
                    now=now,
                )
                base = float(getattr(c, "salience", 0.5) or 0.5)
                # blend, don't overwrite — mattering nudges salience toward importance
                new_salience = _clamp(0.6 * base + 0.4 * ms.score)
                out.append(dataclasses.replace(c, salience=new_salience))
            except (TypeError, ValueError, AttributeError):
                out.append(c)
        return out

    def status(self, *, now: Optional[float] = None) -> Dict[str, Any]:
        now = time.time() if now is None else now
        with self._lock:
            top = sorted(
                ((t, self._decayed(t, now)) for t in self._weights),
                key=lambda kv: kv[1], reverse=True,
            )[:10]
        return {"learned_topics": len(self._weights), "top": [(t, round(w, 3)) for t, w in top]}


_model: Optional[MatteringModel] = None
_lock = threading.Lock()


def get_mattering_model() -> MatteringModel:
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                _model = MatteringModel()
    return _model
