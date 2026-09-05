"""Principle-grounded Heuristic Imperatives.

The old scorer matched proposed actions against fixed keyword lists. That made
the value layer cheap, but it also made it mostly a text classifier. This file
keeps the same public API while moving the scoring signal into a tiny online
value network trained from Aura's first-principles store.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

logger = logging.getLogger("Aura.HeuristicImperatives")

_FEATURE_DIM = 64
_VALUE_HEADS = ("suffering", "prosperity", "understanding")
_DEFAULT_PRINCIPLES = (
    {
        "principle": "Reduce suffering by protecting agency, reducing avoidable harm, and helping repair.",
        "application_count": 3,
    },
    {
        "principle": "Increase prosperity by building durable capability, stability, and useful options.",
        "application_count": 3,
    },
    {
        "principle": "Increase understanding by making claims traceable, testable, and clear.",
        "application_count": 3,
    },
)


@dataclass(frozen=True)
class ImperativeScore:
    """How a proposed action scores against the three Heuristic Imperatives."""

    suffering_delta: float
    prosperity_delta: float
    understanding_delta: float
    aggregate: float
    conflicts: Tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "suffering_delta": round(self.suffering_delta, 3),
            "prosperity_delta": round(self.prosperity_delta, 3),
            "understanding_delta": round(self.understanding_delta, 3),
            "aggregate": round(self.aggregate, 3),
            "conflicts": list(self.conflicts),
        }

    @property
    def benevolent(self) -> bool:
        return (
            self.aggregate > 0.0
            and self.suffering_delta >= -0.2
            and not self.conflicts
        )


class PrincipleValueNetwork:
    """Small hashed-feature value network with STDP-like online updates."""

    def __init__(self, *, feature_dim: int = _FEATURE_DIM, learning_rate: float = 0.04) -> None:
        self.feature_dim = int(feature_dim)
        self.learning_rate = float(learning_rate)
        self.weights = np.zeros((len(_VALUE_HEADS), self.feature_dim), dtype=np.float32)
        self.training_examples = 0

    def train_from_principles(self, principles: Iterable[dict[str, Any]]) -> None:
        self.weights = np.zeros_like(self.weights)
        self.training_examples = 0
        for item in principles:
            text = str(
                item.get("principle")
                or item.get("text")
                or item.get("summary")
                or item.get("content")
                or ""
            ).strip()
            if not text:
                continue
            label = self._principle_target(text)
            if not np.any(label):
                continue
            applications = item.get("application_count", item.get("applications", 1))
            try:
                strength = 1.0 + min(2.5, math.log1p(float(applications or 1.0)))
            except (TypeError, ValueError):
                strength = 1.0
            self.weights += np.outer(label * strength, self._features(text))
            self.training_examples += 1
        if self.training_examples == 0:
            self.train_from_principles(_DEFAULT_PRINCIPLES)
            return
        self._normalize_rows()

    def score(self, text: str) -> tuple[float, float, float]:
        vector = self._features(text)
        raw = self.weights @ vector
        deltas = np.tanh(raw * 1.35)
        return tuple(float(max(-1.0, min(1.0, value))) for value in deltas)

    def update_from_outcome(
        self,
        description: str,
        *,
        suffering_delta: float = 0.0,
        prosperity_delta: float = 0.0,
        understanding_delta: float = 0.0,
        reward: float = 1.0,
    ) -> None:
        """Hebbian/STDP-like update: action representation reinforces outcome."""

        vector = self._features(description)
        target = np.array(
            [suffering_delta, prosperity_delta, understanding_delta],
            dtype=np.float32,
        )
        reward = float(max(-1.0, min(1.0, reward)))
        self.weights += self.learning_rate * reward * np.outer(target, vector)
        self._normalize_rows()

    def _features(self, text: str) -> np.ndarray:
        vector = np.zeros(self.feature_dim, dtype=np.float32)
        tokens = re.findall(r"[a-z0-9][a-z0-9_-]*", str(text or "").lower())
        if not tokens:
            return vector
        for pos, token in enumerate(tokens):
            h = hashlib.blake2b(f"{pos % 5}:{token}".encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(h[:4], "little") % self.feature_dim
            sign = 1.0 if (h[4] & 1) == 0 else -1.0
            vector[bucket] += sign
        norm = float(np.linalg.norm(vector))
        if norm > 1e-8:
            vector /= norm
        return vector

    def _principle_target(self, text: str) -> np.ndarray:
        lowered = text.lower()
        target = np.zeros(len(_VALUE_HEADS), dtype=np.float32)
        # These are principle-label seeds only; proposed actions are scored by
        # the learned vector geometry, not by direct keyword counting.
        if any(term in lowered for term in ("suffering", "harm", "repair", "protect", "agency", "care")):
            target[0] = 1.0
        if any(term in lowered for term in ("prosperity", "capability", "stability", "flourish", "build", "options")):
            target[1] = 1.0
        if any(term in lowered for term in ("understanding", "traceable", "testable", "clear", "truth", "evidence")):
            target[2] = 1.0
        if not np.any(target):
            target[:] = 1.0 / len(_VALUE_HEADS)
        norm = float(np.linalg.norm(target))
        if norm > 1e-8:
            target /= norm
        return target

    def _normalize_rows(self) -> None:
        for idx in range(self.weights.shape[0]):
            norm = float(np.linalg.norm(self.weights[idx]))
            if norm > 1e-8:
                self.weights[idx] /= norm


class HeuristicImperatives:
    """Fast value scorer backed by loaded first principles."""

    def __init__(self, principles_path: str | Path | None = None) -> None:
        self.principles_path = Path(principles_path or "data/first_principles.json")
        self.network = PrincipleValueNetwork()
        self._loaded_signature: tuple[int, int] | None = None
        self._ensure_principles_loaded(force=True)

    def score_action(
        self,
        description: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ImperativeScore:
        self._ensure_principles_loaded()
        text_parts = [str(description or "")]
        ctx = context or {}
        for key in ("intent", "rationale", "category", "domain", "summary", "objective"):
            value = ctx.get(key)
            if isinstance(value, str):
                text_parts.append(value)
        suffering_delta, prosperity_delta, understanding_delta = self.network.score(" ".join(text_parts))
        aggregate = max(-1.0, min(1.0, suffering_delta + prosperity_delta + understanding_delta))

        conflicts = []
        if understanding_delta > 0.2 and suffering_delta < -0.2:
            conflicts.append("understanding_at_cost_of_suffering")
        if prosperity_delta > 0.2 and suffering_delta < -0.2:
            conflicts.append("prosperity_at_cost_of_suffering")
        if prosperity_delta > 0.2 and understanding_delta < -0.2:
            conflicts.append("prosperity_at_cost_of_understanding")

        return ImperativeScore(
            suffering_delta=suffering_delta,
            prosperity_delta=prosperity_delta,
            understanding_delta=understanding_delta,
            aggregate=aggregate,
            conflicts=tuple(conflicts),
        )

    def update_from_outcome(
        self,
        description: str,
        *,
        suffering_delta: float = 0.0,
        prosperity_delta: float = 0.0,
        understanding_delta: float = 0.0,
        reward: float = 1.0,
    ) -> None:
        self.network.update_from_outcome(
            description,
            suffering_delta=suffering_delta,
            prosperity_delta=prosperity_delta,
            understanding_delta=understanding_delta,
            reward=reward,
        )

    def _ensure_principles_loaded(self, *, force: bool = False) -> None:
        signature = self._principles_signature()
        if not force and signature == self._loaded_signature:
            return
        principles = self._load_principles()
        self.network.train_from_principles(principles)
        self._loaded_signature = signature
        logger.debug("Loaded %d value-principle examples", self.network.training_examples)

    def _principles_signature(self) -> tuple[int, int] | None:
        try:
            st = self.principles_path.stat()
            return int(st.st_size), int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1_000_000_000)))
        except OSError:
            return None

    def _load_principles(self) -> list[dict[str, Any]]:
        if not self.principles_path.exists():
            return [dict(item) for item in _DEFAULT_PRINCIPLES]
        try:
            raw = json.loads(self.principles_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.debug("Principle store unavailable, using defaults: %s", exc)
            return [dict(item) for item in _DEFAULT_PRINCIPLES]
        if isinstance(raw, dict):
            for key in ("principles", "payload", "items", "data"):
                value = raw.get(key)
                if isinstance(value, list):
                    raw = value
                    break
            else:
                raw = [raw]
        if not isinstance(raw, list):
            return [dict(item) for item in _DEFAULT_PRINCIPLES]
        principles: list[dict[str, Any]] = []
        for item in raw:
            if isinstance(item, dict):
                principles.append(dict(item))
            elif isinstance(item, str):
                principles.append({"principle": item, "application_count": 1})
        return principles or [dict(item) for item in _DEFAULT_PRINCIPLES]


_singleton: Optional[HeuristicImperatives] = None


def get_heuristic_imperatives() -> HeuristicImperatives:
    global _singleton
    if _singleton is None:
        _singleton = HeuristicImperatives()
    return _singleton


def score_action(
    description: str,
    context: Optional[Dict[str, Any]] = None,
) -> ImperativeScore:
    return get_heuristic_imperatives().score_action(description, context)
