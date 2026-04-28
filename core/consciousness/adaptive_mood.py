"""Adaptive mood coefficients.

Addresses the tautological-correlation critique: the hardcoded formula
``valence = 0.25*DA + 0.30*5HT + 0.20*END + 0.10*OXY - 0.45*CORT`` guarantees
correlation with its inputs by definition. This module replaces the fixed
weights with online-learned coefficients that adapt based on outcome feedback,
so the mapping becomes a learned prediction rather than a programmed identity.

The coefficients are persisted across restarts, bounded, and subject to
decay so a pathological feedback signal cannot drive them to extremes.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Dict, Mapping, Optional

import numpy as np


DEFAULT_MOODS = (
    "valence",
    "arousal",
    "motivation",
    "sociality",
    "stress",
    "calm",
    "wakefulness",
)

DEFAULT_CHEMICALS = (
    "glutamate",
    "gaba",
    "dopamine",
    "serotonin",
    "norepinephrine",
    "acetylcholine",
    "endorphin",
    "oxytocin",
    "cortisol",
    "orexin",
)


# Initial coefficients match the legacy formula so behavior is continuous at
# startup but can adapt from there.
_SEED_WEIGHTS: Dict[str, Dict[str, float]] = {
    "valence": {"dopamine": 0.25, "serotonin": 0.30, "endorphin": 0.20, "oxytocin": 0.10, "cortisol": -0.45},
    "arousal": {"norepinephrine": 0.30, "dopamine": 0.15, "cortisol": 0.20, "glutamate": 0.15, "orexin": 0.20, "gaba": -0.40, "serotonin": -0.10},
    "motivation": {"dopamine": 0.40, "norepinephrine": 0.15, "orexin": 0.20, "gaba": -0.25},
    "sociality": {"oxytocin": 0.60, "serotonin": 0.20, "endorphin": 0.10},
    "stress": {"cortisol": 0.50, "norepinephrine": 0.30, "serotonin": -0.20, "gaba": -0.20},
    "calm": {"gaba": 0.35, "serotonin": 0.30, "endorphin": 0.10, "norepinephrine": -0.20, "cortisol": -0.25, "glutamate": -0.10},
    "wakefulness": {"orexin": 0.50, "norepinephrine": 0.20, "glutamate": 0.15, "gaba": -0.30},
}

_SEED_BIAS: Dict[str, float] = {
    "valence": -0.10,
    "arousal": 0.0,
    "motivation": 0.0,
    "sociality": 0.0,
    "stress": 0.0,
    "calm": 0.0,
    "wakefulness": 0.0,
}

MAX_ABS_COEF = 1.25
MIN_LR = 5e-4
MAX_LR = 5e-2


class AdaptiveMoodCoefficients:
    """Online-learned mood weights with persistence."""

    def __init__(
        self,
        db_path: Optional[str | Path] = None,
        *,
        chemicals: tuple[str, ...] = DEFAULT_CHEMICALS,
        moods: tuple[str, ...] = DEFAULT_MOODS,
        learning_rate: float = 5e-3,
        weight_decay: float = 1e-4,
    ) -> None:
        self.chemicals = tuple(chemicals)
        self.moods = tuple(moods)
        self.learning_rate = float(max(MIN_LR, min(MAX_LR, learning_rate)))
        self.weight_decay = float(max(0.0, min(1e-2, weight_decay)))
        self._lock = threading.RLock()
        self._weights: Dict[str, np.ndarray] = {
            mood: np.array([_SEED_WEIGHTS.get(mood, {}).get(ch, 0.0) for ch in self.chemicals], dtype=np.float64)
            for mood in self.moods
        }
        self._bias: Dict[str, float] = {mood: float(_SEED_BIAS.get(mood, 0.0)) for mood in self.moods}
        self._updates: Dict[str, int] = {mood: 0 for mood in self.moods}
        self._last_prediction: Dict[str, float] = {mood: 0.0 for mood in self.moods}

        if db_path is not None:
            self._db_path = Path(db_path)
            get_task_tracker().create_task(get_storage_gateway().create_dir(self._db_path.parent, cause='AdaptiveMoodCoefficients.__init__'))
            self._init_db()
            self._load()
        else:
            self._db_path = None

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS adaptive_mood_weights (
                    mood TEXT PRIMARY KEY,
                    weights TEXT NOT NULL,
                    bias REAL NOT NULL,
                    updates INTEGER NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )

    def _load(self) -> None:
        if self._db_path is None:
            return
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute("SELECT mood, weights, bias, updates FROM adaptive_mood_weights").fetchall()
        for mood, weights_json, bias, updates in rows:
            if mood not in self._weights:
                continue
            try:
                w = json.loads(weights_json)
                vec = np.array([float(w.get(ch, 0.0)) for ch in self.chemicals], dtype=np.float64)
                self._weights[mood] = vec
                self._bias[mood] = float(bias)
                self._updates[mood] = int(updates)
            except Exception:
                continue

    def _save(self, mood: str) -> None:
        if self._db_path is None:
            return
        w = {ch: float(val) for ch, val in zip(self.chemicals, self._weights[mood])}
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                INSERT INTO adaptive_mood_weights (mood, weights, bias, updates, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(mood) DO UPDATE SET
                    weights = excluded.weights,
                    bias = excluded.bias,
                    updates = excluded.updates,
                    updated_at = excluded.updated_at
                """,
                (mood, json.dumps(w, sort_keys=True), float(self._bias[mood]), int(self._updates[mood]), time.time()),
            )

    def predict(self, chemicals: Mapping[str, float]) -> Dict[str, float]:
        """Current mood prediction from chemistry — the learned formula."""
        x = np.array([float(chemicals.get(ch, 0.0)) for ch in self.chemicals], dtype=np.float64)
        out: Dict[str, float] = {}
        with self._lock:
            for mood in self.moods:
                value = float(np.dot(self._weights[mood], x) + self._bias[mood])
                out[mood] = value
                self._last_prediction[mood] = value
        return out

    def update_from_outcome(
        self,
        chemicals: Mapping[str, float],
        observed: Mapping[str, float],
    ) -> Dict[str, float]:
        """Gradient-descent step toward observed mood signal.

        ``observed`` is the empirical mood measure derived from behavior or
        downstream signals (e.g. action success, affect report, homeostatic
        error). The learned coefficients move toward whatever empirically
        predicts the outcome, rather than being fixed by the author.
        """
        x = np.array([float(chemicals.get(ch, 0.0)) for ch in self.chemicals], dtype=np.float64)
        residuals: Dict[str, float] = {}
        with self._lock:
            for mood in self.moods:
                target = float(observed.get(mood, float("nan")))
                if not np.isfinite(target):
                    continue
                pred = float(np.dot(self._weights[mood], x) + self._bias[mood])
                error = target - pred
                grad = -error * x
                self._weights[mood] -= self.learning_rate * grad
                self._weights[mood] *= (1.0 - self.weight_decay)
                self._weights[mood] = np.clip(self._weights[mood], -MAX_ABS_COEF, MAX_ABS_COEF)
                self._bias[mood] = float(np.clip(
                    self._bias[mood] + self.learning_rate * error,
                    -MAX_ABS_COEF,
                    MAX_ABS_COEF,
                ))
                self._updates[mood] += 1
                residuals[mood] = error
                self._save(mood)
        return residuals

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        with self._lock:
            return {
                mood: {
                    "bias": self._bias[mood],
                    "updates": self._updates[mood],
                    "weights": {ch: float(val) for ch, val in zip(self.chemicals, self._weights[mood])},
                }
                for mood in self.moods
            }

    def total_updates(self) -> int:
        with self._lock:
            return int(sum(self._updates.values()))

    def drift_from_seed(self) -> float:
        """L2 distance from the initial seed weights across all moods."""
        with self._lock:
            total = 0.0
            for mood in self.moods:
                seed = np.array([_SEED_WEIGHTS.get(mood, {}).get(ch, 0.0) for ch in self.chemicals], dtype=np.float64)
                total += float(np.linalg.norm(self._weights[mood] - seed))
            return total


_singleton: Optional[AdaptiveMoodCoefficients] = None
_lock = threading.Lock()


def get_adaptive_mood(db_path: Optional[str | Path] = None) -> AdaptiveMoodCoefficients:
    global _singleton
    with _lock:
        if _singleton is None:
            if db_path is None:
                try:
                    from core.config import config
                    db_path = Path(config.paths.data_dir) / "adaptive_mood.sqlite3"
                except Exception:
                    db_path = Path.home() / ".aura" / "adaptive_mood.sqlite3"
            _singleton = AdaptiveMoodCoefficients(db_path=db_path)
        return _singleton


def reset_singleton_for_test() -> None:
    global _singleton
    with _lock:
        _singleton = None
