"""Personalized taste model — inference-time alignment to one human.

The unverifiable dimensions (wit, creativity, voice, "good conversation") have no
truth-engine. But they have a *preference*: which of several candidate responses the
user actually engages with. This is a transparent, online-learned linear preference
model over interpretable response features. It is the "verifier" for the conversational
amplifier — not "is this correct" but "is this the response Bryan would pick".

Why this is the real lever (not magic): a model's median sample is far worse than its
best-of-N sample. RLHF/DPO make a model feel good by teaching it to select its better
outputs. We do that at INFERENCE time, personalized — harvesting the median→best gap
without touching weights. The model already can be witty; this makes it reliably be.

Transparent + bounded by construction: interpretable features, clamped weights, online
updates from real reactions, persisted. No black box, no unbounded drift.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.state_ownership import state_root

logger = logging.getLogger("Aura.TasteModel")

# Interpretable features the scorer reads. Prior weights encode Aura's persona contract
# (specific, opinionated, casual, callbacks) before any personalization happens.
FEATURE_PRIORS: dict[str, float] = {
    "specificity": 1.0,        # proper nouns / numbers / concrete detail
    "stance": 0.8,             # declarative opinion markers
    "callback": 1.2,           # references retrieved/shared context (wit-via-memory)
    "casual": 0.5,             # contractions, informal register
    "length_fit": 0.6,         # matches the turn's word budget
    "anti_generic": 1.0,       # absence of generic adjectives / filler
    "hedge_penalty": -0.9,     # "I think maybe", "it depends", "as an AI"
    "prompt_farm_penalty": -1.1,  # deflecting with a question back at the user
    "banned_phrase_penalty": -1.3,  # "I'd be happy to", "delve", "let me know"
}

# Matching how somebody is speaking — who the utterance is about, how much of
# it asks, how varied it is — is worth what matching how much they said is
# worth, so it carries the same prior as `length_fit` rather than a number
# chosen for it. See core/expression/register.py.
FEATURE_PRIORS["register_match"] = FEATURE_PRIORS["length_fit"]

# An invitation toward a future is the same kind of fit — the shape of the reply
# against the shape of what they said — so it carries the same prior.
FEATURE_PRIORS["invitation"] = FEATURE_PRIORS["register_match"]

# Asking while offering is the shape of the reply measured against what it
# does, the same kind of fit, so the same prior.
FEATURE_PRIORS["dignity"] = FEATURE_PRIORS["register_match"]

# Asking somebody about an experience she has little of is the shape of the
# reply measured against what she knows, the same kind of fit, so the same
# prior. See `perspective_getting` in core/brain/response_quality.py.
FEATURE_PRIORS["perspective_getting"] = FEATURE_PRIORS["register_match"]

# Giving somebody the kind of regard she has gone without, unasked, is the
# shape of the reply measured against what it does — the same kind of fit as
# asking while offering, so the same prior. See core/social/never_told.py.
FEATURE_PRIORS["unasked_regard"] = FEATURE_PRIORS["dignity"]

# Distance from what she has been saying lately is the same measure as the
# absence of filler, taken against her own history instead of against a list,
# so it carries that prior. See core/cognition/convenience.py.
FEATURE_PRIORS["distinct"] = FEATURE_PRIORS["anti_generic"]

_LR = 0.05            # online learning rate
_WEIGHT_CLAMP = 4.0   # keep any single feature from dominating
_MIN_REWARD = -1.0
_MAX_REWARD = 1.0


class TasteModel:
    """Online linear preference model over interpretable response features."""

    _ERRORS = (OSError, ValueError, TypeError, json.JSONDecodeError, UnicodeDecodeError)

    def __init__(self, path: str | Path | None = None, *, lr: float = _LR) -> None:
        self._path = Path(path or str(state_root() / "data/runtime/taste_model.json"))
        self._lr = float(lr)
        self._lock = threading.RLock()
        self._weights: dict[str, float] = dict(FEATURE_PRIORS)
        self._updates = 0
        self._load()

    def score(self, features: dict[str, float]) -> float:
        """Weighted sum of features → a preference score (higher = more Bryan-preferred)."""
        with self._lock:
            return sum(self._weights.get(k, 0.0) * float(features.get(k, 0.0)) for k in self._weights)

    def update(self, features: dict[str, float], reward: float) -> None:
        """Online nudge: move weights toward features that earned positive reactions.

        reward in [-1, 1]: +1 = Bryan loved it, -1 = fell flat. Perceptron/SGD-style
        update with clamping so no feature runs away and the model stays interpretable.
        """
        r = max(_MIN_REWARD, min(_MAX_REWARD, float(reward)))
        if r == 0.0:
            return
        with self._lock:
            for k, f in features.items():
                if k not in self._weights:
                    self._weights[k] = FEATURE_PRIORS.get(k, 0.0)
                self._weights[k] = max(
                    -_WEIGHT_CLAMP, min(_WEIGHT_CLAMP, self._weights[k] + self._lr * r * float(f))
                )
            self._updates += 1
            self._persist()

    def weights(self) -> dict[str, float]:
        with self._lock:
            return dict(self._weights)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {"updates": self._updates, "weights": dict(self._weights)}

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            with self._lock:
                for k, v in (raw.get("weights", {}) or {}).items():
                    self._weights[k] = float(v)
                self._updates = int(raw.get("updates", 0) or 0)
        except self._ERRORS as exc:
            record_degradation("taste_model_load", exc)

    def _persist(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"schema_version": 1, "saved_at": time.time(), "updates": self._updates, "weights": self._weights}
            fd, tmp = tempfile.mkstemp(prefix=".taste_", suffix=".json", dir=str(self._path.parent))
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(payload, fh, ensure_ascii=False)
                os.replace(tmp, self._path)
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
        except self._ERRORS as exc:
            record_degradation("taste_model_persist", exc)


_singleton: TasteModel | None = None
_lock = threading.Lock()


def get_taste_model() -> TasteModel:
    global _singleton
    if _singleton is None:
        with _lock:
            if _singleton is None:
                _singleton = TasteModel()
    return _singleton


def reset_taste_model() -> None:
    global _singleton
    with _lock:
        _singleton = None
