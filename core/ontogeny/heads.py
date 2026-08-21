"""L3c — the readouts: small, calibrated, and able to explain themselves.

A head is a multinomial logistic readout over [standardised features, presence
bits, reservoir state]. That is a deliberately modest model, and the modesty is
the engineering.

With roughly ten thousand genuinely independent graded episodes, capacity is
not the binding constraint — evidence is. A deep network fitted here would find
structure in noise, and its errors would be confident and unattributable. A
linear readout on a rich fixed reservoir is the standard, well-understood
answer for this regime: the reservoir supplies the nonlinearity and the memory,
the readout supplies the decision, and the readout's parameters remain few
enough that the held-out comparison in ``authority.py`` means something.

It also buys the property governance actually requires. A head that holds
authority over one of Aura's decisions must be able to say why it chose what it
chose, in terms of named features, on that specific episode. Here that is
arithmetic — the contribution of feature *i* to option *k* is exactly
``W[k, i] · x[i]`` — not a saliency heuristic. There is no interpretability
gap to apologise for, because there is nothing in the model that is not a
weighted sum of named quantities.

Confidence is temperature-scaled against held-out episodes. An uncalibrated
head is worse than no head: Aura is going to *say* how sure she is, and a
number that does not track reality is a way of lying with sincerity.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger("Aura.Ontogeny.Heads")

HEAD_SCHEMA = "aura.ontogeny.head.v1"

#: AdaGrad base rate for online updates. Conservative: a head that lurches on
#: one episode is a head that will lurch on one bad episode.
DEFAULT_LEARNING_RATE = 0.05

#: L2 penalty. With few samples and many reservoir inputs, the prior that most
#: weights are near zero is the correct one.
DEFAULT_L2 = 1e-3

#: Minimum evidence before a head will produce a decision at all. Below this
#: it still predicts — the predictions are recorded and scored — but it reports
#: itself as not ready, and nothing will let it act.
MIN_FIT_SAMPLES = 200


@dataclass
class Attribution:
    """Why this head chose this option on this episode."""

    option: str
    total_logit: float
    contributions: tuple[tuple[str, float], ...]

    def top(self, n: int = 5) -> list[tuple[str, float]]:
        ranked = sorted(self.contributions, key=lambda kv: abs(kv[1]), reverse=True)
        return [(name, round(value, 4)) for name, value in ranked[:n]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "option": self.option,
            "logit": round(self.total_logit, 4),
            "top_factors": self.top(),
        }


@dataclass
class Prediction:
    """A head's answer, with everything needed to score and audit it later."""

    control_point: str
    probabilities: dict[str, float]
    choice: str
    confidence: float
    version: int
    ready: bool
    attribution: Attribution | None = None
    at: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, Any]:
        return {
            "control_point": self.control_point,
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
            "choice": self.choice,
            "confidence": round(self.confidence, 4),
            "version": self.version,
            "ready": self.ready,
            "attribution": self.attribution.as_dict() if self.attribution else None,
        }


class PredictionHead:
    """One learned readout for one control point."""

    def __init__(
        self,
        control_point: str,
        options: Sequence[str],
        input_width: int,
        input_names: Sequence[str] = (),
        *,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        l2: float = DEFAULT_L2,
    ) -> None:
        if len(options) < 2:
            raise ValueError("a head needs at least two options to choose between")
        self.control_point = control_point
        self.options = tuple(options)
        self.input_width = int(input_width)
        self.input_names = tuple(input_names) or tuple(f"x{i}" for i in range(self.input_width))
        self.learning_rate = float(learning_rate)
        self.l2 = float(l2)

        k = len(self.options)
        self.w = np.zeros((k, self.input_width), dtype=np.float64)
        self.b = np.zeros(k, dtype=np.float64)
        self._grad_sq_w = np.zeros_like(self.w)
        self._grad_sq_b = np.zeros_like(self.b)
        self.temperature = 1.0
        self.version = 0
        self.samples_seen = 0
        self.fitted_at = 0.0
        self.fit_evidence: dict[str, Any] = {}

    # ── inference ────────────────────────────────────────────────────────

    @property
    def ready(self) -> bool:
        """Has this head seen enough graded experience to be allowed to act?"""
        return self.samples_seen >= MIN_FIT_SAMPLES and self.version > 0

    def logits(self, row: np.ndarray) -> np.ndarray:
        return self.w @ row + self.b

    def predict(self, row: np.ndarray, *, attribute: bool = False) -> Prediction:
        row = np.asarray(row, dtype=np.float64).reshape(-1)
        if row.shape[0] != self.input_width:
            raise ValueError(
                f"head {self.control_point} expects width {self.input_width}, got {row.shape[0]}"
            )
        probs = _softmax(self.logits(row) / max(self.temperature, 1e-3))
        index = int(np.argmax(probs))
        choice = self.options[index]
        attribution = None
        if attribute:
            contributions = tuple(
                (name, float(self.w[index, i] * row[i]))
                for i, name in enumerate(self.input_names)
            )
            attribution = Attribution(
                option=choice,
                total_logit=float(self.logits(row)[index]),
                contributions=contributions,
            )
        return Prediction(
            control_point=self.control_point,
            probabilities={opt: float(p) for opt, p in zip(self.options, probs, strict=True)},
            choice=choice,
            confidence=float(probs[index]),
            version=self.version,
            ready=self.ready,
            attribution=attribution,
        )

    # ── learning ─────────────────────────────────────────────────────────

    def observe(self, row: np.ndarray, label: str, *, weight: float = 1.0) -> float:
        """One online AdaGrad step. Returns the loss on this example."""
        if label not in self.options:
            return 0.0
        row = np.asarray(row, dtype=np.float64).reshape(-1)
        target = self.options.index(label)
        probs = _softmax(self.logits(row))
        error = probs.copy()
        error[target] -= 1.0
        error *= float(weight)

        grad_w = np.outer(error, row) + self.l2 * self.w
        grad_b = error
        self._grad_sq_w += grad_w * grad_w
        self._grad_sq_b += grad_b * grad_b
        self.w -= self.learning_rate * grad_w / (np.sqrt(self._grad_sq_w) + 1e-8)
        self.b -= self.learning_rate * grad_b / (np.sqrt(self._grad_sq_b) + 1e-8)
        self.samples_seen += 1
        return float(-math.log(max(probs[target], 1e-12)))

    def fit(
        self,
        rows: np.ndarray,
        labels: Sequence[str],
        *,
        weights: Sequence[float] | None = None,
        epochs: int = 12,
        seed: int = 7,
        should_stop: Callable[[], bool] | None = None,
        cooperate: Callable[[], None] | None = None,
    ) -> dict[str, Any]:
        """Batch-fit from the corpus. Returns the evidence, not just the model."""
        rows = np.asarray(rows, dtype=np.float64)
        if rows.ndim != 2 or rows.shape[0] == 0:
            return {"fitted": False, "reason": "no rows"}
        if rows.shape[1] != self.input_width:
            return {"fitted": False, "reason": f"width {rows.shape[1]} != {self.input_width}"}
        index = {opt: i for i, opt in enumerate(self.options)}
        y = np.array([index.get(label, -1) for label in labels], dtype=np.int64)
        keep = y >= 0
        rows, y = rows[keep], y[keep]
        if rows.shape[0] == 0:
            return {"fitted": False, "reason": "no labels within the option set"}
        w = np.asarray(weights, dtype=np.float64)[keep] if weights is not None else np.ones(len(y))

        k = len(self.options)
        self.w = np.zeros((k, self.input_width), dtype=np.float64)
        # Start from the class prior so an unfitted feature set predicts the
        # base rate rather than a uniform guess.
        counts = np.bincount(y, weights=w, minlength=k) + 1e-6
        self.b = np.log(counts / counts.sum())
        self._grad_sq_w = np.zeros_like(self.w)
        self._grad_sq_b = np.zeros_like(self.b)

        rng = np.random.default_rng(seed)
        order = np.arange(len(y))
        losses: list[float] = []
        for _ in range(max(1, epochs)):
            if cooperate is not None:
                cooperate()
            if should_stop is not None and should_stop():
                return {"fitted": False, "reason": "foreground_preempted"}
            rng.shuffle(order)
            epoch_loss = 0.0
            for offset, i in enumerate(order):
                if (
                    should_stop is not None
                    and offset % 64 == 0
                    and should_stop()
                ):
                    return {"fitted": False, "reason": "foreground_preempted"}
                if cooperate is not None and offset % 64 == 0:
                    cooperate()
                probs = _softmax(self.logits(rows[i]))
                error = probs.copy()
                error[y[i]] -= 1.0
                error *= w[i]
                grad_w = np.outer(error, rows[i]) + self.l2 * self.w
                grad_b = error
                self._grad_sq_w += grad_w * grad_w
                self._grad_sq_b += grad_b * grad_b
                self.w -= self.learning_rate * grad_w / (np.sqrt(self._grad_sq_w) + 1e-8)
                self.b -= self.learning_rate * grad_b / (np.sqrt(self._grad_sq_b) + 1e-8)
                epoch_loss += -math.log(max(probs[y[i]], 1e-12)) * w[i]
            losses.append(epoch_loss / max(w.sum(), 1e-9))

        self.samples_seen = int(rows.shape[0])
        self.version += 1
        self.fitted_at = time.time()
        predictions = np.argmax(rows @ self.w.T + self.b, axis=1)
        self.fit_evidence = {
            "fitted": True,
            "samples": int(rows.shape[0]),
            "epochs": epochs,
            "final_loss": round(losses[-1], 5) if losses else None,
            "train_accuracy": round(float(np.mean(predictions == y)), 4),
            "class_balance": {
                opt: int(np.sum(y == i)) for i, opt in enumerate(self.options)
            },
            "version": self.version,
        }
        return dict(self.fit_evidence)

    def calibrate(self, rows: np.ndarray, labels: Sequence[str]) -> float:
        """Fit the temperature on held-out episodes. Returns the temperature.

        Temperature scaling is one parameter, which is all the calibration a
        held-out set of this size can honestly support.
        """
        rows = np.asarray(rows, dtype=np.float64)
        index = {opt: i for i, opt in enumerate(self.options)}
        y = np.array([index.get(label, -1) for label in labels], dtype=np.int64)
        keep = y >= 0
        rows, y = rows[keep], y[keep]
        if rows.shape[0] < 30:
            self.temperature = 1.0
            return self.temperature
        raw = rows @ self.w.T + self.b
        best_t, best_nll = 1.0, math.inf
        for t in np.linspace(0.5, 4.0, 36):
            probs = np.apply_along_axis(_softmax, 1, raw / t)
            nll = float(-np.mean(np.log(np.maximum(probs[np.arange(len(y)), y], 1e-12))))
            if nll < best_nll:
                best_nll, best_t = nll, float(t)
        self.temperature = best_t
        return best_t

    # ── persistence ──────────────────────────────────────────────────────

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": HEAD_SCHEMA,
            "control_point": self.control_point,
            "options": list(self.options),
            "input_width": self.input_width,
            "input_names": list(self.input_names),
            "w": self.w.tolist(),
            "b": self.b.tolist(),
            "temperature": self.temperature,
            "version": self.version,
            "samples_seen": self.samples_seen,
            "fitted_at": self.fitted_at,
            "fit_evidence": dict(self.fit_evidence),
            "learning_rate": self.learning_rate,
            "l2": self.l2,
        }

    def load_state(self, state: Mapping[str, Any]) -> bool:
        """Restore weights. A shape or option mismatch is refused, never coerced."""
        try:
            if tuple(state.get("options", ())) != self.options:
                return False
            w = np.asarray(state["w"], dtype=np.float64)
            b = np.asarray(state["b"], dtype=np.float64)
            if w.shape != self.w.shape or b.shape != self.b.shape:
                return False
            self.w, self.b = w, b
            self.temperature = float(state.get("temperature", 1.0))
            self.version = int(state.get("version", 0))
            self.samples_seen = int(state.get("samples_seen", 0))
            self.fitted_at = float(state.get("fitted_at", 0.0))
            self.fit_evidence = dict(state.get("fit_evidence", {}))
            self._grad_sq_w = np.zeros_like(self.w)
            self._grad_sq_b = np.zeros_like(self.b)
            return True
        except (KeyError, TypeError, ValueError) as exc:
            logger.info("ontogeny: head %s checkpoint rejected (%s)", self.control_point, exc)
            return False

    def report(self) -> dict[str, Any]:
        return {
            "control_point": self.control_point,
            "options": list(self.options),
            "version": self.version,
            "ready": self.ready,
            "samples_seen": self.samples_seen,
            "temperature": round(self.temperature, 3),
            "weight_norm": round(float(np.linalg.norm(self.w)), 4),
            "fit_evidence": dict(self.fit_evidence),
            "fitted_age_s": round(time.time() - self.fitted_at, 1) if self.fitted_at else None,
        }


def _softmax(z: np.ndarray) -> np.ndarray:
    shifted = z - np.max(z)
    exp = np.exp(shifted)
    return exp / np.maximum(exp.sum(), 1e-12)


__all__ = [
    "Attribution",
    "DEFAULT_L2",
    "DEFAULT_LEARNING_RATE",
    "HEAD_SCHEMA",
    "MIN_FIT_SAMPLES",
    "Prediction",
    "PredictionHead",
]
