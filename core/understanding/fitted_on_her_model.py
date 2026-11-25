"""A reading fitted on the model she is running: kinds told apart by its states at the end of a transcript.

The states belong to a model, so a fit does not outlive it: it is made in her own process, on the model that is running,
from invented examples. Each example's states are read once, kept by the hash of its transcript, and only what is new is
read again. The probabilities are calibrated on predictions made out of fold, since a classifier fitted on a few hundred
examples in thousands of dimensions is surer than it is right.
"""
from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from core.runtime.errors import record_degradation
from core.runtime.lockdep import checked_lock

logger = logging.getLogger("Aura.Understanding")

__all__ = ["Fitted", "FittedOnHerModel", "fit"]

#: How strongly the classifier is held back. Chosen on the examples' own folds, 23 November 2026.
REGULARISED = 0.01
#: How many folds the calibration is taken over.
FOLDS = 5
#: How many examples one job of the worker reads.
A_BATCH = 8
#: How many of the states just read are held, for whatever else the same words are read for.
JUST_READ = 8
_JUST_READ: dict[tuple[str, str], Any] = {}
#: How long a batch waits for room on the lane, in all, and the longest pause between two tries at it.
PATIENCE_S, LONGEST_PAUSE_S = 1800.0, 60.0


@dataclass
class Fitted:
    """The classifiers, one a depth, and what their probabilities are worth."""

    classes: tuple[str, ...]
    #: For each depth: the mean and spread its states are put on, the weights, and the intercepts.
    parts: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
    #: What the log-odds are divided by so that a probability means what it says.
    temperature: float = 1.0
    #: How the fit did on examples it was not fitted on: the kind told, whether anything was asked, and how many.
    told: float = 0.0
    told_asking: float = 0.0
    over: int = 0

    def probabilities(self, state: np.ndarray) -> dict[str, float]:
        """The calibrated probability of each kind for one state of (depths, sketch)."""
        total = np.zeros(len(self.classes))
        for depth, (mean, spread, weights, bias) in enumerate(self.parts):
            logits = weights @ ((state[depth] - mean) / spread) + bias
            total += _softmax(logits)
        mean_p = np.clip(total / len(self.parts), 1e-9, 1.0)
        tempered = _softmax(np.log(mean_p) / self.temperature)
        return {name: float(p) for name, p in zip(self.classes, tempered, strict=True)}


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def _one_depth(X: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    from sklearn.linear_model import LogisticRegression

    mean, spread = X.mean(axis=0), X.std(axis=0)
    spread = np.where(spread > 1e-6, spread, 1.0)
    model = LogisticRegression(C=REGULARISED, max_iter=2000, class_weight="balanced")
    model.fit((X - mean) / spread, labels)
    weights, bias = np.asarray(model.coef_), np.asarray(model.intercept_)
    if len(model.classes_) == 2:  # two kinds are fitted as one boundary: the first kind is where it is not crossed
        weights, bias = np.vstack([np.zeros_like(weights), weights]), np.concatenate([np.zeros_like(bias), bias])
    return mean, spread, weights, bias, tuple(str(c) for c in model.classes_)


def _averaged(states: np.ndarray, labels: np.ndarray) -> Fitted:
    parts, classes = [], ()
    for depth in range(states.shape[1]):
        mean, spread, weights, bias, classes = _one_depth(states[:, depth].astype(np.float64), labels)
        parts.append((mean, spread, weights, bias))
    return Fitted(classes=classes, parts=parts)


def _temperature(log_p: np.ndarray, truth: np.ndarray) -> float:
    """The temperature under which the truth was likeliest, over predictions made out of fold (Guo and others 2017)."""
    best, best_nll = 1.0, math.inf
    for exponent in np.linspace(-1.5, 1.5, 61):
        t = float(math.exp(exponent))
        p = _softmax(log_p / t)
        nll = -float(np.log(np.clip(p[np.arange(len(truth)), truth], 1e-12, 1.0)).mean())
        if nll < best_nll:
            best, best_nll = t, nll
    return best


def fit(states: np.ndarray, labels: Sequence[str], *, seed: int = 0, kind_of_nothing: str = "talk") -> Fitted:
    """The classifiers over every example, and what they are worth on examples they were not fitted on.
    ``kind_of_nothing`` is the kind that is the absence of the rest, told apart from them all."""
    from sklearn.model_selection import StratifiedKFold

    said = np.asarray([str(label) for label in labels])
    whole = _averaged(states, said)
    index = {name: i for i, name in enumerate(whole.classes)}
    out = np.zeros((len(said), len(whole.classes)))
    smallest = min(int((said == name).sum()) for name in whole.classes)
    folds = min(FOLDS, smallest)
    if folds < 2:
        return whole
    for train, held in StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed).split(states, said):
        part = _averaged(states[train], said[train])
        for row in held:
            for name, p in part.probabilities(states[row]).items():
                out[row, index[name]] = p
    log_p = np.log(np.clip(out, 1e-9, 1.0))
    truth = np.asarray([index[name] for name in said])
    whole.temperature = _temperature(log_p, truth)
    picked = out.argmax(axis=1)
    whole.told = float((picked == truth).mean())
    nothing = index.get(kind_of_nothing, -1)
    whole.told_asking = float(((picked != nothing) == (truth != nothing)).mean())
    whole.over = len(said)
    return whole


@dataclass
class _Held:
    fitted: Fitted | None = None
    states: dict[str, np.ndarray] = field(default_factory=dict)
    model: str = ""
    learning: bool = False


class FittedOnHerModel:
    """Kinds told apart by her model's states, from examples of each: (kind, transcript)."""

    def __init__(self, name: str, examples: Callable[[], Sequence[tuple[str, str]]], *, kind_of_nothing: str,
                 root: Path | None = None) -> None:
        self.name, self._examples, self._nothing, self._root = name, examples, kind_of_nothing, root
        self._lock = checked_lock(f"understanding.{name}")
        self._held = _Held()

    # ── its examples and what it keeps ───────────────────────────────

    def examples(self) -> tuple[tuple[str, str], ...]:
        return tuple(self._examples())

    def _where(self, model: str) -> Path:
        from core.runtime.state_ownership import state_root

        root = self._root or (state_root() / "data" / "understanding")
        return root / f"{self.name}-{hashlib.sha256(model.encode()).hexdigest()[:16]}.npz"

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:24]

    def _load(self, model: str) -> dict[str, np.ndarray]:
        path = self._where(model)
        if not path.is_file():
            return {}
        try:
            with np.load(path, allow_pickle=False) as kept:
                keys, states = kept["keys"], kept["states"]
                return {str(key): states[i] for i, key in enumerate(keys)}
        except (OSError, ValueError, KeyError) as exc:
            record_degradation(self.name, exc, severity="info", action="the states kept for the examples are read again")
            return {}

    async def _save(self, model: str, states: dict[str, np.ndarray]) -> None:
        from core.governance_context import local_internal_governed_scope
        from core.runtime.file_write_gateway import get_file_write_gateway

        keys = sorted(states)
        buffer = io.BytesIO()
        np.savez(buffer, keys=np.asarray(keys), states=np.stack([states[k] for k in keys]).astype(np.float16))
        with local_internal_governed_scope("understanding.states", domain="file_write"):
            await get_file_write_gateway().write_bytes_async(self._where(model), buffer.getvalue(), source="understanding.states")

    @staticmethod
    def model_of(client: Any) -> str:
        """Which model a client runs: its path, resolved, so a link moved to another model is another model."""
        path = str(getattr(client, "model_path", "") or "")
        try:
            return str(Path(path).resolve()) if path else ""
        except OSError:
            return path

    def ready(self, client: Any) -> bool:
        with self._lock:
            return self._held.fitted is not None and self._held.model == self.model_of(client)

    def status(self) -> dict[str, Any]:
        with self._lock:
            held = self._held
            fitted = held.fitted
            return {"ready": fitted is not None, "learning": held.learning, "examples": len(held.states),
                    "of": len(self.examples()),
                    **({"told": round(fitted.told, 3), "told_asking": round(fitted.told_asking, 3),
                        "temperature": round(fitted.temperature, 3)} if fitted else {})}

    # ── fitting it, while nobody is waiting ──────────────────────────

    async def learn(self, client: Any, *, waiting: Any = None) -> bool:
        """Read what examples are not yet read and fit. ``waiting`` says whether a person is; while one is, nothing is read."""
        model = self.model_of(client)
        ask = getattr(client, "states_at_turn", None)
        if not model or not callable(ask):
            return False
        with self._lock:
            if self._held.learning or (self._held.fitted is not None and self._held.model == model):
                return self._held.fitted is not None
            self._held.learning = True
        try:
            states = await asyncio.to_thread(self._load, model)
            given = self.examples()
            wanted = [(self._key(text), text) for _kind, text in given]
            missing = [(key, text) for key, text in dict(wanted).items() if key not in states]
            logger.info("🧭 %s: %d examples held, %d to read.", self.name, len(wanted) - len(missing), len(missing))
            for at in range(0, len(missing), A_BATCH):
                batch = missing[at:at + A_BATCH]
                got = await self._read_when_it_can([text for _key, text in batch], ask, waiting)
                if not got:
                    break
                states.update({key: state for (key, _text), state in zip(batch, got, strict=True)})
            if any(key not in states for key, _text in wanted):
                if states:
                    await self._save(model, states)
                logger.info("🧭 %s: %d of %d examples are read; the rest wait for a quieter time.", self.name,
                            sum(1 for key, _ in wanted if key in states), len(wanted))
                return False
            if missing:
                await self._save(model, {key: states[key] for key, _text in wanted})
            fitted = await asyncio.to_thread(fit, np.stack([states[key] for key, _text in wanted]),
                                             [kind for kind, _text in given], kind_of_nothing=self._nothing)
            with self._lock:
                self._held.fitted, self._held.states, self._held.model = fitted, {k: states[k] for k, _ in wanted}, model
            logger.info("🧭 %s is read on %s: the kind told on %.1f%% of %d examples kept out, and whether it is any of "
                        "them on %.1f%%; its probabilities are tempered by %.2f.", self.name, Path(model).name,
                        100 * fitted.told, fitted.over, 100 * fitted.told_asking, fitted.temperature)
            return True
        except (RuntimeError, OSError, TypeError, ValueError, ImportError) as exc:
            record_degradation(self.name, exc, severity="warning", action=f"{self.name} is not read this time")
            return False
        finally:
            with self._lock:
                self._held.learning = False

    @staticmethod
    async def _read_when_it_can(texts: list[str], ask: Any, waiting: Any) -> list[np.ndarray]:
        """One batch, read when nobody is waiting; tried again, at longer pauses, where the lane has no room for it.

        It asks as a turn does. A batch takes a second or two of a model that is resident already, and as background work
        it was refused for as long as the host was short of memory, which it does not need any of."""
        waited, pause = 0.0, 1.0
        while waited < PATIENCE_S:
            if not (callable(waiting) and waiting()):
                got = list(await ask(texts, in_a_turn=True))
                if len(got) == len(texts):
                    return got
            await asyncio.sleep(pause)
            waited += pause
            pause = min(pause * 2.0, LONGEST_PAUSE_S)
        return []

    # ── reading one ───────────────────────────────────────────────────

    async def read_one(self, text: str, *, client: Any) -> dict[str, float] | None:
        """The probability of each kind for one transcript, or None where it is not read."""
        with self._lock:
            fitted, model = self._held.fitted, self._held.model
        if fitted is None or model != self.model_of(client):
            return None
        # One turn is read for several things, and her model's states at its end are the same for all of them.
        key = (model, self._key(text))
        state = _JUST_READ.get(key)
        if state is None:
            got = await client.states_at_turn([text], in_a_turn=True)
            if not got:
                return None
            state = _JUST_READ[key] = got[0]
            while len(_JUST_READ) > JUST_READ:
                _JUST_READ.pop(next(iter(_JUST_READ)), None)
        return fitted.probabilities(state)
