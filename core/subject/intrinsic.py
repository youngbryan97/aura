"""Whether the next internal state needs the last one, or only the world.

The definition being tested is the sharp one. Same environment, different
internal state, different future:

    E_t = E'_t, K_t != K'_t  =>  K_{t+tau} != K'_{t+tau}

and its predictive shadow, the gain from being allowed to see K_t at all:

    D_intr = (L[f(E_t)] - L[g(E_t, K_t)]) / L[f(E_t)]

That ratio is easy to fake, and easy to break. K_t is a hundred columns and
E_t is five, so the richer model wins on capacity alone unless the comparison
is careful. Four things make it honest here.

Both models are scored on rows neither was fitted on, split contiguously in
time.

Both are given the same representation: four principal components per domain,
fitted on the training rows. A hundred raw columns against a few hundred turns
is not a fair test of what the state knows, it is a test of how much room the
model had to memorise.

What is predicted is the change, not the level. Several of these columns drift
— the substrate wanders, the world model's surprise falls as it learns — and
predicting a drifting level from its own past is trivially good inside the
training range and catastrophic beyond it. Measured on the level, the state
made prediction five times worse than the environment alone on the world model
and two and a half times worse on recurrent cognition, which is what
extrapolating a drift off the end of the training data looks like.

And the same wide model is fitted again with K_t's rows shuffled against their
targets: it keeps every column and every degree of freedom and loses only the
alignment between past and future. Whatever survives that shuffle is
information; whatever does not was capacity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.subject.estimate import fit_predict, split_rows
from core.subject.irreducibility import COMPONENTS, LOWER_BOUND_Z, _basis, _block_columns, _folds
from core.subject.recording import Recording
from core.subject.state import DOMAINS

__all__ = ["IntrinsicReport", "PersistenceReport", "intrinsic_gain", "persistence"]


@dataclass(frozen=True, slots=True)
class IntrinsicReport:
    loss_environment: float
    loss_environment_and_state: float
    loss_shuffled_state: float
    gain: float
    gain_over_shuffle: float
    pairs: int
    env_width: int
    note: str = ""

    @property
    def passes(self) -> bool:
        return self.gain > 0.0 and self.gain_over_shuffle > 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "loss_environment_only": round(self.loss_environment, 6),
            "loss_environment_and_state": round(self.loss_environment_and_state, 6),
            "loss_shuffled_state": round(self.loss_shuffled_state, 6),
            "delta_intrinsic": round(self.gain, 6),
            "delta_over_shuffled_state": round(self.gain_over_shuffle, 6),
            "pairs": self.pairs,
            "env_width": self.env_width,
            "passes": self.passes,
            "note": self.note,
        }


def intrinsic_gain(recording: Recording, *, seed: int = 0) -> IntrinsicReport:
    """How much of the next state the previous state explains beyond the world."""
    frames = recording.frames
    if frames < 60:
        return IntrinsicReport(1.0, 1.0, 1.0, 0.0, 0.0, frames, 0, "too few frames")
    env = recording.env[:-1]
    if env.shape[1] == 0:
        return IntrinsicReport(1.0, 1.0, 1.0, 0.0, 0.0, frames - 1, 0, "no environment recorded")

    live = recording.live_domains()
    if not live:
        return IntrinsicReport(1.0, 1.0, 1.0, 0.0, 0.0, frames - 1, 0, "no domain moved")
    train, validate, test = split_rows(frames - 1)
    sources: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for key in DOMAINS:
        if key not in live:
            continue
        columns = _block_columns(recording, (key,))
        basis = _basis(recording.x[:-1][:, columns], train, COMPONENTS)
        before = basis(recording.x[:-1][:, columns])
        after = basis(recording.x[1:][:, columns])
        sources.append(before)
        targets.append(after - before)
    now = np.hstack(sources)
    nxt = np.hstack(targets)
    only_env = fit_predict(env, nxt, train=train, validate=validate, test=test)
    both = fit_predict(
        np.hstack([env, now]),
        nxt,
        train=train,
        validate=validate,
        test=test,
        own_width=env.shape[1],
    )
    rng = np.random.default_rng(seed)
    order = rng.permutation(now.shape[0])
    shuffled = fit_predict(
        np.hstack([env, now[order]]),
        nxt,
        train=train,
        validate=validate,
        test=test,
        own_width=env.shape[1],
    )

    base = only_env.loss if only_env.loss > 1e-12 else 1.0
    gain = (only_env.loss - both.loss) / base
    over_shuffle = (shuffled.loss - both.loss) / (shuffled.loss if shuffled.loss > 1e-12 else 1.0)
    return IntrinsicReport(
        loss_environment=only_env.loss,
        loss_environment_and_state=both.loss,
        loss_shuffled_state=shuffled.loss,
        gain=float(gain),
        gain_over_shuffle=float(over_shuffle),
        pairs=int(now.shape[0]),
        env_width=int(env.shape[1]),
    )


# ── ISC-v2's persistence line ────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class PersistenceReport:
    """Whether the state carries its own history into its next level. See `persistence`."""

    gain: float
    gain_lower_bound: float
    over_shuffle: float
    over_shuffle_lower_bound: float
    folds: tuple[float, ...]
    pairs: int
    domains: tuple[str, ...]
    note: str = ""

    @property
    def passes(self) -> bool:
        return self.gain_lower_bound > 0.0 and self.over_shuffle_lower_bound > 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "gain": round(self.gain, 6),
            "gain_lower_bound": round(self.gain_lower_bound, 6),
            "over_shuffle": round(self.over_shuffle, 6),
            "over_shuffle_lower_bound": round(self.over_shuffle_lower_bound, 6),
            "folds": [round(value, 6) for value in self.folds],
            "pairs": self.pairs,
            "domains": list(self.domains),
            "passes": self.passes,
            "note": self.note,
        }


def _bound(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, mean
    error = float(np.std(values, ddof=1)) / math.sqrt(len(values))
    return mean, mean - LOWER_BOUND_Z * error


def persistence(recording: Recording, *, seed: int = 0, exclude: tuple[str, ...] = ()) -> PersistenceReport:
    """How much of the state's next level its present level explains, beyond the world and the clock.

    `intrinsic_gain` predicts each domain's change, and a change is predictable
    from the present level whenever the level has no memory at all: for
    independent noise the next change is minus the present value plus noise,
    so memoryless noise scored a gain near one half and passed, while a random
    walk, which carries all of its history, scored below zero and failed. The
    measure rewarded reverting to the mean, which is the opposite of persistence.

    This predicts the next level instead. The drift that made the level hard to
    score is given to both models as elapsed time, so a trend explains nothing
    the state is credited with, and the comparison is read over forward-chaining
    folds with the bound irreducibility uses, so the state has to establish its
    advantage rather than show it on one split. The shuffled arm keeps every
    column and permutes the state's rows, as `intrinsic_gain` does.

    ``exclude`` leaves domains out of the state and the target, so the reading
    can be taken without a long-lived store.
    """
    frames = recording.frames
    env = recording.env[:-1] if recording.env.size else np.zeros((max(0, frames - 1), 0))
    live = tuple(key for key in recording.live_domains() if key not in set(exclude))
    if frames < 60 or not live:
        return PersistenceReport(0.0, 0.0, 0.0, 0.0, (), max(0, frames - 1), live, "too few frames or no live domain")
    pairs = frames - 1
    folds = _folds(pairs)
    first_train = folds[0][0]
    sources: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for key in live:
        columns = _block_columns(recording, (key,))
        basis = _basis(recording.x[:-1][:, columns], first_train, COMPONENTS)
        sources.append(basis(recording.x[:-1][:, columns]))
        targets.append(basis(recording.x[1:][:, columns]))
    now = np.hstack(sources)
    following = np.hstack(targets)
    clock = (np.arange(pairs, dtype=np.float64) / max(1, pairs - 1)).reshape(-1, 1)
    world = np.hstack([env, clock])
    rng = np.random.default_rng(seed)
    shuffled_now = now[rng.permutation(pairs)]
    gains: list[float] = []
    over: list[float] = []
    for train, validate, test in folds:
        base = fit_predict(world, following, train=train, validate=validate, test=test)
        state = fit_predict(np.hstack([world, now]), following, train=train, validate=validate, test=test, own_width=world.shape[1])
        shuffled = fit_predict(
            np.hstack([world, shuffled_now]), following, train=train, validate=validate, test=test, own_width=world.shape[1]
        )
        gains.append((base.loss - state.loss) / base.loss if base.loss > 1e-12 else 0.0)
        over.append((shuffled.loss - state.loss) / shuffled.loss if shuffled.loss > 1e-12 else 0.0)
    gain, gain_bound = _bound(gains)
    shuffle_gain, shuffle_bound = _bound(over)
    return PersistenceReport(
        gain=gain,
        gain_lower_bound=gain_bound,
        over_shuffle=shuffle_gain,
        over_shuffle_lower_bound=shuffle_bound,
        folds=tuple(gains),
        pairs=pairs,
        domains=live,
        note=f"next level from the present level, beyond environment and elapsed time, over {len(gains)} forward folds",
    )
