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

from dataclasses import dataclass
from typing import Any

import numpy as np

from core.subject.estimate import fit_predict, split_rows
from core.subject.irreducibility import COMPONENTS, _basis, _block_columns
from core.subject.recording import Recording
from core.subject.state import DOMAINS

__all__ = ["IntrinsicReport", "intrinsic_gain"]


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
