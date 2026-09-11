"""Systems that look integrated and are not, to find out what the measures reward.

A statistic that only ever runs on the thing it was written for cannot be
trusted, because there is no way to tell whether it is detecting integration or
detecting complexity. Each null here keeps something superficial and destroys
one thing that matters, and every measure in the battery is run over all of
them.

Two kinds.

Surrogates are built from the real recording. Sliding each domain against the
others in time keeps every marginal, every autocorrelation and every
within-domain relationship, and deletes the alignment between domains; shuffling
rows keeps the marginals and deletes time altogether. Anything that survives
either was never about the coupling.

Architectures are small dynamical systems with the same ten domains, the same
widths and comparable coupling strength, wired differently on purpose. The star
routes every influence through one broker. The one-way system keeps every
forward path and no feedback. The prompt-only system lets each domain see a
single scalar summary of everything instead of the states themselves. Each can
be perturbed and graphed exactly like the real one, so the graph measures get a
null too, which matters most for the star: it is strongly connected, and only
vertex connectivity tells it apart from a mind.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.subject.recording import Recording
from core.subject.state import DOMAINS

__all__ = [
    "ARCHITECTURES",
    "NULL_HORIZON",
    "SELF_EFFECT_TARGET",
    "ToySystem",
    "architecture",
    "replay_surrogate",
    "shuffle_surrogate",
    "toy_doses",
    "toy_edges",
    "toy_recording",
]


# ── surrogates over the real recording ───────────────────────────────────


#: The largest share of a recording a replay offset may consume. The window
#: every domain is read through is the rest of it, so a quarter buys eight
#: distinct offsets on a run of eight conditions while keeping three quarters
#: of the rows.
REPLAY_MARGIN: float = 0.25


def replay_surrogate(recording: Recording, *, seed: int = 0, cycle: int = 0) -> Recording:
    """Every domain replayed from its own time origin. Marginals kept, coupling gone.

    Read through a window rather than rolled. `np.roll` is circular, so each
    domain acquired one discontinuity where its end met its beginning — and the
    intact model, seeing every domain at once, can locate that jump from the
    other domains' positions and predict it, which is real cross-domain
    information the cut models lack. The surrogate scored 0.12 against a real
    system's 0.03 for that reason alone: the artifact was the signal.

    The offsets are whole multiples of the condition cycle, so every domain
    still sees the same condition at the same row. Otherwise the surrogate also
    destroys the alignment between a domain and its own condition, which is
    structure the real system is entitled to and the null is not meant to take.
    """
    rng = np.random.default_rng(seed)
    span = recording.frames
    margin = max(1, int(span * REPLAY_MARGIN))
    keep = span - margin
    if keep < 8:
        return recording
    step = max(1, cycle or len({*recording.conditions}) or 1)
    offsets = max(1, margin // step)
    x = np.empty((keep, recording.width), dtype=recording.x.dtype)
    for key in DOMAINS:
        block = recording.slices[key]
        start = int(rng.integers(0, offsets)) * step
        x[:, block] = recording.x[start : start + keep, block]
    return Recording(
        x=x,
        conditions=recording.conditions[:keep],
        tags=recording.tags[:keep],
        times=recording.times[:keep],
        env=recording.env[:keep],
        env_names=recording.env_names,
        columns=recording.columns,
        slices=recording.slices,
        notes={**recording.notes, "null": "replay"},
    )


def shuffle_surrogate(recording: Recording, *, seed: int = 0) -> Recording:
    """Rows permuted. Every distribution kept, every trajectory destroyed."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(recording.frames)
    return Recording(
        x=recording.x[order],
        conditions=tuple(recording.conditions[i] for i in order),
        tags=recording.tags,
        times=recording.times,
        env=recording.env[order],
        env_names=recording.env_names,
        columns=recording.columns,
        slices=recording.slices,
        notes={**recording.notes, "null": "time_shuffle"},
    )


# ── architectures ────────────────────────────────────────────────────────


@dataclass
class ToySystem:
    """Ten domains and one wiring rule, small enough to perturb thousands of times."""

    name: str
    widths: dict[str, int]
    coupling: dict[str, dict[str, np.ndarray]]
    decay: dict[str, np.ndarray]
    noise: float
    hub_width: int = 0
    hub_in: dict[str, np.ndarray] = field(default_factory=dict)
    hub_out: dict[str, np.ndarray] = field(default_factory=dict)
    hub_decay: float = 0.5
    frozen: tuple[str, ...] = ()

    def start(self, rng: np.random.Generator) -> dict[str, np.ndarray]:
        state = {key: rng.normal(scale=0.5, size=width) for key, width in self.widths.items()}
        if self.hub_width:
            state["_hub"] = rng.normal(scale=0.5, size=self.hub_width)
        return state

    def step(
        self, state: dict[str, np.ndarray], rng: np.random.Generator
    ) -> dict[str, np.ndarray]:
        nxt: dict[str, np.ndarray] = {}
        hub = state.get("_hub")
        if self.hub_width:
            drive = np.zeros(self.hub_width)
            for key, matrix in self.hub_in.items():
                drive += matrix @ state[key]
            hub_next = self.hub_decay * (hub if hub is not None else 0.0) + np.tanh(drive)
            nxt["_hub"] = hub_next + self.noise * rng.normal(size=self.hub_width)
        for key, width in self.widths.items():
            if key in self.frozen:
                nxt[key] = state[key]
                continue
            drive = self.decay[key] * state[key]
            for other, matrix in self.coupling.get(key, {}).items():
                drive = drive + matrix @ state[other]
            if self.hub_width and key in self.hub_out:
                drive = drive + self.hub_out[key] @ (hub if hub is not None else np.zeros(self.hub_width))
            nxt[key] = np.tanh(drive) + self.noise * rng.normal(size=width)
        return nxt


def _matrix(rng: np.random.Generator, rows: int, cols: int, strength: float) -> np.ndarray:
    return rng.normal(scale=strength / max(1.0, np.sqrt(cols)), size=(rows, cols))


def architecture(
    name: str,
    *,
    widths: dict[str, int] | None = None,
    seed: int = 0,
    strength: float = 0.6,
    noise: float = 0.05,
) -> ToySystem:
    """One of the wirings, with matched widths, decay and noise."""
    rng = np.random.default_rng(seed)
    widths = widths or {key: 4 for key in DOMAINS}
    decay = {key: rng.uniform(0.3, 0.7, size=width) for key, width in widths.items()}
    coupling: dict[str, dict[str, np.ndarray]] = {key: {} for key in widths}
    hub_width = 0
    hub_in: dict[str, np.ndarray] = {}
    hub_out: dict[str, np.ndarray] = {}
    frozen: tuple[str, ...] = ()
    order = list(widths)

    if name == "recurrent":
        # The comparison case: sparse, reciprocal, no broker.
        for index, key in enumerate(order):
            for offset in (1, 2, -1):
                other = order[(index + offset) % len(order)]
                coupling[key][other] = _matrix(rng, widths[key], widths[other], strength)
    elif name in {"star", "hub"}:
        hub_width = 4
        for key in order:
            hub_in[key] = _matrix(rng, hub_width, widths[key], strength)
            hub_out[key] = _matrix(rng, widths[key], hub_width, strength)
    elif name == "one_way":
        for index, key in enumerate(order):
            if index:
                other = order[index - 1]
                coupling[key][other] = _matrix(rng, widths[key], widths[other], strength)
    elif name == "prompt_only":
        # Everything reaches everything, through one scalar.
        hub_width = 1
        for key in order:
            hub_in[key] = _matrix(rng, hub_width, widths[key], strength)
            hub_out[key] = _matrix(rng, widths[key], hub_width, strength)
    elif name == "frozen_slow":
        for index, key in enumerate(order):
            for offset in (1, 2, -1):
                other = order[(index + offset) % len(order)]
                coupling[key][other] = _matrix(rng, widths[key], widths[other], strength)
        frozen = ("S", "M", "W", "N")
    else:
        raise ValueError(f"no architecture called {name!r}")

    return ToySystem(
        name=name,
        widths=widths,
        coupling=coupling,
        decay=decay,
        noise=noise,
        hub_width=hub_width,
        hub_in=hub_in,
        hub_out=hub_out,
        # The hub of a "hub" null carries its own state across steps; the hub of
        # a "star" is a pure relay that keeps nothing. The difference is whether
        # the broker is itself part of the mind.
        hub_decay=0.6 if name == "hub" else 0.0,
        frozen=frozen,
    )


#: Every wiring the battery is compared against, plus the recurrent reference.
ARCHITECTURES: tuple[str, ...] = (
    "recurrent",
    "star",
    "hub",
    "one_way",
    "prompt_only",
    "frozen_slow",
)


def toy_recording(system: ToySystem, *, steps: int = 4000, seed: int = 0) -> Recording:
    """Run one architecture and shape the result like a real recording."""
    rng = np.random.default_rng(seed)
    state = system.start(rng)
    rows: list[np.ndarray] = []
    for _ in range(steps):
        state = system.step(state, rng)
        rows.append(np.concatenate([state[key] for key in DOMAINS]))
    x = np.vstack(rows)
    start = 0
    slices: dict[str, slice] = {}
    for key in DOMAINS:
        slices[key] = slice(start, start + system.widths[key])
        start += system.widths[key]
    columns = tuple(
        f"{key}.{index}" for key in DOMAINS for index in range(system.widths[key])
    )
    return Recording(
        x=x,
        conditions=tuple("toy" for _ in range(steps)),
        tags=tuple("" for _ in range(steps)),
        times=np.arange(steps, dtype=np.float64),
        env=np.zeros((steps, 1)),
        env_names=("clock",),
        columns=columns,
        slices=slices,
        notes={"null": system.name},
    )


#: How far a null's own domain has to move before the displacement counts as
#: one. A displacement is not a dose until the thing displaced has moved, and
#: the natural unit is that domain's own ordinary variation: one standard
#: deviation, measured on the same warm-up the effects are scaled against.
#:
#: A flat delta is not a matched intervention. At the 0.5 this used, the
#: reference recurrent architecture's workspace domain moved itself by 0.29
#: standard deviations and its self-state by 0.52 — so some domains were poked
#: twice as hard as others, the reference's own declared wiring did not come
#: back out of the measurement, and the positive control the battery needs in
#: order to be able to say yes to anything failed on the instrument rather
#: than on the system. Dose-matching per source removes that: what differs
#: between architectures is then what escapes, not how hard each was hit.
SELF_EFFECT_TARGET: float = 1.0

#: How many frames of propagation a null is given. The real run reads a
#: displaced arm for two turns, which is sixty-six frames; giving the nulls six
#: measured a different experiment and left the reference's longest paths
#: unrecoverable.
NULL_HORIZON: int = 66


def _peak_effects(
    system: ToySystem,
    deltas: dict[str, float],
    scale: dict[str, float],
    *,
    trials: int,
    horizon: int,
    seed: int,
) -> dict[tuple[str, str], float]:
    """Mean peak standardized displacement for every ordered pair, self included.

    The two arms share a noise stream, so the sham floor is exactly zero and
    the comparison is the cleanest possible version of the one run against the
    real system.
    """
    effects: dict[tuple[str, str], list[float]] = {}
    for trial in range(trials):
        base_rng = np.random.default_rng(seed + 100 + trial)
        state = system.start(base_rng)
        for _ in range(20):
            state = system.step(state, base_rng)
        for source in DOMAINS:
            sham_state = {k: v.copy() for k, v in state.items()}
            pert_state = {k: v.copy() for k, v in state.items()}
            pert_state[source] = pert_state[source] + deltas[source]
            sham_rng = np.random.default_rng(seed + 500 + trial)
            pert_rng = np.random.default_rng(seed + 500 + trial)
            peak = {key: 0.0 for key in DOMAINS}
            for _ in range(horizon):
                sham_state = system.step(sham_state, sham_rng)
                pert_state = system.step(pert_state, pert_rng)
                for key in DOMAINS:
                    gap = float(
                        np.sqrt(np.mean((pert_state[key] - sham_state[key]) ** 2))
                    ) / scale[key]
                    peak[key] = max(peak[key], gap)
            for target in DOMAINS:
                effects.setdefault((source, target), []).append(peak[target])
    return {pair: float(np.mean(values)) for pair, values in effects.items()}


def toy_doses(
    system: ToySystem,
    *,
    target: float = SELF_EFFECT_TARGET,
    horizon: int = NULL_HORIZON,
    seed: int = 0,
    rounds: int = 6,
) -> tuple[dict[str, float], dict[str, float]]:
    """The displacement each domain needs to move itself by `target`, and the scales.

    Found by repeated proportional correction rather than a search: the
    response is close enough to linear in the displacement that six rounds of
    four trials settle every domain, and the rounds are cheap.
    """
    warm = toy_recording(system, steps=800, seed=seed + 1)
    scale = {key: float(np.mean(warm.domain(key).std(axis=0))) or 1.0 for key in DOMAINS}
    deltas = {key: 0.5 for key in DOMAINS}
    for _ in range(max(1, rounds)):
        measured = _peak_effects(
            system, deltas, scale, trials=4, horizon=horizon, seed=seed
        )
        for key in DOMAINS:
            own = max(1e-6, measured[(key, key)])
            deltas[key] = float(np.clip(deltas[key] * (target / own), 0.05, 20.0))
    return deltas, scale


def toy_edges(
    system: ToySystem,
    *,
    trials: int = 40,
    horizon: int = NULL_HORIZON,
    seed: int = 0,
    effect_min: float = 0.3,
    target: float = SELF_EFFECT_TARGET,
) -> list[tuple[str, str]]:
    """Perturb each domain to a matched dose and keep the targets that moved.

    Same bar as the real run, and the same number of frames to reach it, so
    that "this architecture's graph is not strongly connected" is a statement
    about the architecture.
    """
    deltas, scale = toy_doses(system, target=target, horizon=horizon, seed=seed)
    measured = _peak_effects(
        system, deltas, scale, trials=trials, horizon=horizon, seed=seed
    )
    return [
        pair
        for pair, value in measured.items()
        if pair[0] != pair[1] and value >= effect_min
    ]
