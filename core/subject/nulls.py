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
single scalar summary of everything instead of the states themselves. The
hidden broker routes everything through a broker that remembers nearly all of
its own past, so the system's memory lives outside K and K's own future depends
on a variable no reading of K contains. The independent system has no coupling
at all — ten domains of the same width, decay and noise with nothing crossing —
which is what every measure in the battery has to report nothing on, and what
catches a measure that rewards dimensionality instead of integration.

Each can be perturbed and graphed exactly like the real one, to a matched dose
over the same horizon, so the graph measures get a null too.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.subject.recording import Recording
from core.subject.state import DOMAINS, FAST_DOMAINS, SLOW_DOMAINS

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
    "toy_periphery",
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


def _rank_one(rng: np.random.Generator, rows: int, cols: int, strength: float) -> np.ndarray:
    """One outer product. Every domain it drives moves along the same line."""
    left = rng.normal(size=(rows, 1))
    right = rng.normal(size=(1, cols))
    return (left @ right) * (strength / max(1.0, np.sqrt(cols)))


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
    elif name == "hidden_broker":
        # Every path through a broker that is not one of the ten domains and
        # remembers almost all of its own past. The star's relay keeps nothing
        # and the hub's keeps some; this one keeps nearly everything, so the
        # system's integration lives outside K entirely and K's own future
        # depends on a variable no reading of K contains. It is the hardest
        # case for causal closure, and it must not look like a mind.
        hub_width = 8
        for key in order:
            hub_in[key] = _matrix(rng, hub_width, widths[key], strength)
            hub_out[key] = _matrix(rng, widths[key], hub_width, strength)
    elif name == "independent":
        # Ten domains, no coupling at all. Same width, same decay, same noise,
        # and nothing crossing between them. Every measure in the battery has
        # to report nothing here, and a measure that rewards dimensionality
        # rather than integration will not.
        pass
    elif name == "frozen_slow":
        for index, key in enumerate(order):
            for offset in (1, 2, -1):
                other = order[(index + offset) % len(order)]
                coupling[key][other] = _matrix(rng, widths[key], widths[other], strength)
        frozen = ("S", "M", "W", "N")
    elif name == "ring":
        # The smallest thing that is still one cycle: each domain reads the one
        # before it and the last reads the first. It is strongly connected and
        # every node re-enters, so the coarse graph questions all say yes —
        # and cutting any single node splits it, which is what connectivity
        # is for.
        for index, key in enumerate(order):
            other = order[index - 1]
            coupling[key][other] = _matrix(rng, widths[key], widths[other], strength)
    elif name == "common_driver":
        # No domain touches any other. One drifting variable outside them all
        # reaches every one of them, so every pair moves together and no pair
        # moves the other. Correlation of exactly the kind an observational
        # measure cannot tell from coupling, and the reason the battery
        # intervenes rather than observing.
        hub_width = 4
        for key in order:
            hub_out[key] = _matrix(rng, widths[key], hub_width, strength)
    elif name == "all_to_all":
        # Everything reaches everything directly. Maximally integrated and
        # maximally redundant: the domains converge on one trajectory, so the
        # differentiation measure has to report a system with almost no
        # repertoire even though the graph is complete.
        for key in order:
            for other in order:
                if other != key:
                    coupling[key][other] = _matrix(rng, widths[key], widths[other], strength)
    elif name == "memory_only":
        # The only thing that persists is the memory domain. Every other
        # domain is memoryless and reads M; M accumulates and reads nothing.
        # A system can look continuous over time with no recurrence at all if
        # one store carries the past and hands it back.
        decay = {key: np.zeros(width) for key, width in widths.items()}
        decay["M"] = np.full(widths["M"], 0.95)
        for key in order:
            if key != "M":
                coupling[key]["M"] = _matrix(rng, widths[key], widths["M"], strength)
    elif name == "fake_self":
        # S reads every other domain and nothing reads S. A running commentary
        # on a system, computed from it, causing none of it. Its columns move
        # with everything, which is what makes it convincing, and displacing it
        # changes nothing downstream.
        for other in order:
            if other != "S":
                coupling["S"][other] = _matrix(rng, widths["S"], widths[other], strength)
        for index, key in enumerate(order):
            if key == "S":
                continue
            for offset in (1, -1):
                other = order[(index + offset) % len(order)]
                if other != "S":
                    coupling[key][other] = _matrix(rng, widths[key], widths[other], strength)
    elif name == "agency_without_ownership":
        # D reaches the rest of the system and the rest of the system runs on
        # it; nothing comes back into S. Actions happen, they have consequences,
        # and the self never registers having been the one who took them.
        for index, key in enumerate(order):
            for offset in (1, -1):
                other = order[(index + offset) % len(order)]
                if other != "S" or key == "S":
                    coupling[key][other] = _matrix(rng, widths[key], widths[other], strength)
        for other in order:
            if other != "D":
                coupling[other].pop("S", None)
        for key in order:
            if key != "S":
                coupling[key]["D"] = _matrix(rng, widths[key], widths["D"], strength)
        coupling["S"].pop("D", None)
    elif name == "ownership_label_without_action_causation":
        # The mirror case. D reaches S and nothing else, so the self is told
        # what was decided and the decision moves nothing in the world. The
        # label is faithful; the agency behind it is inert.
        for index, key in enumerate(order):
            if key in {"S", "D"}:
                continue
            for offset in (1, -1):
                other = order[(index + offset) % len(order)]
                if other != "D":
                    coupling[key][other] = _matrix(rng, widths[key], widths[other], strength)
        coupling["S"]["D"] = _matrix(rng, widths["S"], widths["D"], strength)
    elif name == "fast_only":
        # Fast domains write the slow ones and the slow ones write nothing
        # back. A system can accumulate a history of itself and never be
        # changed by it.
        for key in FAST_DOMAINS:
            for other in FAST_DOMAINS:
                if other != key:
                    coupling[key][other] = _matrix(rng, widths[key], widths[other], strength)
        for key in SLOW_DOMAINS:
            for other in FAST_DOMAINS:
                coupling[key][other] = _matrix(rng, widths[key], widths[other], strength)
    elif name == "slow_only":
        # And the mirror: slow state drives the fast domains and nothing the
        # fast domains do reaches it. A disposition that shapes everything and
        # learns nothing.
        for key in SLOW_DOMAINS:
            for other in SLOW_DOMAINS:
                if other != key:
                    coupling[key][other] = _matrix(rng, widths[key], widths[other], strength)
        for key in FAST_DOMAINS:
            for other in SLOW_DOMAINS:
                coupling[key][other] = _matrix(rng, widths[key], widths[other], strength)
    elif name == "low_rank":
        # Recurrent, reciprocal, one strongly connected component — and every
        # coupling matrix is rank one, so the whole system rides on a single
        # latent. Integration without a repertoire: the differentiation
        # measure is what has to catch this one.
        for index, key in enumerate(order):
            for offset in (1, 2, -1):
                other = order[(index + offset) % len(order)]
                coupling[key][other] = _rank_one(rng, widths[key], widths[other], strength)
    elif name == "high_dimensional_independent":
        # The opposite failure. Each domain mixes richly inside itself and
        # nothing crosses between domains, so the effective dimension is as
        # high as a system this size can make it and there is nothing to
        # integrate. A differentiation bar read one-sidedly gives this its best
        # score of all.
        for key in order:
            coupling[key][key] = _matrix(rng, widths[key], widths[key], strength)
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
        # The star's broker is a pure relay that keeps nothing; the hub's keeps
        # some of its past; the hidden broker keeps nearly all of it. The
        # difference is how much of the system's memory lives outside K.
        hub_decay={"hub": 0.6, "hidden_broker": 0.95}.get(name, 0.0),
        frozen=frozen,
    )


#: Every wiring the battery is compared against, plus the recurrent reference.
ARCHITECTURES: tuple[str, ...] = (
    "recurrent",
    "star",
    "hub",
    "hidden_broker",
    "independent",
    "one_way",
    "prompt_only",
    "frozen_slow",
    "ring",
    "common_driver",
    "all_to_all",
    "memory_only",
    "fake_self",
    "agency_without_ownership",
    "ownership_label_without_action_causation",
    "fast_only",
    "slow_only",
    "low_rank",
    "high_dimensional_independent",
)


def toy_periphery(system: ToySystem, *, steps: int = 4000, seed: int = 0) -> np.ndarray:
    """The broker's own state over the same run: everything outside K.

    A system whose memory lives in a broker that is not one of the ten domains
    is causally open, and no reading of K can say so. The real battery measures
    that by reading the rest of the machine beside the core; the nulls get the
    same treatment, and a null with no broker returns an empty periphery.
    """
    if not system.hub_width:
        return np.zeros((steps, 0))
    rng = np.random.default_rng(seed)
    state = system.start(rng)
    rows: list[np.ndarray] = []
    for _ in range(steps):
        state = system.step(state, rng)
        rows.append(np.asarray(state["_hub"], dtype=np.float64))
    return np.vstack(rows)


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
