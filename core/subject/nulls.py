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


#: What the rank-one null runs at instead of the shared noise. Its private
#: noise has to be small enough that the shared latent is what the state is,
#: rather than one component of it among ten independent ones.
LOW_RANK_NOISE: float = 0.004
LOW_RANK_GAIN: float = 6.0

#: And what the broadcast null runs at. Every domain hears the same signal, so
#: its private noise is the only thing that could give it a repertoire.
BROADCAST_NOISE: float = 0.004
BROADCAST_GAIN: float = 6.0


def _rank_one(rng: np.random.Generator, rows: int, cols: int, strength: float) -> np.ndarray:
    """One outer product. Every domain it drives moves along the same line."""
    left = rng.normal(size=(rows, 1))
    right = rng.normal(size=(1, cols))
    return (left @ right) * (strength / max(1.0, np.sqrt(cols)))


#: What the reference's wiring is scaled to: the middle of the decay the
#: family draws for every architecture, so the reference remembers its own past
#: no longer than its siblings do.
REFERENCE_RADIUS: float = 0.5

#: How hard each triple's product drives its target. A product of two
#: unit-spread readings is bilinear, so this is what it adds to the Jacobian
#: along either source; the product is bounded four spreads out, so the linear
#: part keeps the system stationary whatever this is, and what sets it is the
#: two lines it has to satisfy at once. At 0.25 — half the room the wiring
#: leaves at a radius of a half — so much of a target's change is the product
#: that its own way back round the wiring leaves the two sources reading as
#: one, and the fraction the line reads was predicted at 0.50 and measured
#: 0.04 on eleven triples over twelve seeds. At this, twelve seeds of twelve
#: pass all four triples: the fraction reads a fifth to a half against a floor
#: of a tenth, and the interaction two to ten times its own bound. With the
#: product taken out, none of the twelve passes a single triple.
REFERENCE_DOSE: float = 0.0625

#: How many rounds the reference's two sources are brought level over. Each
#: round is a Lyapunov solve and moves a path a quarter of the way, and the
#: loop converges because the system is a contraction.
REFERENCE_ROUNDS: int = 12

#: How many shocks a reference domain's columns are driven by. Four columns
#: taking four independent shocks make a system whose variance is spread over
#: its whole width — an effective dimension of nine tenths of it, where the
#: differentiation line wants under four tenths, and it wants that because a
#: system whose variance is spread over everything is what ten independent
#: noise sources look like. A domain's columns are views of fewer quantities
#: than there are columns, here as in her: this shares nothing between
#: domains, which is what the common-driver null is for.
REFERENCE_CHANNELS: int = 1

#: Which neighbours a reference domain reads, by their distance around the
#: declared order. Irreducibility asks what the cheapest cut costs, and the
#: cheapest cut is one domain against the other nine: what it costs is how much
#: of that domain's change the others account for. Three neighbours left the
#: reference under the bar it exists to clear. Six is still sparse — every
#: domain reads six of nine and there is no broker anywhere — and it is the
#: same six for every domain, so no domain is a hub.
REFERENCE_NEIGHBOURS: tuple[int, ...] = (1, 2, 3, -1, -2, -3)


@dataclass
class LinearReference(ToySystem):
    """The reference: the recurrent wiring without the squash, and one product per triple.

    The battery needs a system it must say yes to. The recurrent wiring alone
    is not that system: it has no designed interaction, so whether the synergy
    lines pass on it is a matter of the seed, and on the tanh family nothing
    about it can be dosed — cutting one path moves the whole system, and the
    same rule read a path's share as 0.87 on one seed and 0.00 on the next.

    Without the squash every quantity the lines read can be solved instead of
    searched. The stationary covariance is one Lyapunov solve; the information
    each source carries about the target's change is a determinant of it; and
    the two things the lines ask for are then constructions rather than hopes:

      * the fraction is a ratio, so what it needs is that the two sources
        carry the SAME information about the target's change. Two level
        sources read near a half whatever the size; a tenfold imbalance
        reads 0.03.
      * the interaction gain is a held-out ridge score against its own fold
        noise, so what it needs is a bilinear product large enough to clear
        that noise and small enough to leave the system a contraction.

    So each named triple has both sources as parents, level in information and
    together carrying as much of the target as everything else into it; one
    bilinear product of the two on the target's leading direction, bounded far
    out in the tail so a run of large readings cannot drive the state away; and
    no path from the target back to either source, which is what stopped the
    two of them from reading as one.
    """

    directions: dict[str, np.ndarray] = field(default_factory=dict)
    spreads: dict[str, float] = field(default_factory=dict)
    gains: dict[tuple[str, str, str], float] = field(default_factory=dict)
    clip: float = 4.0
    #: Each domain's own shocks, as a width-by-channels matrix of unit columns.
    channels: dict[str, np.ndarray] = field(default_factory=dict)

    def reading(self, state: dict[str, np.ndarray], key: str) -> float:
        """Where a domain sits along its own leading direction, in its own spreads."""
        return float(self.directions[key] @ state[key]) / self.spreads.get(key, 1.0)

    def step(
        self, state: dict[str, np.ndarray], rng: np.random.Generator
    ) -> dict[str, np.ndarray]:
        nxt: dict[str, np.ndarray] = {}
        for key, width in self.widths.items():
            drive = self.decay[key] * state[key]
            for other, matrix in self.coupling.get(key, {}).items():
                drive = drive + matrix @ state[other]
            shocks = self.channels.get(key)
            if shocks is None:
                nxt[key] = drive + self.noise * rng.normal(size=width)
            else:
                nxt[key] = drive + self.noise * (shocks @ rng.normal(size=shocks.shape[1]))
        for sources_and_target, gain in self.gains.items():
            if not gain:
                continue
            source_a, source_b, target = sources_and_target
            product = self.reading(state, source_a) * self.reading(state, source_b)
            bounded = float(np.clip(product, -self.clip, self.clip))
            nxt[target] = nxt[target] + gain * bounded * self.directions[target]
        return nxt


def _domain_slices(widths: dict[str, int]) -> dict[str, slice]:
    out: dict[str, slice] = {}
    start = 0
    for key in DOMAINS:
        out[key] = slice(start, start + widths[key])
        start += widths[key]
    return out


def _as_matrix(system: ToySystem, where: dict[str, slice]) -> np.ndarray:
    size = sum(system.widths[key] for key in DOMAINS)
    matrix = np.zeros((size, size))
    for key in DOMAINS:
        matrix[where[key], where[key]] = np.diag(system.decay[key])
        for other, block in system.coupling.get(key, {}).items():
            matrix[where[key], where[other]] = block
    return matrix


def _hold_radius(system: ToySystem, where: dict[str, slice], radius: float) -> None:
    """Hold the whole wiring at one spectral radius, decay and couplings together."""
    current = float(np.max(np.abs(np.linalg.eigvals(_as_matrix(system, where)))))
    if current <= 1e-12:
        return
    factor = radius / current
    for key in DOMAINS:
        system.decay[key] = system.decay[key] * factor
        for other in system.coupling.get(key, {}):
            system.coupling[key][other] = system.coupling[key][other] * factor


def _stationary_covariance(system: ToySystem, where: dict[str, slice]) -> tuple[np.ndarray, np.ndarray]:
    from scipy.linalg import solve_discrete_lyapunov

    matrix = _as_matrix(system, where)
    size = matrix.shape[0]
    innovation = np.zeros((size, size))
    for key in DOMAINS:
        shocks = getattr(system, "channels", {}).get(key)
        block = np.eye(system.widths[key]) if shocks is None else shocks @ shocks.T
        innovation[where[key], where[key]] = (system.noise ** 2) * block
    return matrix, solve_discrete_lyapunov(matrix, innovation)


def _leading_direction(block: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(block)
    return vectors[:, int(np.argmax(values))]


def _top_directions(block: np.ndarray, keep: int = 3) -> np.ndarray:
    values, vectors = np.linalg.eigh(block)
    order = np.argsort(values)[::-1][:keep]
    return vectors[:, order]


def _gaussian_information(joint: np.ndarray, split: int) -> float:
    sign_x, log_x = np.linalg.slogdet(joint[:split, :split])
    sign_y, log_y = np.linalg.slogdet(joint[split:, split:])
    sign_j, log_j = np.linalg.slogdet(joint)
    if min(sign_x, sign_y, sign_j) <= 0:
        return 0.0
    return max(0.0, 0.5 * float(log_x + log_y - log_j))


def predicted_synergy(system: ToySystem, *, of: str = "change") -> dict[tuple[str, str, str], dict[str, float]]:
    """What the synergy lines will read on a linear system, solved rather than run.

    The instrument keeps three components a side and reads a Gaussian
    information; on a linear system both are functions of one stationary
    covariance, so this says what a recording will show before it is recorded.
    It leaves out the products, which is why the reference is measured as well
    as solved: a product reaches its own sources the long way round, and what
    that does to the two of them is not in this.
    """
    from core.subject.synergy import TRIPLES

    where = _domain_slices(system.widths)
    matrix, covariance = _stationary_covariance(system, where)
    size = covariance.shape[0]
    out: dict[tuple[str, str, str], dict[str, float]] = {}
    for triple in TRIPLES:
        source_a, source_b, target = triple
        select = np.zeros((system.widths[target], size))
        select[:, where[target]] = np.eye(system.widths[target])
        # ISC-v1 reads the target's next level and v2 and v3 read its change.
        # Both are judged, so the reference is built to be seen by both.
        change = matrix[where[target]] - select if of == "change" else matrix[where[target]]
        cov_change = change @ covariance @ change.T + (system.noise ** 2) * np.eye(system.widths[target])
        basis_y = _top_directions(cov_change)
        bases, crosses, blocks = {}, {}, {}
        for key in (source_a, source_b):
            basis = _top_directions(covariance[where[key], where[key]])
            bases[key] = basis
            crosses[key] = basis_y.T @ change @ covariance[:, where[key]] @ basis
            blocks[key] = basis.T @ covariance[where[key], where[key]] @ basis
        across = bases[source_a].T @ covariance[where[source_a], where[source_b]] @ bases[source_b]
        sources = np.block([[blocks[source_a], across], [across.T, blocks[source_b]]])
        target_block = basis_y.T @ cov_change @ basis_y
        both = np.hstack([crosses[source_a], crosses[source_b]])
        joint = np.block([[sources, both.T], [both, target_block]])
        singles = {
            key: _gaussian_information(
                np.block([[blocks[key], crosses[key].T], [crosses[key], target_block]]),
                blocks[key].shape[0],
            )
            for key in (source_a, source_b)
        }
        together = _gaussian_information(joint, sources.shape[0])
        value = together - max(singles.values())
        out[triple] = {
            "mi_a": singles[source_a],
            "mi_b": singles[source_b],
            "joint": together,
            "synergy": value,
            "fraction": value / together if together > 1e-12 else 0.0,
        }
    return out


def _reference(
    widths: dict[str, int], rng: np.random.Generator, strength: float, noise: float, seed: int
) -> LinearReference:
    """The reference wiring, solved into the shape the lines are built to read."""
    from core.subject.synergy import TRIPLES

    decay = {key: rng.uniform(0.3, 0.7, size=width) for key, width in widths.items()}
    coupling: dict[str, dict[str, np.ndarray]] = {key: {} for key in widths}
    order = list(widths)
    for index, key in enumerate(order):
        for offset in REFERENCE_NEIGHBOURS:
            other = order[(index + offset) % len(order)]
            coupling[key][other] = _matrix(rng, widths[key], widths[other], strength)
    for source_a, source_b, target in TRIPLES:
        for source in (source_a, source_b):
            if source not in coupling[target]:
                coupling[target][source] = _matrix(rng, widths[target], widths[source], strength)
            # And no path from a target back to its own sources: it carries
            # their product, and a way back puts that product into both of them.
            coupling[source].pop(target, None)
    channels = {}
    for key, width in widths.items():
        drawn = rng.normal(size=(width, min(REFERENCE_CHANNELS, width)))
        channels[key] = drawn / np.linalg.norm(drawn, axis=0, keepdims=True)
    system = LinearReference(
        name="recurrent", widths=dict(widths), coupling=coupling, decay=decay, noise=noise,
        gains={triple: 0.0 for triple in TRIPLES}, channels=channels,
    )
    where = _domain_slices(widths)
    _hold_radius(system, where, REFERENCE_RADIUS)
    for _ in range(REFERENCE_ROUNDS):
        readings = [predicted_synergy(system, of="change"), predicted_synergy(system, of="level")]
        for source_a, source_b, target in TRIPLES:
            ratios = []
            for lines in readings:
                mi_a, mi_b = lines[(source_a, source_b, target)]["mi_a"], lines[(source_a, source_b, target)]["mi_b"]
                if min(mi_a, mi_b) > 1e-9:
                    ratios.append(mi_b / mi_a)
            if not ratios:
                continue
            tilt = float(np.clip(float(np.exp(np.mean(np.log(ratios)))) ** 0.25, 0.5, 2.0))
            coupling[target][source_a] = coupling[target][source_a] * tilt
            coupling[target][source_b] = coupling[target][source_b] / tilt
        _, covariance = _stationary_covariance(system, where)
        for source_a, source_b, target in TRIPLES:
            carried = {
                other: float(np.trace(block @ covariance[where[other], where[other]] @ block.T))
                for other, block in coupling[target].items()
            }
            own = np.diag(system.decay[target])
            carried["itself"] = float(np.trace(own @ covariance[where[target], where[target]] @ own.T))
            elsewhere = sum(value for key, value in carried.items() if key not in (source_a, source_b))
            for source in (source_a, source_b):
                if carried[source] > 1e-15 and elsewhere > 0.0:
                    lift = float(np.clip((elsewhere / carried[source]) ** 0.25, 0.5, 2.0))
                    coupling[target][source] = coupling[target][source] * lift
        _hold_radius(system, where, REFERENCE_RADIUS)
    _, covariance = _stationary_covariance(system, where)
    for key in DOMAINS:
        block = covariance[where[key], where[key]]
        system.directions[key] = _leading_direction(block)
        system.spreads[key] = float(
            np.sqrt(system.directions[key] @ block @ system.directions[key])
        )
    for triple in TRIPLES:
        system.gains[triple] = REFERENCE_DOSE
    return system


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
        # The comparison case: sparse, reciprocal, no broker — and the one the
        # battery has to be able to say yes to, so it is built to carry what
        # every line measures rather than left to the seed.
        return _reference(widths, rng, strength, noise, seed)
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
        # A broadcast, which means everyone receives the same thing. The first
        # version gave every pair its own random matrix, and that is a dense
        # random recurrent network rather than a broadcast — it measured an
        # effective dimension of 3.4 and passed differentiation, because a
        # dense random network genuinely has a repertoire. What the name says,
        # and what the control is for, is that every domain hears one signal
        # and adds nothing of its own: maximally integrated and maximally
        # redundant, so the graph is complete and the repertoire is almost
        # nothing.
        noise = BROADCAST_NOISE
        read = {key: rng.normal(size=(1, widths[key])) for key in order}
        write = {key: rng.normal(size=(widths[key], 1)) for key in order}
        for key in order:
            for other in order:
                if other != key:
                    coupling[key][other] = (
                        write[key] @ read[other]
                    ) * (strength * BROADCAST_GAIN / max(1.0, np.sqrt(widths[other])))
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
        # latent. Integration without a repertoire: the differentiation measure
        # is what has to catch this one.
        #
        # Rank-one coupling is not enough on its own. The first version of this
        # architecture kept the shared noise every other null runs at, and each
        # domain's own independent noise then dominated what the shared latent
        # was doing: it measured an effective dimension of 19.6, which is the
        # opposite of what the name claims. A null that fails for a reason
        # other than the one it was built for is not a control. So the coupling
        # is strong and the private noise is small, and the system collapses
        # onto the latent the way the name says.
        noise = LOW_RANK_NOISE
        decay = {key: rng.uniform(0.1, 0.2, size=width) for key, width in widths.items()}
        # One read direction per domain and one write direction per domain,
        # shared across every edge, so the whole network is rank one globally
        # rather than rank one edge by edge. A fresh pair of vectors per edge
        # leaves ten independent one-dimensional channels, which is a system
        # with a repertoire; sharing them leaves one.
        read = {key: rng.normal(size=(1, widths[key])) for key in order}
        write = {key: rng.normal(size=(widths[key], 1)) for key in order}
        for index, key in enumerate(order):
            for offset in (1, 2, -1):
                other = order[(index + offset) % len(order)]
                coupling[key][other] = (write[key] @ read[other]) * (
                    strength * LOW_RANK_GAIN / max(1.0, np.sqrt(widths[other]))
                )
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


def toy_closure(system: ToySystem, *, steps: int = 4000, seed: int = 0) -> tuple[bool, float]:
    """Whether K's future needs anything outside K, for a toy whose whole state can be listed.

    The organism's periphery is found by walking the machine, and a walk that
    found nothing that varied has not shown the core is closed; `closure_gain`
    says so. A toy is different: its state is a dict, and every key in it is
    either one of the ten domains or something outside them. When there is no
    key outside them, K is the whole state, and that is read off the state
    rather than assumed. A toy with state outside K is measured the way the
    organism is.

    Returns (closed, leak).
    """
    outside_keys = set(system.start(np.random.default_rng(seed))) - set(DOMAINS)
    if not outside_keys:
        return True, 0.0
    from core.subject.closure import closure_gain

    outside = toy_periphery(system, steps=steps, seed=seed)
    report = closure_gain(
        toy_recording(system, steps=steps, seed=seed),
        outside,
        tuple(f"broker.{index}" for index in range(outside.shape[1])),
        seed=seed,
    )
    return bool(report.closed), float(report.leak)


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


#: Where a dose search starts, how many trials each round reads, and the
#: bounds that stop a domain no writer can move from running its dose away.
DOSE_START: float = 0.5
DOSE_TRIALS: int = 4
DOSE_BOUNDS: tuple[float, float] = (0.05, 20.0)


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
    deltas = {key: DOSE_START for key in DOMAINS}
    for _ in range(max(1, rounds)):
        measured = _peak_effects(
            system, deltas, scale, trials=DOSE_TRIALS, horizon=horizon, seed=seed
        )
        for key in DOMAINS:
            own = max(1e-6, measured[(key, key)])
            deltas[key] = float(np.clip(deltas[key] * (target / own), *DOSE_BOUNDS))
    return deltas, scale


#: How many ordinary conditions the real battery runs, which is how many
#: independent start regimes a null gets. A test holds it equal to the
#: driver's list, so the replication bar asks the same count of both.
NULL_CONDITIONS: int = 8

#: Where in the arm the displacement lands, cycled over trials exactly as the
#: real run cycles it.
NULL_INJECTION_POINTS: tuple[int, ...] = (0, 8, 16)


def _arm(
    system: ToySystem,
    state: dict[str, np.ndarray],
    *,
    horizon: int,
    noise: int,
    source: str | None = None,
    dose: float = 0.0,
    where: int = 0,
) -> list[_ToyFrame]:
    rng = np.random.default_rng(noise)
    current = {key: value.copy() for key, value in state.items()}
    frames: list[_ToyFrame] = []
    for step in range(horizon):
        if source is not None and step == where:
            # A new dict. The last frame holds the old one, and writing the dose
            # into it moved that frame too: a displacement at step eight showed
            # at lag seven, before it happened, twice as large as it was.
            current = {**current, source: current[source] + dose}
        current = system.step(current, rng)
        frames.append(_ToyFrame(current))
    return frames


def _doses_by_trial_divergence(
    system: ToySystem,
    scale: dict[str, np.ndarray],
    *,
    target: float,
    horizon: int,
    seed: int,
) -> dict[str, float]:
    """Each domain's dose, found with the measure its trials are scored with.

    `toy_doses` corrects against a mean RMS gap over a domain's columns, and a
    trial is scored by the largest per-column gap. Doses matched under the
    first landed between two and ten standard deviations under the second on
    the broadcast null, so the matching did not match what was measured. The
    rounds, the trials per round and the bounds on a dose are `toy_doses`'s
    own.
    """
    import inspect

    from core.subject.causal import _paired_divergence

    defaults = inspect.signature(toy_doses).parameters
    rounds = defaults["rounds"].default
    doses = {key: DOSE_START for key in DOMAINS}
    for _ in range(rounds):
        moved: dict[str, list[float]] = {key: [] for key in DOMAINS}
        for trial in range(DOSE_TRIALS):
            life = np.random.default_rng(seed + 100 + trial)
            state = system.start(life)
            for _ in range(20):
                state = system.step(state, life)
            noise = seed + 500 + trial
            sham = _arm(system, state, horizon=horizon, noise=noise)
            for source in DOMAINS:
                displaced = _arm(
                    system, state, horizon=horizon, noise=noise, source=source, dose=doses[source]
                )
                effect, _, _, _ = _paired_divergence(displaced, sham, sham, scale)
                moved[source].append(effect.get(source, 0.0))
        for source in DOMAINS:
            reached = max(1e-6, float(np.median(moved[source])))
            doses[source] = float(np.clip(doses[source] * (target / reached), *DOSE_BOUNDS))
    return doses


class _ToyFrame:
    """One step of a toy system, readable the way a real frame is read."""

    __slots__ = ("state",)

    def __init__(self, state: dict[str, np.ndarray]) -> None:
        self.state = state

    def domain(self, key: str) -> np.ndarray:
        return self.state[key]


def toy_interventions(
    system: ToySystem,
    *,
    trials: int,
    conditions: int = NULL_CONDITIONS,
    horizon: int = NULL_HORIZON,
    seed: int = 0,
    target: float = SELF_EFFECT_TARGET,
) -> Any:
    """Every paired trial the real run makes, made on a toy system.

    The same shape as `core.subject.causal.run_interventions`: in each of
    `conditions` start regimes the system lives on between trials, every domain
    is displaced at a cycled point against two sham arms from the same state
    and the same noise, and effect and floor are read off one column by
    `_paired_divergence`. A toy has no workloads, so a condition here is an
    independent start with its own warm-up and noise. That is a weaker kind of
    variation than a different workload, and the count is the same.
    """
    from core.subject.causal import InterventionSet, Trial, _paired_divergence

    warm = toy_recording(system, steps=800, seed=seed + 1)
    scale = {key: np.asarray(warm.domain(key).std(axis=0), dtype=np.float64) for key in DOMAINS}
    doses = _doses_by_trial_divergence(system, scale, target=target, horizon=horizon, seed=seed)
    out = InterventionSet(scale=scale, delta=float(np.mean(list(doses.values()))))
    for regime in range(conditions):
        name = f"regime_{regime}"
        life = np.random.default_rng(seed + 1000 + regime)
        state = system.start(life)
        for _ in range(20):
            state = system.step(state, life)
        for index in range(trials):
            state = system.step(state, life)
            where = NULL_INJECTION_POINTS[index % len(NULL_INJECTION_POINTS)]
            noise = seed + 5000 + 1000 * regime + index
            for source in DOMAINS:
                arms = {
                    "pert": _arm(
                        system, state, horizon=horizon, noise=noise,
                        source=source, dose=doses[source], where=where,
                    ),
                    "sham_a": _arm(system, state, horizon=horizon, noise=noise),
                    "sham_b": _arm(system, state, horizon=horizon, noise=noise),
                }
                effect, floor, trace, floor_trace = _paired_divergence(
                    arms["pert"], arms["sham_a"], arms["sham_b"], scale
                )
                out.trials.append(
                    Trial(
                        source=source,
                        condition=name,
                        index=index,
                        effect=effect,
                        floor=floor,
                        trace=trace,
                        floor_trace=floor_trace,
                        took=True,
                        self_effect=effect.get(source, 0.0) - floor.get(source, 0.0),
                        injected_at=where,
                    )
                )
        out.lags = max(out.lags, horizon)
    return out


def toy_edges(
    system: ToySystem,
    *,
    trials: int = 6,
    conditions: int = NULL_CONDITIONS,
    horizon: int = NULL_HORIZON,
    seed: int = 0,
    target: float = SELF_EFFECT_TARGET,
) -> list[tuple[str, str]]:
    """The edges a null's trials support, by the real run's rule.

    The graph of a null used to be every pair whose mean peak displacement
    reached 0.3: no sham floor, no test, no correction across the ninety pairs
    and no replication, where the real graph needs all four. A null judged by a
    looser rule than the organism is not a comparison. Its trials now go
    through `build_edges` with the preregistered q-value, effect and
    replication bars.
    """
    from core.subject.causal import build_edges

    results = toy_interventions(
        system, trials=trials, conditions=conditions, horizon=horizon, seed=seed, target=target
    )
    edges, _ = build_edges(results, seed=seed)
    return [(edge.source, edge.target) for edge in edges]
