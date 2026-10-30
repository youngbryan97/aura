"""Systems built to pass every integration line but one, and an ordinary language-model agent.

Every architecture in the null library fails two lines or more (seed 97's table, read line by line), so the library
shows the lines are distinct and not that each is needed on its own. A near miss is the reference with one property
taken out by a stated rule and nothing else changed. If a line is doing its job, the battery fails the near miss on
that line. If it fails it on others too, the construction leaked into a second property, and that is reported rather
than tuned away: a near miss is built from the property it removes, never fitted to what the lines read.

The near misses are kept out of `ARCHITECTURES`, so no campaign's null table or comparison set moves.

The agent is the comparator the paper is asked about most: a stateless language model, a memory store it reads and
appends to, and a fixed persona, with none of the other domains doing anything. It is a toy of that architecture, as
every null is a toy of its own, and it says nothing about any particular product. Its model is called only when
input arrives, so with no input nothing in it moves: that is a fact of the architecture, stated here and not
discovered by running it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.subject.nulls import (
    LinearReference,
    ToySystem,
    _as_matrix,
    _domain_slices,
    _matrix,
    architecture,
)
from core.subject.state import DOMAINS

__all__ = ["AGENT", "NEAR_MISSES", "NAMES", "build", "is_built_here"]

#: Each near miss, and the line it is built to fail and no other. Strong connectivity, vertex connectivity and
#: re-entry are one family here: a graph that is not strongly connected cannot have vertex connectivity two, so
#: the one-way near miss is expected to fail the family together.
NEAR_MISSES: dict[str, str] = {
    "near_miss_no_synergy": "synergy",
    "near_miss_driven": "closure",
    "near_miss_one_cut_vertex": "vertex connectivity",
    "near_miss_one_way": "strong connectivity (and with it vertex connectivity and re-entry)",
    "near_miss_one_shock": "differentiation",
    "near_miss_weakly_coupled": "irreducibility",
    # Second constructions, added after the smoke run of 30 October showed the two above did not take their
    # property out (one shared shock left the largest component at 0.36 of the variance, and weakening every coupling
    # to 0.3 left irreducibility at 0.25). The first two stay in the experiment and are reported as they read.
    "near_miss_slow_mode": "differentiation",
    "near_miss_loose_domain": "irreducibility",
}

AGENT: str = "stateless_agent"

NAMES: tuple[str, ...] = (*NEAR_MISSES, AGENT)

#: How hard the outside driver pushes, as a share of the reference's own coupling strength, and how much of its
#: own past it keeps. Small, so the driver adds a cause outside K without becoming the system's main mode.
DRIVER_SHARE: float = 0.25
DRIVER_MEMORY: float = 0.95

#: The cut vertex and the two domains that reach the rest only through it. A is a source in two triples and I and
#: N in none, so no triple's edges cross the cut.
CUT_VERTEX: str = "A"
BEHIND_THE_CUT: tuple[str, ...] = ("I", "N")

#: The halves of the one-way near miss: every triple's edges run within a half or from the first to the second,
#: so removing the way back removes no triple's own coupling.
UPSTREAM: tuple[str, ...] = ("P", "M", "A", "S", "I")
DOWNSTREAM: tuple[str, ...] = ("W", "D", "C", "G", "N")

#: The share of its cross-domain coupling the weakly coupled near miss keeps.
WEAK_SHARE: float = 0.3

#: Where the slow mode's eigenvalue is put. The reference's wiring sits at a spectral radius of 0.7, so a mode at
#: 0.95 keeps about ten times the variance of the fastest of the rest, which is what a dominant mode is.
SLOW_MODE_RADIUS: float = 0.95

#: The domain the loose near miss leaves almost on its own, and the share of its coupling, in and out, it keeps. N is
#: in no synergy triple, so no triple loses a source.
LOOSE: str = "N"
LOOSE_SHARE: float = 0.05


@dataclass
class DrivenReference(LinearReference):
    """The reference with one variable outside its ten domains that remembers its own past and pushes all of them."""

    driver_out: dict[str, np.ndarray] = field(default_factory=dict)

    def step(self, state: dict[str, np.ndarray], rng: np.random.Generator) -> dict[str, np.ndarray]:
        nxt = super().step({key: value for key, value in state.items() if key != "_hub"}, rng)
        hub = state["_hub"]
        for key, matrix in self.driver_out.items():
            nxt[key] = nxt[key] + matrix @ hub
        nxt["_hub"] = self.hub_decay * hub + self.noise * rng.normal(size=self.hub_width)
        return nxt


@dataclass
class OneShockReference(LinearReference):
    """The reference with every domain's shocks drawn from one shared shock rather than its own."""

    def step(self, state: dict[str, np.ndarray], rng: np.random.Generator) -> dict[str, np.ndarray]:
        shared = rng.normal(size=1)

        class _Shared:
            @staticmethod
            def normal(size: int | tuple[int, ...] = 1) -> np.ndarray:
                width = size if isinstance(size, int) else int(np.prod(size))
                return np.resize(shared, width)

        return super().step(state, _Shared())  # type: ignore[arg-type]


@dataclass
class SlowModeReference(LinearReference):
    """The reference with one collective direction fed back on itself until it is the slowest mode by far.

    Every domain writes along its own leading direction a share of one sum read from all of them, so the whole system
    gains a single slow mode that carries most of its variance: a broadcast of one quantity laid over the wiring.
    """

    mode_read: dict[str, np.ndarray] = field(default_factory=dict)
    mode_write: dict[str, np.ndarray] = field(default_factory=dict)
    mode_gain: float = 0.0

    def step(self, state: dict[str, np.ndarray], rng: np.random.Generator) -> dict[str, np.ndarray]:
        nxt = super().step(state, rng)
        collective = sum(float(self.mode_read[key] @ state[key]) for key in DOMAINS)
        for key in DOMAINS:
            nxt[key] = nxt[key] + self.mode_gain * collective * self.mode_write[key]
        return nxt

    def linear_part(self) -> np.ndarray:
        where = _domain_slices(self.widths)
        matrix = _as_matrix(self, where)
        write = np.concatenate([self.mode_write[key] for key in DOMAINS])
        read = np.concatenate([self.mode_read[key] for key in DOMAINS])
        return matrix + self.mode_gain * np.outer(write, read)


def _slow_mode(base: LinearReference, name: str) -> SlowModeReference:
    system = SlowModeReference(**{f: getattr(base, f) for f in base.__dataclass_fields__})
    system.name = name
    system.mode_write = {key: base.directions[key].copy() for key in DOMAINS}
    system.mode_read = {key: base.directions[key] / np.sqrt(len(DOMAINS)) for key in DOMAINS}

    def radius(gain: float) -> float:
        system.mode_gain = gain
        return float(np.max(np.abs(np.linalg.eigvals(system.linear_part()))))

    low, high = 0.0, 1.0
    while radius(high) < SLOW_MODE_RADIUS and high < 1e3:
        high *= 2.0
    for _ in range(60):
        middle = 0.5 * (low + high)
        low, high = (middle, high) if radius(middle) < SLOW_MODE_RADIUS else (low, middle)
    system.mode_gain = 0.5 * (low + high)
    return system


@dataclass
class StatelessAgent(ToySystem):
    """A language model with no state of its own, a memory store, a fixed persona, and nothing else.

    P is what arrives, G the context assembled from the input, the store and the persona, D the reply, and M the
    store, which keeps what was written and appends the reply. S is the persona: read on every call, never changed.
    The other five domains have nothing in this architecture to be, so they hold still. Each stage reads what the
    one before it held on the last step, as every toy does, so a displaced input, context or reply reaches what
    comes after it. The context and the reply are rebuilt on every call and keep nothing. The model runs only when
    something arrives; `input_rate` is how often that is, and at zero nothing in the agent moves.
    """

    input_rate: float = 1.0
    store_keep: float = 0.98

    def step(self, state: dict[str, np.ndarray], rng: np.random.Generator) -> dict[str, np.ndarray]:
        nxt = {key: value.copy() for key, value in state.items()}
        if rng.random() >= self.input_rate:
            return nxt
        nxt["G"] = np.tanh(
            self.coupling["G"]["P"] @ state["P"]
            + self.coupling["G"]["M"] @ state["M"]
            + self.coupling["G"]["S"] @ state["S"]
        )
        nxt["D"] = np.tanh(self.coupling["D"]["G"] @ state["G"]) + self.noise * rng.normal(size=self.widths["D"])
        nxt["M"] = self.store_keep * state["M"] + self.coupling["M"]["D"] @ state["D"]
        nxt["P"] = rng.normal(size=self.widths["P"])
        return nxt


def is_built_here(name: str) -> bool:
    return name in NAMES


def _reference(seed: int, widths: dict[str, int] | None, strength: float, noise: float) -> LinearReference:
    return architecture("recurrent", widths=widths, seed=seed, strength=strength, noise=noise)  # type: ignore[return-value]


def build(
    name: str,
    *,
    widths: dict[str, int] | None = None,
    seed: int = 0,
    strength: float = 0.6,
    noise: float = 0.05,
) -> ToySystem:
    """One near miss, or the agent, from the same seed the reference would be built from."""
    if name == AGENT:
        return _agent(seed=seed, strength=strength, noise=noise)
    if name not in NEAR_MISSES:
        raise KeyError(name)
    base = _reference(seed, widths, strength, noise)
    rng = np.random.default_rng(seed + 7919)
    if name == "near_miss_no_synergy":
        base.gains = {triple: 0.0 for triple in base.gains}
        base.name = name
        return base
    if name == "near_miss_driven":
        driven = DrivenReference(**{f: getattr(base, f) for f in base.__dataclass_fields__})
        driven.name, driven.hub_width, driven.hub_decay = name, 4, DRIVER_MEMORY
        driven.driver_out = {
            key: _matrix(rng, driven.widths[key], 4, strength * DRIVER_SHARE) for key in DOMAINS
        }
        return driven
    if name == "near_miss_one_shock":
        shocked = OneShockReference(**{f: getattr(base, f) for f in base.__dataclass_fields__})
        shocked.name = name
        return shocked
    if name == "near_miss_slow_mode":
        return _slow_mode(base, name)
    coupling = base.coupling
    if name == "near_miss_one_cut_vertex":
        rest = [key for key in DOMAINS if key not in BEHIND_THE_CUT and key != CUT_VERTEX]
        for behind in BEHIND_THE_CUT:
            for other in rest:
                coupling[behind].pop(other, None)
                coupling[other].pop(behind, None)
            for other in (CUT_VERTEX, *BEHIND_THE_CUT):
                if other != behind:
                    coupling[behind].setdefault(other, _matrix(rng, base.widths[behind], base.widths[other], strength))
                    coupling[other].setdefault(behind, _matrix(rng, base.widths[other], base.widths[behind], strength))
    elif name == "near_miss_one_way":
        for up in UPSTREAM:
            for down in DOWNSTREAM:
                coupling[up].pop(down, None)
    elif name == "near_miss_weakly_coupled":
        for key in coupling:
            coupling[key] = {other: matrix * WEAK_SHARE for other, matrix in coupling[key].items()}
    elif name == "near_miss_loose_domain":
        for key in coupling:
            for other in list(coupling[key]):
                if LOOSE in (key, other) and key != other:
                    coupling[key][other] = coupling[key][other] * LOOSE_SHARE
    base.name = name
    return base


def _agent(*, seed: int, strength: float, noise: float) -> StatelessAgent:
    rng = np.random.default_rng(seed)
    widths = {key: 4 for key in DOMAINS}
    coupling: dict[str, dict[str, np.ndarray]] = {key: {} for key in DOMAINS}
    coupling["G"] = {other: _matrix(rng, 4, 4, strength) for other in ("P", "M", "S")}
    coupling["D"] = {"G": _matrix(rng, 4, 4, strength)}
    coupling["M"] = {"D": _matrix(rng, 4, 4, strength * 0.2)}
    return StatelessAgent(
        name=AGENT,
        widths=widths,
        coupling=coupling,
        decay={key: np.zeros(4) for key in DOMAINS},
        noise=noise,
        frozen=tuple(key for key in DOMAINS if key not in ("P", "G", "D", "M")),
    )
