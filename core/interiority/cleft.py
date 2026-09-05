"""core/interiority/cleft.py — where a signal changes kind.

A chemical synapse is not a wire. It is a gap of a few tens of
nanometres across which one cell's decision becomes another's input, and
four of its properties are computational rather than incidental:

**Release is quantal and probabilistic** (Katz 1969). Transmitter comes
in packets and an arriving spike releases zero, one or a few. Release
probability is a variable the synapse controls. Transmission is
therefore unreliable by design, and the unreliability is a resource: it
decorrelates, it samples, and it makes the rate rather than the event
the carrier.

**The gap is shared.** Transmitter spills to neighbours and reaches
receptors it was not aimed at. Volume transmission is how neuromodulators
act on a whole region instead of a wire.

**Clearance sets the time constant**, and therefore what counts as "now"
for the receiving element.

**The receiving side decides what it hears.** The same molecule is
excitatory at one receptor and inhibitory at another, so meaning is
postsynaptic.

Taken as an architectural rule rather than a biological one, this says:
subsystems should not be wired by direct calls with guaranteed delivery.
They should publish into a medium with a time constant, a release
probability, a clearance rate, and receiver-side interpretation. Three
things follow that this runtime has needed:

* one subsystem cannot synchronously block another, which is the shape
  of the fsync-under-lock stalls in this codebase's own history;
* a subsystem that stops publishing degrades the receiver gradually
  instead of removing a term from a sum;
* a third party can change a channel's gain without touching either
  end, which is what modulation is.

So every faculty publishes into a cleft and every consumer reads from
one. It is item 19 on the list and it is also the transport the other
forty-two use, which is the honest reading of what a synapse is.
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Mapping
from dataclasses import dataclass, field

from core.interiority.params import Param, ParamKind, declare
from core.interiority.receptors import ReceptorBank, get_receptor_bank
from core.runtime.lockdep import checked_lock


def _p(name: str, value: float, basis: str, sensitivity: str, **kw) -> Param:
    return declare(
        f"interiority.cleft.{name}",
        value,
        basis=basis,
        sensitivity=sensitivity,
        owner="core/interiority/cleft.py",
        **kw,
    )


_QUANTUM = _p(
    "quantal_size", 0.1,
    "One vesicle's worth of signal on the normalised scale, so a maximal "
    "activation is ten quanta. Ten is the granularity at which a graded state "
    "still reads as graded after probabilistic release; at one quantum the "
    "channel is a coin flip and at a hundred the stochasticity is invisible and "
    "buys nothing.",
    "Coarser and every activation becomes all-or-nothing; finer and release "
    "noise stops decorrelating anything.",
    unit="signal", kind=ParamKind.DERIVED, lower=0.01, upper=0.5,
)
_BASE_RELEASE_P = _p(
    "release_probability", 0.6,
    "Baseline probability that a given quantum is released. Central synapses "
    "span roughly 0.1 to 0.9 and the middle of that range is where release "
    "probability is most modulable in both directions, which is the property "
    "this uses.",
    "At 1.0 transmission is deterministic and the medium is a wire again; near "
    "0 a real state fails to reach any consumer.",
    unit="probability", kind=ParamKind.CITED, lower=0.05, upper=0.99,
)
_CLEARANCE = _p(
    "clearance_rate", 4.0,
    "Reuptake plus enzymatic degradation plus diffusion, as one first-order "
    "rate. 4/s gives a 250 ms time constant, which is the order of an affective "
    "signal's persistence rather than a spike's, and it is what sets how long "
    "'now' lasts for a consumer.",
    "Slow clearance makes states smear into each other; fast clearance means a "
    "consumer that polls between ticks sees nothing.",
    unit="1/s", kind=ParamKind.CALIBRATION, lower=0.1, upper=100.0,
    sweep_range=(1.0, 20.0),
)
_SPILLOVER = _p(
    "spillover_fraction", 0.12,
    "Fraction of released transmitter reaching neighbouring channels. Non-zero "
    "because volume transmission is real and is how one faculty's state colours "
    "a related one without a wire between them; small because a large value "
    "would make every channel the same channel.",
    "At zero, faculties are independent and nothing generalises; above about a "
    "quarter, the interior loses its distinctions.",
    unit="fraction", kind=ParamKind.CALIBRATION, lower=0.0, upper=0.5,
    sweep_range=(0.0, 0.25),
)
_FACILITATION = _p(
    "facilitation_gain", 0.25,
    "Residual calcium raises release probability for a short window after a "
    "release, so a repeated signal transmits more reliably than an isolated "
    "one. This is why saying a thing twice lands harder than saying it once as "
    "loudly, and it is a property of the medium rather than of the sender.",
    "Zero removes short-term facilitation, so repetition carries no extra "
    "weight and a persistent state is no more audible than a transient one.",
    unit="probability", kind=ParamKind.CALIBRATION, lower=0.0, upper=0.8,
    sweep_range=(0.0, 0.5),
)
_FACILITATION_TAU = _p(
    "facilitation_tau", 2.0,
    "Decay time of the facilitation window. Two seconds is the residual-calcium "
    "scale and, at the affective time constant used here, means facilitation "
    "spans a few consecutive ticks rather than a whole episode.",
    "Long windows turn facilitation into a second, slower integrator that "
    "competes with the receptor bank's scaling.",
    unit="s", kind=ParamKind.CALIBRATION, lower=0.05, upper=60.0,
    sweep_range=(0.5, 10.0),
)


@dataclass
class _Terminal:
    """One channel's presynaptic side."""

    channel: str
    concentration: float = 0.0
    last_release: float = 0.0
    facilitation: float = 0.0
    releases: int = 0
    failures: int = 0
    last_step: float = field(default_factory=time.time)


@dataclass(frozen=True)
class Transmission:
    """What actually crossed, and what it cost to say so."""

    channel: str
    intended: float
    released: float
    concentration: float
    postsynaptic: float
    quanta_attempted: int
    quanta_released: int
    spillover: Mapping[str, float]

    @property
    def fidelity(self) -> float:
        if self.intended <= 0.0:
            return 1.0
        return max(0.0, min(1.0, self.released / self.intended))

    def to_dict(self) -> dict[str, object]:
        return {
            "channel": self.channel,
            "intended": self.intended,
            "released": self.released,
            "concentration": self.concentration,
            "postsynaptic": self.postsynaptic,
            "quanta": [self.quanta_attempted, self.quanta_released],
            "fidelity": self.fidelity,
            "spillover": dict(self.spillover),
        }


class SynapticCleft:
    """The medium every faculty publishes into and every consumer reads from."""

    def __init__(
        self,
        *,
        bank: ReceptorBank | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self._lock = checked_lock("core.interiority.cleft.SynapticCleft", reentrant=True)
        self._terminals: dict[str, _Terminal] = {}
        self._bank = bank or get_receptor_bank()
        self._rng = rng or random.Random()
        #: Neighbourhoods for volume transmission. A channel spills only
        #: to channels declared adjacent to it, because spilling to
        #: everything is the same as having one channel.
        self._neighbours: dict[str, tuple[str, ...]] = {}
        self._modulators: dict[str, float] = {}

    def declare_neighbourhood(self, channel: str, neighbours: tuple[str, ...]) -> None:
        with self._lock:
            self._neighbours[channel] = tuple(n for n in neighbours if n != channel)

    def modulate(self, channel: str, gain: float) -> None:
        """Third-party gain change, without touching either endpoint.

        This is what a neuromodulator does and it is the reason the
        medium is worth having: regulation can act on a channel without
        the sender or the receiver knowing.
        """
        with self._lock:
            self._modulators[channel] = max(0.0, min(4.0, gain))

    def _terminal(self, channel: str) -> _Terminal:
        t = self._terminals.get(channel)
        if t is None:
            t = _Terminal(channel=channel)
            self._terminals[channel] = t
        return t

    def release(
        self, channel: str, signal: float, dt: float | None = None
    ) -> Transmission:
        """Publish a signal. Delivery is probabilistic and the caller is told."""
        with self._lock:
            now = time.time()
            terminal = self._terminal(channel)
            step = dt if dt is not None else max(0.0, min(60.0, now - terminal.last_step))
            terminal.last_step = now

            # Clear what is already in the gap before adding to it.
            terminal.concentration *= math.exp(-_CLEARANCE.value * step)
            terminal.facilitation *= math.exp(-step / _FACILITATION_TAU.value)

            intended = max(0.0, min(1.0, signal))
            quanta = int(math.ceil(intended / _QUANTUM.value)) if intended > 0.0 else 0
            p = min(0.99, _BASE_RELEASE_P.value + terminal.facilitation)
            released_quanta = sum(1 for _ in range(quanta) if self._rng.random() < p)
            released = released_quanta * _QUANTUM.value

            terminal.concentration = min(2.0, terminal.concentration + released)
            terminal.releases += released_quanta
            terminal.failures += quanta - released_quanta
            if released_quanta:
                terminal.facilitation = min(
                    _FACILITATION.value, terminal.facilitation + _FACILITATION.value * 0.5
                )
                terminal.last_release = now

            spill: dict[str, float] = {}
            if released > 0.0:
                for neighbour in self._neighbours.get(channel, ()):  # volume transmission
                    amount = released * _SPILLOVER.value
                    n_terminal = self._terminal(neighbour)
                    n_terminal.concentration = min(2.0, n_terminal.concentration + amount)
                    spill[neighbour] = amount

            modulator = self._modulators.get(channel, 1.0)
            post = self._bank.transduce(channel, terminal.concentration * modulator, step)

            return Transmission(
                channel=channel,
                intended=intended,
                released=released,
                concentration=terminal.concentration,
                postsynaptic=post,
                quanta_attempted=quanta,
                quanta_released=released_quanta,
                spillover=spill,
            )

    def read(self, channel: str, dt: float | None = None) -> float:
        """What a consumer hears now, with nothing newly released."""
        with self._lock:
            terminal = self._terminals.get(channel)
            if terminal is None:
                return 0.0
            now = time.time()
            step = dt if dt is not None else max(0.0, min(60.0, now - terminal.last_step))
            terminal.last_step = now
            terminal.concentration *= math.exp(-_CLEARANCE.value * step)
            modulator = self._modulators.get(channel, 1.0)
            return self._bank.transduce(channel, terminal.concentration * modulator, step)

    def reliability(self, channel: str) -> float:
        with self._lock:
            terminal = self._terminals.get(channel)
            if terminal is None:
                return 0.0
            total = terminal.releases + terminal.failures
            return terminal.releases / total if total else 0.0

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "channels": {
                    name: {
                        "concentration": t.concentration,
                        "releases": t.releases,
                        "failures": t.failures,
                        "reliability": (
                            t.releases / (t.releases + t.failures)
                            if (t.releases + t.failures)
                            else 0.0
                        ),
                        "facilitation": t.facilitation,
                    }
                    for name, t in self._terminals.items()
                },
                "modulators": dict(self._modulators),
                "neighbourhoods": {k: list(v) for k, v in self._neighbours.items()},
            }

    def reset_for_test(self) -> None:
        with self._lock:
            self._terminals.clear()
            self._modulators.clear()


_CLEFT: SynapticCleft | None = None
_CLEFT_LOCK = checked_lock("core.interiority.cleft.singleton")


def get_cleft() -> SynapticCleft:
    global _CLEFT
    with _CLEFT_LOCK:
        if _CLEFT is None:
            _CLEFT = SynapticCleft()
        return _CLEFT


__all__ = ["SynapticCleft", "Transmission", "get_cleft"]
