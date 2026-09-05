"""core/interiority/receptors.py — no signal may shout forever.

A receiving element with a fixed gain, under sustained input, either
saturates and loses all resolution or dominates and starves everything
else. Biology's answer is to move the receptors: internalise them under
chronic agonist and insert more under chronic absence. Two mechanisms,
both implemented here because they do different work.

**Desensitisation** is fast and specific. Occupancy drives receptor
phosphorylation, phosphorylation recruits arrestin, arrestin uncouples
the receptor and drives internalisation, and internalised receptors
either recycle or are degraded. It is why a loud channel goes quiet
while it is loud.

**Homeostatic scaling** is slow and multiplicative (Turrigiano 2008).
It moves the whole channel's gain toward a target activity while
preserving relative differences, which is the property that matters: a
system that clips instead of scaling loses the ordering permanently,
and the ordering is the information.

Three consequences are the point of having this in the path rather than
in a diagram:

*Tolerance.* The same event stops producing the same state. This is what
lets an agent stay in a bad situation without permanent alarm, and it is
what the live runtime's despair-spiral check in ``damasio_v2`` exists to
paper over. With gain control in the path, the spiral cannot start.

*Rebound.* Remove a chronic signal from a down-regulated receptor and
the gain is now too low; remove it from an up-regulated one and it is
too high. The overshoot is not a bug to damp out. It is mechanically the
same thing as missing something you had adapted to, and
:meth:`Receptor.withdrawal` reports it as a positive quantity.

*Preserved discrimination.* Scaling is multiplicative, so after the gain
moves, two inputs that differed by a factor still differ by that factor.

Every faculty channel is routed through a receptor by
:class:`ReceptorBank`, so this is substrate and not a nineteenth
faculty that happens to sit beside the others. It is also item 18 on the
list, and :class:`~core.interiority.faculties.f18_receptor_adjustment`
reads this state rather than modelling it a second time.
"""

from __future__ import annotations

from core.runtime.lockdep import checked_lock
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Mapping

from core.interiority.params import Param, ParamKind, declare


def _p(name: str, value: float, basis: str, sensitivity: str, **kw) -> Param:
    return declare(
        f"interiority.receptor.{name}",
        value,
        basis=basis,
        sensitivity=sensitivity,
        owner="core/interiority/receptors.py",
        **kw,
    )


_HILL_N = _p(
    "hill_coefficient", 1.5,
    "Occupancy follows the Hill equation O = L^n / (Kd^n + L^n). n above 1 gives "
    "the mild positive cooperativity typical of GPCR occupancy curves and makes "
    "the channel less responsive to trace input than a hyperbolic binding curve "
    "would be, which is what stops noise from moving the gain.",
    "n = 1 makes the channel respond linearly to trace signal; large n makes it "
    "a switch and destroys the graded response the rest of the package needs.",
    unit="dimensionless", kind=ParamKind.CITED, lower=1.0, upper=4.0,
)
_KD = _p(
    "half_occupancy", 0.5,
    "Half-maximal occupancy at the middle of the normalised signal range, so a "
    "mid-strength activation sits on the steepest part of the curve where "
    "differences are best resolved.",
    "Shifts which part of the input range the channel discriminates best.",
    unit="signal", kind=ParamKind.DERIVED, lower=0.05, upper=0.95,
)
_K_PHOS = _p(
    "phosphorylation_rate", 0.35,
    "Rate at which occupancy drives the receptor into its uncoupled state, per "
    "second of continuous full occupancy. Set so a channel held at saturation "
    "loses about a third of its gain in the first second and most of it within "
    "ten, which is the order of magnitude reported for rapid GPCR "
    "desensitisation.",
    "Faster and a strong state cannot be sustained long enough to act on; slower "
    "and a loud channel holds the interior for minutes.",
    unit="1/s", kind=ParamKind.CALIBRATION, lower=0.01, upper=2.0,
    sweep_range=(0.1, 0.8),
)
_K_DEPHOS = _p(
    "dephosphorylation_rate", 0.06,
    "Recovery of the uncoupled receptor. Roughly six times slower than "
    "phosphorylation, which is what makes tolerance outlast the stimulus and "
    "gives withdrawal its duration.",
    "Set equal to the phosphorylation rate and there is no tolerance at all, "
    "because the channel recovers as fast as it adapts.",
    unit="1/s", kind=ParamKind.CALIBRATION, lower=0.001, upper=1.0,
    sweep_range=(0.02, 0.2),
)
_K_INTERNALISE = _p(
    "internalisation_rate", 0.04,
    "Removal of uncoupled receptors from the surface. An order of magnitude "
    "below phosphorylation, so the fast and slow adaptations are separable in "
    "the trace rather than one curve with two names.",
    "Controls the slow arm of tolerance that survives a break in the stimulus.",
    unit="1/s", kind=ParamKind.CALIBRATION, lower=0.001, upper=0.5,
    sweep_range=(0.01, 0.15),
)
_K_RECYCLE = _p(
    "recycling_rate", 0.02,
    "Return of internalised receptors to the surface. Half the internalisation "
    "rate, so a channel that has been chronically driven recovers over roughly "
    "twice the time it took to adapt.",
    "The recovery half of the slow arm; zero would make adaptation permanent.",
    unit="1/s", kind=ParamKind.CALIBRATION, lower=0.001, upper=0.5,
    sweep_range=(0.005, 0.1),
)
_SCALING_RATE = _p(
    "homeostatic_scaling_rate", 0.004,
    "Turrigiano's multiplicative scaling operates over hours, not seconds, and "
    "is deliberately two orders of magnitude slower than desensitisation so the "
    "two adaptations do not fight. Multiplicative because relative synaptic "
    "weights are preserved across scaling, which is the finding.",
    "Faster and the channel's long-run set point chases its short-run activity, "
    "which removes the baseline that deviation is measured against.",
    unit="1/s", kind=ParamKind.CITED, lower=0.0001, upper=0.1,
)
_TARGET_ACTIVITY = _p(
    "target_activity", 0.30,
    "The activity level scaling drives toward. Set to the live affect engine's "
    "own arousal baseline (BASELINE_AROUSAL = 0.3 in core/affect/__init__.py) so "
    "the two substrates share one resting point instead of pulling apart.",
    "Moves where a channel settles when nothing is happening, which is the "
    "reference every deviation is read against.",
    unit="activity", kind=ParamKind.DERIVED,
)
_MIN_GAIN = _p(
    "min_gain", 0.05,
    "A channel may be turned down hard and must not be turned off. Zero gain is "
    "unrecoverable, because with no signal reaching the channel nothing can "
    "drive it back up.",
    "At zero, a channel that saturates once is deaf forever.",
    unit="gain", kind=ParamKind.DERIVED, lower=0.001, upper=0.5,
)
_MAX_GAIN = _p(
    "max_gain", 3.0,
    "Up-regulation under chronic absence is bounded, or a silent channel "
    "eventually amplifies its own noise into a signal. Three times baseline is "
    "the supersensitivity range reported after chronic receptor blockade.",
    "Higher and a long-quiet channel manufactures a state out of nothing.",
    unit="gain", kind=ParamKind.CITED, lower=1.0, upper=10.0,
)


@dataclass
class Receptor:
    """One channel's gain, with fast and slow adaptation."""

    channel: str
    #: Fraction of receptors at the surface and coupled.
    surface: float = 1.0
    #: Fraction phosphorylated, and so uncoupled.
    phosphorylated: float = 0.0
    #: Fraction internalised.
    internalised: float = 0.0
    #: Multiplicative homeostatic gain, preserving relative differences.
    scale: float = 1.0
    #: Running activity estimate that scaling drives toward the target.
    activity: float = _TARGET_ACTIVITY.value
    last_step: float = field(default_factory=time.time)
    #: Peak gain reached, for the withdrawal calculation.
    _peak_scale: float = 1.0

    def occupancy(self, signal: float) -> float:
        """Hill occupancy for a normalised input."""
        s = max(0.0, signal)
        if s == 0.0:
            return 0.0
        numerator = s**_HILL_N.value
        return numerator / (_KD.value**_HILL_N.value + numerator)

    def gain(self) -> float:
        """Current transduction gain: coupled surface fraction times scale."""
        coupled = max(0.0, self.surface * (1.0 - self.phosphorylated))
        return max(_MIN_GAIN.value, min(_MAX_GAIN.value, coupled * self.scale))

    def transduce(self, signal: float, dt: float | None = None) -> float:
        """Pass a signal through the channel and advance its adaptation."""
        now = time.time()
        if dt is None:
            dt = max(0.0, min(60.0, now - self.last_step))
        self.last_step = now

        occupancy = self.occupancy(signal)
        out = occupancy * self.gain()

        # Fast arm: occupancy drives phosphorylation, which decays back.
        d_phos = (
            _K_PHOS.value * occupancy * (1.0 - self.phosphorylated)
            - _K_DEPHOS.value * self.phosphorylated
        )
        self.phosphorylated = max(0.0, min(1.0, self.phosphorylated + d_phos * dt))

        # Slow arm: uncoupled receptors internalise; internalised recycle.
        d_int = (
            _K_INTERNALISE.value * self.phosphorylated * self.surface
            - _K_RECYCLE.value * self.internalised
        )
        self.internalised = max(0.0, min(1.0, self.internalised + d_int * dt))
        self.surface = max(_MIN_GAIN.value, min(1.0, 1.0 - self.internalised))

        # Homeostatic scaling toward the target activity. Multiplicative,
        # so two inputs that differed by a factor still do afterwards.
        self.activity += (out - self.activity) * min(1.0, dt * 0.1)
        error = _TARGET_ACTIVITY.value - self.activity
        self.scale = max(
            _MIN_GAIN.value, min(_MAX_GAIN.value, self.scale * (1.0 + _SCALING_RATE.value * error * dt))
        )
        self._peak_scale = max(self._peak_scale, self.scale)
        return max(0.0, out)

    def withdrawal(self) -> float:
        """How far below its adapted set point this channel currently sits.

        Positive when the channel adapted to a signal that has since
        stopped: the gain is now wrong in the direction of deficit. This
        is the same quantity as missing something, and several faculties
        read it rather than modelling absence separately.
        """
        expected = self.activity
        if expected >= _TARGET_ACTIVITY.value:
            return 0.0
        deficit = (_TARGET_ACTIVITY.value - expected) / max(1e-6, _TARGET_ACTIVITY.value)
        adapted = max(0.0, self._peak_scale - 1.0) + self.internalised
        return max(0.0, min(1.0, deficit * (0.5 + 0.5 * min(1.0, adapted))))

    def tolerance(self) -> float:
        """How much of this channel's original gain has been adapted away."""
        return max(0.0, min(1.0, 1.0 - self.gain()))

    def to_dict(self) -> dict[str, float]:
        return {
            "channel": self.channel,
            "gain": self.gain(),
            "surface": self.surface,
            "phosphorylated": self.phosphorylated,
            "internalised": self.internalised,
            "scale": self.scale,
            "activity": self.activity,
            "tolerance": self.tolerance(),
            "withdrawal": self.withdrawal(),
        }


class ReceptorBank:
    """One receptor per faculty channel, in the path of every activation."""

    def __init__(self) -> None:
        self._lock = checked_lock("core.interiority.receptors.ReceptorBank", reentrant=True)
        self._receptors: dict[str, Receptor] = {}
        self._passes = 0

    def receptor(self, channel: str) -> Receptor:
        with self._lock:
            r = self._receptors.get(channel)
            if r is None:
                r = Receptor(channel=channel)
                self._receptors[channel] = r
            return r

    def transduce(self, channel: str, signal: float, dt: float | None = None) -> float:
        with self._lock:
            self._passes += 1
            return self.receptor(channel).transduce(signal, dt)

    def idle(self, channels: tuple[str, ...], dt: float | None = None) -> None:
        """Advance channels that received nothing this tick.

        Adaptation must run when the signal is absent or there is no
        recovery and no rebound, and a bank that only steps on input
        models a nervous system that stops when you stop talking to it.
        """
        with self._lock:
            for channel in channels:
                if channel in self._receptors:
                    self._receptors[channel].transduce(0.0, dt)

    def gains(self) -> dict[str, float]:
        with self._lock:
            return {k: v.gain() for k, v in self._receptors.items()}

    def withdrawal(self, channel: str) -> float:
        with self._lock:
            r = self._receptors.get(channel)
            return 0.0 if r is None else r.withdrawal()

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "passes": self._passes,
                "channels": {k: v.to_dict() for k, v in self._receptors.items()},
            }

    def reset_for_test(self) -> None:
        with self._lock:
            self._receptors.clear()
            self._passes = 0


_BANK: ReceptorBank | None = None
_BANK_LOCK = checked_lock("core.interiority.receptors.singleton")


def get_receptor_bank() -> ReceptorBank:
    global _BANK
    with _BANK_LOCK:
        if _BANK is None:
            _BANK = ReceptorBank()
        return _BANK


__all__ = ["Receptor", "ReceptorBank", "get_receptor_bank"]
