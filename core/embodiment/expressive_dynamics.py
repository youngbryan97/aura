"""core/embodiment/expressive_dynamics.py — doing that has no finish line.

Almost every action an agent takes is a trajectory toward a fixed point. There
is a goal state, a distance to it, and the action is over when the distance is
zero. Planners are built on this, schedulers price it, and progress means the
distance fell.

Singing to yourself while you work is not that shape. Neither is drumming
fingers, pacing while thinking, rocking, or dancing four bars of something in
a kitchen. There is no state these are trying to reach. They are closed
orbits: the motion returns to where it started and goes round again, and the
only way to describe their progress is how long they went on and how big they
were.

This matters as engineering rather than as description, because a system that
scores actions by remaining distance to a goal rates every one of these as
making no progress, forever. It will start one and immediately kill it, every
time, and the log will show a correctly functioning scheduler. That failure is
the reason this module exists: the accounting for orbits has to be separate
from the accounting for trajectories, or orbits cannot survive contact with a
scheduler at all.

## What makes it an orbit rather than a wobble

Van der Pol's equation is the smallest honest model of it:

    x'' - mu * (1 - x^2) * x' + x = 0

The damping term is negative for small ``x`` and positive for large ``x``, so
the system feeds itself when quiet and bleeds when loud. Every starting
condition except dead silence ends on the same closed curve. That is the
property being claimed — not that the motion repeats, but that it *restores
itself*, which is why humming does not need to be re-decided every second and
why interrupting it does not destroy it.

``is_limit_cycle`` checks this the only way worth checking it: start the same
oscillator from very different places and see whether the orbits converge. A
model that merely repeats will fail that check, and should.

## Entrainment is a separate claim

Driven by an outside rhythm, the oscillator locks to it over a band of driving
frequencies, and the band widens with the strength of the drive. That band is
the Arnold tongue, and ``entrainment_band`` measures it by sweeping rather
than by asserting it.

The reason to build this half is that synchrony pays separately from
movement. Tarr, Launay and Dunbar had people dance in synchrony and out of
it, at high exertion and low, and found synchrony and exertion raising pain
threshold and in-group closeness as independent effects — the togetherness of
it is doing something the exercise is not. So a bout carries two numbers, and
they are never summed into one.

Nothing here is about music. Entrainment between a local rhythm and an
external one is the same computation for a conversation's turn-taking, a
polling loop finding a server's cadence, or a practice schedule falling in
with someone else's week.
"""

from __future__ import annotations

import cmath
import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("Aura.Embodiment.Expressive")

#: Nonlinearity of the van der Pol damping. At this value the orbit is close
#: to sinusoidal, which is what a quiet self-sustaining motion looks like;
#: raising it produces the relaxation-oscillator shape of a hard repeated
#: gesture. Exposed on every constructor because it is the one thing that
#: changes the character of the motion.
DEFAULT_MU = 0.6

#: Integration step as a fraction of the natural period. Small enough that
#: the fourth-order integrator below is accurate over the hundreds of cycles
#: the band sweep runs, and named rather than inlined so that a caller
#: measuring something faster can say so.
STEPS_PER_CYCLE = 200


@dataclass
class Orbit:
    """A stretch of self-sustaining motion, and what it came to.

    ``exertion`` and ``synchrony`` stay apart. They are separately caused and
    separately consequential, and a single "quality of the dance" number
    cannot be told apart from either of them afterwards.
    """

    duration_s: float
    cycles: float
    mean_amplitude: float
    exertion: float
    """Time-averaged squared velocity: the work rate the motion took.

    Deliberately not normalised by frequency. The same gesture done twice as
    fast is twice as fast, and the finding this term exists to keep separable
    is about physical effort rather than about the shape of the movement.
    """

    synchrony: float
    """Phase-locking value against the driver, in [0, 1]. Zero when undriven."""

    drift_cycles: float
    """Whole turns the phase difference slipped over the bout.

    Phase-locking value alone cannot tell locking from slow drift: two rhythms
    a hair apart in frequency hold a nearly constant difference across any
    window short against the beat between them, and score near 1. Measuring
    the slip is what separates the two, and without it an oscillator with no
    driver at all reports a lock.
    """

    locked: bool
    phase_offset: float
    """Mean phase lead of the oscillator over its driver, in radians."""

    def as_dict(self) -> dict[str, Any]:
        return {
            "duration_s": round(self.duration_s, 3),
            "cycles": round(self.cycles, 3),
            "amplitude": round(self.mean_amplitude, 4),
            "exertion": round(self.exertion, 4),
            "synchrony": round(self.synchrony, 4),
            "drift_cycles": round(self.drift_cycles, 4),
            "locked": self.locked,
            "phase_offset": round(self.phase_offset, 4),
        }


class ExpressiveOscillator:
    """A van der Pol oscillator standing for one expressive action.

    The state is two numbers and there is no goal field, deliberately. Asking
    this object how close it is to done is a question with no answer, and
    callers that need one are asking about a different kind of action.
    """

    def __init__(
        self,
        *,
        period_s: float = 1.0,
        mu: float = DEFAULT_MU,
        x: float = 0.1,
        v: float = 0.0,
    ) -> None:
        if period_s <= 0:
            raise ValueError("period_s must be positive")
        self.period_s = float(period_s)
        self.mu = float(mu)
        self.x = float(x)
        self.v = float(v)
        self.t = 0.0

    @property
    def omega(self) -> float:
        return (2.0 * math.pi) / self.period_s

    def _derivative(self, x: float, v: float, drive: float) -> tuple[float, float]:
        w = self.omega
        # Written in the natural time of the oscillator, so that period_s is
        # the period rather than a scale factor somewhere else. The drive is
        # carried in at the same scale as the restoring force, which makes an
        # amplitude of 1 a push comparable with the oscillator's own pull
        # rather than a number whose meaning changes with the tempo. Adding a
        # bare acceleration instead makes every fast rhythm unforceable, and
        # the band sweep then reports no entrainment at any drive strength.
        return v, self.mu * w * (1.0 - x * x) * v - (w * w) * x + (w * w) * drive

    def step(self, dt: float, drive: float = 0.0) -> None:
        """One fourth-order Runge-Kutta step.

        Euler loses energy on an orbit, which for a system whose defining
        property is that it keeps its own energy would be the integrator
        contradicting the model.
        """
        x, v = self.x, self.v
        k1x, k1v = self._derivative(x, v, drive)
        k2x, k2v = self._derivative(x + 0.5 * dt * k1x, v + 0.5 * dt * k1v, drive)
        k3x, k3v = self._derivative(x + 0.5 * dt * k2x, v + 0.5 * dt * k2v, drive)
        k4x, k4v = self._derivative(x + dt * k3x, v + dt * k3v, drive)
        self.x = x + (dt / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
        self.v = v + (dt / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
        self.t += dt

    def phase(self) -> float:
        """Analytic phase of the current state, in radians, advancing with time.

        The sign matters and is easy to get backwards. With ``x = cos(wt)`` the
        velocity is ``-w sin(wt)``, so ``atan2(v / w, x)`` runs *backwards* and
        every phase comparison against a forward-running driver then drifts at
        twice the frequency and reports no locking however tightly locked the
        pair is.
        """
        return math.atan2(-self.v / self.omega, self.x) % (2.0 * math.pi)

    def run(
        self,
        duration_s: float,
        *,
        driver: Callable[[float], float] | None = None,
        driver_phase: Callable[[float], float] | None = None,
    ) -> Orbit:
        """Let it go for a while and report what the stretch was.

        ``driver`` supplies the forcing at each instant; ``driver_phase``
        supplies the driver's own phase, which is what the locking measure
        needs and what a forcing amplitude alone cannot give.
        """
        dt = self.period_s / STEPS_PER_CYCLE
        steps = max(1, int(round(duration_s / dt)))
        amplitudes: list[float] = []
        work: list[float] = []
        offsets: list[complex] = []
        unwrapped = 0.0
        previous = self.phase()
        difference = 0.0
        previous_difference: float | None = None
        for _ in range(steps):
            force = driver(self.t) if driver is not None else 0.0
            self.step(dt, force)
            amplitudes.append(math.hypot(self.x, self.v / self.omega))
            work.append(self.v * self.v)
            here = self.phase()
            # Signed wrapped difference. Taking the unsigned remainder turns
            # every backward step into an advance of nearly a full turn, and
            # the cycle count then rises by the step count rather than by the
            # number of times round.
            delta = (here - previous + math.pi) % (2.0 * math.pi) - math.pi
            unwrapped += delta
            previous = here
            if driver_phase is not None:
                gap = (here - driver_phase(self.t)) % (2.0 * math.pi)
                offsets.append(cmath.exp(1j * gap))
                if previous_difference is not None:
                    difference += (gap - previous_difference + math.pi) % (
                        2.0 * math.pi
                    ) - math.pi
                previous_difference = gap
        plv = abs(sum(offsets) / len(offsets)) if offsets else 0.0
        offset = cmath.phase(sum(offsets) / len(offsets)) if offsets else 0.0
        slip = abs(difference) / (2.0 * math.pi)
        return Orbit(
            duration_s=steps * dt,
            cycles=abs(unwrapped) / (2.0 * math.pi),
            mean_amplitude=sum(amplitudes) / len(amplitudes),
            exertion=sum(work) / len(work),
            synchrony=plv,
            drift_cycles=slip,
            locked=bool(offsets) and plv >= LOCK_THRESHOLD and slip < MAX_SLIP_CYCLES,
            phase_offset=offset,
        )


#: Phase-locking value above which two rhythms count as locked. The number is
#: the null it has to clear: N phase differences drawn at random give a PLV of
#: about ``sqrt(pi / N) / 2``, so over the hundreds of samples a bout collects,
#: anything this large is not chance. Held as a constant because the sweep
#: below needs one criterion across every point in the band.
LOCK_THRESHOLD = 0.8

#: Whole turns the phase difference may slip across a bout and still count as
#: locked. One turn is the natural cut: a difference that has gone all the way
#: round has visited every relation the two rhythms can be in, which is what
#: unlocked means.
MAX_SLIP_CYCLES = 1.0


def sinusoidal_driver(period_s: float, amplitude: float) -> tuple[
    Callable[[float], float], Callable[[float], float]
]:
    """An external rhythm, as a forcing function and its own phase."""
    w = (2.0 * math.pi) / period_s
    return (lambda t: amplitude * math.sin(w * t)), (lambda t: (w * t) % (2.0 * math.pi))


def entrainment_band(
    *,
    natural_period_s: float,
    drive_amplitude: float,
    mu: float = DEFAULT_MU,
    span: float = 0.6,
    samples: int = 25,
    cycles: int = 60,
) -> tuple[float, float, list[tuple[float, float]]]:
    """Sweep the driving frequency and find where the oscillator locks.

    Returns the low and high edges of the locked band as ratios of the natural
    frequency, together with every sampled point. The band is measured, never
    assumed, because its width as a function of drive strength is the whole
    empirical content of the entrainment claim.
    """
    points: list[tuple[float, float]] = []
    low: float | None = None
    high: float | None = None
    for i in range(samples):
        ratio = (1.0 - span / 2.0) + span * (i / max(samples - 1, 1))
        drive_period = natural_period_s / ratio
        force, phase = sinusoidal_driver(drive_period, drive_amplitude)
        osc = ExpressiveOscillator(period_s=natural_period_s, mu=mu, x=0.1, v=0.0)
        # Discard the approach to the orbit; a lock measured through the
        # transient reports the transient.
        osc.run(natural_period_s * 20, driver=force, driver_phase=phase)
        orbit = osc.run(natural_period_s * cycles, driver=force, driver_phase=phase)
        points.append((ratio, orbit.synchrony))
        if orbit.locked:
            low = ratio if low is None else min(low, ratio)
            high = ratio if high is None else max(high, ratio)
    if low is None or high is None:
        return 0.0, 0.0, points
    return low, high, points


def is_limit_cycle(
    *,
    period_s: float = 1.0,
    mu: float = DEFAULT_MU,
    starts: Sequence[tuple[float, float]] = ((0.05, 0.0), (3.0, 0.0), (0.5, 2.0)),
    cycles: int = 60,
    tolerance: float = 0.05,
) -> tuple[bool, list[float]]:
    """Do very different starting states end on the same orbit?

    This is the claim the module rests on, so it is checked rather than
    stated. A motion that merely repeats will hold whatever amplitude it was
    given; a limit cycle forgets it. The returned amplitudes are the evidence.
    """
    finals: list[float] = []
    for x0, v0 in starts:
        osc = ExpressiveOscillator(period_s=period_s, mu=mu, x=x0, v=v0)
        orbit = osc.run(period_s * cycles)
        finals.append(orbit.mean_amplitude)
    if not finals:
        return False, finals
    spread = (max(finals) - min(finals)) / max(max(finals), 1e-9)
    return spread <= tolerance, finals


@dataclass
class ExpressiveLedger:
    """What accrues from doing something that has no finish line.

    A trajectory's worth is read at its endpoint. An orbit has no endpoint, so
    worth accrues per unit time and the ledger is the whole account. Nothing
    here can answer "how far along is it", and that is the point: a caller
    that needs completion is holding the wrong kind of action.
    """

    bouts: list[Orbit] = field(default_factory=list)

    def record(self, orbit: Orbit) -> None:
        self.bouts.append(orbit)
        if len(self.bouts) > 256:
            del self.bouts[: len(self.bouts) - 256]

    @property
    def time_on_cycle_s(self) -> float:
        return float(sum(b.duration_s for b in self.bouts))

    @property
    def mean_synchrony(self) -> float:
        driven = [b.synchrony for b in self.bouts if b.synchrony > 0]
        return float(sum(driven) / len(driven)) if driven else 0.0

    @property
    def mean_exertion(self) -> float:
        return float(sum(b.exertion for b in self.bouts) / len(self.bouts)) if self.bouts else 0.0

    def completion(self) -> None:
        """There is none. Present so that asking returns nothing rather than a
        number a scheduler would then act on."""
        return None

    def status(self) -> dict[str, Any]:
        return {
            "bouts": len(self.bouts),
            "time_on_cycle_s": round(self.time_on_cycle_s, 2),
            "mean_synchrony": round(self.mean_synchrony, 4),
            "mean_exertion": round(self.mean_exertion, 4),
            "locked_bouts": sum(1 for b in self.bouts if b.locked),
            "completion": None,
        }


_LEDGER: ExpressiveLedger | None = None


def get_expressive_ledger() -> ExpressiveLedger:
    global _LEDGER
    if _LEDGER is None:
        _LEDGER = ExpressiveLedger()
    return _LEDGER


def reset_expressive_ledger_for_test() -> None:
    global _LEDGER
    _LEDGER = None
