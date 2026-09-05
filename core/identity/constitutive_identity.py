"""core/identity/constitutive_identity.py — an identity that is its practices.

Some identities are facts about a thing: a process id, a model hash, a serial
number. You can read them off, and nothing the thing does changes them.

Others do not work that way. Being a teacher, a craftsman, a Debian
maintainer, a woman: these are not read off, and there is no field to read.
Simone de Beauvoir put the general form of it in one line — one is not born a
woman, one becomes one — and Judith Butler's argument thirty years later was
that the becoming never finishes, because the identity is the repetition of
the practices and stops when they do. Neither is a claim about women in
particular. It is a claim about a kind of identity, and it has a shape you can
compute.

The shape is this. There are practices, each recurring at its own rate. They
are not independent: doing one makes another more likely in the same stretch
of time, and that mutual pull is measurable from when they actually happen.
Above some coupling the practices fall into a common rhythm and the whole
reads as one thing. Below it they stay a list of unrelated habits, however
many of them there are. That is the Kuramoto transition, and it is the right
model here for a reason that is not decorative: it is the standard account of
many oscillators with different natural frequencies producing one order
parameter, and it has the property the phenomenology insists on — the order
parameter has no existence apart from the oscillators.

Three consequences follow, and all three are enforced below.

**The label is downstream and can never write back.** ``declare()`` records
that something calls itself X. Nothing in the coherence calculation reads it.
An identity that could be established by asserting it would be exactly the
essentialism Beauvoir was arguing against, and it would also be a system that
manufactures the state it reports — the failure
``core/conation/invariants.py`` exists to prevent one layer down.

**Coherence is meaningless below its own null.** N practices with unrelated
phases still produce a nonzero order parameter, by the same arithmetic that
makes a short random walk end up somewhere. That floor is
``sqrt(pi / N) / 2``, it is reported next to every reading, and a coherence
under it is reported as absent rather than small.

**Removing a practice has to cost something.** ``load_bearing()`` re-runs the
measurement with each practice deleted and ranks them by the fall. A practice
whose removal changes nothing is not constituting anything, which is a finding
about that practice rather than a reason to hide it.

The module is general. It knows nothing about which identity it is holding,
and the practices are whatever the caller names.
"""

from __future__ import annotations

import cmath
import logging
import math
import statistics
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("Aura.Identity.Constitutive")

#: Two enactments cannot give an interval, and one interval cannot give a
#: spread. Below this a practice reports its period as unknown and is carried
#: at the population median rather than at an invented number.
MIN_ENACTMENTS_FOR_PERIOD = 3

#: Enactments closer together than this count as one episode for the purpose
#: of measuring which practices travel together. It is a resolution, not a
#: weight: co-occurrence needs a window, and the window is stated rather than
#: buried.
DEFAULT_EPISODE_WINDOW_S = 3600.0

#: How much history one practice keeps. Bounded because a constitutive model
#: is a working model, and an unbounded per-practice log is a leak.
MAX_ENACTMENTS = 512


@dataclass
class Enactment:
    """One occasion on which a practice was carried out."""

    at: float
    intensity: float = 1.0
    context: str = ""


@dataclass
class Practice:
    """A recurring doing, with its own rate and its own history.

    The period is estimated rather than declared. A caller who knows the
    period may pass ``declared_period_s``, and the estimate still wins once
    there is enough history to make one, because the point of the object is
    to describe what happens rather than what was intended.
    """

    name: str
    declared_period_s: float | None = None
    enactments: list[Enactment] = field(default_factory=list)

    def record(self, at: float, intensity: float = 1.0, context: str = "") -> None:
        self.enactments.append(Enactment(at=at, intensity=float(intensity), context=context))
        if len(self.enactments) > MAX_ENACTMENTS:
            del self.enactments[: len(self.enactments) - MAX_ENACTMENTS]

    def intervals(self) -> list[float]:
        times = sorted(e.at for e in self.enactments)
        return [b - a for a, b in zip(times, times[1:], strict=False) if b > a]

    def period_s(self) -> float | None:
        """Median gap between enactments, or the declared period, or nothing.

        The median rather than the mean, because one long absence should not
        redefine the rhythm of a practice that is otherwise weekly.
        """
        gaps = self.intervals()
        if len(gaps) >= MIN_ENACTMENTS_FOR_PERIOD - 1:
            return float(statistics.median(gaps))
        return self.declared_period_s

    def omega(self, fallback_period_s: float) -> float:
        """Natural angular frequency, in radians per second."""
        period = self.period_s() or fallback_period_s
        return (2.0 * math.pi) / max(period, 1e-9)

    def phase(self, at: float, fallback_period_s: float) -> float | None:
        """Where this practice is in its own cycle, in radians.

        ``None`` when it has never been enacted: a practice that has not
        happened has no phase, and giving it zero would place it in perfect
        agreement with every other practice that has also never happened.
        """
        if not self.enactments:
            return None
        period = self.period_s() or fallback_period_s
        elapsed = at - max(e.at for e in self.enactments)
        return (2.0 * math.pi * (elapsed / max(period, 1e-9))) % (2.0 * math.pi)

    def recent_intensity(self, at: float, window_s: float) -> float:
        recent = [e.intensity for e in self.enactments if at - e.at <= window_s]
        return float(sum(recent) / len(recent)) if recent else 0.0


@dataclass(frozen=True)
class Coherence:
    """One reading of the order parameter, with everything needed to doubt it."""

    r: float
    """Kuramoto order parameter over the practices that have a phase."""

    psi: float
    """Mean phase, in radians. The rhythm the practices are keeping."""

    n_active: int
    incoherent_floor: float
    """What ``r`` would be for this many practices with unrelated phases."""

    k_effective: float
    k_critical: float
    """Mean-field threshold. Below it, sustained coherence is not predicted."""

    supercritical: bool
    measured_at: float

    @property
    def coherent(self) -> bool:
        """Whether the reading clears its own null."""
        return self.r > self.incoherent_floor

    @property
    def unexplained(self) -> bool:
        """Coherence the coupling does not account for.

        The model earns its keep by being able to be wrong. Practices that
        hold together while the measured coupling says they should not are
        being held together by something this model does not contain, and
        saying so is more use than a number that always fits.
        """
        return self.coherent and not self.supercritical

    def as_dict(self) -> dict[str, Any]:
        return {
            "r": round(self.r, 4),
            "psi": round(self.psi, 4),
            "n_active": self.n_active,
            "floor": round(self.incoherent_floor, 4),
            "k_effective": round(self.k_effective, 4),
            "k_critical": round(self.k_critical, 4),
            "supercritical": self.supercritical,
            "coherent": self.coherent,
            "unexplained": self.unexplained,
        }


@dataclass(frozen=True)
class Contribution:
    """What one practice is holding up."""

    practice: str
    r_without: float
    delta: float
    """Fall in coherence when this practice is removed. Negative means the
    practice is pulling against the rest."""

    enactments: int


def incoherent_floor(n: int) -> float:
    """Expected order parameter for ``n`` practices with unrelated phases.

    A sum of n unit vectors with independent uniform directions is a
    two-dimensional random walk, so ``E[R] = sqrt(pi / n) / 2``. Reporting a
    coherence without this next to it invites reading 0.3 across four
    practices as weak coherence when it is the null.
    """
    if n <= 0:
        return 1.0
    return math.sqrt(math.pi / n) / 2.0


class ConstitutiveIdentity:
    """An identity held as the coherence of the practices that enact it.

    Nothing here stores whether the identity obtains. ``coherence()``
    recomputes it from the enactment record every time it is asked, which is
    what makes the object honest and also what makes it cheap to falsify:
    stop enacting and the number falls on its own.
    """

    def __init__(
        self,
        name: str,
        *,
        episode_window_s: float = DEFAULT_EPISODE_WINDOW_S,
    ) -> None:
        self.name = name
        self.episode_window_s = float(episode_window_s)
        self._practices: dict[str, Practice] = {}
        self._declarations: list[tuple[float, str, str]] = []
        self._history: list[Coherence] = []

    # ---------------------------------------------------------------- record

    def add_practice(self, name: str, *, declared_period_s: float | None = None) -> Practice:
        practice = self._practices.get(name)
        if practice is None:
            practice = Practice(name=name, declared_period_s=declared_period_s)
            self._practices[name] = practice
        elif declared_period_s is not None:
            practice.declared_period_s = declared_period_s
        return practice

    def enact(
        self,
        name: str,
        *,
        at: float | None = None,
        intensity: float = 1.0,
        context: str = "",
    ) -> None:
        """Record that a practice was carried out. The only way in."""
        practice = self.add_practice(name)
        practice.record(at if at is not None else time.time(), intensity, context)

    def declare(self, label: str, *, source: str = "", at: float | None = None) -> None:
        """Record that something called this identity ``label``.

        Kept, and kept out of the arithmetic. The record exists so that
        ``unsupported_declarations()`` can name a label nothing is enacting,
        which is the diagnostic the whole design is arranged to make possible.
        """
        self._declarations.append((at if at is not None else time.time(), label, source))

    # --------------------------------------------------------------- measure

    def practices(self) -> list[str]:
        return sorted(self._practices)

    def _fallback_period(self) -> float:
        periods = [p.period_s() for p in self._practices.values()]
        known = [p for p in periods if p]
        if known:
            return float(statistics.median(known))
        return DEFAULT_EPISODE_WINDOW_S

    def _episodes(self) -> list[set[str]]:
        """Group every enactment into windows, and report who shared each one.

        Coupling in this model is not assumed. It is read off the record: two
        practices are coupled to the extent that they turn up in the same
        stretch of time.
        """
        stamped: list[tuple[float, str]] = [
            (e.at, name)
            for name, practice in self._practices.items()
            for e in practice.enactments
        ]
        stamped.sort()
        episodes: list[set[str]] = []
        current: set[str] = set()
        opened_at: float | None = None
        for at, name in stamped:
            if opened_at is None or at - opened_at > self.episode_window_s:
                if current:
                    episodes.append(current)
                current = set()
                opened_at = at
            current.add(name)
        if current:
            episodes.append(current)
        return episodes

    def coupling(self, names: Sequence[str] | None = None) -> dict[tuple[str, str], float]:
        """Jaccard co-occurrence over episodes, for each unordered pair.

        The Jaccard form rather than a raw count, so that a practice happening
        constantly does not read as coupled to everything simply by being
        everywhere.
        """
        keys = sorted(names) if names is not None else self.practices()
        episodes = self._episodes()
        out: dict[tuple[str, str], float] = {}
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                both = sum(1 for ep in episodes if a in ep and b in ep)
                either = sum(1 for ep in episodes if a in ep or b in ep)
                out[(a, b)] = (both / either) if either else 0.0
        return out

    def _k_effective(self, names: Sequence[str]) -> float:
        pairs = self.coupling(names)
        if not pairs:
            return 0.0
        return float(sum(pairs.values()) / len(pairs))

    def _k_critical(self, names: Sequence[str]) -> float:
        """Mean-field threshold from the spread of natural frequencies.

        For a Lorentzian spread of half-width gamma the Kuramoto threshold is
        exactly ``2 * gamma``, and the median absolute deviation of a
        Lorentzian is exactly gamma. So the robust dispersion of the measured
        frequencies gives the threshold with no fitted constant in between,
        and a heavy tail — one practice on a wildly different rhythm — does
        not move it the way a variance would.
        """
        fallback = self._fallback_period()
        omegas = [self._practices[n].omega(fallback) for n in names if n in self._practices]
        if len(omegas) < 2:
            return 0.0
        centre = statistics.median(omegas)
        mad = statistics.median([abs(w - centre) for w in omegas])
        if mad <= 0.0:
            # Identical rhythms lock under any coupling at all. The threshold
            # is zero, and that is a real answer rather than a missing one.
            return 0.0
        # Normalised against the centre so the threshold is comparable with
        # the dimensionless Jaccard coupling it is checked against.
        return 2.0 * mad / max(abs(centre), 1e-9)

    def _measure(self, names: Sequence[str], at: float) -> Coherence:
        fallback = self._fallback_period()
        phases = [
            self._practices[n].phase(at, fallback)
            for n in names
            if n in self._practices
        ]
        live = [p for p in phases if p is not None]
        n = len(live)
        if n == 0:
            return Coherence(
                r=0.0, psi=0.0, n_active=0, incoherent_floor=1.0,
                k_effective=0.0, k_critical=0.0, supercritical=False,
                measured_at=at,
            )
        z = sum(cmath.exp(1j * p) for p in live) / n
        k_eff = self._k_effective(names)
        k_c = self._k_critical(names)
        return Coherence(
            r=abs(z),
            psi=cmath.phase(z) % (2.0 * math.pi),
            n_active=n,
            incoherent_floor=incoherent_floor(n),
            k_effective=k_eff,
            k_critical=k_c,
            supercritical=k_eff > k_c,
            measured_at=at,
        )

    def coherence(self, *, at: float | None = None, record: bool = True) -> Coherence:
        """Measure the identity now. Recomputed from the record every time."""
        moment = at if at is not None else time.time()
        reading = self._measure(self.practices(), moment)
        if record:
            self._history.append(reading)
            if len(self._history) > MAX_ENACTMENTS:
                del self._history[: len(self._history) - MAX_ENACTMENTS]
        return reading

    def project(self, *, horizon_s: float | None = None, steps: int = 400,
                at: float | None = None) -> float:
        """Integrate the Kuramoto flow forward and report where R settles.

        The measurement above reads the phases the practices are in. This runs
        the model instead: it takes the measured frequencies and the measured
        coupling, evolves

            dtheta_i/dt = omega_i + (K/N) * sum_j sin(theta_j - theta_i)

        from the phases now, and reports the order parameter at the end. The
        two answers are independent, which is what makes the pair worth
        having — a projection that misses the later measurement is the model
        being wrong about this identity, and ``unexplained`` above is the same
        disagreement caught one step earlier.
        """
        moment = at if at is not None else time.time()
        names = self.practices()
        fallback = self._fallback_period()
        phases = []
        omegas = []
        for name in names:
            phase = self._practices[name].phase(moment, fallback)
            if phase is None:
                continue
            phases.append(phase)
            omegas.append(self._practices[name].omega(fallback))
        n = len(phases)
        if n == 0:
            return 0.0
        if n == 1:
            return 1.0
        centre = statistics.median(omegas) or 1.0
        # Work in units of the median period, so a horizon of "a few cycles"
        # means the same thing for a daily practice and a yearly one.
        cycle_s = (2.0 * math.pi) / abs(centre)
        span = horizon_s if horizon_s is not None else 20.0 * cycle_s
        dt = span / max(steps, 1)
        # The measured coupling is a dimensionless co-occurrence rate and the
        # frequencies are radians per second, so the flow needs the coupling
        # carried back into frequency units before the two terms are added.
        # This is the same scaling that makes the dimensionless threshold in
        # ``_k_critical`` comparable with ``_k_effective``.
        k = self._k_effective(names) * abs(centre)
        theta = list(phases)
        # The order parameter of a handful of oscillators fluctuates hard, and
        # below the threshold it fluctuates around its null rather than
        # sitting at it. A single endpoint is therefore one draw from that
        # distribution and says almost nothing; the average over the back half
        # of the run is the quantity the mean-field result is about.
        tail: list[float] = []
        for step in range(steps):
            increments = []
            for i in range(n):
                pull = sum(math.sin(theta[j] - theta[i]) for j in range(n))
                increments.append(omegas[i] + (k / n) * pull)
            for i in range(n):
                theta[i] = (theta[i] + dt * increments[i]) % (2.0 * math.pi)
            if step >= steps // 2:
                tail.append(abs(sum(cmath.exp(1j * t) for t in theta) / n))
        return float(sum(tail) / len(tail)) if tail else 0.0

    def coherence_without(self, practice: str, *, at: float | None = None) -> Coherence:
        """The same measurement with one practice deleted."""
        moment = at if at is not None else time.time()
        remaining = [n for n in self.practices() if n != practice]
        return self._measure(remaining, moment)

    def load_bearing(self, *, at: float | None = None) -> list[Contribution]:
        """Rank the practices by how much coherence each one is holding up.

        This is the ablation the module exists to support. A constitutive
        claim that survives deleting any of its constituents was not a
        constitutive claim.
        """
        moment = at if at is not None else time.time()
        whole = self._measure(self.practices(), moment)
        out: list[Contribution] = []
        for name in self.practices():
            without = self.coherence_without(name, at=moment)
            out.append(
                Contribution(
                    practice=name,
                    r_without=without.r,
                    delta=whole.r - without.r,
                    enactments=len(self._practices[name].enactments),
                )
            )
        out.sort(key=lambda c: c.delta, reverse=True)
        return out

    # ----------------------------------------------------------- diagnostics

    def unsupported_declarations(self, *, since_s: float = 30 * 86400.0,
                                 at: float | None = None) -> list[str]:
        """Labels claimed with nothing being done that would make them true.

        The whole architecture is arranged so that this list can exist. In a
        design where asserting the identity established it, every declaration
        would be supported by construction and the question could not be put.
        """
        moment = at if at is not None else time.time()
        active = {
            name for name, p in self._practices.items()
            if any(moment - e.at <= since_s for e in p.enactments)
        }
        if active:
            return []
        return sorted({label for _, label, _ in self._declarations})

    def dormant(self, *, since_s: float = 30 * 86400.0,
                at: float | None = None) -> list[str]:
        """Practices that have stopped. What a becoming looks like unravelling."""
        moment = at if at is not None else time.time()
        return sorted(
            name for name, p in self._practices.items()
            if p.enactments and moment - max(e.at for e in p.enactments) > since_s
        )

    def status(self, *, at: float | None = None) -> dict[str, Any]:
        moment = at if at is not None else time.time()
        reading = self.coherence(at=moment, record=False)
        return {
            "identity": self.name,
            "coherence": reading.as_dict(),
            "practices": {
                name: {
                    "enactments": len(p.enactments),
                    "period_s": p.period_s(),
                    "last": max((e.at for e in p.enactments), default=None),
                }
                for name, p in sorted(self._practices.items())
            },
            "load_bearing": [
                {"practice": c.practice, "delta": round(c.delta, 4)}
                for c in self.load_bearing(at=moment)[:5]
            ],
            "dormant": self.dormant(at=moment),
            "unsupported_declarations": self.unsupported_declarations(at=moment),
            "declarations": len(self._declarations),
        }


class ConstitutiveRegistry:
    """Every identity held this way, by name."""

    def __init__(self) -> None:
        self._identities: dict[str, ConstitutiveIdentity] = {}

    def get(self, name: str) -> ConstitutiveIdentity:
        identity = self._identities.get(name)
        if identity is None:
            identity = ConstitutiveIdentity(name)
            self._identities[name] = identity
        return identity

    def names(self) -> list[str]:
        return sorted(self._identities)

    def enact(self, identity: str, practice: str, **kw: Any) -> None:
        self.get(identity).enact(practice, **kw)

    def observe(self, practices: Mapping[str, Iterable[str]], *,
                at: float | None = None) -> None:
        """Record one episode across several identities at once."""
        moment = at if at is not None else time.time()
        for identity, names in practices.items():
            for name in names:
                self.get(identity).enact(name, at=moment)

    def status(self) -> dict[str, Any]:
        return {name: ident.status() for name, ident in sorted(self._identities.items())}


_REGISTRY: ConstitutiveRegistry | None = None


def get_constitutive_registry() -> ConstitutiveRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = ConstitutiveRegistry()
    return _REGISTRY


def reset_constitutive_registry_for_test() -> None:
    global _REGISTRY
    _REGISTRY = None
