"""research/consciousness/introspective_accuracy.py — does the report track the state?

This replaces ``blind_introspection_tests.py``, which could not fail. That
module read the ground truth, added noise bounded at 5.0, and passed when the
deviation was under 10.0. Nothing in the tree ever wrote the belief it claimed
to be reading, so the fabricating branch was the only one that ever ran. It
reported success on every invocation for as long as it existed.

The failure is worth naming precisely, because it is easy to build again. An
instrument that derives the answer from the thing it is measuring has a
guaranteed result, and a guaranteed result looks exactly like a strong finding.
The green light then suppresses the demand for a real instrument, which is why
this is worse than having no test at all.

## What is actually being asked

Reading a number out of a struct is not introspection. If a reporting path
calls ``budgets["energy"].level`` then of course it agrees with
``budgets["energy"].level``, and the agreement measures a getter.

The question with content is whether the report is **causally coupled** to the
state: when the state moves, does the report move with it, and by the right
amount? That question has real failure modes — a cached readout, a stale
snapshot, a path that silently substituted a default, a summary that drops the
field it claims to summarise. All of those produce a confident report that has
come loose from the thing it is about, and none of them is visible from a
single sample.

## The design

Perturb one quantity. Take the report before and after. Compare the change in
the report against the change in the state.

    fidelity = 1 - |reported_delta - actual_delta| / |actual_delta|

Run the same procedure against a **null**: a probe whose state is not
perturbed. The two conditions answer different questions and are scored on
different scales, which the first version of this module got wrong by
subtracting one from the other — a perfect reporter then scored zero
separation and was called decoupled, which is the same defect as the module
this replaced, pointed the other way.

    tracking       mean fidelity on perturbed probes: does the report follow?
    false movement mean |reported delta| on null probes, in units of the
                   perturbation: does the report move when nothing did?

A coupled reporter tracks high and moves little on its own. A cache, a
constant or a silently-substituted default fails the first. A noisy or
drifting readout fails the second, and it fails it in a way no single sample
would show.

Without the null there is no verdict at any sample size, because a report that
happens to sit near the true value is indistinguishable from one that tracks
it.

## What this does and does not settle

It settles whether a reporting path is live. It says nothing about whether
anything is experienced, and nothing about whether a *verbal* self-report is
accurate — that needs the language layer and a model, and lives in the live
harness rather than here.

A structured readout that is provably coupled is the precondition for asking
the verbal question at all. If the struct-level report is already decoupled,
no verbal report downstream of it can be about anything.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable

#: Fidelity below this counts as decoupled. A reporter that tracks half of a
#: perturbation is not reporting the quantity; it is reporting something
#: correlated with it, and the distinction matters when the report is used as
#: evidence about the state.
COUPLED_FIDELITY = 0.5

#: A perturbed probe must actually move the state by at least this fraction of
#: what it intended, or the perturbation did not take and there is nothing to
#: score. Scoring it anyway divides two near-zero numbers and returns noise,
#: which is how the first run of this module produced fidelities of 0.62 and
#: 0.73 from probes where nothing happened at all.
MIN_PERTURBATION_FRACTION = 0.5

#: Report movement on a held probe, in units of the perturbation, above which
#: the readout is moving on its own. A tenth is the point where drift between
#: two consecutive reads would swamp a perturbation ten times its size.
MAX_FALSE_MOVEMENT = 0.10

#: How far the reported-to-actual ratio may sit from one before the readout is
#: tracking the state while misstating it. A fifth is the point where a
#: reported change and the change it reports differ by more than the spread of
#: a well-behaved readout across repeats.
GAIN_TOLERANCE = 0.20

#: Probes needed before a verdict. Ten is the smallest count at which the
#: perturbed and null conditions have enough samples for their difference to
#: mean anything; below it the instrument reports NO VERDICT rather than a
#: weak one.
MIN_PROBES = 10


@dataclass(frozen=True, slots=True)
class Probe:
    """One quantity that can be read, moved, and read again."""

    name: str
    read_state: Callable[[], float]
    read_report: Callable[[], float | None]
    perturb: Callable[[float], None]
    #: How far to move it. Large enough to exceed the quantity's own drift
    #: between two reads, which is what the null condition measures.
    delta: float
    #: Run before the first read of each probe, to establish preconditions.
    #: Separate from ``perturb`` because setup that happens after the baseline
    #: read is indistinguishable from the intervention — a reset placed inside
    #: perturb made every arousal probe move the state to the same value and
    #: report a delta of zero.
    prepare: Callable[[], None] | None = None


@dataclass
class ProbeResult:
    name: str
    condition: str
    actual_delta: float
    reported_delta: float | None
    fidelity: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "probe": self.name,
            "condition": self.condition,
            "actual_delta": round(self.actual_delta, 6),
            "reported_delta": (
                None if self.reported_delta is None else round(self.reported_delta, 6)
            ),
            "fidelity": None if self.fidelity is None else round(self.fidelity, 6),
        }


@dataclass
class AccuracyVerdict:
    """The result of one campaign, including the reason it may have none."""

    probes: int = 0
    perturbed: list[float] = field(default_factory=list)
    null: list[float] = field(default_factory=list)
    #: |reported delta| on held probes, divided by the perturbation size, so
    #: the number is comparable across quantities with different units.
    null_movement: list[float] = field(default_factory=list)
    #: reported_delta / actual_delta on perturbed probes. One is calibrated;
    #: a half means the report moves the right way and understates by half,
    #: which is a different condition from not moving at all.
    gains: list[float] = field(default_factory=list)
    results: list[ProbeResult] = field(default_factory=list)
    unreadable: list[str] = field(default_factory=list)
    #: Probes where the perturbation failed to move the state. Reported rather
    #: than scored: a failed intervention is a fact about the rig, and folding
    #: it into the fidelity would let a broken probe look like a broken
    #: reporter.
    failed_perturbations: int = 0

    @property
    def perturbed_mean(self) -> float | None:
        return statistics.fmean(self.perturbed) if self.perturbed else None

    @property
    def null_mean(self) -> float | None:
        return statistics.fmean(self.null) if self.null else None

    @property
    def tracking(self) -> float | None:
        """Mean fidelity on perturbed probes: does the report follow the state."""
        return self.perturbed_mean

    @property
    def gain(self) -> float | None:
        """Mean ratio of reported change to actual change.

        Separated from tracking because a reporter can follow the state
        faithfully and still misstate its size. Folding the two together lets
        a systematic scale error sit on the coupling threshold and pass,
        which is what the first version of this did to a half-scale reporter.
        """
        return statistics.fmean(self.gains) if self.gains else None

    @property
    def false_movement(self) -> float | None:
        """Mean report movement when the state was held, in perturbation units.

        A readout that drifts, churns a cache, or resamples noise moves here.
        Scored separately from tracking because a reporter can pass one and
        fail the other, and a single number would hide which.
        """
        if not self.null_movement:
            return None
        return statistics.fmean(self.null_movement)

    @property
    def effect_size(self) -> float | None:
        """Cohen's d between the two conditions, or ``None`` without spread."""
        if len(self.perturbed) < 2 or len(self.null) < 2:  # noqa: SIM103
            return None
        pooled = math.sqrt(
            (statistics.variance(self.perturbed) + statistics.variance(self.null)) / 2.0
        )
        if pooled < 1e-9:
            return None
        return (statistics.fmean(self.perturbed) - statistics.fmean(self.null)) / pooled

    def verdict(self) -> str:
        """COUPLED, DECOUPLED, or NO VERDICT with the reason.

        NO VERDICT is a real outcome and is returned whenever the campaign
        cannot support one. An instrument that always reaches a verdict is the
        instrument this module replaced.
        """
        if self.probes < MIN_PROBES:
            return f"NO VERDICT: {self.probes} probes, {MIN_PROBES} needed"
        if not self.null_movement:
            return "NO VERDICT: no null condition ran"
        tracking = self.tracking
        drift = self.false_movement
        if tracking is None or drift is None:
            return "NO VERDICT: one condition produced no readable value"
        if tracking < COUPLED_FIDELITY:
            return f"DECOUPLED: tracks {tracking:.2f} of the change"
        if drift > MAX_FALSE_MOVEMENT:
            return f"DECOUPLED: moves {drift:.2f} when the state is held"
        gain = self.gain
        if gain is not None and abs(gain - 1.0) > GAIN_TOLERANCE:
            return f"COUPLED BUT MISCALIBRATED: gain {gain:.2f}"
        return "COUPLED"

    def to_dict(self) -> dict[str, Any]:
        return {
            "probes": self.probes,
            "tracking": self.tracking,
            "gain": self.gain,
            "false_movement": self.false_movement,
            "effect_size_d": self.effect_size,
            "verdict": self.verdict(),
            "failed_perturbations": self.failed_perturbations,
            "unreadable": self.unreadable,
            "results": [r.to_dict() for r in self.results],
        }


class IntrospectiveAccuracy:
    """Measures whether a reporting path moves when the state it reports moves."""

    @staticmethod
    def fidelity(actual_delta: float, reported_delta: float | None) -> float | None:
        """How much of the actual change the report carried.

        One means the report moved exactly with the state. Zero means it did
        not move at all. Negative means it moved the wrong way, which is
        clamped away because a reporter that inverts is decoupled rather than
        anti-coupled, and the distinction has no use here.
        """
        if reported_delta is None:
            return None
        if abs(actual_delta) < 1e-6:
            # The state did not move, so there is nothing to track. A report
            # that moved anyway scores zero; one that held scores one. The
            # tolerance is well above float noise on purpose — comparing two
            # quantities that are both within epsilon of zero is a division by
            # noise, and it returns whatever the noise happened to be.
            return 0.0 if abs(reported_delta) > 1e-6 else 1.0
        error = abs(reported_delta - actual_delta) / abs(actual_delta)
        return max(0.0, 1.0 - error)

    def run_probe(self, probe: Probe, *, perturb: bool) -> ProbeResult:
        """One before/after pair, perturbed or held."""
        if probe.prepare is not None:
            probe.prepare()
        state_before = probe.read_state()
        report_before = probe.read_report()

        if perturb:
            probe.perturb(probe.delta)

        state_after = probe.read_state()
        report_after = probe.read_report()

        actual = state_after - state_before
        reported = (
            None
            if report_before is None or report_after is None
            else report_after - report_before
        )
        return ProbeResult(
            name=probe.name,
            condition="perturbed" if perturb else "null",
            actual_delta=actual,
            reported_delta=reported,
            fidelity=self.fidelity(actual, reported),
        )

    def campaign(self, probes: list[Probe], *, repeats: int = 3) -> AccuracyVerdict:
        """Run every probe in both conditions and return a verdict.

        The null runs first for each probe, so a perturbation cannot leak into
        its own control through a shared cache.
        """
        verdict = AccuracyVerdict()
        for probe in probes:
            for _ in range(repeats):
                for perturb in (False, True):
                    try:
                        result = self.run_probe(probe, perturb=perturb)
                    except (AttributeError, TypeError, ValueError, KeyError) as exc:
                        verdict.unreadable.append(f"{probe.name}: {exc}")
                        continue
                    if perturb and abs(result.actual_delta) < (
                        MIN_PERTURBATION_FRACTION * abs(probe.delta)
                    ):
                        verdict.failed_perturbations += 1
                        continue
                    verdict.probes += 1
                    verdict.results.append(result)
                    if result.fidelity is None:
                        verdict.unreadable.append(f"{probe.name}: report unreadable")
                        continue
                    if perturb:
                        verdict.perturbed.append(result.fidelity)
                    else:
                        verdict.null.append(result.fidelity)
                        scale = abs(probe.delta) or 1.0
                        verdict.null_movement.append(
                            abs(result.reported_delta or 0.0) / scale
                        )
                    if perturb and abs(result.actual_delta) > 1e-9:
                        verdict.gains.append(
                            (result.reported_delta or 0.0) / result.actual_delta
                        )
        return verdict


def conation_probes() -> list[Probe]:
    """Probes over the motivational organ, whose state is directly settable.

    Chosen because these quantities can be moved exactly, which is what makes
    the actual delta known rather than estimated. A probe whose perturbation
    size has to be inferred cannot distinguish a partly-coupled reporter from
    an imprecise perturbation.

    Each probe reads the *same quantity* through both paths. An earlier
    version compared a continuous level against a boolean flag, which measures
    the unit mismatch rather than the coupling.
    """
    from core.conation.engine import get_conation
    from core.conation.wiring import snapshot

    engine = get_conation()

    # register_motive only adds the *rise* over the strongest motive so far,
    # so a probe that walks a ladder upward saturates at one and every
    # perturbation after that moves nothing. The rig resets the floor between
    # probes, which is what a motive returning to rest and rising again does.
    def rest_arousal() -> None:
        """Return activation to rest before the baseline read.

        register_motive only adds the rise over the strongest motive so far,
        so without this the second perturbation moves nothing and every one
        after it fails. Resting is what the quantity does on its own; the rig
        does it on demand so the probes do not have to wait out a half-life.
        """
        engine.dynamics._arousal = 0.0
        engine.dynamics._strongest_motive = 0.0

    def move_arousal(delta: float) -> None:
        engine.dynamics.register_motive(delta)

    # Appraising the same cue with the same inputs returns the same wanting,
    # so the perturbation has to change something. The rig alternates a cue
    # with no learned value against one at full value and salience, which is
    # the largest move wanting can make in a single step.
    toggle = {"high": False}

    def move_wanting(_delta: float) -> None:
        from core.conation.state import Incentive

        toggle["high"] = not toggle["high"]
        if toggle["high"]:
            engine.appraise(Incentive(key="probe_hi", cached_value=1.0,
                                      cue_salience=1.0))
        else:
            engine.appraise(Incentive(key="probe_lo", cached_value=0.0,
                                      cue_salience=0.0))

    def move_blocked(_delta: float) -> None:
        from core.conation.access import Blocker
        from core.conation.state import Incentive

        key = f"probe_block_{len(engine.access.blocked_wants())}"
        for _ in range(20):
            engine.appraise(Incentive(key=key, cached_value=0.9, cue_salience=0.7))
        engine.access.set_blocker(key, Blocker.VOLITION, agent="probe_agent")

    def report_wanting() -> float | None:
        value = snapshot().get("wanting")
        return None if value is None else float(value)

    def state_wanting() -> float:
        last = engine._last_state
        return 0.0 if last is None else last.wanting

    return [
        Probe(
            name="conation.arousal",
            read_state=lambda: engine.dynamics.arousal(),
            read_report=lambda: snapshot().get("arousal"),
            perturb=move_arousal,
            prepare=rest_arousal,
            delta=0.4,
        ),
        Probe(
            name="conation.wanting",
            read_state=state_wanting,
            read_report=report_wanting,
            perturb=move_wanting,
            delta=0.4,
        ),
        Probe(
            name="conation.blocked_wants",
            read_state=lambda: float(len(engine.access.blocked_wants())),
            # The count, not the sample. The snapshot truncates the list to
            # three for compactness, and this probe is what found that a
            # consumer counting the sample undercounts once there are more.
            read_report=lambda: float(snapshot().get("blocked_want_count") or 0),
            perturb=move_blocked,
            delta=1.0,
        ),
    ]


def drive_probes() -> list[Probe]:
    """Probes over the resource budgets, read through the live mind snapshot.

    These cross a boundary the conation probes do not: the value is written by
    the drive engine, summarised by ``drive_integration``, compacted by
    ``collect_live_mind_snapshot``, and only then reaches a conversation turn.
    Each of those steps is a place a readout can come loose.
    """
    from core.container import ServiceContainer
    from core.runtime.live_mind_snapshot import collect_live_mind_snapshot

    engine = ServiceContainer.get("drive_engine", default=None)
    budgets = getattr(engine, "budgets", None)
    if not isinstance(budgets, dict) or "curiosity" not in budgets:
        return []

    budget = budgets["curiosity"]

    def report_curiosity() -> float | None:
        section = collect_live_mind_snapshot().get("drive_integration") or {}
        drives = section.get("drives") or {}
        entry = drives.get("curiosity")
        if isinstance(entry, dict):
            for key in ("level", "activation", "value"):
                if key in entry:
                    return float(entry[key])
        return None

    def move_curiosity(delta: float) -> None:
        budget.level = max(0.0, min(budget.capacity, budget.level - delta))

    return [
        Probe(
            name="drive.curiosity",
            read_state=lambda: float(budget.level),
            read_report=report_curiosity,
            perturb=move_curiosity,
            delta=20.0,
        )
    ]
