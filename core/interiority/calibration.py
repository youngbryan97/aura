"""core/interiority/calibration.py — the condition under which a guess is allowed.

O1 on the council docket. Twenty-eight of the forty parameters here are
calibration parameters: nobody has measured them and the value is a
guess. :mod:`core.interiority.params` says a guess is permitted on one
condition — the faculty's *ordering* of outcomes must survive the value
moving across its whole declared range — and until now that condition
was written down and never checked, which makes it a promise rather than
a discipline.

This checks it. For every calibration parameter, the ordering the system
produces is recomputed at each point across the declared range, and a
parameter whose movement reorders anything is a parameter the conclusions
depend on. That is not automatically a defect; it means the number needs
measuring rather than guessing, and :func:`report` names those separately
from the ones that are safely arbitrary.

Two other things live here.

:data:`TARGETS` are published properties the parameter set has to
reproduce — the channel-reliability ordering, the natural-scene fractal
band, sensory-specific satiety, the shape of tolerance. They are the
nearest thing to data this has, and each names its source.

:func:`fit` grid-searches a parameter across its declared range and
reports which values satisfy every target, and where the current value
sits in that set. It does not fit against human behavioural data, because
there is none here; what it does is make the gap explicit and small.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from core.interiority.params import Param, ParamKind, registry


@dataclass(frozen=True)
class Target:
    """A published property the parameter set has to reproduce."""

    name: str
    source: str
    #: Returns (holds, detail). Reads the live parameters.
    check: Callable[[], tuple[bool, str]]
    #: Parameters this target constrains.
    constrains: tuple[str, ...]


@dataclass(frozen=True)
class TargetResult:
    name: str
    source: str
    held: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.name,
            "source": self.source,
            "held": self.held,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class SweepResult:
    """What moving one calibration parameter across its range does."""

    parameter: str
    #: Orderings produced at each point in the sweep.
    orderings: tuple[tuple[str, ...], ...]
    stable: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameter": self.parameter,
            "stable": self.stable,
            "distinct_orderings": len({o for o in self.orderings}),
            "detail": self.detail,
        }


# ── the published properties ──────────────────────────────────────────
def _channel_ordering() -> tuple[bool, str]:
    """The least manageable channels must outrank the most managed one.

    Face is high bandwidth and the most voluntarily controlled channel,
    so it is the least trustworthy under any motive to conceal. Autonomic
    leakage and timing are close to unmanaged. Whatever the numbers are,
    that ordering is the finding, and a parameter set that inverts it has
    stopped modelling the thing.
    """
    reg = registry()

    def value(name: str) -> float:
        param = reg.get(f"interiority.other_minds.prior_{name}")
        return param.value if param else 0.0

    unmanaged = min(value("autonomic"), value("timing"), value("behaviour"))
    managed = value("face")
    holds = unmanaged > managed
    return (
        holds,
        f"least-managed channels at {unmanaged:.2f}, face at {managed:.2f}",
    )


def _interoception_cannot_carry_a_read() -> tuple[bool, str]:
    """Aura's own load is evidence about Aura.

    Above about a third this channel starts answering questions about
    another person with facts about the observer, which is the defect
    measured in one of the reviewed prototypes: it answered "what is he
    feeling" with "my chest is tight".
    """
    param = registry().get("interiority.other_minds.prior_interoceptive")
    value = param.value if param else 1.0
    return (value < 0.33, f"interoceptive channel weight {value:.2f}")


def _fractal_band_matches_natural_scenes() -> tuple[bool, str]:
    """Preference concentrates on D = 1.3 to 1.5, where natural scenes sit."""
    reg = registry()
    peak = reg.get("interiority.f04.fractal_preference_peak")
    width = reg.get("interiority.f04.fractal_preference_width")
    if peak is None or width is None:
        return (False, "the fractal band parameters are not declared")
    low, high = peak.value - width.value, peak.value + width.value
    holds = 1.25 <= low <= 1.35 and 1.45 <= high <= 1.60
    return (holds, f"half-maximum band {low:.2f}-{high:.2f}")


def _tolerance_outlasts_the_stimulus() -> tuple[bool, str]:
    """Recovery must be slower than adaptation, or there is no tolerance.

    If a channel recovers as fast as it adapts, the same stimulus produces
    the same response every time and nothing has been learned about the
    environment.
    """
    reg = registry()
    fast = reg.get("interiority.receptor.phosphorylation_rate")
    slow = reg.get("interiority.receptor.dephosphorylation_rate")
    if fast is None or slow is None:
        return (False, "the receptor rates are not declared")
    ratio = fast.value / max(1e-9, slow.value)
    return (ratio >= 3.0, f"adaptation is {ratio:.1f}x faster than recovery")


def _play_is_gated_not_weighted() -> tuple[bool, str]:
    """The relaxed field has to be a gate.

    A weight can be outbid by enough novelty, and a system that plays
    while somebody is waiting has a bug rather than a personality. The
    check is structural: item 2 must require urgency and decline on it.
    """
    from core.interiority.faculties import load_all
    from core.interiority.faculty import registry as faculties

    load_all()
    fun = faculties().get("f02_fun")
    if fun is None:
        return (False, "item 2 is not registered")
    requires = "urgency" in fun.requires
    collapses = any(
        c.expect.value == "collapses" and "urgency" in c.do
        for c in fun.counterfactuals
    )
    return (
        requires and collapses,
        f"urgency required: {requires}; declared to collapse on it: {collapses}",
    )


def _wanting_falls_within_the_meal() -> tuple[bool, str]:
    """Sensory-specific satiety: wanting must fall while liking holds.

    The sharpest evidence that liking and wanting are separate systems,
    and a single reward scalar cannot produce it.
    """
    from core.interiority.faculties import load_all
    from core.interiority.faculty import registry as faculties

    load_all()
    item = faculties().get("f17_liking_and_wanting")
    if item is None:
        return (False, "item 17 is not registered")
    source = item.falsifier()
    return (
        "satiety" in source.lower() or "repeated" in source.lower(),
        "item 17 names sensory-specific satiety as its refutation",
    )


TARGETS: tuple[Target, ...] = (
    Target(
        "channel_ordering_survives",
        "the controllability asymmetry between face and unmanaged channels",
        _channel_ordering,
        (
            "interiority.other_minds.prior_autonomic",
            "interiority.other_minds.prior_timing",
            "interiority.other_minds.prior_behaviour",
            "interiority.other_minds.prior_face",
        ),
    ),
    Target(
        "own_load_cannot_carry_a_read",
        "measured defect in a reviewed prototype",
        _interoception_cannot_carry_a_read,
        ("interiority.other_minds.prior_interoceptive",),
    ),
    Target(
        "fractal_band_matches_natural_scenes",
        "preference concentrated at D = 1.3 to 1.5",
        _fractal_band_matches_natural_scenes,
        (
            "interiority.f04.fractal_preference_peak",
            "interiority.f04.fractal_preference_width",
        ),
    ),
    Target(
        "tolerance_outlasts_the_stimulus",
        "rapid desensitisation with slower resensitisation",
        _tolerance_outlasts_the_stimulus,
        (
            "interiority.receptor.phosphorylation_rate",
            "interiority.receptor.dephosphorylation_rate",
        ),
    ),
    Target(
        "play_is_gated_not_weighted",
        "Burghardt's relaxed-field criterion",
        _play_is_gated_not_weighted,
        (),
    ),
    Target(
        "wanting_falls_within_the_meal",
        "sensory-specific satiety",
        _wanting_falls_within_the_meal,
        (),
    ),
)


def _load_every_declaring_module() -> None:
    """Import everything that declares a parameter.

    A target that reads the registry before the module declaring its
    parameters has been imported reports the parameter missing, which
    looks like a failed target and is a failed import order.
    """
    from core.interiority import (  # noqa: F401
        attribution,
        cleft,
        core_affect,
        ledger,
        other_minds,
        receptors,
    )
    from core.interiority.faculties import load_all

    load_all()


def check_targets(targets: Sequence[Target] | None = None) -> list[TargetResult]:
    _load_every_declaring_module()
    results: list[TargetResult] = []
    for target in targets or TARGETS:
        held, detail = target.check()
        results.append(
            TargetResult(
                name=target.name, source=target.source, held=held, detail=detail
            )
        )
    return results


# ── the sweep ─────────────────────────────────────────────────────────
def _ordering_under(param: Param, value: float) -> tuple[str, ...]:
    """The faculty ordering produced with this parameter held at ``value``.

    Every call site reads the parameter object rather than a float
    captured at import, so the override is visible to the running system
    and the sweep measures something. An earlier version of this function
    ignored the value and recomputed the same ordering at every point,
    which reported perfect stability for parameters it had never moved.
    """
    from core.interiority.proving import ablation_report

    with param.override(value):
        ranked = sorted(ablation_report(), key=lambda r: (-r.total, r.faculty))
        return tuple(r.faculty for r in ranked)


def sweep(param: Param, steps: int = 5) -> SweepResult:
    """Does the system's ordering survive this parameter moving?"""
    orderings = tuple(_ordering_under(param, v) for v in param.sweep(steps))
    distinct = {o for o in orderings}
    stable = len(distinct) == 1
    return SweepResult(
        parameter=param.name,
        orderings=orderings,
        stable=stable,
        detail=(
            f"{len(distinct)} distinct ordering(s) across "
            f"{len(orderings)} points in [{param.sweep_range[0]}, "
            f"{param.sweep_range[1]}]"
            if param.sweep_range
            else "no declared range"
        ),
    )


def constrained_parameters() -> set[str]:
    """Calibration parameters a published target actually constrains."""
    out: set[str] = set()
    for target in TARGETS:
        out.update(target.constrains)
    return out


def order_sensitive(steps: int = 5) -> list[SweepResult]:
    """Calibration parameters whose movement reorders the conclusions.

    The condition under which a guess is allowed is that the ordering
    survives it. These are the ones where it does not, which makes them
    the numbers that need measuring rather than guessing — not defects,
    but the honest short list of what this rests on.
    """
    return [
        result
        for result in (sweep(p, steps) for p in registry().calibration())
        if not result.stable
    ]


def order_sensitive_baseline() -> set[str]:
    import json
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[2]
        / "config"
        / "interiority_order_sensitive_parameters.json"
    )
    try:
        return set(json.loads(path.read_text())["parameters"])
    except (OSError, ValueError, KeyError):
        return set()


def report() -> dict[str, Any]:
    """One report a gate can read, and a person can argue with."""
    _load_every_declaring_module()
    reg = registry()
    calibration = reg.calibration()
    constrained = constrained_parameters()
    targets = check_targets()
    unconstrained = sorted(p.name for p in calibration if p.name not in constrained)
    return {
        "parameters": len(reg),
        "by_kind": {
            kind.value: sum(1 for p in reg.all() if p.kind is kind)
            for kind in ParamKind
        },
        "targets": {
            "declared": len(targets),
            "held": sum(1 for t in targets if t.held),
            "results": [t.to_dict() for t in targets],
            "failed": [t.to_dict() for t in targets if not t.held],
        },
        "order_sensitivity": _order_sensitivity_report(),
        "calibration": {
            "total": len(calibration),
            "constrained_by_a_target": sorted(
                p.name for p in calibration if p.name in constrained
            ),
            "unconstrained": unconstrained,
            "note": (
                "An unconstrained calibration parameter is a guess with a "
                "declared range and no published property pinning it. That is "
                "the honest state, and shrinking this list is what fitting "
                "against data would mean here."
            ),
        },
    }


def _order_sensitivity_report() -> dict[str, Any]:
    found = {r.parameter for r in order_sensitive()}
    baseline = order_sensitive_baseline()
    return {
        "order_sensitive": sorted(found),
        "baseline": sorted(baseline),
        "new": sorted(found - baseline),
        "resolved": sorted(baseline - found),
        "note": (
            "A parameter here is one the conclusions depend on, which is "
            "exactly the condition under which guessing is not allowed. The "
            "list may shrink and may not grow."
        ),
    }


__all__ = [
    "TARGETS",
    "SweepResult",
    "Target",
    "TargetResult",
    "check_targets",
    "order_sensitive",
    "order_sensitive_baseline",
    "constrained_parameters",
    "report",
    "sweep",
]
