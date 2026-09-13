"""Grading and provenance for the latent-cortex experiments.

Everything that turns arm counts into a tier, and the record that says
which code produced the tier. Split out of ``experiments.py`` when that
module crossed the 2,000-line ceiling.

The implementation digest covers EVERY module that participates in a
verdict, not just the file it happens to live in — a digest that stops
covering the grading code the moment the grading code moves is worse than
no digest, because it still reads as provenance.
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Final

from core.brain.llm.latent_cortex.resource_accounting import (
    certify_comparison_accounting,
)
from core.brain.verifiers.foundry import wilson_lower_bound, wilson_upper_bound

logger = logging.getLogger("Aura.LatentCortex.Experiments")

__all__ = [
    "CONJECTURE",
    "PROVEN",
    "REFUTED",
    "SUPPORTED",
    "ArmResult",
    "Claim",
    "ExperimentProvenance",
    "PairedObservation",
    "experiments_implementation_sha256",
    "grade_paired_treatment_vs_control",
    "grade_treatment_vs_control",
]

PROVEN = "PROVEN"
SUPPORTED = "SUPPORTED"
CONJECTURE = "CONJECTURE"
REFUTED = "REFUTED"

_MIN_N_FOR_VERDICT = 20  # below this, everything is CONJECTURE
_BOOTSTRAP_RESAMPLES = 10_000

#: Every source file that can change a verdict. Ordered, so the digest is
#: stable across filesystems.
_GRADING_SOURCES = ("experiment_tasks.py", "experiment_grading.py", "experiments.py")

_MODULE_DIGEST = ""

# ── Claims ──────────────────────────────────────────────────────────────


@dataclass
class ArmResult:
    """One experimental arm: n trials, k successes, plus cost accounting."""

    name: str
    n: int = 0
    successes: int = 0
    layer_apps: int = 0

    @property
    def accuracy(self) -> float:
        return self.successes / self.n if self.n else 0.0

    @property
    def lb(self) -> float:
        return wilson_lower_bound(self.successes, self.n)

    @property
    def ub(self) -> float:
        return wilson_upper_bound(self.successes, self.n)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n": self.n,
            "successes": self.successes,
            "accuracy": round(self.accuracy, 4),
            "wilson_lb": round(self.lb, 4),
            "wilson_ub": round(self.ub, 4),
            "layer_apps": self.layer_apps,
        }


@dataclass(frozen=True)
class PairedObservation:
    """One task evaluated by treatment and control under measured compute."""

    task_id: str
    family: str
    treatment_success: bool
    control_success: bool
    treatment_layer_apps: int | None = None
    control_layer_apps: int | None = None
    treatment_resource: dict[str, Any] | None = None
    control_resource: dict[str, Any] | None = None
    treatment_information: dict[str, Any] | None = None
    control_information: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "family": self.family,
            "treatment_success": self.treatment_success,
            "control_success": self.control_success,
            "treatment_layer_apps": self.treatment_layer_apps,
            "control_layer_apps": self.control_layer_apps,
            "treatment_resource": self.treatment_resource,
            "control_resource": self.control_resource,
            "treatment_information": self.treatment_information,
            "control_information": self.control_information,
        }


def _coerce_solver_outcome(value: Any) -> tuple[bool, int | None]:
    if isinstance(value, tuple) and len(value) == 2:
        success, layer_apps = value
        if not isinstance(success, bool):
            raise ValueError("solver success must be boolean")
        if type(layer_apps) is not int or layer_apps < 0:
            raise ValueError("solver layer-app receipt must be a non-negative integer")
        return success, layer_apps
    if isinstance(value, bool):
        return value, None
    raise ValueError("solver must return bool or (bool, non-negative layer_apps)")


def _coerce_accounted_solver_outcome(
    value: Any,
) -> tuple[bool, int, dict[str, Any], dict[str, Any]]:
    if not isinstance(value, tuple) or len(value) != 4:
        raise ValueError(
            "claim-grade solver must return "
            "(bool, layer_apps, resource_receipt, information_receipt)"
        )
    success, layer_apps, resource, information = value
    if not isinstance(success, bool):
        raise ValueError("solver success must be boolean")
    if type(layer_apps) is not int or layer_apps <= 0:
        raise ValueError("solver layer-app receipt must be a positive integer")
    if not isinstance(resource, dict) or not isinstance(information, dict):
        raise ValueError("solver accounting receipts must be mappings")
    return success, layer_apps, resource, information


def _coerce_role_outcome(value: Any) -> tuple[bool, int, float | None]:
    """Strict contract for role runners: (success, layer_apps, divergence).

    The two-field contract above does not cover divergence, so this extends
    it rather than letting a third field arrive unchecked. Divergence is a
    finite non-negative real when measured and ``None`` when exchange
    telemetry is structurally absent. Non-finite numbers are never absence
    sentinels because they poison downstream arithmetic and JSON evidence.
    """
    if not isinstance(value, tuple) or len(value) != 3:
        raise ValueError(
            "role solver must return (bool, non-negative layer_apps, divergence)"
        )
    success, layer_apps, divergence = value
    if not isinstance(success, bool):
        raise ValueError("role solver success must be boolean")
    if type(layer_apps) is not int or layer_apps < 0:
        raise ValueError("role solver layer-app receipt must be a non-negative integer")
    if divergence is None:
        return success, layer_apps, None
    if isinstance(divergence, bool) or not isinstance(divergence, (int, float)):
        raise ValueError("role solver divergence must be a real number or None")
    divergence_value = float(divergence)
    # NaN must be refused, not merely infinities. `isinf(nan)` is False and
    # `isfinite(nan)` is False, so the earlier form let NaN straight through
    # — and NaN is the most damaging value here: it propagates silently
    # through every downstream mean, and every comparison against it is
    # False, so a poisoned divergence looks like a small one forever. If a
    # runner cannot measure divergence it must say so structurally, not by
    # emitting a float that quietly disables the statistics.
    if not math.isfinite(divergence_value) or divergence_value < 0.0:
        raise ValueError("role solver divergence must be finite and non-negative")
    return success, layer_apps, divergence_value


def _exact_paired_pvalue_greater(wins: int, losses: int) -> float:
    """Exact one-sided McNemar/binomial p for treatment wins > losses."""
    discordant = wins + losses
    if discordant <= 0:
        return 1.0
    numerator = sum(math.comb(discordant, k) for k in range(wins, discordant + 1))
    return min(1.0, numerator / (2**discordant))


def _paired_bootstrap_interval(
    differences: list[int], *, alpha: float, seed: int = 20260715
) -> tuple[float, float]:
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not math.isfinite(float(alpha))
        or not 0.0 < alpha <= 0.5
    ):
        raise ValueError("bootstrap alpha must be inside (0, 0.5]")
    if not differences:
        return 0.0, 0.0
    if len(set(differences)) == 1:
        value = float(differences[0])
        return value, value
    import numpy as np

    values = np.asarray(differences, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty((_BOOTSTRAP_RESAMPLES,), dtype=np.float64)
    for start in range(0, _BOOTSTRAP_RESAMPLES, 500):
        count = min(500, _BOOTSTRAP_RESAMPLES - start)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        means[start : start + count] = values[indices].mean(axis=1)
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return float(low), float(high)


def _holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for index, (name, pvalue) in enumerate(ordered):
        running = max(running, min(1.0, (count - index) * pvalue))
        adjusted[name] = running
    return adjusted


def grade_paired_treatment_vs_control(
    experiment: str,
    statement: str,
    observations_by_family: dict[str, list[PairedObservation]],
    *,
    alpha: float = 0.05,
    minimum_effect: float = 0.0,
    compute_tolerance: float = 0.05,
    require_compute: bool = True,
    require_resource_accounting: bool = False,
    provenance: ExperimentProvenance | None = None,
) -> Claim:
    """Paired, multiplicity-corrected capability comparison."""
    for name, value in (
        ("alpha", alpha),
        ("minimum_effect", minimum_effect),
        ("compute_tolerance", compute_tolerance),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(
            float(value)
        ):
            raise ValueError(f"{name} must be a finite number")
    if not 0.0 < alpha <= 0.5:
        raise ValueError("alpha must be inside (0, 0.5]")
    if not 0.0 <= minimum_effect < 1.0:
        raise ValueError("minimum_effect must be inside [0, 1)")
    if not 0.0 <= compute_tolerance <= 1.0:
        raise ValueError("compute_tolerance must be inside [0, 1]")
    if type(require_compute) is not bool:
        raise ValueError("require_compute must be boolean")
    if type(require_resource_accounting) is not bool:
        raise ValueError("require_resource_accounting must be boolean")

    family_stats: dict[str, dict[str, Any]] = {}
    raw_pvalues: dict[str, float] = {}
    all_differences: list[int] = []
    invalid_compute: list[str] = []
    invalid_resource_accounting: list[str] = []
    accounting_certificates: dict[str, list[dict[str, Any]]] = {}
    underpowered: list[str] = []
    seen_task_ids: set[str] = set()
    family_bound_alpha = alpha / max(1, len(observations_by_family))
    for family, observations in observations_by_family.items():
        if not isinstance(family, str) or not family.strip():
            raise ValueError("paired evidence family names must be non-empty strings")
        if not isinstance(observations, list):
            raise ValueError(f"paired evidence for {family} must be a list")
        for observation in observations:
            if not isinstance(observation, PairedObservation):
                raise ValueError(f"paired evidence for {family} contains an invalid row")
            if not observation.task_id or observation.task_id in seen_task_ids:
                raise ValueError("paired evidence task ids must be non-empty and unique")
            seen_task_ids.add(observation.task_id)
            if observation.family != family:
                raise ValueError(
                    f"paired evidence family mismatch: {observation.family!r} != {family!r}"
                )
            if type(observation.treatment_success) is not bool or type(
                observation.control_success
            ) is not bool:
                raise ValueError("paired evidence outcomes must be boolean")
            for cost in (
                observation.treatment_layer_apps,
                observation.control_layer_apps,
            ):
                if cost is not None and (type(cost) is not int or cost < 0):
                    raise ValueError(
                        "paired evidence compute must be non-negative integers or null"
                    )
        differences = [
            int(obs.treatment_success) - int(obs.control_success) for obs in observations
        ]
        wins = differences.count(1)
        losses = differences.count(-1)
        missing_compute = any(
            obs.treatment_layer_apps is None or obs.control_layer_apps is None
            for obs in observations
        )
        nonpositive_compute = any(
            obs.treatment_layer_apps is not None
            and obs.control_layer_apps is not None
            and (obs.treatment_layer_apps <= 0 or obs.control_layer_apps <= 0)
            for obs in observations
        )
        mismatched = [
            obs.task_id
            for obs in observations
            if obs.treatment_layer_apps is not None
            and obs.control_layer_apps is not None
            and (
                abs(obs.treatment_layer_apps - obs.control_layer_apps)
                / max(1, obs.control_layer_apps)
            )
            > compute_tolerance
        ]
        family_certificates: list[dict[str, Any]] = []
        if require_resource_accounting:
            tolerance = Fraction(str(compute_tolerance)).limit_denominator(10_000)
            for obs in observations:
                receipts = (
                    obs.treatment_resource,
                    obs.control_resource,
                    obs.treatment_information,
                    obs.control_information,
                )
                if any(receipt is None for receipt in receipts):
                    invalid_resource_accounting.append(family)
                    continue
                certificate = certify_comparison_accounting(
                    treatment_resource=obs.treatment_resource,
                    control_resource=obs.control_resource,
                    treatment_information=obs.treatment_information,
                    control_information=obs.control_information,
                    tolerance_numerator=tolerance.numerator,
                    tolerance_denominator=tolerance.denominator,
                    require_compute_parity=require_compute,
                )
                family_certificates.append(certificate)
                if not certificate["admitted"]:
                    invalid_resource_accounting.append(family)
            accounting_certificates[family] = family_certificates
        if require_compute and (
            missing_compute
            or nonpositive_compute
            or mismatched
            or family in invalid_resource_accounting
        ):
            invalid_compute.append(family)
        effect = sum(differences) / len(differences) if differences else 0.0
        ci_low, ci_high = _paired_bootstrap_interval(
            differences,
            alpha=family_bound_alpha,
        )
        pvalue = _exact_paired_pvalue_greater(wins, losses)
        if len(observations) < _MIN_N_FOR_VERDICT:
            underpowered.append(family)
        else:
            raw_pvalues[family] = pvalue
        family_stats[family] = {
            "n": len(observations),
            "treatment_wins": wins,
            "control_wins": losses,
            "ties": len(observations) - wins - losses,
            "paired_effect": round(effect, 6),
            "effect_interval": [round(ci_low, 6), round(ci_high, 6)],
            "effect_bound_alpha": family_bound_alpha,
            "one_sided_exact_p": pvalue,
            "missing_compute": missing_compute,
            "nonpositive_compute": nonpositive_compute,
            "compute_mismatch_task_ids": mismatched,
            "resource_accounting_invalid": family in invalid_resource_accounting,
        }
        all_differences.extend(differences)

    adjusted = _holm_adjust(raw_pvalues)
    positive_families = [
        family
        for family, stats in family_stats.items()
        if family in adjusted
        and adjusted[family] < alpha
        and stats["effect_interval"][0] > minimum_effect
        and family not in invalid_compute
        and family not in invalid_resource_accounting
    ]
    regressed_families = [
        family
        for family, stats in family_stats.items()
        if stats["effect_interval"][1] < -minimum_effect
    ]
    pooled_wins = all_differences.count(1)
    pooled_losses = all_differences.count(-1)
    pooled_effect = (
        sum(all_differences) / len(all_differences) if all_differences else 0.0
    )
    pooled_low, pooled_high = _paired_bootstrap_interval(
        all_differences,
        alpha=alpha,
        seed=20260716,
    )
    pooled_p = _exact_paired_pvalue_greater(pooled_wins, pooled_losses)
    evidence = {
        "method": (
            "paired exact McNemar/binomial + Holm correction + "
            "alpha-derived one-sided percentile bounds"
        ),
        "alpha": alpha,
        "minimum_effect": minimum_effect,
        "compute_tolerance": compute_tolerance,
        # Whether compute parity was actually VALIDATED for this claim.
        # Several callers legitimately disable it (arms that intentionally
        # spend different compute), but a claim graded without compute
        # matching must not read as clean causal attribution — the observed
        # difference may be bought with extra compute rather than by the
        # named mechanism.
        "compute_matched": bool(require_compute),
        "resource_accounting_required": require_resource_accounting,
        "resource_accounting_matched": bool(
            require_resource_accounting and not invalid_resource_accounting
        ),
        "accounting_certificates": accounting_certificates,
        "bootstrap_resamples": _BOOTSTRAP_RESAMPLES,
        "families": family_stats,
        "holm_adjusted_p": adjusted,
        "positive_families": positive_families,
        "regressed_families": regressed_families,
        "underpowered_families": underpowered,
        "invalid_compute_families": invalid_compute,
        "invalid_resource_accounting_families": sorted(
            set(invalid_resource_accounting)
        ),
        "pooled": {
            "n": len(all_differences),
            "treatment_wins": pooled_wins,
            "control_wins": pooled_losses,
            "paired_effect": round(pooled_effect, 6),
            "effect_interval": [round(pooled_low, 6), round(pooled_high, 6)],
            "effect_bound_alpha": alpha,
            "one_sided_exact_p": pooled_p,
        },
    }
    pooled_positive = (
        len(all_differences) >= _MIN_N_FOR_VERDICT
        and pooled_p < alpha
        and pooled_low > minimum_effect
    )
    required_positive = max(2, math.ceil(len(family_stats) * 2 / 3))
    evidence["required_positive_families"] = required_positive
    if (
        invalid_compute
        or underpowered
        or (require_resource_accounting and invalid_resource_accounting)
    ):
        tier = CONJECTURE
    elif (
        len(positive_families) >= required_positive
        and pooled_positive
        and not regressed_families
    ):
        tier = PROVEN
    elif positive_families and pooled_positive and not regressed_families:
        tier = SUPPORTED
    elif regressed_families or (all_differences and pooled_high <= 0.0):
        tier = REFUTED
    else:
        tier = CONJECTURE
    provenance_payload, provenance_gaps = _provenance_payload(provenance)
    tier = _tier_under_provenance(tier, provenance_gaps, evidence)
    return Claim(
        experiment=experiment,
        statement=statement,
        tier=tier,
        evidence=evidence,
        provenance=provenance_payload,
    )


#: Everything a third party needs to re-run an experiment and get the same
#: verdict. A Claim used to carry a name, a statement, a tier, evidence and a
#: wall-clock time — enough to READ the result, nothing like enough to
#: reproduce it. The runners take opaque callbacks, so most of this has to
#: come from the caller; what the module can measure about itself, it does.
_PROVENANCE_FIELDS: Final = (
    "task_manifest_sha256",
    "checkpoint_fingerprint",
    "schedule_sha256",
    "verifier_version",
    "environment_sha256",
)
_MODULE_DIGEST: str = ""


def experiments_implementation_sha256() -> str:
    """Digest of every source that can change a verdict.

    The one piece of provenance nobody has to be trusted for: a claim graded
    by different code is a different claim, and the grader can measure that
    itself. It covers the whole grading set rather than one file, because a
    digest that stops covering the grading code the moment that code moves
    to a sibling module is worse than no digest — it still reads as
    provenance.
    """
    global _MODULE_DIGEST
    if not _MODULE_DIGEST:
        here = Path(__file__).parent
        digest = hashlib.sha256()
        for name in _GRADING_SOURCES:
            try:
                digest.update(name.encode("utf-8"))
                digest.update(b"\x1e")
                digest.update((here / name).read_bytes())
            except OSError:
                return ""
        _MODULE_DIGEST = digest.hexdigest()
    return _MODULE_DIGEST


@dataclass(frozen=True, slots=True)
class ExperimentProvenance:
    """What the caller pins so the verdict can be reproduced.

    Every field is required. An experiment that cannot name the tasks it ran,
    the checkpoint it ran against, the schedule, the verifier, or the
    environment has produced a number, not a result — and a number nobody can
    re-derive must not be published above CONJECTURE.
    """

    task_manifest_sha256: str
    checkpoint_fingerprint: str
    schedule_sha256: str
    verifier_version: str
    environment_sha256: str

    def gaps(self) -> tuple[str, ...]:
        return tuple(
            name
            for name in _PROVENANCE_FIELDS
            if not str(getattr(self, name) or "").strip()
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {name: getattr(self, name) for name in _PROVENANCE_FIELDS}
        payload["implementation_sha256"] = experiments_implementation_sha256()
        return payload


def _provenance_payload(
    provenance: ExperimentProvenance | None,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Return the recorded provenance and the fields still missing."""
    if provenance is None:
        return (
            {"implementation_sha256": experiments_implementation_sha256()},
            _PROVENANCE_FIELDS,
        )
    return provenance.to_dict(), provenance.gaps()


def _tier_under_provenance(
    tier: str, gaps: tuple[str, ...], evidence: dict[str, Any]
) -> str:
    """Cap a verdict that nobody else could reproduce.

    REFUTED survives: a failure that cannot be reproduced is still a failure
    observed, and downgrading it would turn missing paperwork into good news.
    """
    evidence["provenance_gaps"] = list(gaps)
    evidence["reproducible"] = not gaps
    if not gaps or tier in {CONJECTURE, REFUTED}:
        return tier
    return CONJECTURE


@dataclass
class Claim:
    experiment: str
    statement: str
    tier: str
    evidence: dict[str, Any] = field(default_factory=dict)
    graded_at: float = field(default_factory=lambda: time.time())
    #: What the verdict can be reproduced from. See ExperimentProvenance.
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment": self.experiment,
            "statement": self.statement,
            "tier": self.tier,
            "evidence": self.evidence,
            "graded_at": self.graded_at,
            "provenance": self.provenance,
        }


def grade_treatment_vs_control(
    experiment: str,
    statement: str,
    treatment_by_family: dict[str, ArmResult],
    control_by_family: dict[str, ArmResult],
    *,
    provenance: ExperimentProvenance | None = None,
) -> Claim:
    """The conservative comparison grader shared by Experiments 1, 4, 5."""
    wins, losses, small = [], [], []
    # Iterate the UNION of families. Walking only the treatment side let a
    # family that exists in the control but was dropped from the treatment
    # vanish silently — selective omission that can only improve the claim.
    for family in sorted(set(treatment_by_family) | set(control_by_family)):
        treat = treatment_by_family.get(family)
        control = control_by_family.get(family)
        if (
            treat is None
            or control is None
            or treat.n < _MIN_N_FOR_VERDICT
            or control.n < _MIN_N_FOR_VERDICT
        ):
            small.append(family)
            continue
        if treat.lb > control.ub:
            wins.append(family)
        elif treat.accuracy <= control.accuracy:
            losses.append(family)
    # A family measured for the control but missing from the treatment is
    # named explicitly so its absence cannot read as absence of evidence.
    missing_treatment = sorted(set(control_by_family) - set(treatment_by_family))
    evidence = {
        "treatment": {f: a.to_dict() for f, a in treatment_by_family.items()},
        "control": {f: a.to_dict() for f, a in control_by_family.items()},
        "separated_families": wins,
        "not_better_families": losses,
        "underpowered_families": small,
        "families_missing_from_treatment": missing_treatment,
    }
    evidence["aggregate_only"] = True
    evidence["limitation"] = (
        "aggregate Wilson intervals lack paired task outcomes and cannot earn PROVEN"
    )
    if missing_treatment:
        # Selective omission cannot be rewarded: an incomplete treatment arm
        # is undecided evidence, whatever the reported families show.
        tier = CONJECTURE
        evidence["limitation"] = (
            "families measured for the control are missing from the treatment; "
            "the comparison is incomplete"
        )
    elif wins:
        tier = SUPPORTED
    elif small and not losses:
        tier = CONJECTURE
    elif losses and not wins:
        tier = REFUTED
    else:
        tier = CONJECTURE
    provenance_payload, provenance_gaps = _provenance_payload(provenance)
    tier = _tier_under_provenance(tier, provenance_gaps, evidence)
    return Claim(
        experiment=experiment,
        statement=statement,
        tier=tier,
        evidence=evidence,
        provenance=provenance_payload,
    )


