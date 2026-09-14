"""Fix what a campaign is measuring BEFORE it sees the data.

Two results in this repo were reported as established and are not, for the
same reason: a decision that should have preceded the data was made after it.

* The Grassmann encoder width. Three widths were run (8/12/16), 12 scored best,
  and 12 became ``GRASSMANN_ANCHORS_DEFAULT`` — described in the source as
  "the measured optimum, not the smallest number that works". With three arms
  and one campaign, the best of three IS the expected shape of noise. That is
  not a claim that 12 is wrong; it is a statement that this campaign cannot
  distinguish "12 resolves real structure" from "12 won a three-way draw".

* The CAA steering A/B's pass criterion, which was whatever the one significant
  comparison happened to be.

Neither was dishonest. Both are what happens when the analysis plan is written
in the same session as the analysis.

What this module provides
-------------------------
A ``Preregistration`` is a frozen declaration — parameters, metrics, decision
thresholds, arms — carrying a content hash. Register it, run, then attach the
same object to the result. Two things then become checkable rather than
remembered:

1. **Whether the reported analysis is the declared one.** ``verify_result``
   cannot confirm missing measurements, parameters, or arms. Additional metrics
   are exploratory without invalidating the declared comparisons.

2. **Which findings were declared.** Anything outside the registration is
   EXPLORATORY. The content hash does not prove chronology: a campaign must also
   retain publication evidence from before its data were exposed.

Exploratory findings are not second-class science — they are how the width
sweep found a real encoder bug. They are second-class EVIDENCE, and the only
thing being asked here is that an artifact say which kind it holds.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any

from core.runtime.file_write_gateway import get_file_write_gateway
from core.verify.invariants import invariant

__all__ = [
    "EvidenceStatus",
    "Finding",
    "Preregistration",
    "canonical_hash",
    "load_preregistration",
]


class EvidenceStatus(StrEnum):
    """What a number is allowed to be cited as."""

    #: Declared before the run, measured as declared, threshold met.
    CONFIRMATORY = "confirmatory"
    #: Declared before the run, measured as declared, threshold NOT met.
    #: A real result — the one an honest campaign produces most often.
    NEGATIVE = "negative"
    #: Selected, tuned, or defined after seeing the data. Worth reporting,
    #: worth replicating, not citable as established.
    EXPLORATORY = "exploratory"
    #: Declared but not actually measured. Never a pass.
    UNMEASURED = "unmeasured"


def canonical_hash(payload: Any) -> str:
    """Stable content hash. Key order and float repr cannot change it."""
    encoded = json.dumps(
        _json_value(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_value(value: Any) -> Any:
    """Copy the declared JSON domain without repr-based identity or nonfinite values."""
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("preregistration keys must be strings")
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise ValueError("preregistration values must be finite JSON data")


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _finite_metric(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("a boolean is not a measured metric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("metric must be finite")
    return number


@dataclass(frozen=True)
class Finding:
    """One declared metric and what the run did to it."""

    metric: str
    value: float | None
    threshold: float | None
    status: EvidenceStatus
    note: str = ""

    @property
    def meets_threshold(self) -> bool:
        if self.value is None or self.threshold is None:
            return False
        try:
            return _finite_metric(self.value) >= _finite_metric(self.threshold)
        except (TypeError, ValueError, OverflowError):
            return False

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "status": str(self.status)}


@dataclass(frozen=True)
class Preregistration:
    """A campaign's analysis plan, fixed before the campaign runs.

    ``parameters`` are the knobs the run is NOT free to choose (encoder width,
    alpha, trial count). ``metrics`` maps each declared metric to the threshold
    it must clear. ``arms`` names the conditions, so an arm that quietly
    appears or vanishes is visible.
    """

    campaign: str
    hypothesis: str
    parameters: Mapping[str, Any]
    metrics: Mapping[str, float]
    arms: Sequence[str] = field(default_factory=tuple)
    registered_at: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", _freeze(_json_value(self.parameters)))
        object.__setattr__(self, "metrics", MappingProxyType(
            {k: _finite_metric(v) for k, v in self.metrics.items()}))
        object.__setattr__(self, "arms", tuple(str(a) for a in self.arms))
        if len(self.arms) != len(set(self.arms)) or any(not arm for arm in self.arms):
            raise ValueError("preregistered arms must be unique and nonempty")
        if not self.registered_at:
            object.__setattr__(
                self, "registered_at", datetime.now(UTC).isoformat()
            )

    @property
    def plan_hash(self) -> str:
        """Identity of the PLAN — deliberately excludes the timestamp.

        Two people registering the same plan get the same hash, and a plan
        edited after the fact gets a different one.
        """
        return canonical_hash(
            {
                "campaign": self.campaign,
                "hypothesis": self.hypothesis,
                "parameters": _json_value(self.parameters),
                "metrics": dict(self.metrics),
                "arms": list(self.arms),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "campaign": self.campaign,
            "hypothesis": self.hypothesis,
            "parameters": _json_value(self.parameters),
            "metrics": dict(self.metrics),
            "arms": list(self.arms),
            "registered_at": self.registered_at,
            "notes": self.notes,
            "plan_hash": self.plan_hash,
        }

    def write(self, storage_root: str | Path | None = None) -> Path:
        """Publish this immutable plan under its content hash.

        Production uses Aura's fixed evaluation store. Tests and offline tools
        may inject a root, but never a filename: the plan hash owns identity and
        create-once publication prevents after-the-data replacement.
        """

        if storage_root is None:
            from core.config import config

            root = Path(config.paths.data_dir) / "evaluation" / "preregistrations"
        else:
            root = Path(storage_root)
        target = root / f"{self.plan_hash}.json"
        payload = (json.dumps(self.to_dict(), indent=2) + "\n").encode("utf-8")
        created = get_file_write_gateway().write_bytes_if_absent(
            target,
            payload,
            source="evaluation.preregistration",
        )
        if not created and target.read_bytes() != payload:
            raise FileExistsError(
                f"preregistration hash collision or replacement attempt at {target}"
            )
        return target

    # ── reading a run against the plan ────────────────────────────────────

    def classify(
        self,
        observed: Mapping[str, float | None],
        *,
        parameters_used: Mapping[str, Any] | None = None,
        arms_used: Sequence[str] | None = None,
    ) -> tuple[Finding, ...]:
        """Label every metric this run produced.

        A declared metric that was measured and cleared its threshold is
        CONFIRMATORY; measured and short is NEGATIVE; declared and absent is
        UNMEASURED. Anything observed that was never declared is EXPLORATORY,
        and so is EVERY metric of a run whose parameters differ from the
        registered ones — because a run at unregistered settings is a
        different experiment wearing this plan's name.
        """
        drift = self.parameter_drift(parameters_used or {})
        arm_drift = self.arm_drift(arms_used)
        findings: list[Finding] = []
        for metric, threshold in sorted(self.metrics.items()):
            if metric not in observed or observed[metric] is None:
                findings.append(
                    Finding(metric, None, threshold, EvidenceStatus.UNMEASURED)
                )
                continue
            try:
                value = _finite_metric(observed[metric])
            except (TypeError, ValueError, OverflowError):
                findings.append(Finding(metric, None, threshold, EvidenceStatus.UNMEASURED,
                                        note="no finite numerical measurement"))
                continue
            if drift or arm_drift:
                findings.append(
                    Finding(
                        metric,
                        value,
                        threshold,
                        EvidenceStatus.EXPLORATORY,
                        note=(f"parameter drift: {', '.join(sorted(drift))}; "
                              f"arm drift: {', '.join(arm_drift)}"),
                    )
                )
                continue
            findings.append(
                Finding(
                    metric,
                    value,
                    threshold,
                    EvidenceStatus.CONFIRMATORY
                    if value >= float(threshold)
                    else EvidenceStatus.NEGATIVE,
                )
            )
        for metric in sorted(set(observed) - set(self.metrics)):
            value = observed[metric]
            try:
                value = None if value is None else _finite_metric(value)
            except (TypeError, ValueError, OverflowError):
                value = None
            findings.append(
                Finding(
                    metric,
                    value,
                    None,
                    EvidenceStatus.EXPLORATORY,
                    note="not declared before the run",
                )
            )
        return tuple(findings)

    def parameter_drift(self, used: Mapping[str, Any]) -> dict[str, str]:
        """Registered parameters this run did not honour."""
        drift: dict[str, str] = {}
        for key, registered in self.parameters.items():
            if key not in used:
                drift[key] = "registered parameter not reported"
                continue
            try:
                same = canonical_hash(used[key]) == canonical_hash(registered)
            except (TypeError, ValueError, OverflowError):
                same = False
            if not same:
                drift[key] = f"{registered!r}->{used[key]!r}"
        return drift

    def verify_result(
        self,
        observed: Mapping[str, float | None],
        *,
        parameters_used: Mapping[str, Any] | None = None,
        arms_used: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """The whole verdict, as a dict an artifact can carry verbatim."""
        findings = self.classify(observed, parameters_used=parameters_used, arms_used=arms_used)
        confirmatory = [f for f in findings if f.status is EvidenceStatus.CONFIRMATORY]
        return {
            "plan_hash": self.plan_hash,
            "campaign": self.campaign,
            "hypothesis": self.hypothesis,
            "parameter_drift": self.parameter_drift(parameters_used or {}),
            "arm_drift": self.arm_drift(arms_used),
            "findings": [f.to_dict() for f in findings],
            # Every declared metric confirmatory, and at least one declared.
            "confirms_hypothesis": bool(self.metrics)
            and len(confirmatory) == len(self.metrics),
            "exploratory_metrics": [
                f.metric for f in findings if f.status is EvidenceStatus.EXPLORATORY
            ],
            "unmeasured_metrics": [
                f.metric for f in findings if f.status is EvidenceStatus.UNMEASURED
            ],
        }

    def arm_drift(self, used: Sequence[str] | None) -> tuple[str, ...]:
        """Compare measured conditions, without treating declaration as execution."""
        if used is None:
            return ("registered arms not reported",) if self.arms else ()
        if isinstance(used, str) or any(not isinstance(arm, str) for arm in used):
            return ("invalid arm inventory",)
        if len(used) != len(set(used)):
            return ("duplicate arm names",)
        return tuple(
            [f"missing:{arm}" for arm in sorted(set(self.arms) - set(used))]
            + [f"undeclared:{arm}" for arm in sorted(set(used) - set(self.arms))]
        )


@invariant("evaluation.preregistered_evidence", scope="evaluation",
           owner="core/evaluation/preregistration.py", observational=False)
def _preregistered_evidence() -> tuple:
    """Missing configuration and nonfinite observations cannot confirm a plan."""
    plan = Preregistration("probe", "gain", {"seed": 1}, {"gain": 0.1}, ("base", "treatment"))
    complete = {"parameters_used": {"seed": 1}, "arms_used": ("base", "treatment")}
    assert plan.verify_result({"gain": 0.2}, **complete)["confirms_hypothesis"]
    assert not plan.verify_result({"gain": 0.2})["confirms_hypothesis"]
    assert not plan.verify_result({"gain": float("inf")}, **complete)["confirms_hypothesis"]
    assert not plan.verify_result({"gain": 0.2}, parameters_used={"seed": 1})["confirms_hypothesis"]
    return ()


def load_preregistration(path: str | Path) -> Preregistration:
    """Read a plan back, and refuse one whose hash no longer matches itself."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    recorded = str(data.pop("plan_hash", ""))
    if len(recorded) != 64 or any(char not in "0123456789abcdef" for char in recorded):
        raise ValueError("preregistration has no valid recorded plan hash")
    plan = Preregistration(
        campaign=str(data.get("campaign", "")),
        hypothesis=str(data.get("hypothesis", "")),
        parameters=data.get("parameters") or {},
        metrics=data.get("metrics") or {},
        arms=data.get("arms") or (),
        registered_at=str(data.get("registered_at", "")),
        notes=str(data.get("notes", "")),
    )
    if recorded and recorded != plan.plan_hash:
        raise ValueError(
            f"preregistration at {path} was edited after registration: "
            f"recorded {recorded[:12]}, recomputed {plan.plan_hash[:12]}"
        )
    return plan
