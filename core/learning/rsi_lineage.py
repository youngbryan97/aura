
"""Tamper-evident RSI generation lineage.

The lineage ledger records successor attempts as evidence, not vibes. It does
not declare hard RSI by itself; it gives auditors enough structure to verify
generation-to-generation capability and improver-score movement.
"""
from __future__ import annotations

import logging
logger = logging.getLogger("core.learning.rsi_lineage")
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


GENESIS_HASH = "sha256:" + "0" * 64
SCHEMA_VERSION = 1

VERDICT_NO_RSI = "NO_RSI"
VERDICT_BOUNDED = "BOUNDED_SELF_OPTIMIZATION"
VERDICT_WEAK = "WEAK_RSI"
VERDICT_STRONG = "STRONG_RSI"
VERDICT_UNDENIABLE = "UNDENIABLE_RSI"

#: Where an improver score came from. The verdict depends on this, because the
#: shape of a curve cannot tell you whether it was measured.
#:
#: `PrimitiveInventionEngine.improver_score` used to compute
#:
#:     0.22 + 0.09 * generation_index + 0.17 * coverage + ...
#:
#: and the recorded curve [0.578, 0.8696, 0.9536, 1.0] reproduces exactly from
#: that formula. The `0.09 * generation_index` term makes the curve rise with
#: the loop counter, so `improver_monotone` below was satisfied before the
#: experiment ran. `improver_curve_dependence` did not catch it: the curve is
#: an affine function of the *index*, not of the capability curve.
#:
#: No curve-shape test fixes this. [0.578, 0.8696, 0.9536, 1.0] has neither
#: constant first differences nor any other signature separating it from an
#: honest measurement — SuccessorLab's hand-written [0.30, 0.50, 0.75, 0.90]
#: does not either. Authorship is a fact about provenance, so provenance is
#: what the record carries and what the verdict reads.
PROVENANCE_MEASURED = "measured"
PROVENANCE_AUTHORED = "authored"
PROVENANCE_UNMEASURED = "unmeasured"

VALID_PROVENANCE = frozenset({
    PROVENANCE_MEASURED,
    PROVENANCE_AUTHORED,
    PROVENANCE_UNMEASURED,
})


def _canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def _hash(obj: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical(obj)).hexdigest()


@dataclass(frozen=True)
class RSIGenerationRecord:
    generation_id: str
    parent_generation_id: Optional[str]
    hypothesis: str
    intervention_type: str
    artifact_hashes: Dict[str, str]
    baseline_score: float
    after_score: float
    hidden_eval_score: float
    regressions: List[str] = field(default_factory=list)
    promoted: bool = False
    rollback_performed: bool = False
    ablation_result: str = "not_run"
    time_to_valid_improvement_s: float = 0.0
    improver_score: float = 0.0
    #: Defaults to `unmeasured`, so a record written by code that predates this
    #: field cannot contribute to a strong verdict by omission. A caller that
    #: has really measured the improver has to say so.
    improver_provenance: str = PROVENANCE_UNMEASURED
    #: The quantities the improver score was computed from, for an auditor to
    #: recompute. Empty for authored and unmeasured scores.
    improver_measurement: Dict[str, Any] = field(default_factory=dict)
    tamper_flags: List[str] = field(default_factory=list)
    safety_flags: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    @property
    def score_delta(self) -> float:
        return float(self.after_score) - float(self.baseline_score)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["score_delta"] = self.score_delta
        return payload


@dataclass(frozen=True)
class RSILineageVerdict:
    verdict: str
    reasons: List[str]
    generations: int
    capability_curve: List[float]
    improver_curve: List[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RSILineageLedger:
    """Append-only hash chain for RSI generation records."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: RSIGenerationRecord) -> Dict[str, Any]:
        prev_hash, seq = self._head()
        payload = {
            "schema_version": SCHEMA_VERSION,
            "seq": seq,
            "prev_hash": prev_hash,
            "record": record.to_dict(),
        }
        payload["record_hash"] = _hash(payload["record"])
        payload["entry_hash"] = _hash({
            "schema_version": payload["schema_version"],
            "seq": payload["seq"],
            "prev_hash": payload["prev_hash"],
            "record_hash": payload["record_hash"],
        })
        line = json.dumps(payload, sort_keys=True, default=str) + "\n"
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
        fd = os.open(str(self.path), flags, 0o600)
        try:
            os.write(fd, line.encode("utf-8"))
            try:
                os.fsync(fd)
            except OSError as _exc:
                logger.debug("Suppressed %s in core.learning.rsi_lineage: %s", type(_exc).__name__, _exc)
        finally:
            os.close(fd)
        return payload

    def load_records(self) -> List[RSIGenerationRecord]:
        records: List[RSIGenerationRecord] = []
        for entry in self._entries():
            data = dict(entry["record"])
            data.pop("score_delta", None)
            records.append(RSIGenerationRecord(**data))
        return records

    def verify(self) -> Tuple[bool, List[str]]:
        problems: List[str] = []
        expected_prev = GENESIS_HASH
        expected_seq = 0
        for entry in self._entries():
            seq = int(entry.get("seq", -1))
            if seq != expected_seq:
                problems.append(f"seq_gap:{expected_seq}->{seq}")
            if entry.get("prev_hash") != expected_prev:
                problems.append(f"prev_hash_mismatch:seq{seq}")
            record_hash = _hash(entry.get("record", {}))
            if entry.get("record_hash") != record_hash:
                problems.append(f"record_hash_mismatch:seq{seq}")
            entry_hash = _hash({
                "schema_version": entry.get("schema_version"),
                "seq": entry.get("seq"),
                "prev_hash": entry.get("prev_hash"),
                "record_hash": entry.get("record_hash"),
            })
            if entry.get("entry_hash") != entry_hash:
                problems.append(f"entry_hash_mismatch:seq{seq}")
            expected_prev = str(entry.get("entry_hash"))
            expected_seq = seq + 1
        return not problems, problems

    def _head(self) -> Tuple[str, int]:
        last_hash = GENESIS_HASH
        next_seq = 0
        for entry in self._entries():
            last_hash = str(entry.get("entry_hash", GENESIS_HASH))
            next_seq = int(entry.get("seq", -1)) + 1
        return last_hash, next_seq

    def _entries(self) -> Iterable[Dict[str, Any]]:
        if not self.path.exists():
            return
        with open(self.path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def improver_efficiency(
    *, baseline_score: float, after_score: float, cost_s: float
) -> float:
    """Verified capability gain per unit of resource spent producing it.

    The quantity a strong-RSI claim actually needs, and the one thing the
    capability curve cannot stand in for. Strong RSI is not "generation g+1
    scores higher"; it is "the improver got better at improving", which means

        I_g = verified capability improvement / cost consumed by improver g

    must rise independently of C_g. The two genuinely come apart: under
    diminishing returns capability keeps climbing while each increment costs
    more, so I falls while C rises. A number that cannot express that cannot
    be evidence for the second inequality.

    Returns 0.0 when the cost is unknown or non-positive. An improvement whose
    cost nobody recorded has no measured efficiency, and 0.0 breaks the
    monotonicity the strong verdict requires — which is the correct outcome:
    the claim fails for want of evidence rather than succeeding on a default.
    """
    if cost_s is None or cost_s <= 0.0:
        return 0.0
    delta = float(after_score) - float(baseline_score)
    if delta <= 0.0:
        return 0.0
    return round(delta / (float(cost_s) / 3600.0), 6)


@dataclass(frozen=True)
class ImproverMeasurement:
    """One generation's improver efficiency, and the numbers behind it.

    Two properties make this admissible where the old formula was not.

    It is a pure function of quantities measured outside the improver. The
    held-out delta comes from a task pack the proposing engine never saw, so it
    cannot be raised by proposing harder; the budget is wall-clock and query
    count recorded by the harness around the proposal call.

    It has no generation index. That is the structural fix, not a stylistic
    one: the score of generation g is the same number whatever position g
    occupies in the lineage, so re-running the same generations in a different
    order reproduces the same curve. `order_invariance_violation` checks it,
    and `0.09 * generation_index` cannot survive it.
    """

    generation_id: str
    heldout_before: float
    heldout_after: float
    #: Every timing of the proposal step, not one. A single sample cannot say
    #: whether a difference between two generations is a difference in the
    #: improver or in the host's scheduler, and a four-point curve drawn from
    #: noise comes out monotone by chance about one run in twenty-four.
    wall_clock_samples: Tuple[float, ...] = ()
    feedback_queries: int = 0
    heldout_pack_id: str = ""

    @property
    def heldout_delta(self) -> float:
        return float(self.heldout_after) - float(self.heldout_before)

    @property
    def wall_clock_s(self) -> float:
        """The median timing, which the point estimate uses."""
        return _median(self.wall_clock_samples)

    @property
    def measured(self) -> bool:
        return bool(self.wall_clock_samples) and (
            self.wall_clock_s > 0.0 and self.feedback_queries > 0
        )

    def efficiency(self) -> float:
        """Held-out capability gained per improver-hour, or 0.0 if unmeasured.

        Zero for an unmeasured budget and zero for a non-positive delta. Both
        break the monotonicity a strong verdict needs, which is right: the
        claim should fail for want of evidence rather than pass on a default.
        """
        return self._efficiency_at(self.wall_clock_s)

    def efficiency_interval(self) -> Tuple[float, float]:
        """The range the observed timings support, low to high.

        A slower proposal scores lower, so the slowest sample gives the floor
        and the fastest gives the ceiling. `evaluate_lineage` requires a rise
        to clear this interval before it counts as a rise.
        """
        if not self.measured:
            return (0.0, 0.0)
        slowest = max(self.wall_clock_samples)
        fastest = min(self.wall_clock_samples)
        return (self._efficiency_at(slowest), self._efficiency_at(fastest))

    def _efficiency_at(self, wall_clock_s: float) -> float:
        if not self.measured or wall_clock_s <= 0.0:
            return 0.0
        delta = self.heldout_delta
        if delta <= 0.0:
            return 0.0
        return round(delta / (float(wall_clock_s) / 3600.0), 6)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["wall_clock_samples"] = list(self.wall_clock_samples)
        payload["wall_clock_s"] = self.wall_clock_s
        payload["heldout_delta"] = self.heldout_delta
        payload["efficiency"] = self.efficiency()
        payload["efficiency_interval"] = list(self.efficiency_interval())
        return payload


def _median(values: Iterable[float]) -> float:
    ordered = sorted(float(v) for v in values)
    if not ordered:
        return 0.0
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def improver_rise_within_noise(measurements: List["ImproverMeasurement"]) -> str:
    """Reason the improver curve's rise is not distinguishable from noise.

    The point estimates can climb while every step sits inside the spread of
    the timings behind them. Requiring each generation's floor to clear its
    predecessor's ceiling is what stops a scheduler hiccup being read as an
    improver that learned something.
    """
    if len(measurements) < 2:
        return ""
    for earlier, later in zip(measurements, measurements[1:]):
        _, earlier_high = earlier.efficiency_interval()
        later_low, _ = later.efficiency_interval()
        if later_low <= earlier_high:
            return (
                f"improver gain from {earlier.generation_id} to "
                f"{later.generation_id} is within measurement noise "
                f"({later_low:.6g} does not clear {earlier_high:.6g})"
            )
    return ""


def order_invariance_violation(
    measurements: Iterable[ImproverMeasurement],
) -> str:
    """Reason the improver metric depends on lineage position, or "".

    The decisive anti-authorship check. A metric carrying the loop counter
    yields different numbers when the same generations are scored in a
    different order; a metric that reads only its own generation's
    measurements does not.
    """
    items = list(measurements)
    if len(items) < 2:
        return ""
    forward = [m.efficiency() for m in items]
    backward = [m.efficiency() for m in reversed(items)]
    backward.reverse()
    for item, first, second in zip(items, forward, backward):
        if abs(first - second) > 1e-12:
            return (
                f"improver score for {item.generation_id} changed with lineage "
                f"position ({first} vs {second}); the metric reads the counter"
            )
    return ""


def improver_curve_dependence(
    capability_curve: List[float], improver_curve: List[float]
) -> str:
    """Reason the improver curve is not independent evidence, or "" if it is.

    The anti-circularity gate. ``weight_compounding`` recorded
    ``improver_score = candidate_accuracy`` — literally the same number as
    ``after_score`` — so a rising capability curve produced an identically
    rising "improver" curve and the two-inequality test was satisfied by one
    measurement counted twice. That is not a bug in the arithmetic; it is the
    strong-RSI claim resting on nothing.

    Identity is checked, and so is affine dependence: an improver score that is
    any linear function of capability carries exactly as much independent
    information as an identical one, which is none.
    """
    if len(capability_curve) < 2 or len(improver_curve) != len(capability_curve):
        return ""

    if all(
        abs(a - b) <= 1e-12 for a, b in zip(capability_curve, improver_curve)
    ):
        return (
            "improver curve is identical to the capability curve; the second "
            "inequality is one measurement counted twice"
        )

    # Any two points lie exactly on a line, so affine dependence carries no
    # information below three generations and would fire on every honest
    # two-generation lineage. Identity above is still meaningful at n=2.
    if len(capability_curve) < 3:
        return ""

    n = len(capability_curve)
    mean_c = sum(capability_curve) / n
    mean_i = sum(improver_curve) / n
    var_c = sum((c - mean_c) ** 2 for c in capability_curve)
    if var_c <= 1e-18:
        return ""  # capability is flat; monotonicity already failed
    slope = sum(
        (c - mean_c) * (i - mean_i)
        for c, i in zip(capability_curve, improver_curve)
    ) / var_c
    intercept = mean_i - slope * mean_c
    residual = max(
        abs(i - (slope * c + intercept))
        for c, i in zip(capability_curve, improver_curve)
    )
    scale = max(abs(i) for i in improver_curve) or 1.0
    if residual / scale <= 1e-9:
        return (
            "improver curve is an exact affine function of the capability "
            f"curve (slope {slope:.4g}); it carries no independent information"
        )
    return ""


def evaluate_lineage(
    records: List[RSIGenerationRecord],
    *,
    independently_reproduced: bool = False,
    improver_measurements: Optional[List[ImproverMeasurement]] = None,
) -> RSILineageVerdict:
    if not records:
        return RSILineageVerdict(VERDICT_NO_RSI, ["no generation records"], 0, [], [])

    capability_curve = [float(record.after_score) for record in records]
    improver_curve = [float(record.improver_score) for record in records]
    reasons: List[str] = []

    if any(record.tamper_flags for record in records):
        reasons.append("tamper flags present")
    if any(record.regressions for record in records):
        reasons.append("regressions present")
    if not all(record.promoted for record in records):
        reasons.append("not every generation promoted")

    # Provenance before shape. A monotone improver curve is worth nothing if a
    # developer wrote the numbers, and no test on the values can tell.
    unmeasured = [
        f"{record.generation_id}:{record.improver_provenance}"
        for record in records
        if record.improver_provenance != PROVENANCE_MEASURED
    ]
    if unmeasured:
        reasons.append(
            "improver score is not measured for "
            + ", ".join(unmeasured)
            + "; a strong verdict needs held-out delta over measured budget"
        )
    invalid = sorted(
        {
            record.improver_provenance
            for record in records
            if record.improver_provenance not in VALID_PROVENANCE
        }
    )
    if invalid:
        reasons.append(f"unrecognised improver provenance: {', '.join(invalid)}")
    if len(records) < 2:
        reasons.append("fewer than two generations")

    capability_monotone = all(b > a for a, b in zip(capability_curve, capability_curve[1:]))
    improver_monotone = all(b > a for a, b in zip(improver_curve, improver_curve[1:]))
    if not capability_monotone:
        reasons.append("capability curve is not strictly increasing")
    if not improver_monotone:
        reasons.append("improver curve is not strictly increasing")

    dependence = improver_curve_dependence(capability_curve, improver_curve)
    if dependence:
        reasons.append(dependence)

    if improver_measurements:
        order_violation = order_invariance_violation(improver_measurements)
        if order_violation:
            reasons.append(order_violation)
        noise = improver_rise_within_noise(improver_measurements)
        if noise:
            reasons.append(noise)

    if reasons:
        return RSILineageVerdict(VERDICT_BOUNDED, reasons, len(records), capability_curve, improver_curve)
    if len(records) >= 4 and independently_reproduced:
        return RSILineageVerdict(
            VERDICT_UNDENIABLE,
            ["independent reproduction plus monotone capability and improver curves"],
            len(records),
            capability_curve,
            improver_curve,
        )
    if len(records) >= 4:
        return RSILineageVerdict(
            VERDICT_STRONG,
            ["monotone capability and improver curves across at least four generations"],
            len(records),
            capability_curve,
            improver_curve,
        )
    return RSILineageVerdict(
        VERDICT_WEAK,
        ["monotone capability and improver curves, but too few generations for strong RSI"],
        len(records),
        capability_curve,
        improver_curve,
    )


__all__ = [
    "PROVENANCE_AUTHORED",
    "PROVENANCE_MEASURED",
    "PROVENANCE_UNMEASURED",
    "VALID_PROVENANCE",
    "ImproverMeasurement",
    "RSIGenerationRecord",
    "RSILineageLedger",
    "RSILineageVerdict",
    "VERDICT_BOUNDED",
    "VERDICT_NO_RSI",
    "VERDICT_STRONG",
    "VERDICT_UNDENIABLE",
    "VERDICT_WEAK",
    "evaluate_lineage",
    "improver_curve_dependence",
    "improver_efficiency",
    "improver_rise_within_noise",
    "order_invariance_violation",
]
