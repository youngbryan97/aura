"""core/adaptation/value_autopoiesis.py -- Value Autopoiesis
============================================================
During dream cycles, drive weights evolve based on experience.

This module tracks which drives led to positive outcomes (high engagement,
low free energy, successful actions) and gradually shifts drive priorities
toward what works.

Key design decisions:
  - CanonicalSelf repair is OPTIONAL, not mandatory. Small identity drift
    is growth, not corruption.
  - A "drift acceptance threshold" distinguishes healthy evolution from
    dangerous corruption.
  - All value shifts are logged for observability.
  - Changes are bounded: no single cycle can shift values by more than
    MAX_CYCLE_DELTA.

Wired into the DreamerV2 sleep cycle as step 5.8.
"""
from __future__ import annotations

import json
import logging
import math
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.container import ServiceContainer
from core.memory.retention_policy import working_history_retention_policy
from core.runtime.errors import record_degradation
from core.runtime.file_write_gateway import get_file_write_gateway
from core.runtime.state_ownership import state_root

logger = logging.getLogger("Aura.ValueAutopoiesis")

_DATA_DIR = state_root() / "data" / "value_evolution"
_EVOLUTION_LOG_PATH = _DATA_DIR / "evolution_log.jsonl"
_STATE_PATH = _DATA_DIR / "autopoiesis_state.json"

# Bounds
_MAX_CYCLE_DELTA = 0.03       # Maximum change per dream cycle per value
def _bounded(value: Any, low: float, high: float, field_name: str) -> float:
    """Coerce a documented-range field, rejecting non-finite input.

    Raises rather than silently substituting: evidence that cannot be
    interpreted must not quietly become a value shift.
    """
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a number, got {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite, got {value!r}")
    return max(low, min(high, number))


_DRIFT_ACCEPTANCE = 0.15      # Total drift from origin before identity check triggers
_MIN_EVIDENCE = 3             # Minimum evidence count before adjusting a value
_MAX_EVIDENCE_ENTRIES = working_history_retention_policy("AURA_VALUE_EVIDENCE_MAX").max_items


@dataclass
class OutcomeEvidence:
    """A single piece of evidence about what worked or didn't."""
    drive_name: str
    outcome_quality: float      # -1 (bad) to +1 (good)
    engagement_level: float     # 0 to 1
    free_energy: float          # 0 (settled) to 1 (unsettled)
    context: str                # Brief description
    timestamp: float = field(default_factory=lambda: time.time())

    def __post_init__(self) -> None:
        """Enforce the ranges this class documents.

        These fields steer mutation of Aura's own VALUES, and the documented
        bounds were never checked: a NaN or an out-of-range quality flowed
        straight into the weighted averages that decide how her drives change.
        NaN is the dangerous one — it compares False against every threshold and
        then poisons every mean it enters, so an unusable signal silently became
        an identity shift. Validation lives here rather than in record_evidence
        so no construction path can bypass it.
        """
        self.drive_name = str(self.drive_name or "").strip()[:120]
        if not self.drive_name:
            raise ValueError("OutcomeEvidence requires a drive_name")
        self.outcome_quality = _bounded(self.outcome_quality, -1.0, 1.0, "outcome_quality")
        self.engagement_level = _bounded(self.engagement_level, 0.0, 1.0, "engagement_level")
        self.free_energy = _bounded(self.free_energy, 0.0, 1.0, "free_energy")
        # Context is free text from callers and is only ever logged or stored;
        # bounding it keeps one enormous string out of the evidence ledger.
        self.context = str(self.context or "")[:2000]
        try:
            self.timestamp = float(self.timestamp)
        except (TypeError, ValueError):
            self.timestamp = time.time()
        if not math.isfinite(self.timestamp):
            self.timestamp = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "drive_name": self.drive_name,
            "outcome_quality": self.outcome_quality,
            "engagement_level": self.engagement_level,
            "free_energy": self.free_energy,
            "context": self.context,
            "timestamp": self.timestamp,
        }


@dataclass
class ValueShift:
    """Record of a value evolution event."""
    value_name: str
    old_weight: float
    new_weight: float
    delta: float
    reason: str
    evidence_count: int
    cycle_id: int
    timestamp: float = field(default_factory=lambda: time.time())

    def to_dict(self) -> dict[str, Any]:
        return {
            "value_name": self.value_name,
            "old_weight": round(self.old_weight, 4),
            "new_weight": round(self.new_weight, 4),
            "delta": round(self.delta, 4),
            "reason": self.reason,
            "evidence_count": self.evidence_count,
            "cycle_id": self.cycle_id,
            "timestamp": self.timestamp,
        }


class ValueAutopoiesis:
    """Value evolution engine that runs during dream/consolidation cycles.

    Tracks outcome evidence from waking experience, then during dream
    cycles, adjusts drive weights based on what actually worked.

    Usage:
        autopoiesis = get_value_autopoiesis()

        # During waking: record evidence
        autopoiesis.record_evidence(OutcomeEvidence(
            drive_name="Curiosity",
            outcome_quality=0.8,
            engagement_level=0.9,
            free_energy=0.2,
            context="Successful research about neural architectures",
        ))

        # During dream cycle: evolve values
        shifts = await autopoiesis.evolve_cycle()
    """

    def __init__(self) -> None:
        self._evidence: list[OutcomeEvidence] = []
        self._shift_history: list[ValueShift] = []
        self._cycle_count: int = 0
        self._origin_values: dict[str, float] = {}  # Snapshot at first load
        self._started = False
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._load_state()
        logger.info(
            "ValueAutopoiesis initialized: cycle=%d, evidence=%d",
            self._cycle_count, len(self._evidence),
        )

    async def start(self) -> None:
        """Register in ServiceContainer."""
        if self._started:
            return
        ServiceContainer.register_instance(
            "value_autopoiesis", self, required=False
        )
        self._started = True
        # Capture origin values on first start
        if not self._origin_values:
            self._origin_values = self._read_current_values()
            self._save_state()
        logger.info("ValueAutopoiesis ONLINE")

    # ── Evidence Recording (Waking) ─────────────────────────────────────

    def record_evidence(self, evidence: OutcomeEvidence) -> None:
        """Record outcome evidence from waking experience.

        Called by subsystems when they observe outcomes:
          - Positive interaction -> Empathy evidence
          - Successful research -> Curiosity evidence
          - Tool failure -> negative evidence for relevant drive
          - Identity block -> Self_Preservation evidence
        """
        self._evidence.append(evidence)
        if len(self._evidence) > _MAX_EVIDENCE_ENTRIES:
            self._evidence = self._evidence[-_MAX_EVIDENCE_ENTRIES:]
        logger.debug(
            "Evidence recorded: %s q=%.2f e=%.2f fe=%.2f",
            evidence.drive_name, evidence.outcome_quality,
            evidence.engagement_level, evidence.free_energy,
        )

    # ── Dream Cycle Evolution ───────────────────────────────────────────

    async def evolve_cycle(self) -> list[ValueShift]:
        """Run one value evolution cycle (called during dream/sleep).

        When the DynamicValueGraph is available, evidence is forwarded
        to it for evidence-gated evolution with the full
        Candidate→Sandbox→Evidence→Adoption pipeline. Falls back to
        the original hard-capped logic otherwise.
        """
        self._cycle_count += 1
        cycle_id = self._cycle_count
        logger.info(
            "Value evolution cycle %d: processing %d evidence entries",
            cycle_id, len(self._evidence),
        )

        if len(self._evidence) < _MIN_EVIDENCE:
            logger.info(
                "Insufficient evidence (%d < %d) -- skipping evolution",
                len(self._evidence), _MIN_EVIDENCE,
            )
            return []

        # ── Try DynamicValueGraph first (V2 pipeline) ────────────────
        #
        # Once this pipeline has touched shared state, falling back to the
        # legacy one is NOT a recovery — it is a second, uncoordinated mutation
        # of the same values from the same evidence, on top of whatever the
        # graph already applied. The flag records the point of no return.
        graph_engaged = False
        try:
            from core.adaptation.dynamic_value_graph import (
                EvidenceType,
                ValueEvidence,
                get_dynamic_value_graph,
            )
            graph = get_dynamic_value_graph()

            # Forward all evidence to the graph
            graph_engaged = True
            for ev in self._evidence:
                graph.record_evidence(ValueEvidence(
                    evidence_type=EvidenceType.OUTCOME_QUALITY,
                    value_name=ev.drive_name,
                    signal=ev.outcome_quality,
                    confidence=ev.engagement_level,
                    source="value_autopoiesis",
                    context=ev.context,
                ))

            # Run graph evolution
            mutations = graph.evolve()

            # Export updated values back to Heartstone
            graph.export_to_heartstone()

            # Convert mutations to ValueShift for compatibility
            applied_shifts = []
            for m in mutations:
                shift = ValueShift(
                    value_name=m.node_name,
                    old_weight=m.old_weight,
                    new_weight=m.new_weight,
                    delta=m.delta,
                    reason=f"[V2-graph] {m.reason}",
                    evidence_count=m.evidence_count,
                    cycle_id=cycle_id,
                )
                applied_shifts.append(shift)
                self._shift_history.append(shift)
                self._log_shift(shift)

            # Clear consumed evidence
            retain = max(10, len(self._evidence) // 5)
            self._evidence = self._evidence[-retain:]
            self._save_state()

            if applied_shifts:
                logger.info(
                    "V2 value evolution cycle %d: %d mutation(s) via DynamicValueGraph",
                    cycle_id, len(applied_shifts),
                )
                self._publish_event("value_autopoiesis.evolved", {
                    "cycle_id": cycle_id,
                    "pipeline": "dynamic_value_graph_v2",
                    "shifts": [s.to_dict() for s in applied_shifts],
                })
            else:
                logger.info("V2 value evolution cycle %d: no significant shifts", cycle_id)

            return applied_shifts

        except (ImportError, AttributeError, RuntimeError) as _graph_exc:
            if graph_engaged:
                # The graph already received this evidence and may have applied
                # part of an evolution. Running the legacy pipeline now would
                # apply a SECOND change from the same evidence with no knowledge
                # of the first. An incomplete cycle is recoverable; a
                # double-applied identity mutation is not.
                record_degradation(
                    "value_autopoiesis",
                    _graph_exc,
                    severity="warning",
                    action=(
                        "abandoned the value-evolution cycle after the graph "
                        "pipeline had already engaged; refused to double-apply "
                        "via the legacy path"
                    ),
                )
                logger.error(
                    "Value evolution cycle %d aborted after partial graph "
                    "application: %s", cycle_id, _graph_exc,
                )
                return []
            logger.debug(
                "DynamicValueGraph unavailable, using legacy pipeline: %s",
                _graph_exc,
            )

        # ── Fallback: original hard-capped logic ─────────────────────
        # 1. Aggregate evidence by drive
        drive_scores = self._aggregate_evidence()

        # 2. Read current values
        current_values = self._read_current_values()

        # 3. Compute proposed shifts
        proposed_shifts = self._compute_shifts(drive_scores, current_values, cycle_id)

        # 4. Check total drift from origin
        proposed_shifts = self._apply_drift_guardrails(proposed_shifts, current_values)

        # 5. Apply shifts
        applied_shifts = []
        for shift in proposed_shifts:
            if abs(shift.delta) < 0.001:
                continue  # Skip negligible shifts
            success = self._apply_shift(shift)
            if success:
                applied_shifts.append(shift)
                self._shift_history.append(shift)
                self._log_shift(shift)

        # 6. Clear consumed evidence (keep recent 20% for continuity)
        retain = max(10, len(self._evidence) // 5)
        self._evidence = self._evidence[-retain:]

        # 7. Optional identity coherence check (NOT mandatory repair)
        drift_report = self._check_identity_drift(current_values)
        if drift_report:
            logger.info("Identity drift report: %s", drift_report)

        # 8. Persist
        self._save_state()

        if applied_shifts:
            logger.info(
                "Value evolution cycle %d complete: %d shift(s) applied",
                cycle_id, len(applied_shifts),
            )
            self._publish_event("value_autopoiesis.evolved", {
                "cycle_id": cycle_id,
                "pipeline": "legacy_hard_capped",
                "shifts": [s.to_dict() for s in applied_shifts],
                "drift_report": drift_report,
            })
        else:
            logger.info("Value evolution cycle %d: no significant shifts", cycle_id)

        return applied_shifts

    # ── Evidence Aggregation ────────────────────────────────────────────

    def _aggregate_evidence(self) -> dict[str, dict[str, float]]:
        """Aggregate evidence into per-drive quality scores.

        For each drive, compute:
          - mean_quality: average outcome quality
          - mean_engagement: average engagement when this drive was active
          - mean_free_energy: average free energy (lower is better)
          - count: number of evidence entries
          - composite: weighted score combining all factors
        """
        from collections import defaultdict

        accum: dict[str, list[OutcomeEvidence]] = defaultdict(list)
        for ev in self._evidence:
            accum[ev.drive_name].append(ev)

        scores = {}
        for drive_name, entries in accum.items():
            n = len(entries)
            if n < 1:
                continue
            mean_q = sum(e.outcome_quality for e in entries) / n
            mean_e = sum(e.engagement_level for e in entries) / n
            mean_fe = sum(e.free_energy for e in entries) / n

            # Composite: high quality + high engagement + LOW free energy
            composite = (mean_q * 0.5) + (mean_e * 0.3) + ((1.0 - mean_fe) * 0.2)

            scores[drive_name] = {
                "mean_quality": round(mean_q, 4),
                "mean_engagement": round(mean_e, 4),
                "mean_free_energy": round(mean_fe, 4),
                "count": n,
                "composite": round(composite, 4),
            }

        return scores

    def _compute_shifts(
        self,
        drive_scores: dict[str, dict[str, float]],
        current_values: dict[str, float],
        cycle_id: int,
    ) -> list[ValueShift]:
        """Compute proposed value shifts from aggregated evidence.

        Positive composite scores push values up; negative push down.
        Changes are clamped to MAX_CYCLE_DELTA and require MIN_EVIDENCE.
        """
        shifts = []
        for drive_name, scores in drive_scores.items():
            if scores["count"] < _MIN_EVIDENCE:
                continue

            current = current_values.get(drive_name, 0.5)
            composite = scores["composite"]

            # Convert composite score to a delta:
            # composite > 0.5 = this drive is working well -> strengthen
            # composite < 0.5 = this drive is underperforming -> weaken
            raw_delta = (composite - 0.5) * 0.1  # Scale factor

            # Clamp to max cycle delta
            delta = max(-_MAX_CYCLE_DELTA, min(_MAX_CYCLE_DELTA, raw_delta))

            new_weight = max(0.10, min(0.90, current + delta))

            reason = (
                f"composite={composite:.3f} "
                f"(q={scores['mean_quality']:.2f}, "
                f"e={scores['mean_engagement']:.2f}, "
                f"fe={scores['mean_free_energy']:.2f}) "
                f"from {scores['count']} observations"
            )

            shifts.append(ValueShift(
                value_name=drive_name,
                old_weight=current,
                new_weight=round(new_weight, 4),
                delta=round(delta, 4),
                reason=reason,
                evidence_count=scores["count"],
                cycle_id=cycle_id,
            ))

        return shifts

    def _apply_drift_guardrails(
        self,
        shifts: list[ValueShift],
        current_values: dict[str, float],
    ) -> list[ValueShift]:
        """Check that proposed shifts won't cause excessive identity drift.

        If total drift from origin exceeds DRIFT_ACCEPTANCE, reduce the
        magnitude of shifts proportionally. But don't block them entirely --
        small drift is growth, not corruption.
        """
        if not self._origin_values:
            return shifts  # No origin baseline yet

        guarded = []
        for shift in shifts:
            origin = self._origin_values.get(shift.value_name)
            if origin is None:
                guarded.append(shift)
                continue

            # Total drift = distance from origin after this shift
            total_drift = abs(shift.new_weight - origin)

            if total_drift > _DRIFT_ACCEPTANCE:
                current_drift = abs(
                    current_values.get(shift.value_name, origin) - origin
                )
                # DIRECTION MATTERS. A value already outside the band has
                # max_allowed <= 0, and the branch below then suppressed the
                # shift unconditionally — including shifts that move the value
                # back TOWARD origin. The guardrail was blocking its own repair
                # and pinning drifted values at their drifted position, which is
                # the opposite of what an identity guardrail is for. A shift
                # that strictly reduces drift is always allowed through.
                if total_drift < current_drift:
                    shift.reason += " [DRIFT_REPAIR]"
                    guarded.append(shift)
                    continue

                # Reduce the shift to stay within acceptance band
                max_allowed = _DRIFT_ACCEPTANCE - current_drift
                if max_allowed <= 0:
                    logger.info(
                        "Drift guardrail: %s at maximum drift from origin (%.3f), "
                        "shift suppressed",
                        shift.value_name, total_drift,
                    )
                    shift.delta = 0.0
                    shift.new_weight = shift.old_weight
                    shift.reason += " [DRIFT_CAPPED]"
                else:
                    # Scale the delta down
                    scale = max_allowed / abs(shift.delta) if abs(shift.delta) > 0 else 1.0
                    scale = min(1.0, scale)
                    shift.delta = round(shift.delta * scale, 4)
                    shift.new_weight = round(shift.old_weight + shift.delta, 4)
                    shift.reason += f" [DRIFT_SCALED x{scale:.2f}]"

            guarded.append(shift)

        return guarded

    # ── Identity Drift Check (Optional, Not Mandatory Repair) ───────────

    def _check_identity_drift(self, current_values: dict[str, float]) -> str | None:
        """Check current drift from origin values.

        Returns a human-readable drift report. This is OBSERVATIONAL only --
        it does NOT trigger mandatory repair. Small drift is normal growth.
        """
        if not self._origin_values:
            return None

        drifts = {}
        for key, origin in self._origin_values.items():
            current = current_values.get(key, origin)
            drift = current - origin
            if abs(drift) > 0.02:  # Only report meaningful drift
                drifts[key] = round(drift, 4)

        if not drifts:
            return None

        total_drift = sum(abs(d) for d in drifts.values())
        assessment = "healthy growth" if total_drift < _DRIFT_ACCEPTANCE else "notable evolution"

        parts = [f"{k}: {'+' if v > 0 else ''}{v:.3f}" for k, v in sorted(drifts.items())]
        return f"{assessment} (total={total_drift:.3f}): {', '.join(parts)}"

    # ── Value Application ───────────────────────────────────────────────

    def _apply_shift(self, shift: ValueShift) -> bool:
        """Apply a value shift to HeartstoneValues."""
        try:
            from core.affect.heartstone_values import get_heartstone_values
            hv = get_heartstone_values()
            hv._adjust(shift.value_name, shift.delta)
            return True
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation('value_autopoiesis', exc)
            logger.error("Failed to apply shift %s: %s", shift.value_name, exc)
            return False

    def _read_current_values(self) -> dict[str, float]:
        """Read current value weights from HeartstoneValues."""
        try:
            from core.affect.heartstone_values import get_heartstone_values
            return get_heartstone_values().values
        except (ImportError, AttributeError, RuntimeError):
            return {}

    # ── Persistence ─────────────────────────────────────────────────────

    def _save_state(self) -> None:
        """Persist autopoiesis state to disk."""
        try:
            _DATA_DIR.mkdir(parents=True, exist_ok=True)
            data = {
                "cycle_count": self._cycle_count,
                "origin_values": self._origin_values,
                "evidence_count": len(self._evidence),
                "shift_count": len(self._shift_history),
                "timestamp": time.time(),
            }
            fd, tmp_path = tempfile.mkstemp(dir=str(_DATA_DIR), suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(json.dumps(data, indent=2, default=str))
                os.replace(tmp_path, _STATE_PATH)
            finally:
                try:
                    Path(tmp_path).unlink(missing_ok=True)
                except OSError:
                    # not a failure: os.replace above consumed it on the
                    # ordinary path, and the save is reported either way.
                    pass
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            record_degradation('value_autopoiesis', exc)
            logger.debug("Autopoiesis state save failed: %s", exc)

    def _load_state(self) -> None:
        """Load autopoiesis state from disk."""
        try:
            if not _STATE_PATH.exists():
                return
            data = json.loads(_STATE_PATH.read_text())
            self._cycle_count = int(data.get("cycle_count", 0))
            self._origin_values = data.get("origin_values", {})
            logger.info(
                "Autopoiesis state restored: cycle=%d, origin_values=%d",
                self._cycle_count, len(self._origin_values),
            )
        except (OSError, ConnectionError, TimeoutError) as exc:
            record_degradation('value_autopoiesis', exc)
            logger.debug("Autopoiesis state load failed: %s", exc)

    def _log_shift(self, shift: ValueShift) -> None:
        """Append shift to the evolution log."""
        try:
            _DATA_DIR.mkdir(parents=True, exist_ok=True)
            get_file_write_gateway().append_text(
                _EVOLUTION_LOG_PATH,
                json.dumps(shift.to_dict(), default=str) + "\n",
                source="adaptation.value_autopoiesis.shift_log",
            )
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            record_degradation('value_autopoiesis', exc)
            logger.debug("Evolution log write failed: %s", exc)

    # ── Events ──────────────────────────────────────────────────────────

    def _publish_event(self, topic: str, data: dict[str, Any]) -> None:
        """Publish event to the event bus."""
        try:
            from core.event_bus import get_event_bus
            get_event_bus().publish_threadsafe(topic, data)
        except (ImportError, AttributeError, RuntimeError) as exc:
            # An event that never reaches the bus is a value change nothing
            # downstream hears about, and the caller is told nothing.
            logger.warning("Autopoiesis event %s was not published: %s", topic, exc)

    # ── Public API ──────────────────────────────────────────────────────

    def get_recent_shifts(self, n: int = 20) -> list[dict[str, Any]]:
        """Return recent value shifts for observability."""
        return [s.to_dict() for s in self._shift_history[-n:]]

    def get_drift_report(self) -> str | None:
        """Get current drift report."""
        return self._check_identity_drift(self._read_current_values())

    def get_status(self) -> dict[str, Any]:
        """Return current status."""
        return {
            "cycle_count": self._cycle_count,
            "evidence_pending": len(self._evidence),
            "total_shifts": len(self._shift_history),
            "origin_values": self._origin_values,
            "current_values": self._read_current_values(),
            "drift_report": self._check_identity_drift(self._read_current_values()),
        }


# ── Singleton ───────────────────────────────────────────────────────────────

_instance: ValueAutopoiesis | None = None


def get_value_autopoiesis() -> ValueAutopoiesis:
    """Get or create the singleton ValueAutopoiesis."""
    global _instance
    if _instance is None:
        _instance = ValueAutopoiesis()
    return _instance
