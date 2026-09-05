"""Long-horizon stability checks for continual learning and memory drift.

CP126 found ten defects here, three critical, and they compose into one
sentence: **an engine with no evidence reported that everything was stable,
and then recommended controls nobody ran.**

  * Every unmeasurable axis returned 0.0 — no feature windows, no canaries,
    fewer than six paired metrics, no value memories — so a freshly
    constructed engine scored perfectly on all five and reported "stable".
    Unmeasured is now ``None`` and propagates to a ``not_ready`` status. A
    stability engine that cannot see is not a stable system.
  * Interventions were strings: "freeze_new_learning", "quarantine",
    "rollback", "block_self_modification". No owner was invoked and nothing
    acknowledged. They are labelled recommendations now, each with a named
    owner and ``enforced=False``, and there is a seam an owner can register
    against — at which point the acknowledgement, or the failure, is recorded
    on the intervention itself.
  * Re-storing identical content re-applied contradiction penalties to its
    counterparts, overwrote the map entry and appended a second JSONL record.
    Storing is idempotent by record id.
  * Contradiction was field-name equality over three alias triples, one of
    which mapped ``observation_id``/``domain`` to "subject" and ``outcome``
    to "value" — so two ordinary episodes in one domain with different
    outcomes were a contradiction. Only records that declare an explicit
    subject/predicate/value claim participate.
  * Contradiction updates mutated the counterpart records in memory and
    persisted only the new one, so a restart kept half of each update.
  * Pruning removed records from the map and never from the log, which grew
    forever; ``load`` then took the last N LINES rather than the retained
    set, so pruned records came back and contradiction ids dangled.
  * ``_append_jsonl`` read the entire file, concatenated one line, and
    atomically rewrote the whole thing — quadratic I/O on every memory and
    every report.
  * ``load`` raised on the first malformed line, during construction.
"""
from __future__ import annotations

import json
import logging
import math
import statistics
import time
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from core.governance_context import local_internal_governed_scope
from core.runtime.errors import record_degradation
from core.runtime.file_write_gateway import get_file_write_gateway

from .schemas import Episode, clamp, jaccard, stable_hash

logger = logging.getLogger("Aura.ContinualLearningStability")

#: Below this many paired samples, a train/hidden comparison is not a
#: measurement. Six was the previous floor and it was applied to UNPAIRED
#: histories, which is a different thing entirely.
_MIN_PAIRED_SAMPLES = 6
#: Feature windows needed before a drift comparison means anything.
_MIN_DRIFT_WINDOWS = 4
#: Log lines above which the JSONL is compacted to the retained set.
_COMPACT_THRESHOLD_LINES = 4096
#: Refuse payloads larger than this rather than writing them.
_MAX_CONTENT_BYTES = 256 * 1024


class StabilityEvidenceError(ValueError):
    """A caller-supplied value cannot be recorded as evidence."""


def _finite(value: Any, *, field_name: str) -> float:
    """A finite float, or a refusal naming the field.

    Metrics, confidences, utilities and canary scores were accepted with
    ``float(value)``, so NaN and infinity entered the histories that later
    decided whether learning was stable.
    """
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise StabilityEvidenceError(f"{field_name} is not a number: {value!r}") from exc
    if not math.isfinite(number):
        raise StabilityEvidenceError(f"{field_name} must be finite, got {number!r}")
    return number


@dataclass(frozen=True)
class Measurement:
    """A stability axis, or an honest statement that it was not measured.

    Every one of these returned 0.0 when it had nothing to look at, and 0.0
    on a risk axis reads as "no risk". The distinction between "measured, and
    it is fine" and "there was nothing to measure" is the whole of this type.
    """

    value: float | None
    basis: str
    samples: int = 0
    paired: bool = True

    @property
    def measured(self) -> bool:
        return self.value is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "measured": self.measured,
            "basis": self.basis,
            "samples": self.samples,
            "paired": self.paired,
        }


@dataclass
class MemoryRecord:
    record_id: str
    kind: str
    content: dict[str, Any]
    provenance: dict[str, Any]
    confidence: float
    created_at: float = field(default_factory=time.time)
    last_verified: float = field(default_factory=time.time)
    contradictions: list[str] = field(default_factory=list)
    utility_score: float = 0.0
    decay_rate: float = 0.002
    #: Set when the record has been pruned. Written to the log as a tombstone
    #: so a reload does not resurrect it.
    retired: bool = False

    def current_confidence(self, now: float | None = None) -> float:
        age_hours = max(0.0, ((now or time.time()) - self.last_verified) / 3600)
        return clamp(self.confidence * math.exp(-self.decay_rate * age_hours) - 0.08 * len(self.contradictions))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StabilityReport:
    report_id: str
    drift_score: float
    contradiction_score: float
    forgetting_score: float
    overfit_score: float
    value_drift_score: float
    status: str
    interventions: list[dict[str, Any]]
    metrics: dict[str, Any]
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


#: Who would carry out each recommendation if anyone did. Named so a reader
#: of a report can tell that nothing has been asked to.
_INTERVENTION_OWNERS = {
    "drift_review": "learning_scheduler",
    "belief_reconciliation": "belief_store",
    "canary_regression": "training_promotion_gate",
    "overfit_guard": "training_promotion_gate",
    "value_integrity": "self_modification_gate",
    "not_ready": "operator",
    "continue": "none",
}


class ContinualLearningStabilityEngine:
    """Immune system for memory drift, contradictions, canaries, and values.

    It DETECTS. It does not enforce: every intervention it emits is a
    recommendation addressed to an owner, and says so. Register a handler
    with :meth:`set_intervention_handler` to close that loop, and the
    acknowledgement is recorded on the intervention.
    """

    def __init__(self, *, state_dir: str | Path | None = None, horizon_records: int = 10000):
        self.state_dir = Path(state_dir) if state_dir else None
        self.horizon_records = horizon_records
        self.memories: dict[str, MemoryRecord] = {}
        self.feature_windows: dict[str, deque[Counter[str]]] = defaultdict(lambda: deque(maxlen=64))
        self.metric_history: dict[str, deque[tuple[str, float]]] = defaultdict(lambda: deque(maxlen=1024))
        self.canaries: dict[str, dict[str, Any]] = {}
        self.checkpoints: dict[str, dict[str, Any]] = {}
        self._intervention_handler: Callable[[Mapping[str, Any]], Any] | None = None
        self._quarantined_lines = 0
        self._log_lines = 0
        if self.state_dir:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            self.load()

    # ── enforcement seam ──────────────────────────────────────────────

    def set_intervention_handler(
        self, handler: Callable[[Mapping[str, Any]], Any] | None
    ) -> None:
        """Register the thing that actually carries interventions out.

        Without one, every intervention in a report carries
        ``enforced=False`` and ``acknowledgement="no handler registered"`` —
        which is what was true all along, and was not written down.
        """
        self._intervention_handler = handler

    # ── ingestion ─────────────────────────────────────────────────────

    def store_memory(
        self,
        *,
        kind: str,
        content: Mapping[str, Any],
        provenance: Mapping[str, Any],
        confidence: float = 0.7,
        utility_score: float = 0.0,
    ) -> MemoryRecord:
        """Record a memory, once.

        Idempotent by record id: the id is derived from kind + content +
        provenance, so a repeat ingestion is the SAME memory. It used to
        re-apply a confidence penalty to every counterpart, overwrite the map
        entry and append a second log record each time it arrived.
        """
        provenance = self._validated_provenance(provenance)
        content = self._validated_content(content)
        confidence = clamp(_finite(confidence, field_name="confidence"))
        utility = _finite(utility_score, field_name="utility_score")

        record_id = stable_hash(
            {"kind": kind, "content": content, "prov": provenance}, prefix="mem_"
        )
        existing = self.memories.get(record_id)
        if existing is not None and not existing.retired:
            existing.last_verified = time.time()
            self._persist_memory(existing)
            return existing

        rec = MemoryRecord(
            record_id, kind, dict(content), dict(provenance), confidence,
            utility_score=utility,
        )
        touched: list[MemoryRecord] = []
        for other in self._contradictions(rec):
            rec.contradictions.append(other.record_id)
            other.contradictions.append(rec.record_id)
            other.confidence = clamp(other.confidence - 0.04)
            touched.append(other)
        self.memories[record_id] = rec
        self._prune()
        self._persist_memory(rec)
        # The counterparts changed too. Persisting only the new record left
        # half of every contradiction update behind on restart.
        for other in touched:
            self._persist_memory(other)
        return rec

    @staticmethod
    def _validated_provenance(provenance: Mapping[str, Any]) -> dict[str, Any]:
        prov = dict(provenance or {})
        source = str(prov.get("source") or "").strip()
        if not source:
            raise StabilityEvidenceError(
                "provenance.source is required: a memory with no stated origin "
                "cannot be weighed against one that has an origin"
            )
        prov["source"] = source
        return prov

    @staticmethod
    def _validated_content(content: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(content or {})
        try:
            size = len(json.dumps(payload, sort_keys=True, default=str).encode("utf-8"))
        except (TypeError, ValueError) as exc:
            raise StabilityEvidenceError(f"content is not serialisable: {exc}") from exc
        if size > _MAX_CONTENT_BYTES:
            raise StabilityEvidenceError(
                f"content is {size} bytes, over the {_MAX_CONTENT_BYTES} limit"
            )
        return payload

    def ingest_episode(self, ep: Episode) -> MemoryRecord:
        return self.store_memory(
            kind="episode",
            content={
                "observation_id": ep.observation.observation_id,
                "domain": ep.observation.domain,
                "action": ep.action.to_dict(),
                "predicted": ep.predicted,
                "outcome": ep.outcome.to_dict(),
                "features": sorted(ep.features())[:256],
            },
            provenance={"episode_id": ep.episode_id, "source": "advanced_cognition"},
            confidence=max(0.3, ep.observation.confidence),
            utility_score=ep.outcome.utility,
        )

    def observe_feature_distribution(self, domain: str, features: Sequence[str]) -> None:
        self.feature_windows[domain].append(Counter(features))

    def record_metric(self, name: str, value: float, *, run_id: str = "") -> None:
        """Record a metric sample, optionally naming the run it came from.

        ``run_id`` is what makes a train/hidden comparison a PAIRED one. It
        is optional because most callers do not have it yet, and a report
        built from unpaired history says ``paired=false`` rather than
        presenting the difference of two unrelated series as an overfit
        estimate.
        """
        self.metric_history[name].append(
            (str(run_id or ""), _finite(value, field_name=f"metric[{name}]"))
        )

    def register_canary(
        self,
        name: str,
        *,
        baseline_score: float,
        min_score: float | None = None,
        description: str = "",
        tags: Sequence[str] = (),
    ) -> None:
        baseline = _finite(baseline_score, field_name="baseline_score")
        floor = (
            _finite(min_score, field_name="min_score")
            if min_score is not None
            else baseline * 0.9
        )
        self.canaries[name] = {
            "baseline_score": baseline,
            "min_score": floor,
            "description": description,
            "tags": list(tags),
            "history": [],
        }
        self._persist_state()

    def update_canary(
        self,
        name: str,
        score: float,
        *,
        details: Mapping[str, Any] | None = None,
        run_id: str = "",
    ) -> dict[str, Any]:
        value = _finite(score, field_name="canary_score")
        if name not in self.canaries:
            self.register_canary(name, baseline_score=value)
        canary = self.canaries[name]
        entry = {
            "score": value,
            "details": dict(details or {}),
            "run_id": str(run_id or ""),
            "ts": time.time(),
        }
        canary["history"].append(entry)
        canary["history"] = canary["history"][-256:]
        self._persist_state()
        return {
            "name": name,
            "degraded": value < canary["min_score"],
            "score": value,
            "min_score": canary["min_score"],
            "run_id": entry["run_id"],
        }

    def checkpoint(self, label: str, payload: Mapping[str, Any]) -> str:
        checkpoint_id = stable_hash({"label": label, "payload": payload, "ts": round(time.time(), 3)}, prefix="ckpt_")
        self.checkpoints[checkpoint_id] = {"label": label, "payload": dict(payload), "created_at": time.time()}
        self._persist_state()
        return checkpoint_id

    # ── assessment ────────────────────────────────────────────────────

    def assess_stability(self) -> StabilityReport:
        """Score the five axes, and say which of them were measured at all."""
        axes = {
            "drift": self._drift(),
            "contradiction": self._contradiction_score(),
            "forgetting": self._forgetting(),
            "overfit": self._overfit(),
            "value_drift": self._value_drift(),
        }
        measured = {name: m for name, m in axes.items() if m.measured}
        unmeasured = sorted(name for name, m in axes.items() if not m.measured)

        if not measured:
            # No evidence at all. The old engine scored this 0.0 across the
            # board and called it stable.
            status = "not_ready"
            worst = 0.0
        else:
            worst = max(m.value or 0.0 for m in measured.values())
            if unmeasured:
                # Some axes are blind. "watch" is the most this can honestly
                # claim, however good the ones it CAN see look.
                status = "watch" if worst < 0.65 else "unstable"
            else:
                status = "stable" if worst < 0.35 else "watch" if worst < 0.65 else "unstable"

        interventions = self._interventions(axes, unmeasured)
        metrics = {
            "memory_count": len(self.memories),
            "canary_count": len(self.canaries),
            "checkpoint_count": len(self.checkpoints),
            "measured_axes": sorted(measured),
            "unmeasured_axes": unmeasured,
            "worst_measured": worst if measured else None,
            "quarantined_log_lines": self._quarantined_lines,
            "axes": {name: m.as_dict() for name, m in axes.items()},
            # The five bare numbers callers already read. Unmeasured axes are
            # None here rather than 0.0.
            "drift": axes["drift"].value,
            "contradiction": axes["contradiction"].value,
            "forgetting": axes["forgetting"].value,
            "overfit": axes["overfit"].value,
            "value_drift": axes["value_drift"].value,
        }
        report = StabilityReport(
            stable_hash({"metrics": metrics, "ts": round(time.time(), 3)}, prefix="stab_"),
            axes["drift"].value or 0.0,
            axes["contradiction"].value or 0.0,
            axes["forgetting"].value or 0.0,
            axes["overfit"].value or 0.0,
            axes["value_drift"].value or 0.0,
            status,
            interventions,
            metrics,
        )
        self._persist_report(report)
        return report

    def retrieve(
        self,
        query_features: Sequence[str],
        *,
        limit: int = 12,
        min_confidence: float = 0.15,
    ) -> list[dict[str, Any]]:
        query = set(query_features)
        scored = []
        for memory in self.memories.values():
            if memory.retired:
                continue
            features = set()
            for k, v in memory.content.items():
                features.add(str(k).lower())
                features.add(str(v).lower()[:64])
            score = 0.55 * jaccard(query, features) + 0.35 * memory.current_confidence() + 0.1 * clamp(memory.utility_score + 0.5)
            if memory.current_confidence() >= min_confidence:
                scored.append((score, memory))
        return [{"score": s, "memory": m.to_dict()} for s, m in sorted(scored, key=lambda x: x[0], reverse=True)[:limit]]

    # ── contradiction ─────────────────────────────────────────────────

    @staticmethod
    def _claim(record: MemoryRecord) -> tuple[str, str, str] | None:
        """The explicit subject/predicate/value claim, if the record makes one.

        The old version fell back through aliases: ``observation_id`` or
        ``domain`` became the subject, ``outcome`` became the value. Two
        ordinary episodes in one domain with different outcomes were
        therefore a contradiction, and every episode ingested into a busy
        domain contradicted every other. Only a record that literally
        declares a claim participates.
        """
        content = record.content
        subject = content.get("subject")
        predicate = content.get("predicate") or content.get("relation")
        if subject is None or predicate is None or "value" not in content:
            return None
        return (
            str(subject).strip().lower(),
            str(predicate).strip().lower(),
            json.dumps(content["value"], sort_keys=True, default=str),
        )

    def _contradictions(self, rec: MemoryRecord) -> list[MemoryRecord]:
        claim = self._claim(rec)
        if claim is None:
            return []
        subject, predicate, value = claim
        out = []
        for other in self.memories.values():
            if other.retired:
                continue
            other_claim = self._claim(other)
            if other_claim is None:
                continue
            osubject, opredicate, ovalue = other_claim
            if (
                subject == osubject
                and predicate == opredicate
                and value != ovalue
                and rec.current_confidence() > 0.2
                and other.current_confidence() > 0.2
            ):
                out.append(other)
        return out[:16]

    # ── the five axes ─────────────────────────────────────────────────

    def _drift(self) -> Measurement:
        scores = []
        windows = 0
        for window in self.feature_windows.values():
            if len(window) < _MIN_DRIFT_WINDOWS:
                continue
            windows += 1
            first = self._merge(list(window)[: max(1, len(window) // 3)])
            last = self._merge(list(window)[-max(1, len(window) // 3) :])
            scores.append(1 - self._cos(first, last))
        if not scores:
            return Measurement(
                None,
                f"no domain has {_MIN_DRIFT_WINDOWS} feature windows yet",
            )
        return Measurement(clamp(max(scores)), "max cosine drift across domains", windows)

    def _contradiction_score(self) -> Measurement:
        live = [m for m in self.memories.values() if not m.retired]
        if not live:
            return Measurement(None, "no memories stored")
        conflicted = sum(1 for m in live if m.contradictions)
        return Measurement(
            clamp(conflicted / len(live)), "fraction of memories with a contradiction", len(live)
        )

    def _forgetting(self) -> Measurement:
        drops = []
        samples = 0
        for canary in self.canaries.values():
            history = canary.get("history", [])
            if not history:
                continue
            window = history[-min(8, len(history)) :]
            samples += len(window)
            recent = statistics.mean([x["score"] for x in window])
            drops.append(
                max(0.0, canary["baseline_score"] - recent)
                / max(1e-6, abs(canary["baseline_score"]))
            )
        if not drops:
            # A canary that has never been run has not passed. This returned
            # 0.0, so "no regression" and "never checked" were the same number.
            return Measurement(None, "no canary has been scored")
        return Measurement(clamp(max(drops)), "worst canary drop from baseline", samples)

    def _overfit(self) -> Measurement:
        train = self._metric_series("train_score", "known_score")
        hidden = self._metric_series("hidden_score", "canary_score")
        if not train or not hidden:
            return Measurement(None, "no train/hidden metric history")

        paired_runs = sorted(
            {run for run, _ in train if run} & {run for run, _ in hidden if run}
        )
        if len(paired_runs) >= _MIN_PAIRED_SAMPLES:
            train_by_run = dict(train)
            hidden_by_run = dict(hidden)
            gaps = [train_by_run[r] - hidden_by_run[r] for r in paired_runs]
            early = statistics.mean(gaps[: len(gaps) // 2])
            late = statistics.mean(gaps[len(gaps) // 2 :])
            return Measurement(
                clamp(max(0.0, late - early) / 2),
                "widening train-minus-hidden gap over paired runs",
                len(paired_runs),
                paired=True,
            )

        if len(train) < _MIN_PAIRED_SAMPLES or len(hidden) < _MIN_PAIRED_SAMPLES:
            return Measurement(
                None,
                f"fewer than {_MIN_PAIRED_SAMPLES} samples on one side",
                min(len(train), len(hidden)),
            )
        train_values = [v for _, v in train]
        hidden_values = [v for _, v in hidden]
        train_gain = statistics.mean(train_values[-3:]) - statistics.mean(train_values[:3])
        hidden_gain = statistics.mean(hidden_values[-3:]) - statistics.mean(hidden_values[:3])
        return Measurement(
            clamp(max(0.0, train_gain - hidden_gain) / 2),
            "UNPAIRED first/last means — no run ids to pair on",
            min(len(train_values), len(hidden_values)),
            paired=False,
        )

    def _metric_series(self, *names: str) -> list[tuple[str, float]]:
        for name in names:
            series = self.metric_history.get(name)
            if series:
                return list(series)
        return []

    def _value_drift(self) -> Measurement:
        values = [
            memory
            for memory in self.memories.values()
            if not memory.retired
            and (
                memory.kind in {"value", "identity", "governance"}
                or memory.content.get("tag") in {"value", "identity", "governance"}
            )
        ]
        if not values:
            # No value memories is not evidence that values are intact. It is
            # the absence of the thing that would be evidence either way.
            return Measurement(None, "no value/identity/governance memories stored")
        worst = max(
            [1 - memory.current_confidence() for memory in values]
            + [0.2 * len(memory.contradictions) for memory in values]
        )
        return Measurement(clamp(worst), "worst value-memory degradation", len(values))

    # ── interventions ─────────────────────────────────────────────────

    def _interventions(
        self, axes: Mapping[str, Measurement], unmeasured: Sequence[str]
    ) -> list[dict[str, Any]]:
        proposals: list[dict[str, Any]] = []

        def _propose(kind: str, action: str, severity: float) -> None:
            proposals.append({"kind": kind, "action": action, "severity": severity})

        if unmeasured:
            _propose(
                "not_ready",
                "instrument_missing_axes_before_trusting_this_report:" + ",".join(unmeasured),
                1.0,
            )

        thresholds = (
            ("drift", 0.35, "drift_review", "freeze_new_learning_and_compare_recent_feature_distributions"),
            ("contradiction", 0.25, "belief_reconciliation", "quarantine_conflicting_memories_and_refresh_evidence"),
            ("forgetting", 0.25, "canary_regression", "rollback_or_rehearse_failed_canaries"),
            ("overfit", 0.25, "overfit_guard", "increase_hidden_eval_weight_and_reject_update"),
            ("value_drift", 0.2, "value_integrity", "block_self_modification_until_governance_memories_reverified"),
        )
        for axis, threshold, kind, action in thresholds:
            measurement = axes[axis]
            if measurement.measured and (measurement.value or 0.0) > threshold:
                _propose(kind, action, measurement.value or 0.0)

        if not proposals:
            _propose("continue", "learning_stability_within_thresholds", 0.0)

        return [self._dispatch(proposal) for proposal in proposals]

    def _dispatch(self, proposal: dict[str, Any]) -> dict[str, Any]:
        """Hand the recommendation to its owner, or say that nobody has one.

        These read as controls — "freeze", "quarantine", "rollback", "block
        self-modification" — and nothing was invoked by any of them. The
        recommendation is now labelled as one, addressed to an owner, and
        carries whatever came back.
        """
        intervention = dict(proposal)
        intervention["owner"] = _INTERVENTION_OWNERS.get(proposal["kind"], "unassigned")
        intervention["enforced"] = False
        if self._intervention_handler is None:
            intervention["acknowledgement"] = "no handler registered — recommendation only"
            return intervention
        try:
            acknowledgement = self._intervention_handler(dict(intervention))
        except Exception as exc:  # noqa: BLE001 - an owner's failure is evidence
            record_degradation(
                "continual_learning_stability",
                exc,
                action=f"intervention {proposal['kind']} was not carried out",
            )
            intervention["acknowledgement"] = f"handler raised: {type(exc).__name__}"
            return intervention
        intervention["enforced"] = bool(acknowledgement)
        intervention["acknowledgement"] = (
            acknowledgement if isinstance(acknowledgement, (str, dict)) else bool(acknowledgement)
        )
        return intervention

    # ── retention ─────────────────────────────────────────────────────

    def _prune(self) -> None:
        live = [m for m in self.memories.values() if not m.retired]
        if len(live) <= self.horizon_records:
            return
        overflow = len(live) - self.horizon_records
        ranked = sorted(
            live, key=lambda m: (m.current_confidence() + 0.2 * m.utility_score, m.last_verified)
        )
        for rec in ranked[:overflow]:
            rec.retired = True
            # A tombstone, not a silent map deletion: load() replayed the last
            # N lines of the log and brought pruned records straight back.
            self._persist_memory(rec)
            self.memories.pop(rec.record_id, None)
            for other in self.memories.values():
                if rec.record_id in other.contradictions:
                    other.contradictions.remove(rec.record_id)

    # ── helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _merge(counters: list[Counter[str]]) -> Counter[str]:
        out: Counter[str] = Counter()
        for counter in counters:
            out.update(counter)
        return out

    @staticmethod
    def _cos(a: Counter[str], b: Counter[str]) -> float:
        keys = set(a) | set(b)
        dot = sum(a[k] * b[k] for k in keys)
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        return clamp(dot / (na * nb)) if na and nb else 0.0

    # ── persistence ───────────────────────────────────────────────────

    def _persist_memory(self, rec: MemoryRecord) -> None:
        if self.state_dir:
            self._append_jsonl(self.state_dir / "memory.jsonl", rec.to_dict())
            self._maybe_compact()

    def _persist_report(self, report: StabilityReport) -> None:
        if self.state_dir:
            self._append_jsonl(self.state_dir / "stability_reports.jsonl", report.to_dict())

    def _append_jsonl(self, path: Path, payload: Mapping[str, Any]) -> None:
        """Append one line.

        This read the entire file, concatenated one line and atomically
        rewrote the whole thing — for every memory and every report. At ten
        thousand records that is a hundred million characters of I/O to add
        one row, on whatever thread happened to be storing a memory.
        """
        line = json.dumps(payload, sort_keys=True, default=str) + "\n"
        self._write(path, line, append=True)
        self._log_lines += 1

    @staticmethod
    def _write(path: Path, text: str, *, append: bool = False) -> None:
        """Every write on one governed lane.

        The module wrote through atomic_writer directly, which is the
        migration debt the ownership baseline counts. There is nothing
        special about this state that needs its own path.
        """
        source = "advanced_cognition.continual_learning_stability"
        gateway = get_file_write_gateway()
        with local_internal_governed_scope(source, domain="state_mutation"):
            if append:
                gateway.append_text(path, text, source=source)
            else:
                gateway.write_text(path, text, source=source)

    def _maybe_compact(self) -> None:
        """Rewrite the log to the retained set once it outgrows the map.

        Pruning removed records from the map and never from the log, so the
        file grew without bound and a reload read rows for memories that had
        been dropped on purpose.
        """
        if not self.state_dir or self._log_lines < _COMPACT_THRESHOLD_LINES:
            return
        path = self.state_dir / "memory.jsonl"
        lines = [
            json.dumps(rec.to_dict(), sort_keys=True, default=str)
            for rec in self.memories.values()
            if not rec.retired
        ]
        self._write(path, "\n".join(lines) + ("\n" if lines else ""))
        self._log_lines = len(lines)
        logger.info("Compacted stability memory log to %d retained records", len(lines))

    def _persist_state(self) -> None:
        if not self.state_dir:
            return
        payload = {
            "canaries": self.canaries,
            "checkpoints": self.checkpoints,
            "metric_history": {
                k: [[run, value] for run, value in v] for k, v in self.metric_history.items()
            },
            "feature_windows": {k: [dict(c) for c in v] for k, v in self.feature_windows.items()},
        }
        path = self.state_dir / "stability_state.json"
        self._write(path, json.dumps(payload, indent=2, sort_keys=True))

    # ── recovery ──────────────────────────────────────────────────────

    def load(self) -> None:
        """Rebuild from disk, quarantining anything unreadable.

        Every one of a malformed line, a truncated write, an unknown field
        and a non-numeric confidence used to raise — inside ``__init__``, so
        one bad row made the engine unconstructible and took the runtime that
        built it with it. Bad rows are counted and skipped; the count is in
        every report.
        """
        if not self.state_dir:
            return
        self._load_memories(self.state_dir / "memory.jsonl")
        self._load_state(self.state_dir / "stability_state.json")

    def _load_memories(self, path: Path) -> None:
        if not path.exists():
            return
        try:
            raw_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            record_degradation("continual_learning_stability", exc, action="memory log unreadable")
            return
        self._log_lines = len(raw_lines)
        # Last write wins per record id, so a contradiction update recorded
        # after the original replaces it rather than sitting behind it.
        for line in raw_lines:
            if not line.strip():
                continue
            record = self._decode_memory(line)
            if record is None:
                self._quarantined_lines += 1
                continue
            if record.retired:
                self.memories.pop(record.record_id, None)
                continue
            self.memories[record.record_id] = record
        self._drop_dangling_contradictions()
        if self._quarantined_lines:
            record_degradation(
                "continual_learning_stability",
                StabilityEvidenceError(f"{self._quarantined_lines} unreadable memory rows skipped"),
                action="rows quarantined; engine constructed from the remainder",
            )

    @staticmethod
    def _decode_memory(line: str) -> MemoryRecord | None:
        try:
            raw = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            return None
        if not isinstance(raw, dict) or not raw.get("record_id"):
            return None
        known = {f for f in MemoryRecord.__dataclass_fields__}
        payload = {k: v for k, v in raw.items() if k in known}
        try:
            payload["confidence"] = _finite(payload.get("confidence", 0.0), field_name="confidence")
            payload["utility_score"] = _finite(payload.get("utility_score", 0.0), field_name="utility")
            payload["decay_rate"] = _finite(payload.get("decay_rate", 0.002), field_name="decay_rate")
            return MemoryRecord(**payload)
        except (StabilityEvidenceError, TypeError, ValueError):
            return None

    def _drop_dangling_contradictions(self) -> None:
        present = set(self.memories)
        for record in self.memories.values():
            record.contradictions = [c for c in record.contradictions if c in present]

    def _load_state(self, path: Path) -> None:
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except (json.JSONDecodeError, ValueError, OSError) as exc:
            record_degradation(
                "continual_learning_stability", exc,
                action="stability state discarded; canaries and metrics start empty",
            )
            return
        if not isinstance(data, dict):
            return
        self.canaries = data.get("canaries", {}) if isinstance(data.get("canaries"), dict) else {}
        self.checkpoints = (
            data.get("checkpoints", {}) if isinstance(data.get("checkpoints"), dict) else {}
        )
        for key, vals in (data.get("metric_history") or {}).items():
            for sample in vals:
                try:
                    if isinstance(sample, (list, tuple)) and len(sample) == 2:
                        run, value = sample
                    else:
                        run, value = "", sample
                    self.metric_history[key].append(
                        (str(run), _finite(value, field_name=f"metric[{key}]"))
                    )
                except StabilityEvidenceError:
                    self._quarantined_lines += 1
        for key, windows in (data.get("feature_windows") or {}).items():
            for counter in windows:
                if isinstance(counter, dict):
                    self.feature_windows[key].append(Counter(counter))
