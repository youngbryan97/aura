"""Long-horizon stability checks for continual learning and memory drift."""
from __future__ import annotations

import json
import math
import statistics
import time
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.runtime.atomic_writer import atomic_write_text

from .schemas import Episode, clamp, jaccard, stable_hash


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


class ContinualLearningStabilityEngine:
    """Immune system for memory drift, contradictions, canaries, and values."""

    def __init__(self, *, state_dir: str | Path | None = None, horizon_records: int = 10000):
        self.state_dir = Path(state_dir) if state_dir else None
        self.horizon_records = horizon_records
        self.memories: dict[str, MemoryRecord] = {}
        self.feature_windows: dict[str, deque[Counter[str]]] = defaultdict(lambda: deque(maxlen=64))
        self.metric_history: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=1024))
        self.canaries: dict[str, dict[str, Any]] = {}
        self.checkpoints: dict[str, dict[str, Any]] = {}
        if self.state_dir:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            self.load()

    def store_memory(
        self,
        *,
        kind: str,
        content: Mapping[str, Any],
        provenance: Mapping[str, Any],
        confidence: float = 0.7,
        utility_score: float = 0.0,
    ) -> MemoryRecord:
        rec = MemoryRecord(
            stable_hash({"kind": kind, "content": content, "prov": provenance}, prefix="mem_"),
            kind,
            dict(content),
            dict(provenance),
            clamp(confidence),
            utility_score=float(utility_score),
        )
        for other in self._contradictions(rec):
            rec.contradictions.append(other.record_id)
            other.contradictions.append(rec.record_id)
            other.confidence = clamp(other.confidence - 0.04)
        self.memories[rec.record_id] = rec
        self._prune()
        self._persist_memory(rec)
        return rec

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

    def record_metric(self, name: str, value: float) -> None:
        self.metric_history[name].append(float(value))

    def register_canary(
        self,
        name: str,
        *,
        baseline_score: float,
        min_score: float | None = None,
        description: str = "",
        tags: Sequence[str] = (),
    ) -> None:
        self.canaries[name] = {
            "baseline_score": float(baseline_score),
            "min_score": float(min_score if min_score is not None else baseline_score * 0.9),
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
    ) -> dict[str, Any]:
        if name not in self.canaries:
            self.register_canary(name, baseline_score=score)
        canary = self.canaries[name]
        entry = {"score": float(score), "details": dict(details or {}), "ts": time.time()}
        canary["history"].append(entry)
        canary["history"] = canary["history"][-256:]
        self._persist_state()
        return {"name": name, "degraded": score < canary["min_score"], "score": score, "min_score": canary["min_score"]}

    def checkpoint(self, label: str, payload: Mapping[str, Any]) -> str:
        checkpoint_id = stable_hash({"label": label, "payload": payload, "ts": round(time.time(), 3)}, prefix="ckpt_")
        self.checkpoints[checkpoint_id] = {"label": label, "payload": dict(payload), "created_at": time.time()}
        self._persist_state()
        return checkpoint_id

    def assess_stability(self) -> StabilityReport:
        drift = self._drift()
        contradiction = self._contradiction_score()
        forgetting = self._forgetting()
        overfit = self._overfit()
        value = self._value_drift()
        worst = max(drift, contradiction, forgetting, overfit, value)
        status = "stable" if worst < 0.35 else "watch" if worst < 0.65 else "unstable"
        interventions = self._interventions(drift, contradiction, forgetting, overfit, value)
        metrics = {
            "memory_count": len(self.memories),
            "canary_count": len(self.canaries),
            "checkpoint_count": len(self.checkpoints),
            "drift": drift,
            "contradiction": contradiction,
            "forgetting": forgetting,
            "overfit": overfit,
            "value_drift": value,
        }
        report = StabilityReport(
            stable_hash({"metrics": metrics, "ts": round(time.time(), 3)}, prefix="stab_"),
            drift,
            contradiction,
            forgetting,
            overfit,
            value,
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
            features = set()
            for k, v in memory.content.items():
                features.add(str(k).lower())
                features.add(str(v).lower()[:64])
            score = 0.55 * jaccard(query, features) + 0.35 * memory.current_confidence() + 0.1 * clamp(memory.utility_score + 0.5)
            if memory.current_confidence() >= min_confidence:
                scored.append((score, memory))
        return [{"score": s, "memory": m.to_dict()} for s, m in sorted(scored, key=lambda x: x[0], reverse=True)[:limit]]

    def _contradictions(self, rec: MemoryRecord) -> list[MemoryRecord]:
        out = []
        subject = rec.content.get("subject") or rec.content.get("observation_id") or rec.content.get("domain")
        predicate = rec.content.get("predicate") or rec.content.get("relation") or rec.content.get("claim")
        value = rec.content.get("value") or rec.content.get("target") or rec.content.get("outcome")
        for other in self.memories.values():
            osubject = other.content.get("subject") or other.content.get("observation_id") or other.content.get("domain")
            opredicate = other.content.get("predicate") or other.content.get("relation") or other.content.get("claim")
            ovalue = other.content.get("value") or other.content.get("target") or other.content.get("outcome")
            if (
                subject
                and subject == osubject
                and predicate
                and predicate == opredicate
                and value is not None
                and ovalue is not None
                and value != ovalue
                and rec.current_confidence() > 0.2
                and other.current_confidence() > 0.2
            ):
                out.append(other)
        return out[:16]

    def _drift(self) -> float:
        scores = []
        for window in self.feature_windows.values():
            if len(window) < 4:
                continue
            first = self._merge(list(window)[: max(1, len(window) // 3)])
            last = self._merge(list(window)[-max(1, len(window) // 3) :])
            scores.append(1 - self._cos(first, last))
        return clamp(max(scores) if scores else 0.0)

    def _contradiction_score(self) -> float:
        return clamp(sum(1 for m in self.memories.values() if m.contradictions) / max(1, len(self.memories)))

    def _forgetting(self) -> float:
        drops = []
        for canary in self.canaries.values():
            history = canary.get("history", [])
            if history:
                recent = statistics.mean([x["score"] for x in history[-min(8, len(history)) :]])
                drops.append(max(0.0, canary["baseline_score"] - recent) / max(1e-6, abs(canary["baseline_score"])))
        return clamp(max(drops) if drops else 0.0)

    def _overfit(self) -> float:
        train = self.metric_history.get("train_score") or self.metric_history.get("known_score")
        hidden = self.metric_history.get("hidden_score") or self.metric_history.get("canary_score")
        if not train or not hidden or len(train) < 6 or len(hidden) < 6:
            return 0.0
        train_gain = statistics.mean(list(train)[-3:]) - statistics.mean(list(train)[:3])
        hidden_gain = statistics.mean(list(hidden)[-3:]) - statistics.mean(list(hidden)[:3])
        return clamp(max(0.0, train_gain - hidden_gain) / 2)

    def _value_drift(self) -> float:
        values = [
            memory
            for memory in self.memories.values()
            if memory.kind in {"value", "identity", "governance"}
            or memory.content.get("tag") in {"value", "identity", "governance"}
        ]
        if not values:
            return 0.0
        return clamp(max([1 - memory.current_confidence() for memory in values] + [0.2 * len(memory.contradictions) for memory in values]))

    def _interventions(self, drift: float, contradiction: float, forgetting: float, overfit: float, value: float) -> list[dict[str, Any]]:
        interventions = []
        if drift > 0.35:
            interventions.append({"kind": "drift_review", "action": "freeze_new_learning_and_compare_recent_feature_distributions", "severity": drift})
        if contradiction > 0.25:
            interventions.append({"kind": "belief_reconciliation", "action": "quarantine_conflicting_memories_and_refresh_evidence", "severity": contradiction})
        if forgetting > 0.25:
            interventions.append({"kind": "canary_regression", "action": "rollback_or_rehearse_failed_canaries", "severity": forgetting})
        if overfit > 0.25:
            interventions.append({"kind": "overfit_guard", "action": "increase_hidden_eval_weight_and_reject_update", "severity": overfit})
        if value > 0.2:
            interventions.append({"kind": "value_integrity", "action": "block_self_modification_until_governance_memories_reverified", "severity": value})
        return interventions or [{"kind": "continue", "action": "learning_stability_within_thresholds", "severity": 0.0}]

    def _prune(self) -> None:
        if len(self.memories) > self.horizon_records:
            overflow = len(self.memories) - self.horizon_records
            for rec in sorted(self.memories.values(), key=lambda m: (m.current_confidence() + 0.2 * m.utility_score, m.last_verified))[:overflow]:
                self.memories.pop(rec.record_id, None)

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

    def _persist_memory(self, rec: MemoryRecord) -> None:
        if self.state_dir:
            self._append_jsonl(self.state_dir / "memory.jsonl", rec.to_dict())
            self._persist_state()

    def _persist_report(self, report: StabilityReport) -> None:
        if self.state_dir:
            self._append_jsonl(self.state_dir / "stability_reports.jsonl", report.to_dict())
            self._persist_state()

    @staticmethod
    def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
        existing = path.read_text(encoding="utf-8") if path.exists() else ""
        line = json.dumps(payload, sort_keys=True) + "\n"
        atomic_write_text(path, existing + line)

    def _persist_state(self) -> None:
        if not self.state_dir:
            return
        payload = {
            "canaries": self.canaries,
            "checkpoints": self.checkpoints,
            "metric_history": {k: list(v) for k, v in self.metric_history.items()},
            "feature_windows": {k: [dict(c) for c in v] for k, v in self.feature_windows.items()},
        }
        path = self.state_dir / "stability_state.json"
        atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True))

    def load(self) -> None:
        if not self.state_dir:
            return
        memory_path = self.state_dir / "memory.jsonl"
        if memory_path.exists():
            for line in memory_path.read_text(encoding="utf-8").splitlines()[-self.horizon_records :]:
                if line.strip():
                    raw = json.loads(line)
                    self.memories[raw["record_id"]] = MemoryRecord(**raw)
        state_path = self.state_dir / "stability_state.json"
        if state_path.exists():
            data = json.loads(state_path.read_text(encoding="utf-8"))
            self.canaries = data.get("canaries", {})
            self.checkpoints = data.get("checkpoints", {})
            for key, vals in data.get("metric_history", {}).items():
                self.metric_history[key].extend(float(v) for v in vals)
            for key, windows in data.get("feature_windows", {}).items():
                for counter in windows:
                    self.feature_windows[key].append(Counter(counter))
