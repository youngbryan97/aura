"""Memory engineering and benchmark utilities.

Aura treats memory as a first-class component: records have explicit scope,
agent write stamps, vector clocks, graph links, and measurable retrieval
trade-offs.  The benchmark mirrors LOCOMO/Mem0-style concerns using local,
deterministic scoring: accuracy/F1 proxy, latency, token load, and duplication.
"""
from __future__ import annotations

import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping


class MemoryScope(str, Enum):
    USER = "user"
    SESSION = "session"
    AGENT = "agent"
    APPLICATION = "application"


@dataclass(frozen=True)
class VectorClock:
    counters: Mapping[str, int] = field(default_factory=dict)

    def tick(self, actor: str) -> "VectorClock":
        data = dict(self.counters)
        data[actor] = data.get(actor, 0) + 1
        return VectorClock(data)

    def merge(self, other: "VectorClock") -> "VectorClock":
        keys = set(self.counters) | set(other.counters)
        return VectorClock({key: max(self.counters.get(key, 0), other.counters.get(key, 0)) for key in keys})

    def happens_before(self, other: "VectorClock") -> bool:
        keys = set(self.counters) | set(other.counters)
        return all(self.counters.get(key, 0) <= other.counters.get(key, 0) for key in keys) and any(
            self.counters.get(key, 0) < other.counters.get(key, 0) for key in keys
        )

    def to_dict(self) -> dict[str, int]:
        return dict(self.counters)


@dataclass(frozen=True)
class AgentWriteStamp:
    actor: str
    scope: MemoryScope
    clock: VectorClock
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "actor": self.actor,
            "scope": self.scope.value,
            "clock": self.clock.to_dict(),
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class ScopedMemoryRecord:
    record_id: str
    text: str
    scope: MemoryScope
    owner: str
    agent: str
    metadata: dict[str, Any] = field(default_factory=dict)
    links: tuple[str, ...] = ()
    stamp: AgentWriteStamp | None = None

    def visible_to(self, *, user: str, session: str, agent: str, application: str) -> bool:
        if self.scope is MemoryScope.USER:
            return self.owner == user
        if self.scope is MemoryScope.SESSION:
            return self.metadata.get("session") == session
        if self.scope is MemoryScope.AGENT:
            return self.agent == agent
        if self.scope is MemoryScope.APPLICATION:
            return self.metadata.get("application", application) == application
        return False


@dataclass(frozen=True)
class MemoryBenchmarkCase:
    query: str
    relevant_ids: tuple[str, ...]
    context: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryBenchmarkResult:
    strategy: str
    accuracy: float
    f1: float
    p95_latency_ms: float
    mean_tokens: float
    duplication_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "accuracy": round(self.accuracy, 4),
            "f1": round(self.f1, 4),
            "p95_latency_ms": round(self.p95_latency_ms, 3),
            "mean_tokens": round(self.mean_tokens, 2),
            "duplication_rate": round(self.duplication_rate, 4),
        }


class GraphMemoryIndex:
    """Tiny graph-enhanced index for selective memory retrieval."""

    def __init__(self) -> None:
        self.records: dict[str, ScopedMemoryRecord] = {}
        self.edges: dict[str, set[str]] = defaultdict(set)
        self.cache: dict[str, list[str]] = {}

    def add(self, record: ScopedMemoryRecord) -> None:
        self.records[record.record_id] = record
        for linked in record.links:
            self.edges[record.record_id].add(linked)
            self.edges[linked].add(record.record_id)

    def share_cache(self, key: str, record_ids: Iterable[str]) -> None:
        self.cache[key] = list(dict.fromkeys(str(r) for r in record_ids))

    def invalidate(self, record_id: str) -> None:
        for key, values in list(self.cache.items()):
            if record_id in values:
                self.cache.pop(key, None)

    def retrieve_full_context(self, case: MemoryBenchmarkCase) -> list[ScopedMemoryRecord]:
        return list(self.records.values())

    def retrieve_graph_selective(self, case: MemoryBenchmarkCase, *, top_k: int = 8) -> list[ScopedMemoryRecord]:
        query_terms = self._terms(case.query)
        scored: list[tuple[float, str]] = []
        for record_id, record in self.records.items():
            score = self._score(query_terms, record)
            if score > 0:
                scored.append((score, record_id))
        scored.sort(reverse=True)
        seeds = [record_id for _, record_id in scored[: max(1, top_k // 2)]]
        expanded = self._expand(seeds, limit=top_k)
        return [self.records[rid] for rid in expanded if rid in self.records]

    def _expand(self, seeds: Iterable[str], limit: int) -> list[str]:
        seen: set[str] = set()
        queue: deque[str] = deque(seeds)
        while queue and len(seen) < limit:
            current = queue.popleft()
            if current in seen:
                continue
            seen.add(current)
            for neighbor in sorted(self.edges.get(current, ())):
                if neighbor not in seen:
                    queue.append(neighbor)
        return list(seen)

    @staticmethod
    def _terms(text: str) -> set[str]:
        return {tok.strip(".,:;!?()[]{}").lower() for tok in text.split() if len(tok.strip(".,:;!?()[]{}")) > 2}

    def _score(self, query_terms: set[str], record: ScopedMemoryRecord) -> float:
        terms = self._terms(record.text)
        if not query_terms or not terms:
            return 0.0
        overlap = len(query_terms & terms)
        metadata_boost = sum(1 for value in record.metadata.values() if str(value).lower() in query_terms)
        return overlap + metadata_boost * 0.5


class MemoryBenchmarkRunner:
    def __init__(self, index: GraphMemoryIndex) -> None:
        self.index = index

    def run(self, cases: Iterable[MemoryBenchmarkCase]) -> dict[str, MemoryBenchmarkResult]:
        case_list = list(cases)
        return {
            "full_context": self._evaluate(case_list, "full_context"),
            "graph_selective": self._evaluate(case_list, "graph_selective"),
        }

    def _evaluate(self, cases: list[MemoryBenchmarkCase], strategy: str) -> MemoryBenchmarkResult:
        latencies: list[float] = []
        token_counts: list[int] = []
        f1s: list[float] = []
        accuracies: list[float] = []
        duplicate_counts = 0
        total_returned = 0
        for case in cases:
            start = time.perf_counter()
            if strategy == "full_context":
                records = self.index.retrieve_full_context(case)
            else:
                records = self.index.retrieve_graph_selective(case)
            elapsed = (time.perf_counter() - start) * 1000.0
            ids = [r.record_id for r in records]
            duplicate_counts += len(ids) - len(set(ids))
            total_returned += len(ids)
            latencies.append(elapsed)
            token_counts.append(sum(len(r.text.split()) for r in records))
            precision, recall, f1 = self._precision_recall_f1(ids, case.relevant_ids)
            f1s.append(f1)
            accuracies.append(1.0 if recall >= 0.999 else 0.0)
        return MemoryBenchmarkResult(
            strategy=strategy,
            accuracy=sum(accuracies) / max(1, len(accuracies)),
            f1=sum(f1s) / max(1, len(f1s)),
            p95_latency_ms=self._p95(latencies),
            mean_tokens=sum(token_counts) / max(1, len(token_counts)),
            duplication_rate=duplicate_counts / max(1, total_returned),
        )

    @staticmethod
    def _precision_recall_f1(ids: Iterable[str], relevant: Iterable[str]) -> tuple[float, float, float]:
        retrieved = set(ids)
        gold = set(relevant)
        if not gold:
            return 1.0, 1.0, 1.0
        tp = len(retrieved & gold)
        precision = tp / max(1, len(retrieved))
        recall = tp / max(1, len(gold))
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        return precision, recall, f1

    @staticmethod
    def _p95(values: list[float]) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        return ordered[min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))]


def context_consistency_ratio(records: Iterable[ScopedMemoryRecord]) -> float:
    grouped: dict[str, list[VectorClock]] = defaultdict(list)
    for record in records:
        if record.stamp:
            grouped[record.owner].append(record.stamp.clock)
    ratios: list[float] = []
    for clocks in grouped.values():
        if len(clocks) < 2:
            continue
        ordered_pairs = 0
        total_pairs = 0
        for idx, left in enumerate(clocks):
            for right in clocks[idx + 1 :]:
                total_pairs += 1
                if left.happens_before(right) or right.happens_before(left) or left.counters == right.counters:
                    ordered_pairs += 1
        ratios.append(ordered_pairs / max(1, total_pairs))
    return sum(ratios) / max(1, len(ratios))


__all__ = [
    "MemoryScope",
    "VectorClock",
    "AgentWriteStamp",
    "ScopedMemoryRecord",
    "MemoryBenchmarkCase",
    "MemoryBenchmarkResult",
    "GraphMemoryIndex",
    "MemoryBenchmarkRunner",
    "context_consistency_ratio",
]
