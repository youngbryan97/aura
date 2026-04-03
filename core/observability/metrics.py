"""
In-process metrics collection. No Prometheus dependency required —
metrics are stored in memory and exposed via the health endpoint.
"""
import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MetricSnapshot:
    count: int = 0
    total: float = 0.0
    min: float = float("inf")
    max: float = 0.0
    last_10: deque = field(default_factory=lambda: deque(maxlen=10))

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else 0.0

    @property
    def p90(self) -> float:
        if not self.last_10:
            return 0.0
        s = sorted(self.last_10)
        idx = max(0, int(len(s) * 0.9) - 1)
        return s[idx]

    def record(self, value: float):
        self.count += 1
        self.total += value
        self.min = min(self.min, value)
        self.max = max(self.max, value)
        self.last_10.append(value)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "count": self.count,
            "mean_ms": round(self.mean * 1000, 1),
            "min_ms": round(self.min * 1000, 1) if self.min != float("inf") else 0,
            "max_ms": round(self.max * 1000, 1),
            "p90_ms": round(self.p90 * 1000, 1),
        }


class MetricsCollector:
    """Thread-safe in-process metrics."""

    def __init__(self):
        self._lock = threading.Lock()
        self._timers: Dict[str, MetricSnapshot] = defaultdict(MetricSnapshot)
        self._counters: Dict[str, int] = defaultdict(int)
        self._gauges: Dict[str, float] = {}

    def increment(self, name: str, by: int = 1):
        with self._lock:
            self._counters[name] += by

    def gauge(self, name: str, value: float):
        with self._lock:
            self._gauges[name] = value

    def record_duration(self, name: str, seconds: float):
        with self._lock:
            self._timers[name].record(seconds)

    def timer(self, name: str):
        """Context manager for timing a block."""
        return _TimerContext(self, name)

    def get_snapshot(self, name: str) -> Optional[Dict[str, Any]]:
        """Retrieve a snapshot of a specific timer metric's current state."""
        with self._lock:
            if name not in self._timers:
                return None
            return self._timers[name].to_dict()

    def get_all_snapshots(self) -> Dict[str, Dict[str, Any]]:
        """Retrieve snapshots of all registered timer metrics."""
        with self._lock:
            return {name: m.to_dict() for name, m in self._timers.items()}

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "counters": dict(self._counters),
                "gauges":   dict(self._gauges),
                "timers":   {k: v.to_dict() for k, v in self._timers.items()},
            }


class _TimerContext:
    def __init__(self, collector: MetricsCollector, name: str):
        self._c = collector
        self._name = name
        self._start = 0.0

    def __enter__(self):
        self._start = time.monotonic()
        return self

    def __exit__(self, *_):
        self._c.record_duration(self._name, time.monotonic() - self._start)


# Singleton
_collector: Optional[MetricsCollector] = None


def get_metrics() -> MetricsCollector:
    global _collector
    if _collector is None:
        _collector = MetricsCollector()
    return _collector
