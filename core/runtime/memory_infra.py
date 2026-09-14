"""core/runtime/memory_infra.py — attributed memory dumps.

Clean-room adoption of Chromium's memory-infra: per-component allocator
dumps, ownership edges to resolve double counting, detail levels, and
periodic global dumps.

Aura's open memory problem is documented and unresolved: a four-hour soak
grows ~242MB/h, linearly, and the two candidate explanations (a real leak
versus proof machinery deferring reclamation) could not be separated
because nothing attributes bytes to a component. `psutil` says the process
is bigger. `tracemalloc` says which *allocation site* is bigger, which is
a different and less useful question — a dict that grew is not the same as
knowing which subsystem's cache it belonged to.

memory-infra answers the useful question by inverting the direction:
components declare what they are holding. Then:

* **A global dump is a snapshot of the whole system's attribution**, taken
  on a schedule and on pressure. Cheap at BACKGROUND detail so it can run
  always.
* **The diff between two dumps is the leak report.** "Between 02:00 and
  06:00 the process grew 970MB; the episodic buffer grew 12MB, the probe
  cache grew 940MB" ends the argument in one line. This is the tool that
  was missing.
* **Ownership edges resolve double counting.** Two components that both
  reference one buffer would otherwise each report its size and the total
  would exceed the process. An edge says who owns it; the other side's
  claim is subtracted.
* **Unattributed bytes are reported explicitly.** Process RSS minus the
  sum of dumps is a real number and it is the most important one on the
  page: it is how much of the runtime nobody is accounting for.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

logger = logging.getLogger("Aura.MemoryInfra")


class DetailLevel(StrEnum):
    #: Cheap enough to run every minute forever. Totals only.
    BACKGROUND = "background"
    #: Per-subsystem breakdown. Suitable on pressure.
    LIGHT = "light"
    #: Everything, including per-entry detail. Explicitly requested only.
    DETAILED = "detailed"


_LEVEL_ORDER = {DetailLevel.BACKGROUND: 0, DetailLevel.LIGHT: 1, DetailLevel.DETAILED: 2}


@dataclass
class AllocatorDump:
    """One component's claim about what it is holding."""

    name: str
    size_bytes: int
    object_count: int = 0
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "size_bytes": self.size_bytes,
            "size_mb": round(self.size_bytes / 1e6, 3),
            "object_count": self.object_count,
            "detail": dict(self.detail),
        }


@dataclass(frozen=True)
class OwnershipEdge:
    """``owner`` is charged for the bytes; ``target``'s claim is discounted."""

    owner: str
    target: str
    importance: int = 1


ProviderFn = Callable[[DetailLevel], "AllocatorDump | list[AllocatorDump] | int | None"]


@dataclass
class _Provider:
    name: str
    fn: ProviderFn
    min_level: DetailLevel
    owner: str
    failures: int = 0


@dataclass
class GlobalDump:
    at: float
    level: DetailLevel
    dumps: dict[str, AllocatorDump]
    process_rss_bytes: int
    edges: tuple[OwnershipEdge, ...] = ()

    @property
    def attributed_bytes(self) -> int:
        """Sum of claims, with owned duplicates removed."""
        discounted = {edge.target for edge in self.edges}
        return sum(
            dump.size_bytes for name, dump in self.dumps.items() if name not in discounted
        )

    @property
    def unattributed_bytes(self) -> int:
        return max(0, self.process_rss_bytes - self.attributed_bytes)

    def to_dict(self) -> dict[str, Any]:
        attributed = self.attributed_bytes
        return {
            "at": self.at,
            "level": str(self.level),
            "process_rss_mb": round(self.process_rss_bytes / 1e6, 2),
            "attributed_mb": round(attributed / 1e6, 2),
            "unattributed_mb": round(self.unattributed_bytes / 1e6, 2),
            "attributed_fraction": (
                round(attributed / self.process_rss_bytes, 4)
                if self.process_rss_bytes
                else 0.0
            ),
            "components": {
                name: dump.to_dict()
                for name, dump in sorted(
                    self.dumps.items(), key=lambda kv: -kv[1].size_bytes
                )
            },
            "edges": [
                {"owner": e.owner, "target": e.target, "importance": e.importance}
                for e in self.edges
            ],
        }


@dataclass
class DumpDiff:
    """The leak report: who grew between two dumps."""

    earlier_at: float
    later_at: float
    process_growth_bytes: int
    growth_by_component: dict[str, int]
    unattributed_growth_bytes: int

    @property
    def elapsed_s(self) -> float:
        return max(1e-9, self.later_at - self.earlier_at)

    @property
    def rate_mb_per_hour(self) -> float:
        return (self.process_growth_bytes / 1e6) / (self.elapsed_s / 3600.0)

    def top_growers(self, limit: int = 5) -> list[tuple[str, int]]:
        return sorted(self.growth_by_component.items(), key=lambda kv: -kv[1])[:limit]

    def to_dict(self) -> dict[str, Any]:
        return {
            "elapsed_s": round(self.elapsed_s, 1),
            "process_growth_mb": round(self.process_growth_bytes / 1e6, 2),
            "rate_mb_per_hour": round(self.rate_mb_per_hour, 2),
            "unattributed_growth_mb": round(self.unattributed_growth_bytes / 1e6, 2),
            "top_growers": [
                {"component": name, "growth_mb": round(delta / 1e6, 3)}
                for name, delta in self.top_growers(8)
            ],
        }

    @property
    def component_growth_bytes(self) -> int:
        """Total growth across components that grew."""
        return sum(delta for delta in self.growth_by_component.values() if delta > 0)

    def narrative(self) -> str:
        """One sentence naming what grew.

        Deliberately reports component growth even when process RSS is
        flat. A component that grew 900MB while RSS did not is not "no
        growth" — it is either something else shrinking by the same amount
        or an attribution that is wrong, and both are worth knowing. The
        earlier version of this method said "process did not grow" in
        exactly that case, which is the kind of true-but-misleading
        summary this module exists to replace.
        """
        window = f"{self.elapsed_s / 60:.0f}m"
        top = [(name, delta) for name, delta in self.top_growers(3) if delta > 0]
        attribution = ", ".join(f"{name} +{delta / 1e6:.1f}MB" for name, delta in top)

        if self.process_growth_bytes > 0:
            return (
                f"process grew {self.process_growth_bytes / 1e6:.0f}MB over {window} "
                f"({self.rate_mb_per_hour:.0f}MB/h): "
                f"{attribution or 'no component reported growth'}; "
                f"unattributed {self.unattributed_growth_bytes / 1e6:.0f}MB"
            )
        if not top:
            return f"no growth over {window}: process and every component held steady"
        return (
            f"process RSS did not grow over {window}, but attributed components did "
            f"({self.component_growth_bytes / 1e6:.0f}MB: {attribution}) — either "
            "something unattributed shrank by the same amount, or an attribution is wrong"
        )


class MemoryInfra:
    def __init__(self, *, history: int = 64) -> None:
        self._lock = threading.Lock()
        self._providers: dict[str, _Provider] = {}
        self._edges: list[OwnershipEdge] = []
        self._history: list[GlobalDump] = []
        self._history_limit = history
        self.dumps_taken = 0

    # ── registration ──────────────────────────────────────────────────
    def register(
        self,
        name: str,
        fn: ProviderFn,
        *,
        owner: str = "unknown",
        min_level: DetailLevel = DetailLevel.BACKGROUND,
    ) -> None:
        """Declare what a component is holding.

        The provider receives the detail level and returns an
        :class:`AllocatorDump`, a list of them, or a plain byte count.
        """
        with self._lock:
            self._providers[name] = _Provider(
                name=name, fn=fn, min_level=min_level, owner=owner
            )

    def unregister(self, name: str) -> None:
        with self._lock:
            self._providers.pop(name, None)

    def add_ownership_edge(self, owner: str, target: str, *, importance: int = 1) -> None:
        """Two components claiming one buffer would double-count it."""
        with self._lock:
            self._edges = [
                e for e in self._edges if not (e.owner == owner and e.target == target)
            ]
            self._edges.append(OwnershipEdge(owner=owner, target=target, importance=importance))

    def providers(self) -> list[str]:
        with self._lock:
            return sorted(self._providers)

    # ── dumping ───────────────────────────────────────────────────────
    def dump(self, level: DetailLevel = DetailLevel.BACKGROUND) -> GlobalDump:
        with self._lock:
            providers = [
                p for p in self._providers.values() if _LEVEL_ORDER[level] >= _LEVEL_ORDER[p.min_level]
            ]
            edges = tuple(self._edges)

        dumps: dict[str, AllocatorDump] = {}
        for provider in providers:
            try:
                outcome = provider.fn(level)
            except Exception:  # noqa: BLE001 — one broken provider must not blind the rest
                provider.failures += 1
                logger.debug("memory provider %s failed", provider.name, exc_info=True)
                continue
            for dump in _coerce_dumps(provider.name, outcome):
                dumps[dump.name] = dump

        global_dump = GlobalDump(
            at=time.time(),
            level=level,
            dumps=dumps,
            process_rss_bytes=_process_rss_bytes(),
            edges=edges,
        )
        with self._lock:
            self._history.append(global_dump)
            if len(self._history) > self._history_limit:
                del self._history[: -self._history_limit]
            self.dumps_taken += 1
        return global_dump

    def diff(self, earlier: GlobalDump, later: GlobalDump) -> DumpDiff:
        growth: dict[str, int] = {}
        for name, dump in later.dumps.items():
            before = earlier.dumps.get(name)
            growth[name] = dump.size_bytes - (before.size_bytes if before else 0)
        for name, dump in earlier.dumps.items():
            growth.setdefault(name, -dump.size_bytes)
        return DumpDiff(
            earlier_at=earlier.at,
            later_at=later.at,
            process_growth_bytes=later.process_rss_bytes - earlier.process_rss_bytes,
            growth_by_component=growth,
            unattributed_growth_bytes=later.unattributed_bytes - earlier.unattributed_bytes,
        )

    def diff_since(self, seconds: float) -> DumpDiff | None:
        """Growth over the last N seconds, from retained dumps."""
        with self._lock:
            history = list(self._history)
        if len(history) < 2:
            return None
        cutoff = time.time() - seconds
        earlier = next((d for d in history if d.at >= cutoff), history[0])
        later = history[-1]
        if earlier is later:
            earlier = history[0]
        if earlier is later:
            return None
        return self.diff(earlier, later)

    def leak_report(self, *, window_s: float = 3600.0) -> dict[str, Any]:
        """The answer to 'what is growing', in one call."""
        diff = self.diff_since(window_s)
        if diff is None:
            return {"available": False, "reason": "need at least two dumps in the window"}
        return {
            "available": True,
            "narrative": diff.narrative(),
            **diff.to_dict(),
        }

    def report(self) -> dict[str, Any]:
        with self._lock:
            history = list(self._history)
            providers = {
                p.name: {"owner": p.owner, "min_level": str(p.min_level), "failures": p.failures}
                for p in self._providers.values()
            }
        latest = history[-1] if history else None
        return {
            "providers": providers,
            "dumps_taken": self.dumps_taken,
            "retained_dumps": len(history),
            "latest": latest.to_dict() if latest else None,
            "leak_report": self.leak_report(),
        }

    def reset_for_test(self) -> None:
        with self._lock:
            self._providers.clear()
            self._edges.clear()
            self._history.clear()
            self.dumps_taken = 0


def _coerce_dumps(name: str, outcome: Any) -> list[AllocatorDump]:
    if outcome is None:
        return []
    if isinstance(outcome, AllocatorDump):
        return [outcome if outcome.name else AllocatorDump(name, outcome.size_bytes)]
    if isinstance(outcome, list):
        return [d for d in outcome if isinstance(d, AllocatorDump)]
    if isinstance(outcome, (int, float)):
        return [AllocatorDump(name=name, size_bytes=int(outcome))]
    return []


def _process_rss_bytes() -> int:
    try:
        from core.runtime.resource_observation import get_resource_observer

        # Own RSS only: the default observation adds a full process-table
        # scan whose result this function discards.
        return int(
            get_resource_observer().memory(include_process_tree=False).process_rss_bytes
            or 0
        )
    except Exception:  # noqa: BLE001
        logger.debug("process RSS probe failed", exc_info=True)
        return 0


_INFRA = MemoryInfra()


def get_memory_infra() -> MemoryInfra:
    return _INFRA


def register_provider(
    name: str,
    fn: ProviderFn,
    *,
    owner: str = "unknown",
    min_level: DetailLevel = DetailLevel.BACKGROUND,
) -> None:
    _INFRA.register(name, fn, owner=owner, min_level=min_level)


def register_sized_container(
    name: str,
    container: Any,
    *,
    owner: str = "unknown",
    bytes_per_entry: int = 512,
) -> None:
    """Attribute a plain cache or buffer without instrumenting it.

    An estimate that is attributed beats an exact number that is not: the
    diff between two dumps is what finds the leak, and a consistent
    estimate diffs correctly.
    """

    def provider(_level: DetailLevel) -> AllocatorDump:
        try:
            count = len(container)
        except (TypeError, AttributeError):
            count = 0
        return AllocatorDump(
            name=name,
            size_bytes=count * bytes_per_entry,
            object_count=count,
            detail={"estimated": True, "bytes_per_entry": bytes_per_entry},
        )

    register_provider(name, provider, owner=owner)


def install_runtime_providers() -> list[str]:
    """Attribute the buffers this codebase's own disciplines hold."""
    from core.observability.bus_recorder import get_bus_recorder
    from core.observability.trace_events import get_tracer

    def bus_ring(_level: DetailLevel) -> AllocatorDump:
        report = get_bus_recorder().report()
        return AllocatorDump(
            name="observability.bus_ring",
            size_bytes=int(report["ring_size"]) * 1024,
            object_count=int(report["ring_size"]),
            detail={"span_s": report["ring_span_s"], "estimated": True},
        )

    def trace_ring(_level: DetailLevel) -> AllocatorDump:
        report = get_tracer().report()
        return AllocatorDump(
            name="observability.trace_ring",
            size_bytes=int(report["buffered"]) * 256,
            object_count=int(report["buffered"]),
            detail={"estimated": True},
        )

    def histogram_registry(_level: DetailLevel) -> AllocatorDump:
        from core.observability.histograms import histograms_report

        report = histograms_report()
        return AllocatorDump(
            name="observability.histograms",
            size_bytes=int(report["count"]) * 4096,
            object_count=int(report["count"]),
            detail={"estimated": True},
        )

    def atomspace(_level: DetailLevel) -> AllocatorDump:
        from core.knowledge.atomspace import get_atomspace

        space = get_atomspace()
        count = len(space)
        return AllocatorDump(
            name="knowledge.atomspace",
            # An atom plus its truth value, attention value, and index
            # entries. Estimated, and labelled as such — a wrong-by-2x
            # number that tracks growth beats no number at all.
            size_bytes=count * 512,
            object_count=count,
            detail={"estimated": True},
        )

    def telemetry_history(_level: DetailLevel) -> AllocatorDump:
        from core.fsw.telemetry_dictionary import get_telemetry

        report = get_telemetry().report()
        samples = sum(count for _name, count in report["busiest"])
        return AllocatorDump(
            name="fsw.telemetry_history",
            size_bytes=(report["channels"] * 256) + (samples * 64),
            object_count=report["channels"],
            detail={"estimated": True, "events_emitted": report["events_emitted"]},
        )

    def diagnostic_logs(_level: DetailLevel) -> AllocatorDump:
        """The append-only forensic records: sanitizer, assertion, lockdep."""
        from core.fsw.assertions import assertions_report
        from core.runtime.lockdep import lockdep_report
        from core.runtime.sanitizers import sanitizer_report

        entries = (
            assertions_report()["distinct_sites"]
            + sanitizer_report()["distinct_findings"]
            + len(lockdep_report()["splats"])
            + len(lockdep_report()["order_edges"])
        )
        return AllocatorDump(
            name="runtime.diagnostic_logs",
            size_bytes=entries * 2048,
            object_count=entries,
            detail={"estimated": True},
        )

    def controller_queues(_level: DetailLevel) -> AllocatorDump:
        from core.runtime.reconcile import reconcile_report

        report = reconcile_report()
        depth = int(report["total_queue_depth"])
        return AllocatorDump(
            name="runtime.controller_queues",
            size_bytes=depth * 512,
            object_count=depth,
            detail={"estimated": True, "controllers": report["count"]},
        )

    def pass_records(_level: DetailLevel) -> AllocatorDump:
        from core.pipeline.pass_manager import get_instrumentation

        records = len(get_instrumentation().records())
        return AllocatorDump(
            name="pipeline.pass_records",
            size_bytes=records * 512,
            object_count=records,
            detail={"estimated": True},
        )

    def embedding_model(_level: DetailLevel) -> AllocatorDump:
        """The encoder the serving process holds in its own memory, exactly.

        LIVE, 2026-09-10: 4,213MB of RSS with 0.6% attributed. The serving
        process loads Qwen3-Embedding-0.6B through sentence-transformers and
        moves it to the GPU, which on Apple Silicon is the same memory, and no
        provider had ever claimed it — so a growth diff could not name the one
        component holding most of what there was to hold. The bytes are the
        parameters' own count and element size, not an estimate.
        """
        from core.container import ServiceContainer

        engine = ServiceContainer.get("vector_memory_engine", default=None)
        model = getattr(engine, "_model", None)
        if model is None or not callable(getattr(model, "parameters", None)):
            return AllocatorDump(
                name="memory.embedding_model",
                size_bytes=0,
                object_count=0,
                detail={"loaded": False},
            )
        total = 0
        tensors = 0
        device = ""
        for parameter in model.parameters():
            total += int(parameter.numel()) * int(parameter.element_size())
            tensors += 1
            if not device:
                device = str(getattr(parameter, "device", "") or "")
        return AllocatorDump(
            name="memory.embedding_model",
            size_bytes=total,
            object_count=tensors,
            detail={
                "loaded": True,
                "model": str(getattr(engine, "PREFERRED_MODEL", "") or ""),
                "device": device,
            },
        )

    def torch_device_memory(_level: DetailLevel) -> AllocatorDump:
        """What torch holds on the GPU beyond the parameters: activations,
        the allocator's cache. Exact, from the allocator; zero when torch is
        not loaded here, which is a fact and not a failure."""
        try:
            import torch
        except ImportError:
            return AllocatorDump(name="torch.device_memory", size_bytes=0, detail={"loaded": False})
        mps = getattr(torch, "mps", None)
        if mps is None or not torch.backends.mps.is_available():
            return AllocatorDump(name="torch.device_memory", size_bytes=0, detail={"device": "none"})
        allocated = int(mps.current_allocated_memory())
        driver = int(getattr(mps, "driver_allocated_memory", lambda: 0)())
        return AllocatorDump(
            name="torch.device_memory",
            size_bytes=max(allocated, driver),
            detail={"device": "mps", "allocated": allocated, "driver": driver},
        )

    def mlx_device_memory(_level: DetailLevel) -> AllocatorDump:
        """MLX arrays held in THIS process. The resident cortex lives in the
        worker and is not here; anything that is — a reflex model, a probe —
        is exact from MLX's own accounting."""
        try:
            import mlx.core as mx
        except ImportError:
            return AllocatorDump(name="mlx.device_memory", size_bytes=0, detail={"loaded": False})
        active = int(mx.get_active_memory())
        cache = int(mx.get_cache_memory())
        return AllocatorDump(
            name="mlx.device_memory",
            size_bytes=active + cache,
            detail={"active": active, "cache": cache, "peak": int(mx.get_peak_memory())},
        )

    register_provider(
        "memory.embedding_model", embedding_model, owner="core/memory/vector_memory_engine.py"
    )
    register_provider("torch.device_memory", torch_device_memory, owner="core/runtime/memory_infra.py")
    register_provider("mlx.device_memory", mlx_device_memory, owner="core/runtime/memory_infra.py")
    register_provider("observability.bus_ring", bus_ring, owner="core/observability/bus_recorder.py")
    register_provider("observability.trace_ring", trace_ring, owner="core/observability/trace_events.py")
    register_provider(
        "observability.histograms", histogram_registry, owner="core/observability/histograms.py"
    )
    register_provider("knowledge.atomspace", atomspace, owner="core/knowledge/atomspace.py")
    register_provider(
        "fsw.telemetry_history", telemetry_history, owner="core/fsw/telemetry_dictionary.py"
    )
    register_provider(
        "runtime.diagnostic_logs", diagnostic_logs, owner="core/runtime/sanitizers.py"
    )
    register_provider(
        "runtime.controller_queues", controller_queues, owner="core/runtime/reconcile.py"
    )
    register_provider("pipeline.pass_records", pass_records, owner="core/pipeline/pass_manager.py")
    return _INFRA.providers()


def memory_infra_report() -> dict[str, Any]:
    return _INFRA.report()


def reset_memory_infra_for_test() -> None:
    _INFRA.reset_for_test()


__all__ = [
    "AllocatorDump",
    "DetailLevel",
    "DumpDiff",
    "GlobalDump",
    "MemoryInfra",
    "OwnershipEdge",
    "get_memory_infra",
    "install_runtime_providers",
    "memory_infra_report",
    "register_provider",
    "register_sized_container",
    "reset_memory_infra_for_test",
]
