"""The providers that say what each discipline is holding.

Taken out of `core/runtime/memory_infra.py`, which is the register and the
dump — the instrument. These are the readings: each one asks a subsystem for
what it holds and reports it as an AllocatorDump. A reading that is an
estimate says so in its detail; the encoder and the device allocators are
exact.
"""

from __future__ import annotations

from core.runtime.memory_infra import AllocatorDump, DetailLevel, register_provider

__all__ = ["install_runtime_providers"]


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
    from core.runtime.memory_infra import get_memory_infra

    return get_memory_infra().providers()
