"""Host-safe MLX memory envelope for ANY tool that loads a model.

An unguarded evaluation tool drove this host to 103 GB and forced a
shutdown. Training already ran inside a resource envelope
(``tools/run_recurrence_training_envelope.py``); evaluation, probes, and
one-off scripts did not, so a single unbounded decode loop could exhaust
the machine. Memory safety cannot be a property of one lane — it has to be
a property of loading a model at all.

This module makes the envelope a two-line call any script can make, with
limits derived from the ACTUAL host rather than hardcoded, and with a
periodic reclaim hook for long generation loops.

    from core.runtime.mlx_memory_guard import mlx_memory_envelope

    with mlx_memory_envelope(fraction=0.5) as envelope:
        model, tokenizer = load(...)
        ...
        envelope.reclaim(step)   # inside any long loop

MLX's memory limit controls allocator reclamation; it is not a hard process
RSS limit. Model owners must also enforce their host resource envelope.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("Aura.MLXMemoryGuard")

MLX_MEMORY_GUARD_SCHEMA = "aura.mlx_memory_guard.v1"

# Never hand MLX more than this share of physical RAM by default. The host
# still needs room for the OS, the window server, and whatever else the
# operator is running; jetsam kills the largest process, which would be us.
DEFAULT_FRACTION = 0.5
MIN_LIMIT_BYTES = 2 * 1024**3
# Reclaim cadence for generation loops. Cheap relative to a forward pass.
DEFAULT_RECLAIM_EVERY = 16
_SWAP_USED = re.compile(r"\bused\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*([MG])\b", re.IGNORECASE)


def _parse_swap_used_gb(output: str) -> float | None:
    """Parse the labeled used field; vm.swapusage prints total first."""

    match = _SWAP_USED.search(output)
    if match is None:
        return None
    value = float(match.group(1))
    return value if match.group(2).upper() == "G" else value / 1024.0


def _pressure_reasons(
    *,
    host_gb: float,
    reclaimable_gb: float,
    compressed_gb: float,
    swap_used_gb: float | None,
) -> tuple[str, ...]:
    """Classify present pressure; allocated swap is corroboration, not history."""

    reasons = []
    if compressed_gb > 0.25 * host_gb:
        reasons.append("compressor_high")
    if reclaimable_gb < 0.08 * host_gb:
        reasons.append("reclaimable_critical")
    if (
        swap_used_gb is not None
        and swap_used_gb > 2.0
        and reclaimable_gb < 0.15 * host_gb
    ):
        reasons.append("swap_correlated_scarcity")
    return tuple(reasons)


def _synchronize_and_reclaim() -> None:
    """Finish queued Metal work before releasing allocator buffers.

    MLX execution is asynchronous. Clearing its allocator cache while a stream
    can still reference a buffer creates a native lifetime race that Python
    cannot catch. The second barrier makes the reclaim itself observable before
    a caller starts the next generation.
    """
    import mlx.core as mx

    mx.synchronize()
    mx.clear_cache()
    mx.synchronize()


def host_memory_bytes() -> int:
    """Physical RAM, or a conservative floor when it cannot be read."""
    try:
        from core.runtime.resource_observation import get_resource_observer

        size = int(get_resource_observer().memory(include_process_tree=False).total_bytes)
        if size > 0:
            return size
    except (ImportError, AttributeError, ValueError, OSError, RuntimeError, TypeError) as exc:
        # The floor below is a 64GB host pretending to be 8GB, which changes
        # every ceiling computed from it.
        logger.warning("host RAM unreadable (%s: %s); using the 8GB floor", type(exc).__name__, exc)
    return 8 * 1024**3


def _host_probe_output(
    command: list[str],
    *,
    source: str,
    broker_stdout_path: Path | None,
) -> str:
    """Run one host probe directly or through the detached exact-command broker."""

    if broker_stdout_path is None:
        from core.runtime.subprocess_gateway import get_subprocess_gateway

        return get_subprocess_gateway().run(
            command,
            timeout=5,
            check=True,
            read_only=True,
            source=source,
            accelerator_capability="none",
        ).stdout

    from core.runtime.detached_subprocess_broker import (
        broker_available,
        run_brokered_process,
    )

    path = broker_stdout_path.expanduser()
    if (
        not path.is_absolute()
        or path.exists()
        or not path.parent.is_dir()
        or not broker_available()
    ):
        raise RuntimeError("detached host-pressure broker evidence is unavailable")
    result = run_brokered_process(
        command,
        cwd=Path.cwd(),
        stdout_path=path,
        timeout_s=5.0,
    )
    if (
        result.returncode != 0
        or result.status != "passed"
        or result.containment_verified is not True
    ):
        raise RuntimeError("detached host-pressure probe failed")
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise RuntimeError("detached host-pressure evidence is unreadable") from exc


def host_pressure(
    *,
    broker_vm_stat_path: Path | None = None,
    broker_swapusage_path: Path | None = None,
) -> dict[str, Any]:
    """Real memory pressure, not the misleading 'Pages free' number.

    On macOS, ``Pages free`` excludes inactive/purgeable pages that the
    kernel will reclaim on demand, so a healthy host can report ~1 GB free
    while 65% of RAM is actually available. Deciding to abort a run on that
    number is a false alarm; the signals that actually preceded this host's
    jetsam kill were SWAP and COMPRESSOR growth.
    """
    broker_values = (broker_vm_stat_path, broker_swapusage_path)
    if any(value is not None for value in broker_values) != all(
        value is not None for value in broker_values
    ):
        return {"available": False, "pressure_reasons": ["broker_contract_incomplete"]}

    stats: dict[str, int] = {}
    try:
        output = _host_probe_output(
            ["vm_stat"],
            source="mlx_memory_guard.host_pressure.vm_stat",
            broker_stdout_path=broker_vm_stat_path,
        )
    except (OSError, RuntimeError, ValueError):
        return {"available": False}
    page_size = 16384
    for line in output.splitlines():
        if "page size of" in line:
            for token in line.split():
                if token.isdigit():
                    page_size = int(token)
                    break
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        digits = value.strip().rstrip(".")
        if digits.isdigit():
            stats[key.strip()] = int(digits)

    def gb(name: str) -> float:
        return stats.get(name, 0) * page_size / 1024**3

    free = gb("Pages free")
    inactive = gb("Pages inactive")
    speculative = gb("Pages speculative")
    purgeable = gb("Pages purgeable")
    compressed = gb("Pages occupied by compressor")
    host = host_memory_bytes() / 1024**3
    reclaimable = free + inactive + speculative + purgeable
    swap_used: float | None = None
    try:
        swap = _host_probe_output(
            ["sysctl", "-n", "vm.swapusage"],
            source="mlx_memory_guard.host_pressure.swapusage",
            broker_stdout_path=broker_swapusage_path,
        )
        swap_used = _parse_swap_used_gb(swap)
    except (OSError, RuntimeError, ValueError) as exc:
        # swap_used stays None, which the pressure reasons below read as
        # "no swap in use" rather than "we could not look".
        logger.debug(
            "swap usage unreadable (%s: %s); pressure is judged without it", type(exc).__name__, exc
        )
    reasons = _pressure_reasons(
        host_gb=host,
        reclaimable_gb=reclaimable,
        compressed_gb=compressed,
        swap_used_gb=swap_used,
    )
    return {
        "available": True,
        "host_gb": round(host, 2),
        "free_gb": round(free, 2),
        "reclaimable_gb": round(reclaimable, 2),
        "available_fraction": round(reclaimable / max(host, 1e-9), 3),
        "compressed_gb": round(compressed, 2),
        "swap_available": swap_used is not None,
        "swap_used_gb": None if swap_used is None else round(swap_used, 2),
        "pressure_reasons": list(reasons),
        "under_pressure": bool(reasons),
    }


@dataclass
class MemoryEnvelope:
    """Applied limits plus the reclaim hook for long loops."""

    memory_bytes: int
    cache_bytes: int
    wired_bytes: int
    reclaim_every: int = DEFAULT_RECLAIM_EVERY
    requested_wired_bytes: int | None = None
    device_wired_cap_bytes: int | None = None

    def reclaim(self, step: int | None = None, *, force: bool = False) -> bool:
        """Release MLX's buffer cache. Call inside generation/eval loops.

        Returns True when a reclaim actually ran, so callers can receipt it.
        """
        if not force and step is not None and self.reclaim_every > 0:
            if step % self.reclaim_every != 0:
                return False
        try:
            _synchronize_and_reclaim()
            return True
        except (ImportError, RuntimeError) as exc:
            # False here is receipted as "no reclaim ran", and a reclaim that
            # RAISED is the case a caller under memory pressure needs named.
            logger.warning("MLX buffer cache reclaim failed (%s: %s)", type(exc).__name__, exc)
            return False

    def to_receipt(self) -> dict[str, Any]:
        return {
            "schema": MLX_MEMORY_GUARD_SCHEMA,
            "memory_limit_gb": round(self.memory_bytes / 1024**3, 3),
            "cache_limit_gb": round(self.cache_bytes / 1024**3, 3),
            "wired_limit_gb": round(self.wired_bytes / 1024**3, 3),
            "requested_wired_limit_gb": (
                None if self.requested_wired_bytes is None
                else round(self.requested_wired_bytes / 1024**3, 3)
            ),
            "device_wired_cap_gb": (
                None if self.device_wired_cap_bytes is None
                else round(self.device_wired_cap_bytes / 1024**3, 3)
            ),
            "reclaim_every": self.reclaim_every,
            "host_memory_gb": round(host_memory_bytes() / 1024**3, 3),
        }


def _resolve_bytes(
    value: float | None,
    *,
    host: int,
    fraction: float,
    floor: int = 0,
) -> int:
    """Resolve one limit in bytes, validated against the real host.

    ``floor`` applies only to the working-memory limit: a small CACHE
    limit is a legitimate choice (it just means more frequent reclaim),
    whereas a tiny working limit cannot load a model at all.
    """
    if value is None:
        return max(floor, int(host * fraction))
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("memory limits must be numeric gigabytes or None")
    resolved = int(float(value) * 1024**3)
    if resolved <= 0:
        raise ValueError("memory limits must be positive")
    if floor and resolved < floor:
        raise ValueError("memory limit below the 2 GiB floor is unusable")
    if resolved > host:
        raise ValueError(
            "memory limit exceeds physical RAM; the host would be swapped "
            "to death rather than the run failing"
        )
    return resolved


@contextmanager
def mlx_memory_envelope(
    *,
    fraction: float = DEFAULT_FRACTION,
    memory_gb: float | None = None,
    cache_gb: float | None = 2.0,
    wired_gb: float | None = None,
    reclaim_every: int = DEFAULT_RECLAIM_EVERY,
    restore_limits_on_exit: bool = True,
) -> Iterator[MemoryEnvelope]:
    """Bound MLX memory for the duration of the block.

    ``fraction`` sizes the default limit from real host RAM. Explicit
    gigabyte values override it and are validated against the host, so a
    typo cannot silently authorize an unbounded run.
    """
    if not isinstance(restore_limits_on_exit, bool):
        raise ValueError("restore_limits_on_exit must be boolean")
    if not 0.05 <= float(fraction) <= 0.9:
        raise ValueError("fraction must be inside [0.05, 0.9]")
    host = host_memory_bytes()
    memory_bytes = _resolve_bytes(
        memory_gb, host=host, fraction=fraction, floor=MIN_LIMIT_BYTES
    )
    cache_bytes = _resolve_bytes(
        cache_gb, host=host, fraction=min(fraction, 0.05)
    )
    wired_bytes = _resolve_bytes(
        wired_gb, host=host, fraction=min(fraction + 0.15, 0.85)
    )
    if cache_bytes > memory_bytes:
        raise ValueError("cache limit cannot exceed the memory limit")

    import mlx.core as mx

    device_cap = mx.device_info().get("max_recommended_working_set_size")
    if type(device_cap) is not int or device_cap <= 0:
        raise ValueError("MLX device wired-memory capacity is unavailable")
    device_cap = min(device_cap, host - 1)
    requested_wired_bytes = wired_bytes
    if wired_gb is not None and wired_bytes > device_cap:
        raise ValueError("wired limit exceeds the MLX device working-set capacity")
    wired_bytes = min(wired_bytes, device_cap)
    setters = (
        ("memory", mx.set_memory_limit, memory_bytes),
        ("cache", mx.set_cache_limit, cache_bytes),
        ("wired", mx.set_wired_limit, wired_bytes),
    )
    applied = []

    def restore_applied() -> None:
        # Restore every successful setter even if another restoration fails.
        for name, setter, previous in reversed(applied):
            try:
                setter(previous)
            except (RuntimeError, ValueError) as exc:
                logger.warning("Could not restore MLX %s limit: %s", name, exc)

    try:
        for name, setter, value in setters:
            applied.append((name, setter, setter(value)))
    except BaseException:
        restore_applied()
        raise
    envelope = MemoryEnvelope(
        memory_bytes=memory_bytes,
        cache_bytes=cache_bytes,
        wired_bytes=wired_bytes,
        reclaim_every=reclaim_every,
        requested_wired_bytes=requested_wired_bytes,
        device_wired_cap_bytes=device_cap,
    )
    logger.info("MLX memory envelope applied: %s", envelope.to_receipt())
    try:
        yield envelope
    finally:
        try:
            if restore_limits_on_exit:
                _synchronize_and_reclaim()
            else:
                # A process-isolated model owner can let process teardown
                # reclaim Metal buffers. Avoid exercising allocator cache
                # reclamation immediately before the process exits.
                mx.synchronize()
        except (RuntimeError, ValueError) as exc:  # pragma: no cover
            logger.warning("Could not synchronize MLX envelope cleanup: %s", exc)
        finally:
            if restore_limits_on_exit:
                restore_applied()


__all__ = [
    "DEFAULT_FRACTION",
    "MLX_MEMORY_GUARD_SCHEMA",
    "MemoryEnvelope",
    "_synchronize_and_reclaim",
    "host_memory_bytes",
    "host_pressure",
    "mlx_memory_envelope",
]
