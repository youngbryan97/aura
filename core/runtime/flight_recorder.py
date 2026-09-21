"""core/runtime/flight_recorder.py — black-box flight recorder (roadmap A5).

Aerospace keeps a crash-survivable record of the final moments; Aura now does
too. An always-on, fixed-size, mmap-backed ring receives one compact
"mind-moment" frame per cognitive tick — wall/monotonic time, tick id, stage,
mode, RSS, tick duration, consecutive loop failures, unhealthy typed
conditions (K6), degradation count. Appends are a single 512-byte memcpy into
a shared mapping: no syscalls, no fsync, nothing on the event loop. Because
the mapping is MAP_SHARED, the kernel owns the pages the instant they are
written — the ring survives SIGKILL, OOM-kill, segfaults, every process death
short of the whole machine going down.

On boot, the previous ring is inspected. A clean-shutdown marker present means
a graceful exit; absent means Aura died hard — the last recorded moments are
extracted into a governed death report under ``data/error_logs/flight/``,
where the incident narrator picks it up ("what happened?" gets receipts, not
confabulation) and the continuity waking sequence gets a grounded note about
how the previous life ended. The forensic record of a death is written by the
death itself, not reconstructed from inference afterwards.

Honesty properties: every frame carries a CRC so torn writes are skipped, not
misread; an unreadable previous ring is reported as unreadable; a clean
shutdown produces no death report at all.
"""
from __future__ import annotations

import asyncio
import json
import logging
import mmap
import os
import struct
import time
import uuid
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.governance_context import local_internal_governed_scope
from core.runtime.errors import record_degradation
from core.runtime.file_write_gateway import get_file_write_gateway
from core.runtime.flags import FlagKind, declare

logger = logging.getLogger("Aura.FlightRecorder")

_MAGIC = b"AURAFDR1"
_VERSION = 1
_HEADER_SIZE = 4096
_SLOT_SIZE = 512

# magic, version, slot_size, slot_count, boot_id, boot_wall, pid,
# clean, closed_wall, close_reason
_HEADER_FMT = "<8s3I32sdIId64s"
_HEADER_LEN = struct.calcsize(_HEADER_FMT)

# used_len, seq, wall_ts, mono_ts, tick, rss_mb, tick_duration_ms,
# consecutive_failures, degradation_count  (crc32 prefixes the packed body)
_SLOT_BODY_FMT = "<IQddQffII"
_SLOT_BODY_LEN = struct.calcsize(_SLOT_BODY_FMT)
_SLOT_PREFIX_LEN = 4 + _SLOT_BODY_LEN
_PAYLOAD_CAP = _SLOT_SIZE - _SLOT_PREFIX_LEN

_DEFAULT_SLOT_COUNT = 4096
def _default_flight_dir() -> Path:
    """Resolved per construction through the shared forensics root.

    A module-level relative Path baked in whatever directory happened to be
    current at import, which put death reports somewhere the readers of those
    reports did not look.
    """
    from core.utils.paths import forensics_dir

    return Path(forensics_dir("flight"))
_RING_NAME = "flight_ring.bin"
_PREV_RING_NAME = "flight_ring.prev"
_LOCK_NAME = "flight_ring.lock"
_DEATH_ARTIFACT_GLOB = "death_*.json"
_MAX_DEATH_ARTIFACTS = 20
# Refresh the expensive frame fields (conditions digest, degradation count)
# every Nth frame; the cached values ride along in between.
_SLOW_FIELD_PERIOD = 5
# A death artifact this old is a previous life's news, not this waking's.
_WAKING_NOTE_MAX_AGE_S = 900.0
_DEGRADATION_THROTTLE_S = 60.0

_ENABLED_FLAG = declare(
    "AURA_FLIGHT_RECORDER",
    kind=FlagKind.BOOL,
    default=True,
    description="Black-box flight recorder: crash-survivable ring of per-tick mind-moments.",
    owner="core.runtime.flight_recorder",
)
_SLOTS_FLAG = declare(
    "AURA_FLIGHT_RECORDER_SLOTS",
    kind=FlagKind.INT,
    default=_DEFAULT_SLOT_COUNT,
    description="Flight recorder ring capacity in frames (512 bytes each).",
    owner="core.runtime.flight_recorder",
)


@dataclass(frozen=True)
class RecordedFrame:
    """One mind-moment recovered from a ring."""

    seq: int
    wall_ts: float
    mono_ts: float
    tick: int
    rss_mb: float
    tick_duration_ms: float
    consecutive_failures: int
    degradation_count: int
    payload: dict[str, Any] = field(default_factory=dict)

    def compact(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "at": self.wall_ts,
            "tick": self.tick,
            "stage": self.payload.get("stage", ""),
            "mode": self.payload.get("mode", ""),
            "rss_mb": round(self.rss_mb, 1),
            "dur_ms": round(self.tick_duration_ms, 1),
            "fails": self.consecutive_failures,
        }


@dataclass(frozen=True)
class RingInspection:
    """Everything a previous ring file has to say."""

    readable: bool
    clean: bool
    boot_id: str
    boot_started_at: float
    pid: int
    close_reason: str
    closed_at: float
    frames: list[RecordedFrame]


def inspect_ring_file(path: Path, *, last_n: int = 240) -> RingInspection | None:
    """Read a ring file without mapping it. Returns None if absent."""
    try:
        raw = Path(path).read_bytes()
    except FileNotFoundError:
        # not a failure: the docstring says None means absent, and this is
        # the read that finds out.
        return None
    except OSError as exc:
        record_degradation(
            "flight_recorder", exc, action="treated previous ring as unreadable"
        )
        return _unreadable_inspection()
    if len(raw) < _HEADER_SIZE:
        return _unreadable_inspection()
    try:
        (
            magic,
            version,
            slot_size,
            slot_count,
            boot_id_raw,
            boot_wall,
            pid,
            clean,
            closed_wall,
            reason_raw,
        ) = struct.unpack(_HEADER_FMT, raw[:_HEADER_LEN])
    except struct.error:
        return _unreadable_inspection()
    if magic != _MAGIC or version != _VERSION or slot_size < _SLOT_PREFIX_LEN:
        return _unreadable_inspection()
    frames: list[RecordedFrame] = []
    for index in range(slot_count):
        offset = _HEADER_SIZE + index * slot_size
        slot = raw[offset : offset + slot_size]
        if len(slot) < slot_size:
            break
        frame = _decode_slot(slot, slot_size)
        if frame is not None:
            frames.append(frame)
    frames.sort(key=lambda item: item.seq)
    return RingInspection(
        readable=True,
        clean=bool(clean),
        boot_id=boot_id_raw.rstrip(b"\x00").decode("ascii", "replace"),
        boot_started_at=float(boot_wall),
        pid=int(pid),
        close_reason=reason_raw.rstrip(b"\x00").decode("utf-8", "replace"),
        closed_at=float(closed_wall),
        frames=frames[-last_n:],
    )


def _unreadable_inspection() -> RingInspection:
    return RingInspection(
        readable=False,
        clean=False,
        boot_id="",
        boot_started_at=0.0,
        pid=0,
        close_reason="",
        closed_at=0.0,
        frames=[],
    )


def _decode_slot(slot: bytes, slot_size: int) -> RecordedFrame | None:
    (crc,) = struct.unpack_from("<I", slot, 0)
    try:
        (
            used_len,
            seq,
            wall_ts,
            mono_ts,
            tick,
            rss_mb,
            duration_ms,
            failures,
            degradations,
        ) = struct.unpack_from(_SLOT_BODY_FMT, slot, 4)
    except struct.error:
        # not a failure: a slot shorter than a record is one that was never
        # written, and the caller skips it like any empty slot.
        return None
    if seq == 0 or used_len > slot_size - _SLOT_PREFIX_LEN:
        return None
    body = slot[4 : _SLOT_PREFIX_LEN + used_len]
    if zlib.crc32(body) != crc:
        return None
    raw_payload = slot[_SLOT_PREFIX_LEN : _SLOT_PREFIX_LEN + used_len]
    try:
        payload = json.loads(raw_payload.decode("utf-8"))
        if not isinstance(payload, dict):
            payload = {"raw": payload}
    except (ValueError, UnicodeDecodeError):
        payload = {"raw": raw_payload.decode("utf-8", "replace")}
    return RecordedFrame(
        seq=int(seq),
        wall_ts=float(wall_ts),
        mono_ts=float(mono_ts),
        tick=int(tick),
        rss_mb=float(rss_mb),
        tick_duration_ms=float(duration_ms),
        consecutive_failures=int(failures),
        degradation_count=int(degradations),
        payload=payload,
    )


class FlightRecorder:
    """Crash-survivable ring of mind-moments plus its post-mortem reader."""

    def __init__(
        self,
        flight_dir: Path | str | None = None,
        *,
        slot_count: int | None = None,
    ) -> None:
        self._flight_dir = (
            Path(flight_dir) if flight_dir is not None else _default_flight_dir()
        )
        self._ring_path = self._flight_dir / _RING_NAME
        self._prev_ring_path = self._flight_dir / _PREV_RING_NAME
        requested = slot_count if slot_count is not None else int(_SLOTS_FLAG.value())
        self._slot_count = max(16, int(requested))
        self._file: Any = None
        self._lock_file: Any = None
        self._mm: mmap.mmap | None = None
        self._started = False
        self._boot_id = uuid.uuid4().hex
        self._boot_wall = 0.0
        self._next_seq = 1
        self._frames_written = 0
        self._last_death_report: dict[str, Any] | None = None
        self._cached_conditions: dict[str, str] = {}
        self._cached_degradations = 0
        self._last_error_at = 0.0
        self._last_tick = 0

    # ── lifecycle ──────────────────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        return bool(_ENABLED_FLAG.value())

    @property
    def started(self) -> bool:
        return self._started

    @property
    def ring_path(self) -> Path:
        return self._ring_path

    def start_sync(self) -> dict[str, Any] | None:
        """Inspect the previous ring, open a fresh one, return a death report
        if the previous run ended without a clean shutdown. Blocking: call
        off the event loop (``start()`` does)."""
        if self._started:
            return self._last_death_report
        if not self.enabled:
            logger.info("Flight recorder disabled by AURA_FLIGHT_RECORDER.")
            return None
        gateway = get_file_write_gateway()
        with local_internal_governed_scope(
            "flight_recorder.ring_lifecycle",
            domain="file_write",
        ):
            gateway.ensure_directory(
                self._flight_dir,
                source="core.runtime.flight_recorder.start",
            )
            # The runtime lock is taken BEFORE the previous ring is even read:
            # a second live runtime (the duplicate-runtime cascade) must not be
            # able to rotate the ring the first one is still writing.
            self._acquire_runtime_lock()
            rotated_previous = False
            try:
                previous = inspect_ring_file(self._ring_path)
                death_report: dict[str, Any] | None = None
                if previous is not None and not previous.clean:
                    death_report = self._build_death_report(previous)
                if previous is not None:
                    try:
                        gateway.replace_file(
                            self._ring_path,
                            self._prev_ring_path,
                            source="core.runtime.flight_recorder.rotate",
                        )
                        rotated_previous = True
                    except OSError as exc:
                        record_degradation(
                            "flight_recorder",
                            exc,
                            action="preserved previous ring and refused destructive overwrite",
                        )
                        raise RuntimeError(
                            "could not rotate previous flight ring without data loss"
                        ) from exc
                self._open_fresh_ring()
            except (OSError, RuntimeError, TypeError, ValueError, struct.error):
                if rotated_previous and self._prev_ring_path.exists():
                    try:
                        gateway.replace_file(
                            self._prev_ring_path,
                            self._ring_path,
                            source="core.runtime.flight_recorder.restore_failed_start",
                        )
                    except (OSError, RuntimeError, TypeError, ValueError) as restore_exc:
                        record_degradation(
                            "flight_recorder",
                            restore_exc,
                            action="previous ring remained at flight_ring.prev after failed start",
                            severity="warning",
                        )
                self.close()
                raise
        self._last_death_report = death_report
        self._started = True
        logger.info(
            "🛬 Flight recorder ONLINE — ring %s (%d slots, %d KB)%s",
            self._ring_path,
            self._slot_count,
            (self._slot_count * _SLOT_SIZE + _HEADER_SIZE) // 1024,
            "; previous run died uncleanly" if death_report else "",
        )
        return death_report

    async def start(self) -> dict[str, Any] | None:
        """Boot entry point: open the ring off-loop, publish any death report
        through the governed async write lane, register the service."""
        death_report = await asyncio.to_thread(self.start_sync)
        if death_report is not None:
            try:
                await self._publish_death_artifact(death_report)
            except (OSError, RuntimeError, ValueError, TypeError) as exc:
                record_degradation(
                    "flight_recorder",
                    exc,
                    action="kept death report in memory; artifact write failed",
                    severity="warning",
                )
        try:
            from core.container import ServiceContainer

            ServiceContainer.register_instance("flight_recorder", self, required=False)
        except (ImportError, AttributeError, RuntimeError, TypeError) as exc:
            record_degradation(
                "flight_recorder",
                exc,
                action="continued without service registration",
                severity="debug",
            )
        return death_report

    def _acquire_runtime_lock(self) -> None:
        if self._lock_file is not None:
            return
        import fcntl

        lock_path = self._flight_dir / _LOCK_NAME
        lock_file = get_file_write_gateway().open_owned_binary(
            lock_path,
            mode="a+b",
            permissions=0o600,
            source="core.runtime.flight_recorder.runtime_lock",
        )
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            lock_file.close()
            raise RuntimeError(
                f"flight ring already locked by another runtime: {lock_path}"
            ) from exc
        self._lock_file = lock_file

    def _open_fresh_ring(self) -> None:
        total = _HEADER_SIZE + self._slot_count * _SLOT_SIZE
        self._boot_wall = time.time()
        file = get_file_write_gateway().open_owned_binary(
            self._ring_path,
            mode="w+b",
            permissions=0o600,
            source="core.runtime.flight_recorder.open_ring",
        )
        try:
            file.truncate(total)
            file.seek(0)
            file.write(self._pack_header(clean=False, closed_wall=0.0, reason=""))
            file.flush()
            os.fsync(file.fileno())
            self._mm = mmap.mmap(file.fileno(), total, access=mmap.ACCESS_WRITE)
            self._file = file
        except (OSError, ValueError):
            if self._mm is not None:
                self._mm.close()
                self._mm = None
            if not file.closed:
                file.close()
            raise
        self._next_seq = 1
        self._frames_written = 0

    def _pack_header(self, *, clean: bool, closed_wall: float, reason: str) -> bytes:
        packed = struct.pack(
            _HEADER_FMT,
            _MAGIC,
            _VERSION,
            _SLOT_SIZE,
            self._slot_count,
            self._boot_id.encode("ascii")[:32].ljust(32, b"\x00"),
            self._boot_wall,
            os.getpid(),
            1 if clean else 0,
            closed_wall,
            reason.encode("utf-8", "ignore")[:64].ljust(64, b"\x00"),
        )
        return packed.ljust(_HEADER_SIZE, b"\x00")

    def mark_clean_shutdown(self, reason: str = "graceful") -> bool:
        """Stamp the header so the next boot knows this was not a death."""
        if self._mm is None:
            return False
        try:
            self._mm[0:_HEADER_SIZE] = self._pack_header(
                clean=True, closed_wall=time.time(), reason=str(reason)[:64]
            )
            self._mm.flush()
            return True
        except (OSError, ValueError) as exc:
            record_degradation(
                "flight_recorder", exc, action="clean-shutdown marker not written"
            )
            return False

    def close(self) -> None:
        if self._mm is not None:
            try:
                self._mm.flush()
                self._mm.close()
            except (OSError, ValueError) as exc:
                record_degradation(
                    "flight_recorder", exc, action="ring closed without final flush",
                    severity="debug",
                )
            self._mm = None
        if self._file is not None:
            try:
                self._file.close()
            except OSError as exc:
                # The handle is dropped below either way, so a close that
                # refuses is how the ring's descriptor leaks.
                logger.warning("Flight recorder file would not close: %s", exc)
            self._file = None
        if self._lock_file is not None:
            try:
                self._lock_file.close()
            except OSError as exc:
                logger.warning("Flight recorder lock would not close: %s", exc)
            self._lock_file = None
        try:
            from core.container import ServiceContainer

            ServiceContainer.unregister_instance(
                "flight_recorder",
                expected_instance=self,
            )
        except (ImportError, AttributeError, RuntimeError, TypeError) as exc:
            record_degradation(
                "flight_recorder",
                exc,
                action="closed ring but could not release service registration",
                severity="debug",
            )
        self._started = False

    def on_stop(self) -> None:
        """Seal the ring while its mmap still belongs to the live service."""

        if self._started:
            self.mark_clean_shutdown("container_shutdown")
        self.close()

    # ── recording ──────────────────────────────────────────────────────

    def record_frame(
        self,
        *,
        tick: int,
        stage: str = "",
        mode: str = "",
        tick_duration_ms: float = 0.0,
        consecutive_failures: int = 0,
        extra: dict[str, Any] | None = None,
        refresh_slow_fields: bool = True,
    ) -> bool:
        """Append one mind-moment. One bounded memcpy into the shared
        mapping — safe from the tick loop, never raises."""
        if self._mm is None or not self._started:
            return False
        try:
            if refresh_slow_fields and self._frames_written % _SLOW_FIELD_PERIOD == 0:
                self._refresh_slow_fields()
            payload: dict[str, Any] = {
                "stage": str(stage)[:80],
                "mode": str(mode)[:32],
            }
            if self._cached_conditions:
                payload["cond"] = self._cached_conditions
            if extra:
                payload["extra"] = extra
            encoded = self._bounded_payload(payload)
            seq = self._next_seq
            body = struct.pack(
                _SLOT_BODY_FMT,
                len(encoded),
                seq,
                time.time(),
                time.monotonic(),
                int(tick),
                float(self._sample_rss_mb()),
                float(tick_duration_ms),
                int(consecutive_failures),
                int(self._cached_degradations),
            ) + encoded
            slot = struct.pack("<I", zlib.crc32(body)) + body
            slot = slot.ljust(_SLOT_SIZE, b"\x00")
            offset = _HEADER_SIZE + ((seq - 1) % self._slot_count) * _SLOT_SIZE
            self._mm[offset : offset + _SLOT_SIZE] = slot
            self._next_seq = seq + 1
            self._frames_written += 1
            if tick > 0:
                self._last_tick = int(tick)
            return True
        except (OSError, ValueError, TypeError, struct.error) as exc:
            now = time.monotonic()
            if now - self._last_error_at >= _DEGRADATION_THROTTLE_S:
                self._last_error_at = now
                record_degradation(
                    "flight_recorder", exc, action="dropped one mind-moment frame"
                )
            return False

    def record_event(
        self,
        *,
        kind: str,
        source: str,
        summary: str,
        lane: str = "",
    ) -> bool:
        """Append one event-moment (degradation, condition flip, reconciler
        action) into the same crash-survivable ring as the tick frames. Rides
        on the last-known tick so post-mortem ordering stays meaningful, and
        NEVER refreshes the slow fields — event feeds fire from inside other
        subsystems' locks (conditions.set, record_degradation), and a refresh
        re-enters those same subsystems (all_conditions_report / the
        degradation tracker). Pure memcpy, no re-entrancy, never raises."""
        extra: dict[str, Any] = {"src": str(source)[:48], "sum": str(summary)[:160]}
        if lane:
            extra["lane"] = str(lane)[:32]
        return self.record_frame(
            tick=self._last_tick,
            stage=f"event:{kind}"[:80],
            mode="event",
            extra=extra,
            refresh_slow_fields=False,
        )

    @staticmethod
    def _bounded_payload(payload: dict[str, Any]) -> bytes:
        encoded = json.dumps(payload, separators=(",", ":"), default=str).encode(
            "utf-8", "ignore"
        )
        if len(encoded) <= _PAYLOAD_CAP:
            return encoded
        for key in ("extra", "cond"):
            payload.pop(key, None)
            encoded = json.dumps(payload, separators=(",", ":"), default=str).encode(
                "utf-8", "ignore"
            )
            if len(encoded) <= _PAYLOAD_CAP:
                return encoded
        return json.dumps(
            {"stage": str(payload.get("stage", ""))[:64]}, separators=(",", ":")
        ).encode("utf-8", "ignore")[:_PAYLOAD_CAP]

    def _sample_rss_mb(self) -> float:
        """This process's RSS, and nothing that costs a process-table scan.

        Every frame calls this, including the event frames that record_event
        promises are pure memcpy — and record_degradation feeds those from the
        event loop. The default observation walks the process tree, which
        enumerates every pid on the host, so recording a degradation blocked
        the loop long enough to cause the next one: on 2026-07-29 a 5.2s lag
        became a 63.5s freeze that way, each degradation buying the next.
        Only process_rss_bytes is read here, so only that is asked for.
        """
        try:
            from core.runtime.resource_observation import get_resource_observer

            memory = get_resource_observer().memory(include_process_tree=False)
            return float(memory.process_rss_bytes) / (1024.0 * 1024.0)
        except (ImportError, OSError, RuntimeError, AttributeError):
            # not a failure: an RSS this cannot read is written as zero, and
            # a zero in the ring says the sample was not taken.
            return 0.0

    def _refresh_slow_fields(self) -> None:
        try:
            from core.runtime.conditions import all_conditions_report

            unhealthy: dict[str, str] = {}
            for component, conditions in all_conditions_report().items():
                for kind, condition in conditions.items():
                    status = bool(condition.get("status"))
                    if (kind == "Ready" and not status) or (
                        kind == "Degraded" and status
                    ):
                        reason = str(condition.get("reason", ""))[:48]
                        unhealthy[component] = f"{kind}={status}:{reason}"
            self._cached_conditions = unhealthy
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            self._cached_conditions = {}
        try:
            from core.runtime.errors import get_degradation_tracker

            self._cached_degradations = int(
                get_degradation_tracker().status().get("total_degradations", 0)
            )
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            # not a failure: no tracker to count, and the ring records the
            # zero as it would any unread sample.
            self._cached_degradations = 0

    # ── post-mortem ────────────────────────────────────────────────────

    def _build_death_report(self, previous: RingInspection) -> dict[str, Any]:
        frames = previous.frames
        last = frames[-1] if frames else None
        died_at = last.wall_ts if last else None
        uptime_s = (
            max(0.0, died_at - previous.boot_started_at)
            if died_at and previous.boot_started_at
            else None
        )
        rss_delta = None
        if last is not None:
            final_window = [
                frame for frame in frames if last.wall_ts - frame.wall_ts <= 60.0
            ]
            if len(final_window) >= 2:
                rss_delta = round(final_window[-1].rss_mb - final_window[0].rss_mb, 1)
        report: dict[str, Any] = {
            "schema": "aura.flight_recorder.death.v1",
            "generated_at": time.time(),
            "ring_readable": previous.readable,
            "previous_boot_id": previous.boot_id,
            "previous_boot_started_at": previous.boot_started_at or None,
            "previous_pid": previous.pid or None,
            "died_at": died_at,
            "uptime_s": round(uptime_s, 1) if uptime_s is not None else None,
            "frames_recovered": len(frames),
            "final_tick": last.tick if last else None,
            "final_stage": last.payload.get("stage", "") if last else "",
            "final_mode": last.payload.get("mode", "") if last else "",
            "final_rss_mb": round(last.rss_mb, 1) if last else None,
            "rss_delta_final_minute_mb": rss_delta,
            "max_consecutive_failures": max(
                (frame.consecutive_failures for frame in frames), default=0
            ),
            "unhealthy_conditions_at_end": (
                dict(last.payload.get("cond", {})) if last else {}
            ),
            "degradation_count_at_end": last.degradation_count if last else 0,
            "last_frames": [frame.compact() for frame in frames[-24:]],
        }
        report["narrative"] = self._compose_death_narrative(report)
        return report

    @staticmethod
    def _compose_death_narrative(report: dict[str, Any]) -> str:
        if not report.get("ring_readable", False):
            return (
                "The previous run ended without a clean shutdown and its flight "
                "ring was unreadable — the death is on record, its final moments "
                "are not."
            )
        died_at = report.get("died_at")
        if not died_at:
            return (
                "The previous run ended without a clean shutdown before its "
                "first mind-moment was recorded — it died during boot."
            )
        when = time.strftime("%H:%M:%S", time.localtime(float(died_at)))
        parts = [
            f"The previous run went down hard; its last recorded moment was {when}"
        ]
        uptime_s = report.get("uptime_s")
        if uptime_s:
            parts.append(f"after {float(uptime_s) / 3600.0:.1f}h alive")
        stage = report.get("final_stage")
        tick = report.get("final_tick")
        if stage:
            parts.append(f"in stage '{stage}' (tick {tick})")
        rss = report.get("final_rss_mb")
        if rss:
            climb = report.get("rss_delta_final_minute_mb")
            if climb and float(climb) > 50.0:
                parts.append(f"with RSS {rss:.0f} MB and climbing (+{climb:.0f} MB over the final minute)")
            else:
                parts.append(f"with RSS {rss:.0f} MB")
        sentence = ", ".join(parts) + "."
        unhealthy = report.get("unhealthy_conditions_at_end") or {}
        if unhealthy:
            listed = "; ".join(
                f"{component}: {state}" for component, state in sorted(unhealthy.items())[:4]
            )
            sentence += f" Unhealthy at the end — {listed}."
        failures = int(report.get("max_consecutive_failures") or 0)
        if failures:
            sentence += f" The tick loop had logged {failures} consecutive failure(s)."
        return sentence

    async def _publish_death_artifact(self, report: dict[str, Any]) -> None:
        died_at = report.get("died_at") or report.get("generated_at") or time.time()
        artifact_path = self._flight_dir / f"death_{int(float(died_at))}.json"
        report["artifact_path"] = str(artifact_path)
        gateway = get_file_write_gateway()
        with local_internal_governed_scope(
            "flight_recorder.postmortem", domain="file_write"
        ):
            await gateway.write_text_async(
                artifact_path,
                json.dumps(report, indent=2, default=str),
                source="flight_recorder.postmortem",
            )
            for stale in sorted(self._flight_dir.glob(_DEATH_ARTIFACT_GLOB))[
                :-_MAX_DEATH_ARTIFACTS
            ]:
                await gateway.delete_path_async(
                    stale, source="flight_recorder.postmortem_prune"
                )
        logger.warning("🛬 Death report published: %s", artifact_path)

    # ── consumers ──────────────────────────────────────────────────────

    def get_last_death_report(self) -> dict[str, Any] | None:
        return self._last_death_report

    def waking_note(self) -> str:
        """A grounded sentence about how the previous life ended, for the
        continuity waking sequence. Empty when the last shutdown was clean
        or the death is stale news."""
        report = self._last_death_report
        if report is None:
            report = self._newest_recent_artifact()
        if not report:
            return ""
        narrative = str(report.get("narrative", "")).strip()
        if not narrative:
            return ""
        return f"Black-box record of the gap: {narrative}"

    def _newest_recent_artifact(self) -> dict[str, Any] | None:
        try:
            candidates = sorted(
                self._flight_dir.glob(_DEATH_ARTIFACT_GLOB),
                key=lambda item: item.stat().st_mtime,
            )
        except OSError as exc:
            # This is the death-artifact reader. A directory it cannot walk
            # is a crash report nobody will ever see.
            logger.warning("Could not list %s for death artifacts: %s", self._flight_dir, exc)
            return None
        if not candidates:
            return None
        newest = candidates[-1]
        try:
            report = json.loads(newest.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            # The newest death artifact exists and will not open, which is
            # the one case where returning None hides a real report.
            logger.warning("Death artifact %s will not parse: %s", newest, exc)
            return None
        generated = float(report.get("generated_at") or 0.0)
        if time.time() - generated > _WAKING_NOTE_MAX_AGE_S:
            return None
        return report if isinstance(report, dict) else None

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "started": self._started,
            "ring_path": str(self._ring_path),
            "slot_count": self._slot_count,
            "frames_written": self._frames_written,
            "last_death_at": (self._last_death_report or {}).get("died_at"),
        }


_recorder: FlightRecorder | None = None


def get_flight_recorder() -> FlightRecorder:
    global _recorder
    if _recorder is None:
        _recorder = FlightRecorder()
    return _recorder


def set_flight_recorder_for_test(recorder: FlightRecorder | None) -> None:
    global _recorder
    _recorder = recorder


def record_mind_moment(
    *,
    tick: int,
    stage: str = "",
    mode: str = "",
    tick_duration_ms: float = 0.0,
    consecutive_failures: int = 0,
    extra: dict[str, Any] | None = None,
) -> bool:
    """Tick-loop entry point. No-op (never instantiates, never raises)
    until a recorder has been started by boot or a test."""
    recorder = _recorder
    if recorder is None or not recorder.started:
        return False
    return recorder.record_frame(
        tick=tick,
        stage=stage,
        mode=mode,
        tick_duration_ms=tick_duration_ms,
        consecutive_failures=consecutive_failures,
        extra=extra,
    )


def record_event(
    *,
    kind: str,
    source: str,
    summary: str,
    lane: str = "",
) -> bool:
    """Event-feed entry point (degradations, condition flips, reconciler
    actions). No-op (never instantiates, never raises) until a recorder has
    been started by boot or a test."""
    recorder = _recorder
    if recorder is None or not recorder.started:
        return False
    return recorder.record_event(kind=kind, source=source, summary=summary, lane=lane)


__all__ = [
    "FlightRecorder",
    "RecordedFrame",
    "RingInspection",
    "get_flight_recorder",
    "inspect_ring_file",
    "record_event",
    "record_mind_moment",
    "set_flight_recorder_for_test",
]
