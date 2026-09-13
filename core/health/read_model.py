"""Bounded stale-while-revalidate snapshots for the public health surface."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger("Aura.Health.ReadModel")


@dataclass(frozen=True, slots=True)
class HealthReadModelConfig:
    refresh_interval_s: float = 5.0
    max_stale_s: float = 30.0
    collection_timeout_s: float = 8.0
    retry_base_s: float = 2.0
    retry_max_s: float = 30.0
    schema_version: str = "aura.health.snapshot.v1"
    metadata_key: str = "health_read_model"
    worker_name_prefix: str = "AuraHealthSnapshot"
    incident_prefix: str = "health-refresh"
    log_label: str = "Health snapshot"

    def normalized(self) -> HealthReadModelConfig:
        refresh = max(0.05, float(self.refresh_interval_s))
        retry_base = max(0.05, float(self.retry_base_s))
        return HealthReadModelConfig(
            refresh_interval_s=refresh,
            max_stale_s=max(refresh, float(self.max_stale_s)),
            collection_timeout_s=max(0.05, float(self.collection_timeout_s)),
            retry_base_s=retry_base,
            retry_max_s=max(retry_base, float(self.retry_max_s)),
            schema_version=str(self.schema_version or "aura.health.snapshot.v1"),
            metadata_key=str(self.metadata_key or "health_read_model"),
            worker_name_prefix=str(
                self.worker_name_prefix or "AuraHealthSnapshot"
            ),
            incident_prefix=str(self.incident_prefix or "health-refresh"),
            log_label=str(self.log_label or "Health snapshot"),
        )


class HealthSnapshotReadModel:
    """Serve immutable snapshots while one daemon worker refreshes them.

    HTTP callers never join the collector. A wedged dependency can therefore
    make the snapshot stale, but it cannot consume the health request budget or
    create a poll storm. The single worker is deliberately not replaced while
    stuck; that bounds abandoned work to one thread and makes the condition
    visible in metadata.
    """

    def __init__(
        self,
        collector: Callable[[], dict[str, Any]],
        fallback_factory: Callable[[], dict[str, Any]],
        *,
        config: HealthReadModelConfig | None = None,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = lambda: time.time(),
    ) -> None:
        self._collector = collector
        self._fallback_factory = fallback_factory
        self._config = (config or HealthReadModelConfig()).normalized()
        self._clock = clock
        self._wall_clock = wall_clock
        self._lock = threading.RLock()

        self._epoch = 0
        self._closed = False
        self._payload: dict[str, Any] | None = None
        self._captured_at_monotonic = 0.0
        self._captured_at_unix = 0.0
        self._generation = 0
        self._snapshot_generation = 0
        self._active_generation = 0
        self._active_started_at = 0.0
        self._active_thread: threading.Thread | None = None
        self._timed_out_generation = 0
        self._next_refresh_not_before = 0.0

        self._consecutive_failures = 0
        self._total_failures = 0
        self._total_timeouts = 0
        self._total_refreshes = 0
        self._last_error = ""
        self._last_error_at_unix = 0.0
        self._last_duration_s = 0.0
        self._incident_sequence = 0
        self._active_incident_id = ""
        self._last_recovery: dict[str, Any] | None = None

    @property
    def config(self) -> HealthReadModelConfig:
        return self._config

    def start(self) -> bool:
        with self._lock:
            self._closed = False
        return self.request_refresh(force=True)

    def close(self) -> None:
        """Stop admitting refreshes and invalidate any late worker result."""

        with self._lock:
            self._closed = True
            self._epoch += 1
            self._active_generation = 0
            self._active_started_at = 0.0
            # Retain a live worker reference so a subsequent lifespan cannot
            # admit another collector until the invalidated worker exits.
            if self._active_thread is not None and not self._active_thread.is_alive():
                self._active_thread = None

    def reset_for_test(self) -> None:
        """Clear state without waiting for an intentionally blocked test worker."""

        with self._lock:
            self._epoch += 1
            self._closed = False
            self._payload = None
            self._captured_at_monotonic = 0.0
            self._captured_at_unix = 0.0
            self._generation = 0
            self._snapshot_generation = 0
            self._active_generation = 0
            self._active_started_at = 0.0
            self._active_thread = None
            self._timed_out_generation = 0
            self._next_refresh_not_before = 0.0
            self._consecutive_failures = 0
            self._total_failures = 0
            self._total_timeouts = 0
            self._total_refreshes = 0
            self._last_error = ""
            self._last_error_at_unix = 0.0
            self._last_duration_s = 0.0
            self._incident_sequence = 0
            self._active_incident_id = ""
            self._last_recovery = None

    def request_refresh(self, *, force: bool = False) -> bool:
        now = self._clock()
        with self._lock:
            if self._closed:
                return False
            active = self._active_thread
            if active is not None and active.is_alive():
                return False
            if not force and now < self._next_refresh_not_before:
                return False
            if (
                not force
                and self._payload is not None
                and now - self._captured_at_monotonic
                < self._config.refresh_interval_s
            ):
                return False

            self._generation += 1
            generation = self._generation
            epoch = self._epoch
            self._active_generation = generation
            self._active_started_at = now
            self._timed_out_generation = 0
            worker = threading.Thread(
                target=self._run_refresh,
                args=(epoch, generation, now),
                name=f"{self._config.worker_name_prefix}-{generation}",
                daemon=True,
            )
            self._active_thread = worker
            worker.start()
            return True

    def _run_refresh(self, epoch: int, generation: int, started_at: float) -> None:
        try:
            payload = self._collector()
            if not isinstance(payload, dict):
                raise TypeError(
                    f"{self._config.log_label.lower()} collector returned "
                    f"{type(payload).__name__}, expected dict"
                )
        except Exception as exc:  # noqa: BLE001 - terminal worker boundary
            self._finish_failure(epoch, generation, started_at, exc)
            return
        self._finish_success(epoch, generation, started_at, payload)

    def _finish_success(
        self,
        epoch: int,
        generation: int,
        started_at: float,
        payload: dict[str, Any],
    ) -> None:
        now = self._clock()
        recovered: tuple[str, int] | None = None
        with self._lock:
            if epoch != self._epoch or generation != self._active_generation:
                return
            if self._active_incident_id:
                recovered = (
                    self._active_incident_id,
                    self._consecutive_failures,
                )
                self._last_recovery = {
                    "incident_id": self._active_incident_id,
                    "failed_refreshes": self._consecutive_failures,
                    "recovered_at_unix": self._wall_clock(),
                }
            self._payload = payload
            self._snapshot_generation = generation
            self._captured_at_monotonic = now
            self._captured_at_unix = self._wall_clock()
            self._total_refreshes += 1
            self._consecutive_failures = 0
            self._last_error = ""
            self._active_incident_id = ""
            self._last_duration_s = max(0.0, now - started_at)
            self._next_refresh_not_before = now + self._config.refresh_interval_s
            self._clear_active_locked(generation)
        if recovered:
            logger.info(
                "%s refresh recovered incident %s after %d failed refreshes",
                self._config.log_label,
                recovered[0],
                recovered[1],
            )

    def _finish_failure(
        self,
        epoch: int,
        generation: int,
        started_at: float,
        exc: Exception,
    ) -> None:
        now = self._clock()
        with self._lock:
            if epoch != self._epoch or generation != self._active_generation:
                return
            reason = f"{type(exc).__name__}: {str(exc) or '<no message>'}"
            if self._timed_out_generation == generation:
                first_failure = False
                self._last_error = reason[:320]
                self._last_error_at_unix = self._wall_clock()
            else:
                first_failure = self._record_failure_locked(reason, now=now)
            self._last_duration_s = max(0.0, now - started_at)
            self._clear_active_locked(generation)
            incident_id = self._active_incident_id
            failure_count = self._consecutive_failures
        log = logger.warning if first_failure else logger.debug
        log(
            "%s refresh incident %s failed (streak=%d): %s",
            self._config.log_label,
            incident_id,
            failure_count,
            exc,
        )

    def _record_failure_locked(self, reason: str, *, now: float) -> bool:
        first_failure = self._consecutive_failures == 0
        if first_failure:
            self._incident_sequence += 1
            self._active_incident_id = (
                f"{self._config.incident_prefix}-{self._incident_sequence:06d}"
            )
        self._consecutive_failures += 1
        self._total_failures += 1
        self._last_error = reason[:320]
        self._last_error_at_unix = self._wall_clock()
        exponent = min(10, max(0, self._consecutive_failures - 1))
        retry_s = min(
            self._config.retry_max_s,
            self._config.retry_base_s * (2**exponent),
        )
        self._next_refresh_not_before = now + retry_s
        return first_failure

    def _clear_active_locked(self, generation: int) -> None:
        if generation != self._active_generation:
            return
        self._active_generation = 0
        self._active_started_at = 0.0
        self._active_thread = None
        self._timed_out_generation = 0

    def _record_timeout_if_needed_locked(self, now: float) -> bool:
        if self._active_generation <= 0 or self._active_started_at <= 0.0:
            return False
        age = max(0.0, now - self._active_started_at)
        if age < self._config.collection_timeout_s:
            return False
        if self._timed_out_generation == self._active_generation:
            return False
        self._timed_out_generation = self._active_generation
        self._total_timeouts += 1
        self._record_failure_locked(
            f"TimeoutError: {self._config.log_label.lower()} refresh exceeded "
            f"{self._config.collection_timeout_s:.3f}s",
            now=now,
        )
        return True

    def read(self) -> dict[str, Any]:
        """Return immediately with the latest snapshot and freshness metadata."""

        self.request_refresh()
        now = self._clock()
        timed_out = False
        with self._lock:
            timed_out = self._record_timeout_if_needed_locked(now)
            payload = dict(self._payload or self._fallback_factory())
            has_snapshot = self._payload is not None
            age_s = (
                max(0.0, now - self._captured_at_monotonic)
                if has_snapshot
                else None
            )
            fresh = bool(
                has_snapshot
                and age_s is not None
                and age_s <= self._config.refresh_interval_s
            )
            expired = bool(
                not has_snapshot
                or age_s is None
                or age_s > self._config.max_stale_s
            )
            active = bool(
                self._active_thread is not None and self._active_thread.is_alive()
            )
            refresh_age_s = (
                max(0.0, now - self._active_started_at)
                if active and self._active_started_at > 0.0
                else 0.0
            )
            serving = (
                "initializing"
                if not has_snapshot
                else "fresh"
                if fresh
                else "expired"
                if expired
                else "stale_while_revalidate"
            )
            metadata = {
                "schema_version": self._config.schema_version,
                "generation": self._generation,
                "snapshot_generation": self._snapshot_generation,
                "captured_at": (
                    datetime.fromtimestamp(self._captured_at_unix, tz=UTC).isoformat()
                    if self._captured_at_unix > 0.0
                    else None
                ),
                "captured_at_unix": self._captured_at_unix or None,
                "age_s": round(age_s, 3) if age_s is not None else None,
                "fresh": fresh,
                "stale": bool(has_snapshot and not fresh),
                "expired": expired,
                "serving": serving,
                "refresh_in_flight": active,
                "refresh_generation": self._active_generation,
                "refresh_age_s": round(refresh_age_s, 3),
                "refresh_timed_out": bool(
                    active
                    and self._timed_out_generation == self._active_generation
                ),
                "refresh_interval_s": self._config.refresh_interval_s,
                "max_stale_s": self._config.max_stale_s,
                "collection_timeout_s": self._config.collection_timeout_s,
                "next_retry_in_s": round(
                    max(0.0, self._next_refresh_not_before - now), 3
                ),
                "consecutive_failures": self._consecutive_failures,
                "total_failures": self._total_failures,
                "total_timeouts": self._total_timeouts,
                "total_refreshes": self._total_refreshes,
                "last_error": self._last_error,
                "last_error_at_unix": self._last_error_at_unix or None,
                "last_duration_s": round(self._last_duration_s, 3),
                "incident_id": self._active_incident_id or None,
                "last_recovery": dict(self._last_recovery)
                if self._last_recovery
                else None,
            }
        if timed_out:
            logger.warning(
                "%s refresh incident %s exceeded %.3fs; serving %s snapshot",
                self._config.log_label,
                metadata.get("incident_id"),
                self._config.collection_timeout_s,
                metadata.get("serving"),
            )
        payload[self._config.metadata_key] = metadata
        return payload


__all__ = ["HealthReadModelConfig", "HealthSnapshotReadModel"]
