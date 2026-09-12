"""A single-slot background writer for state that is rebuilt, not appended.

The immune ecology is serialised under the lock that guards it — it has to
be, because the payload reads the cell population, the tissue field, the
lineage table and the expansion engine, and a snapshot taken while another
thread mutates them is a snapshot of nothing. The fsync that follows needs
none of that, and doing it there is how the runtime freezes: lockdep
reported ``fsync attempted while holding
['core.adaptation.adaptive_immunity._lock']`` during a subject-core run,
the same class as the on-loop fsync that once stopped the live event loop
for twenty minutes.

So the payload is handed here and the caller releases the lock. One slot,
replaced rather than queued: a newer snapshot of the same state makes the
older one worthless, and a queue would make a burst of observations cost a
burst of writes — which is the amplification the coalescing interval in
``_save_state`` already exists to stop. The slot is the same idea one
layer down, and it holds across the window where a write is in flight.

Durability is explicit. :meth:`flush` drains the slot and waits for the
write to finish, and it does the write itself when no thread is running,
so a caller that needs a durable point gets one without the writer having
to exist. Callers that hold the serialising lock must not call it: waiting
on a disk write with the lock held costs exactly what doing the write
there cost.

The thread retires after an idle period and restarts on the next deferred
submit. A test suite that constructs thousands of these would otherwise
accumulate one idle thread apiece for the life of the process.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable
from typing import Any

from core.runtime.lockdep import LockRank, checked_lock

logger = logging.getLogger("Aura.ImmuneStateWriter")

#: How long the writer waits with an empty slot before retiring. Long enough
#: that a conversation's worth of observations reuses one thread, short
#: enough that a process holding many writers does not hold many threads.
DEFAULT_IDLE_TIMEOUT_S = 30.0

#: Default ceiling on how long a durable caller waits for the slot to drain.
#: A write that has not landed in this long is a sick disk, and blocking
#: past it turns one slow write into a stalled subsystem.
DEFAULT_FLUSH_TIMEOUT_S = 5.0


class SingleSlotStateWriter:
    """Write the newest payload off the caller's thread, one at a time."""

    def __init__(
        self,
        name: str,
        write: Callable[[Any], None],
        *,
        idle_timeout_s: float = DEFAULT_IDLE_TIMEOUT_S,
        on_error: Callable[[BaseException], None] | None = None,
    ) -> None:
        self._name = name
        self._write = write
        self._idle_timeout_s = float(idle_timeout_s)
        self._on_error = on_error
        self._slot_lock = checked_lock(f"{name}.slot", rank=LockRank.LEAF)
        self._slot: Any | None = None
        self._writing = False
        self._stopping = False
        self._thread: threading.Thread | None = None
        self._wakeup = threading.Event()
        self._idle = threading.Event()
        self._idle.set()
        self._submitted = 0
        self._written = 0
        self._replaced = 0
        self._failed = 0
        self._inline = 0
        self._last_error: str | None = None

    @property
    def name(self) -> str:
        return self._name

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------

    def submit(self, payload: Any, *, background: bool) -> None:
        """Take the payload as the one to write next.

        ``background=True`` is the locked caller's path: it cannot wait, so
        a thread is started to do the write. ``background=False`` leaves the
        payload in the slot for the :meth:`flush` the caller is about to
        make, which keeps a durable write on the caller's own thread and
        starts no thread at all.
        """
        with self._slot_lock:
            if self._slot is not None:
                self._replaced += 1
            self._slot = payload
            self._submitted += 1
            self._idle.clear()
            if background and not self._stopping:
                self._ensure_thread_locked()
        self._wakeup.set()

    def flush(self, timeout: float = DEFAULT_FLUSH_TIMEOUT_S) -> bool:
        """Make the pending payload durable. True when the slot is empty.

        Never call this while holding the lock the payload was built under.

        A write already in flight is waited for rather than claimed, so two
        callers flushing at once do not both report failure while the write
        one of them started is landing.
        """
        deadline = time.monotonic() + max(0.0, float(timeout))
        while True:
            inline: Any | None = None
            with self._slot_lock:
                if self._slot is None and not self._writing:
                    return True
                thread = self._thread
                if (thread is None or not thread.is_alive()) and self._slot is not None:
                    inline = self._slot
                    self._slot = None
                    self._writing = True
            if inline is not None:
                self._inline += 1
                self._write_once(inline)
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return False
            self._wakeup.set()
            self._idle.wait(min(remaining, 0.25))

    def stop(self, timeout: float = DEFAULT_FLUSH_TIMEOUT_S) -> bool:
        """Drain what is pending and retire the thread, inside one budget.

        The drain and the join share ``timeout`` rather than each taking
        it, because a shutdown hook is given a budget for the whole call
        and a hook that doubles it reads as a hung subsystem.
        """
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._slot_lock:
            self._stopping = True
            thread = self._thread
        self._wakeup.set()
        drained = self.flush(max(0.0, deadline - time.monotonic()))
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
        return drained

    def stats(self) -> dict[str, Any]:
        with self._slot_lock:
            pending = self._slot is not None
            writing = self._writing
            running = self._thread is not None and self._thread.is_alive()
        return {
            "name": self._name,
            "submitted": self._submitted,
            "written": self._written,
            "coalesced": self._replaced,
            "failed": self._failed,
            "inline_writes": self._inline,
            "pending": pending,
            "writing": writing,
            "thread_running": running,
            "last_error": self._last_error,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ensure_thread_locked(self) -> None:
        """Start the writer if none is running. Caller holds the slot lock."""
        thread = self._thread
        if thread is not None and thread.is_alive():
            return
        try:
            thread = threading.Thread(
                target=self._run,
                name=f"{self._name}-writer",
                daemon=True,
            )
            thread.start()
        except RuntimeError as exc:
            # Out of threads, or interpreter shutdown. The payload stays in
            # the slot; the next flush writes it on the caller's thread.
            self._thread = None
            self._last_error = f"thread_start_failed: {exc}"
            logger.debug("%s: writer thread would not start: %s", self._name, exc)
            return
        self._thread = thread

    def _run(self) -> None:
        while True:
            woke = self._wakeup.wait(self._idle_timeout_s)
            self._wakeup.clear()
            self._drain()
            with self._slot_lock:
                if self._slot is not None:
                    continue
                self._idle.set()
                if self._stopping or not woke:
                    self._thread = None
                    return

    def _drain(self) -> None:
        while True:
            with self._slot_lock:
                payload = self._slot
                if payload is None:
                    return
                self._slot = None
                self._writing = True
            self._write_once(payload)

    def _write_once(self, payload: Any) -> None:
        try:
            self._write(payload)
            self._written += 1
        except Exception as exc:  # noqa: BLE001 — the writer must survive
            self._failed += 1
            self._last_error = f"{type(exc).__name__}: {exc}"
            if self._on_error is not None:
                try:
                    self._on_error(exc)
                except Exception:  # noqa: BLE001 — reporting must not kill it
                    logger.debug("%s: error handler failed", self._name, exc_info=True)
            else:
                logger.debug("%s: state write failed: %s", self._name, exc)
        finally:
            with self._slot_lock:
                self._writing = False
                if self._slot is None:
                    self._idle.set()


__all__ = [
    "DEFAULT_FLUSH_TIMEOUT_S",
    "DEFAULT_IDLE_TIMEOUT_S",
    "ImmuneStatePersistence",
    "SingleSlotStateWriter",
]


class ImmuneStatePersistence:
    """How the immune ecology reaches disk, and how a caller waits for it.

    Lifted off ``AdaptiveImmuneSystem`` because that class had grown past the
    method ceiling and this is the part of it with a boundary of its own: it
    owns the coalescing interval, the snapshot, the digest, the handover to
    the writer thread and the shutdown drain, and it reads the ecology only
    through attributes the host class already holds.

    A mixin rather than module functions. Two suites patch these by name —
    one on the instance, one on ``AdaptiveImmuneSystem`` itself — and a method
    set on the class shadows the mixin, so both keep working. Module functions
    would have broken the second.

    What the host must provide: ``_lock``, ``_state_writer``, ``_state_path``,
    ``_state_dirty``, ``_last_save_at``, ``_save_min_interval_s``,
    ``_deferred_saves``, ``_cells``, ``_tissue``, ``_lineage_stats``,
    ``_observation_count``, ``_last_dream_at``, ``_recent_antigens``,
    ``_recent_responses``, ``_recurrence_tracker`` and ``expansion_engine``.

    The names it reads out of :mod:`core.adaptation.adaptive_immunity` are
    imported inside the methods. That module imports this one, so a top-level
    import would be a cycle — and the call-time form is also what keeps a
    test's patch of a module-level constant visible from here.
    """

    def _save_state(self, *, force: bool = False) -> None:
        """Persist the immune ecology, coalescing bursts.

        CP126 df9f2a05: core observation writes state, reinforcement can write
        again, and the response summary writes again — so ONE event
        serialized the whole ecology several times. During a failure storm,
        which is exactly when many events arrive at once, that multiplied I/O
        and lock time in the subsystem meant to be responding to the storm.

        Writes inside the coalescing interval are deferred, not dropped: the
        dirty flag survives and the next call past the interval writes the
        latest state. The honest cost is that a crash can lose up to
        ``_save_min_interval_s`` of fitness updates — which is a far better
        trade than amplifying the storm that causes the crash, and callers
        that need a durable point (boot seeding, consolidation) pass
        force=True.

        force=True means the snapshot is taken now and written next, not
        that the disk has it when this returns. The write happens on the
        caller's thread only where blocking is the caller's own cost —
        off the ecology lock and off the event loop. Everywhere else it
        goes to the writer thread, and :meth:`flush_state` is how a caller
        waits for it.
        """
        from core.adaptation.adaptive_immunity import (
            IMMUNE_STATE_SCHEMA_VERSION,
            EffectorKind,
            _immune_state_digest,
            _on_event_loop,
        )

        now = time.time()
        self._state_dirty = True
        if not force and (now - self._last_save_at) < self._save_min_interval_s:
            self._deferred_saves += 1
            return
        self._last_save_at = now
        self._state_dirty = False
        payload = {
            "cells": [cell.to_dict() for cell in self._cells],
            "tissue": self._tissue.to_dict(),
            "lineage_stats": {
                lineage_id: {
                    "successes": int(stats["successes"]),
                    "failures": int(stats["failures"]),
                    "best_effector": (
                        stats["best_effector"].value
                        if isinstance(stats["best_effector"], EffectorKind)
                        else None
                    ),
                    "best_fitness": float(stats["best_fitness"]),
                }
                for lineage_id, stats in self._lineage_stats.items()
            },
            "observation_count": self._observation_count,
            "last_dream_at": self._last_dream_at,
            "recent_antigens": [antigen.to_dict() for antigen in list(self._recent_antigens)[-24:]],
            "recent_responses": list(self._recent_responses)[-24:],
            "recurrence_tracker": {
                key: {
                    "occurrences": int(stats.get("occurrences", 0)),
                    "last_seen": float(stats.get("last_seen", 0.0)),
                    "interval_ewma": float(stats.get("interval_ewma", 0.0)),
                    "last_interval": (
                        float(stats["last_interval"])
                        if stats.get("last_interval") is not None
                        else None
                    ),
                    "streak": int(stats.get("streak", 0)),
                    "peak_streak": int(stats.get("peak_streak", 0)),
                    "verified_repairs": int(stats.get("verified_repairs", 0)),
                    "failed_repairs": int(stats.get("failed_repairs", 0)),
                    "last_verified_at": float(stats.get("last_verified_at", 0.0)),
                }
                for key, stats in self._recurrence_tracker.items()
            },
            "expansion_engine": self.expansion_engine.to_dict(),
        }
        payload["schema_version"] = IMMUNE_STATE_SCHEMA_VERSION
        # An unkeyed digest over the body: it detects CORRUPTION and truncation,
        # not tampering by anyone who can write the file. Labelled as what it
        # is rather than as a trust root — a signed state file needs a key this
        # subsystem does not hold (CP126 5c214831).
        payload["integrity"] = {
            "algorithm": "sha256-unkeyed",
            "digest": _immune_state_digest(payload),
        }
        # The snapshot had to be built under the caller's lock. The write
        # does not: lockdep reported the fsync under _lock during a
        # subject-core run, which is the class of defect that froze the live
        # event loop for twenty minutes. Hand the payload over and let the
        # disk work happen where nothing is waiting on the ecology.
        here = force and not self._lock.held_by_current_thread() and not _on_event_loop()
        self._state_writer.submit(payload, background=not here)
        if here:
            self.flush_state()

    def flush_state(self, timeout: float = DEFAULT_FLUSH_TIMEOUT_S) -> bool:
        """Make the newest snapshot durable. True when nothing is pending.

        Refuses under ``_lock``, and on the event loop, and says so with
        False. Both are places where waiting for a disk write costs what
        doing the write there cost — the lock because every other caller of
        the immune ecology queues behind it, the loop because everything
        else in the runtime does. The queued payload is not lost either
        way: the writer thread still has it.
        """
        from core.adaptation.adaptive_immunity import _on_event_loop

        if self._lock.held_by_current_thread() or _on_event_loop():
            return False
        return self._state_writer.flush(timeout)

    def on_stop(self) -> None:
        """Container shutdown hook: the last snapshot has to land.

        Bounded well inside the container's per-hook budget. A state file
        this size writes in milliseconds, so a drain that needs seconds is
        a sick disk, and holding shutdown for it helps nobody.
        """
        from core.adaptation.adaptive_immunity import STATE_SHUTDOWN_TIMEOUT_S

        self._state_writer.stop(timeout=STATE_SHUTDOWN_TIMEOUT_S)

    def _record_state_write_failure(self, exc: BaseException) -> None:
        """Report a failed state write from whichever thread ran it.

        The write used to run inline, so a narrow except list was enough —
        anything unlisted reached the caller. Off the caller's thread there
        is no caller to reach, so every failure is recorded here instead of
        one class of them being lost with the thread.
        """
        from core.adaptation.adaptive_immunity import (
            _record_adaptive_immunity_degradation,
            logger,
        )

        _record_adaptive_immunity_degradation(
            exc,
            action="Skipped adaptive immune persistence write and kept in-memory immune state active",
            extra={"state_path": str(self._state_path), "cells": len(self._cells)},
        )
        logger.debug("Adaptive immune state save skipped: %s", exc)

    def _write_state_payload(self, payload: dict[str, Any]) -> None:
        """Put one serialized ecology on disk. Never called under ``_lock``.

        Route through the governed file-write gateway: a repair-capable,
        behavior-evolving state file is a consequential write and must be
        authorized and receipt-bound like every other one. The scope is
        built here rather than inherited, so it holds on the writer thread.
        """
        from core.governance_context import local_internal_governed_scope
        from core.runtime.file_write_gateway import get_file_write_gateway

        with local_internal_governed_scope(
            "adaptation.adaptive_immunity.state",
            domain="file_write",
            receipt_prefix="adaptive-immunity-state",
        ):
            get_file_write_gateway().write_text(
                self._state_path,
                json.dumps(payload, indent=2),
                source="adaptation.adaptive_immunity.state",
            )

    def _load_state(self) -> bool:
        from collections import defaultdict, deque

        from core.adaptation.adaptive_immunity import (
            IMMUNE_STATE_SCHEMA_VERSION,
            MAX_IMMUNE_STATE_BYTES,
            Antigen,
            CellKind,
            EffectorKind,
            ImmuneCell,
            TissueField,
            _immune_state_digest,
            _live_rule_vocabulary,
            _normalize_behavioral_rule,
            _record_adaptive_immunity_degradation,
            logger,
        )

        if not self._state_path.exists():
            return False
        try:
            size = self._state_path.stat().st_size
            if size > MAX_IMMUNE_STATE_BYTES:
                raise ValueError(
                    f"immune state file is {size} bytes, over the "
                    f"{MAX_IMMUNE_STATE_BYTES} bound"
                )
            payload = json.loads(self._state_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("immune state must be a JSON object")
            # A file written by a different layout is quarantined to a reseed
            # rather than parsed field-by-field into a live repair-capable
            # population (CP126 5c214831).
            found_version = int(payload.get("schema_version", 0) or 0)
            if found_version != IMMUNE_STATE_SCHEMA_VERSION:
                raise ValueError(
                    f"immune state schema {found_version} != "
                    f"{IMMUNE_STATE_SCHEMA_VERSION}"
                )
            integrity = payload.get("integrity")
            if isinstance(integrity, dict) and integrity.get("digest"):
                if _immune_state_digest(payload) != str(integrity["digest"]):
                    raise ValueError("immune state digest does not match its contents")
            if "expansion_engine" in payload:
                from core.adaptation.dimensional_expansion import DimensionalExpansionEngine

                self.expansion_engine = DimensionalExpansionEngine.from_dict(
                    payload["expansion_engine"]
                )

            self._cells = [ImmuneCell.from_dict(item) for item in payload.get("cells", [])]
            vocabulary = _live_rule_vocabulary()
            migrated_rules = 0
            for cell in self._cells:
                if cell.kind not in {CellKind.B, CellKind.MEMORY}:
                    if cell.behavioral_rule is not None:
                        cell.behavioral_rule = None
                        migrated_rules += 1
                    continue
                normalized, migrated = _normalize_behavioral_rule(
                    cell.behavioral_rule,
                    self._rng,
                    vocabulary=vocabulary,
                )
                cell.behavioral_rule = normalized
                migrated_rules += int(migrated)
            self._migrated_behavioral_rules = migrated_rules
            if migrated_rules:
                logger.info(
                    "Migrated %d persisted immune behavioral rule(s) to the bounded grammar",
                    migrated_rules,
                )

            # Reconcile receptor vectors of loaded cells with system current_dim
            target_dim = self.expansion_engine.current_dim
            for cell in self._cells:
                cell.resize_receptor(target_dim, self._rng)

            self._tissue = TissueField.from_dict(
                payload.get("tissue", {}),
                diffusion=self.cfg.tissue_diffusion,
                decay=self.cfg.tissue_decay,
            )
            self._lineage_stats = defaultdict(
                lambda: {
                    "successes": 0,
                    "failures": 0,
                    "best_effector": None,
                    "best_fitness": 0.0,
                }
            )
            for lineage_id, stats in payload.get("lineage_stats", {}).items():
                self._lineage_stats[lineage_id] = {
                    "successes": int(stats.get("successes", 0)),
                    "failures": int(stats.get("failures", 0)),
                    "best_effector": (
                        EffectorKind(stats["best_effector"]) if stats.get("best_effector") else None
                    ),
                    "best_fitness": float(stats.get("best_fitness", 0.0)),
                }
            self._observation_count = int(payload.get("observation_count", 0))
            self._last_dream_at = int(payload.get("last_dream_at", 0))
            self._recent_antigens = deque(
                [Antigen.from_dict(item) for item in payload.get("recent_antigens", [])],
                maxlen=self.cfg.replay_buffer_size,
            )
            self._recent_responses = deque(
                [dict(item) for item in payload.get("recent_responses", [])],
                maxlen=self.cfg.recent_response_buffer,
            )
            self._recurrence_tracker = defaultdict(
                lambda: {
                    "occurrences": 0,
                    "last_seen": 0.0,
                    "interval_ewma": 0.0,
                    "last_interval": None,
                    "streak": 0,
                    "peak_streak": 0,
                    "verified_repairs": 0,
                    "failed_repairs": 0,
                    "last_verified_at": 0.0,
                }
            )
            for key, stats in payload.get("recurrence_tracker", {}).items():
                self._recurrence_tracker[str(key)] = {
                    "occurrences": int(stats.get("occurrences", 0)),
                    "last_seen": float(stats.get("last_seen", 0.0)),
                    "interval_ewma": float(stats.get("interval_ewma", 0.0)),
                    "last_interval": self._coerce_optional_float(stats.get("last_interval")),
                    "streak": int(stats.get("streak", 0)),
                    "peak_streak": int(stats.get("peak_streak", 0)),
                    "verified_repairs": int(stats.get("verified_repairs", 0)),
                    "failed_repairs": int(stats.get("failed_repairs", 0)),
                    "last_verified_at": float(stats.get("last_verified_at", 0.0)),
                }
            self._assign_species()
            return bool(self._cells)
        except (
            OSError,
            ConnectionError,
            TimeoutError,
            # Corrupt or hostile persisted state must quarantine to a reseed,
            # never abort immune construction: JSON, schema, enum, and
            # numeric failures were previously uncaught here.
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            _record_adaptive_immunity_degradation(
                exc,
                action="Rejected persisted adaptive immune state and reseeded immune population",
                severity="degraded",
                extra={"state_path": str(self._state_path)},
            )
            logger.warning("Adaptive immune state load failed; reseeding: %s", exc)
            return False
