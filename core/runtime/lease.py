"""core/runtime/lease.py — lease-based leader election.

Clean-room adoption of Kubernetes' `coordination.k8s.io/Lease` and the
`leaderelection` client package.

Aura's recorded worst incident class is two runtimes doing the same job at
once. A false-death verdict made the launcher respawn a second 32B without
reaping the wedged one; memory doubled; the doubling made false deaths
more likely; the cascade fed itself. The fix applied at the time was
manual — `pkill` and a single relaunch. That is not a fix, it is a
recovery.

Leader election is the actual fix, and the reason it works is a detail
that is easy to get wrong and that Kubernetes gets right:

**The old leader must give up before the new one can take over.** A lease
has a duration and a renew deadline, with the renew deadline strictly
shorter. The holder renews continuously; if it cannot renew before the
renew deadline expires, it *stops leading immediately, on its own*,
without being told. A challenger may only acquire after the full lease
duration has passed since the last successful renewal. The gap between
those two moments is the safety margin, and it means there is never an
instant when two processes both believe they hold the lease — even if the
old one is wedged, paused, or on the far side of a clock jump.

Everything else follows:

* Identity includes pid and boot id, so a lease left behind by a crashed
  process is distinguishable from one held by a live one and can be
  reclaimed immediately rather than after a timeout.
* Observing a *live* foreign holder taints the runtime with
  DUPLICATE_RUNTIME, because two runtimes on one host is a fact every
  later report needs to carry.
* ``transitions`` counts how many times leadership actually moved. A
  system that is flapping leadership looks healthy in every instantaneous
  snapshot and is obvious in this one number.

Use it for anything that must have exactly one owner across processes:
the resident model lane, the autonomy conductor, scheduled convergence,
the training launcher.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import socket
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Aura.Lease")

#: A lease is valid this long after its last renewal. A challenger may not
#: acquire before it expires.
DEFAULT_LEASE_DURATION_S = 15.0
#: The holder gives up if it cannot renew within this. Strictly shorter
#: than the duration — that difference is the whole safety margin.
DEFAULT_RENEW_DEADLINE_S = 10.0
#: How often to attempt a renewal or an acquisition.
DEFAULT_RETRY_PERIOD_S = 2.0


@dataclass(frozen=True)
class Identity:
    """Who is holding. pid + boot id make crash detection cheap."""

    holder: str
    pid: int
    boot_id: str
    host: str
    started_at: float

    @classmethod
    def current(cls, holder: str | None = None) -> Identity:
        return cls(
            holder=holder or f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}",
            pid=os.getpid(),
            boot_id=_boot_id(),
            host=socket.gethostname(),
            started_at=time.time(),
        )

    def same_process(self, other: Identity) -> bool:
        return self.pid == other.pid and self.boot_id == other.boot_id

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _boot_id() -> str:
    """A value that changes when the host reboots.

    Without it a recycled pid on a fresh boot looks like a live holder.
    """
    try:
        return f"{int(_host_boot_time())}"
    except Exception:  # noqa: BLE001
        logger.debug("host boot identity unavailable", exc_info=True)
        return "unknown"


def _host_boot_time() -> float:
    from core.runtime.resource_psutil import boot_time

    return float(boot_time())


@dataclass
class LeaseRecord:
    name: str
    identity: Identity
    acquired_at: float
    renewed_at: float
    lease_duration_s: float
    transitions: int = 0

    def expires_at(self) -> float:
        return self.renewed_at + self.lease_duration_s

    def expired(self, now: float | None = None) -> bool:
        return (now if now is not None else time.time()) >= self.expires_at()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "identity": self.identity.to_dict(),
            "acquired_at": self.acquired_at,
            "renewed_at": self.renewed_at,
            "lease_duration_s": self.lease_duration_s,
            "expires_at": self.expires_at(),
            "transitions": self.transitions,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> LeaseRecord:
        identity = payload.get("identity") or {}
        return cls(
            name=str(payload.get("name", "")),
            identity=Identity(
                holder=str(identity.get("holder", "")),
                pid=int(identity.get("pid", 0) or 0),
                boot_id=str(identity.get("boot_id", "")),
                host=str(identity.get("host", "")),
                started_at=float(identity.get("started_at", 0.0) or 0.0),
            ),
            acquired_at=float(payload.get("acquired_at", 0.0) or 0.0),
            renewed_at=float(payload.get("renewed_at", 0.0) or 0.0),
            lease_duration_s=float(
                payload.get("lease_duration_s", DEFAULT_LEASE_DURATION_S) or DEFAULT_LEASE_DURATION_S
            ),
            transitions=int(payload.get("transitions", 0) or 0),
        )


def _lease_path(name: str) -> Path:
    """Where this lease lives.

    Overridable because the default is the shared data dir, which every
    concurrent process resolves to the same file — including the live runtime
    and every parallel test chunk. Leader election contending across unrelated
    processes is exactly the failure a lease exists to prevent, and it made
    tests pass alone and fail together for reasons no individual test could
    explain. An operator relocating runtime state has the same need.
    """
    import os

    override = str(os.environ.get("AURA_RUNTIME_LEASE_DIR", "") or "").strip()
    if override:
        return Path(override).expanduser() / f"{name}.json"

    from core.config import config

    return Path(config.paths.data_dir) / "runtime" / "leases" / f"{name}.json"


def _holder_is_live(identity: Identity) -> bool:
    """Is the recorded holder still a running process on this boot?

    A lease from a crashed process is reclaimable immediately; one from a
    live process must wait out its duration.
    """
    if identity.boot_id != _boot_id():
        return False
    if identity.host != socket.gethostname():
        # A different host cannot be checked; assume live and wait it out.
        return True
    if identity.pid <= 0:
        return False
    try:
        os.kill(identity.pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return True
    return True


class LeaderElector:
    """One lease, one holder, with a give-up-before-takeover guarantee."""

    def __init__(
        self,
        name: str,
        *,
        identity: Identity | None = None,
        lease_duration_s: float = DEFAULT_LEASE_DURATION_S,
        renew_deadline_s: float = DEFAULT_RENEW_DEADLINE_S,
        retry_period_s: float = DEFAULT_RETRY_PERIOD_S,
        on_started_leading: Callable[[], Any] | None = None,
        on_stopped_leading: Callable[[], Any] | None = None,
        on_new_leader: Callable[[str], Any] | None = None,
    ) -> None:
        if renew_deadline_s >= lease_duration_s:
            raise ValueError(
                "renew_deadline_s must be strictly less than lease_duration_s; "
                "that difference is the margin that prevents two leaders"
            )
        self.name = name
        self.identity = identity or Identity.current()
        self.lease_duration_s = lease_duration_s
        self.renew_deadline_s = renew_deadline_s
        self.retry_period_s = retry_period_s
        self._on_started = on_started_leading
        self._on_stopped = on_stopped_leading
        self._on_new_leader = on_new_leader

        self._lock = threading.Lock()
        self._is_leader = False
        self._last_renew_ok = 0.0
        self._observed_leader = ""
        self._task: asyncio.Task | None = None
        self._stopping = asyncio.Event()
        self.acquisitions = 0
        self.renewals = 0
        self.renew_failures = 0
        self.lost_leadership = 0
        self._duplicate_reported = False

    # ── state ─────────────────────────────────────────────────────────
    @property
    def is_leader(self) -> bool:
        with self._lock:
            return self._is_leader

    def observed_leader(self) -> str:
        with self._lock:
            return self._observed_leader

    # ── storage ───────────────────────────────────────────────────────
    def _read(self) -> LeaseRecord | None:
        path = _lease_path(self.name)
        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except OSError:
            logger.debug("lease %s unreadable", self.name, exc_info=True)
            return None
        try:
            return LeaseRecord.from_dict(json.loads(raw))
        except (ValueError, TypeError):
            logger.warning("lease %s is corrupt; treating as unheld", self.name)
            return None

    async def _read_async(self) -> LeaseRecord | None:
        """Read lease storage without lending filesystem latency to the loop."""
        return await asyncio.to_thread(self._read)

    def _write_sync(self, record: LeaseRecord) -> bool:
        from core.governance_context import local_internal_governed_scope
        from core.runtime.file_write_gateway import get_file_write_gateway

        path = _lease_path(self.name)
        try:
            with local_internal_governed_scope(f"lease.{self.name}"):
                gateway = get_file_write_gateway()
                gateway.ensure_directory(path.parent, source=f"lease.{self.name}")
                gateway.write_text(
                    path,
                    json.dumps(record.to_dict(), indent=2, sort_keys=True),
                    durable=True,
                    source=f"lease.{self.name}",
                )
            return True
        except Exception:  # noqa: BLE001 — a failed write means we do not hold it
            logger.warning("lease %s write failed", self.name, exc_info=True)
            return False

    async def _write(self, record: LeaseRecord) -> bool:
        """Async lane: the fsync happens on a worker thread, never on the loop."""
        from core.governance_context import local_internal_governed_scope
        from core.runtime.file_write_gateway import get_file_write_gateway

        path = _lease_path(self.name)
        try:
            with local_internal_governed_scope(f"lease.{self.name}"):
                gateway = get_file_write_gateway()
                await gateway.ensure_directory_async(
                    path.parent,
                    source=f"lease.{self.name}",
                )
                await gateway.write_text_async(
                    path,
                    json.dumps(record.to_dict(), indent=2, sort_keys=True),
                    durable=True,
                    source=f"lease.{self.name}",
                )
            return True
        except Exception:  # noqa: BLE001
            logger.warning("lease %s async write failed", self.name, exc_info=True)
            return False

    # ── the election ──────────────────────────────────────────────────
    def _decide(self, existing: LeaseRecord | None, now: float) -> tuple[str, LeaseRecord | None]:
        """Pure decision: ('acquire'|'renew'|'observe', record)."""
        if existing is None:
            return "acquire", LeaseRecord(
                name=self.name,
                identity=self.identity,
                acquired_at=now,
                renewed_at=now,
                lease_duration_s=self.lease_duration_s,
                transitions=0,
            )
        if existing.identity.same_process(self.identity):
            existing.identity = self.identity
            existing.renewed_at = now
            existing.lease_duration_s = self.lease_duration_s
            return "renew", existing

        # Someone else holds it. Reclaim only if the lease has genuinely
        # expired, or if the holder is provably gone.
        holder_live = _holder_is_live(existing.identity)
        if existing.expired(now) or not holder_live:
            return "acquire", LeaseRecord(
                name=self.name,
                identity=self.identity,
                acquired_at=now,
                renewed_at=now,
                lease_duration_s=self.lease_duration_s,
                transitions=existing.transitions + 1,
            )
        return "observe", existing

    async def try_acquire_or_renew(self) -> bool:
        """One attempt. Returns whether we hold the lease afterwards."""
        now = time.time()
        existing = await self._read_async()
        action, record = self._decide(existing, now)

        if action == "observe" and record is not None:
            self._note_foreign_holder(record)
            await self._relinquish("another process holds the lease")
            return False

        assert record is not None
        if not await self._write(record):
            # We could not persist the claim, so we do not hold it. This
            # is the branch that matters: never keep leading on a write we
            # could not make.
            self.renew_failures += 1
            await self._check_renew_deadline()
            return False

        self._last_renew_ok = now
        with self._lock:
            self._observed_leader = self.identity.holder
            became_leader = not self._is_leader
            self._is_leader = True
        if action == "acquire":
            self.acquisitions += 1
        else:
            self.renewals += 1
        if became_leader:
            logger.info(
                "👑 acquired lease %r as %s (duration %.0fs, transitions %d)",
                self.name,
                self.identity.holder,
                self.lease_duration_s,
                record.transitions,
            )
            await _maybe_await(self._on_started)
        return True

    def _note_foreign_holder(self, record: LeaseRecord) -> None:
        with self._lock:
            changed = self._observed_leader != record.identity.holder
            self._observed_leader = record.identity.holder
        if changed:
            logger.info(
                "lease %r is held by %s (pid %d), expires in %.1fs",
                self.name,
                record.identity.holder,
                record.identity.pid,
                max(0.0, record.expires_at() - time.time()),
            )
            if self._on_new_leader is not None:
                with contextlib.suppress(Exception):
                    self._on_new_leader(record.identity.holder)

        if (
            not self._duplicate_reported
            and record.identity.host == self.identity.host
            and record.identity.boot_id == self.identity.boot_id
            and record.identity.pid != self.identity.pid
            and _holder_is_live(record.identity)
        ):
            self._duplicate_reported = True
            from core.runtime.taint import TaintFlag, taint

            taint(
                TaintFlag.DUPLICATE_RUNTIME,
                f"lease {self.name!r} is held by live pid {record.identity.pid} on this "
                f"host while pid {self.identity.pid} also wants it",
                subsystem="lease",
            )

    async def _check_renew_deadline(self) -> None:
        """Give up leadership if we have not renewed within the deadline."""
        if not self.is_leader:
            return
        since = time.time() - self._last_renew_ok
        if since >= self.renew_deadline_s:
            await self._relinquish(
                f"renew deadline exceeded ({since:.1f}s >= {self.renew_deadline_s:.1f}s)"
            )

    async def _relinquish(self, reason: str, *, expected: bool = False) -> None:
        with self._lock:
            if not self._is_leader:
                return
            self._is_leader = False
            self.lost_leadership += 1
        log = logger.info if expected else logger.warning
        log("👑 %s lease %r: %s", "released" if expected else "lost", self.name, reason)
        await _maybe_await(self._on_stopped)

    # ── lifecycle ─────────────────────────────────────────────────────
    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._stopping.clear()
        self._task = get_task_tracker().create_task(
            self._run(),
            name=f"lease.{self.name}",
        )

    async def _run(self) -> None:
        while not self._stopping.is_set():
            try:
                await self.try_acquire_or_renew()
                await self._check_renew_deadline()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — the elector never dies quietly
                self.renew_failures += 1
                logger.warning("lease %s loop error: %s", self.name, exc)
                await self._check_renew_deadline()
            try:
                await asyncio.wait_for(self._stopping.wait(), timeout=self.retry_period_s)
            except TimeoutError:
                continue

    async def stop(self, *, release: bool = True) -> None:
        """Stop renewing and, by default, hand the lease back immediately."""
        self._stopping.set()
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        if release and self.is_leader:
            await self._release_lease()
        await self._relinquish("stopped", expected=True)

    async def _release_lease(self) -> None:
        """Expire our own lease so a successor takes over now, not in 15s."""
        record = self._read()
        if record is None or not record.identity.same_process(self.identity):
            return
        record.renewed_at = 0.0  # immediately expired
        await self._write(record)

    def report(self) -> dict[str, Any]:
        record = self._read()
        return {
            "name": self.name,
            "identity": self.identity.to_dict(),
            "is_leader": self.is_leader,
            "observed_leader": self.observed_leader(),
            "lease_duration_s": self.lease_duration_s,
            "renew_deadline_s": self.renew_deadline_s,
            "acquisitions": self.acquisitions,
            "renewals": self.renewals,
            "renew_failures": self.renew_failures,
            "lost_leadership": self.lost_leadership,
            "seconds_since_renew": round(time.time() - self._last_renew_ok, 2)
            if self._last_renew_ok
            else None,
            "record": record.to_dict() if record else None,
        }


async def _maybe_await(fn: Callable[[], Any] | None) -> None:
    if fn is None:
        return
    try:
        outcome = fn()
        if asyncio.iscoroutine(outcome):
            await outcome
    except Exception:  # noqa: BLE001 — a callback must not break the election
        logger.warning("lease callback failed", exc_info=True)


@dataclass
class _Registry:
    electors: dict[str, LeaderElector] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)


_REGISTRY = _Registry()


def get_elector(
    name: str,
    **kwargs: Any,
) -> LeaderElector:
    """Fetch or create the elector for a named lease. Idempotent."""
    with _REGISTRY.lock:
        existing = _REGISTRY.electors.get(name)
        if existing is not None:
            return existing
        elector = LeaderElector(name, **kwargs)
        _REGISTRY.electors[name] = elector
        return elector


def is_leader(name: str) -> bool:
    """Gate leader-only work.

    Returns False when no elector exists for the lease — refusing to act
    is the safe answer when nobody is running the election.
    """
    with _REGISTRY.lock:
        elector = _REGISTRY.electors.get(name)
    return bool(elector and elector.is_leader)


def should_act_as_singleton(name: str) -> bool:
    """Fail-open variant of :func:`is_leader`, for *protective* work.

    Exclusivity has two failure directions and they are not symmetric.
    For mutative work — launching a model, writing a shared ledger —
    acting twice is worse than not acting, so :func:`is_leader` fails
    closed. For protective work — shedding memory, enforcing eviction —
    *not* acting is worse than acting twice, so this returns True unless
    another process is provably alive and holding the lease.

    Choosing the wrong one of these is how a safety mechanism gets
    disabled by the machinery meant to coordinate it.
    """
    with _REGISTRY.lock:
        elector = _REGISTRY.electors.get(name)
    if elector is None:
        return True
    if elector.is_leader:
        return True
    record = elector._read()  # noqa: SLF001 — same-module accessor
    if record is None or record.expired():
        return True
    if record.identity.same_process(elector.identity):
        return True
    return not _holder_is_live(record.identity)


async def start_all_electors() -> list[str]:
    with _REGISTRY.lock:
        electors = list(_REGISTRY.electors.values())
    started: list[str] = []
    for elector in electors:
        with contextlib.suppress(Exception):
            await elector.start()
            started.append(elector.name)
    return started


async def stop_all_electors() -> None:
    with _REGISTRY.lock:
        electors = list(_REGISTRY.electors.values())
    for elector in electors:
        with contextlib.suppress(Exception):
            await elector.stop()


def lease_report() -> dict[str, Any]:
    with _REGISTRY.lock:
        electors = list(_REGISTRY.electors.values())
    return {
        "count": len(electors),
        "leases": [e.report() for e in electors],
        "held": [e.name for e in electors if e.is_leader],
    }


def reset_leases_for_test() -> None:
    with _REGISTRY.lock:
        _REGISTRY.electors.clear()


__all__ = [
    "DEFAULT_LEASE_DURATION_S",
    "DEFAULT_RENEW_DEADLINE_S",
    "DEFAULT_RETRY_PERIOD_S",
    "Identity",
    "LeaderElector",
    "LeaseRecord",
    "get_elector",
    "is_leader",
    "lease_report",
    "reset_leases_for_test",
    "should_act_as_singleton",
    "start_all_electors",
    "stop_all_electors",
]

#: The lease every runtime process contends for. Holding it means "I am
#: the runtime on this host"; observing a live foreign holder is the
#: duplicate-runtime condition that used to be found only by reading
#: memory graphs after the cascade.
RUNTIME_LEASE = "aura_runtime"
