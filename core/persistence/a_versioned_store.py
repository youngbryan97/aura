"""A store that knows what version it wrote and refuses to guess.

A blind comparison gave Home Assistant the highest maturity score and said the
persistence code earned it alone: versioned records, minor versions with a
forward-readable constraint, serialised loads, bounded concurrent loads, deep
copies of pending writes, migration hooks, atomic writes, a read-only mode,
final writes tied into shutdown, corruption detection, and corrupt state put
aside rather than casually overwritten.

Aura had most of the pieces in different files and none of them together. The
one that was missing everywhere is the minor-version rule, and it is the one
that matters for a system that keeps changing shape:

* A **major** version this reader does not know is unreadable. Migrate it or
  refuse; do not guess.
* A **minor** version ahead of this reader is readable. A newer writer may
  only add fields, so an older reader can still use what it understands —
  which is what lets a rollback keep the data written since.

The other rule with teeth: a file that will not parse is moved aside, not
overwritten. A store that writes over corruption has destroyed the only
evidence of what went wrong, and the next question after any corruption is
always what the bytes were.
"""
from __future__ import annotations

import copy
import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from core.runtime.lockdep import checked_lock

logger = logging.getLogger("Aura.AVersionedStore")

__all__ = [
    "AVersionedStore",
    "CannotRead",
    "Kept",
    "TOO_MANY_AT_ONCE",
]

#: How many stores may be loading at once. Loading parses and migrates, and a
#: boot that opens forty of them together turns a fast disk into a stall.
#: Bounded rather than serialised: one at a time makes boot the sum of every
#: load, which is the other way to be wrong about this.
TOO_MANY_AT_ONCE = 4
_LOADING = threading.Semaphore(TOO_MANY_AT_ONCE)


class CannotRead(RuntimeError):
    """The file is there and this reader cannot use it.

    Carries where the unreadable copy was put, so the next question — what
    were the bytes — has an answer.
    """

    def __init__(self, why: str, *, put_aside_at: Path | None = None) -> None:
        super().__init__(why)
        self.put_aside_at = put_aside_at


@dataclass(frozen=True)
class Kept:
    """What came back, with the version it was written under."""

    data: dict[str, Any]
    major: int
    minor: int
    written_at: float


class AVersionedStore:
    """One file, versioned, migrated, and never overwritten when unreadable."""

    def __init__(
        self,
        path: Any,
        *,
        major: int,
        minor: int = 0,
        migrate: Callable[[dict[str, Any], int, int], dict[str, Any]] | None = None,
        read_only: bool = False,
        source: str = "a_versioned_store",
    ) -> None:
        self._path = Path(path)
        self._major = int(major)
        self._minor = int(minor)
        self._migrate = migrate
        self._read_only = bool(read_only)
        self._source = str(source)
        self._lock = checked_lock(f"a_versioned_store:{self._path.name}")
        self._pending: dict[str, Any] | None = None
        self._written_at = 0.0

    # -------------------------------------------------------------- reading

    @property
    def path(self) -> Path:
        return self._path

    @property
    def read_only(self) -> bool:
        return self._read_only

    def load(self) -> Kept | None:
        """Read it, migrating where the version is behind this reader.

        None when there is nothing there. ``CannotRead`` when there is
        something there that this reader must not act on.
        """
        with _LOADING:
            return self._load()

    def _load(self) -> Kept | None:
        with self._lock:
            if not self._path.exists():
                return None
            try:
                raw = json.loads(self._path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                where = self._put_aside("unreadable")
                raise CannotRead(
                    f"{self._path.name} would not parse: {exc}", put_aside_at=where
                ) from exc
            if not isinstance(raw, dict):
                where = self._put_aside("not_a_record")
                raise CannotRead(
                    f"{self._path.name} is not a versioned record", put_aside_at=where
                )
            if "data" not in raw and "major" not in raw:
                # A file written before this store existed. It is version 0
                # and the whole object is the data. Only when a migration was
                # supplied: without one, this reader has been given a file it
                # has no account of, and reading it as version 0 would be a
                # guess wearing a rule's clothes.
                if self._migrate is None:
                    where = self._put_aside("not_a_record")
                    raise CannotRead(
                        f"{self._path.name} is not a versioned record",
                        put_aside_at=where,
                    )
                raw = {"major": 0, "minor": 0, "data": raw, "written_at": 0.0}
            if "data" not in raw:
                where = self._put_aside("not_a_record")
                raise CannotRead(
                    f"{self._path.name} is not a versioned record", put_aside_at=where
                )

            major = int(raw.get("major", 0))
            minor = int(raw.get("minor", 0))
            data = raw.get("data")
            if not isinstance(data, dict):
                where = self._put_aside("data_is_not_an_object")
                raise CannotRead(
                    f"{self._path.name} holds {type(data).__name__} where an "
                    "object belongs",
                    put_aside_at=where,
                )

            if major > self._major:
                # Forward major versions are not readable. Nothing about this
                # reader lets it know what a later shape means, and acting on
                # a guess is how a rollback loses data it could have kept.
                raise CannotRead(
                    f"{self._path.name} was written at version {major}.{minor} "
                    f"and this reader knows {self._major}.{self._minor}"
                )
            if major < self._major:
                if self._migrate is None:
                    raise CannotRead(
                        f"{self._path.name} is at version {major}.{minor} and "
                        f"nothing was given to migrate it to {self._major}"
                    )
                data = dict(self._migrate(dict(data), major, minor))
                logger.info(
                    "migrated %s from %d.%d to %d.%d",
                    self._path.name, major, minor, self._major, self._minor,
                )
                major, minor = self._major, self._minor
            # A minor version ahead is fine: a newer writer may only add
            # fields, so what this reader understands is still there.
            return Kept(
                data=data,
                major=major,
                minor=minor,
                written_at=float(raw.get("written_at", 0.0)),
            )

    # -------------------------------------------------------------- writing

    def hold(self, data: dict[str, Any]) -> None:
        """Take a deep copy now, write it later.

        The copy is the point. A caller that hands over its live dict and
        keeps mutating it has written whatever the dict happened to hold when
        the flush ran, which is a different thing from what it meant to save.
        """
        if self._read_only:
            raise PermissionError(f"{self._path.name} is open read-only")
        with self._lock:
            self._pending = copy.deepcopy(dict(data))

    def flush(self) -> bool:
        """Write what is held. False when there is nothing to write."""
        with self._lock:
            if self._read_only or self._pending is None:
                return False
            body = json.dumps(
                {
                    "major": self._major,
                    "minor": self._minor,
                    "written_at": time.time(),
                    "data": self._pending,
                },
                indent=2,
                sort_keys=True,
                default=str,
            )
            try:
                from core.runtime.file_write_gateway import get_file_write_gateway

                gateway = get_file_write_gateway()
                gateway.ensure_directory(self._path.parent, source=self._source)
                gateway.write_text(self._path, body, source=self._source)
            except Exception as exc:  # noqa: BLE001 — a failed save is not a crash
                logger.warning("%s was not saved: %s", self._path.name, exc)
                return False
            self._written_at = time.time()
            self._pending = None
            return True

    def save(self, data: dict[str, Any]) -> bool:
        """Hold and flush in one step."""
        self.hold(data)
        return self.flush()

    def final_write(self) -> bool:
        """The write that happens as the process goes down.

        Separate from ``flush`` so a shutdown path can be read for what it
        does: nothing held means nothing outstanding, and that is worth
        saying rather than inferring from a False.
        """
        with self._lock:
            outstanding = self._pending is not None
        if not outstanding:
            logger.debug("%s had nothing outstanding at shutdown", self._path.name)
            return False
        return self.flush()

    # ------------------------------------------------------------ internals

    def _put_aside(self, why: str) -> Path | None:
        """Move an unreadable file rather than write over it.

        Caller holds the lock. A store that overwrites corruption has
        destroyed the only evidence of what went wrong, and the first question
        after any corruption is what the bytes were.
        """
        if self._read_only:
            return None
        where = self._path.with_name(
            f"{self._path.stem}.{why}.{int(time.time())}{self._path.suffix}"
        )
        try:
            from core.runtime.file_write_gateway import get_file_write_gateway

            gateway = get_file_write_gateway()
            gateway.write_text(
                where, self._path.read_bytes().decode("utf-8", "replace"),
                source=f"{self._source}.quarantine",
            )
            gateway.delete_file(self._path, source=f"{self._source}.quarantine")
        except Exception as exc:  # noqa: BLE001 — failing to quarantine is not fatal
            logger.warning("%s could not be put aside: %s", self._path.name, exc)
            return None
        logger.warning("%s was unreadable (%s); put aside at %s",
                       self._path.name, why, where.name)
        return where

    def report(self) -> dict[str, Any]:
        with self._lock:
            return {
                "path": str(self._path),
                "version": f"{self._major}.{self._minor}",
                "read_only": self._read_only,
                "holding": self._pending is not None,
                "written_at": self._written_at,
                "exists": self._path.exists(),
            }
