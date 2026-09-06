"""One turn state, one lease, and cancelling is not idle.

An external blind comparison scored Letta's runtime highest, and said why: its
turn lifecycle is a discriminated state machine, every active turn owns an
immutable lease carrying an id and an abort signal, mutation APIs verify the
lease is still the current owner, and cancellation is not treated as
instantaneous idleness — the state stays ``cancelling`` until both the owner
and the external cleanup have settled.

That last part is the one that matters. A runtime that snaps to ``idle`` the
moment cancel is called will start the next turn on top of the one it is
still tearing down, and every symptom of that looks like something else: a
reply attributed to the wrong turn, a generation nobody asked for, a stop
button that appeared to work.

So the states are:

* ``idle`` — nothing owns the runtime.
* ``command`` — a takeover is being arranged. Brief and atomic: a command
  either becomes the active turn or leaves the previous one alone.
* ``active`` — one lease owns it.
* ``cancelling`` — the owner has been told to stop. Two things have to settle
  before this becomes ``idle``, in either order: the owner finishes, and the
  external cleanup reports. Whichever arrives second is the one that ends it.

Nothing here cancels anything by itself. It holds who owns the turn and what
has settled; the owner reads its own lease's ``stopping`` and stops.
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from core.runtime.lockdep import checked_lock
from core.runtime.what_stops_it import Stopping

logger = logging.getLogger("Aura.WhoseTurnItIs")

__all__ = [
    "ALease",
    "NotTheOwner",
    "TheTurn",
    "TurnStatus",
    "the_turn",
]


class TurnStatus(StrEnum):
    """The four states. Nothing else is a state."""

    IDLE = "idle"
    COMMAND = "command"
    ACTIVE = "active"
    CANCELLING = "cancelling"


class NotTheOwner(RuntimeError):
    """A lease that is not the current one tried to change the turn.

    Raised rather than ignored. A stale owner writing into a live turn is the
    defect this whole module exists to make impossible, and swallowing it
    would turn a loud wrong answer into a quiet one.
    """


@dataclass(frozen=True)
class ALease:
    """Proof that a particular turn owns the runtime. Immutable on purpose.

    Handed to the owner when the turn begins and never handed out again. An
    owner holding a lease can be told, by the lease itself, whether it is
    still the owner — which a bare turn id cannot do, because an id can be
    copied and a copy of an id is indistinguishable from the original.
    """

    turn_id: str
    stopping: Stopping
    began_at: float = field(default_factory=time.monotonic)
    origin: str = "user"

    @property
    def held_for_s(self) -> float:
        return time.monotonic() - self.began_at


@dataclass
class _Settling:
    """What still has to report before a cancelled turn is over."""

    owner_finished: bool = False
    cleanup_reported: bool = False

    @property
    def settled(self) -> bool:
        return self.owner_finished and self.cleanup_reported


class TheTurn:
    """The one holder of who owns the runtime right now."""

    def __init__(self) -> None:
        self._lock = checked_lock("whose_turn_it_is")
        self._status = TurnStatus.IDLE
        self._lease: ALease | None = None
        self._settling: _Settling | None = None
        self._why_cancelled = ""
        #: Every turn that has held it, newest last, capped. For reading a
        #: sequence back, not for storing conversation.
        self._history: list[tuple[str, str, float]] = []

    # ----------------------------------------------------------- reading

    @property
    def status(self) -> TurnStatus:
        with self._lock:
            return self._status

    @property
    def lease(self) -> ALease | None:
        with self._lock:
            return self._lease

    def owns_it(self, lease: ALease | None) -> bool:
        """Whether this lease is the current owner. The one question to ask."""
        with self._lock:
            return lease is not None and self._lease is lease

    def report(self) -> dict[str, Any]:
        with self._lock:
            settling = self._settling
            return {
                "status": str(self._status),
                "turn_id": self._lease.turn_id if self._lease else "",
                "origin": self._lease.origin if self._lease else "",
                "held_for_s": round(self._lease.held_for_s, 3) if self._lease else 0.0,
                "why_cancelled": self._why_cancelled,
                "waiting_on": [] if settling is None else [
                    name
                    for name, done in (
                        ("the owner to finish", settling.owner_finished),
                        ("the cleanup to report", settling.cleanup_reported),
                    )
                    if not done
                ],
                "turns_held": len(self._history),
            }

    # ----------------------------------------------------------- writing

    def begin(self, *, origin: str = "user", turn_id: str = "") -> ALease:
        """Take the turn. Refuses unless nothing else owns it.

        ``NotTheOwner`` rather than a bare False: a caller that starts working
        after a refused begin is the overlapping-ownership defect, and a
        boolean is the easiest thing in the world not to read.
        """
        with self._lock:
            if self._status is not TurnStatus.IDLE:
                raise NotTheOwner(
                    f"cannot begin a turn while the runtime is {self._status}"
                    + (f" ({self._lease.turn_id})" if self._lease else "")
                )
            return self._install(origin=origin, turn_id=turn_id)

    def take_over(self, *, origin: str = "command", turn_id: str = "") -> ALease:
        """Replace whatever is running with this turn, atomically.

        The command case. Either this returns a lease that owns the runtime,
        or it raises and the previous turn is untouched — never a window where
        the old owner has been told to stop and nothing owns the runtime yet.
        """
        with self._lock:
            previous = self._lease
            self._status = TurnStatus.COMMAND
            if previous is not None:
                previous.stopping.stop("taken over by a command")
            lease = self._install(origin=origin, turn_id=turn_id)
            if previous is not None:
                logger.info(
                    "🔁 turn %s took over from %s", lease.turn_id, previous.turn_id
                )
            return lease

    def cancel(self, why: str = "") -> bool:
        """Tell the owner to stop. The turn is not over when this returns.

        Returns whether there was anything to cancel. The state becomes
        ``cancelling`` and stays there until the owner finishes AND the
        cleanup reports — snapping straight to idle here is how the next turn
        starts on top of the one still being torn down.
        """
        with self._lock:
            if self._status in (TurnStatus.IDLE, TurnStatus.CANCELLING):
                return False
            lease = self._lease
            self._status = TurnStatus.CANCELLING
            self._why_cancelled = why or "cancelled"
            self._settling = _Settling()
            if lease is not None:
                lease.stopping.stop(self._why_cancelled)
            return True

    def finish(self, lease: ALease) -> TurnStatus:
        """The owner is done. Only the owner may say so."""
        with self._lock:
            self._must_own(lease)
            if self._status is TurnStatus.CANCELLING:
                assert self._settling is not None
                self._settling.owner_finished = True
                return self._settle_if_ready()
            return self._go_idle()

    def cleanup_reported(self, lease: ALease) -> TurnStatus:
        """External cleanup for a cancelled turn has settled.

        Takes the lease so a report about a turn that is already gone cannot
        end the turn that replaced it — the cleanup arriving late is exactly
        the case this is for.
        """
        with self._lock:
            self._must_own(lease)
            if self._status is not TurnStatus.CANCELLING:
                return self._status
            assert self._settling is not None
            self._settling.cleanup_reported = True
            return self._settle_if_ready()

    def reset(self, why: str = "reset") -> None:
        """Drop everything. Every outstanding lease becomes stale.

        For a runtime coming up or going down, not for ending a turn.
        """
        with self._lock:
            if self._lease is not None:
                self._lease.stopping.stop(why)
            self._status = TurnStatus.IDLE
            self._lease = None
            self._settling = None
            self._why_cancelled = ""

    # ---------------------------------------------------------- internals

    def _install(self, *, origin: str, turn_id: str) -> ALease:
        """Caller holds the lock."""
        chosen = str(turn_id or uuid.uuid4().hex)
        lease = ALease(
            turn_id=chosen, stopping=Stopping(f"turn:{chosen}"), origin=str(origin)
        )
        self._lease = lease
        self._status = TurnStatus.ACTIVE
        self._settling = None
        self._why_cancelled = ""
        self._history.append((chosen, str(origin), time.time()))
        del self._history[:-64]
        return lease

    def _must_own(self, lease: ALease) -> None:
        """Caller holds the lock."""
        if self._lease is not lease:
            raise NotTheOwner(
                f"lease {getattr(lease, 'turn_id', '?')} is not the current owner"
                + (f" ({self._lease.turn_id})" if self._lease else " (nothing is)")
            )

    def _settle_if_ready(self) -> TurnStatus:
        """Caller holds the lock."""
        assert self._settling is not None
        if not self._settling.settled:
            return TurnStatus.CANCELLING
        return self._go_idle()

    def _go_idle(self) -> TurnStatus:
        """Caller holds the lock."""
        self._status = TurnStatus.IDLE
        self._lease = None
        self._settling = None
        self._why_cancelled = ""
        return TurnStatus.IDLE


_THE_TURN = TheTurn()


def the_turn() -> TheTurn:
    """The one turn state for this process."""
    return _THE_TURN
