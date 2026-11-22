"""How long a change takes to settle, learned by watching changes settle.

Anything watched while it changes (a screen drawing a move, a page filling in, a gauge swinging) shows one reading,
then others, then stays. A reading is believed once it has stayed: read twice, the second look as long after the first
as changes here have been seen to keep going. That wait is learned. Each change watched to its end says how long it
kept changing, from the first look that showed it changing to the last; the wait is the longest of the changes watched
of late. A reading believed and then found to have been caught mid-change says the wait was too short, and counts as a
change that kept going for as long as the wait first given.

Nothing here knows what is watched.
"""
from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["HowLongItSettles"]

#: Changes remembered: the wait is the longest of these. The wait first given counts as one of them, so it is let go
#: only once this many changes have been watched since.
REMEMBERED = 8


@dataclass
class HowLongItSettles:
    """What one watcher has seen of how long changes take to settle where it watches."""

    first_wait: float
    kept_changing: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.kept_changing:
            self.kept_changing = [float(self.first_wait)]

    def wait(self) -> float:
        """How long after one look the next is taken, for the two to say whether it has stayed."""
        return max(self.kept_changing[-REMEMBERED:])

    def watched(self, kept_changing_s: float) -> None:
        """A change watched to its end kept changing for ``kept_changing_s`` between the looks that saw it change."""
        self.kept_changing = [*self.kept_changing[-(REMEMBERED - 1):], max(0.0, float(kept_changing_s))]

    def too_soon(self) -> None:
        """A reading believed was caught mid-change: the wait first given is waited again until changes since say
        otherwise."""
        self.watched(self.first_wait)
