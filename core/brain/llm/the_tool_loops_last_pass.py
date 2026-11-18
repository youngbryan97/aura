"""When her tool loop stops calling and writes what it found.

LIVE 18 November 2026 a question about her own source took ten calls in 24 minutes, the tenth on the loop's last turn,
and the time ran out before anything was written. The person got no answer, though each call had worked.
"""
from __future__ import annotations

__all__ = ["this_pass_writes"]


def this_pass_writes(turn: int, max_turns: int, write_by: float, longest_pass_s: float, now: float) -> bool:
    """Whether this pass is offered no tools: it is the last of several, or a call and then a reply would not fit by
    ``write_by``, going by the longest pass so far. A loop of one pass has only that pass to call in. ``write_by`` is
    on the monotonic clock; 0 means no limit is known."""
    if turn > 0 and turn + 1 >= max_turns:
        return True
    return write_by > 0 and longest_pass_s > 0 and now + 2 * longest_pass_s >= write_by
