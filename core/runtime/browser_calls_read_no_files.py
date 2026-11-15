"""Her browser's calls, without a walk of the file system on every one.

Playwright's Python client wraps each call to the browser (a key down, the
pointer moved, a frame acknowledged) in ``inspect.stack(0)``. For every frame
on the stack that asks the file system whether the frame's source file exists
and checks the line cache. The client then uses three things of each frame:
its file's name, its line, and the frame itself.

Measured 2026-10-12: 0.67 ms a call at a stack 45 deep, against 0.014 ms for
reading the same three things off the frames. Through one game 7% of her main
thread went to it, thirty frames a second acknowledged at that price, and
every key she pressed in time waited on it.

``install()`` gives the client's connection module a stand-in for ``inspect``
whose ``stack`` reads the frames directly. Everything else it is asked for is
the real module's. What the client makes of the stack (the name of the call
in its error messages, where it was called from) comes out the same; the test
holds the two side by side.

The client also keeps the last ten frames of each call as a traceback, for an
error to show. ``traceback.extract_stack`` checks every one of those files
against the disk as it takes them. The stand-in for ``traceback`` takes the
same frames and leaves their lines to be read if the traceback is ever shown.
"""
from __future__ import annotations

import inspect
import sys
import traceback
from typing import Any

__all__ = ["install", "stack_read_off_the_frames"]


def stack_read_off_the_frames(skip: int = 1) -> list[inspect.FrameInfo]:
    """The caller's stack as ``inspect.stack(0)`` gives it, innermost first, read off the frames with no file looked at."""
    frame = sys._getframe(skip)
    frames: list[inspect.FrameInfo] = []
    while frame is not None:
        code = frame.f_code
        frames.append(inspect.FrameInfo(frame, code.co_filename, frame.f_lineno, code.co_name, None, None))
        frame = frame.f_back
    return frames


class _InspectThatReadsFramesOnly:
    """``inspect``, but for ``stack(0)``."""

    def __getattr__(self, name: str) -> Any:
        return getattr(inspect, name)

    @staticmethod
    def stack(context: int = 1) -> list[inspect.FrameInfo]:
        if context:
            # Lines of source asked for: the real one reads them, less this frame of ours.
            return inspect.stack(context)[1:]
        return stack_read_off_the_frames(skip=2)


class _TracebackThatReadsLinesWhenShown:
    """``traceback``, but for ``extract_stack``."""

    def __getattr__(self, name: str) -> Any:
        return getattr(traceback, name)

    @staticmethod
    def extract_stack(f: Any = None, limit: int | None = None) -> traceback.StackSummary:
        frame = f if f is not None else sys._getframe(1)
        stack = traceback.StackSummary.extract(traceback.walk_stack(frame), limit=limit, lookup_lines=False)
        stack.reverse()
        return stack


def install() -> bool:
    """Have the browser client read its stacks off the frames. Whether it does now: False where the client is not
    there, or is built some other way than the one this was measured on, and is left as it is."""
    try:
        from playwright._impl import _connection
    # not a failure: no browser client here, or one built otherwise: there is nothing to change.
    except Exception:  # noqa: BLE001 - no client, or one that will not import: nothing to speed up
        return False
    seen = getattr(_connection, "inspect", None)
    if isinstance(seen, _InspectThatReadsFramesOnly):
        return True
    if seen is not inspect:
        return False
    _connection.inspect = _InspectThatReadsFramesOnly()
    if getattr(_connection, "traceback", None) is traceback:
        _connection.traceback = _TracebackThatReadsLinesWhenShown()
    return True
