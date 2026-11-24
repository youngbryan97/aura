"""What her model has written so far of what it is writing, for whoever asked to watch it.

A generation is worth watching where somebody is watching the work it is for. The caller sets a watcher for its context;
the request then asks the worker to send what it has written with its progress, and each such message reaches the watcher.
Nothing is sent for any other generation.
"""
from __future__ import annotations

import contextvars
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

__all__ = ["MOST_SENT", "pass_on", "watch_for", "watched", "watcher"]

#: The most of what was written that a progress message carries: its end.
MOST_SENT = 24000

Watcher = Callable[[str, int], None]
_WATCHER: contextvars.ContextVar[Watcher | None] = contextvars.ContextVar("aura_what_is_being_written", default=None)
_FOR: dict[str, Watcher] = {}


@contextmanager
def watched(by: Watcher) -> Iterator[None]:
    """What is written in this context is passed to ``by(text, tokens)`` as it is."""
    token = _WATCHER.set(by)
    try:
        yield
    finally:
        _WATCHER.reset(token)


def watcher() -> Watcher | None:
    return _WATCHER.get()


def watch_for(request_id: str) -> bool:
    """Pass what is written for this request to this context's watcher; whether there is one."""
    by = _WATCHER.get()
    if by is None:
        return False
    _FOR[str(request_id)] = by
    while len(_FOR) > 32:
        _FOR.pop(next(iter(_FOR)))
    return True


def pass_on(message: Any) -> None:
    """Pass a worker's progress message to its watcher, where it has one."""
    by = _FOR.get(str(message.get("id"))) if isinstance(message, dict) else None
    if by is not None and isinstance(message.get("written"), str):
        by(message["written"], int(message.get("tokens_generated") or 0))


def forget(request_id: str) -> None:
    _FOR.pop(str(request_id), None)
