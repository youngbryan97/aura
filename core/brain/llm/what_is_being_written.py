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

__all__ = ["more_of", "pass_on", "watch_for", "watched", "watcher"]

Watcher = Callable[[str, int], None]
_WATCHER: contextvars.ContextVar[Watcher | None] = contextvars.ContextVar("aura_what_is_being_written", default=None)
_FOR: dict[str, Watcher] = {}
#: What has come so far of what is being written for each request that is watched.
_SO_FAR: dict[str, str] = {}


def more_of(text: str, job: dict[str, Any]) -> dict[str, Any]:
    """What a progress message carries of what has been written: what is new since the last one, and where it goes on from.
    The whole of it every few tokens is most of a megabyte a second by the end of a long one."""
    at, before = int(job.get("_written_sent") or 0), str(job.get("_written_before") or "")
    if at > len(text) or text[max(0, at - 40):at] != before:
        at = 0
    job["_written_sent"], job["_written_before"] = len(text), text[-40:]
    return {"written": text[at:], "written_at": at}


@contextmanager
def watched(by: Watcher) -> Iterator[None]:
    """What is written in this context is passed to ``by(text, tokens)`` as it is, and how much of what it is given has been
    read to ``by.reading(read, of)`` where it has one."""
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
        forget(next(iter(_FOR)))
    return True


def pass_on(message: Any) -> None:
    """Pass a worker's progress message to its watcher, where it has one."""
    request_id = str(message.get("id")) if isinstance(message, dict) else ""
    by = _FOR.get(request_id)
    if by is None:
        return
    if message.get("phase") == "prefill":
        reading = getattr(by, "reading", None)
        if reading is not None:
            reading(int(message.get("prompt_tokens_processed") or 0), int(message.get("prompt_tokens_total") or 0))
    elif isinstance(message.get("written"), str):
        _SO_FAR[request_id] = _SO_FAR.get(request_id, "")[:max(0, int(message.get("written_at") or 0))] + message["written"]
        by(_SO_FAR[request_id], int(message.get("tokens_generated") or 0))


def forget(request_id: str) -> None:
    _FOR.pop(str(request_id), None)
    _SO_FAR.pop(str(request_id), None)
