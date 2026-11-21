"""What each host she asks things of has left for her, as the host itself says.

A lesson from competitive Pokémon (core/cognition/what_an_act_costs_beyond_now.py): each act has its own count of uses,
and an act with none left is not an act. Every host she reaches is one. A host that answers "429 Too Many Requests"
has said it has none left for her for a while, and usually for how long; one that sends a count of requests remaining
has said how many she has. Asking again before that time spends a request on a refusal she was already given, and
looks to the host like a client that does not listen. Measured at Wikimedia: seven image topics asked in a tight loop,
and the last five came back "no image available" (core/skills/computer_use.py).

So what a host says of its own supply is kept, host by host: the time it asked her to wait until, from Retry-After on
a refusal (seconds, or a date), or from a count remaining of none with the time its count resets. Until then a request
to that host is answered here with the host's own refusal and the time still to wait, and does not go out. Nothing is
held on a guess: a host that refuses without saying for how long is asked again as before.

The count each host started with and has left is kept as well, for a caller deciding what a request is worth spending.
"""
from __future__ import annotations

import email.utils
import logging
import time
import urllib.parse
from dataclasses import dataclass
from typing import Any

from core.runtime.lockdep import LockRank, checked_lock

logger = logging.getLogger("Aura.NetworkGateway.WhatAHostHasLeft")

__all__ = ["HostSupply", "held_back", "heard_from", "what_each_host_has_left"]

#: The statuses with which a host says it has nothing for her yet.
_NOT_YET = (429, 503)
#: A reset written as a moment rather than a wait: larger than any wait a host would ask for in seconds.
_A_MOMENT = 1_000_000_000


@dataclass
class HostSupply:
    """What one host has said of its supply for her."""

    #: When, on the wall clock, it asked her to wait until; 0 where it has not.
    wait_until: float = 0.0
    #: How many requests it says she has left, and how many it allows, where it says.
    left: int | None = None
    allows: int | None = None
    #: How many times it has refused her and said for how long.
    refusals: int = 0


_HOSTS: dict[str, HostSupply] = {}
_LOCK = checked_lock("network_gateway.what_a_host_has_left", rank=LockRank.LEAF)


def _host(url: str) -> str:
    try:
        return (urllib.parse.urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""


def _header(headers: dict[str, Any], *names: str) -> str:
    lowered = {str(key).lower(): value for key, value in (headers or {}).items()}
    for name in names:
        value = lowered.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _seconds_from(text: str, now: float) -> float | None:
    """A wait as a host writes one: seconds, or an HTTP date to wait until."""
    if not text:
        return None
    try:
        return max(0.0, float(text))
    except ValueError:
        pass
    try:
        when = email.utils.parsedate_to_datetime(text)
    except (TypeError, ValueError, IndexError):
        return None
    if when is None:
        return None
    return max(0.0, when.timestamp() - now)


def _a_count(text: str) -> int | None:
    try:
        return int(float(text))
    except ValueError:
        return None


def held_back(url: str, *, now: float | None = None) -> dict[str, Any] | None:
    """The host's own refusal, where it has asked her to wait and the time has not come; None where she may ask."""
    host = _host(url)
    if not host:
        return None
    moment = time.time() if now is None else now
    with _LOCK:
        supply = _HOSTS.get(host)
        waiting = 0.0 if supply is None else supply.wait_until - moment
    if waiting <= 0:
        return None
    return {"status_code": 429, "headers": {"Retry-After": str(int(waiting + 0.999))}, "content": b"", "ok": False,
            "error": f"{host} asked her to wait before asking again; {waiting:.0f} s are still to go",
            "retry_after_s": waiting, "held_back": True}


def heard_from(url: str, status: int, headers: dict[str, Any] | None, *, now: float | None = None) -> None:
    """What a host's answer says of its supply: how many she has left and when it resets, or how long to wait."""
    host = _host(url)
    if not host:
        return
    moment = time.time() if now is None else now
    headers = headers or {}
    remaining = _a_count(_header(headers, "ratelimit-remaining", "x-ratelimit-remaining"))
    limit = _a_count(_header(headers, "ratelimit-limit", "x-ratelimit-limit"))
    reset_text = _header(headers, "ratelimit-reset", "x-ratelimit-reset")
    reset = _a_count(reset_text) if reset_text else None
    if reset is not None and reset > _A_MOMENT:
        reset = max(0, int(reset - moment))
    wait = _seconds_from(_header(headers, "retry-after"), moment) if status in _NOT_YET else None
    if wait is None and status in _NOT_YET and reset is not None:
        wait = float(reset)
    if wait is None and remaining == 0 and reset is not None:
        wait = float(reset)
    if wait is None and status in _NOT_YET:
        # Held on nothing: a refusal that names no time says nothing of when to ask again.
        logger.info("%s refused her (%d) and did not say how long to wait", host, status)
    if remaining is None and limit is None and wait is None:
        return
    with _LOCK:
        supply = _HOSTS.setdefault(host, HostSupply())
        if remaining is not None:
            supply.left = remaining
        if limit is not None:
            supply.allows = limit
        if wait is not None and wait > 0:
            supply.wait_until = max(supply.wait_until, moment + wait)
            if status in _NOT_YET:
                supply.refusals += 1


def what_each_host_has_left() -> dict[str, HostSupply]:
    """Her own copy of what each host has said of its supply."""
    with _LOCK:
        return {host: HostSupply(**vars(supply)) for host, supply in _HOSTS.items()}


def forget_hosts_for_test() -> None:
    with _LOCK:
        _HOSTS.clear()
