"""Having a faculty and being able to use it are different facts.

A person who knows how to search the web does not get confused when the wifi
drops. They do not conclude they have forgotten how to search. They separate
two things without effort:

    the capability      "I know how to look things up"      — still true
    its precondition    "there is a network"                — false right now

and they combine them into a third thing they can say out loud: "I can't look
that up at the moment, there's no internet."

Aura had only the first axis. The capability registry knows which skills are
registered and enabled, and nothing in it knows whether the world outside the
process is currently in a state where those skills can do anything. So a
missing network looked identical to a missing skill, and she would either
claim a capability that could not possibly work, or deny one she has.

This module is the second axis. It probes the preconditions themselves —
cheaply, with a short cache, never blocking a turn — and reports them as
facts. Composition happens in capability_condition, so the derived verdict is
computed, not asserted: capability AND preconditions => usable. Remove the
network and the conclusion changes on its own, without a prompt mentioning
networks anywhere.

Preconditions are declared per capability rather than inferred from names, so
adding a skill means declaring what the world must provide for it — which is
the honest place for that knowledge to live.
"""

from __future__ import annotations

import logging
import os
import socket
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("Aura.CapabilityPreconditions")

__all__ = [
    "PreconditionState",
    "declared_preconditions",
    "failing_preconditions",
    "precondition_state",
    "request_precondition_refresh",
    "reset_precondition_cache",
]

#: How long a probe result stands. Long enough that a burst of turns costs one
#: probe; short enough that unplugging the network is noticed within a turn or
#: two.
_CACHE_TTL_SECONDS = 12.0

#: A probe must never be the reason a reply is slow.
_PROBE_TIMEOUT_SECONDS = 1.0

# Network availability is not synonymous with permission to reach a public DNS
# server on port 53. Corporate, hotel and mobile networks commonly block that
# traffic while their local resolver and HTTPS work normally. These independent
# providers plus direct TCP/443 reachability avoid making any one provider,
# resolver or protocol the source of truth.
_HTTPS_PROBE_URLS = (
    "https://www.apple.com/library/test/success.html",
    "https://connectivitycheck.gstatic.com/generate_204",
    "https://www.cloudflare.com/cdn-cgi/trace",
)
_TCP_HTTPS_TARGETS = (
    ("1.1.1.1", 443),
    ("8.8.8.8", 443),
)


@dataclass(frozen=True)
class PreconditionState:
    name: str
    satisfied: bool
    #: The plain fact, for her to say however she says things.
    fact: str
    #: True when the probe itself could not run. Unknown is not failure: a
    #: probe that cannot answer must not be reported as a missing world.
    unknown: bool = False


def _https_endpoint_reachable(url: str) -> bool:
    from core.runtime.network_gateway import get_network_gateway

    response = get_network_gateway().request(
        "HEAD",
        url,
        headers={"User-Agent": "Aura-Connectivity-Probe/1"},
        timeout=_PROBE_TIMEOUT_SECONDS,
        source="capability_preconditions.network_probe",
        read_only=True,
        suppress_degradation=True,
    )
    # Any valid HTTP response proves DNS, routing, TCP and TLS. The exact
    # status is not a service-health contract, including a legitimate 4xx/5xx.
    status = int(response.get("status_code", 0) or 0)
    return 100 <= status < 600


def _tcp_https_endpoint_reachable(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=_PROBE_TIMEOUT_SECONDS):
            return True
    # not a failure: a port that does not answer is not a reachable endpoint, which
    # is the question being asked.
    except (OSError, TimeoutError):
        return False


def _probe_network() -> PreconditionState:
    """Measure off-host reachability without depending on public DNS/53.

    A successful HTTPS observation is strongest because it exercises the path
    web capabilities use. Direct TCP/443 is an independent route signal when
    DNS or a connectivity-check provider is unavailable. Only unanimous,
    expected connection failures establish offline; an unexpected probe fault
    remains unknown rather than becoming a confident false negative.
    """
    unexpected_error = False
    for url in _HTTPS_PROBE_URLS:
        try:
            if _https_endpoint_reachable(url):
                return PreconditionState("network", True, "there is a network connection")
        except Exception as exc:  # noqa: BLE001 - a probe may never raise upward
            unexpected_error = True
            logger.debug("HTTPS network probe error for %s: %s", url, exc)

    for host, port in _TCP_HTTPS_TARGETS:
        try:
            if _tcp_https_endpoint_reachable(host, port):
                return PreconditionState("network", True, "there is a network connection")
        except Exception as exc:  # noqa: BLE001 - a probe may never raise upward
            unexpected_error = True
            logger.debug("TCP network probe error for %s:%s: %s", host, port, exc)

    if unexpected_error:
        return PreconditionState(
            "network", True, "network state could not be determined", unknown=True
        )
    return PreconditionState("network", False, "there is no network connection right now")


def _probe_desktop_session() -> PreconditionState:
    """Is there a windowing session to drive?

    Over SSH or in a headless daemon there is no desktop to click, and that
    is a fact about the world rather than about her.
    """
    if os.environ.get("AURA_HEADLESS") == "1":
        return PreconditionState("desktop_session", False, "this runtime is headless")
    try:
        if os.uname().sysname == "Darwin":
            # A login session that owns the window server has these set; a
            # bare daemon does not.
            has_session = bool(os.environ.get("SSH_CONNECTION")) is False
            return PreconditionState(
                "desktop_session",
                has_session,
                "there is a desktop session"
                if has_session
                else "this session has no desktop attached",
            )
    except Exception as exc:  # noqa: BLE001
        logger.debug("Desktop session probe error: %s", exc)
        return PreconditionState(
            "desktop_session", True, "desktop session state is unknown", unknown=True
        )
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return PreconditionState("desktop_session", True, "there is a desktop session")
    return PreconditionState("desktop_session", False, "there is no desktop session attached")


def _probe_writable_home() -> PreconditionState:
    try:
        home = os.path.expanduser("~")
        writable = os.access(home, os.W_OK)
        return PreconditionState(
            "writable_storage",
            writable,
            "local storage is writable" if writable else "local storage is not writable",
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("Storage probe error: %s", exc)
        return PreconditionState("writable_storage", True, "storage state is unknown", unknown=True)


def _probe_accessibility_permission() -> PreconditionState:
    """Has macOS granted this process the Accessibility right?

    Reading the screen through the accessibility API needs a grant a person
    makes once in System Settings. Without it every read returns
    "[Accessibility error or UI unresponsive]" — which is a fact about
    permission, not about her, and she should be able to say which.
    """
    if sys.platform != "darwin":
        return PreconditionState(
            "accessibility_permission", True, "not applicable on this platform"
        )
    try:
        from ApplicationServices import AXIsProcessTrusted

        granted = bool(AXIsProcessTrusted())
        return PreconditionState(
            "accessibility_permission",
            granted,
            "macOS Accessibility access is granted"
            if granted
            else "macOS has not granted Accessibility access to this app",
        )
    except Exception as exc:  # noqa: BLE001 - a probe may never raise upward
        logger.debug("Accessibility probe error: %s", exc)
        return PreconditionState(
            "accessibility_permission", True, "permission state is unknown", unknown=True
        )


_PROBES: dict[str, Callable[[], PreconditionState]] = {
    "network": _probe_network,
    "desktop_session": _probe_desktop_session,
    "writable_storage": _probe_writable_home,
    "accessibility_permission": _probe_accessibility_permission,
}

#: What the world must provide for each capability to be usable at all.
#:
#: Declared, not guessed from the name: a new skill states what it needs, and
#: the reasoning above it keeps working without being taught about that skill.
_CAPABILITY_PRECONDITIONS: dict[str, tuple[str, ...]] = {
    "web_search": ("network",),
    "sovereign_browser": ("network", "desktop_session"),
    "browser_action": ("network", "desktop_session"),
    "email_adapter": ("network",),
    "reddit_adapter": ("network",),
    "computer_use": ("desktop_session", "accessibility_permission"),
    "desktop_task": ("desktop_session", "accessibility_permission"),
    "os_manipulation": ("desktop_session", "accessibility_permission"),
    "build_app": ("writable_storage",),
    "file_operation": ("writable_storage",),
}

_CACHE: dict[str, tuple[float, PreconditionState]] = {}
_CACHE_LOCK = threading.Lock()
_CACHE_EPOCH = 0
_IN_FLIGHT: dict[str, int] = {}


def reset_precondition_cache() -> None:
    global _CACHE_EPOCH
    with _CACHE_LOCK:
        _CACHE_EPOCH += 1
        _CACHE.clear()
        _IN_FLIGHT.clear()


def _run_probe_refresh(
    key: str,
    probe: Callable[[], PreconditionState],
    epoch: int,
) -> None:
    try:
        state = probe()
        if not isinstance(state, PreconditionState) or state.name != key:
            raise TypeError(f"probe {key!r} returned an invalid precondition state")
    except Exception as exc:  # noqa: BLE001 - terminal background boundary
        logger.debug("Precondition probe %s failed: %s", key, exc)
        state = PreconditionState(
            key,
            True,
            f"{key.replace('_', ' ')} state could not be determined",
            unknown=True,
        )

    with _CACHE_LOCK:
        if epoch != _CACHE_EPOCH:
            return
        _CACHE[key] = (time.monotonic(), state)
        if _IN_FLIGHT.get(key) == epoch:
            _IN_FLIGHT.pop(key, None)


def request_precondition_refresh(name: str) -> bool:
    """Schedule one bounded refresh without waiting for external I/O.

    Returns ``True`` only for the caller that started the generation's worker.
    Runtime services can use this to prewarm the cache; chat callers get the
    same single-flight behavior through :func:`precondition_state`.
    """
    key = str(name or "").strip()
    probe = _PROBES.get(key)
    if probe is None:
        return False

    with _CACHE_LOCK:
        cached = _CACHE.get(key)
        if cached is not None and (time.monotonic() - cached[0]) < _CACHE_TTL_SECONDS:
            return False
        epoch = _CACHE_EPOCH
        if _IN_FLIGHT.get(key) == epoch:
            return False
        _IN_FLIGHT[key] = epoch

    threading.Thread(
        target=_run_probe_refresh,
        args=(key, probe, epoch),
        name=f"AuraPreconditionProbe-{key}",
        daemon=True,
    ).start()
    return True


def precondition_state(name: str) -> PreconditionState:
    """Return a fresh cached state and refresh misses without blocking chat."""
    key = str(name or "").strip()
    probe = _PROBES.get(key)
    if probe is None:
        return PreconditionState(key, True, "no precondition declared", unknown=True)

    now = time.monotonic()
    with _CACHE_LOCK:
        cached = _CACHE.get(key)
        if cached is not None and (now - cached[0]) < _CACHE_TTL_SECONDS:
            return cached[1]

    request_precondition_refresh(key)
    return PreconditionState(
        key,
        True,
        f"{key.replace('_', ' ')} state is being measured",
        unknown=True,
    )


def declared_preconditions(capability: Any) -> tuple[str, ...]:
    return _CAPABILITY_PRECONDITIONS.get(str(capability or "").strip(), ())


def failing_preconditions(capability: Any) -> tuple[PreconditionState, ...]:
    """Preconditions this capability needs that the world is not providing.

    An UNKNOWN probe never appears here. Reporting "there's no network"
    because a socket call raised something unexpected would be the same
    confident-lie failure as reporting a missing skill because a registry
    read failed.
    """
    failures: list[PreconditionState] = []
    for name in declared_preconditions(capability):
        state = precondition_state(name)
        if not state.satisfied and not state.unknown:
            failures.append(state)
    return tuple(failures)
