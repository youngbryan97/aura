"""Raw Chrome DevTools Protocol websocket transport.

This is the only sanctioned home for the raw CDP websocket send/receive pair.
Higher-level capabilities (visible web interlocutor, browser controllers) must
route their CDP traffic through :func:`cdp_call` so raw environment sinks stay
inside the approved adapter layer, per the no-raw-bypass final blocker.

Being the sanctioned sink is exactly why it needs a contract. CDP can navigate,
read any page's DOM, run arbitrary JavaScript, download files and clear
cookies, so "route it through here" only helps if *here* checks something.
CP126 found it opening any URL a caller passed, forwarding any method with no
allowlist or receipt, dropping every event that was not the reply it wanted,
and reading frames of unbounded size.

CP126 57ab9887 / 45ccffeb / f47118c0 / fc411124 / 7e9bb8a8 / e39c11e5.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import time
import urllib.parse
from typing import Any

logger = logging.getLogger("Aura.ChromeCDP")

#: A debugger endpoint is a local-only surface. CP126 57ab9887: any
#: ``target_ws_url`` was opened, so a caller could point this at a remote host.
_ALLOWED_WS_SCHEMES = frozenset({"ws", "wss"})
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "[::1]"})

#: Frames larger than this are refused rather than buffered (CP126 e39c11e5).
MAX_FRAME_BYTES = 8 * 1024 * 1024

#: Events observed while waiting for the reply, kept rather than discarded.
MAX_RETAINED_EVENTS = 64

MIN_TIMEOUT_S = 0.1
MAX_TIMEOUT_S = 120.0
DEFAULT_TIMEOUT_S = 5.0

#: CDP methods this adapter will forward. CP126 45ccffeb: the sanctioned sink
#: accepted ANY method with no scoped authority or destructive-action class,
#: which made "route raw CDP through the adapter" a naming convention rather
#: than a control.
READ_METHODS = frozenset({
    "Browser.getVersion",
    "DOM.getDocument",
    "DOM.getOuterHTML",
    "Page.getFrameTree",
    "Page.getLayoutMetrics",
    "Page.captureScreenshot",
    "Runtime.getProperties",
    "Target.getTargets",
    "Target.getTargetInfo",
    "Network.getCookies",
    "Accessibility.getFullAXTree",
})

#: Methods that change page or browser state. Permitted, but classified and
#: receipted so a caller cannot mistake one for a read.
MUTATING_METHODS = frozenset({
    "Page.bringToFront",
    "Page.navigate",
    "Page.reload",
    "Input.dispatchKeyEvent",
    "Input.dispatchMouseEvent",
    "Input.insertText",
    "Runtime.evaluate",
    "Runtime.callFunctionOn",
    "DOM.setOuterHTML",
    "Emulation.setDeviceMetricsOverride",
    "Page.enable",
    "DOM.enable",
    "Runtime.enable",
    "Network.enable",
    "Accessibility.enable",
})

#: Methods that destroy state or reach outside the page. Refused unless the
#: caller passes ``allow_destructive=True`` and says why.
DESTRUCTIVE_METHODS = frozenset({
    "Network.clearBrowserCookies",
    "Network.clearBrowserCache",
    "Storage.clearDataForOrigin",
    "Browser.close",
    "Target.closeTarget",
    "Page.setDownloadBehavior",
    "Browser.setDownloadBehavior",
    "Page.handleJavaScriptDialog",
})

ALLOWED_METHODS = READ_METHODS | MUTATING_METHODS | DESTRUCTIVE_METHODS


class CdpPolicyError(RuntimeError):
    """The request was refused before any websocket was opened."""


def classify_method(method: str) -> str:
    """``read`` | ``mutating`` | ``destructive`` | ``unknown``."""
    name = str(method or "").strip()
    if name in READ_METHODS:
        return "read"
    if name in MUTATING_METHODS:
        return "mutating"
    if name in DESTRUCTIVE_METHODS:
        return "destructive"
    return "unknown"


def validate_target(target_ws_url: Any) -> str:
    """A loopback CDP websocket URL, or raise.

    CP126 57ab9887: the caller controlled the raw destination with no scheme,
    host, port or session check, so this sink could be aimed anywhere.
    """
    url = str(target_ws_url or "").strip()
    if not url:
        raise CdpPolicyError("CDP target URL is empty")
    if any(char in url for char in ("\n", "\r", "\x00", " ")):
        raise CdpPolicyError("CDP target URL contains control characters")
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in _ALLOWED_WS_SCHEMES:
        raise CdpPolicyError(f"CDP target scheme {parsed.scheme!r} is not a websocket scheme")
    host = (parsed.hostname or "").lower()
    if host not in _LOOPBACK_HOSTS:
        # A debugger port reachable off-box is a remote-control channel.
        raise CdpPolicyError(f"CDP target host {host!r} is not loopback")
    try:
        port = parsed.port
    except ValueError as exc:
        raise CdpPolicyError("CDP target port is invalid") from exc
    if port is not None and not (1 <= port <= 65535):
        raise CdpPolicyError(f"CDP target port {port} is out of range")
    return url


def validated_timeout(timeout: Any) -> float:
    """A finite, positive, bounded timeout.

    CP126 7e9bb8a8: a non-positive or non-finite value was passed straight to
    the websocket library, whose own recv timeout could then fire with a
    different exception than the documented TimeoutError.
    """
    try:
        value = float(timeout)
    except (TypeError, ValueError):
        return DEFAULT_TIMEOUT_S
    if not math.isfinite(value) or value <= 0:
        return DEFAULT_TIMEOUT_S
    return max(MIN_TIMEOUT_S, min(MAX_TIMEOUT_S, value))


def _authorize(method: str, allow_destructive: bool, reason: str) -> str:
    """Policy gate. Returns the method class or raises CdpPolicyError."""
    kind = classify_method(method)
    if kind == "unknown":
        raise CdpPolicyError(
            f"CDP method {method!r} is not in the adapter allowlist; add it "
            "with an explicit classification rather than forwarding blind"
        )
    if kind == "destructive" and not allow_destructive:
        raise CdpPolicyError(
            f"CDP method {method!r} is destructive; pass allow_destructive=True "
            "with a reason to proceed"
        )
    if kind == "destructive" and not str(reason or "").strip():
        raise CdpPolicyError("a destructive CDP call must state a reason")
    return kind


def cdp_call(
    target_ws_url: str,
    method: str,
    params: dict[str, Any],
    *,
    timeout: float = DEFAULT_TIMEOUT_S,
    allow_destructive: bool = False,
    reason: str = "",
) -> dict[str, Any]:
    """Execute one CDP JSON-RPC request and wait for its matching response.

    Raises :class:`CdpPolicyError` when the target or method is refused,
    RuntimeError for a missing dependency or a CDP-level error, and
    TimeoutError when the response never arrives inside ``timeout`` seconds.

    Blocking. Async callers must use :func:`cdp_call_async` — CP126 f47118c0:
    connection, send, the receive loop, JSON parsing and close all block, so
    calling this on an event loop stalls the runtime for up to the timeout.
    """
    url = validate_target(target_ws_url)
    kind = _authorize(method, allow_destructive, reason)
    budget = validated_timeout(timeout)

    try:
        import websocket
    except ImportError as exc:
        raise RuntimeError("websocket-client is required for Chrome CDP control") from exc

    started = time.monotonic()
    # CP126 45ccffeb: an audit receipt for a raw browser-control call.
    receipt: dict[str, Any] = {
        "method": method,
        "method_class": kind,
        "target": url,
        "reason": reason,
        "allow_destructive": bool(allow_destructive),
        "at": time.time(),
    }
    if kind != "read":
        logger.info("🖥️ CDP %s call: %s (%s)", kind, method, reason or "no reason given")

    ws = websocket.create_connection(url, timeout=budget)
    events: list[dict[str, Any]] = []
    try:
        message_id = 1
        ws.send(json.dumps({"id": message_id, "method": method, "params": params or {}}))
        deadline = time.monotonic() + budget
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Timed out waiting for Chrome CDP method {method}")
            try:
                ws.settimeout(remaining)
            except (AttributeError, OSError):
                # not a failure: the deadline above is the real bound, and
                # the socket timeout only narrows how long one recv waits.
                pass
            try:
                raw = ws.recv()
            except Exception as exc:  # noqa: BLE001 - normalized below
                # CP126 7e9bb8a8: the library raises its own timeout type; the
                # documented contract is TimeoutError.
                if _is_timeout(exc):
                    raise TimeoutError(
                        f"Timed out waiting for Chrome CDP method {method}"
                    ) from exc
                raise
            if not raw:
                continue
            # CP126 e39c11e5: refuse an oversized frame instead of parsing it.
            size = len(raw) if isinstance(raw, (bytes, bytearray, str)) else 0
            if size > MAX_FRAME_BYTES:
                raise RuntimeError(
                    f"CDP frame of {size} bytes exceeds the {MAX_FRAME_BYTES}-byte bound"
                )
            try:
                data = json.loads(raw)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise RuntimeError(f"Chrome CDP returned unparseable JSON: {exc}") from exc
            if not isinstance(data, dict):
                continue
            if data.get("id") == message_id:
                receipt["elapsed_s"] = round(time.monotonic() - started, 4)
                receipt["events_observed"] = len(events)
                if "error" in data:
                    receipt["ok"] = False
                    raise RuntimeError(str(data["error"]))
                receipt["ok"] = True
                # CP126 fc411124: events seen while waiting are RETURNED, not
                # silently dropped — a higher-level state machine needs them.
                return {"result": data, "events": events, "receipt": receipt}
            if len(events) < MAX_RETAINED_EVENTS:
                events.append(data)
            elif len(events) == MAX_RETAINED_EVENTS:
                events.append({"_truncated": True, "reason": "event buffer full"})
    finally:
        try:
            ws.close()
        except Exception as exc:  # noqa: BLE001 - close must not mask the result
            logger.debug("CDP websocket close failed: %s", exc)


def _is_timeout(exc: BaseException) -> bool:
    """Whether an exception from the websocket library means "timed out"."""
    if isinstance(exc, TimeoutError):
        return True
    name = type(exc).__name__.lower()
    return "timeout" in name or "timed out" in str(exc).lower()


async def cdp_call_async(
    target_ws_url: str,
    method: str,
    params: dict[str, Any],
    *,
    timeout: float = DEFAULT_TIMEOUT_S,
    allow_destructive: bool = False,
    reason: str = "",
) -> dict[str, Any]:
    """Await one CDP call without blocking the event loop.

    CP126 f47118c0: the transport is entirely synchronous, so an async caller
    invoking :func:`cdp_call` directly stalls its runtime for up to the network
    timeout. The policy checks run inline (they are cheap and must refuse
    before a thread is spent); only the blocking I/O is offloaded.
    """
    validate_target(target_ws_url)
    _authorize(method, allow_destructive, reason)
    budget = validated_timeout(timeout)
    return await asyncio.wait_for(
        asyncio.to_thread(
            cdp_call,
            target_ws_url,
            method,
            params,
            timeout=budget,
            allow_destructive=allow_destructive,
            reason=reason,
        ),
        # A little headroom so the inner deadline reports first, with its
        # method name attached.
        timeout=budget + 5.0,
    )
