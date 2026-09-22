"""interface/auth.py
──────────────────
Extracted from server.py — shared authentication, authorization,
rate-limiting, and session management utilities used across route files.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import secrets
import threading
import time
from http.cookies import CookieError, SimpleCookie
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import SplitResult, urlsplit

from fastapi import Header, HTTPException, Request

from core.config import config
from core.runtime.errors import record_degradation

if TYPE_CHECKING:
    from core.security.device_pairing import PairedDevice

logger = logging.getLogger("Aura.Server.Auth")


# ── Constants ─────────────────────────────────────────────────

TRUSTED_IPS = {"127.0.0.1", "::1"}
TRUSTED_LOCAL_UI_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_BROWSER_ORIGIN_SCHEMES = frozenset({"http", "https"})
_DEFAULT_PORTS = {"http": 80, "https": 443}
HEALTH_PATHS = {"/", "/api/health", "/api/health/live", "/api/health/ready"}
LOCAL_UI_ORIGINS = {
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
}
PROTECTED_LOCAL_POST_PATHS = {
    "/api/reality-reach/acceptance/acoustic/a1/mandate",
    "/api/reality-reach/acceptance/acoustic/a1/run",
    "/api/reality-reach/acceptance/acoustic/provision",
    "/api/reality-reach/acceptance/mandate",
    "/api/reality-reach/acceptance/run",
    "/api/skill/execute",
    "/api/reboot",
    "/api/system/hot-reload",
    "/api/terminal/send",
}

CHEAT_CODE_COOKIE_NAME = "aura_owner_session"
CHEAT_CODE_COOKIE_TTL_SECS = 60 * 60 * 24 * 30

# ── Paired-device (LAN embodiment) surface ───────────────────
#
# Paired devices carry a revocable, per-device token minted by
# core/security/device_pairing.py. They are authorized ONLY for the
# conversation surface below — never the sovereign control surface
# (skill execution, reboot, hot-reload, settings, privacy, memory
# administration). Widening this allowlist is a security decision:
# keep it deliberate and reviewed.
DEVICE_SESSION_COOKIE_NAME = "aura_device_session"
DEVICE_SESSION_COOKIE_TTL_SECS = 60 * 60 * 24 * 180
# Reachable without credentials: the pairing ceremony IS the
# authentication. Both are rate-limited and code/TTL/attempt bounded.
PAIRING_PUBLIC_PATHS = {"/pair", "/api/devices/pair/complete"}
DEVICE_ALLOWED_READONLY_EXACT_PATHS = {
    "/",
    "/pair",
    "/api/sessions",
    "/api/ui/bootstrap",
}
# Read-only surfaces for devices: GET/HEAD only. A paired phone may
# watch Aura's worlds; it may not rewrite them.
DEVICE_ALLOWED_READONLY_PREFIXES = (
    "/static",
    "/ws",
    "/worlds",
    "/api/worlds",
)
_DEVICE_DENIAL_LOG_INTERVAL_S = 60.0
_DEVICE_DENIAL_LOG_MAX_KEYS = 512
_DEVICE_DENIAL_LOG_LOCK = threading.Lock()
_DEVICE_DENIAL_LOG_STATE: dict[tuple[str, str, str], dict[str, float | int]] = {}
_VERIFY_TOKEN_WARNED_MISSING = False
_VERIFY_TOKEN_WARNED_LOCAL = False

_DEVICE_LOOKUP_RECOVERABLE_ERRORS = (
    ImportError,
    AttributeError,
    LookupError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


# ── Internal-only guard ──────────────────────────────────────

def _request_host(request: Request) -> str:
    client = getattr(request, "client", None)
    return str(getattr(client, "host", "unknown") or "unknown")


def _is_trusted_local_host(host: str) -> bool:
    return str(host or "").strip().lower() in TRUSTED_LOCAL_UI_HOSTS


def _extract_request_token(request: Request) -> str | None:
    auth_header = str(request.headers.get("Authorization", "") or "")
    if auth_header.startswith("Bearer "):
        return auth_header[7:]
    x_api_token = str(request.headers.get("X-Api-Token", "") or "")
    return x_api_token or None


def allowed_local_ui_origins() -> list[str]:
    """Origins allowed to drive Aura's local UI API from a browser.

    Aura is a localhost desktop app, but browser security still matters:
    arbitrary websites must not be able to treat the user's loopback server as
    an authenticated capability surface.
    """
    return sorted(LOCAL_UI_ORIGINS)


def _header_value(request: Request, name: str) -> str:
    headers = getattr(request, "headers", None) or {}
    try:
        return str(headers.get(name) or headers.get(name.lower()) or "")
    except (AttributeError, TypeError):
        return ""


def _request_method(request: Request) -> str:
    return str(getattr(request, "method", "GET") or "GET").upper()


def _request_scheme(request: Request) -> str:
    scheme = str(getattr(getattr(request, "url", None), "scheme", "") or "").lower()
    if scheme == "ws":
        return "http"
    if scheme == "wss":
        return "https"
    return scheme if scheme in _BROWSER_ORIGIN_SCHEMES else "http"


def _header_values(request: Request, name: str) -> tuple[str, ...]:
    """Return every value for a security-sensitive header.

    ``Headers.get()`` hides duplicate Host headers.  Different proxies can
    select different duplicates, so ambiguous authorities must fail closed.
    """

    headers = getattr(request, "headers", None)
    getlist = getattr(headers, "getlist", None)
    if callable(getlist):
        try:
            values = tuple(str(value) for value in getlist(name))
        except (AttributeError, TypeError, ValueError):
            values = ()
        if values:
            return values

    scope = getattr(request, "scope", None)
    raw_headers = scope.get("headers") if isinstance(scope, dict) else None
    if isinstance(raw_headers, (list, tuple)):
        expected = name.lower().encode("ascii")
        collected: list[str] = []
        for item in raw_headers:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                continue
            raw_name, raw_value = item
            if isinstance(raw_name, str):
                raw_name = raw_name.encode("latin-1", errors="ignore")
            if raw_name.lower() != expected:
                continue
            if isinstance(raw_value, bytes):
                collected.append(raw_value.decode("latin-1", errors="strict"))
            else:
                collected.append(str(raw_value))
        if collected:
            return tuple(collected)

    value = _header_value(request, name)
    return (value,) if value else ()


def _parse_authority(value: str, *, default_port: int) -> tuple[str, int] | None:
    """Parse one RFC-style authority without DNS resolution or normalization tricks."""

    raw = str(value or "").strip()
    if not raw or any(char in raw for char in "\r\n\t /?#@\\,"):
        return None
    try:
        parsed = urlsplit(f"//{raw}")
        port = parsed.port
    # not a failure: a value that will not parse as a URL has no port or origin to
    # read, and this refuses rather than guessing.
    except (TypeError, ValueError, UnicodeError):
        return None
    if (
        parsed.scheme
        or parsed.path
        or parsed.query
        or parsed.fragment
        or parsed.username is not None
        or parsed.password is not None
    ):
        return None
    host = str(parsed.hostname or "").strip().lower()
    if not host or "%" in host:
        return None
    return host, int(port if port is not None else default_port)


def _parse_browser_origin(value: str) -> tuple[str, str, int] | None:
    raw = str(value or "").strip()
    if not raw or any(char in raw for char in "\r\n\t"):
        return None
    try:
        parsed: SplitResult = urlsplit(raw)
        port = parsed.port
    # not a failure: a value that will not parse as a URL has no port or origin to
    # read, and this refuses rather than guessing.
    except (TypeError, ValueError, UnicodeError):
        return None
    scheme = str(parsed.scheme or "").lower()
    host = str(parsed.hostname or "").strip().lower()
    if (
        scheme not in _BROWSER_ORIGIN_SCHEMES
        or not host
        or "%" in host
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        return None
    return scheme, host, int(port if port is not None else _DEFAULT_PORTS[scheme])


def _request_authority(
    request: Request,
    *,
    require_trusted_loopback: bool,
) -> tuple[str, int] | None:
    host_values = _header_values(request, "Host")
    if len(host_values) != 1:
        return None
    scheme = _request_scheme(request)
    authority = _parse_authority(host_values[0], default_port=_DEFAULT_PORTS[scheme])
    if authority is None:
        return None
    if require_trusted_loopback and authority[0] not in TRUSTED_LOCAL_UI_HOSTS:
        return None
    return authority


def _request_targets_trusted_loopback_authority(request: Request) -> bool:
    """Whether Host names an exact built-in loopback UI authority.

    This intentionally does not resolve hostnames.  DNS aliases, dotted
    lookalikes, integer IPv4 forms, and trailing-dot variants are not trusted.
    """

    return _request_authority(request, require_trusted_loopback=True) is not None


def _is_allowed_local_ui_origin(value: str) -> bool:
    value = str(value or "").strip().rstrip("/")
    return bool(value and value in LOCAL_UI_ORIGINS)


def _origin_matches_request_authority(
    request: Request,
    origin: str,
    *,
    require_trusted_loopback: bool,
) -> bool:
    parsed_origin = _parse_browser_origin(origin)
    if parsed_origin is None:
        return False
    scheme, origin_host, origin_port = parsed_origin
    if scheme != _request_scheme(request):
        return False
    authority = _request_authority(
        request,
        require_trusted_loopback=require_trusted_loopback,
    )
    if authority is None:
        return False
    if require_trusted_loopback and origin_host not in TRUSTED_LOCAL_UI_HOSTS:
        return False
    return (origin_host, origin_port) == authority


def _origin_matches_request_host(request: Request, origin: str) -> bool:
    """True only for an exact, literal loopback Origin/Host authority pair."""

    return _origin_matches_request_authority(
        request,
        origin,
        require_trusted_loopback=True,
    )


def _is_cross_site_browser_request(
    request: Request,
    *,
    allow_authenticated_request_authority: bool = False,
) -> bool:
    """Return True when browser metadata says this is not Aura's own UI.

    Cross-origin CSRF against localhost carries either an Origin header or
    Fetch Metadata such as ``Sec-Fetch-Site: cross-site``.  Treat those as
    hostile unless the request also supplies the real API token.
    """
    origin = _header_value(request, "Origin").strip()
    allowed_origin = bool(
        origin
        and (
            _is_allowed_local_ui_origin(origin)
            or _origin_matches_request_host(request, origin)
            or (
                allow_authenticated_request_authority
                and _origin_matches_request_authority(
                    request,
                    origin,
                    require_trusted_loopback=False,
                )
            )
        )
    )
    if origin and not allowed_origin:
        return True
    fetch_site = _header_value(request, "Sec-Fetch-Site").strip().lower()
    if fetch_site in {"cross-site", "same-site"} and not allowed_origin:
        return True
    return False


def _has_same_origin_browser_context(request: Request) -> bool:
    origin = _header_value(request, "Origin").strip()
    if _is_allowed_local_ui_origin(origin) or _origin_matches_request_host(request, origin):
        return True
    referer = _header_value(request, "Referer").strip()
    if any(referer.startswith(f"{allowed}/") or referer == allowed for allowed in LOCAL_UI_ORIGINS):
        return True
    parsed_referer = _parse_browser_origin(referer)
    if parsed_referer is None:
        try:
            parsed = urlsplit(referer)
            referer_origin = f"{parsed.scheme}://{parsed.netloc}"
        # not a failure: a value that will not parse as a URL has no port or origin to
        # read, and this refuses rather than guessing.
        except (TypeError, ValueError, UnicodeError):
            return False
    else:
        referer_origin = referer
    return _origin_matches_request_host(request, referer_origin)


def _has_desktop_ui_marker(request: Request) -> bool:
    surface = _header_value(request, "X-Aura-Surface").strip().lower()
    desktop_marker = _header_value(request, "X-Aura-Desktop-Request").strip().lower()
    return surface in {"desktop", "desktop-ui", "messages", "voice"} or desktop_marker in {"1", "true", "same-origin"}


def _allow_local_without_token(request: Request, *, protected_route: bool) -> bool:
    if not _is_trusted_local_host(_request_host(request)):
        return False
    if not _request_targets_trusted_loopback_authority(request):
        return False
    if _is_cross_site_browser_request(request):
        return False
    if not protected_route:
        return True
    return _has_same_origin_browser_context(request) or _has_desktop_ui_marker(request)


def request_has_allowed_local_browser_origin(request: Request) -> bool:
    """True when a browser-originated local request came from Aura's UI."""
    if not _is_trusted_local_host(_request_host(request)):
        return False
    if not _request_targets_trusted_loopback_authority(request):
        return False
    if _is_cross_site_browser_request(request):
        return False
    origin = _header_value(request, "Origin").strip()
    if origin:
        return _is_allowed_local_ui_origin(origin) or _origin_matches_request_host(
            request,
            origin,
        )
    return True


# ── Paired-device credentials ────────────────────────────────

def _path_at_or_below(path: str, prefix: str) -> bool:
    return path == prefix or path.startswith(f"{prefix}/")


def device_path_allowed(path: str, method: str = "GET") -> bool:
    """Return whether a paired device may use this exact HTTP operation."""
    normalized = str(path or "")
    normalized_method = str(method or "GET").upper()
    if normalized == "/api/chat":
        return normalized_method == "POST"
    if _path_at_or_below(normalized, "/api/chat/delivery"):
        return normalized_method in {"GET", "HEAD"}
    if normalized == "/api/devices/pair/complete":
        return normalized_method == "POST"
    if normalized_method not in {"GET", "HEAD"}:
        return False
    if normalized in DEVICE_ALLOWED_READONLY_EXACT_PATHS or normalized in HEALTH_PATHS:
        return True
    return any(
        _path_at_or_below(normalized, prefix)
        for prefix in DEVICE_ALLOWED_READONLY_PREFIXES
    )


def _extract_device_token(request: Request) -> str | None:
    header_token = _header_value(request, "X-Aura-Device-Token").strip()
    if header_token:
        return header_token
    bearer = _extract_request_token(request)
    if bearer and bearer.startswith("adt1."):
        return bearer
    cookies = getattr(request, "cookies", None)
    if cookies is not None:
        try:
            cookie_token = cookies.get(DEVICE_SESSION_COOKIE_NAME)
        except _COOKIE_READ_RECOVERABLE_ERRORS as exc:
            record_degradation("auth.device_cookie", exc)
            cookie_token = None
        if cookie_token:
            return str(cookie_token)
    cookie_header = _header_value(request, "Cookie")
    if cookie_header:
        parsed = SimpleCookie()
        try:
            parsed.load(cookie_header)
        except _COOKIE_READ_RECOVERABLE_ERRORS as exc:
            record_degradation("auth.device_cookie", exc)
            return None
        morsel = parsed.get(DEVICE_SESSION_COOKIE_NAME)
        if morsel is not None:
            return morsel.value
    return None


def device_for_request(request: Request) -> PairedDevice | None:
    """Resolve a paired device from the request, or None. Fails closed."""
    token = _extract_device_token(request)
    if not token:
        return None
    try:
        from core.security.device_pairing import get_device_registry

        return get_device_registry().verify_token(token)
    except _DEVICE_LOOKUP_RECOVERABLE_ERRORS as exc:
        record_degradation("auth.device_lookup", exc)
        return None


def paired_device_session_id(request: Request) -> str | None:
    """Return the stable private chat-session key for a paired principal."""

    device = device_for_request(request)
    if device is None:
        return None
    device_id = str(getattr(device, "device_id", "") or "").strip()
    return f"paired-device:{device_id}" if device_id else None


def local_owner_principal_id() -> str | None:
    """Resolve the configured primary operator without using mutable turn state."""
    try:
        from core.container import ServiceContainer

        identity_kernel = ServiceContainer.get("identity_kernel", default=None)
        if identity_kernel is not None and hasattr(identity_kernel, "get_current_identity"):
            current = identity_kernel.get_current_identity()
            if isinstance(current, dict):
                principal = " ".join(
                    str(current.get("primary_operator") or "").strip().split()
                ).casefold()[:160]
                if principal:
                    return principal
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation("auth.owner_principal", exc)
    try:
        from core.identity.self_contract import SelfContract

        principal = " ".join(
            str(
                SelfContract().get_relationship_constraints().get("primary_operator")
                or ""
            ).strip().split()
        ).casefold()[:160]
        return principal or None
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation("auth.owner_principal", exc)
        return None


def relational_principal_id_for_request(request: Request) -> str | None:
    """Bind relational state to this authenticated request, never prior turn state."""
    device = device_for_request(request)
    if device is not None:
        principal = " ".join(
            str(getattr(device, "principal_id", "") or "").strip().split()
        ).casefold()[:160]
        return principal or None
    profile = request_access_profile(request)
    if profile.get("surface") == "owner":
        return local_owner_principal_id()
    return None


def _handoff_scope(surface: str, identity: str) -> str:
    material = f"aura-ui-handoff-v1\0{surface}\0{identity}".encode(
        "utf-8", errors="surrogateescape"
    )
    return hashlib.sha256(material).hexdigest()


def request_access_profile(request: Request | None) -> dict[str, Any]:
    """Describe the authenticated UI surface without exposing credentials."""

    if request is None:
        return {
            "surface": "internal",
            "conversation_only": False,
            "handoff_scope": _handoff_scope("internal", "runtime"),
            "capabilities": {
                "chat": True,
                "sessions": True,
                "websocket": True,
                "world_read": True,
                "desktop_control": True,
                "performance_telemetry": True,
                "voice_stream": True,
                "interaction_signals": True,
                "tools_catalog": True,
                "learning_status": True,
                "diagnostics": True,
            },
        }
    supplied = _extract_request_token(request)
    expected = str(config.api_token or "")
    host = _request_host(request)
    paired_device = device_for_request(request)
    synthetic_internal_request = (
        not isinstance(request, Request)
        and host in {"test", "testclient", "unknown"}
        and paired_device is None
    )
    # An explicit paired credential defines the principal even when a reverse
    # proxy or local transport makes the peer address look like loopback. UI
    # capability advertising must agree with the later authorization decision.
    if paired_device is not None:
        device_identity = "\0".join(
            (
                str(getattr(paired_device, "device_id", "") or "").strip(),
                str(getattr(paired_device, "principal_id", "") or "").strip().casefold(),
            )
        )
        return {
            "surface": "paired_device",
            "conversation_only": True,
            "handoff_scope": _handoff_scope("paired_device", device_identity),
            "capabilities": {
                "chat": True,
                "sessions": True,
                "websocket": True,
                "world_read": True,
                "desktop_control": False,
                "performance_telemetry": False,
                "voice_stream": False,
                "interaction_signals": False,
                "tools_catalog": False,
                "learning_status": False,
                "diagnostics": False,
            },
        }
    trusted_local_owner = bool(
        _is_trusted_local_host(host)
        and _request_targets_trusted_loopback_authority(request)
        and not _is_cross_site_browser_request(request)
    )
    owner_authenticated = trusted_local_owner or bool(
        supplied
        and expected
        and not supplied.startswith("adt1.")
        and hmac.compare_digest(supplied, expected)
    ) or synthetic_internal_request
    if owner_authenticated:
        owner_identity = local_owner_principal_id() or "configured-owner"
        return {
            "surface": "owner",
            "conversation_only": False,
            "handoff_scope": _handoff_scope("owner", owner_identity),
            "capabilities": {
                "chat": True,
                "sessions": True,
                "websocket": True,
                "world_read": True,
                "desktop_control": True,
                "performance_telemetry": True,
                "voice_stream": True,
                "interaction_signals": True,
                "tools_catalog": True,
                "learning_status": True,
                "diagnostics": True,
            },
        }
    return {
        "surface": "unknown",
        "conversation_only": True,
        "handoff_scope": "",
        "capabilities": {},
    }


def _log_device_scope_denial(device_id: str, path: str, method: str) -> None:
    key = (str(device_id), str(method).upper(), str(path))
    now = time.monotonic()
    suppressed = 0
    should_log = False
    with _DEVICE_DENIAL_LOG_LOCK:
        state = _DEVICE_DENIAL_LOG_STATE.get(key)
        if state is None:
            should_log = True
            _DEVICE_DENIAL_LOG_STATE[key] = {"last_logged": now, "suppressed": 0}
        elif now - float(state.get("last_logged", 0.0)) >= _DEVICE_DENIAL_LOG_INTERVAL_S:
            should_log = True
            suppressed = int(state.get("suppressed", 0))
            state["last_logged"] = now
            state["suppressed"] = 0
        else:
            state["suppressed"] = int(state.get("suppressed", 0)) + 1
        if len(_DEVICE_DENIAL_LOG_STATE) > _DEVICE_DENIAL_LOG_MAX_KEYS:
            oldest = min(
                _DEVICE_DENIAL_LOG_STATE,
                key=lambda item: float(
                    _DEVICE_DENIAL_LOG_STATE[item].get("last_logged", 0.0)
                ),
            )
            if oldest != key:
                _DEVICE_DENIAL_LOG_STATE.pop(oldest, None)
    if should_log:
        suffix = f" ({suppressed} similar attempts suppressed)" if suppressed else ""
        logger.warning(
            "Paired device %s attempted out-of-scope operation %s %s%s",
            device_id,
            str(method).upper(),
            path,
            suffix,
        )


def _device_authorizes_request(
    request: Request,
    path: str,
    *,
    resolved_device: PairedDevice | None = None,
) -> bool:
    """True when a valid paired device may access this path.

    Raises 403 when the device is valid but the path is outside the
    conversation surface, so a stolen device token cannot even probe
    the control plane quietly.
    """
    device = resolved_device if resolved_device is not None else device_for_request(request)
    if device is None:
        return False
    method = _request_method(request)
    if device_path_allowed(path, method):
        return True
    _log_device_scope_denial(device.device_id, path, method)
    raise HTTPException(status_code=403, detail="Device session lacks access to this surface")


def validate_runtime_security_request(request: Request) -> None:
    """Fail closed on every request if security config drifts after startup."""
    path = str(getattr(request.url, "path", "") or "")
    host = _request_host(request)
    internal_only = bool(getattr(config.security, "internal_only_mode", False))
    expected = str(config.api_token or "")
    supplied = _extract_request_token(request)
    master_authenticated = bool(
        expected and supplied and hmac.compare_digest(supplied, expected)
    )
    paired_device = None if master_authenticated else device_for_request(request)
    public_path = path in HEALTH_PATHS or path in PAIRING_PUBLIC_PATHS

    # A loopback TCP peer does not prove that a browser loaded Aura's own UI.
    # DNS rebinding leaves the peer on 127.0.0.1 while Host remains an
    # attacker-controlled domain.  Reject that authority before public-path,
    # internal-only, CSRF, or desktop-marker exceptions are considered.
    if (
        _is_trusted_local_host(host)
        and not _request_targets_trusted_loopback_authority(request)
        and not master_authenticated
        and paired_device is None
    ):
        raise HTTPException(status_code=403, detail="Untrusted local Host authority denied")

    # Browser-originated cross-site requests to localhost are CSRF attempts.
    # A valid bearer/API token can still authorize automation clients, but
    # loopback alone is not authentication.
    if (
        _is_cross_site_browser_request(
            request,
            allow_authenticated_request_authority=bool(paired_device or public_path),
        )
        and not master_authenticated
    ):
        raise HTTPException(status_code=403, detail="Cross-origin local API request denied")

    if internal_only:
        if not _is_trusted_local_host(host):
            raise HTTPException(status_code=403, detail="External access denied")
        if not master_authenticated and not _allow_local_without_token(
            request,
            protected_route=False,
        ):
            raise HTTPException(status_code=403, detail="Untrusted local request denied")
        return

    # Health endpoints can stay unauthenticated for monitors, but the rest of
    # the API must fail closed if the token disappears at runtime.
    if public_path:
        return

    if not expected:
        logger.error("AURA_API_TOKEN not set and service is not internal-only. Blocking request to %s.", path)
        raise HTTPException(status_code=503, detail="Authentication not configured")

    if master_authenticated:
        return

    if paired_device is not None:
        if _device_authorizes_request(request, path, resolved_device=paired_device):
            return

    protected_local_post = (
        _request_method(request) not in {"GET", "HEAD", "OPTIONS"}
        and path in PROTECTED_LOCAL_POST_PATHS
    )
    if _allow_local_without_token(request, protected_route=protected_local_post):
        return

    if not supplied or not hmac.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="Unauthorized")


def _require_internal(request: Request) -> None:
    """Block non-localhost requests when AURA_INTERNAL_ONLY=1."""
    if not config.security.internal_only_mode:
        return
    host = _request_host(request)
    if not _is_trusted_local_host(host):
        raise HTTPException(status_code=403, detail="External access denied")
    expected = str(config.api_token or "")
    supplied = _extract_request_token(request)
    master_authenticated = bool(
        expected and supplied and hmac.compare_digest(supplied, expected)
    )
    if not master_authenticated and not _allow_local_without_token(
        request,
        protected_route=False,
    ):
        raise HTTPException(status_code=403, detail="Untrusted local request denied")


# ── Token verification ───────────────────────────────────────

def _verify_token(request: Request, x_api_token: str | None = Header(default=None)) -> None:
    """Bearer-token check. Ensures fail-closed unless running in strict internal_only_mode."""
    global _VERIFY_TOKEN_WARNED_LOCAL, _VERIFY_TOKEN_WARNED_MISSING
    expected = config.api_token
    internal_only = getattr(config.security, "internal_only_mode", False)
    supplied = _extract_request_token(request) or x_api_token

    if not expected:
        # Only allow missing token if we are strictly bound to localhost
        if internal_only and _allow_local_without_token(request, protected_route=False):
            if not _VERIFY_TOKEN_WARNED_MISSING:
                logger.warning("AURA_API_TOKEN not set but running in internal_only_mode.")
                _VERIFY_TOKEN_WARNED_MISSING = True
            return

        logger.error("AURA_API_TOKEN not set and service is not internal-only. Blocking.")
        raise HTTPException(status_code=503, detail="Authentication not configured")

    if supplied and hmac.compare_digest(supplied, expected):
        return

    if not internal_only and _device_authorizes_request(
        request, str(getattr(request.url, "path", "") or "")
    ):
        return

    if internal_only and _allow_local_without_token(request, protected_route=False):
        return

    if _allow_local_without_token(request, protected_route=True):
        if not _VERIFY_TOKEN_WARNED_LOCAL:
            logger.info("Trusted same-origin Aura UI request accepted without exposing API token.")
            _VERIFY_TOKEN_WARNED_LOCAL = True
        return

    if not supplied or not hmac.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="Unauthorized")


# ── Cookie management ────────────────────────────────────────

_CHEAT_CODE_COOKIE_SECRET: bytes | None = None
_COOKIE_SECRET_RECOVERABLE_ERRORS = (
    ImportError,
    AttributeError,
    LookupError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)
_COOKIE_READ_RECOVERABLE_ERRORS = (AttributeError, CookieError, KeyError, TypeError, ValueError)
_OWNER_SESSION_RECOVERABLE_ERRORS = (
    ImportError,
    AttributeError,
    LookupError,
    RuntimeError,
    TypeError,
    ValueError,
)
_CHEAT_CODE_RECOVERABLE_ERRORS = (
    ImportError,
    AttributeError,
    LookupError,
    RuntimeError,
    TypeError,
    ValueError,
)


def _get_cheat_code_cookie_secret() -> bytes:
    global _CHEAT_CODE_COOKIE_SECRET
    if _CHEAT_CODE_COOKIE_SECRET is None:
        secret_value: str | None = None
        try:
            from core.security.zenith_secrets import get_secret

            secret_value = get_secret("AURA_CHEAT_CODE_COOKIE_SECRET")
        except _COOKIE_SECRET_RECOVERABLE_ERRORS as exc:
            record_degradation('auth', exc)
            secret_value = None
        secret_value = secret_value or config.api_token or secrets.token_urlsafe(32)
        _CHEAT_CODE_COOKIE_SECRET = secret_value.encode("utf-8")
    return _CHEAT_CODE_COOKIE_SECRET


def _encode_owner_session_cookie() -> str:
    payload = {
        "scope": "sovereign_owner",
        "issued_at": int(time.time()),
        "exp": int(time.time()) + CHEAT_CODE_COOKIE_TTL_SECS,
    }
    encoded = base64.urlsafe_b64encode(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).decode("ascii").rstrip("=")
    signature = hmac.new(
        _get_cheat_code_cookie_secret(),
        encoded.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return f"{encoded}.{signature}"


def _decode_owner_session_cookie(token: str | None) -> dict[str, Any] | None:
    if not token or "." not in token:
        return None
    encoded, signature = token.rsplit(".", 1)
    expected = hmac.new(
        _get_cheat_code_cookie_secret(),
        encoded.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(signature, expected):
        return None
    padded = encoded + "=" * (-len(encoded) % 4)
    try:
        payload = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8"))
    # not a failure: text that is not the JSON this expects is not a record it can
    # read back.
    except (ValueError, UnicodeDecodeError):
        return None
    if payload.get("scope") != "sovereign_owner":
        return None
    if int(payload.get("exp", 0) or 0) < int(time.time()):
        return None
    return cast(dict[str, Any], payload)


def _restore_owner_session_from_request(request: Request | None) -> bool:
    if request is None:
        return False
    token = None
    cookies = getattr(request, "cookies", None)
    if cookies is not None:
        try:
            token = cookies.get(CHEAT_CODE_COOKIE_NAME)
        except _COOKIE_READ_RECOVERABLE_ERRORS as exc:
            record_degradation('auth', exc)
            token = None
    if not token:
        headers = getattr(request, "headers", None) or {}
        cookie_header = headers.get("cookie") or headers.get("Cookie")
        if cookie_header:
            parsed = SimpleCookie()
            try:
                parsed.load(cookie_header)
            except _COOKIE_READ_RECOVERABLE_ERRORS as exc:
                record_degradation('auth', exc)
                parsed = SimpleCookie()
            morsel = parsed.get(CHEAT_CODE_COOKIE_NAME)
            if morsel is not None:
                token = morsel.value
    payload = _decode_owner_session_cookie(token)
    if not payload:
        return False
    try:
        from core.security.trust_engine import get_trust_engine
        from core.security.user_recognizer import get_user_recognizer

        get_user_recognizer().override_session_owner(reason="owner_session_cookie")
        get_trust_engine().establish_sovereign_session(
            reason="owner_session_cookie",
            announce=False,
        )
        return True
    except _OWNER_SESSION_RECOVERABLE_ERRORS as exc:
        record_degradation('auth', exc)
        logger.debug("Owner session cookie restore failed: %s", exc)
        return False


def _activate_cheat_code_for_request(code: str | None, *, silent: bool, source: str) -> dict[str, Any] | None:
    if not code:
        return None
    try:
        from core.security.cheat_codes import activate_cheat_code

        result = activate_cheat_code(code, silent=silent, source=source)
        return cast(dict[str, Any], result)
    except _CHEAT_CODE_RECOVERABLE_ERRORS as exc:
        record_degradation('auth', exc)
        logger.debug("Cheat code activation failed: %s", exc)
        return {
            "ok": False,
            "status": "error",
            "message": "Cheat code activation failed.",
        }


# ── Rate Limiter ──────────────────────────────────────────────

class _RateLimiter:
    """Token-bucket rate limiter per client IP with automatic cleanup."""
    def __init__(self, max_requests: int = 30, window_seconds: float = 60.0):
        self._max = max_requests
        self._window = window_seconds
        self._clients: dict[str, list[float]] = {}
        self._lock = threading.Lock()
        self._last_cleanup = time.time()

    def check(self, client_ip: str) -> bool:
        now = time.time()
        with self._lock:
            # Periodic cleanup: evict stale IPs every 5 minutes
            if now - self._last_cleanup > 300:
                stale = [ip for ip, hits in self._clients.items() if not hits or now - hits[-1] > self._window]
                for ip in stale:
                    del self._clients[ip]
                self._last_cleanup = now

            hits = self._clients.get(client_ip, [])
            hits = [t for t in hits if now - t < self._window]
            if len(hits) >= self._max:
                return False
            hits.append(now)
            self._clients[client_ip] = hits
            return True

_rate_limiter = _RateLimiter(max_requests=30, window_seconds=60.0)


def _check_rate_limit(request: Request) -> None:
    """H-02: Rate limit check with Trusted IP bypass.

    Security: ``X-Forwarded-For``/``X-Real-IP`` are attacker-controlled on a
    direct connection, so the trusted-IP bypass and the rate-limit bucket key
    are both anchored to the real socket peer. Forwarded headers are only
    honored when the direct peer is loopback — i.e. a genuine local reverse
    proxy fronting Aura — so that per-real-IP limiting still works behind a
    trusted proxy without letting a remote client spoof ``127.0.0.1`` to
    bypass the limiter or rotate a forged header to evade per-IP buckets.
    """
    peer_ip = request.client.host if request.client else "unknown"
    peer_is_local_proxy = peer_ip in TRUSTED_IPS

    if peer_is_local_proxy:
        forwarded = request.headers.get("X-Forwarded-For")
        real_ip = request.headers.get("X-Real-IP")
        if forwarded:
            # A local proxy forwarding a real external client: rate-limit by
            # that client, do not grant the local-trust bypass.
            client_ip = forwarded.split(",")[0].strip() or peer_ip
        elif real_ip:
            client_ip = real_ip.strip() or peer_ip
        else:
            # Genuine local traffic (the desktop UI / same-host probes).
            return
        if client_ip in TRUSTED_IPS:
            return
    else:
        # Direct remote connection: ignore attacker-controlled headers entirely.
        client_ip = peer_ip

    if not _rate_limiter.check(client_ip):
        try:
            from core.security.defensive_runtime import observe_rate_limit_violation

            observe_rate_limit_violation(client_ip, route=str(getattr(request.url, "path", "") or "unknown"))
        except (
            ImportError,
            AttributeError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            record_degradation("auth.rate_limit_defense", exc)
            logger.debug("Rate-limit defensive reporting skipped: %s", exc)
        raise HTTPException(status_code=429, detail="Too many requests")
