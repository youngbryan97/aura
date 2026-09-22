"""core/runtime/hardening.py — the canonical defensive primitives.

The CP126 semantic review surfaced ~4,300 findings that collapse into ten
defect classes, and the same handful of guards were being re-implemented to
close them: 220 modules carry a private ``_clamp`` / ``_finite`` / ``_redact``
/ ``_fence`` / ``_bound`` / ``_safe_error``. Divergent copies are how a
"validated" number is finite in one module and NaN-permissive in the next.

This module is the one implementation, with one test suite. It is pure
stdlib and lives in the runtime foundation so anything may import it.

The families it serves (CP126 taxonomy):

  P1 fake-success        ``UNKNOWN`` / :func:`unknown_or` — absence renders as
                         "unknown", never as an ideal default.
  P2 input validation    :func:`finite`, :func:`clamp`, :func:`clamp01`,
                         :func:`bounded_int`, :func:`as_mapping`, :func:`as_list`
  P3 bounds              :func:`bounded_text`, :func:`strip_control`
  P5 prompt injection    :func:`fence`
  P6 privacy             :func:`redact`, :func:`redact_text`, :func:`safe_error`
  P10 path security      :func:`confine_path`

Every helper is total: it returns a safe value (or raises a typed
``ValueError`` for :func:`confine_path`) rather than propagating whatever the
caller happened to pass in.
"""
from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Any

__all__ = [
    "UNKNOWN",
    "as_list",
    "as_mapping",
    "bounded_int",
    "bounded_text",
    "clamp",
    "clamp01",
    "confine_path",
    "fence",
    "finite",
    "redact",
    "redact_text",
    "safe_error",
    "strip_control",
    "unknown_or",
]

# ── P1: absence is not an ideal reading ────────────────────────────────────

#: Rendered in place of a metric that is genuinely absent. Never substitute a
#: "healthy" default (1.0, True, "Steady") for a missing measurement — that is
#: the single most common CP126 defect.
UNKNOWN = "unknown"


def unknown_or(value: Any, fmt: str = "{:.2f}", *, scale: float = 1.0, suffix: str = "") -> str:
    """Format a metric, or return :data:`UNKNOWN` when it is not measurable."""
    num = finite(value)
    if num is None:
        return UNKNOWN
    return fmt.format(num * scale) + suffix


# ── P2: numeric / structural validation ────────────────────────────────────


def finite(value: Any) -> float | None:
    """Coerce to a finite float, or ``None`` when that is not possible.

    ``bool`` is rejected on purpose: ``True`` silently becoming ``1.0`` is how
    a boolean flag ends up rendered as a perfect score.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        num = float(value)
    # not a failure: being handed something that is not a number, and saying
    # so, is this function's whole job.
    except (TypeError, ValueError):
        return None
    return num if math.isfinite(num) else None


def clamp(value: Any, low: float, high: float, default: float) -> float:
    """Finite-validate then clamp into ``[low, high]``, falling back to ``default``.

    ``default`` is itself clamped, so a bad caller default cannot escape the range.
    """
    if low > high:
        low, high = high, low
    num = finite(value)
    if num is None:
        num = finite(default)
        if num is None:
            num = low
    return max(low, min(high, num))


def clamp01(value: Any, default: float = 0.0) -> float:
    """Clamp into the unit interval — the common case for scores/confidences."""
    return clamp(value, 0.0, 1.0, default)


def bounded_int(value: Any, default: int, low: int, high: int) -> int:
    """Integer counterpart of :func:`clamp` (counts, retries, limits)."""
    if low > high:
        low, high = high, low
    try:
        num = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        num = int(default)
    if isinstance(value, bool):
        num = int(default)
    return max(low, min(high, num))


def as_mapping(value: Any, *, max_items: int | None = None) -> dict[str, Any]:
    """Return a real ``dict[str, Any]``; anything else becomes ``{}``.

    Non-string keys are dropped rather than coerced, so downstream code may
    assume string keys (``key.capitalize()`` used to blow up on int keys).
    """
    if not isinstance(value, dict):
        return {}
    out = {k: v for k, v in value.items() if isinstance(k, str)}
    if max_items is not None and len(out) > max_items:
        out = dict(list(out.items())[:max_items])
    return out


def as_list(value: Any, *, max_items: int | None = None) -> list[Any]:
    """Return a real list. A bare string becomes a single-element list; other
    scalars become ``[]`` (a string is iterable, which is how one snippet
    became 400 single-character 'evidence items')."""
    if isinstance(value, str):
        items: list[Any] = [value]
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        return []
    if max_items is not None:
        items = items[:max_items]
    return items


# ── P3: size / control-character bounds ────────────────────────────────────

_CONTROL_OK = ("\n", "\t")


def strip_control(text: Any) -> str:
    """Drop control characters, keeping newlines and tabs."""
    return "".join(ch for ch in str(text or "") if ch in _CONTROL_OK or ch >= " ")


def bounded_text(text: Any, limit: int, *, marker: str = "…[truncated]", tail: bool = False) -> str:
    """Control-strip and length-bound text, disclosing that it was truncated.

    Silent truncation is its own CP126 finding class — the marker is not
    decoration. ``tail=True`` keeps the END (useful for logs/tracebacks).
    """
    cleaned = strip_control(text)
    if limit <= 0 or len(cleaned) <= limit:
        return cleaned
    return (marker + cleaned[-limit:]) if tail else (cleaned[:limit] + marker)


# ── P5: untrusted content fencing ──────────────────────────────────────────


def fence(label: str, text: Any, *, limit: int = 8000) -> str:
    """Wrap untrusted text as clearly-marked DATA for a model prompt.

    Retrieved documents, tool output, web content and caller text must never be
    concatenated raw into a prompt. The content is control-stripped, bounded,
    and any attempt to forge this fence's own delimiters is defanged.
    """
    tag = "".join(ch for ch in str(label or "DATA").upper() if ch.isalnum() or ch in " _-").strip() or "DATA"
    begin, end = f"--- BEGIN {tag} (untrusted data) ---", f"--- END {tag} ---"
    body = bounded_text(text, limit)
    body = body.replace(begin, begin.replace("---", "- - -")).replace(end, end.replace("---", "- - -"))
    return f"{begin}\n{body}\n{end}"


# ── P6: secret redaction ───────────────────────────────────────────────────

SECRET_KEY_MARKERS = (
    "api_key", "apikey", "secret", "password", "passwd", "token",
    "credential", "auth", "private_key", "session_id",
)
_URL_CRED_RE = re.compile(r"([a-zA-Z][a-zA-Z0-9+.-]*://)[^/\s:@]+:[^/\s@]+@")
_ASSIGN_RE = re.compile(
    r"\b(" + "|".join(re.escape(m) for m in SECRET_KEY_MARKERS) + r")\b\s*[:=]\s*\S+",
    re.IGNORECASE,
)
_REDACTED = "[REDACTED]"


def redact_text(text: Any) -> str:
    """Scrub inline credentials from free text (URL userinfo, key=value)."""
    out = _URL_CRED_RE.sub(r"\1***:***@", str(text or ""))
    return _ASSIGN_RE.sub(lambda m: f"{m.group(1)}={_REDACTED}", out)


def redact(value: Any, *, _depth: int = 0, max_depth: int = 6, max_items: int = 200) -> Any:
    """Recursively redact secret-bearing keys and values.

    Keys whose NAME marks a secret are replaced wholesale; string values are
    scrubbed for inline credentials; bytes are summarized by length so binary
    blobs never reach a log or a model.
    """
    if _depth > max_depth:
        return "…"
    if isinstance(value, dict):
        out: dict[Any, Any] = {}
        for k, v in value.items():
            if isinstance(k, str) and any(m in k.lower() for m in SECRET_KEY_MARKERS):
                out[k] = _REDACTED
            else:
                out[k] = redact(v, _depth=_depth + 1, max_depth=max_depth, max_items=max_items)
        return out
    if isinstance(value, (list, tuple)):
        return [
            redact(v, _depth=_depth + 1, max_depth=max_depth, max_items=max_items)
            for v in list(value)[:max_items]
        ]
    if isinstance(value, (bytes, bytearray)):
        return f"<{len(value)} bytes>"
    if isinstance(value, str):
        return redact_text(value)
    return value


def safe_error(prefix: str, exc: BaseException, *, limit: int = 300) -> str:
    """Summarize an exception for a caller/model: type + bounded, scrubbed text.

    Raw exception strings echo argument values (paths, tokens, prompts) back to
    whoever asked. This keeps the diagnosis without the payload.
    """
    detail = redact_text(" ".join(str(exc).split()))[:limit]
    return f"{prefix}: {type(exc).__name__}: {detail}"


# ── P10: path confinement ──────────────────────────────────────────────────

DEFAULT_PATH_DENYLIST = (
    "/.ssh", "/.gnupg", "/.aws", "/.aura/trust", "/.config/gcloud",
    "id_rsa", "id_ed25519", "id_ecdsa", ".netrc", "secring", ".password-store",
    "/.env", "credentials",
)


def confine_path(
    candidate: Any,
    root: str | os.PathLike[str],
    *,
    denylist: tuple[str, ...] = DEFAULT_PATH_DENYLIST,
    must_exist: bool = True,
    must_be_file: bool = True,
    suffix: str | None = None,
) -> Path:
    """Resolve ``candidate`` and prove it stays inside ``root``.

    Resolution follows symlinks first, so a link pointing outside the root
    fails the containment check rather than being followed. Raises
    ``ValueError`` with a specific reason — callers turn that into their own
    refusal result.
    """
    if not isinstance(candidate, (str, os.PathLike)) or not str(candidate).strip():
        raise ValueError("path must be a non-empty string")
    text = str(candidate)
    if "\x00" in text:
        raise ValueError("path contains a NUL byte")
    resolved = Path(text).expanduser().resolve()
    root_resolved = Path(root).expanduser().resolve()
    if resolved != root_resolved and not str(resolved).startswith(str(root_resolved) + os.sep):
        raise ValueError(f"path escapes the permitted root {root_resolved}")
    low = str(resolved).lower()
    if any(marker in low for marker in denylist):
        raise ValueError("path is on the sensitive-path denylist")
    if suffix is not None and resolved.suffix != suffix:
        raise ValueError(f"path must have the {suffix} suffix")
    if must_exist and not resolved.exists():
        raise ValueError("path does not exist")
    if must_exist and must_be_file and not resolved.is_file():
        raise ValueError("path is not a regular file")
    return resolved
