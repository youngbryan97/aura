"""Redaction that goes all the way down, and stops growing.

CP126 against ``core/capability_engine.py``: "Only top-level parameter keys
containing a small secret-word set are redacted. Nested credentials,
headers, URLs, content, files, and full successful results can be persisted
without structural redaction or size limits."

The audit path did this::

    safe_params = params.copy()          # shallow
    for k in safe_params:                # top level only
        if any(s in k.lower() for s in sensitive_keys):
            safe_params[k] = "[REDACTED]"

so ``{"api_key": "sk-..."}`` was caught and
``{"request": {"headers": {"Authorization": "Bearer ..."}}}`` was written to
the database intact — and ``result`` was persisted whole, unredacted and
unbounded, for every successful call.

Six modules in this codebase carry their own redaction helper and each
knows a different subset of the problem: one matches key names, one matches
``sk-`` prefixes in text, one scrubs IPs. None of them recurses AND bounds
AND matches values. That is why the same gap keeps reappearing — fixing one
teaches the others nothing.

This is the shared one, and it does four things the partial versions do
not:

**Recurses.** Dicts inside lists inside dicts. Depth is bounded so a cyclic
or pathological structure cannot hang the audit path.

**Matches values, not just keys.** A key called ``note`` holding
``"Bearer eyJhbGciOi..."`` is a leaked credential whatever it is called.
Key-name matching alone is a naming convention, not a control.

**Bounds size.** An audit row is not a place to store a 40MB tool result.
Truncation is stated in the output — ``<truncated 39,900,000 of 40,000,000
chars>`` — because a silently shortened value in an audit log is worse than
a missing one: it reads as the whole truth.

**Reports.** :class:`RedactionReport` says how many values were redacted
and how much was elided, so a caller can record that the row is partial
rather than discovering it during an incident.
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

REDACTED = "[REDACTED]"

#: Substrings that make a key sensitive, matched case-insensitively at any
#: depth. Substring rather than exact match, because the wild has
#: ``x-api-key``, ``AWS_SECRET_ACCESS_KEY``, ``db_password_2`` and
#: ``authorization``, and an exact-match list is a list that will be
#: incomplete forever.
SENSITIVE_KEY_MARKERS: frozenset[str] = frozenset({
    "apikey", "api_key", "auth", "bearer", "cert", "cookie", "credential",
    "passphrase", "password", "passwd", "private_key", "privatekey", "pwd",
    "secret", "session_id", "signature", "ssn", "token",
})

#: Key names that contain a marker but are not secrets. Without these,
#: ``token_count`` and ``auth_method`` are redacted, the audit trail loses
#: the fields an investigator actually needs, and someone deletes the
#: redaction rather than fix it.
SENSITIVE_KEY_EXEMPTIONS: frozenset[str] = frozenset({
    "auth_method", "auth_required", "auth_type", "authenticated",
    "authority_verified", "authorized", "cert_expiry", "has_token",
    # The shell's working directory is not a credential. "pwd" is a marker so
    # that pwd_hash and pwd_salt are caught, and because exemptions match the
    # WHOLE key those still are — but PWD and OLDPWD are exported by every
    # login shell, so without these two the sandbox refused to launch on any
    # developer machine, reporting "untrusted_code may not hold secrets;
    # environment carries ... OLDPWD, PWD". A guard that fires on the current
    # directory teaches people to route around it.
    "oldpwd", "pwd",
    "requires_auth", "secret_count", "token_budget", "token_count",
    "token_limit", "tokens_used",
})

#: Values that are credentials whatever key they arrived under.
CREDENTIAL_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"sk-[A-Za-z0-9\-_]{20,}"), "[REDACTED_API_KEY]"),
    (re.compile(r"\bBearer\s+[A-Za-z0-9\-_.=]{10,}", re.IGNORECASE), "[REDACTED_BEARER]"),
    (re.compile(r"\beyJ[A-Za-z0-9\-_]{10,}\.[A-Za-z0-9\-_]{10,}\.[A-Za-z0-9\-_]+"), "[REDACTED_JWT]"),
    (re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), "[REDACTED_AWS_KEY]"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{16,}\b"), "[REDACTED_GITHUB_TOKEN]"),
    (re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}\b"), "[REDACTED_SLACK_TOKEN]"),
    (
        re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.DOTALL),
        "[REDACTED_PRIVATE_KEY]",
    ),
    # Credentials embedded in a URL: scheme://user:password@host
    (re.compile(r"\b([a-z][a-z0-9+.\-]*://)[^\s/:@]+:[^\s/@]+@"), r"\1[REDACTED_USERINFO]@"),
)

#: Direct personal identifiers.
#:
#: Added when core/constitution.py's tool telemetry was routed through here:
#: that bus carries every tool call in the system, so "credentials only" was
#: the wrong boundary — a shell argument or an email tool's payload carries
#: the person, not just their keys.
#:
#: Kept as a separate tuple because the two tiers answer to different
#: authorities. A credential leaving this machine is never right, so
#: :data:`CREDENTIAL_PATTERNS` applies everywhere unconditionally. A person's
#: email address leaving this machine is the whole point of an email tool, so
#: the personal tier is applied where the destination — not the content —
#: makes it wrong, which today means a third-party model provider. Ordering
#: is preserved by composition below: these still run AFTER the credential
#: patterns whenever both tiers apply, so an email inside a userinfo URL is
#: already gone by the time this runs.
PERSONAL_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "[EMAIL_REDACTED]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN_REDACTED]"),
    (re.compile(r"\b(?:\d[ -]?){13,19}\b"), "[CARD_REDACTED]"),
    (
        re.compile(
            r"\b(?:\+?\d{1,3}[ .-]?)?(?:\(\d{2,4}\)[ .-]?)?\d{3,4}[ .-]\d{3,4}"
            r"(?:[ .-]\d{2,4})?\b"
        ),
        "[PHONE_REDACTED]",
    ),
)

#: Every pattern, in the order the audit paths have always applied them.
_VALUE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    CREDENTIAL_PATTERNS + PERSONAL_PATTERNS
)

DEFAULT_MAX_DEPTH = 8
DEFAULT_MAX_ITEMS = 200
DEFAULT_MAX_STRING = 4_096
DEFAULT_MAX_TOTAL_CHARS = 256 * 1024


@dataclass
class RedactionReport:
    """What the redactor changed, so a caller can say the row is partial."""

    redacted_values: int = 0
    truncated_strings: int = 0
    elided_items: int = 0
    depth_exceeded: int = 0
    cycles_broken: int = 0
    budget_exhausted: bool = False
    sensitive_keys: list[str] = field(default_factory=list)

    @property
    def modified(self) -> bool:
        return bool(
            self.redacted_values
            or self.truncated_strings
            or self.elided_items
            or self.depth_exceeded
            or self.cycles_broken
            or self.budget_exhausted
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "redacted_values": self.redacted_values,
            "truncated_strings": self.truncated_strings,
            "elided_items": self.elided_items,
            "depth_exceeded": self.depth_exceeded,
            "cycles_broken": self.cycles_broken,
            "budget_exhausted": self.budget_exhausted,
            # Names only. The names are the diagnostic; the values are the
            # thing being protected.
            "sensitive_keys": sorted(set(self.sensitive_keys))[:50],
        }


def is_sensitive_key(key: Any) -> bool:
    """Does this key name mark its value as a secret?"""
    if not isinstance(key, str):
        return False
    lowered = key.strip().lower()
    if lowered in SENSITIVE_KEY_EXEMPTIONS:
        return False
    collapsed = lowered.replace("-", "_")
    return any(marker in collapsed for marker in SENSITIVE_KEY_MARKERS)


def redact_text(
    text: str,
    *,
    patterns: Sequence[tuple[re.Pattern[str], str]] | None = None,
) -> tuple[str, bool]:
    """Strip credential-shaped substrings. Returns (text, changed).

    ``patterns`` narrows the tier — :data:`CREDENTIAL_PATTERNS` alone, for a
    boundary where a person's own contact details are the payload rather than
    the leak. Default is every pattern, which is what every existing caller
    means.
    """
    if not text:
        return text, False
    redacted = text
    for pattern, replacement in patterns if patterns is not None else _VALUE_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted, redacted != text


def redact_structure(
    value: Any,
    *,
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_items: int = DEFAULT_MAX_ITEMS,
    max_string: int = DEFAULT_MAX_STRING,
    max_total_chars: int = DEFAULT_MAX_TOTAL_CHARS,
    report: RedactionReport | None = None,
) -> tuple[Any, RedactionReport]:
    """A redacted, bounded, JSON-shaped copy of ``value``.

    The input is never mutated — an audit path that redacted in place would
    hand the caller back a crippled object and break the very execution it
    was recording.
    """
    result = report if report is not None else RedactionReport()
    budget = _Budget(max_total_chars)
    redacted = _walk(
        value,
        depth=0,
        max_depth=max_depth,
        max_items=max_items,
        max_string=max_string,
        budget=budget,
        report=result,
        seen=set(),
    )
    result.budget_exhausted = budget.exhausted
    return redacted, result


def redact_mapping(
    mapping: Mapping[str, Any] | None,
    **kwargs: Any,
) -> tuple[dict[str, Any], RedactionReport]:
    """Convenience wrapper for the common "redact these params" case."""
    redacted, report = redact_structure(dict(mapping or {}), **kwargs)
    if not isinstance(redacted, dict):  # pragma: no cover - defensive
        return {}, report
    return redacted, report


class _Budget:
    __slots__ = ("remaining", "exhausted")

    def __init__(self, total: int) -> None:
        self.remaining = max(0, int(total))
        self.exhausted = False

    def take(self, amount: int) -> int:
        """Consume up to ``amount``; return what was actually available."""
        if self.remaining <= 0:
            self.exhausted = True
            return 0
        granted = min(amount, self.remaining)
        self.remaining -= granted
        if granted < amount:
            self.exhausted = True
        return granted


def _walk(
    value: Any,
    *,
    depth: int,
    max_depth: int,
    max_items: int,
    max_string: int,
    budget: _Budget,
    report: RedactionReport,
    seen: set[int],
) -> Any:
    if depth > max_depth:
        report.depth_exceeded += 1
        return f"<depth limit {max_depth} reached>"

    if isinstance(value, (str, bytes)):
        return _walk_text(value, max_string=max_string, budget=budget, report=report)

    if value is None or isinstance(value, (bool, int, float)):
        return value

    # Containers can be self-referential. An audit path is the last place
    # that should hang, so cycles are named and cut rather than followed.
    identity = id(value)
    if isinstance(value, (Mapping, list, tuple, set, frozenset)):
        if identity in seen:
            report.cycles_broken += 1
            return "<cycle>"
        seen = seen | {identity}

    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= max_items:
                remaining = len(value) - max_items
                report.elided_items += remaining
                out["<elided>"] = f"{remaining} more key(s) omitted"
                break
            raw_key = key if isinstance(key, str) else repr(key)
            if is_sensitive_key(raw_key):
                report.redacted_values += 1
                report.sensitive_keys.append(raw_key)
                _place(out, _redact_key(raw_key, report), REDACTED)
                continue
            _place(
                out,
                _redact_key(raw_key, report),
                _walk(
                    item,
                    depth=depth + 1,
                    max_depth=max_depth,
                    max_items=max_items,
                    max_string=max_string,
                    budget=budget,
                    report=report,
                    seen=seen,
                ),
            )
        return out

    if isinstance(value, (list, tuple, set, frozenset)):
        items = list(value)
        out_list: list[Any] = []
        for index, item in enumerate(items):
            if index >= max_items:
                remaining = len(items) - max_items
                report.elided_items += remaining
                out_list.append(f"<{remaining} more item(s) omitted>")
                break
            out_list.append(
                _walk(
                    item,
                    depth=depth + 1,
                    max_depth=max_depth,
                    max_items=max_items,
                    max_string=max_string,
                    budget=budget,
                    report=report,
                    seen=seen,
                )
            )
        return out_list

    # Anything else — a live object, a file handle, a model. Its repr is the
    # only safe thing to record, and it goes through text redaction too
    # because plenty of objects put credentials in their repr.
    return _walk_text(repr(value), max_string=max_string, budget=budget, report=report)


def _redact_key(key_text: str, report: RedactionReport) -> str:
    """A key is text, and text is where credentials hide.

    :func:`is_sensitive_key` answers a different question — whether the key
    *names* a secret, so its value must go. It says nothing about a key that
    *is* one. An object keyed by API key wrote the credential straight into
    the audit row while the identical string one field to the right was
    stripped, which is the opposite of what this module exists to do.

    No truncation and no budget here: a key is short, and a half-written key
    would corrupt the shape of the record rather than protect anything.
    """
    redacted, changed = redact_text(key_text)
    if changed:
        report.redacted_values += 1
    return redacted


def _place(out: dict[str, Any], key_text: str, value: Any) -> None:
    """Insert without letting a redacted key overwrite an earlier entry.

    Two distinct keys can redact — or ``repr`` — to the same text, and a plain
    assignment would drop one of the values. An audit record that silently
    holds fewer rows than the call it describes is worse than a noisy one.
    """
    if key_text in out:
        discriminator = 2
        while f"{key_text}~{discriminator}" in out:
            discriminator += 1
        key_text = f"{key_text}~{discriminator}"
    out[key_text] = value


def _walk_text(
    value: str | bytes,
    *,
    max_string: int,
    budget: _Budget,
    report: RedactionReport,
) -> str:
    if isinstance(value, bytes):
        try:
            text = value.decode("utf-8", errors="replace")
        except (UnicodeDecodeError, AttributeError):  # pragma: no cover - defensive
            text = repr(value)
    else:
        text = value

    text, changed = redact_text(text)
    if changed:
        report.redacted_values += 1

    original_length = len(text)
    allowed = min(max_string, budget.take(min(original_length, max_string)))
    if original_length > allowed:
        report.truncated_strings += 1
        # State the loss. A silently shortened value in an audit log reads
        # as the whole truth, which is the failure mode this exists to stop.
        return (
            text[:allowed]
            + f"…<truncated {original_length - allowed:,} of {original_length:,} chars>"
        )
    return text


def redaction_marker(report: RedactionReport) -> dict[str, Any] | None:
    """A field to store beside a redacted row, or None if nothing changed."""
    if not report.modified:
        return None
    return {"schema": "aura.redaction.v1", **report.to_dict()}


__all__ = [
    "CREDENTIAL_PATTERNS",
    "DEFAULT_MAX_DEPTH",
    "DEFAULT_MAX_ITEMS",
    "DEFAULT_MAX_STRING",
    "DEFAULT_MAX_TOTAL_CHARS",
    "PERSONAL_PATTERNS",
    "REDACTED",
    "SENSITIVE_KEY_EXEMPTIONS",
    "SENSITIVE_KEY_MARKERS",
    "RedactionReport",
    "is_sensitive_key",
    "redact_mapping",
    "redact_structure",
    "redact_text",
    "redaction_marker",
]
