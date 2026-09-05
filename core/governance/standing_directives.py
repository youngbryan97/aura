"""Standing directives — durable, user-authored prohibitions.

Why this exists
---------------
OpenClaw deleted a user's entire email inbox. The user had said "Do not
delete any emails"; a context-compression pass evicted that sentence to
make room, and the agent, no longer holding the constraint, proceeded
(arXiv:2603.12644, "Instruction Amnesia via Context Compression").

Aura already resists that specific failure by *class* rather than by
instruction: ``additional_confirmation_required`` forces a confirmation
whenever the effect scope is destructive, in code, so no amount of
compaction can bypass it. But the narrower gap was real — a user's own
standing instruction was not a durable object anywhere in this codebase.
The Will's constraints are Aura's (welfare limits, behavioural scars);
"never touch ~/Documents" lived only as long as it stayed in a context
window or came back from a memory lookup. Same eviction surface, smaller.

This module is the durable half. A directive is written to disk, and the
authority gateway reads it from disk on every consequential action. The
model is not asked to remember anything and cannot be talked out of it,
because the model is not consulted.

Deny-only, structurally
-----------------------
There is no allow/grant field, and adding one would be a mistake worth
naming here. A store that could also say "always permit X without asking"
would be an authority-*granting* path into the most safety-critical gate
in the system: one successful prompt injection that writes a directive
becomes a permanent, audited-looking backdoor. Prohibitions can only ever
tighten the gate, so a hostile write is at worst a denial of service, and
a denial of service is recoverable. That asymmetry is the whole design.

Failure posture
---------------
- No file at all -> no directives. Normal, quiet, allows.
- File present but unreadable/corrupt -> we know the user wrote
  prohibitions and we cannot tell what they were. Refuse anything that is
  not read-only and record a degradation. Guessing "probably fine" here
  would defeat the point of writing them down.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.config import config
from core.runtime.errors import record_degradation
from core.runtime.lockdep import checked_lock

SCHEMA_VERSION = 1
SCHEMA_NAME = "standing_directives"

# A directive may guard a filesystem location or a named tool. Both are
# exact, checkable things. Deliberately no free-text/semantic matcher: a
# fuzzy rule in a security gate fails in both directions, and the one that
# matters (silently not matching) is invisible.
KIND_PATH = "path"
KIND_TOOL = "tool"
_KINDS = frozenset({KIND_PATH, KIND_TOOL})

# What a directive covers. "write" is the default because "don't touch my
# Documents" almost always means "don't change them"; "any" is available
# for someone who means reads too, but they have to say so.
SCOPE_WRITE = "write"
SCOPE_ANY = "any"
_SCOPES = frozenset({SCOPE_WRITE, SCOPE_ANY})

# Effect scopes that read but do not change anything. A write-scoped
# directive lets these through.
_READ_ONLY_EFFECT_SCOPES = frozenset({"read_only", "status"})

# Argument keys that carry a filesystem location. Checked structurally;
# every other string argument is additionally checked as text (see
# ``_matches_path``), which is what catches a path buried in a shell
# command.
_PATH_ARG_KEYS = frozenset(
    {
        "path",
        "paths",
        "destination",
        "dest",
        "directory",
        "dir",
        "file",
        "filename",
        "file_path",
        "source",
        "src",
        "target",
        "target_path",
        "output",
        "output_path",
    }
)

_MAX_SCAN_DEPTH = 6


@dataclass(frozen=True)
class StandingDirective:
    """One prohibition. Note the absence of any field that could permit."""

    directive_id: str
    kind: str
    value: str
    reason: str
    scope: str = SCOPE_WRITE
    created_at: float = 0.0
    created_by: str = "owner"

    def to_dict(self) -> dict[str, Any]:
        return {
            "directive_id": self.directive_id,
            "kind": self.kind,
            "value": self.value,
            "reason": self.reason,
            "scope": self.scope,
            "created_at": self.created_at,
            "created_by": self.created_by,
        }

    @staticmethod
    def from_dict(raw: Any) -> StandingDirective:
        if not isinstance(raw, dict):
            raise ValueError("directive entry is not an object")
        kind = str(raw.get("kind") or "").strip().lower()
        if kind not in _KINDS:
            raise ValueError(f"unknown directive kind: {kind!r}")
        value = str(raw.get("value") or "").strip()
        if not value:
            raise ValueError("directive has no value")
        scope = str(raw.get("scope") or SCOPE_WRITE).strip().lower()
        if scope not in _SCOPES:
            raise ValueError(f"unknown directive scope: {scope!r}")
        return StandingDirective(
            directive_id=str(raw.get("directive_id") or uuid.uuid4().hex),
            kind=kind,
            value=value,
            reason=str(raw.get("reason") or "").strip(),
            scope=scope,
            created_at=float(raw.get("created_at") or 0.0),
            created_by=str(raw.get("created_by") or "owner"),
        )


@dataclass(frozen=True)
class DirectiveMatch:
    """Why an action was refused, in terms the refusal can quote back."""

    directive: StandingDirective
    matched_on: str


@dataclass
class _LoadResult:
    directives: list[StandingDirective] = field(default_factory=list)
    unreadable: bool = False
    detail: str = ""
    mtime: float = 0.0
    present: bool = False


def directives_path() -> Path:
    return config.paths.data_dir / "governance" / "standing_directives.json"


def _normalize_path(raw: Any) -> str:
    """Absolute, symlink-resolved, without requiring the path to exist.

    ``resolve()`` collapses ``..`` and follows symlinks, which is what stops
    ``~/Documents/../Documents/x`` and a symlink into a guarded directory
    from walking around a directive.
    """
    text = str(raw or "").strip()
    if not text:
        return ""
    try:
        return str(Path(text).expanduser().resolve())
    except (OSError, RuntimeError, ValueError):
        return ""


def _is_within(child: str, parent: str) -> bool:
    if not child or not parent:
        return False
    if child == parent:
        return True
    return child.startswith(parent.rstrip(os.sep) + os.sep)


def _iter_strings(value: Any, depth: int = 0) -> list[tuple[str, str]]:
    """Every string in the argument tree, paired with the key that held it."""
    if depth > _MAX_SCAN_DEPTH:
        return []
    found: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, str):
                found.append((str(key), item))
            else:
                found.extend(_iter_strings(item, depth + 1))
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            if isinstance(item, str):
                found.append(("", item))
            else:
                found.extend(_iter_strings(item, depth + 1))
    elif isinstance(value, str):
        found.append(("", value))
    return found


class StandingDirectiveStore:
    """Reads directives from disk; caches only until the file changes."""

    def __init__(self) -> None:
        self._lock = checked_lock("governance.standing_directives")
        self._cache: _LoadResult | None = None
        self._cache_key: tuple[float, int] | None = None

    def _load(self) -> _LoadResult:
        path = directives_path()
        try:
            stat = path.stat()
            key = (stat.st_mtime, stat.st_size)
        except FileNotFoundError:
            # The ordinary case: the user has written no prohibitions.
            self._cache = None
            self._cache_key = None
            return _LoadResult(present=False)
        except OSError as exc:
            return _LoadResult(
                unreadable=True,
                detail=f"stat failed: {exc}",
                present=True,
            )

        if self._cache is not None and self._cache_key == key:
            return self._cache

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            entries = payload.get("directives") if isinstance(payload, dict) else payload
            if not isinstance(entries, list):
                raise ValueError("payload has no directive list")
            directives = [StandingDirective.from_dict(item) for item in entries]
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            # Do not cache a failure: a half-written file during an edit
            # should stop failing as soon as the write completes.
            record_degradation(
                "governance",
                exc,
                action="standing_directives_unreadable",
            )
            return _LoadResult(
                unreadable=True,
                detail=str(exc),
                present=True,
            )

        result = _LoadResult(
            directives=directives,
            mtime=key[0],
            present=True,
        )
        self._cache = result
        self._cache_key = key
        return result

    def load(self) -> _LoadResult:
        with self._lock:
            return self._load()

    def list_directives(self) -> list[StandingDirective]:
        return list(self.load().directives)

    def check(
        self,
        *,
        tool_name: str,
        args: dict[str, Any] | None,
        effect_scope: str = "",
    ) -> tuple[DirectiveMatch | None, _LoadResult]:
        """Return the first directive this action violates, if any."""
        loaded = self.load()
        if loaded.unreadable or not loaded.directives:
            return None, loaded

        read_only = str(effect_scope or "").strip().lower() in _READ_ONLY_EFFECT_SCOPES
        strings = _iter_strings(dict(args or {}))
        normalized_tool = str(tool_name or "").strip().lower()

        for directive in loaded.directives:
            if directive.scope == SCOPE_WRITE and read_only:
                continue
            if directive.kind == KIND_TOOL:
                if normalized_tool == directive.value.strip().lower():
                    return DirectiveMatch(directive, f"tool={normalized_tool}"), loaded
                continue
            matched_on = _matches_path(directive, strings)
            if matched_on:
                return DirectiveMatch(directive, matched_on), loaded

        return None, loaded


def _matches_path(directive: StandingDirective, strings: list[tuple[str, str]]) -> str:
    """Structural match on path-ish keys, then a literal text match.

    The text pass exists because a guarded location can arrive inside a
    shell command rather than in a ``path`` argument, and a rule that only
    inspects well-named keys would wave that through. It can over-match —
    mentioning the path in a comment would trip it — and that is the
    direction to be wrong in: the cost is one refusal that names exactly
    which directive fired, which the user can then narrow or remove.
    """
    guarded = _normalize_path(directive.value)
    if not guarded:
        return ""

    for key, text in strings:
        if key.lower() in _PATH_ARG_KEYS:
            if _is_within(_normalize_path(text), guarded):
                return f"{key}={text}"

    raw = str(directive.value or "").strip()
    needles = {guarded, raw}
    if raw.startswith("~"):
        needles.add(str(Path(raw).expanduser()))
    for key, text in strings:
        for needle in needles:
            if needle and needle in text:
                return f"{key or 'argument'} contains {needle}"
    return ""


_store: StandingDirectiveStore | None = None
_store_lock = checked_lock("governance.standing_directives.singleton")


def get_standing_directives() -> StandingDirectiveStore:
    global _store
    with _store_lock:
        if _store is None:
            _store = StandingDirectiveStore()
        return _store


def reset_standing_directives_for_test() -> None:
    global _store
    with _store_lock:
        _store = None


def add_directive(
    *,
    kind: str,
    value: str,
    reason: str,
    scope: str = SCOPE_WRITE,
    created_by: str = "owner",
) -> StandingDirective:
    """Append a prohibition. There is deliberately no counterpart that grants."""
    from core.governance_context import local_internal_governed_scope
    from core.runtime.file_write_gateway import get_file_write_gateway

    directive = StandingDirective.from_dict(
        {
            "directive_id": uuid.uuid4().hex,
            "kind": kind,
            "value": value,
            "reason": reason,
            "scope": scope,
            "created_at": time.time(),
            "created_by": created_by,
        }
    )

    store = get_standing_directives()
    existing = store.list_directives()
    payload = {
        "directives": [item.to_dict() for item in existing] + [directive.to_dict()],
    }

    path = directives_path()
    with local_internal_governed_scope(
        "standing_directives.add",
        receipt_prefix="standing_directive",
    ):
        gateway = get_file_write_gateway()
        gateway.ensure_directory(path.parent, source="standing_directives")
        gateway.write_json(
            path,
            payload,
            schema_version=SCHEMA_VERSION,
            schema_name=SCHEMA_NAME,
            source="standing_directives.add",
        )
    return directive


def remove_directive(directive_id: str) -> bool:
    """Drop a prohibition by id. Returns whether anything was removed."""
    from core.governance_context import local_internal_governed_scope
    from core.runtime.file_write_gateway import get_file_write_gateway

    store = get_standing_directives()
    existing = store.list_directives()
    remaining = [item for item in existing if item.directive_id != directive_id]
    if len(remaining) == len(existing):
        return False

    path = directives_path()
    with local_internal_governed_scope(
        "standing_directives.remove",
        receipt_prefix="standing_directive",
    ):
        gateway = get_file_write_gateway()
        gateway.write_json(
            path,
            {"directives": [item.to_dict() for item in remaining]},
            schema_version=SCHEMA_VERSION,
            schema_name=SCHEMA_NAME,
            source="standing_directives.remove",
        )
    return True
