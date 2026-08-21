"""core/runtime/flags.py — typed, declared runtime flags (roadmap C1).

Chrome ships risky behavior behind Finch: every trial is *declared*, typed,
defaulted, kill-switchable, and enumerable — you can always ask a binary
"what is active right now, and why?". Aura's equivalent grew as ~686
scattered ``os.environ.get("AURA_*")`` reads: untyped, undiscoverable,
individually parsed, impossible to audit.

This module is the typed layer those reads migrate onto:

  * ``declare(...)`` registers a flag exactly once — name, type, default,
    description, and the module that owns it. Conflicting re-declaration
    is an error (two subsystems silently sharing a knob is how defaults
    drift apart).
  * Value resolution is read-through with explicit precedence:
    **environment variable** (operational override, always wins) →
    **runtime_settings** (persisted configuration) → **declared default**.
    Malformed values fall back to the default — a typo'd env var can
    degrade a flag, never crash a boot path.
  * ``flag_report()`` enumerates every declared flag with its current
    value AND the source it resolved from — the "what trials are active"
    surface for `make doctor`, the health API, and the incident narrator.

Migration is ratcheted, not big-banged: ``tests/test_flag_ratchet.py``
pins the raw ``os.environ.get("AURA_`` count so the sprawl only shrinks;
new reliability flags declare here from birth.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

logger = logging.getLogger("Aura.Flags")

_TRUTHY = {"1", "true", "on", "yes"}
_FALSY = {"0", "false", "off", "no"}


class FlagKind(StrEnum):
    BOOL = "bool"
    INT = "int"
    FLOAT = "float"
    STRING = "string"


@dataclass(frozen=True)
class FlagSpec:
    name: str
    kind: FlagKind
    default: Any
    description: str
    owner: str  # module path that declared it


class Flag:
    """One declared flag. Values are read-through (never cached) so env
    overrides land immediately — tests and operators rely on that."""

    def __init__(self, spec: FlagSpec) -> None:
        self.spec = spec
        #: Modules that also declare this knob. One knob may have many
        #: readers; only the contract has to agree.
        self.additional_owners: set[str] = set()

    @property
    def name(self) -> str:
        return self.spec.name

    def value(self) -> Any:
        resolved, _ = self.value_with_source()
        return resolved

    def value_with_source(self) -> tuple[Any, str]:
        raw = os.environ.get(self.spec.name)
        if raw is not None:
            coerced, ok = self._coerce(raw)
            if ok:
                return coerced, "env"
            logger.warning(
                "Flag %s: malformed env value %r; using default %r",
                self.spec.name,
                raw,
                self.spec.default,
            )
            return self.spec.default, "default(malformed_env)"

        persisted = self._persisted_value()
        if persisted is not None:
            coerced, ok = self._coerce(persisted)
            if ok:
                return coerced, "settings"

        return self.spec.default, "default"

    def _persisted_value(self) -> Any:
        try:
            from core.runtime.runtime_settings import get_runtime_setting

            return get_runtime_setting(self.spec.name, None)
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            return None

    def _coerce(self, raw: Any) -> tuple[Any, bool]:
        try:
            if self.spec.kind is FlagKind.BOOL:
                if isinstance(raw, bool):
                    return raw, True
                lowered = str(raw).strip().lower()
                if lowered in _TRUTHY:
                    return True, True
                if lowered in _FALSY:
                    return False, True
                return self.spec.default, False
            if self.spec.kind is FlagKind.INT:
                return int(str(raw).strip()), True
            if self.spec.kind is FlagKind.FLOAT:
                return float(str(raw).strip()), True
            return str(raw), True
        except (TypeError, ValueError):
            return self.spec.default, False


class _BootstrapFlag(Flag):
    """Observable flag whose value must be known before settings exist."""

    def value_with_source(self) -> tuple[Any, str]:
        from core.runtime.state_ownership import bootstrap_flag_value

        value, source = bootstrap_flag_value(self.name)
        return value, str(source)


def _bootstrap_flags() -> dict[str, Flag]:
    from core.runtime.state_ownership import bootstrap_flag_specs

    return {
        name: _BootstrapFlag(
            FlagSpec(
                name=spec.name,
                kind=FlagKind.STRING,
                default=spec.default,
                description=spec.description,
                owner=spec.owner,
            )
        )
        for name, spec in bootstrap_flag_specs().items()
    }


_REGISTRY: dict[str, Flag] = {}
_REGISTRY_LOCK = threading.Lock()


def declare(
    name: str,
    *,
    kind: FlagKind,
    default: Any,
    description: str,
    owner: str,
) -> Flag:
    """Declare a flag. Idempotent for identical specs; conflicting
    re-declaration raises — a knob must have exactly one meaning."""
    if not str(name).startswith("AURA_"):
        raise ValueError(f"flag names are namespaced: {name!r} must start with AURA_")
    spec = FlagSpec(name=name, kind=kind, default=default, description=description, owner=owner)
    bootstrap = _bootstrap_flags().get(name)
    if bootstrap is not None:
        if bootstrap.spec != spec:
            raise ValueError(
                f"flag {name} is a process-bootstrap flag owned by "
                f"{bootstrap.spec.owner}; refusing the conflicting declaration from {owner}"
            )
        return bootstrap
    with _REGISTRY_LOCK:
        existing = _REGISTRY.get(name)
        if existing is not None:
            # The CONTRACT is kind and default. Two modules disagreeing about
            # whether this is a bool defaulting to True or a string defaulting
            # to "" is a real contradiction and still fails loudly.
            #
            # Description and owner are metadata. Two modules that both read
            # AURA_OPCUA_ENDPOINT and describe it slightly differently are not
            # contradicting each other; they are both readers, which is the
            # normal case and used to be inexpressible — the second one raised
            # a ValueError that surfaced far from the actual problem.
            if (existing.spec.kind, existing.spec.default) != (spec.kind, spec.default):
                raise ValueError(
                    f"flag {name} already declared by {existing.spec.owner} with a "
                    f"different spec; refusing the conflicting declaration from {owner}"
                )
            # Accountability is kept, not dropped: "who reads this knob" is now
            # answerable, which it was not when the second reader could not
            # register at all.
            if owner and owner != existing.spec.owner:
                existing.additional_owners.add(str(owner))
            return existing
        flag = Flag(spec)
        _REGISTRY[name] = flag
        return flag


def get_flag(name: str) -> Flag | None:
    with _REGISTRY_LOCK:
        registered = _REGISTRY.get(name)
    return registered or _bootstrap_flags().get(name)


def declared_flags() -> dict[str, FlagSpec]:
    with _REGISTRY_LOCK:
        declared = {name: flag.spec for name, flag in _REGISTRY.items()}
    declared.update({name: flag.spec for name, flag in _bootstrap_flags().items()})
    return declared


def flag_report() -> list[dict[str, Any]]:
    """Every declared flag with its live value and resolution source —
    the 'what is active right now' surface."""
    with _REGISTRY_LOCK:
        flags = list(_REGISTRY.values())
    registered_names = {flag.name for flag in flags}
    flags.extend(
        flag
        for name, flag in _bootstrap_flags().items()
        if name not in registered_names
    )
    report = []
    for flag in sorted(flags, key=lambda f: f.name):
        value, source = flag.value_with_source()
        report.append(
            {
                "name": flag.spec.name,
                "kind": str(flag.spec.kind),
                "value": value,
                "source": source,
                "default": flag.spec.default,
                "owner": flag.spec.owner,
                "description": flag.spec.description,
            }
        )
    return report


def aura_log_dir_override() -> str:
    """Return the typed override that redirects logs and the forensic record.

    Declared here rather than in each watchdog: the stall watchdog, the memory
    watchdog and the crash handler each read this switch, and a lane that
    interpreted it differently from the lane reading the artifacts back is how
    forensics end up written where nobody looks for them.
    """

    return str(
        declare(
            "AURA_LOG_DIR",
            kind=FlagKind.STRING,
            default="",
            description="Override root for Aura logs and the forensic record",
            owner="core.runtime.flags",
        ).value()
        or ""
    ).strip()


def aura_root_override() -> str:
    """Return the canonical typed override for Aura's writable runtime root."""

    return str(
        declare(
            "AURA_ROOT",
            kind=FlagKind.STRING,
            default="",
            description="Override root for Aura runtime data, logs, and durable state",
            owner="core.runtime.flags",
        ).value()
        or ""
    ).strip()


def user_surface_recurrent_loops_override() -> str:
    """Return the default recurrent depth requested for user-surface decode."""

    return str(
        declare(
            "AURA_USER_SURFACE_RECURRENT_LOOPS",
            kind=FlagKind.STRING,
            default="1",
            description="Default recurrent passes requested for a live user-surface decode",
            owner="core.runtime.flags",
        ).value()
        or "1"
    ).strip()


def user_surface_recurrent_max_loops_override() -> str | None:
    """Return the explicitly authorized maximum live user-surface depth."""

    value = declare(
        "AURA_USER_SURFACE_RECURRENT_MAX_LOOPS",
        kind=FlagKind.STRING,
        default=None,
        description=(
            "Maximum recurrent passes admitted on a live user surface; depth above "
            "one remains an explicit measured opt-in"
        ),
        owner="core.runtime.flags",
    ).value()
    return None if value is None else str(value).strip()


def reset_registry_for_test() -> None:
    with _REGISTRY_LOCK:
        _REGISTRY.clear()


# ── One-line declared reads ───────────────────────────────────────────────
#
# The ratchet in tests/test_flag_ratchet.py caps raw os.environ AURA_* reads,
# and the count grew instead of shrinking because declaring a flag cost five
# lines and a module-level name for something that was one expression. At that
# price the next person writes os.getenv and moves on — which is what happened.
#
# These declare on first use and read through, so complying costs one line and
# the knob still arrives in the registry typed, defaulted, owned and described.
# Nothing is cached: declare() is idempotent for an identical spec, and Flag
# values are read-through by design so a test that sets the variable is seen.


def env_str(name: str, *, default: str = "", description: str, owner: str) -> str:
    """A declared string knob, read through. Empty string when unset."""
    bootstrap = _bootstrap_flags().get(name)
    flag = bootstrap or declare(
        name,
        kind=FlagKind.STRING,
        default=default,
        description=description,
        owner=owner,
    )
    value = flag.value()
    return str(default if value is None else value)


def env_int(name: str, *, default: int, description: str, owner: str) -> int:
    """A declared integer knob. A malformed value yields the default.

    Malformed is not fatal on purpose: an operator typo in one knob should not
    take down a boot, and the flag layer already records the resolution source
    for anyone asking why a value looks wrong.
    """
    value = declare(
        name, kind=FlagKind.INT, default=default, description=description, owner=owner
    ).value()
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def env_float(name: str, *, default: float, description: str, owner: str) -> float:
    """A declared float knob. A malformed value yields the default."""
    value = declare(
        name, kind=FlagKind.FLOAT, default=default, description=description, owner=owner
    ).value()
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def env_present(name: str, *, description: str, owner: str) -> bool:
    """Is this knob SET to something non-empty?

    The declared form of `os.environ.get(X) is not None`, and deliberately not
    a literal translation of it. Migrating that expression to a helper that
    returns "" when unset turns it into `"" is not None` — always true — which
    is exactly the mistake that made `is_test_run` unconditionally True and
    broke 23 tests in one pass.

    Empty counts as absent. `AURA_TESTING=""` meaning "testing is on" would
    surprise everyone, and call sites in this codebase already disagreed about
    it: some checked `is not None`, others checked truthiness, on the same
    three variables in the same file.
    """
    bootstrap = _bootstrap_flags().get(name)
    flag = bootstrap or declare(
        name,
        kind=FlagKind.STRING,
        default="",
        description=description,
        owner=owner,
    )
    return bool(str(flag.value() or "").strip())


def env_bool(name: str, *, default: bool = False, description: str, owner: str) -> bool:
    """A declared boolean knob.

    Truthiness is the flag layer's, not this function's — so "1", "true" and
    "on" mean the same thing everywhere, which was one of the things 647
    independently-parsed raw reads could not promise.
    """
    value = declare(
        name, kind=FlagKind.BOOL, default=default, description=description, owner=owner
    ).value()
    return bool(default if value is None else value)
