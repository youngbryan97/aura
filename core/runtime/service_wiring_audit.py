"""Static audit: does every expected service actually have a provider?

The defect this exists to catch is a declared expectation with nothing
behind it, and it has a distinctive shape — the runtime is *correct*, loudly
and forever, about something no amount of waiting will fix.

LIVE 2026-07-27. ``interface/routes/chat.py`` lists nine conversational
organs a turn is expected to engage, and reports any that stay missing for
twelve consecutive turns. ``soul`` — "identity continuity across turns and
restarts" — raised a CRITICAL fault on every conversation Bryan had. The
report was accurate. The orchestrator constructed a ``Soul`` at boot and
never published it to the service spine, so the lookup answered None for the
life of the process, and the warm-up the message implied was never going to
arrive.

The second-order damage was worse than the noise. ``get_panzer_soul()`` looks
that service up and substitutes a metadata proxy carrying ``logic = None``
when it is missing, so the personality engine ran on a shell while the real
object sat one attribute away on the orchestrator — an absence presented as
a presence.

A runtime probe cannot catch this class, because the honest answer at any
given moment is "not registered *yet*". What distinguishes a warm-up from a
wiring gap is whether a registration site exists in the source at all, which
is a static question. So this asks it statically, and any gate or diagnostic
surface can call it.

It deliberately reports rather than judges: which names have a provider,
which do not, and where each provider lives. The caller decides what an
unwired expectation means for it.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable
from core.runtime.errors import record_degradation

#: Call names that publish something into the service spine.
_REGISTRATION_CALLS = frozenset({
    "register",
    "register_instance",
    "register_runtime_service",
    "register_factory",
    "register_singleton",
})

#: Directories scanned for registration sites.
_DEFAULT_SOURCE_ROOTS = ("core", "interface")

_SKIP_DIR_PARTS = frozenset({
    "__pycache__", ".venv", "node_modules", ".git", "build", "dist",
    ".claude", "artifacts", "tests",
})


@dataclass(frozen=True)
class WiringSite:
    """One place a service name is published."""

    service: str
    path: str
    line: int
    call: str

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{self.path}:{self.line} ({self.call})"


@dataclass
class WiringReport:
    """Which expected services have a provider, and which do not."""

    sites: dict[str, list[WiringSite]] = field(default_factory=dict)
    expected: tuple[str, ...] = ()

    @property
    def wired(self) -> list[str]:
        return sorted(name for name in self.expected if self.sites.get(name))

    @property
    def unwired(self) -> list[str]:
        """Expected services with no registration site anywhere in the source.

        This is the finding. A name here cannot become available by waiting.
        """
        return sorted(name for name in self.expected if not self.sites.get(name))

    @property
    def ok(self) -> bool:
        return not self.unwired

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "aura.service_wiring_audit.v1",
            "expected": list(self.expected),
            "wired": self.wired,
            "unwired": self.unwired,
            "ok": self.ok,
            "sites": {
                name: [str(site) for site in found]
                for name, found in sorted(self.sites.items())
                if found
            },
        }

    def explain(self) -> str:
        """A sentence a human can act on, for a failing gate."""
        if self.ok:
            return f"all {len(self.expected)} expected services have a provider"
        return (
            "expected services with no registration site in the source: "
            + ", ".join(self.unwired)
            + " — these cannot become available by waiting, so a runtime "
            "'not ready yet' for them is a wiring gap, not a warm-up"
        )


@lru_cache(maxsize=1)
def _service_name_constants() -> dict[str, str]:
    """Map ``ServiceNames.ATTR`` -> the string it resolves to.

    Registrations are written both ways — a bare literal and the constant —
    and a scanner that only understands literals reports a wired service as
    missing. ``data_honesty_governor`` is registered exclusively through
    ``ServiceNames.DATA``, and looking for the literal alone finds nothing.
    """
    try:
        from core.service_names import ServiceNames
    except (ImportError, AttributeError):
        return {}
    mapping: dict[str, str] = {}
    for attribute in dir(ServiceNames):
        if attribute.startswith("_"):
            continue
        value = getattr(ServiceNames, attribute, None)
        if isinstance(value, str):
            mapping[attribute] = value
        elif value is not None and hasattr(value, "value"):
            # StrEnum members and similar carriers.
            raw = getattr(value, "value", None)
            if isinstance(raw, str):
                mapping[attribute] = raw
    return mapping


def _iter_source_files(roots: Iterable[str], repo_root: Path) -> Iterable[Path]:
    for root in roots:
        base = repo_root / root
        if not base.is_dir():
            continue
        for path in base.rglob("*.py"):
            # Match against the path RELATIVE to the repo, not the absolute
            # one. Absolute parts include every ancestor directory, so a
            # checkout living under any skipped name disqualified the entire
            # tree: in a worktree at `<repo>/.claude/worktrees/<name>/` this
            # skipped every file, found zero registration sites, and reported
            # EVERY organ as unwired. A blind audit that says "nothing is
            # wired" instead of "I cannot see" is worse than no audit — it is
            # a detector reporting total failure as a measurement.
            try:
                relative = path.relative_to(repo_root)
            except ValueError:
                continue
            if _SKIP_DIR_PARTS.intersection(relative.parts):
                continue
            yield path


def _registered_name(node: ast.Call, constants: dict[str, str]) -> str | None:
    """The service name this call publishes, if it publishes one."""
    if not node.args:
        return None
    first = node.args[0]
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        return first.value
    # ServiceNames.SOUL / ServiceNames.DATA
    if isinstance(first, ast.Attribute) and isinstance(first.value, ast.Name):
        if first.value.id.endswith("ServiceNames"):
            return constants.get(first.attr)
    return None


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def scan_registration_sites(
    expected: Iterable[str],
    *,
    roots: Iterable[str] = _DEFAULT_SOURCE_ROOTS,
    repo_root: Path | None = None,
) -> dict[str, list[WiringSite]]:
    """Find every place each expected service name is published."""
    wanted = {str(name) for name in expected if str(name)}
    root = repo_root or Path(__file__).resolve().parent.parent.parent
    constants = _service_name_constants()
    found: dict[str, list[WiringSite]] = {name: [] for name in wanted}

    for path in _iter_source_files(roots, root):
        try:
            source = path.read_text(encoding="utf-8")
        except OSError:
            continue
        # Cheap prefilter: parsing every file in core/ is the expensive part.
        if not any(name in source for name in wanted) and "ServiceNames" not in source:
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            call = _call_name(node)
            if call not in _REGISTRATION_CALLS:
                continue
            name = _registered_name(node, constants)
            if name in wanted:
                found[name].append(
                    WiringSite(
                        service=name,
                        path=str(path.relative_to(root)),
                        line=node.lineno,
                        call=call,
                    )
                )
    return found


def audit_expected_services(
    expected: Iterable[str],
    *,
    roots: Iterable[str] = _DEFAULT_SOURCE_ROOTS,
    repo_root: Path | None = None,
) -> WiringReport:
    """Audit a roster of expected service names against the source."""
    names = tuple(dict.fromkeys(str(name) for name in expected if str(name)))
    return WiringReport(
        sites=scan_registration_sites(names, roots=roots, repo_root=repo_root),
        expected=names,
    )


def expected_turn_organ_names() -> tuple[str, ...]:
    """The conversational organs a chat turn expects, read from the route.

    Read rather than duplicated: a second copy of this roster would drift
    from the one that actually raises the fault, which is the failure mode
    this module exists to catch.
    """
    routes = Path(__file__).resolve().parent.parent.parent / "interface" / "routes"
    # The chat route is several modules now, and the roster travelled with the
    # turn-contract lane. Reading one filename by name returned an empty
    # roster the moment it moved — and an empty roster reads as "every organ
    # is wired", which is the exact false all-clear this module exists to
    # prevent.
    for route in sorted(routes.glob("chat*.py")):
        try:
            source = route.read_text(encoding="utf-8")
        except OSError:
            continue
        match = re.search(
            r"_EXPECTED_TURN_ORGANS[^=]*=\s*\((.*?)\n\)", source, re.S,
        )
        if not match:
            continue
        names = tuple(re.findall(r'\(\s*"([a-z_]+)"\s*,', match.group(1)))
        if names:
            return names
    record_degradation(
        "runtime.service_wiring_audit",
        RuntimeError("_EXPECTED_TURN_ORGANS was not found in any chat lane module"),
        severity="error",
        action=(
            "reported an empty turn-organ roster, which every caller reads as "
            "nothing missing"
        ),
        enforce_failure_policy=False,
    )
    return ()


def audit_turn_organs() -> WiringReport:
    """Audit the conversational organ roster specifically."""
    return audit_expected_services(expected_turn_organ_names())


__all__ = [
    "WiringReport",
    "WiringSite",
    "audit_expected_services",
    "audit_turn_organs",
    "expected_turn_organ_names",
    "scan_registration_sites",
]
