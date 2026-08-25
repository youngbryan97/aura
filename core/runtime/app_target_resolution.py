"""Resolve ordinary app names against the applications actually on this Mac.

LaunchServices accepts strings, which makes a typo look like an execution
failure even when the intended application is unambiguous.  This boundary
keeps that correction grounded: known language aliases are canonicalized,
then the result is matched against a bounded inventory of real ``.app``
bundles.  Fuzzy correction is accepted only when it has one strong winner.
"""

from __future__ import annotations

import difflib
import os
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from core.runtime.os_automation_effects import canonical_app_target

_APP_SCAN_ROOTS = (
    Path("/Applications"),
    Path("/System/Applications"),
    Path("/System/Library/CoreServices/Applications"),
)
_APP_SCAN_MAX_DEPTH = 3
_APP_SCAN_MAX_BUNDLES = 4096
_APP_CACHE_TTL_S = 60.0
_cache_at = 0.0
_cache: tuple[InstalledApp, ...] = ()


@dataclass(frozen=True, slots=True)
class InstalledApp:
    name: str
    path: str


@dataclass(frozen=True, slots=True)
class AppTargetResolution:
    requested: str
    canonical: str
    resolved: str
    app_path: str = ""
    method: str = ""
    inventory_available: bool = False
    corrected: bool = False
    alternatives: tuple[str, ...] = ()

    @property
    def launchable(self) -> bool:
        return bool(self.resolved)

    def to_dict(self) -> dict[str, object]:
        return {
            "requested": self.requested,
            "canonical": self.canonical,
            "resolved": self.resolved,
            "app_path": self.app_path,
            "method": self.method,
            "inventory_available": self.inventory_available,
            "corrected": self.corrected,
            "alternatives": list(self.alternatives),
        }


def _normal_name(value: str) -> str:
    raw = str(value or "").casefold().strip()
    if raw.endswith(".app"):
        raw = raw[:-4]
    return "".join(character for character in raw if character.isalnum())


def _launchservices_app_path(name: str) -> str | None:
    """Return LaunchServices' bundle path without launching the application."""
    try:
        from AppKit import NSWorkspace

        path = NSWorkspace.sharedWorkspace().fullPathForApplication_(name)
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return str(path).strip() if path else ""


def _scan_root(root: Path) -> list[InstalledApp]:
    if not root.is_dir():
        return []
    found: list[InstalledApp] = []
    root_depth = len(root.parts)
    try:
        walker = os.walk(root, followlinks=False)
        for directory, child_dirs, _files in walker:
            depth = len(Path(directory).parts) - root_depth
            if depth >= _APP_SCAN_MAX_DEPTH:
                child_dirs[:] = []
            bundles = [name for name in child_dirs if name.casefold().endswith(".app")]
            for bundle in bundles:
                found.append(
                    InstalledApp(
                        name=bundle[:-4],
                        path=str(Path(directory) / bundle),
                    )
                )
                if len(found) >= _APP_SCAN_MAX_BUNDLES:
                    return found
            # Never descend into an app bundle. Besides being unnecessary, it
            # would expose nested helpers as applications the user can choose.
            child_dirs[:] = [
                name for name in child_dirs if not name.casefold().endswith(".app")
            ]
    except OSError:
        return found
    return found


def installed_app_inventory(*, refresh: bool = False) -> tuple[InstalledApp, ...]:
    """Return a bounded, de-duplicated inventory of launchable app bundles."""
    global _cache, _cache_at
    now = time.monotonic()
    if not refresh and _cache and now - _cache_at <= _APP_CACHE_TTL_S:
        return _cache

    roots = list(_APP_SCAN_ROOTS)
    home_apps = Path.home() / "Applications"
    if home_apps not in roots:
        roots.append(home_apps)
    by_path: dict[str, InstalledApp] = {}
    for root in roots:
        for app in _scan_root(root):
            by_path.setdefault(app.path, app)
            if len(by_path) >= _APP_SCAN_MAX_BUNDLES:
                break
        if len(by_path) >= _APP_SCAN_MAX_BUNDLES:
            break
    _cache = tuple(sorted(by_path.values(), key=lambda item: (item.name.casefold(), item.path)))
    _cache_at = now
    return _cache


def _resolve_from_inventory(
    requested: str,
    canonical: str,
    apps: tuple[InstalledApp, ...],
    launchservices_lookup: Callable[[str], str | None],
) -> AppTargetResolution:
    canonical_key = _normal_name(canonical)
    requested_key = _normal_name(requested)
    by_key: dict[str, list[InstalledApp]] = {}
    for app in apps:
        by_key.setdefault(_normal_name(app.name), []).append(app)

    exact = by_key.get(canonical_key, ())
    if len(exact) == 1:
        app = exact[0]
        return AppTargetResolution(
            requested=requested,
            canonical=canonical,
            resolved=app.name,
            app_path=app.path,
            method="installed_exact",
            inventory_available=True,
            corrected=_normal_name(requested) != _normal_name(app.name),
        )

    # Singular/plural drift is common in ordinary speech ("Note app") and is
    # much stronger evidence than unconstrained spelling similarity.
    grammatical_keys = {
        canonical_key.removesuffix("s"),
        canonical_key + "s",
        requested_key.removesuffix("s"),
        requested_key + "s",
    }
    grammatical = [
        app
        for key in grammatical_keys
        for app in by_key.get(key, ())
        if key
    ]
    unique_grammatical = {app.path: app for app in grammatical}
    if len(unique_grammatical) == 1:
        app = next(iter(unique_grammatical.values()))
        return AppTargetResolution(
            requested=requested,
            canonical=canonical,
            resolved=app.name,
            app_path=app.path,
            method="installed_inflection",
            inventory_available=True,
            corrected=True,
        )

    ranked = list(
        (
            difflib.SequenceMatcher(None, canonical_key, key).ratio(),
            app,
        )
        for key, candidates in by_key.items()
        for app in candidates
        if canonical_key and key
    )
    ranked.sort(
        key=lambda item: (item[0], item[1].name.casefold(), item[1].path),
        reverse=True,
    )
    best_score = ranked[0][0] if ranked else 0.0
    second_score = ranked[1][0] if len(ranked) > 1 else 0.0
    if ranked and best_score >= 0.92 and best_score - second_score >= 0.06:
        app = ranked[0][1]
        return AppTargetResolution(
            requested=requested,
            canonical=canonical,
            resolved=app.name,
            app_path=app.path,
            method="installed_unique_fuzzy",
            inventory_available=True,
            corrected=True,
            alternatives=tuple(item.name for _score, item in ranked[1:4]),
        )

    alternatives = tuple(item.name for _score, item in ranked[:3])
    launchservices_path = launchservices_lookup(canonical)
    if launchservices_path:
        resolved_name = Path(launchservices_path).stem or canonical
        return AppTargetResolution(
            requested=requested,
            canonical=canonical,
            resolved=resolved_name,
            app_path=launchservices_path,
            method="launchservices_exact",
            inventory_available=bool(apps),
            corrected=_normal_name(requested) != _normal_name(resolved_name),
            alternatives=alternatives,
        )
    return AppTargetResolution(
        requested=requested,
        canonical=canonical,
        resolved="",
        method=(
            "application_not_found"
            if launchservices_path == ""
            else "launchservices_unavailable"
        ),
        inventory_available=bool(apps),
        corrected=_normal_name(requested) != canonical_key,
        alternatives=alternatives,
    )


def resolve_installed_app_target(
    value: str,
    *,
    installed_apps: Iterable[InstalledApp] | None = None,
    refresh: bool = False,
    launchservices_lookup: Callable[[str], str | None] | None = None,
) -> AppTargetResolution:
    requested = str(value or "").strip()
    canonical = canonical_app_target(requested)
    if not canonical:
        return AppTargetResolution(
            requested=requested,
            canonical="",
            resolved="",
            method="missing_target",
        )
    apps = (
        tuple(installed_apps)
        if installed_apps is not None
        else installed_app_inventory(refresh=refresh)
    )
    lookup = launchservices_lookup or _launchservices_app_path
    return _resolve_from_inventory(requested, canonical, apps, lookup)


__all__ = [
    "AppTargetResolution",
    "InstalledApp",
    "installed_app_inventory",
    "resolve_installed_app_target",
]
