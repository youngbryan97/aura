"""core/version.py
───────────────
Single source of truth for Aura's version.

BEFORE: Version strings scattered as:
  - UI_VERSION = "8.1 (Patched)"      in interface/server.py
  - version="6.0.0"                   in setup.py
  - <span class="version-tag">v8.2 (Hardened)</span>  in index.html
  - version: str = "3.5.5-INDEPENDENT" in core/panzer_soul.py
  - AURA SOVEREIGN SOURCE BUNDLE - SECURITY HARDENED (v5.3)  in export scripts
  - version="1.0"  in pyproject.toml

AFTER: Everything imports from here.

Usage:
    from core.version import VERSION, version_string

    # In server.py:
    app = FastAPI(version=VERSION)

    # In index.html (template rendering):
    <span>{{ version_string() }}</span>

    # In setup.py:
    from core.version import VERSION
    setup(version=VERSION)
"""

from __future__ import annotations

import logging

# ── Canonical Version ────────────────────────────────────────

PRODUCT_NAME: str = "Aura Luna"
MAJOR: int = 2026
MINOR: int = 4
PATCH: int = 20

#: Release label — set to "" for stable, "alpha"/"beta"/"rc.1" otherwise
LABEL: str = "Zenith"

#: Full semver string: "6.0.0" or "6.0.0-beta"
VERSION: str = f"{MAJOR}.{MINOR}.{PATCH}" + (f"-{LABEL}" if LABEL else "")

#: Codename shown in UI
CODENAME: str = "Zenith"


def version_string(style: str = "full") -> str:
    """Return a human-readable version string.

    Args:
        style: one of
            "full"    → "Aura Luna v2026.4.20-Zenith"
            "short"   → "v2026.4.20-Zenith"
            "semver"  → "2026.4.20-Zenith"
            "ui"      → "v2026.4"    (major.minor only, for UI badges)

    Returns:
        Formatted version string.

    """
    if style == "full":
        return f"{PRODUCT_NAME} v{VERSION}"
    if style == "short":
        return f"v{VERSION}"
    if style == "semver":
        return VERSION
    if style == "ui":
        return f"v{MAJOR}.{MINOR}"
    raise ValueError(f"Unknown style: {style!r}. Use 'full', 'short', 'semver', or 'ui'.")


def as_tuple() -> tuple[int, int, int]:
    """Return ``(MAJOR, MINOR, PATCH)`` as integers for comparison."""
    return (MAJOR, MINOR, PATCH)


def is_at_least(major: int, minor: int = 0, patch: int = 0) -> bool:
    """Return True if this version is >= the given version."""
    return as_tuple() >= (major, minor, patch)

# ── Migration Reference ──────────────────────────────────────
# Update these strings when you find them in the codebase and
# redirect them to import from here instead.

_LEGACY_STRINGS_TO_MIGRATE = [
    # file                            old string
    ("interface/server.py",           'UI_VERSION = "8.1 (Patched)"'),
    ("setup.py",                      'version="6.0.0"'),
    ("interface/static/index.html",   'v8.2 (Hardened)'),
    ("core/panzer_soul.py",           'version: str = "3.5.5-INDEPENDENT"'),
    ("pyproject.toml",                'version = "1.0"'),
]


if __name__ == "__main__":
    # python -m core.version
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger("Version")
    logger.info("Full:   %s", version_string('full'))
    logger.info("Short:  %s", version_string('short'))
    logger.info("SemVer: %s", version_string('semver'))
    logger.info("UI:     %s", version_string('ui'))
    logger.info("Tuple:  %s", as_tuple())
    logger.info("")
    logger.info("Legacy strings to migrate:")
    for f, s in _LEGACY_STRINGS_TO_MIGRATE:
        logger.info("  %s: %s", f, s)
