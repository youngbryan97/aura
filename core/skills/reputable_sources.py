"""The sites the operator has listed as reputable, each with what kind of source it is (config/reputable_sources.json).

Her fact checks read only hosts on the operator's read allowlist (core/skills/reach_gateway.py). Bryan, 19 October
2026: as many reputable sources as can be listed, of these kinds: official sites, peer-reviewed studies, .edu and .gov
(checked), primary sources, trustworthy tertiary sources such as encyclopedias and fact sites, and secondary sources
counted for less. The list is his, in the repository; it is read, never added to, and nothing she reads changes it.

A host is on the list when it is an entry or under one (www.britannica.com is under britannica.com). A named source
is matched before a class of hosts, and a longer suffix before a shorter one, so plato.stanford.edu is the reference
work it is and not only a university's page.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger("Aura.Reach.Reputable")

__all__ = ["Listed", "LIST_PATH", "engines", "listed", "suffixes"]

LIST_PATH = Path(__file__).resolve().parents[2] / "config" / "reputable_sources.json"


@dataclass(frozen=True)
class Listed:
    suffix: str
    tier: str
    what: str
    checked: str = ""


@lru_cache(maxsize=4)
def _entries(path: str = str(LIST_PATH)) -> tuple[tuple[Listed, ...], tuple[str, ...]]:
    try:
        made = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("the reputable sources list could not be read (%s): %s", path, exc)
        return (), ()
    named = [Listed(str(e["host"]).lower(), str(e["tier"]), str(e.get("what", "")), str(e.get("checked", "")))
             for e in made.get("sources", [])]
    classes = [Listed(str(e["suffix"]).lower(), str(e["tier"]), str(e.get("what", "")), str(e.get("checked", "")))
               for e in made.get("classes", [])]
    ordered = sorted(named, key=lambda e: -len(e.suffix)) + sorted(classes, key=lambda e: -len(e.suffix))
    return tuple(ordered), tuple(str(h).lower() for h in made.get("engines", []))


def _under(host: str, suffix: str) -> bool:
    return host == suffix or host.endswith("." + suffix)


def listed(host: str, path: Path = LIST_PATH) -> Listed | None:
    """The entry a host is on, named sources first and the longest suffix first; None when it is on none."""
    host = (host or "").lower().rstrip(".")
    return next((e for e in _entries(str(path))[0] if _under(host, e.suffix)), None)


def suffixes(path: Path = LIST_PATH) -> frozenset[str]:
    """Every suffix on the list, and its search engines, as hosts the read allowlist admits with what is under them."""
    entries, found_by = _entries(str(path))
    return frozenset({e.suffix for e in entries} | set(found_by))


def engines(path: Path = LIST_PATH) -> tuple[str, ...]:
    return _entries(str(path))[1]
