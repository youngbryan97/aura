"""What a package promises, and the test that would catch it breaking.

The organ audit counts a package as having answered "what do you promise" when
its files contain one of a handful of markers — ``THE_PROMISES``,
``Protocol``, ``result_schema``. That was the right first measurement and it
has an obvious failure mode: the answer is a string, so the number moves by
adding strings. An external review put the same point more generally — the
question that matters is not whether an abstraction is implemented correctly
but whether it governs the organism.

A promise here is three things and cannot be written without all of them:

* **it** — what the package guarantees, in one sentence a person could
  disagree with. "Handles errors gracefully" is not one.
* **checked_by** — the test that fails when it stops being true, as a node id
  something can actually run.
* **if_it_fails** — where the failure goes. A promise whose breach is silent
  is a promise nobody finds out about.

:func:`promises_whose_test_is_missing` is the gate. A promise naming a test
that does not exist is worse than no promise, because it reads as coverage.
"""
from __future__ import annotations

import functools
import importlib
import logging
import pkgutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("Aura.APromiseWithATest")

__all__ = [
    "APromise",
    "the_declared_promises",
    "promises_whose_test_is_missing",
    "packages_that_declare_promises",
    "how_the_promises_stand",
]

ROOT = Path(__file__).resolve().parents[2]

#: The module a package puts its promises in.
THE_MODULE = "_promises"


@dataclass(frozen=True, slots=True)
class APromise:
    """One thing a package guarantees, and what catches it breaking."""

    it: str
    checked_by: str
    if_it_fails: str

    def __post_init__(self) -> None:
        if len(self.it.split()) < 4:
            raise ValueError(
                f"{self.it!r} is too short to be disagreed with; a promise is a "
                "sentence, not a label"
            )
        if "::" not in self.checked_by and not self.checked_by.endswith(".py"):
            raise ValueError(
                f"{self.it!r} names {self.checked_by!r}, which is not a test node "
                "id — something has to be able to run it"
            )
        if not self.if_it_fails.strip():
            raise ValueError(
                f"{self.it!r} does not say where its breach goes; a promise whose "
                "breach is silent is one nobody finds out about"
            )

    @property
    def test_file(self) -> str:
        return self.checked_by.split("::", 1)[0]

    @property
    def test_name(self) -> str:
        return self.checked_by.split("::", 1)[1] if "::" in self.checked_by else ""

    def to_dict(self) -> dict[str, str]:
        return {
            "it": self.it,
            "checked_by": self.checked_by,
            "if_it_fails": self.if_it_fails,
        }


@functools.lru_cache(maxsize=2)
def the_declared_promises(root: str = "") -> dict[str, tuple[APromise, ...]]:
    """Every package under ``core`` that declares promises, and what they are."""
    where = Path(root or ROOT) / "core"
    found: dict[str, tuple[APromise, ...]] = {}
    for entry in sorted(where.iterdir()):
        if not entry.is_dir() or entry.name.startswith("__"):
            continue
        if not (entry / f"{THE_MODULE}.py").exists():
            continue
        try:
            module = importlib.import_module(f"core.{entry.name}.{THE_MODULE}")
        except Exception as exc:  # noqa: BLE001 - a broken declaration is a finding
            logger.warning("core.%s declares promises that will not import: %s",
                           entry.name, exc)
            continue
        declared = getattr(module, "THE_PROMISES", ())
        rows = tuple(one for one in declared if isinstance(one, APromise))
        if rows:
            found[entry.name] = rows
    return found


def packages_that_declare_promises(root: str = "") -> tuple[str, ...]:
    return tuple(sorted(the_declared_promises(root)))


def promises_whose_test_is_missing(root: str = "") -> tuple[str, ...]:
    """Promises naming a test that does not exist.

    Checked against the file on disk and against the test names inside it, so
    a renamed test shows up here rather than as coverage that stopped running.
    """
    base = Path(root or ROOT)
    missing: list[str] = []
    for package, promises in the_declared_promises(root).items():
        for promise in promises:
            path = base / promise.test_file
            if not path.exists():
                missing.append(f"core.{package}: {promise.checked_by} — no such file")
                continue
            if not promise.test_name:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            wanted = promise.test_name.split("::")[-1]
            if f"def {wanted}(" not in text:
                missing.append(
                    f"core.{package}: {promise.checked_by} — the file has no {wanted}"
                )
    return tuple(sorted(missing))


def how_the_promises_stand(root: str = "") -> dict[str, Any]:
    """For the health report: who promises what, and whether it is checked."""
    declared = the_declared_promises(root)
    missing = promises_whose_test_is_missing(root)
    # Namespace packages included. Requiring __init__.py left 42 directories
    # out of the organ audit, and core.state — which declares promises here —
    # was one of them.
    packages = [
        one.name
        for one in sorted((Path(root or ROOT) / "core").iterdir())
        if one.is_dir() and not one.name.startswith("__") and any(one.glob("*.py"))
    ]
    return {
        "packages": len(packages),
        "declaring": len(declared),
        "promises": sum(len(rows) for rows in declared.values()),
        "with_a_missing_test": list(missing),
        "not_declaring": [one for one in packages if one not in declared],
        "each": {
            name: [one.to_dict() for one in rows]
            for name, rows in sorted(declared.items())
        },
    }
