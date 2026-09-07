"""Every release requirement, and the check that can decide it.

The registry says what must be true. Each requirement carries an ``acceptance``
paragraph, and a paragraph is not a check: nothing runs it, and nothing fails
when what it describes stops being true. A release gate built on 313 such
paragraphs measures how much was written down.

So this asks a narrower, answerable question of every requirement: is there
something in this repository that a person can RUN, whose failing would mean
this requirement is not met, and does that thing exist?

Three places a link can come from, in order of how much it is worth:

    the ledger     an evidence entry of class "test", pointing at a receipt
    the registry   a test path, a make target or a tool named in the text
    the map        config/requirement_acceptance_checks.json, hand-written
                   where nothing in the registry names one

Every link is resolved against the tree — a test file that is gone, a make
target that was renamed, a tool that moved. A link nobody can run is counted
as no link, which is the whole point: the failure mode this replaces is a
paragraph that reads like a check.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "config/requirement_registry.json"
LEDGER = ROOT / "config/requirement_evidence_ledger.json"
CURATED = ROOT / "config/requirement_acceptance_checks.json"
MAKEFILE = ROOT / "Makefile"

#: `tests/test_something.py`, with or without a `::test_name` selector.
_TEST_PATH = re.compile(r"\btests/[A-Za-z0-9_./-]+\.py(?:::[A-Za-z0-9_]+)?")
#: `make target-name`, the way the docs write a gate.
_MAKE_TARGET = re.compile(r"\bmake\s+([a-z][a-z0-9-]{2,})\b")
#: `tools/whatever.py`, a runnable audit.
_TOOL_PATH = re.compile(r"\btools/[A-Za-z0-9_./-]+\.py")

_TEXT_FIELDS = ("acceptance", "closure_requires", "notes", "status_detail", "title")


def _make_targets() -> set[str]:
    """Every target the Makefile defines, so a named one can be checked."""

    if not MAKEFILE.is_file():
        return set()
    targets = set()
    for line in MAKEFILE.read_text("utf-8", errors="replace").splitlines():
        match = re.match(r"^([a-z][a-z0-9-]*):", line)
        if match:
            targets.add(match.group(1))
    return targets


def _text_of(requirement: dict[str, Any]) -> str:
    parts = []
    for field in _TEXT_FIELDS:
        value = requirement.get(field)
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, list):
            parts.extend(str(item) for item in value)
    return "\n".join(parts)


def _resolvable(kind: str, name: str, targets: set[str]) -> bool:
    if kind == "test":
        return (ROOT / name.split("::", 1)[0]).is_file()
    if kind == "make":
        return name in targets
    if kind == "tool":
        return (ROOT / name).is_file()
    return False


def _named_in(text: str, targets: set[str]) -> list[dict[str, str]]:
    found: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for kind, pattern in (("test", _TEST_PATH), ("tool", _TOOL_PATH)):
        for match in pattern.finditer(text):
            name = match.group(0)
            if (kind, name) not in seen:
                seen.add((kind, name))
                found.append({"kind": kind, "name": name})
    for match in _MAKE_TARGET.finditer(text):
        name = match.group(1)
        if name in targets and ("make", name) not in seen:
            seen.add(("make", name))
            found.append({"kind": "make", "name": name})
    return found


def build(root: Path = ROOT) -> dict[str, Any]:
    registry = json.loads(REGISTRY.read_text())
    requirements = registry["requirements"]
    targets = _make_targets()

    ledger_tests: dict[str, list[str]] = {}
    if LEDGER.is_file():
        for entry in json.loads(LEDGER.read_text()).get("entries", []):
            if str(entry.get("evidence_class") or "") != "test":
                continue
            ledger_tests.setdefault(str(entry.get("requirement_id") or ""), []).append(
                str(entry.get("ref") or "")
            )

    curated: dict[str, list[dict[str, str]]] = {}
    if CURATED.is_file():
        curated = json.loads(CURATED.read_text()).get("checks", {})

    entries = []
    for requirement in requirements:
        rid = str(requirement.get("id") or "")
        checks: list[dict[str, str]] = []
        for ref in ledger_tests.get(rid, []):
            checks.append({"kind": "receipt", "name": ref, "from": "ledger"})
        for check in _named_in(_text_of(requirement), targets):
            checks.append({**check, "from": "registry"})
        for check in curated.get(rid, []):
            checks.append({**check, "from": "map"})

        runnable = []
        broken = []
        for check in checks:
            kind, name = check["kind"], check["name"]
            if kind == "receipt":
                (runnable if (root / name).is_file() else broken).append(check)
            elif _resolvable(kind, name, targets):
                runnable.append(check)
            else:
                broken.append(check)

        entries.append(
            {
                "id": rid,
                "state": str(requirement.get("state") or ""),
                "mandatory": bool(requirement.get("mandatory")),
                "checks": runnable,
                "broken_links": broken,
                "linked": bool(runnable),
            }
        )

    unlinked = sorted(e["id"] for e in entries if not e["linked"])
    broken = sorted(e["id"] for e in entries if e["broken_links"])
    return {
        "schema_version": 1,
        "summary": {
            "requirements": len(entries),
            "linked": sum(1 for e in entries if e["linked"]),
            "unlinked": unlinked,
            "with_broken_links": broken,
            "by_source": dict(
                sorted(
                    Counter(
                        check["from"] for e in entries for check in e["checks"]
                    ).items()
                )
            ),
            "by_kind": dict(
                sorted(
                    Counter(
                        check["kind"] for e in entries for check in e["checks"]
                    ).items()
                )
            ),
            "complete": not unlinked and not broken,
        },
        "entries": entries,
        "non_claims": [
            "A link says something can decide the requirement, not that it passes.",
            "A check that cannot be resolved in the tree counts as no check.",
            "Linked is not closed: closure is the gate's question, not this one.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    summary = dict(report["summary"])
    summary["unlinked"] = len(summary["unlinked"])
    summary["with_broken_links"] = len(summary["with_broken_links"])
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
