#!/usr/bin/env python3
"""Which registered organs nothing asks for, and which of their abilities nothing uses.

The subject-core battery kept finding the same defect: a mechanism written for
a job, registered under a name, and never called. The grounding engine, the
workspace's processor list, the two halves of the substrate stimulus path, the
lifetime reservoir's reading. None of them was visible from reading the code,
because each one looks finished from the inside.

This looks from the outside. Every name the container registers, against every
place in the tree that asks for a name. A service with no asker is not
necessarily dead — a few are fetched through a variable, and a few are meant to
be reached only from the routes — but the list is short enough to read, and
every entry is a question worth asking once.

There are two halves to the defect and this finds both. An organ nobody asks
for is one. The other is an organ that is asked for constantly and has an
ability nobody ever uses: `encode_text_to_stimulus` and `inject_stimulus` were
written for each other and nothing connected them; `inject_perceptual_frame` is
a dimension-by-dimension mapping of the world onto the substrate with no
sender; `AffectGroundingEngine.gather` was registered and called zero times.
None of those is a dead service. Each is a live service with a dead method.

It reports rather than fails. A gate on either number would be a gate on how
the tree happens to spell a lookup today.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

#: Ways a service name is asked for. Anything not matched here reads as unused,
#: so a new lookup idiom belongs in this list.
LOOKUPS: tuple[str, ...] = (
    "ServiceContainer.get(",
    "container.get(",
    "get_runtime_service(",
    "resolve_",
    "has_runtime_service(",
)


def registered_names() -> list[str]:
    import os

    os.environ.setdefault("AURA_TESTING", "1")
    from core.container import ServiceContainer
    from core.service_registration import register_all_services

    register_all_services()
    services = getattr(ServiceContainer, "_services", {})
    return sorted(services)


def askers(name: str) -> list[str]:
    """Every file that mentions this service name in a lookup-shaped line."""
    # Both quote styles, because the tree uses both and matching one of them
    # reports half the runtime as dead.
    pattern = f"[\"']{re.escape(name)}[\"']"
    try:
        out = subprocess.run(
            ["grep", "-rnE", "--include=*.py", pattern, "core", "interface", "skills"],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=120,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    hits: list[str] = []
    for line in out.splitlines():
        if "register" in line:
            continue
        if any(idiom in line for idiom in LOOKUPS):
            hits.append(line.split(":", 1)[0])
    return sorted(set(hits))


#: Methods every object has or that a framework calls for you. Counting these
#: as dead would bury the ones that matter.
BORING: frozenset[str] = frozenset(
    {
        "start", "stop", "close", "reset", "run", "tick", "update", "shutdown",
        "get_status", "get_snapshot", "snapshot", "status", "to_dict", "as_dict",
        "summary", "get_summary", "health", "is_alive", "is_ready", "initialize",
    }
)


def public_methods(module_path: Path) -> list[str]:
    """Every public method defined in one module, by name."""
    try:
        source = module_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    names = re.findall(r"^\s+(?:async\s+)?def\s+([a-z][a-z0-9_]*)\s*\(", source, re.M)
    return sorted({name for name in names if not name.startswith("_") and name not in BORING})


def method_callers(name: str, own: Path) -> int:
    """How many files outside the defining one call a method of this name."""
    try:
        out = subprocess.run(
            ["grep", "-rln", "--include=*.py", f"\\.{name}(", "core", "interface", "skills"],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=120,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return 0
    files = {line for line in out.splitlines() if line and Path(line) != own}
    return len(files)


def dead_abilities(limit: int = 40) -> list[dict[str, object]]:
    """Public methods on the organs, defined once and called from nowhere else.

    Scoped to the packages the cognitive cycle runs through, because a sweep of
    the whole tree returns more than anybody will read and the interesting ones
    are all here.
    """
    roots = (
        "core/consciousness",
        "core/affect",
        "core/ontogeny",
        "core/world_model",
        "core/memory",
        "core/agency",
        "core/being",
        "core/somatic",
    )
    found: list[dict[str, object]] = []
    for root in roots:
        for module in sorted((REPO / root).glob("*.py")):
            if module.name == "__init__.py":
                continue
            for name in public_methods(module):
                if method_callers(name, module.relative_to(REPO)) == 0:
                    found.append({"module": str(module.relative_to(REPO)), "method": name})
                    if len(found) >= limit * 4:
                        return found
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO / "artifacts" / "dead_organs.json")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip-methods", action="store_true")
    args = parser.parse_args()

    names = registered_names()
    unused: list[str] = []
    thin: list[tuple[str, int]] = []
    for name in names:
        found = askers(name)
        if not found:
            unused.append(name)
        elif len(found) == 1:
            thin.append((name, 1))

    abilities = [] if args.skip_methods else dead_abilities()
    report = {
        "registered": len(names),
        "never_asked_for": unused,
        "asked_for_once": [name for name, _ in thin],
        "methods_nothing_calls": abilities,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    if not args.quiet:
        print(f"{len(names)} services registered")
        print(f"{len(unused)} never asked for:")
        for name in unused:
            print(f"  {name}")
        print(f"{len(thin)} asked for in exactly one place:")
        for name, _ in thin:
            print(f"  {name}")
        print(f"{len(abilities)} public methods called from nowhere else:")
        for entry in abilities:
            print(f"  {entry['module']}::{entry['method']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
