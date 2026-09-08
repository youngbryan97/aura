#!/usr/bin/env python3
"""Which registered organs nothing ever asks for.

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

It reports rather than fails. A gate on this number would be a gate on how the
tree happens to spell a lookup today.
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO / "artifacts" / "dead_organs.json")
    parser.add_argument("--quiet", action="store_true")
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

    report = {
        "registered": len(names),
        "never_asked_for": unused,
        "asked_for_once": [name for name, _ in thin],
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
