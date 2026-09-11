#!/usr/bin/env python3
"""Who asks for a service through a seam that will never build it.

`get_runtime_service` resolves with `ServiceContainer.peek`, and peek
deliberately never invokes a factory: the seam exists for diagnostics and error
sinks, which must be able to look at a runtime without booting an organ while
they look. That is the right rule for a reader.

It is the wrong seam for a caller that wants the thing. A service registered as
a lazy factory has no instance until somebody builds it, so an ask through this
seam returns None for the life of the process — silently, with the subsystem
present, registered and complete. Every mechanism found dead by the subject-core
battery has had this shape, and this one is mechanical to find.

    .venv/bin/python tools/audit_read_only_service_asks.py
    .venv/bin/python tools/audit_read_only_service_asks.py --json out.json

The organism has to be up for the answer to mean anything: which services have
been built by now is the question.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

os.environ.setdefault("AURA_TESTING", "1")

#: `get_runtime_service("name")`, which is the only form that can be read
#: statically. A name built at runtime is not findable here and is not claimed.
ASK = re.compile(r"""get_runtime_service\(\s*["']([A-Za-z0-9_]+)["']""")


def asked_for(root: Path) -> dict[str, list[str]]:
    """Every service name asked for through the read-only seam, by call site."""
    out: dict[str, list[str]] = {}
    for path in sorted(root.rglob("*.py")):
        if "service_registry.py" in path.name:
            continue
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        for name in ASK.findall(text):
            out.setdefault(name, []).append(str(path.relative_to(REPO)))
    return out


async def blind_asks() -> list[dict[str, object]]:
    """The asks that resolve to nothing because the service is built on demand."""
    from core.subject.driver import build_runtime, start_organism

    workdir = REPO / "artifacts" / "subject_core" / "seam_audit"
    runtime = build_runtime(workdir, seed=1)
    await start_organism(runtime, quiet=True)

    from core.container import ServiceContainer

    findings: list[dict[str, object]] = []
    for name, sites in sorted(asked_for(REPO / "core").items()):
        if ServiceContainer.peek(name, default=None) is not None:
            continue
        if ServiceContainer.get(name, default=None) is None:
            # Not registered at all, or genuinely absent in this runtime. That
            # is a different question from this one.
            continue
        findings.append({"service": name, "sites": sites})
    return findings


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    findings = await blind_asks()
    total = sum(len(item["sites"]) for item in findings)  # type: ignore[arg-type]
    print(
        f"{len(findings)} service(s) are asked for through the read-only seam "
        f"and are only built on demand, across {total} call site(s):"
    )
    for item in findings:
        sites = item["sites"]  # type: ignore[index]
        print(f"  {item['service']:34} {len(sites):3d}  {sites[0]}")
        for extra in sites[1:]:
            print(f"  {'':34} {'':3}  {extra}")
    if args.json is not None:
        args.json.write_text(json.dumps(findings, indent=2, sort_keys=True) + "\n")
        print(f"written: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
