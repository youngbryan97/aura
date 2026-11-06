#!/usr/bin/env python3
"""Every service name her code reads from the container, and whether anything registers it.

    python tools/census_service_reads.py [--manifest artifacts/current/SERVICE_OWNERSHIP.md] [--phases-only] [--json]

A read with no registrar returns None for the life of the process, and the code
behind it returns at its first line. Sixteen of her self-maintenance organs were
read that way every tick from April to November 2026, behind a boot audit that
reported 100%. This lists the reads, the static registrars, and, given her last
boot's service manifest, which names her running container did not hold.

A static registrar is a call that names the service as a string: ``register``,
``register_instance`` and their kin, a provider table row, or an alias. A name
built at run time is not seen, so a name listed here may still exist; the
manifest is the check against the running container.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("core", "interface", "skills", "llm", "executors", "security")
#: Initializers that nothing calls. A name registered only there does not exist in her runtime.
UNCALLED = ("core/orchestrator/initializers/cognitive_sensory.py",)

READ = re.compile(
    r"(?:ServiceContainer|container|_container|get_container\(\))\."
    r"(?:get|resolve|get_service|get_optional|peek)\(\s*['\"]([a-zA-Z0-9_]+)['\"]"
)
READ_HELPERS = re.compile(
    r"(?:optional_service|get_runtime_service|resolve_service|_get_service)\(\s*['\"]([a-zA-Z0-9_]+)['\"]"
)
REGISTER = re.compile(
    r"\b(?:register|register_instance|register_singleton|register_runtime_service|register_factory|"
    r"register_lazy|register_service|alias|register_alias|_register)\(\s*(?:name\s*=\s*)?['\"]([a-zA-Z0-9_]+)['\"]"
)
#: A register call whose name is a constant: ``ServiceNames.X`` or a module-level ``NAME = "x"``.
REGISTER_BY_CONSTANT = re.compile(
    r"\b(?:register|register_instance|register_singleton|register_runtime_service|register_factory|"
    r"register_lazy|register_service|alias|register_alias|_register)\(\s*(?:name\s*=\s*)?"
    r"((?:ServiceNames?)\.[A-Z_0-9]+|[A-Z_][A-Z_0-9]*)\s*[,)]"
)
MODULE_CONSTANT = re.compile(r"^([A-Z_][A-Z_0-9]*)\s*(?::\s*str)?\s*=\s*['\"]([a-zA-Z0-9_]+)['\"]", re.M)
#: A provider table row: ``("name", factory, ...)`` or an alias row ``("alias", "target")``.
TABLE_ROW = re.compile(r"\(\s*['\"]([a-z0-9_]+)['\"]\s*,\s*(?:_?[a-zA-Z_][a-zA-Z0-9_]*\s*,|['\"][a-z0-9_]+['\"]\s*\))")
MAPPING_ROW = re.compile(r"['\"]([a-zA-Z0-9_]+)['\"]\s*:\s*\(?\s*['\"]core\.")


def _files() -> list[Path]:
    out = [p for pkg in PACKAGES for p in (ROOT / pkg).rglob("*.py")
           if "/tests/" not in str(p) and "__pycache__" not in str(p)]
    main = ROOT / "aura_main.py"
    return out + ([main] if main.exists() else [])


def _service_names() -> dict[str, str]:
    source = (ROOT / "core" / "service_names.py").read_text()
    return dict(re.findall(r"^\s+([A-Z_][A-Z_0-9]*)\s*=\s*['\"]([a-zA-Z0-9_]+)['\"]", source, re.M))


def census(*, phases_only: bool = False) -> dict[str, dict[str, list[str]]]:
    reads: dict[str, set[str]] = collections.defaultdict(set)
    registrars: dict[str, set[str]] = collections.defaultdict(set)
    named = _service_names()
    for path in _files():
        rel = str(path.relative_to(ROOT))
        text = path.read_text(errors="ignore")
        if not phases_only or rel.startswith("core/phases/"):
            for pattern in (READ, READ_HELPERS):
                for match in pattern.finditer(text):
                    reads[match.group(1)].add(rel)
        if rel in UNCALLED:
            continue
        for pattern in (REGISTER, MAPPING_ROW):
            for match in pattern.finditer(text):
                registrars[match.group(1)].add(rel)
        constants = dict(MODULE_CONSTANT.findall(text))
        for match in REGISTER_BY_CONSTANT.finditer(text):
            ref = match.group(1)
            name = named.get(ref.split(".", 1)[1]) if "." in ref else constants.get(ref)
            if name:
                registrars[name].add(rel)
        if rel.startswith("core/providers/"):
            for match in TABLE_ROW.finditer(text):
                registrars[match.group(1)].add(rel)
    return {
        name: {"read_by": sorted(readers), "registered_by": sorted(registrars.get(name, ()))}
        for name, readers in reads.items()
    }


def manifest_names(path: Path) -> set[str]:
    return set(re.findall(r"^\| `([a-zA-Z0-9_]+)` \|", path.read_text(), re.M))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, default=None, help="a SERVICE_OWNERSHIP.md from a boot")
    parser.add_argument("--phases-only", action="store_true", help="only reads made by the kernel's phases")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    found = census(phases_only=args.phases_only)
    live = manifest_names(args.manifest) if args.manifest else None
    unregistered = {n: v for n, v in found.items() if not v["registered_by"]}
    rows = []
    for name, value in sorted(unregistered.items(), key=lambda item: (-len(item[1]["read_by"]), item[0])):
        rows.append({"name": name, "read_by": value["read_by"],
                     "in_manifest": None if live is None else name in live})
    if args.json:
        print(json.dumps({"reads": len(found), "unregistered": rows}, indent=1))
        return 0
    print(f"{len(found)} names read; {len(rows)} with no static registrar"
          + ("" if live is None else f"; {sum(1 for r in rows if not r['in_manifest'])} also absent from the manifest"))
    for row in rows:
        mark = "" if live is None else ("  (in the manifest)" if row["in_manifest"] else "")
        print(f"  {row['name']:34s} {len(row['read_by']):3d}  {', '.join(row['read_by'][:3])}{mark}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
