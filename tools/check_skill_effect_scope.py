#!/usr/bin/env python3
"""A skill may not reach further than the scope it declares.

Registration already refuses a skill that declares no recognised
``effect_scope``. Nothing compared the declaration with the code, so a skill
could declare ``pure_compute`` and import ``subprocess`` — and the Will, the
catalog and every policy downstream would treat it as harmless arithmetic.

Python cannot isolate an imported module from the interpreter it is imported
into. What it can do is refuse to load one whose reach exceeds its
declaration, which is the achievable form of the control and the one this
gate enforces. `core/skills/effect_reach.py` does the measuring; the same
function runs at registration, so this is not a CI-only opinion.

    python tools/check_skill_effect_scope.py
    python tools/check_skill_effect_scope.py --write-baseline
    python tools/check_skill_effect_scope.py --report
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.skills.catalog_policy import SKILL_EFFECT_SCOPES  # noqa: E402
from core.skills.effect_reach import measure_file, violation  # noqa: E402

BASELINE = ROOT / "config" / "skill_effect_scope_baseline.json"
SKILL_ROOTS = ("core/skills", "skills")


def _declared_name(tree: ast.Module) -> str:
    """The `name = "..."` a skill class declares."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for statement in node.body:
            if (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
                and statement.targets[0].id == "name"
                and isinstance(statement.value, ast.Constant)
                and isinstance(statement.value.value, str)
            ):
                return statement.value.value
    return ""


def skill_modules() -> dict[str, Path]:
    """Skill name -> the module that defines it."""
    found: dict[str, Path] = {}
    for root in SKILL_ROOTS:
        base = ROOT / root
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            if "__pycache__" in str(path) or path.name.startswith("__"):
                continue
            try:
                tree = ast.parse(path.read_text("utf-8", errors="ignore"))
            except SyntaxError:
                continue
            name = _declared_name(tree)
            if name and name in SKILL_EFFECT_SCOPES:
                found.setdefault(name, path)
    return found


def mismatches() -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for name, path in sorted(skill_modules().items()):
        declared = SKILL_EFFECT_SCOPES[name]
        reach = measure_file(path)
        problem = violation(declared, reach)
        if problem:
            out.append(
                {
                    "skill": name,
                    "module": str(path.relative_to(ROOT)),
                    "declared": declared,
                    "reaches": problem,
                    "evidence": list(reach.evidence[:4]),
                }
            )
    return out


def load_baseline() -> set[str]:
    if not BASELINE.exists():
        return set()
    payload = json.loads(BASELINE.read_text("utf-8"))
    return {str(entry) for entry in payload.get("grandfathered", [])}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-baseline", action="store_true")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args(argv)

    found = mismatches()
    keys = {f"{m['skill']}:{m['declared']}->{m['reaches']}" for m in found}

    if args.report:
        print(f"{len(skill_modules())} skills mapped to a module")
        for entry in found:
            print(
                f"   {entry['skill']}: {entry['reaches']} "
                f"({', '.join(entry['evidence'])})"
            )
        return 0

    previous = load_baseline()

    if args.write_baseline:
        grown = keys - previous
        if previous and grown:
            print("refusing to grow the baseline. These are new:", file=sys.stderr)
            for key in sorted(grown):
                print(f"   • {key}", file=sys.stderr)
            return 1
        payload = {
            "description": (
                "Skills whose module reaches further than the effect_scope they "
                "declare. This list may only SHRINK: "
                "tools/check_skill_effect_scope.py fails on any mismatch not "
                "listed here and refuses to add one. Fix by narrowing the "
                "module or by declaring the scope the code actually needs."
            ),
            "count": len(keys),
            "grandfathered": sorted(keys),
        }
        BASELINE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {BASELINE.name}: {len(keys)} grandfathered mismatch(es)")
        return 0

    new = sorted(keys - previous)
    if new:
        print(f"❌ {len(new)} skill(s) reach further than they declare:")
        for entry in found:
            key = f"{entry['skill']}:{entry['declared']}->{entry['reaches']}"
            if key in new:
                print(f"   • {entry['skill']} ({entry['module']}): {entry['reaches']}")
                print(f"     evidence: {', '.join(entry['evidence'])}")
        print(
            "\nDeclare the scope the code needs, or stop reaching for it. The "
            "declaration is what the Will and the catalog decide on."
        )
        return 1

    stale = sorted(previous - keys)
    if stale:
        print(f"❌ {len(stale)} baseline entries no longer describe the code:")
        for key in stale:
            print(f"   • {key}")
        print("\nRemove them in the commit that fixed them: --write-baseline")
        return 1

    print(
        f"✅ every skill declares a scope that covers its reach "
        f"({len(skill_modules())} mapped, {len(previous)} grandfathered)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
