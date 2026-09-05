#!/usr/bin/env python3
"""Modules nothing reaches, counted and held to a falling ceiling.

292 modules under ``core/`` — 27,896 lines — are imported by nothing in the
repository. That is not a style complaint. When a large fraction of the tree is
unreachable, "is X wired?" stops having an answer you can trust, and this
codebase has already been bitten by exactly that: a second affect engine with
no construction path, a fallback that could never be reached, a vision flag with
three different defaults in four files. Dead weight is where half-wired things
hide.

Deleting 292 modules in one pass is the wrong move and this tool does not
propose it. Some are entry points invoked by name, some are loaded dynamically
through ``importlib`` or a service registry, and some are staged work. What can
be established mechanically is which ones nothing references *at all* — by
import or by name — and that the number only falls.

Reachability here means either:

* a static import from anywhere in the repo outside ``archive/``, or
* the dotted module path appearing in a string literal — how ``importlib``,
  plugin registries and config-driven factories reach a module. A module
  referenced only that way is reachable, and calling it dead would be wrong.
* membership of a package that walks its own directory and imports what it
  finds. The path is built at run time from a filename, so it appears in no
  import statement and no string literal. An external review's static pass
  concluded the forty-three interiority faculties were dead on exactly this
  ground, and they run on every boot.

Run: ``python tools/lint_module_reachability.py`` / ``--write-baseline`` /
``--list``
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "config" / "module_reachability_baseline.json"

#: Directories whose imports do not count as reachability — they are copies.
_EXCLUDED_PREFIXES = ("archive/", ".claude/", "build/", "dist/")


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(ROOT).with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _iter_sources() -> list[Path]:
    out = []
    for path in ROOT.rglob("*.py"):
        rel = path.relative_to(ROOT).as_posix()
        if "__pycache__" in rel or rel.startswith(_EXCLUDED_PREFIXES):
            continue
        out.append(path)
    return sorted(out)


def _catalog_reachable(core_modules: dict[str, Path]) -> set[str]:
    """Skills the live catalog loads by scanning the filesystem.

    `CapabilityEngine.reload_skills` builds its catalog from
    `core.skills.discovery.build_skill_catalog`, which walks the skill source
    roots and accepts what it finds. Nothing imports those modules and nothing
    names them as dotted strings, so both of this tool's notions of
    reachability miss them — and it reported ten skills that the running
    system loads and serves as reached by nothing.

    That is the dangerous direction for this tool to be wrong in. Its output
    is read as a retirement list, and a retirement pass built on it once
    proposed deleting 57 modules of which 21 were live. Asking the catalog is
    the fix; globbing `core/skills/*.py` is not, because it would also cover
    skill files the catalog rejects, which are exactly the ones worth seeing.

    If the catalog cannot be built, nothing is added. An unavailable loader
    must not silently promote every skill to reachable.
    """
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        from core.skills.discovery import build_skill_catalog

        catalog = build_skill_catalog()
    except Exception as exc:  # noqa: BLE001 - any failure means "no evidence"
        print(
            f"note: skill catalog unavailable ({type(exc).__name__}: {exc}); "
            "catalog-discovered skills are not counted as reachable",
            file=sys.stderr,
        )
        return set()

    reachable = {
        str(declaration.module_path)
        for declaration in getattr(catalog, "accepted", ())
        if str(getattr(declaration, "module_path", "")) in core_modules
    }
    return reachable


#: How a package walks its own directory and imports what it finds. The module
#: path is BUILT at run time out of the package name and a filename read off
#: the disk, so it exists in no import statement and in no string literal, and
#: a scan looking for either calls every member dead.
_SELF_ENUMERATION = (
    "pkgutil.iter_modules",
    "pkgutil.walk_packages",
    "iter_modules(__path__",
    "walk_packages(__path__",
)


def _reached_by_self_enumeration(
    sources: list[Path], core_modules: dict[str, Path]
) -> set[str]:
    """Members of a package that imports whatever is in its own directory.

    An external review ran a static reachability pass over this tree and
    concluded the forty-three interiority faculties were dead. They are not:
    ``core/interiority/faculties/__init__.py`` walks its own directory with
    ``pkgutil.iter_modules`` and imports every file whose name starts with f,
    and the service calls that on boot. Nothing in the repository ever writes
    ``core.interiority.faculties.f01_reading_others``, so a scan that looks for
    the dotted path finds nothing, and this tool would have said the same.

    That is the failure mode this whole file exists to prevent, made by the
    file itself. A tool that reports live code as dead is worse than no tool,
    because the number it produces is the one somebody deletes from.

    A package counts as self-enumerating when its ``__init__`` both walks its
    own path and imports by a built name. Walking without importing is a
    listing, and this does not treat a listing as a use.
    """

    reached: set[str] = set()
    for path in sources:
        if path.name != "__init__.py":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        walks = any(marker in text for marker in _SELF_ENUMERATION)
        if not walks or "import_module" not in text:
            continue
        package = _module_name(path)
        if package not in core_modules and not package.startswith("core."):
            continue
        for name in core_modules:
            if name.startswith(f"{package}."):
                reached.add(name)
    return reached


def _with_ancestor_packages(name: str, core_modules: dict[str, Any]) -> list[str]:
    """The module, plus every package Python would import on the way to it.

    ``from core.engineering.draw.schematic import x`` executes
    core/engineering/draw/__init__.py. Counting only the leaf left three
    packages reported unreachable while their own submodules were imported
    all over the tree — the same implicit-import blind spot this file already
    names when it excludes packages from the test-only count, applied one
    list over.
    """
    if name not in core_modules:
        return []
    found = [name]
    parts = name.split(".")
    for depth in range(len(parts) - 1, 0, -1):
        ancestor = ".".join(parts[:depth])
        if ancestor in core_modules:
            found.append(ancestor)
    return found


def scan() -> dict[str, object]:
    sources = _iter_sources()
    core_modules = {
        _module_name(p): p
        for p in sources
        if p.relative_to(ROOT).as_posix().startswith("core/")
    }

    referenced: set[str] = set()
    importers: dict[str, set[str]] = defaultdict(set)

    for path in sources:
        try:
            text = path.read_text(encoding="utf-8")
            tree = ast.parse(text)
        except (SyntaxError, UnicodeDecodeError, OSError):
            continue
        me = _module_name(path)
        is_package = path.name == "__init__.py"

        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                # Relative imports were skipped entirely, and that was a real
                # blind spot rather than a rounding error: `from .celery_app
                # import app` made celery_app invisible to this scan, so it was
                # reported unreachable while its own package imported it. A
                # retirement pass built on that reported 57 modules as safe to
                # delete and 21 of them were relatively imported by a sibling.
                #
                # `node.level` counts the leading dots. Level 1 is the
                # importer's own package, and each extra dot climbs one more.
                #
                # The package/module distinction is the whole difficulty and
                # getting it wrong is not visible in the output. For a module
                # `a.b.c` living in c.py, level 1 is `a.b`; for a package
                # `a.b` living in b/__init__.py, `_module_name` has already
                # dropped the `__init__` so level 1 is `a.b` itself. A single
                # formula therefore cannot serve both, and the version that
                # served only packages resolved `from ..cognitive.x import` in
                # core/phases/cognitive_routing.py to core.phases.cognitive.x
                # instead of core.cognitive.x — which reported a module that 51
                # others transitively import as unreachable.
                if node.level:
                    parts = me.split(".")
                    climb = node.level - 1 if is_package else node.level
                    base_parts = parts[: len(parts) - climb] if climb else parts
                    base = ".".join(base_parts)
                    prefix = f"{base}.{node.module}" if node.module else base
                else:
                    prefix = node.module or ""
                if prefix:
                    names = [prefix] + [f"{prefix}.{a.name}" for a in node.names]
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                # importlib.import_module("core.x.y"), registry tables, config
                # strings. A module reached this way is reached.
                candidate = node.value.strip()
                if candidate.startswith("core.") and " " not in candidate:
                    names = [candidate]
            for name in names:
                for candidate in _with_ancestor_packages(name, core_modules):
                    if candidate != me:
                        referenced.add(candidate)
                        importers[candidate].add(me)

    referenced |= _catalog_reachable(core_modules)
    referenced |= _reached_by_self_enumeration(sources, core_modules)

    orphans = sorted(set(core_modules) - referenced)
    lines = 0
    for name in orphans:
        try:
            lines += len(
                core_modules[name].read_text(encoding="utf-8", errors="ignore").splitlines()
            )
        except OSError:
            continue

    # Reached by tests and by nothing else. Not orphans — something imports
    # them — and not safe either, which is why they get their own number.
    #
    # `core/brain/prompt_builder.py` is the case that motivated this. It
    # builds the system prompt, withholds PERSON MODEL and INTERNAL SUBJECTIVE
    # STATE when the prompt may leave the host (the CP126 e18ed993 fix), and
    # emits a provenance manifest recording which components were injected.
    # Its only importer in the repository is its own test file. The live
    # prompt comes from `ContextAssembler.build_system_prompt`, which has none
    # of that — so a privacy protection and a provenance channel both read as
    # present, with twenty green tests behind them, while neither is in force.
    #
    # A test-only module is not automatically wrong: some are fixtures or
    # staged work. What is wrong is not being able to tell, and the count only
    # falls.
    # Packages are excluded. `core.adaptation` is imported implicitly by every
    # submodule import and this scan only sees the explicit ones, so an
    # `__init__.py` lands here whenever the only explicit `from core.x import`
    # in the tree is in a test — which says nothing about whether the package
    # is live.
    declared = _declared_experimental()
    test_only = sorted(
        name
        for name, path in core_modules.items()
        if path.name != "__init__.py"
        and name not in declared
        and importers[name]
        and not _production_importers(importers[name])
    )

    return {
        "core_modules": len(core_modules),
        "orphans": orphans,
        "orphan_count": len(orphans),
        "orphan_lines": lines,
        "test_only": test_only,
        "test_only_count": len(test_only),
    }


def _declared_experimental() -> dict[str, str]:
    """Modules that are a scientific rig, not a runtime path.

    An experiment legitimately has no production consumer: it exists to be run
    deliberately and to produce a result, and wiring it into the runtime to
    satisfy a counter would be worse than leaving it out. The test-only rule is
    right for a module that claims to do something at runtime and wrong for
    one that does not claim to, so the difference is declared rather than
    inferred — and each entry has to say why, because an undeclared list is
    where "experimental" becomes a synonym for "unfinished and forgotten".
    """
    path = ROOT / "config" / "experimental_modules.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        # Fail closed: an unreadable declaration excludes nothing.
        return {}
    modules = data.get("modules", {})
    if not isinstance(modules, dict):
        return {}
    return {str(k): str(v) for k, v in modules.items() if str(v).strip()}


def _production_importers(names: set[str]) -> set[str]:
    """Importers that are not themselves tests."""
    return {
        name
        for name in names
        if not (
            name.startswith("tests.")
            or name.startswith("test_")
            or ".test_" in name
            or name.endswith("_test")
            or name.startswith("conftest")
            or ".conftest" in name
        )
    }


def _load_baseline() -> dict[str, object] | None:
    if not BASELINE.is_file():
        return None
    return json.loads(BASELINE.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-baseline", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    report = scan()
    orphans: list[str] = report["orphans"]  # type: ignore[assignment]

    if args.list:
        for name in orphans:
            print(f"  {name}")
        print()

    if args.write_baseline:
        # The same guard core/runtime's lock ratchet needed: a refresh command
        # that can write a HIGHER count records the debt as the new normal and
        # the gate passes on it forever. Refreshing after a genuine
        # improvement is the only reason to run this.
        if BASELINE.is_file():
            try:
                was = int(json.loads(BASELINE.read_text()).get("orphan_count", 0))
            except (OSError, ValueError):
                was = 0
            if int(report["orphan_count"]) > was:
                print(
                    f"❌ refusing to raise the baseline: {was} -> "
                    f"{report['orphan_count']}. Wire the new orphans to "
                    "something, or retire them."
                )
                return 1
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE.write_text(
            json.dumps(
                {
                    "orphan_count": report["orphan_count"],
                    "orphan_lines": report["orphan_lines"],
                    "orphans": orphans,
                    "test_only_count": report["test_only_count"],
                    "test_only": report["test_only"],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            f"wrote baseline: {report['orphan_count']} orphans, "
            f"{report['orphan_lines']:,} lines"
        )
        return 0

    baseline = _load_baseline()
    print(
        f"🕸  {report['orphan_count']} of {report['core_modules']} core modules are "
        f"reached by nothing ({report['orphan_lines']:,} lines)"
    )

    if baseline is None:
        print("no baseline yet — run with --write-baseline")
        return 0

    known = set(baseline.get("orphans", []))
    new = sorted(set(orphans) - known)
    closed = sorted(known - set(orphans))

    if closed:
        print(f"✅ {len(closed)} newly reachable (baseline should shrink): {closed[:8]}")

    if new:
        print(f"\n❌ {len(new)} module(s) became unreachable:")
        for name in new:
            print(f"   {name}")
        print(
            "\nEither wire it to something, or retire it. A module nothing reaches "
            "is where a half-wired subsystem hides — this repo has shipped a second "
            "affect engine and an unreachable fallback exactly this way."
        )
        return 1

    if len(orphans) > int(baseline.get("orphan_count", 0)):
        print("❌ orphan count rose without a new module name — refresh the baseline")
        return 1

    # Counting them stops the number growing and says nothing about what any
    # one of them IS. "279 unreachable" is not actionable; "279 decided" is.
    dispositions = ROOT / "config" / "orphan_dispositions.json"
    if dispositions.is_file():
        decided = json.loads(dispositions.read_text(encoding="utf-8")).get("modules", {})
        undecided = sorted(set(orphans) - set(decided))
        if undecided:
            print(f"\n❌ {len(undecided)} unreachable module(s) with no recorded decision:")
            for name in undecided[:10]:
                print(f"   {name}")
            print(
                "\nRun tools/triage_orphan_modules.py --write-dispositions to seed "
                "them, then change any default that is wrong. A module nobody has "
                "decided about is how a half-wired subsystem survives."
            )
            return 1
        counts: dict[str, int] = {}
        for entry in decided.values():
            key = str(entry.get("disposition", "?"))
            counts[key] = counts.get(key, 0) + 1
        summary = ", ".join(f"{v} {k.lower()}" for k, v in sorted(counts.items()))
        print(f"   dispositions: {summary}")

    test_only: list[str] = report["test_only"]  # type: ignore[assignment]
    known_test_only = set(baseline.get("test_only", []))
    new_test_only = sorted(set(test_only) - known_test_only)
    print(
        f"🧪 {report['test_only_count']} core module(s) are reached only by tests"
    )
    if new_test_only:
        print(f"\n❌ {len(new_test_only)} module(s) became test-only:")
        for name in new_test_only:
            print(f"   {name}")
        print(
            "\nA module whose only importer is its own test is not exercised by "
            "anything that runs. core/brain/prompt_builder.py sat like this with "
            "a privacy protection and a provenance manifest that were never in "
            "force, behind twenty passing tests."
        )
        return 1

    print("✅ no new unreachable modules")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
