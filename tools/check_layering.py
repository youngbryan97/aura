#!/usr/bin/env python3
"""tools/check_layering.py — architectural layering gate.

Clean-room adoption of Chromium's `DEPS` / `checkdeps` mechanism.

Architecture documents describe intended layering. Code enforces actual
layering. When only the first exists, the two diverge silently — a
foundation module grows an import of a cognitive one, then another, and a
year later "core/runtime is the foundation" is a sentence in a document
that the import graph contradicts. Nobody did it on purpose; each import
was locally reasonable.

Chromium's answer is a `DEPS` file per directory listing `include_rules`:

    include_rules = [
        "+core.runtime",       # allowed
        "-core.brain",         # forbidden
        "!core.brain.legacy",  # forbidden, but existing uses are grandfathered
    ]

Rules inherit down the tree, the most specific matching rule wins, and the
build fails on a violation. The rules are the architecture, and they are
executable.

This runs as a make gate with a **ratchet baseline**: existing violations
are recorded in `config/layering_baseline.json` and tolerated; new ones
fail. The baseline may only shrink. That is the same discipline
`tests/test_async_write_lane_ratchet.py` already applies to sync writes in
async code, and it is the only way to introduce a rule into a large
existing codebase without a flag day.

    python tools/check_layering.py            # check
    python tools/check_layering.py --baseline # rewrite the baseline (shrink only)
    python tools/check_layering.py --stats    # what the graph looks like now
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASELINE_PATH = PROJECT_ROOT / "config" / "layering_baseline.json"
DEPS_FILENAME = "DEPS"

#: Every tree that carries include rules. A tree with no DEPS contributes
#: nothing, so adding one here is safe before its rules exist.
DEFAULT_ROOTS = tuple(
    str(PROJECT_ROOT / name)
    for name in ("core", "interface", "skills", "security", "llm", "executors")
)


@dataclass(frozen=True)
class Rule:
    """One include rule. ``kind`` is '+', '-', or '!'."""

    kind: str
    prefix: str

    def matches(self, module: str) -> bool:
        return module == self.prefix or module.startswith(self.prefix + ".")


@dataclass
class DepsFile:
    directory: Path
    rules: list[Rule] = field(default_factory=list)
    description: str = ""


@dataclass(frozen=True)
class Violation:
    source: str
    imported: str
    rule: str
    line: int

    def key(self) -> str:
        return f"{self.source}::{self.imported}"

    def __str__(self) -> str:
        return f"{self.source}:{self.line}: imports {self.imported} (forbidden by {self.rule})"


def parse_deps(path: Path) -> DepsFile:
    """Parse the two allowed literal assignments in a DEPS file."""
    deps = DepsFile(directory=path.parent)
    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return deps
    namespace: dict[str, object] = {}
    try:
        tree = ast.parse(source, filename=str(path), mode="exec")
        for statement in tree.body:
            if (
                isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Constant)
                and isinstance(statement.value.value, str)
            ):
                continue
            if (
                not isinstance(statement, ast.Assign)
                or len(statement.targets) != 1
                or not isinstance(statement.targets[0], ast.Name)
                or statement.targets[0].id not in {"description", "include_rules"}
            ):
                raise ValueError(
                    "DEPS permits only literal description/include_rules assignments"
                )
            namespace[statement.targets[0].id] = ast.literal_eval(statement.value)
    except (SyntaxError, ValueError, TypeError) as exc:
        print(f"error: cannot parse {path}: {exc}", file=sys.stderr)
        return deps
    deps.description = str(namespace.get("description", "") or "")
    for entry in namespace.get("include_rules", []) or []:
        text = str(entry).strip()
        if not text or text.startswith("#"):
            continue
        kind, prefix = text[0], text[1:].strip()
        if kind not in "+-!" or not prefix:
            print(f"error: {path}: malformed rule {text!r}", file=sys.stderr)
            continue
        deps.rules.append(Rule(kind=kind, prefix=prefix))
    return deps


def collect_deps(root: Path) -> dict[Path, DepsFile]:
    return {
        path.parent: parse_deps(path)
        for path in sorted(root.rglob(DEPS_FILENAME))
        if path.is_file()
    }


def rules_for(directory: Path, deps: dict[Path, DepsFile], root: Path) -> list[Rule]:
    """Rules inherit down the tree; nearest directory's rules come first."""
    collected: list[Rule] = []
    current = directory
    reached_boundary = False
    while not reached_boundary:
        entry = deps.get(current)
        if entry is not None:
            collected.extend(entry.rules)
        parent = current.parent
        reached_boundary = current == root or parent == current
        if not reached_boundary:
            current = parent
    return collected


def module_name(path: Path, root: Path) -> str:
    relative = path.relative_to(root).with_suffix("")
    parts = list(relative.parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def imports_of(path: Path) -> Iterator[tuple[str, int]]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError):
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, node.lineno
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import: same package, never a layering break
                continue
            if not node.module:
                continue
            if "." in node.module:
                yield node.module, node.lineno
                continue
            # `from core import governance_context` is a dependency on
            # core.governance_context, and reporting it as a dependency on
            # `core` made it unmatchable by any per-package rule: a rule
            # specific enough to allow the submodule cannot match the bare
            # package name, and a rule that does match the bare name matches
            # everything under it. A dotted module is already specific, so
            # only the top-level form is expanded, which also leaves the
            # baseline's existing keys untouched.
            for alias in node.names:
                yield f"{node.module}.{alias.name}", node.lineno


def check_module(module: str, rules: list[Rule]) -> Rule | None:
    """First matching rule wins. Returns the rule if it forbids."""
    for rule in rules:
        if rule.matches(module):
            return rule if rule.kind in "-!" else None
    return None


def scan(root: Path, *, package_root: Path | None = None) -> list[Violation]:
    package_root = package_root or root
    deps = collect_deps(root)
    if not deps:
        return []
    violations: list[Violation] = []
    for path in sorted(root.rglob("*.py")):
        if any(part in {"__pycache__", ".venv", "node_modules"} for part in path.parts):
            continue
        directory_rules = rules_for(path.parent, deps, root)
        if not directory_rules:
            continue
        source = str(path.relative_to(package_root.parent))
        for imported, line in imports_of(path):
            broken = check_module(imported, directory_rules)
            if broken is None:
                continue
            violations.append(
                Violation(
                    source=source,
                    imported=imported,
                    rule=f"{broken.kind}{broken.prefix}",
                    line=line,
                )
            )
    return violations


def load_baseline() -> set[str]:
    try:
        payload = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return set()
    return set(payload.get("grandfathered", []))


def write_baseline(violations: list[Violation], *, previous: set[str]) -> tuple[int, int]:
    """Rewrite the baseline. Once seeded, it may only shrink."""
    current = {v.key() for v in violations}
    # The ratchet: anything already fixed stays fixed, and nothing new is
    # ever admitted. On the first run there is no baseline to ratchet
    # against, so the current set seeds it.
    keep = current if not BASELINE_PATH.exists() else current & previous
    removed = len(previous - keep)
    added = len(keep - previous)
    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    BASELINE_PATH.write_text(
        json.dumps(
            {
                "description": (
                    "Grandfathered layering violations. This list may only "
                    "SHRINK: tools/check_layering.py fails on any violation not "
                    "listed here, and refuses to add new entries."
                ),
                "count": len(keep),
                "grandfathered": sorted(keep),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return added, removed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", action="store_true", help="rewrite the ratchet baseline")
    parser.add_argument("--stats", action="store_true", help="print graph statistics")
    parser.add_argument(
        "--root",
        action="append",
        default=None,
        help="a tree to scan; repeatable. Defaults to every tree that has DEPS.",
    )
    args = parser.parse_args(argv)

    # `core` alone was the default, so the trees that import core — the UI, the
    # skills, the security package — had no rules and could not have had any.
    roots = [Path(r).resolve() for r in (args.root or DEFAULT_ROOTS)]
    violations: list[Violation] = []
    for root in roots:
        if root.exists():
            violations.extend(scan(root, package_root=root))
    baseline = load_baseline()

    if args.baseline:
        added, removed = write_baseline(violations, previous=baseline)
        if added:
            print(
                f"refused: {added} violation(s) are new and cannot be added to the "
                "baseline — fix them or change the DEPS rule deliberately",
                file=sys.stderr,
            )
            for violation in violations:
                if violation.key() not in baseline:
                    print(f"  {violation}", file=sys.stderr)
            return 1
        print(f"baseline rewritten: {removed} entr(ies) retired, {len(baseline) - removed} remain")
        return 0

    if args.stats:
        by_rule: dict[str, int] = {}
        for violation in violations:
            by_rule[violation.rule] = by_rule.get(violation.rule, 0) + 1
        print(json.dumps({"total": len(violations), "by_rule": by_rule}, indent=2))
        return 0

    fresh = [v for v in violations if v.key() not in baseline]
    fixed = baseline - {v.key() for v in violations}

    if fresh:
        print(f"❌ {len(fresh)} new layering violation(s):", file=sys.stderr)
        for violation in fresh:
            print(f"  {violation}", file=sys.stderr)
        print(
            "\nEither remove the import, or change the DEPS rule deliberately and "
            "say why in its description.",
            file=sys.stderr,
        )
        return 1

    message = f"✅ layering clean ({len(baseline)} grandfathered)"
    if fixed:
        message += f"; {len(fixed)} baseline entr(ies) now fixed — run --baseline to retire them"
    print(message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
