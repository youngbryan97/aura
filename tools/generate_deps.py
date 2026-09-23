#!/usr/bin/env python3
"""Write a DEPS file for every core package, from the graph that exists.

``tools/check_layering.py`` is a working clean-room ``checkdeps``, and seven
of ``core``'s 158 packages had a DEPS file. So "Aura enforces architectural
layering" was true of 4% of the tree, which is the shape an external review
named: good mechanism, partial deployment.

The seven were hand-written and say what a foundation package may not reach
for — cognition, agency, the interface. Writing 139 more of those means
inventing a layer for every package in the repository and being wrong about
some of them, which produces either false rules or a baseline full of
exceptions. This does something narrower and checkable instead.

**The rule is the graph.** For each package, the generated DEPS allows
exactly the packages that package imports today and forbids the rest of
``core``, plus ``interface``, ``skills`` and ``aura_main``. Nothing that
compiles today stops compiling, the ratchet baseline gains nothing, and every
NEW cross-package dependency becomes an edit to a DEPS file — a line in a
diff a reviewer can see, rather than an import nobody notices.

That is the property a layering rule is for. A taxonomy can be added on top
later, package by package, by replacing a generated file with a written one;
the seven existing files are exactly that and are left alone.

A rule narrower than a package survives regeneration. Where a generated file
already names a module (`"+core.self.what_came_before"` for a package that
reads one ledger from core.self), and every import the package makes into
that package is inside the modules it names, the module rules and the comment
lines above them are written back in place of the package rule. Coverage is
judged with the layering checker's own import resolution and rule matching,
so the generator and the gate cannot disagree about what a rule allows. An
import outside the named modules adds a rule for that module, which is a line
in a diff like any other new edge; a `from core.X import name` resolves to
core.X itself and only the package rule covers it. The same regeneration used
to widen every such rule to the whole package and drop its comment, which is
why nine packages below are written rather than generated.

    python tools/generate_deps.py --check    # regenerate nothing, compare
    python tools/generate_deps.py --write
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORE = ROOT / "core"

# Run as `python tools/generate_deps.py`, which puts tools/ on the import path
# and the repository nowhere on it.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.check_layering import Rule, imports_of  # noqa: E402

#: Hand-written files, left alone. Each states an architectural intent that a
#: generated closed-world list would replace with a weaker fact.
#: `engineering` is written rather than generated because its rule is the
#: reason it exists: it may not reach anything that generates a number. A
#: closed-world list of what it happens to import today would allow a later
#: import of core.brain the moment somebody added one.
#: `conation` is written for the same reason. Its rule is that a motivational
#: state may never be produced by generated text, and a closed-world list of
#: today's imports would allow core.brain the moment somebody added one — which
#: is precisely the import that the organ's own invariant exists to forbid.
#: Packages whose DEPS says something a generator cannot derive. The check is
#: byte-exact against generated content, so a real rule written into a
#: generated file makes the gate red until somebody regenerates it away —
#: which is how core/learning's rule about the procedure currency sat here
#: with `make deps-check` failing, one regeneration from being deleted.
HANDWRITTEN = {
    "conation", "engineering", "fsw", "health", "learning", "observability",
    "persistence", "runtime", "utils", "verify",
    # The measurer must not become part of what it measures. core/subject
    # reads the running organism to take a reading, and the rule is one-way:
    # nothing in the runtime may import subject, because a measurement the
    # measured thing can read is partly about itself. A generated list would
    # say only "what it imports today" and would widen in the direction the
    # rule exists to hold. It was generated once, on 2026-09-07, and the
    # written rule survived by one command.
    "subject",
    # The next nine are written for one reason each: a rule that names a
    # MODULE. Until 22 September 2026 the generator wrote package-level rules
    # only, so regenerating one of these widened its rule to the whole package
    # and dropped the comment. It now keeps a module rule that still covers
    # every import (`narrowings`), and a render of all nine that day gave
    # exactly the rules each file holds. They stay written because their
    # descriptions and header prose are written too; moving those into
    # comments above each rule would let the generator take them back.
    # Three packages whose rule names a MODULE rather than a package.
    # Narrative classification reads core.conversation.word_markers, the
    # heuristic imperatives read the same module to match a principle, and
    # cycle observation reads core.state.percepts; the generator emitted
    # package-level rules only, so regenerating any of them widened it to the
    # whole of core.conversation or the whole of core.state and the narrowing
    # was gone with the gate still green. core/values was the third and was not
    # listed, so `make deps-check` was red on it and the only way to satisfy
    # the gate was to widen the rule.
    "consciousness", "world_model", "values",
    # Two more with a module-level rule. The post-action verifier and the task
    # graph each register an invariant and take nothing else from core.verify,
    # so each names core.verify.invariants. Regenerating widened both to the
    # whole of core.verify, the verifier that checks them included.
    "capabilities", "planning",
    # And one more. Steering and recurrent evaluation decode the public channel
    # and take nothing else from core.brain; regenerating widened that to the
    # cognition the evaluation scores.
    "evaluation",
    # The other direction of that same edge. Frontier certification takes
    # the exact paired test from core.evaluation and nothing else there, so
    # core/brain names core.evaluation.paired_power. Regenerating widened it to
    # the whole of core.evaluation — the scorers of the cognition this package
    # is — with the gate still green.
    "brain",
    # And one more of the module-level kind. The registered
    # knowledge-revision canary exercises the canonical store and takes
    # nothing else from core.knowledge, so core/organism names
    # core.knowledge.revision_validation. Regenerating widened that to the
    # whole of core.knowledge with the gate still green — and until this
    # entry, `make deps-check` was red for exactly the reason the comment
    # at the top of this block describes: a real rule written into a
    # generated file, one regeneration from being deleted.
    "organism",
    # And the kernel. Organ startup warms the shared sentence encoder the
    # memory organ uses, and takes nothing else from core.memory, so
    # core/kernel names core.memory.embedding_runtime. The rule was written
    # into a generated file, so `make deps-check` was red on it and the only
    # command that turned it green widened it to the whole of core.memory.
    "kernel",
    # The judge must not be able to reach the defendant. core/phenomenology
    # decides whether evidence supports a claim about this system, so its rule
    # is "imports nothing from core" rather than "what it imports today" — a
    # generated DEPS would widen it silently the first time a protocol reached
    # for the organ it is meant to perturb.
    "phenomenology",
    # The estimated must not be able to reach the estimators. Every subsystem
    # that used to own its own copy of "how Aura is" writes into
    # core/canonical, so a generated DEPS would widen it the first time the
    # state layer went and fetched a value instead of being told one — and
    # then there would be two answers again, which is what it exists to end.
    "canonical",
}

SKIP_DIRS = {"__pycache__", ".venv", "node_modules", "archive"}

#: Roots outside core that a core package must not reach into. `interface` is
#: the UI, `skills` the loadable skill tree, `aura_main` the entry point.
FORBIDDEN_ROOTS = ("interface", "skills", "aura_main")

#: Trees outside core that import core. Each gets one DEPS at its root, with
#: the same closed-world rule: what it reaches for today is allowed and the
#: rest of core is not, so a new coupling between the UI and a cognitive
#: internal is a line in a diff.
OUTSIDE_ROOTS = ("interface", "skills", "security", "llm", "executors")

OUTSIDE_HEADER = """# {root}/ — layering rules. Generated by tools/generate_deps.py.
#
# The `+` lines are the core packages this tree imports today. The `-core`
# after them forbids the rest, so a new dependency from here into a cognitive
# or runtime internal is an edit to this file. Regenerate with
# `make deps-generate`.
#
# {counted} core packages reached when this was generated.

description = "{description}"

include_rules = [
"""

HEADER = """# core/{package} — layering rules. Generated by tools/generate_deps.py.
#
# The `+` lines are what this package imports today, package by package. The
# `-core` line after them forbids the rest, so a new cross-package dependency
# is an edit to this file rather than an import nobody sees. Regenerate with
# `make deps-generate`; replace this file with a written one when this
# package earns a real architectural rule, the way core/runtime/DEPS has.
#
# {counted} outbound edges when this was generated.

description = "{description}"

include_rules = [
"""


def packages() -> list[str]:
    """Every directory under core that holds Python.

    Not "every directory with an ``__init__.py``": more than half of these are
    namespace packages — ``core/health``, ``core/media``, ``core/learning``
    and ``core/bus`` among them — and requiring the marker file made the first
    run of this generator miss them, which turned real imports into
    violations.
    """
    return sorted(
        d.name
        for d in CORE.iterdir()
        if d.is_dir() and d.name not in SKIP_DIRS and any(d.rglob("*.py"))
    )


def top_level_modules() -> set[str]:
    return {p.stem for p in CORE.glob("*.py") if p.stem != "__init__"}


def imports_by_package() -> dict[str, set[str]]:
    """Which `core.X` and outside roots each package reaches for."""
    known_packages = set(packages())
    known_modules = top_level_modules()
    found: dict[str, set[str]] = defaultdict(set)

    for path in sorted(CORE.rglob("*.py")):
        if SKIP_DIRS & set(path.parts):
            continue
        relative = path.relative_to(CORE)
        if len(relative.parts) < 2:
            continue
        package = relative.parts[0]
        if package not in known_packages:
            continue
        try:
            tree = ast.parse(path.read_text("utf-8", errors="ignore"), filename=str(path))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
                modules = [node.module]
            for module in modules:
                parts = module.split(".")
                if parts[0] != "core":
                    if parts[0] in FORBIDDEN_ROOTS:
                        found[package].add(parts[0])
                    continue
                if len(parts) == 1:
                    continue
                head = parts[1]
                if head in known_packages:
                    found[package].add(f"core.{head}")
                elif head in known_modules:
                    found[package].add(f"core.{head}")
    return found


#: The comment lines this tool writes itself. They are never carried over as
#: the comment of a written rule that happens to follow one.
_SECTION_COMMENTS = frozenset(
    {
        "# What this package reaches for today.",
        "# Outside core. Each of these is a layering break already in",
        "# the tree; recorded here so it is visible where the rule is,",
        "# and so removing it is a one-line diff.",
        "# Everything else.",
    }
)

_MODULE_RULE = re.compile(r'^\s*"\+(core\.[A-Za-z_][\w.]*)",\s*$')


def written_narrowings(text: str) -> dict[str, tuple[str, ...]]:
    """Every module-level `+` rule in a DEPS file, with the comment lines directly above it.

    A module-level rule names something inside a core package, so it has at
    least three dotted parts: `core.self.what_came_before`. A blank line or
    any other line ends the comment that belongs to the next rule.
    """
    known = set(packages())
    found: dict[str, tuple[str, ...]] = {}
    comment: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            if stripped not in _SECTION_COMMENTS:
                comment.append(stripped)
            continue
        matched = _MODULE_RULE.match(line)
        if matched:
            parts = matched.group(1).split(".")
            if len(parts) >= 3 and parts[1] in known:
                found[matched.group(1)] = tuple(comment)
        comment = []
    return found


def checker_imports(package: str) -> set[str]:
    """Every core module a package imports, resolved as tools/check_layering.py resolves it."""
    found: set[str] = set()
    for path in sorted((CORE / package).rglob("*.py")):
        if SKIP_DIRS & set(path.parts):
            continue
        found.update(name for name, _line in imports_of(path) if name.startswith("core."))
    return found


def narrowings(
    package: str,
    allowed: set[str],
    written: dict[str, tuple[str, ...]],
    imported: set[str],
) -> dict[str, list[tuple[str, tuple[str, ...]]]]:
    """Which package rules the written module rules can stand in for, and with what.

    Keyed by the package rule (`core.self`); each value is the module rules to
    write in its place, each with its comment. A written rule no import still
    needs is dropped, as a package rule is when its last import goes. An
    import into the package that none of them covers gets a rule of its own
    at the module the checker resolved. An import of the package itself,
    `from core.self import name`, can only be covered by the package rule, so
    that package is written whole.
    """
    by_target: dict[str, list[str]] = defaultdict(list)
    for prefix in written:
        by_target[".".join(prefix.split(".")[:2])].append(prefix)
    result: dict[str, list[tuple[str, tuple[str, ...]]]] = {}
    for target, prefixes in sorted(by_target.items()):
        if target == f"core.{package}" or target not in allowed:
            continue
        reached = {m for m in imported if m == target or m.startswith(target + ".")}
        if not reached or target in reached:
            continue
        rules = [Rule(kind="+", prefix=prefix) for prefix in prefixes]
        kept = sorted(r.prefix for r in rules if any(r.matches(m) for m in reached))
        added = sorted(m for m in reached if not any(r.matches(m) for r in rules))
        result[target] = [(prefix, written.get(prefix, ())) for prefix in sorted(kept + added)]
    return result


def render(
    package: str,
    allowed: set[str],
    narrowed: dict[str, list[tuple[str, tuple[str, ...]]]] | None = None,
) -> str:
    narrowed = narrowed or {}
    inside = sorted(a for a in allowed if a.startswith("core."))
    outside = sorted(a for a in allowed if not a.startswith("core."))

    lines = [HEADER.format(
        package=package,
        counted=len({a for a in allowed if a != f"core.{package}"}),
        description=(
            f"core.{package}: may import only what it already imports; "
            "every new edge is an edit here."
        ),
    )]
    lines.append(f'    "+core.{package}",\n')
    # The package's own name is already the line above. A submodule importing a
    # sibling puts it in the graph as an outbound edge, and emitting it again
    # here writes a duplicate rule and an outbound count one too high.
    inside = [name for name in inside if name != f"core.{package}"]
    if inside:
        lines.append("\n    # What this package reaches for today.\n")
        for name in inside:
            for prefix, comment in narrowed.get(name, [(name, ())]):
                lines.extend(f"    {remark}\n" for remark in comment)
                lines.append(f'    "+{prefix}",\n')
    if outside:
        lines.append(
            "\n    # Outside core. Each of these is a layering break already in\n"
            "    # the tree; recorded here so it is visible where the rule is,\n"
            "    # and so removing it is a one-line diff.\n"
        )
        for name in outside:
            lines.append(f'    "+{name}",\n')
    lines.append("\n    # Everything else.\n")
    lines.append('    "-core",\n')
    for root in FORBIDDEN_ROOTS:
        if root not in outside:
            lines.append(f'    "-{root}",\n')
    lines.append("]\n")
    return "".join(lines)


def outside_imports(root: str) -> set[str]:
    """Which `core.X` a tree outside core reaches for."""
    known_packages = set(packages())
    known_modules = top_level_modules()
    base = ROOT / root
    found: set[str] = set()
    paths = [base] if base.is_file() else list(base.rglob("*.py"))
    for path in paths:
        if SKIP_DIRS & set(path.parts):
            continue
        try:
            tree = ast.parse(path.read_text("utf-8", errors="ignore"), filename=str(path))
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
                if "." in node.module:
                    modules = [node.module]
                else:
                    modules = [f"{node.module}.{a.name}" for a in node.names]
            for module in modules:
                parts = module.split(".")
                if parts[0] != "core" or len(parts) == 1:
                    continue
                head = parts[1]
                if head in known_packages or head in known_modules:
                    found.add(f"core.{head}")
    return found


def render_outside(root: str, allowed: set[str]) -> str:
    lines = [OUTSIDE_HEADER.format(
        root=root,
        counted=len(allowed),
        description=(
            f"{root}: may import only the core packages it already imports."
        ),
    )]
    for name in sorted(allowed):
        lines.append(f'    "+{name}",\n')
    lines.append("\n    # Everything else in core.\n")
    lines.append('    "-core",\n')
    lines.append("]\n")
    return "".join(lines)


def _generated_paths() -> set[str]:
    """Every DEPS file this tool is responsible for."""
    found = {
        f"{root}/DEPS" for root in OUTSIDE_ROOTS if (ROOT / root).is_dir()
    }
    found |= {
        f"core/{package}/DEPS"
        for package in packages()
        if package not in HANDWRITTEN
    }
    return found


def _unpublished(paths: list[str]) -> list[str]:
    """Rules git does not carry are rules CI never sees.

    `.gitignore` had an unanchored `data/`, which matches a directory of that
    name at any depth — so `core/data/DEPS` was generated here, ignored by
    git, and missing in CI, where the check failed on a file it could not
    read. A generated rule that does not reach the repository is not a rule.
    """
    missing: list[str] = []
    for relative in paths:
        if not (ROOT / relative).exists():
            continue
        tracked = subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["git", "-c", "core.fsmonitor=false", "ls-files", "--error-unmatch", relative],
            capture_output=True,
            cwd=ROOT,
            check=False,
        )
        if tracked.returncode != 0:
            missing.append(f"{relative} (generated but not tracked by git)")
    return missing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    graph = imports_by_package()
    written = 0
    stale: list[str] = []

    for root in OUTSIDE_ROOTS:
        base = ROOT / root
        if not base.is_dir():
            continue
        target = base / "DEPS"
        content = render_outside(root, outside_imports(root))
        if args.check:
            if not target.exists() or target.read_text("utf-8") != content:
                stale.append(f"{root}/DEPS")
        elif args.write:
            target.write_text(content, encoding="utf-8")
            written += 1

    for package in packages():
        if package in HANDWRITTEN:
            continue
        target = CORE / package / "DEPS"
        allowed = graph.get(package, set())
        by_hand = written_narrowings(target.read_text("utf-8")) if target.exists() else {}
        narrowed = narrowings(package, allowed, by_hand, checker_imports(package)) if by_hand else {}
        content = render(package, allowed, narrowed)
        if args.check:
            if not target.exists() or target.read_text("utf-8") != content:
                stale.append(f"core/{package}/DEPS")
            continue
        if args.write:
            target.write_text(content, encoding="utf-8")
            written += 1

    if args.check:
        stale.extend(_unpublished(sorted(_generated_paths())))
        if stale:
            print(f"❌ {len(stale)} DEPS file(s) do not match the import graph")
            for name in stale[:20]:
                print(f"   • {name}")
            print("\nRun `make deps-generate` in the commit that changed the imports.")
            return 1
        print(f"✅ every generated DEPS matches the graph ({len(packages())} packages)")
        return 0

    if args.write:
        print(f"wrote {written} DEPS files ({len(HANDWRITTEN)} hand-written left alone)")
        return 0

    print(json.dumps({p: sorted(v) for p, v in sorted(graph.items())}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
