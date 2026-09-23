"""The layering rule covers the repository, and it refuses a new edge.

`tools/check_layering.py` has been a working checkdeps for months and seven
of core's 158 packages had a DEPS file, so "architectural boundaries are
enforced" described 4% of the tree. The trees that import core — the UI, the
skills, the security package — had no rules at all, because the checker only
ever scanned `core`.

Coverage is one claim and enforcement is another, so both are checked here:
every package has rules, and a cross-package import that no rule allows is
reported. The second half is the one that matters. A gate with complete
coverage and no teeth is the same as no gate, and this repository has found
that shape enough times to stop assuming it.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
CHECKER = ROOT / "tools" / "check_layering.py"

SKIP = {"__pycache__", ".venv", "node_modules", "archive"}


def _load_checker():
    """Import the gate by path, registered under a name.

    `spec_from_file_location` alone is not enough: the module defines
    dataclasses, and `dataclasses` resolves a field's namespace through
    `sys.modules[cls.__module__]`. Leaving it unregistered makes that lookup
    return None and the class construction raise.
    """
    from importlib import util

    name = "_aura_layering_checker"
    if name in sys.modules:
        return sys.modules[name]
    spec = util.spec_from_file_location(name, CHECKER)
    module = util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _core_packages() -> list[Path]:
    return sorted(
        d
        for d in CORE.iterdir()
        if d.is_dir() and d.name not in SKIP and any(d.rglob("*.py"))
    )


def test_every_core_package_has_include_rules():
    missing = [d.name for d in _core_packages() if not (d / "DEPS").exists()]
    assert not missing, (
        f"{len(missing)} core packages have no DEPS and therefore no rules: "
        f"{missing[:15]}"
    )


def test_the_trees_that_import_core_have_rules_too():
    for name in ("interface", "skills", "security", "llm"):
        tree = ROOT / name
        if not tree.is_dir():
            continue
        assert (tree / "DEPS").exists(), (
            f"{name}/ imports core and has no rules; the checker used to scan "
            "core only, so this tree could not have been constrained"
        )


def test_the_checker_scans_more_than_core():
    module = _load_checker()
    roots = {Path(r).name for r in module.DEFAULT_ROOTS}
    assert {"core", "interface", "skills"} <= roots


def test_the_tree_is_clean_against_its_own_rules():
    result = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, str(CHECKER)], capture_output=True, text=True, cwd=ROOT
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_a_new_cross_package_import_is_refused(tmp_path):
    """The negative control.

    Complete coverage proves nothing on its own — a DEPS file that allows
    everything is a DEPS file. This plants an import no rule permits and
    requires the checker to name it.
    """
    module = _load_checker()

    fake_root = tmp_path / "core"
    package = fake_root / "memory"
    package.mkdir(parents=True)
    (package / "DEPS").write_text(
        'description = "test"\n'
        'include_rules = [\n'
        '    "+core.memory",\n'
        '    "+core.config",\n'
        '    "-core",\n'
        '    "-interface",\n'
        ']\n',
        encoding="utf-8",
    )
    (package / "store.py").write_text(
        "from core.config import config\n"
        "from core.brain.inference_gate import InferenceGate\n"
        "from interface.routes import chat\n",
        encoding="utf-8",
    )

    violations = module.scan(fake_root, package_root=fake_root)
    forbidden = {v.imported for v in violations}
    assert "core.brain.inference_gate" in forbidden
    assert "interface.routes" in forbidden
    assert not any(v.imported.startswith("core.config") for v in violations)


def test_a_from_package_import_names_the_submodule():
    """`from core import governance_context` depends on core.governance_context.

    Reported as a dependency on `core`, it was unmatchable: a rule specific
    enough to allow the submodule cannot match the bare package name, and a
    rule that does match the bare name matches everything beneath it.
    """
    module = _load_checker()

    sample = Path(__file__).parent / "_layering_import_sample.py"
    sample.write_text(
        "from core import governance_context\n"
        "from core.runtime.errors import record_degradation\n",
        encoding="utf-8",
    )
    try:
        found = {name for name, _line in module.imports_of(sample)}
    finally:
        sample.unlink()

    assert "core.governance_context" in found
    assert "core" not in found
    assert "core.runtime.errors" in found


@pytest.mark.parametrize("package", ["runtime", "observability", "verify", "fsw"])
def test_the_written_rules_were_not_replaced_by_generated_ones(package):
    """The hand-written files say what a package may NOT reach for.

    A generated closed-world list would replace that intent with a weaker
    fact — "these are the imports that exist" — and the intent is the part
    worth keeping.
    """
    body = (CORE / package / "DEPS").read_text("utf-8")
    assert "Generated by tools/generate_deps.py" not in body
    assert "-core.brain" in body


def _generator():
    """`tools/generate_deps.py`, imported by path."""
    from importlib import util

    name = "_aura_deps_generator"
    if name in sys.modules:
        return sys.modules[name]
    spec = util.spec_from_file_location(name, ROOT / "tools" / "generate_deps.py")
    module = util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


GENERATED_HEADER = "Generated by tools/generate_deps.py"


def test_a_written_rule_is_registered_where_the_generator_can_see_it():
    """No DEPS file states a rule the next regeneration would delete.

    The generator leaves alone the packages named in its own HANDWRITTEN set
    and rewrites every other DEPS file from the import graph, so a rule
    written into an unregistered file survives until somebody runs
    `make deps-generate`. `make deps-check` goes red in the meantime, which
    looks like a stale file and invites exactly the command that deletes the
    rule. core/learning sat in that state once; core/subject sat in it on
    2026-09-07 and was one command from losing "the organism may not read
    the thing that measures it".

    Both directions are checked. An unregistered file must carry the
    generated header, and a registered one must not — a name left in the set
    after its rule became a generated list stops the generator updating a
    file nobody is maintaining by hand.
    """

    handwritten = set(_generator().HANDWRITTEN)
    unregistered_but_written = []
    registered_but_generated = []
    for deps in sorted(CORE.rglob("DEPS")):
        package = deps.parent.name
        body = deps.read_text("utf-8")
        if package in handwritten:
            if GENERATED_HEADER in body:
                registered_but_generated.append(package)
        elif GENERATED_HEADER not in body:
            unregistered_but_written.append(package)

    assert not unregistered_but_written, (
        "written DEPS rules the generator would overwrite: "
        f"{unregistered_but_written} — add each to HANDWRITTEN in "
        "tools/generate_deps.py, saying why the rule cannot be derived"
    )
    assert not registered_but_generated, (
        "packages held back from the generator that no longer say anything "
        f"it could not derive: {registered_but_generated}"
    )


def test_every_registered_name_still_has_a_deps_file():
    """A HANDWRITTEN name whose package moved or was renamed protects nothing."""

    missing = [
        package
        for package in sorted(_generator().HANDWRITTEN)
        if not (CORE / package / "DEPS").is_file()
    ]
    assert not missing, f"HANDWRITTEN names with no DEPS file: {missing}"


# ── a rule narrower than a package survives regeneration ─────────────────
#
# Five generated files named one module where the generator writes a package
# (core/affect's `+core.self.what_came_before`, core/unity's
# `+core.agency.habits_are_hers`), each with a comment saying why. The
# generator rewrote them to the whole package and dropped the comment, so
# `make deps-check` was red on all five and the only command that turned it
# green deleted the narrowing (22 September 2026).

_WRITTEN = '''include_rules = [
    "+core.affect",

    # What this package reaches for today.
    "+core.runtime",
    # Its ledgers are part of her history.
    "+core.self.what_came_before",

    # Everything else.
    "-core",
]
'''


def test_a_module_rule_is_read_with_the_comment_above_it():
    written = _generator().written_narrowings(_WRITTEN)
    assert written == {"core.self.what_came_before": ("# Its ledgers are part of her history.",)}


def test_the_generators_own_section_comment_is_not_a_rules_comment():
    # A module rule first in its section sits right under the section comment
    # this tool writes itself. Only the rule's own comment goes with it.
    text = _WRITTEN.replace('    "+core.runtime",\n', "")
    assert "reaches for today.\n    # Its ledgers" in text
    written = _generator().written_narrowings(text)
    assert written["core.self.what_came_before"] == ("# Its ledgers are part of her history.",)


def test_a_module_rule_that_covers_every_import_is_written_back():
    generator = _generator()
    allowed = {"core.runtime", "core.self"}
    imported = {"core.runtime.errors", "core.self.what_came_before"}
    narrowed = generator.narrowings("affect", allowed, generator.written_narrowings(_WRITTEN), imported)
    body = generator.render("affect", allowed, narrowed)
    assert '"+core.self.what_came_before",' in body
    assert '"+core.self",' not in body
    assert "    # Its ledgers are part of her history.\n    \"+core.self.what_came_before\"," in body


def test_an_import_outside_the_named_modules_gets_a_module_rule_of_its_own():
    generator = _generator()
    allowed = {"core.self"}
    imported = {"core.self.what_came_before", "core.self.recognition"}
    narrowed = generator.narrowings("affect", allowed, generator.written_narrowings(_WRITTEN), imported)
    assert [prefix for prefix, _ in narrowed["core.self"]] == [
        "core.self.recognition",
        "core.self.what_came_before",
    ]


def test_an_import_of_the_package_itself_needs_the_package_rule():
    generator = _generator()
    allowed = {"core.self"}
    # `from core.self import what_came_before` resolves to core.self in the
    # checker, which only a package rule matches.
    imported = {"core.self", "core.self.what_came_before"}
    narrowed = generator.narrowings("affect", allowed, generator.written_narrowings(_WRITTEN), imported)
    assert "core.self" not in narrowed
    assert '"+core.self",' in generator.render("affect", allowed, narrowed)


def test_a_module_rule_no_import_needs_is_dropped():
    generator = _generator()
    written = dict(generator.written_narrowings(_WRITTEN))
    written["core.self.recognition"] = ("# No longer read.",)
    narrowed = generator.narrowings("affect", {"core.self"}, written, {"core.self.what_came_before"})
    assert [prefix for prefix, _ in narrowed["core.self"]] == ["core.self.what_came_before"]


def test_the_generator_judges_coverage_with_the_checkers_matching():
    """A rule the generator keeps must be one the gate accepts, and no wider."""
    generator = _generator()
    checker = _load_checker()
    body = generator.render(
        "affect",
        {"core.self"},
        generator.narrowings(
            "affect", {"core.self"}, generator.written_narrowings(_WRITTEN), {"core.self.what_came_before"}
        ),
    )
    rules = [
        checker.Rule(kind=line.strip()[1], prefix=line.strip()[2:].rstrip('",'))
        for line in body.splitlines()
        if line.strip().startswith('"+') or line.strip().startswith('"-')
    ]
    assert checker.check_module("core.self.what_came_before", rules) is None
    assert checker.check_module("core.self.recognition", rules) is not None
