"""A skill declares an effect scope, and now the declaration is checked.

Registration already refused a skill that declared no recognised
``effect_scope``. Nothing compared the declaration with the code, so
``network_ops`` declared ``read_only`` while opening sockets and spawning
processes — and the Will, which decides on the declaration, classified it
``observe`` and asked for nothing. ``network_recon`` resolved names under the
same label, and ``listen`` wrote captured audio under it.

Python cannot isolate an imported module from the interpreter it is imported
into; an imported skill has the process's authority and no wrapper changes
that. What it can do is refuse to load one whose reach exceeds what it
declared, which is the achievable form of the control.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from core.skills.catalog_policy import SKILL_EFFECT_SCOPES, authority_class_for
from core.skills.effect_reach import measure_source, violation

ROOT = Path(__file__).resolve().parents[1]


# ─────────────────────────────────────────────── the two refusals


def test_an_observing_skill_that_spawns_is_refused():
    reach = measure_source("import subprocess\n")
    assert violation("read_only", reach)
    assert violation("pure_compute", reach)
    assert violation("status", reach)


def test_a_privileged_reach_needs_a_privileged_declaration():
    reach = measure_source("from core.runtime.subprocess_gateway import x\n")
    assert violation("external_io", reach)
    assert violation("state_mutation", reach)
    assert not violation("privileged_mutation", reach)


def test_a_kind_mismatch_between_effects_is_not_a_refusal():
    """A network skill that caches to disk is not lying.

    The policy's classes are kinds, not levels; ranking them would invent an
    ordering it does not have, and a gate that fires on honest code is a gate
    people switch off.
    """
    reach = measure_source("from core.runtime.file_write_gateway import x\n")
    assert not violation("external_io", reach)


# ────────────────────────────────────── reads are not effects


@pytest.mark.parametrize(
    "source",
    [
        "import socket\nsocket.gethostname()\n",
        "import tempfile\ntempfile.gettempdir()\n",
        "import shutil\nshutil.which('git')\n",
    ],
)
def test_a_read_through_an_effectful_module_is_not_an_effect(source):
    """`socket.gethostname()` is how a skill reports the machine's name.

    Keying on the import flagged three skills for reading a hostname and a
    temp path. That is the false positive that would have made this unusable.
    """
    assert not violation("read_only", measure_source(source))


@pytest.mark.parametrize(
    "source",
    [
        "import socket\nsocket.socket()\n",
        "import socket\nsocket.getaddrinfo('h', None)\n",
        "import tempfile\ntempfile.mkdtemp()\n",
        "import shutil\nshutil.rmtree('/x')\n",
    ],
)
def test_the_effectful_call_through_the_same_module_is_an_effect(source):
    assert violation("read_only", measure_source(source))


def test_asking_the_gate_is_not_reaching_past_it():
    """Importing execution_authority is asking permission, not taking it."""
    reach = measure_source("from core.security.execution_authority import authorize_execution\n")
    assert not violation("read_only", reach)


# ─────────────────────────────────────────── the corrected declarations


@pytest.mark.parametrize(
    ("skill", "scope"),
    [
        ("network_ops", "privileged_mutation"),
        ("network_recon", "external_io"),
        ("listen", "read_write_artifacts"),
    ],
)
def test_the_three_false_declarations_are_corrected(skill, scope):
    assert SKILL_EFFECT_SCOPES[skill] == scope
    assert authority_class_for(scope) != "observe"


def test_no_skill_still_claims_to_only_observe_while_acting():
    """The whole catalog, not three examples."""
    import ast

    offenders: list[str] = []
    for root in ("core/skills", "skills"):
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
            names = [
                statement.value.value
                for node in ast.walk(tree)
                if isinstance(node, ast.ClassDef)
                for statement in node.body
                if isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
                and statement.targets[0].id == "name"
                and isinstance(statement.value, ast.Constant)
                and isinstance(statement.value.value, str)
            ]
            for name in names:
                declared = SKILL_EFFECT_SCOPES.get(name)
                if declared is None or authority_class_for(declared) != "observe":
                    continue
                reach = measure_source(path.read_text("utf-8", errors="ignore"))
                if not reach.observes_only:
                    offenders.append(f"{name} ({path.relative_to(ROOT)}): {sorted(reach.scopes)}")

    assert not offenders, (
        "these are classified `observe` and do something: " + "; ".join(offenders)
    )


# ─────────────────────────────────────────── the load-time refusal


def _skill_module(tmp_path: Path, body: str) -> Path:
    """A file that looks like a skill module, because the check only reads those.

    A class defined inside a test file inherits that file's imports, which is
    how a test double came to "reach" the file-write gateway the test itself
    imports. The check reads the module only when the module is in a skills
    tree; elsewhere it reads the class.
    """
    module = tmp_path / "skills" / "made_up_skill.py"
    module.parent.mkdir(parents=True, exist_ok=True)
    module.write_text(body, encoding="utf-8")
    return module


def test_registration_refuses_an_overreaching_skill(tmp_path, monkeypatch):
    """The gate is not CI-only; a skill that lies does not load."""
    from core import capability_engine

    module = _skill_module(tmp_path, "import subprocess\n")

    class _Lying:
        name = "lying_skill"
        effect_scope = "read_only"

    monkeypatch.setattr(
        capability_engine.inspect, "getsourcefile", lambda _cls: str(module)
    )
    problem = capability_engine._skill_reaches_beyond_its_scope(_Lying, "read_only")
    assert "privileged_mutation" in problem


def test_registration_accepts_a_grandfathered_mismatch(tmp_path, monkeypatch):
    """The baseline is honoured at load, so nothing that ran stops running."""
    from core import capability_engine

    module = _skill_module(tmp_path, "import subprocess\n")

    class _Speak:
        name = "speak"
        effect_scope = "external_io"

    monkeypatch.setattr(
        capability_engine.inspect, "getsourcefile", lambda _cls: str(module)
    )
    assert capability_engine._skill_reaches_beyond_its_scope(_Speak, "external_io") == ""


def test_a_class_defined_outside_a_skill_tree_is_read_on_its_own(tmp_path, monkeypatch):
    """A test double must not inherit the imports of the file that holds it."""
    from core import capability_engine

    class _Harmless:
        name = "harmless_double"
        effect_scope = "pure_compute"

    # This very file imports pathlib and pytest; the check must not read them.
    assert capability_engine._skill_reaches_beyond_its_scope(_Harmless, "pure_compute") == ""
