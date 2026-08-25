"""The Python engine can do engineering arithmetic, and still cannot escape.

The sandbox has no ``__import__`` by design, so the only way to give it
dimensional arithmetic used to be to open the import door for everything.
Instead the primitives are imported by the runner, while real builtins still
exist, and handed to the sandboxed code as ready-made names. They are pure
computation with no file, network or process reach, so this adds arithmetic
that carries its units and no capability at all.

The escape tests are the point of the file. If either of them ever passes,
the injection widened the boundary and has to come out.
"""

from __future__ import annotations

import asyncio

import pytest

from core.skills.code_repl import CodeREPLSkill


def _run(code: str) -> dict:
    return asyncio.run(CodeREPLSkill().execute({"code": code}, {}))


@pytest.mark.slow
def test_dimensional_arithmetic_works_without_an_import():
    out = _run(
        'hull = Tube.of("300 mm", "12 mm", "700 mm")\n'
        'print(hull.mass(material("titanium").density).text())\n'
        'print((Q(9.05, "MPa") * Q(0.144, "m") / Q(0.012, "m")).as_("MPa").text())\n'
    )
    assert out.get("ok"), out.get("error")
    assert "33.7 kg" in out["stdout"]
    assert "109 MPa" in out["stdout"]


@pytest.mark.slow
def test_a_dimension_error_is_raised_inside_the_sandbox():
    out = _run(
        "try:\n"
        '    Q(1, "m") + Q(1, "s")\n'
        "except DimensionError as exc:\n"
        '    print("caught", exc)\n'
    )
    assert out.get("ok"), out.get("error")
    assert "caught" in out["stdout"]
    assert "length and time" in out["stdout"]


@pytest.mark.slow
def test_the_sandbox_still_cannot_open_a_file():
    out = _run('print(open("/etc/passwd").read())')
    assert not out.get("ok")
    assert "open" in str(out.get("error"))


@pytest.mark.slow
def test_the_sandbox_still_cannot_import_anything():
    out = _run("import os\nprint(os.listdir('/'))")
    assert not out.get("ok")
    assert "__import__" in str(out.get("error"))


@pytest.mark.slow
def test_the_sandbox_cannot_reach_the_engineering_module_by_import():
    """The names are handed over; the door stays shut."""
    out = _run("from core.engineering.units import Q\nprint(Q)")
    assert not out.get("ok")
    assert "__import__" in str(out.get("error"))


def test_the_injected_names_are_pure_computation():
    """Nothing handed to the sandbox may read, write or spawn."""
    import inspect

    from core.sandbox import runner

    block = runner.RUNNER_PY.split("engineering_namespace = {", 1)[1].split("}", 1)[0]
    for forbidden in ("write_bundle", "design_from", "export", "open", "Path", "subprocess"):
        assert forbidden not in block, f"{forbidden} was handed to untrusted code"


def test_the_untrusted_child_is_given_a_minimal_environment():
    """It was inheriting the whole parent environment, credentials included."""
    from core.sandbox.runner import _untrusted_environment

    env = _untrusted_environment()
    assert env.get("PATH")
    assert env.get("PYTHONNOUSERSITE") == "1"
    for leaked in ("AWS_SECRET_ACCESS_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
                   "AURA_LOG_DIR", "PYTHONPATH"):
        assert leaked not in env


def test_the_spawn_passes_that_environment():
    import inspect

    from core.sandbox import runner

    assert "env=_untrusted_environment()" in inspect.getsource(runner._communicate_process)
