"""A change is guarded again by the tests that ran what it changed, and a test it broke is asked alone before it counts
(Master Todo F16, 26 November 2026).

LIVE 23 to 26 November: her change to MorphogenField.diffuse_step, a shallow copy where a deep one held the event loop
104 ms, passed every gate of hers and was refused by the outside check after 3.9 hours. The check had run 5,121
guarding tests three times over, and refused on one, test_installed_runtime_diagnostics_report_clean_at_rest, which
reads the diagnostics of the whole test process and ran nothing of hers.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from outside_bar.verify_change import module_names_it_changed, verify
from outside_bar.verify_change import tests_that_ran_it as ran_it


def _run(values=(), seen=(), ran=("x.py::f",), here=("x.py::f",), finished=True, whole=True) -> dict:
    return {"values": {key: "1" for key in values}, "seen": list(seen), "ran": list(ran), "ran_here": list(here),
            "finished": finished, "whole": whole}


def test_only_the_test_files_that_ran_a_changed_function_are_asked_again(tmp_path):
    guards = ["tests/test_a.py", "tests/test_b.py", "tests/test_c.py"]
    for name in guards:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text("def test_it():\n    pass\n")
    run = _run(values=["tests/test_a.py::test_it|f|1|x"], seen=["tests/test_c.py::test_other|f|x|every call"])
    assert ran_it(run, guards, tmp_path, {}) == ["tests/test_a.py", "tests/test_c.py"]


def test_a_test_that_reads_a_changed_module_name_is_asked_again_too(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_a.py").write_text("from app.field import json_safe\n")
    (tmp_path / "tests" / "test_b.py").write_text("import app.field as field\n\ndef test_it():\n    field.copy\n")
    (tmp_path / "tests" / "test_c.py").write_text("def test_it():\n    pass\n")
    guards = ["tests/test_a.py", "tests/test_b.py", "tests/test_c.py"]
    assert ran_it(_run(ran=(), here=()), guards, tmp_path, {"app/field.py": {"json_safe", "copy"}}) == guards[:2]


@pytest.mark.parametrize("run", [_run(finished=False), _run(whole=False), _run(ran=("x.py::f", "y.py::g")),
                                 _run(values=["tests/test_a.py::test_it|other|1|x"])])
def test_a_run_that_cannot_say_which_tests_ran_the_change_asks_them_all(tmp_path, run):
    assert ran_it(run, ["tests/test_a.py"], tmp_path, {}) is None
    assert ran_it(_run(), ["tests/test_a.py"], tmp_path, {"app/x.py": {"*"}}) is None


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "core.fsmonitor=false", *args],
                          cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


def _committed(repo: Path, old: dict[str, str], new: dict[str, str]) -> str:
    _git(repo, "init", "-q", "-b", "main")
    for files, message in ((old, "before"), (new, "after")):
        for name, text in files.items():
            (repo / name).parent.mkdir(parents=True, exist_ok=True)
            (repo / name).write_text(text)
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def test_the_names_a_commit_binds_or_stops_binding_at_module_level_are_read_off_it(tmp_path):
    commit = _committed(tmp_path, {"app/field.py": "import copy\nfrom json import dumps\nRATE = 2\n\n\ndef f():\n    return 1\n"},
                        {"app/field.py": "from json import dumps\nRATE = 3\n\n\ndef f():\n    return 2\n"})
    assert module_names_it_changed(tmp_path, commit, "app/field.py") == {"copy", "RATE"}


_SHARED = '''RAN = []


def step(x):
    {body}
    return x * 2
'''


def test_a_test_broken_only_beside_others_is_a_doubt_and_a_test_broken_alone_refuses(tmp_path):
    """The first test of the file calls step(); the second reads a list of the module's. On the change, step() leaves
    something in the list, so the second fails after the first and passes alone: what ran beside it decided it."""
    tests = ("from app.shared import RAN, step\n\n\ndef test_step():\n    assert step(2) == 4\n\n\n"
             "def test_nothing_is_left_at_rest():\n    assert RAN == []\n")
    commit = _committed(tmp_path, {"app/__init__.py": "", "app/shared.py": _SHARED.format(body="pass"),
                                   "tests/test_shared.py": tests},
                        {"app/shared.py": _SHARED.format(body="RAN.append(x)")})
    result = verify(tmp_path, commit)
    assert result["tests_held"]["beside_others"] == ["tests/test_shared.py::test_nothing_is_left_at_rest"], result
    assert not result["tests_held"]["broke"]
    assert any("what ran beside it decided it" in doubt for doubt in result["doubts"])


def test_what_a_fixture_broader_than_one_test_made_guards_the_tests_of_every_file(tmp_path):
    """A session fixture in conftest.py builds a table with the changed function once; a test in another file reads the
    table and never calls the function or names its module. The conftest names the module, so every test beside it
    guards; and the call was made in a fixture broader than one test, so none is narrowed away."""
    old = {"app/__init__.py": "", "app/table.py": "def build():\n    return {'rate': 2}\n",
           "tests/conftest.py": "import pytest\n\nfrom app.table import build\n\n\n@pytest.fixture(scope='session')\n"
                                "def table():\n    return build()\n",
           "tests/test_build.py": "def test_build(table):\n    assert table\n",
           "tests/test_rate.py": "def test_rate(table):\n    assert table['rate'] == 2\n",
           "tests/test_other.py": "def test_other():\n    assert 1 + 1 == 2\n"}
    commit = _committed(tmp_path, old, {"app/table.py": "def build():\n    return {'rate': 3}\n"})
    result = verify(tmp_path, commit)
    assert result["verdict"] == "REFUSED" and "tests/test_rate.py::test_rate" in result["tests_held"]["broke"], result
