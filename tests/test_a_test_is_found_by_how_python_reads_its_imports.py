"""A test is found as a guard of a change by how Python reads its imports, not by the text of one import line (Master
Todo F16, 26 November 2026).

Her mend of ``core.runtime.subprocess_gateway._coerce_argv`` took away the refusal of an empty argument that
tests/test_subprocess_gateway_contract.py asserts as the gateway's contract, and the outside check let it land on
origin. That test imports ``from core.runtime import shutdown_coordinator, subprocess_gateway``. The guards were found
by fixed texts, the module's dotted name and ``import subprocess_gateway``, and that line holds neither, so the one test
written to guard the function was never asked.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from outside_bar.verify_change import imports_it
from outside_bar.verify_change import tests_naming as guards_of

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("line", [
    "from core.runtime import shutdown_coordinator, subprocess_gateway",
    "from core.runtime import (\n    shutdown_coordinator,\n    subprocess_gateway,\n)",
    "from core.runtime import subprocess_gateway as gateway",
    "import core.runtime.subprocess_gateway",
    "from core.runtime.subprocess_gateway import _coerce_argv",
])
def test_every_way_of_importing_a_module_is_read_as_importing_it(line):
    assert imports_it(line.encode(), "core.runtime.subprocess_gateway")


@pytest.mark.parametrize("line", [
    "from core.runtime import shutdown_coordinator",
    "from core.other import subprocess_gateway",
    "# from core.runtime import subprocess_gateway",
    "text = 'from core.runtime import subprocess_gateway'",
])
def test_a_line_that_only_looks_like_it_is_not(line):
    assert not imports_it(line.encode(), "core.runtime.subprocess_gateway")


def test_a_name_imported_from_a_module_over_several_lines_is_read():
    source = b"from app.field import (\n    other,\n    json_safe,\n)\n"
    assert imports_it(source, "app.field", "json_safe") and not imports_it(source, "app.field", "copy")


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "core.fsmonitor=false", *args],
                   cwd=repo, check=True, capture_output=True)


def test_the_contract_test_of_a_module_imported_beside_another_guards_it(tmp_path):
    files = {
        "core/__init__.py": "",
        "core/runtime/__init__.py": "",
        "core/runtime/subprocess_gateway.py": "def _coerce_argv(argv):\n    return list(argv)\n",
        "core/runtime/shutdown_coordinator.py": "",
        "tests/test_subprocess_gateway_contract.py":
            "from core.runtime import shutdown_coordinator, subprocess_gateway\n\n\n"
            "def test_it():\n    assert subprocess_gateway._coerce_argv(['a'])\n",
        "tests/test_unrelated.py": "def test_it():\n    subprocess_gateway = 1\n    assert subprocess_gateway\n",
    }
    for name, text in files.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(text)
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "add", "-A")
    found = guards_of(tmp_path, ["core/runtime/subprocess_gateway.py"])
    assert found == ["tests/test_subprocess_gateway_contract.py"], found
