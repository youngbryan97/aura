"""The change verifier is outside the benchmark's seal, and nothing a result depends on is (Master Todo F16).

A comparison across seals is refused, and every file under outside_bar/ was sealed, the change verifier with them. So
each change to the verifier made the baseline of 19 November, four and a half hours of her time, unusable. The verifier
judges her changes and no result depends on it. These tests hold that: no sealed file imports a file left out, and a
change to the verifier leaves the seal whole while a change to the harness still breaks it.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from outside_bar import run as bar

HERE = Path(bar.__file__).resolve().parent


def _imports_of(path: Path) -> set[str]:
    """The outside_bar modules ``path`` imports, by file name."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.module == "outside_bar":
            names.update(f"{alias.name}.py" for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith("outside_bar."):
            names.add(f"{node.module.split('.')[1]}.py")
        elif isinstance(node, ast.Import):
            names.update(f"{alias.name.split('.')[1]}.py" for alias in node.names if alias.name.startswith("outside_bar."))
    return names


def test_no_sealed_file_imports_a_file_left_out_of_the_seal():
    sealed = {name.split("/", 1)[1] for name in bar.sealed_files() if name.startswith("outside_bar/")}
    for name in sorted(sealed):
        if name.endswith(".py"):
            assert not _imports_of(HERE / name) & set(bar.JUDGES_HER_CHANGES), name


def test_every_python_file_of_the_harness_is_sealed_except_the_verifier():
    sealed = {name.split("/", 1)[1] for name in bar.sealed_files() if name.startswith("outside_bar/")}
    here = {path.name for path in HERE.glob("*.py")}
    assert here - sealed == set(bar.JUDGES_HER_CHANGES)


def test_the_verifier_imports_only_from_itself_or_the_sealed_harness():
    """So leaving it out cannot carry a file a result depends on out of the seal with it."""
    for name in bar.JUDGES_HER_CHANGES:
        assert _imports_of(HERE / name) <= {*bar.JUDGES_HER_CHANGES, *(path.name for path in HERE.glob("*.py"))}


@pytest.mark.parametrize(("changed", "broken"), [("verify_change.py", False), ("value_witness.py", False),
                                                 ("differential.py", False), ("grade.py", True), ("run.py", True)])
def test_a_change_to_the_verifier_leaves_the_seal_whole_and_one_to_the_harness_breaks_it(monkeypatch, changed, broken):
    digest = bar._digest
    monkeypatch.setattr(bar, "_digest", lambda path: "changed" if Path(path).name == changed else digest(path))
    assert (f"outside_bar/{changed}" in bar.what_broke_the_seal()) is broken
