"""A change of hers that survives probation becomes a guard test of what it gives now (Master Todo F16).

The outside check puts a change on probation when it answers some input differently and cannot say the difference is
a loss. When probation keeps the change, those answers are now hers, and nothing in her source said so: the next
change could take them away and only the outside check, comparing with the code before, would notice. Each such input
is now written as a test of what the change gives, landed in her source, so every later check of that module runs it.

A refusal gone is never guarded. Settling speaks for valid questions and for her own calls, and neither speaks for an
input that was refused before, which is how rounds five and six hid a deleted guard. The landing itself is tested in
tests/test_her_mend_lands_on_origin.py, against a bare repository standing in for origin.
"""
from __future__ import annotations

import ast
import os
import subprocess
import sys
import textwrap

import pytest

from core.self_modification.what_her_change_must_keep import guardable, guarding_test_source, path_of_its_guard

pytestmark = pytest.mark.unit

_BEFORE = textwrap.dedent('''
    def scale(x):
        if x < 0:
            raise ValueError("no negative sizes")
        return x * 2


    def label(x):
        return "small" if x < 10 else "large"
''')
# The change kept after probation: sizes are tripled, a zero size is refused, and a negative size is now let through.
_AFTER = textwrap.dedent('''
    def scale(x):
        if x == 0:
            raise ArithmeticError("no empty sizes")
        return x * 3


    def label(x):
        return "small" if x < 10 else "large"
''')


def _doubted(args: dict, before: str, after: str, kind: str, **more) -> dict:
    return {"function": "shapes/sizes.py::scale", "input": {"args": args}, "before": before, "after": after,
            "kind": kind, "counted": "doubt", **more}


_SEEN = [
    _doubted({"x": "4"}, "8", "12", "another value"),
    _doubted({"x": "0"}, "0", "raised ArithmeticError by the code: no empty sizes", "now raises"),
    _doubted({"x": "-1"}, "raised ValueError: no negative sizes", "-3", "a refusal gone"),
    _doubted({"x": "5"}, "10", "15", "another value", **{"as": "digests"}),
    {**_doubted({"x": "6"}, "12", "18", "another value"), "counted": "lost"},
    _doubted({"x": "7"}, "14", "21; and it changed its argument", "another value"),
]


def test_only_a_doubted_difference_settling_speaks_for_is_guarded():
    assert [guardable(one) for one in _SEEN] == [True, True, False, False, False, False]


def test_the_guard_asserts_what_the_change_gives_now():
    source = guarding_test_source(_SEEN, goal='Make "sizes" bigger', commit="0123456789abcdef")
    assert "assert scale(x=4) == 12" in source
    assert "with pytest.raises(ArithmeticError):\n        scale(x=0)" in source
    assert "x=-1" not in source, "a refusal gone is not guarded"
    assert '"""What her change 012345678 gives now' in source and "Make 'sizes' bigger" in source
    ast.parse(source)


def test_nothing_guardable_writes_nothing():
    assert guarding_test_source(_SEEN[2:], goal="g", commit="c") == ""


def _the_guard_on(tmp_path, code: str) -> subprocess.CompletedProcess:
    package = tmp_path / "shapes"
    package.mkdir(exist_ok=True)
    (package / "__init__.py").write_text("")
    (package / "sizes.py").write_text(code)
    (tmp_path / "tests").mkdir(exist_ok=True)
    guard = tmp_path / path_of_its_guard("make sizes bigger")
    guard.write_text(guarding_test_source(_SEEN, goal="Make sizes bigger", commit="0123456789"))
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(tmp_path)}
    return subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--rootdir", str(tmp_path),
                           str(guard)], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)


def test_the_guard_passes_on_the_kept_change_and_fails_on_the_code_before_it(tmp_path):
    kept = _the_guard_on(tmp_path, _AFTER)
    assert kept.returncode == 0, kept.stdout[-800:]
    undone = _the_guard_on(tmp_path, _BEFORE)
    assert undone.returncode != 0 and "2 failed" in undone.stdout, undone.stdout[-800:]
