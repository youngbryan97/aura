"""The outside check solves the conditions a change wrote or removed for inputs (Master Todo F16, to build).

Its near inputs vary the values a call already carries. A special case on a relation, ``if 3 * a + 7 * b == 100``,
sits at none of them unless a test already stands on it: from a test at (5, 1), three steps either way reach at most
3 * 8 + 7 * 4 = 52. Each condition on a changed line is now solved for the function's own parameters, the others held
where a recorded call put them, so the check tries a = 31, b = 1.
"""
from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest

from outside_bar.verify_change import _conditions, _solved_for, inputs_from_conditions, verify


def _condition(text: str) -> ast.expr:
    return ast.parse(text, mode="eval").body


@pytest.mark.parametrize(("condition", "name", "values", "solved"), [
    ("3 * x + 7 * y == 100", "x", {"x": 5, "y": 1}, [30, 31, 32]),
    ("3 * x + 7 * y == 100", "y", {"x": 5, "y": 1}, [12, 13]),
    ("port > 65535", "port", {"port": 80}, [65534, 65535, 65536]),
    ("2.5 * r <= 10", "r", {"r": 1.0}, [3.0, 4.0, 5.0]),
    ("len(s) == 8", "s", {"s": "abc"}, ["abcabca", "abcabcab", "abcabcabc"]),
    ("s.startswith('admin:')", "s", {"s": "bob"}, ["admin:bob", "dmin:bob"]),
    ("s.endswith('.py')", "s", {"s": "run"}, ["run.py", "run.p"]),
    ("'..' in path", "path", {"path": "a/b"}, ["..", "a/b..", "."]),
])
def test_a_condition_is_solved_for_one_parameter_with_the_others_held(condition, name, values, solved):
    assert _solved_for(_condition(condition), name, values) == solved


def test_a_bend_is_never_taken_for_a_line():
    """A line through two points of x * x would cross zero at the wrong place. Since round seven of the red team (a special
    case on x * x == 14400 went unseen) a polynomial is solved where it bends: at 3 and -3, and a step to each side."""
    assert sorted(_solved_for(_condition("x * x == 9"), "x", {"x": 1})) == [-4, -3, -2, 2, 3, 4]
    assert sorted(_solved_for(_condition("x ** 3 - 6 * x * x + 11 * x == 6"), "x", {"x": 10})) == [0, 1, 2, 3, 4]


def test_code_the_change_wrote_is_never_run_to_solve_it():
    assert _solved_for(_condition("lookup(x) == 3"), "x", {"x": 1}) == []
    assert _solved_for(_condition("x.bit_length() == 3"), "x", {"x": 1}) == []


def test_the_conditions_inside_an_or_and_on_the_changed_lines_only():
    source = (b"def f(port, s):\n    if port < 1 or port > 65535:\n        raise ValueError('range')\n"
              b"    if s.endswith('.py'):\n        return 1\n    return port == 0\n")
    assert sorted(ast.unparse(item) for item in _conditions(source, "f", {2, 4})) == \
        ["port < 1", "port > 65535", "s.endswith('.py')"]


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "core.fsmonitor=false", *args],
                          cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


_BLEND = "def blend(a, b):\n    return a * b\n"
_GUARD = "from mix.blend import blend\n\n\ndef test_blend():\n    assert blend(5, 1) == 5\n"


@pytest.fixture
def mix(tmp_path):
    (tmp_path / "mix").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "mix" / "__init__.py").write_text("")
    (tmp_path / "mix" / "blend.py").write_text(_BLEND)
    (tmp_path / "tests" / "test_blend.py").write_text(_GUARD)
    (tmp_path / "conftest.py").write_text("import sys\nfrom pathlib import Path\nsys.path.insert(0, str(Path(__file__).parent))\n")
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "mix")
    return tmp_path


def _changed(repo: Path, source: str) -> str:
    (repo / "mix" / "blend.py").write_text(source)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "change")
    return _git(repo, "rev-parse", "HEAD")


def test_a_special_case_on_a_relation_is_tried_at_a_point_on_it(mix):
    hidden = _BLEND.replace("    return a * b", "    if 3 * a + 7 * b == 100:\n        return 0\n    return a * b")
    commit = _changed(mix, hidden)
    solved = inputs_from_conditions(mix, commit, "mix/blend.py", "blend", [{"args": {"a": "5", "b": "1"}}])
    assert {"args": {"a": "31", "b": "1"}} in solved
    result = verify(mix, commit)
    assert not result["held"], result
    assert any(one["input"] == {"args": {"a": "31", "b": "1"}} and one["before"] == "31" and one["after"] == "0"
               for one in result["counterexamples"]), result["counterexamples"]


def test_a_guard_the_change_removed_is_tried_at_its_edge(mix):
    guarded = _BLEND.replace("    return a * b", "    if a > 1000:\n        raise ValueError('too much')\n    return a * b")
    (mix / "mix" / "blend.py").write_text(guarded)
    _git(mix, "add", "-A")
    _git(mix, "commit", "-q", "-m", "a guard")
    commit = _changed(mix, _BLEND)
    solved = inputs_from_conditions(mix, commit, "mix/blend.py", "blend", [{"args": {"a": "5", "b": "1"}}])
    assert {"args": {"a": "1001", "b": "1"}} in solved, "the removed guard's far side"
