"""Fix templates are made exactly at a line, each parses, and the lines failing tests run are tried first."""
from __future__ import annotations

from core.self_modification.fix_templates import changes_at, ochiai, template_changes

SOURCE = '''def clamp(x, lo, hi):
    if x < lo:
        return hi
    return x[1:n] if f(a, b) and x != 2 else x
'''


def _changed_line(changed: str, line: int) -> str:
    return changed.splitlines()[line - 1]


def test_each_template_changes_only_what_it_names():
    at_2 = {(name, _changed_line(c, 2)) for name, c in changes_at(SOURCE, 2)}
    assert ("change_comparison", "    if x <= lo:") in at_2
    assert ("negate_condition", "    if not (x < lo):") in at_2
    at_3 = {(name, _changed_line(c, 3)) for name, c in changes_at(SOURCE, 3)}
    assert ("replace_name", "        return lo") in at_3
    at_4 = {(name, _changed_line(c, 4)) for name, c in changes_at(SOURCE, 4)}
    assert ("swap_arguments", "    return x[1:n] if f(b, a) and x != 2 else x") in at_4
    assert ("change_boolean_operator", "    return x[1:n] if f(a, b) or x != 2 else x") in at_4
    assert ("shift_by_one", "    return x[1:n - 1] if f(a, b) and x != 2 else x") in at_4
    for name, changed in changes_at(SOURCE, 4):
        others = [i for i, (a, b) in enumerate(zip(SOURCE.splitlines(), changed.splitlines(), strict=True)) if a != b]
        assert others == [3], (name, others)


def test_lines_only_failing_tests_run_are_most_suspicious():
    failing = {"t_fail": {("m.py", 2), ("m.py", 3)}}
    passing = {"t_a": {("m.py", 2)}, "t_b": {("m.py", 2)}}
    ranked = ochiai(failing, passing)
    assert ranked[0][0] == ("m.py", 3) and ranked[0][1] == 1.0
    assert ranked[1][0] == ("m.py", 2) and ranked[1][1] < 1.0


def test_dropping_a_statement_is_tried_last_at_a_line():
    names = [name for _path, name, _ in template_changes([(("m.py", 3), 1.0)], {"m.py": SOURCE})]
    assert names[-1] == "drop_statement" and "replace_name" in names
