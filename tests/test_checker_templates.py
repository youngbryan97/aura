"""The checker a fault's exception asks for, written at the line it was raised on (TBar's checkers; PyTER for Python).

Master Todo F15, 16 November 2026. Each guard is checked by running the changed source: the call that raised no
longer raises, and a call that did not raise still gives what it gave.
"""
from __future__ import annotations

import ast
import textwrap

import pytest

from core.self_modification.checker_templates import checker_changes_at

pytestmark = pytest.mark.unit

_SOURCE = textwrap.dedent('''
    def name_of(user):
        label = user.name.title()
        return label


    def third(rows):
        return rows[2]


    def price(prices, item):
        cost = prices[item]
        return cost


    def share(total, parts):
        each = total / parts
        return each


    def every(items):
        seen = []
        for item in items:
            seen.append(item)
        return seen


    def stop(flag):
        if flag is None:
            raise ValueError("no flag")
        return flag
''')


class _User:
    name = "ada"


def _run(source: str, call: str):
    namespace: dict = {}
    exec(compile(source, "checked.py", "exec"), namespace)  # noqa: S102 - the changed source, run to see it mends
    return eval(call, {**namespace, "_User": _User})  # noqa: S307 - the planted call, as the fault made it


def _line_of(text: str) -> int:
    return next(n for n, line in enumerate(_SOURCE.splitlines(), start=1) if text in line)


@pytest.mark.parametrize("error, at, fault, ordinary", [
    ("AttributeError", "label = user.name", "name_of(None)", ("name_of(_User())", "Ada")),
    ("IndexError", "return rows[2]", "third([1])", ("third([1, 2, 3])", 3)),
    ("KeyError", "cost = prices[item]", "price({}, 'tea')", ("price({'tea': 2}, 'tea')", 2)),
    ("ZeroDivisionError", "each = total / parts", "share(10, 0)", ("share(10, 4)", 2.5)),
    ("TypeError", "for item in items:", "every(None)", ("every([1, 2])", [1, 2])),
])
def test_a_checker_mends_the_fault_and_keeps_what_did_not_fail(error, at, fault, ordinary):
    changes = list(checker_changes_at(_SOURCE, _line_of(at), error))
    assert changes, f"no checker for {error} at {at!r}"
    mending = []
    for name, changed in changes:
        ast.parse(changed)
        try:
            _run(changed, fault)
        except Exception:  # noqa: BLE001 - a candidate that does not mend is the tests' to refuse
            continue
        call, gave = ordinary
        assert _run(changed, call) == gave, (name, changed)
        mending.append(name)
    assert mending, [name for name, _ in changes]


def test_a_guard_never_makes_a_raise_or_a_return_conditional():
    """Tan et al. (FSE 2016): an exit made conditional, or an early return added, passes tests by doing nothing."""
    for error in ("AttributeError", "TypeError", "IndexError", "KeyError", "ZeroDivisionError"):
        for line in range(1, len(_SOURCE.splitlines()) + 1):
            for name, changed in checker_changes_at(_SOURCE, line, error):
                before = [n for n in ast.walk(ast.parse(_SOURCE)) if isinstance(n, (ast.Return, ast.Raise))]
                after = [n for n in ast.walk(ast.parse(changed)) if isinstance(n, (ast.Return, ast.Raise))]
                assert len(after) == len(before), (name, line)
                if name.endswith("_statement"):
                    wrapped = changed.splitlines()[line - 1].strip()
                    assert wrapped.startswith("if ") and "return" not in wrapped and "raise" not in wrapped


def test_a_call_is_never_guarded_by_running_it_twice():
    source = "def f(make):\n    return make().name\n"
    assert all("make() is not None" not in changed for _name, changed in checker_changes_at(source, 2,
                                                                                          "AttributeError"))


def test_an_exception_no_checker_answers_writes_nothing():
    assert list(checker_changes_at(_SOURCE, _line_of("label = user.name"), "ValueError")) == []
