"""Mends of a program she is making that she writes herself, without her model: tried first, because each is exact.

Her repair of programs was engineered before this (core/self_modification): the shapes of code that are almost never meant,
read from the syntax tree of JavaScript or Python with the edits that mend each; the fix templates and the expressions her
own file writes elsewhere, made at the lines the failing runs went through and the passing ones did not. What they lack is
something that says whether an edit helped, and the maker has that: it runs the program in a second or so. So each is tried
before her model is asked, which takes a minute or two and can be wrong in any way.

A change is only proposed here. What keeps one is that less fails with it (core/making/maker.py).
"""
from __future__ import annotations

import ast
import json
import logging
import re
import sys
from collections.abc import Iterator, Mapping, Sequence

logger = logging.getLogger("Making.HerOwnMends")

__all__ = ["MOST", "WHAT_A_RUN_WENT_THROUGH", "known_by_reading", "missing_imports", "of_its_lines", "of_its_shapes", "ran"]

#: The most changes of each sort that are tried for one state of a program.
MOST = 40

#: Runs a Python program as it is run, and writes the lines of it that were gone through beside it as it leaves.
WHAT_A_RUN_WENT_THROUGH = '''import atexit, json, os, runpy, sys
_file, _out = sys.argv[1], sys.argv[2]
sys.argv = [_file, *sys.argv[3:]]
_seen = set()
def _line(frame, event, _arg):
    if event == "line":
        _seen.add(frame.f_lineno)
    return _line
def _call(frame, _event, _arg):
    if frame.f_code.co_filename != _file:
        return None
    _seen.add(frame.f_lineno)
    return _line
def _told():
    sys.settrace(None)
    with open(_out, "w") as told:
        json.dump(sorted(_seen), told)
atexit.register(_told)
sys.path[0] = os.path.dirname(_file)
sys.settrace(_call)
runpy.run_path(_file, run_name="__main__")
'''

_UNDEFINED = re.compile(r"(?:name \\?'|Undefined name `)([A-Za-z_][A-Za-z0-9_]*)(?:\\?' is not defined|`)")


def of_its_shapes(code: str, suffix: str) -> Iterator[tuple[str, str, str]]:
    """(what shape, what was changed, the code with it changed) for each shape of code almost never meant that it has."""
    from core.self_modification.code_that_looks_wrong import applied, what_looks_wrong
    from core.self_modification.repairing_by_behaviour import _candidates

    try:
        found = what_looks_wrong(code, suffix)
    except (ImportError, ValueError, RuntimeError) as exc:
        logger.info("Its shapes are not read: %s", exc)
        return
    seen: set[str] = set()
    for suspicion, edits in _candidates(found):
        changed = applied(code, edits)
        if changed != code and changed not in seen:
            seen.add(changed)
            yield suspicion.pattern, "; ".join(edit.says(code) for edit in edits)[:240], changed


def missing_imports(code: str, failure: str) -> Iterator[tuple[str, str, str]]:
    """The import of Python's own that a failure says is missing, where its name is a module's."""
    for name in dict.fromkeys(_UNDEFINED.findall(failure)):
        if name in sys.stdlib_module_names:
            first = re.match(r'(?:#![^\n]*\n)?(?:\s*"""[\s\S]*?"""\s*\n)?(?:from __future__[^\n]*\n)*', code)
            at = first.end() if first else 0
            yield "missing_import", f"added `import {name}`", f"{code[:at]}import {name}\n{code[at:]}"


def known_by_reading(code: str) -> str:
    """What is wrong with some Python that can be known without running it (core/sandbox/static_check.py); empty for none."""
    from core.sandbox.static_check import what_will_not_work

    return "; ".join(f"line {finding.line}: {finding.said}" for finding in what_will_not_work(code))[:600]


def ran(failing: Mapping[str, set[int]], passing: Mapping[str, set[int]], *, after: int = 0) -> list[tuple[int, float]]:
    """The lines the failing runs went through past line ``after``, the most suspicious first."""
    from core.self_modification.fix_templates import ochiai

    ranked = ochiai({name: {("it", line) for line in lines} for name, lines in failing.items()},
                    {name: {("it", line) for line in lines} for name, lines in passing.items()})
    return [(line, score) for (_path, line), score in ranked if line > after and score > 0]


def of_its_lines(code: str, ranked: Sequence[tuple[int, float]], error: str = "") -> Iterator[tuple[str, str, str]]:
    """(what template, where, the code with it changed) for the fix templates and the ingredients at the suspicious lines
    of some Python, the two turn about."""
    from core.self_modification.fix_templates import template_changes
    from core.self_modification.ingredient_repair import ingredient_changes

    at = [(("it.py", line), score) for line, score in ranked]
    sources = {"it.py": code}

    def whole(changes: Iterator[tuple[str, str, str]], sort: str) -> Iterator[tuple[str, str, str]]:
        try:
            for _path, name, changed in changes:
                if _parses(changed) and changed != code:
                    yield f"{sort}:{name}", _what_changed(code, changed), changed
        except (ValueError, SyntaxError, RuntimeError, KeyError, IndexError) as exc:
            logger.info("No more %s changes: %s", sort, exc)

    ways = [whole(template_changes(at, sources, error), "template"), whole(ingredient_changes(at, sources), "ingredient")]
    while ways:
        for way in list(ways):
            got = next(way, None)
            if got is None:
                ways.remove(way)
            else:
                yield got


def _parses(code: str) -> bool:
    try:
        ast.parse(code)
    except SyntaxError:
        return False
    return True


def _what_changed(before: str, after: str) -> str:
    a, b = before.splitlines(), after.splitlines()
    for one, two in zip(a, b, strict=False):
        if one != two:
            return f"changed `{one.strip()[:100]}` to `{two.strip()[:100]}`"
    return "changed it"


def lines_in(text: str) -> set[int]:
    """The lines a run told of."""
    try:
        return {int(line) for line in json.loads(text)}
    except (ValueError, TypeError):
        return set()
