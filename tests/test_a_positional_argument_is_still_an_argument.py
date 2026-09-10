"""A parameter reached positionally is not one every caller passes empty.

`switched_off_arguments` reads keyword arguments and reports a parameter whose
callers all pass an empty literal. It read only keywords, so a call passing the
parameter positionally was invisible — and the tool reported a mechanism as
unreachable while a caller was using it.

Found by the tool's own gate, 2026-09-08:
`core.brain.llm.latent_cortex.commitment_ablations:_verdict:checks` is called
with `checks={}` on the three inconclusive paths and with a populated `checks`
positionally on the two that decide a verdict.

A call whose arity the source does not state — `f(*args)`, `f(**kwargs)` —
reaches anything, so nothing behind one can be called switched off either.
"""

from __future__ import annotations

import ast

from core.verify.can_this_ever_fire import (
    _REACHES_EVERYTHING,
    _keyword_arguments_by_callee,
)


def _trees(source: str) -> dict[str, ast.Module]:
    return {"m": ast.parse(source)}


def test_positional_reach_is_counted() -> None:
    _passed, reach = _keyword_arguments_by_callee(
        _trees("def f(a, b, c):\n    return a\n\nf(1, 2, 3)\n")
    )
    assert reach["f"] == 3


def test_a_star_argument_reaches_everything() -> None:
    _passed, reach = _keyword_arguments_by_callee(_trees("f(*args)\n"))
    assert reach["f"] == _REACHES_EVERYTHING


def test_a_double_star_argument_reaches_everything() -> None:
    _passed, reach = _keyword_arguments_by_callee(_trees("f(**kwargs)\n"))
    assert reach["f"] == _REACHES_EVERYTHING


def test_keywords_are_still_collected() -> None:
    passed, _reach = _keyword_arguments_by_callee(_trees("f(checks={})\n"))
    assert "checks" in passed["f"]


def test_the_live_false_positive_is_gone() -> None:
    """The shape that produced it, as a whole file."""

    from core.verify.can_this_ever_fire import switched_off_arguments

    source = '''
def verdict(status, reason, table, checks):
    return (status, reason, table, checks)

def decide(table):
    if not table:
        return verdict("INCONCLUSIVE", "nothing ran", table, checks={})
    checks = {"beats_control": True}
    return verdict("SUPPORTED", "beat it", table, checks)
'''
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "core").mkdir()
        (root / "core" / "sample.py").write_text(source, encoding="utf-8")
        found = switched_off_arguments(str(root))
    assert not [item for item in found if item.parameter == "checks"], found
