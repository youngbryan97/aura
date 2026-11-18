"""Repair from ingredients: an expression her own file writes elsewhere, put in place of a suspect one.

Pure: each source is written here, and nothing is run. The first case is a defect from her history, f5da1f5da653,
reduced to the lines that decide it.
"""
from __future__ import annotations

import ast

from core.self_modification.ingredient_repair import ingredient_changes


def _at(source: str, marker: str) -> list[tuple[str, str]]:
    """(context, changed line) for every ingredient at the line holding ``marker``, in the order offered."""
    line = next(n for n, text in enumerate(source.splitlines(), 1) if marker in text)
    return [(context, changed.splitlines()[line - 1].strip())
            for _, context, changed in ingredient_changes([(("m.py", line), 1.0)], {"m.py": source})]


PRESS = '''import time


async def press(look, capture_s, neutral_s, after_capture):
    neutral_began = time.monotonic()
    before = await captured(look, capture_s, not_before=neutral_began + neutral_s)
    if after_capture is not None:
        before = await captured(look, capture_s, not_before=before.upper)
    if time.monotonic() - before.upper > .75:
        raise LookupError("before_outside_neutral")
    return before
'''


def test_her_fix_of_f5da1f5da653_is_the_first_ingredient_offered():
    """Her fix was ``not_before=time.monotonic()``: written beside ``before.upper`` two lines further on."""
    offered = _at(PRESS, "not_before=before.upper")
    assert offered[0] == ("same_place", "before = await captured(look, capture_s, not_before=time.monotonic())")
    assert offered[1] == ("same_place",
                          "before = await captured(look, capture_s, not_before=neutral_began + neutral_s)")


def test_a_name_another_function_binds_is_never_offered():
    source = '''def first(rows):
    total = sum(rows)
    return total


def second(rows, limit):
    kept = [r for r in rows if r < limit]
    return len(kept)
'''
    offered = [line for _, line in _at(source, "return len(kept)")]
    assert offered and not any("total" in line for line in offered)


def test_nothing_a_refusal_says_is_changed():
    offered = [line for _, line in _at(PRESS, "time.monotonic() - before.upper > .75")]
    assert offered == []
    raising = PRESS.replace('raise LookupError("before_outside_neutral")',
                            'raise LookupError(before.upper - neutral_began)')
    assert _at(raising, "raise LookupError") == []


def test_an_awaited_ingredient_stays_out_of_a_function_that_cannot_await():
    source = '''async def fetch(client):
    return await client.read()


def size(client):
    return len(client.cached)
'''
    assert not any("await" in line for _, line in _at(source, "return len(client.cached)"))


def test_a_sum_put_into_a_product_keeps_its_brackets():
    source = '''def scaled(a, b, width):
    span = a + b
    return width * 2
'''
    offered = [line for _, line in _at(source, "return width * 2")]
    assert "return (a + b) * 2" in offered
    for line in offered:
        ast.parse(line.replace("return ", "", 1))


def test_a_number_takes_the_place_only_of_a_number():
    source = '''def settle(frames, seconds):
    if frames > 3:
        return seconds * 1.5
    return seconds * 0.25
'''
    offered = [line for _, line in _at(source, "return seconds * 0.25")]
    assert "return seconds * 1.5" in offered
    assert "return seconds * frames" not in offered
