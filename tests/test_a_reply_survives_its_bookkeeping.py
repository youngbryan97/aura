"""A state commit that did not land does not throw away the answer.

LIVE 2026-08-30: a long question ran the cortex, produced a reply, and the
cycle deadline passed before the state commit. Both records say
severity="warning" and both are followed by code that carries on and extracts
the reply — but cognitive_engine is fail-closed, so the escalation turned a
bookkeeping note into "CRITICAL SERVICE FAILURE" and the turn died with
"I couldn't get to an answer I'd stand behind".
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import core.brain.cognitive_engine as engine


def _records_in(source: str) -> list[ast.Call]:
    tree = ast.parse(source)
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", "") == "record_degradation"
    ]


def _keyword(call: ast.Call, name: str):
    for word in call.keywords:
        if word.arg == name:
            return word.value
    return None


def _mentions(call: ast.Call, text: str) -> bool:
    return text in ast.unparse(call)


def test_the_two_commit_records_do_not_fail_the_turn_closed():
    source = Path(inspect.getfile(engine)).read_text(encoding="utf-8")
    wanted = ("cognitive cycle budget spent before state commit", "foreground_turn_uncommitted")

    found = 0
    for call in _records_in(source):
        if not any(_mentions(call, one) for one in wanted):
            continue
        found += 1
        opted_out = _keyword(call, "enforce_failure_policy")
        assert opted_out is not None, (
            "a record the code deliberately continues past must say so, or the "
            "fail-closed policy discards the reply"
        )
        assert getattr(opted_out, "value", None) is False
    assert found == 2, f"expected both commit records, found {found}"


def test_the_reply_is_still_extracted_after_an_uncommitted_turn():
    """The lines that matter: the record, then the extraction."""
    source = Path(inspect.getfile(engine)).read_text(encoding="utf-8")
    at = source.index("foreground_turn_uncommitted")
    after = source[at : at + 1200]
    assert "_turn_response_message" in after
