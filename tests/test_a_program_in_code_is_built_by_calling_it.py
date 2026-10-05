"""A program that is code is checked by calling it in a sandbox, part by part, as an application is by using it."""
from __future__ import annotations

import pytest

from core.rebuilding.programs_in_code import CodeCall, WrittenCode, run_calls
from core.rebuilding.rebuilding_a_program import rebuild
from core.rebuilding.what_a_program_does import Genome

pytestmark = pytest.mark.asyncio

_COUNT = 'def count_words(text):\n    """How many words."""\n    return len(text.split())\n'
_WRONG = 'def shout(text):\n    """Upper case."""\n    return text.lower()\n'


class _NoCorpus:
    def by_title(self, title):
        return None

    def search(self, text, limit=5):
        return []


async def _script(prompt, schema, max_tokens):
    if schema is Genome:
        return Genome.model_validate({"name": "Textkit", "what_it_is": "text helpers", "kind": "code", "features": [
            {"name": "Count words", "how": "call count_words", "shows": "the number", "weight": 3},
            {"name": "Shout", "how": "call shout", "shows": "upper case", "weight": 2},
        ]})
    if schema.__name__ == "_Calls":
        return schema.model_validate({"calls": [
            {"feature": "Count words", "function": "count_words", "args": ["a b c"], "expect": "equals", "value": 3},
            {"feature": "Shout", "function": "shout", "args": ["hi"], "expect": "equals", "value": "HI"},
        ]})
    if schema is WrittenCode:
        return WrittenCode(code=_COUNT if '"Count words"' in prompt else _WRONG)
    return None


async def test_calls_are_made_in_the_sandbox():
    runs = await run_calls(_COUNT, [
        CodeCall(feature="c", function="count_words", args=["a b"], value=2),
        CodeCall(feature="c", function="count_words", args=[None], expect="raises", value="AttributeError"),
        CodeCall(feature="c", function="missing", args=[], value=1),
    ])
    assert [r.held for r in runs] == [True, True, False]


async def test_a_function_is_kept_for_what_calling_it_shows(tmp_path):
    done = await rebuild("", _script, tmp_path, corpus=_NoCorpus(), online=False, asked="a library of text helpers")
    outcome = {o.feature.name: o for o in done.built.outcomes}
    assert outcome["Count words"].kept and not outcome["Shout"].kept
    source = done.built.path.read_text()
    assert "def count_words" in source and "def shout" not in source
    assert done.built.path.name == "program.py"
