"""Taking something out of her reply closes the gap it left and nothing else: her lines keep their indentation and her
code stays as she wrote it.

LIVE 12 and 17 November 2026: two code answers of hers were stored with every line flush left, a Python function
among them, after a sentence elsewhere in the reply was taken out by a check that closed the gap by rewriting every
line.
"""
from __future__ import annotations

from core.conversation.response_reliability import _plain_text_maths
from core.utils.text_after_removal import closed_up
from interface.routes.chat_desktop_mode import _strip_empty_citations
from interface.routes.chat_reply_shaping import _without_sentences

_CODE = (
    "```python\n"
    "def group_by_extension(paths):\n"
    "    groups = {}\n"
    "    for p in paths:\n"
    "        groups.setdefault(p.rsplit('.', 1)[-1], []).append(p)\n"
    "    return groups\n"
    "```"
)
_REPLY = f"We talked about this last week. Here it is.\n\n{_CODE}\n\nThat groups them  by extension ."


def test_the_gaps_close_and_indentation_and_code_stay():
    assert closed_up("One  two .\n\n\n\nThree\n    four  five") == "One two.\n\nThree\n    four five"
    assert closed_up(_CODE) == _CODE
    assert closed_up("```\na  =  1 .\n```") == "```\na  =  1 .\n```", "inside a fence nothing is touched"


def test_a_sentence_taken_out_leaves_her_code_indented():
    out = _without_sentences(_REPLY, ["We talked about this last week."])
    assert out.startswith("Here it is."), "no stray space where the sentence was"
    assert _CODE in out
    assert out.endswith("That groups them by extension.")
    middle = _without_sentences("    - first. Second.\n    - third.", ["Second."])
    assert middle == "    - first.\n    - third.", "a list item keeps its indentation"


def test_a_citation_with_no_source_taken_out_leaves_her_code_indented():
    assert _CODE in _strip_empty_citations(f"Here it is [1].\n\n{_CODE}")


def test_maths_made_plain_leaves_her_code_indented():
    assert _CODE in _plain_text_maths(f"So \\(x = 2\\).\n\n{_CODE}")
