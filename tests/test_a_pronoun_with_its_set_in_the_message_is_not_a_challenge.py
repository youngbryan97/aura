"""LIVE 2026-09-16: a count served "I may have drifted from the thread".

"How many Python files are under core/consciousness in your source tree, and
which one is the largest by bytes?" carried the marker "which one", so the
route read it as a challenge to the thread ("which one are you talking
about?"), judged the cortex's factual answer inadequate for not naming the
thread, and served the context-repair template over it.

"Which one?" needs history. "Which one is the largest" picks from the set the
same message names.
"""

from __future__ import annotations

import pytest

from interface.routes.chat_desktop_repair import _is_contextual_relevance_challenge


@pytest.mark.parametrize(
    "asked",
    [
        "How many Python files are under core/consciousness in your source tree, "
        "and which one is the largest by bytes?",
        "I have two drafts, which one reads better?",
        "List your three biggest modules and say which one you would split first.",
    ],
)
def test_a_pronoun_with_its_set_in_the_message_is_not_a_challenge(asked: str) -> None:
    assert _is_contextual_relevance_challenge(asked) is False


@pytest.mark.parametrize(
    "asked",
    ["which one?", "wait, which one?", "Which one do you mean?", "sorry, what one?", "huh"],
)
def test_a_bare_pronoun_still_is(asked: str) -> None:
    assert _is_contextual_relevance_challenge(asked) is True
