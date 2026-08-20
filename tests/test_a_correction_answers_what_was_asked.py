"""Correcting a denial nobody asked about makes a bad reply worse.

LIVE, 2026-08-19. A repository-debugging reply degenerated into a loop and
denied three unrelated capabilities on the way. Each denial was faithfully
corrected, so the served answer contained:

    I can read the filesystem — ...
    I can self repair — self_repair are registered and enabled right now ...
    I can execute nethack action — execute_nethack_action are registered and
    enabled right now, so if that failed it was the attempt and not the
    capability.

inside an answer about a failing test. Every correction was individually true
and the result was absurd: a degenerate draft amplified into three status lines
about things nobody asked for.

The registry also lists every skill that could plausibly satisfy a subject, so
"read the filesystem" cited computer_use and desktop_task — neither of which
reads a file — while the reader that was actually offered went unmentioned.
"""

from __future__ import annotations

import pytest

from core.container import ServiceContainer
from core.conversation.session_scope import set_user_question


@pytest.fixture(autouse=True)
def _registry():
    from core.capability_engine import CapabilityEngine

    ServiceContainer.register_instance("capability_engine", CapabilityEngine())
    yield
    set_user_question("")


REPO_TASK = (
    "there's a python project at /tmp/ledger - one test is failing. "
    "read the code and work out why"
)


def test_an_off_topic_denial_is_removed_not_answered():
    from interface.routes.chat import _correct_false_capability_denials

    set_user_question(REPO_TASK)
    served = str(
        _correct_false_capability_denials(
            "I can't read the filesystem. I cannot self repair. "
            "I can't execute nethack action. Please paste the code."
        )
    )
    assert "nethack" not in served
    assert "self repair" not in served
    assert "self_repair" not in served


def test_the_relevant_denial_is_still_corrected():
    from interface.routes.chat import _correct_false_capability_denials

    set_user_question(REPO_TASK)
    served = str(_correct_false_capability_denials("I can't read the filesystem."))
    assert "I can read the filesystem" in served


def test_the_named_skills_are_ones_that_would_actually_do_it():
    """Citing a skill that cannot perform the subject is a new false claim."""
    from interface.routes.chat import _correct_false_capability_denials

    set_user_question(REPO_TASK)
    served = str(_correct_false_capability_denials("I can't read the filesystem."))
    assert "file_operation" in served
    assert "desktop_task" not in served
    assert "computer_use" not in served


def test_with_no_question_in_scope_nothing_is_dropped():
    """The gate must not silently delete corrections when relevance is unknown."""
    from interface.routes.chat import _correct_false_capability_denials

    set_user_question("")
    served = str(_correct_false_capability_denials("I can't read the filesystem."))
    assert "I can read the filesystem" in served
