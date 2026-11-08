"""A turn of hers answered no request, so it is not queued for independent programs to settle.

LIVE 7 November 2026, dress rehearsal: a remark's reading of a video ("playing, at 0:02 of 11:13. 0:01 seen: A man in a
yellow shirt ...") was queued as a problem she answered, taken for mathematics, and checked by three samples on her
model while the person's next request waited 35 seconds for it.
"""

import pytest

from core.brain.cognitive_engine_quick_reply import _AnswersTheDesktopDirectly
from core.conversation.session_scope import her_own_turn_scope
from core.learning import examples_she_meets


@pytest.fixture
def queued(monkeypatch):
    calls = []

    async def note(problem):
        calls.append(problem)
        return True

    monkeypatch.setattr(examples_she_meets, "note_to_check_async", note)
    return calls


@pytest.mark.asyncio
async def test_a_reading_of_hers_is_not_queued(queued):
    with her_own_turn_scope(True):
        assert not await _AnswersTheDesktopDirectly._direct_desktop_quick_reply_queue_for_programs(
            None, "playing, at 0:02 of 11:13. 0:01 seen: A man in a yellow shirt is outdoors, 4 of 9 frames")
    assert queued == []


@pytest.mark.asyncio
async def test_a_persons_request_still_is(queued):
    asked = "What are the prime factors of 21000063?"
    with her_own_turn_scope(False):
        await _AnswersTheDesktopDirectly._direct_desktop_quick_reply_queue_for_programs(None, asked)
    from core.brain.executable_reasoning import should_use_executable_reasoning
    from core.brain.reasoning_amplifier_v2 import is_amplifiable

    task_type = is_amplifiable(asked)
    expected = [asked] if task_type and should_use_executable_reasoning(asked, task_type=task_type) else []
    assert queued == expected
