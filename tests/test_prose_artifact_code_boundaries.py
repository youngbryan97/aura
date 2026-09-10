import pytest

from core.conversation.response_reliability import _has_punctuation_join_artifact


@pytest.mark.parametrize("text", [
    "Use `lock.acquire()` before the update.",
    "Call ``client.compare_exchange()`` to claim it.",
    "The implementation:\n```python\nawait lock.acquire()\n```",
])
def test_member_access_in_code_is_not_broken_prose(text):
    assert not _has_punctuation_join_artifact(text)


def test_code_does_not_hide_adjacent_broken_prose():
    assert _has_punctuation_join_artifact("Use `lock.acquire()`. Evidence.utschein")
