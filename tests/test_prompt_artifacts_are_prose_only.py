"""Inside a fence, "state:" is code — not the prompt leaking into speech.

The prompt-artifact patterns look for the scaffold appearing in her voice:
lines starting "obj:", "state:", "ctx:", "history:". Those are real leaks and
must stay caught.

They were also applied to fenced code, where the same tokens are ordinary
content — a Python annotation `state: str = "x"`, a YAML key, a JSON field. So
any answer that showed a dataclass, a config example, or an annotated
attribute was flagged. Live 2026-08-18 a code answer was rejected with
reasons=escaped_control_artifact,prompt_artifact and the user was told "I
couldn't get to an answer I'd stand behind on that one".

The repair path was worse than the rejection: it deleted the offending line
out of the middle of the code and returned the rest as if it were whole.
"""

from __future__ import annotations

import pytest

from core.phases.dialogue_policy import (
    _contains_prompt_artifact,
    repair_dialogue_surface,
)

_CODE = "Here:\n\n```python\nclass S:\n    state: str = 'x'\n```\n\nThat holds it."
_YAML = "Config:\n\n```yaml\nstate: active\nhistory: kept\n```\n\nUse that."


@pytest.mark.parametrize("reply", [_CODE, _YAML])
def test_code_is_not_a_prompt_artifact(reply: str) -> None:
    assert not _contains_prompt_artifact(reply)


@pytest.mark.parametrize(
    "reply",
    [
        "obj: reverse the string\nstate: thinking\nHere is the answer.",
        "ctx: the user asked about files\nThe file has three lines.",
        "[ACTIVE GROUNDING EVIDENCE]\nThe file has three lines.",
    ],
)
def test_a_real_leak_is_still_caught(reply: str) -> None:
    assert _contains_prompt_artifact(reply)


def test_a_code_line_is_never_deleted_from_the_middle_of_a_block() -> None:
    """Returning code with a line removed is worse than refusing it."""
    kept = repair_dialogue_surface(_CODE, None)

    assert "state: str = 'x'" in kept
    assert "class S:" in kept


def test_a_leaked_line_is_still_removed_from_prose() -> None:
    leaked = (
        "obj: reverse the string\n"
        "Here is the answer, which is long enough to stand on its own."
    )

    kept = repair_dialogue_surface(leaked, None)

    assert "obj: reverse" not in kept
    assert "Here is the answer" in kept


def test_a_reply_with_no_fence_is_unaffected() -> None:
    assert not _contains_prompt_artifact("The function reverses the string.")
