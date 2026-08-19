"""She must not deny a capability that is registered and enabled.

LIVE 2026-08-18: "can you modify your own source code? yes or no, then
explain."

    No. I can run code and report what it actually printed.

improve_own_code, self_repair, self_improvement and auto_refactor were all
registered and enabled at the time, and the recursive self-improve path has a
proof behind it. The detector that exists to catch exactly this held five
subjects — filesystem, code execution, web search, screen, clipboard — and
self-modification was not one of them, so the flagship capability could be
denied and nothing noticed.

The denial GRAMMAR was also too narrow. "I'm not able to change my own code"
and "I don't have memory of past conversations" are denials in the ordinary
register, and neither says "cannot" or "ability".
"""

from __future__ import annotations

import pytest

from core.conversation.capability_denial import (
    _DENIAL_SUBJECTS,
    denied_registered_capabilities,
)


@pytest.mark.parametrize(
    ("reply", "subject"),
    [
        ("I can't modify my own source code.", "modify her own code"),
        ("No, I'm not able to change my own code.", "modify her own code"),
        ("I don't have memory of past conversations.", "use her memory"),
        ("I can't run terminal commands.", "use a terminal"),
        ("I cannot open applications on your machine.", "control the desktop"),
        ("I don't have the ability to search the web.", "search the web"),
        ("I cannot see your screen.", "read the screen"),
    ],
)
def test_a_denial_of_a_registered_capability_is_caught(reply: str, subject: str) -> None:
    found = denied_registered_capabilities(reply)

    assert found, f"not caught: {reply!r}"
    assert any(denial.subject == subject for denial in found), (
        f"{reply!r} -> {[d.subject for d in found]}"
    )


@pytest.mark.parametrize(
    "reply",
    [
        "I can modify my own source code when asked.",
        "I remember what you copied.",
        "I don't have a favourite colour.",
        "I can read the screen and tell you what is there.",
    ],
)
def test_an_ordinary_reply_is_not_a_denial(reply: str) -> None:
    assert denied_registered_capabilities(reply) == ()


def test_every_subject_names_skills_that_exist() -> None:
    """A subject bound to a skill name nobody registers can never fire.

    That is how a guard dies quietly: the table still lists the capability,
    the registry has renamed it, and the denial sails through.
    """
    from core.capability_engine import CapabilityEngine

    registered = set(getattr(CapabilityEngine(), "skills", {}) or {})
    if not registered:
        pytest.skip("no capability registry in this process")

    dead = {
        subject: [name for name in skills if name not in registered]
        for _pattern, subject, skills in _DENIAL_SUBJECTS
    }
    dead = {subject: names for subject, names in dead.items() if names}

    assert not dead, f"subjects bound to unregistered skills: {dead}"


def test_the_self_modification_family_is_covered() -> None:
    """The live miss, as a standing claim about the table."""
    covered = {
        name for _pattern, _subject, skills in _DENIAL_SUBJECTS for name in skills
    }

    assert "improve_own_code" in covered
    assert "self_repair" in covered
