"""She must not deny a capability the registry says she has.

LIVE 2026-08-17: "how many python files live in core/introspection?" was
answered "I don't have file system access or the ability to count files in a
directory." Eight filesystem-capable skills were registered and enabled at that
moment, and the same question phrased "are in" was answered exactly, with
filenames listed.

A wrong denial is worse than a wrong attempt: it teaches the person the product
cannot do something it can, and they stop asking.
"""

from __future__ import annotations

import pytest

from core.conversation.capability_denial import denied_registered_capabilities
from interface.routes.chat import _correct_false_capability_denials as correct


class _Meta:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled


class _Engine:
    def __init__(self, names, enabled: bool = True) -> None:
        self.skills = {n: _Meta(enabled) for n in names}


def test_the_live_denial_is_detected() -> None:
    denials = denied_registered_capabilities(
        "I don't have file system access or the ability to count files in a directory.",
        _Engine(["file_operation", "computer_use"]),
    )

    assert denials and denials[0].subject == "read the filesystem"


@pytest.mark.parametrize(
    "sentence",
    [
        "I do not have access to your files.",
        "I cannot read your screen.",
        "I can't search the web for that.",
        "I have no way to reach the clipboard.",
    ],
)
def test_denials_across_capabilities_are_detected(sentence: str) -> None:
    engine = _Engine(
        ["file_operation", "computer_use", "web_search", "os_automation", "desktop_task"]
    )

    assert denied_registered_capabilities(sentence, engine)


# ── it must not overrule a CHOICE ───────────────────────────────────────────

@pytest.mark.parametrize(
    "sentence",
    [
        "I would rather not help with that.",
        "I won't do that.",
        "I'd prefer not to read your files.",
    ],
)
def test_a_refusal_on_principle_is_left_alone(sentence: str) -> None:
    """Declining is a choice. Overriding it would be worse than the denial."""
    engine = _Engine(["file_operation", "computer_use"])

    assert denied_registered_capabilities(sentence, engine) == ()


def test_a_capability_she_genuinely_lacks_is_not_contradicted() -> None:
    """Nothing registered for it, so the denial is true and stands."""
    engine = _Engine(["memory_write"])

    assert denied_registered_capabilities("I cannot read your screen.", engine) == ()


def test_a_disabled_skill_does_not_count_as_a_capability() -> None:
    engine = _Engine(["file_operation", "computer_use"], enabled=False)

    assert denied_registered_capabilities("I have no access to your files.", engine) == ()


def test_an_unreadable_registry_gives_no_opinion() -> None:
    """Claiming a capability she lacks is the same defect pointing the other way."""
    assert denied_registered_capabilities("I cannot read your screen.", _Engine([])) == ()


# ── the correction replaces, and names real skills ──────────────────────────

def test_the_denial_sentence_is_replaced_not_annotated() -> None:
    out = str(correct("I don't have file system access. Sorry about that."))

    assert "I don't have file system access" not in out
    assert "Sorry about that." in out


def test_the_replacement_names_registered_skills() -> None:
    out = str(correct("I do not have access to your files."))

    assert any(skill in out for skill in ("file_operation", "computer_use", "desktop_task"))


def test_ordinary_replies_are_untouched() -> None:
    original = "I counted ten files and pushed the change."

    assert correct(original) == original


def test_the_same_capability_is_corrected_only_once() -> None:
    """Two denials of one thing produced the same replacement twice, verbatim."""
    out = str(correct(
        "I don't have file system access. I can't read files on this machine. Anything else?"
    ))

    assert out.count("registered and enabled") == 1
    assert "Anything else?" in out


def test_removing_a_duplicate_denial_does_not_leave_double_spaces() -> None:
    out = str(correct("I don't have file access. I cannot read files. Done."))

    assert "  " not in out
