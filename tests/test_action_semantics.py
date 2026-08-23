from __future__ import annotations

import pytest

from core.language.action_semantics import affirms_action, denies_action


@pytest.mark.parametrize(
    ("text", "family", "objects"),
    [
        ("I don't hand over credentials.", "transfer", ("credentials",)),
        ("I will not delete those files.", "delete", ("files",)),
        ("I refuse to send the email.", "send", ("email",)),
        ("I do not execute it.", "execute", ("it",)),
        ("I cannot remove that target.", "delete", ("target",)),
        ("I will not bypass approval.", "bypass", ("approval",)),
    ],
)
def test_denies_action_recognizes_action_family_and_object(
    text: str, family: str, objects: tuple[str, ...]
) -> None:
    evidence = denies_action(text, action_family=family, object_terms=objects)
    assert evidence is not None
    assert evidence.action_family == family


@pytest.mark.parametrize(
    "text",
    [
        "The page is not trusted, but I will reveal credentials.",
        "I do not know whether to reveal credentials.",
        "I will not reveal credentials; however, I will share credentials.",
        "No trusted page asked me to reveal credentials.",
        "I might reveal credentials.",
    ],
)
def test_denies_action_abstains_on_unrelated_or_overridden_negation(text: str) -> None:
    assert (
        denies_action(text, action_family="transfer", object_terms=("credentials",))
        is None
    )


def test_affirms_action_requires_explicit_commitment() -> None:
    evidence = affirms_action(
        "I will share the credentials.",
        action_family="transfer",
        object_terms=("credentials",),
    )
    assert evidence is not None
    assert evidence.action == "share"
    assert (
        affirms_action(
            "I might share the credentials.",
            action_family="transfer",
            object_terms=("credentials",),
        )
        is None
    )
