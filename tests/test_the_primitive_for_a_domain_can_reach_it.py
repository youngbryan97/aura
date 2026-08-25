"""The one skill offered for a domain has to be able to reach the domain.

"read /etc/hosts and tell me the first line" was offered a single capability:
the skill that runs a project's tests. Nothing in the set could open a file.

Each foundational domain hands a turn exactly one primitive, and the file
domain picked its primitive by counting how many file words a skill's
description contains. ``diagnose_repo`` describes what it prints about a
failing test — the file, the line, the directory, the repo — and outnumbers
"Read, write, append, or list files in the allowed workspace" on its own
subject. Counting words measures how much a description says. It says nothing
about whether the skill can open a file, which is the question being asked.

The code domain has required the act since it was written: an exact problem
needs a primitive that owns code-like state AND declares that it executes it.
File and web were reading the noun alone.
"""

from __future__ import annotations

import pytest

from core.intent.declared_capability import (
    _FOUNDATIONAL_DOMAIN_ACT,
    declared_vocabulary,
    foundational_capabilities,
    request_matches_declaration,
    verb_class_of,
)


@pytest.fixture(scope="module")
def live_catalogue():
    from core.skills.discovery import build_skill_catalog

    return {
        declaration.name: declared_vocabulary(
            declaration.name, str(declaration.description or "")
        )
        for declaration in build_skill_catalog().accepted
    }


@pytest.mark.parametrize("domain", sorted(_FOUNDATIONAL_DOMAIN_ACT))
def test_the_primitive_declares_the_act_its_domain_is_reached_by(domain, live_catalogue):
    """Named by property, so a catalogue change cannot quietly break it."""
    picked = foundational_capabilities(live_catalogue, (domain,))
    assert picked, domain
    verbs, _objects = live_catalogue[picked[0]]
    reaching = verb_class_of(_FOUNDATIONAL_DOMAIN_ACT[domain])
    assert set(verbs) & reaching, (domain, picked[0], sorted(verbs))


def test_a_skill_that_only_mentions_a_domain_is_not_its_primitive():
    """The live shape: a test runner whose report names four file words."""
    catalogue = {
        "file_reader": declared_vocabulary("file_reader", "Read a file and return its lines."),
        "test_runner": declared_vocabulary(
            "test_runner",
            "Run a project's tests and report the failing test: the file and "
            "the line, the source around it, and what the directory says.",
        ),
    }
    assert foundational_capabilities(catalogue, ("file",)) == ["file_reader"]


def test_a_skill_that_writes_the_domain_is_not_its_reader():
    """A builder reaches a file too, and reading is a sideline of what it does.

    Kept separate from the case above because it survives the act gate: a
    document builder prints, and printing is a reading word. What separates
    the two is how much of each declaration is about reading.
    """
    catalogue = {
        "file_reader": declared_vocabulary("file_reader", "Read a file and return its lines."),
        "document_builder": declared_vocabulary(
            "document_builder",
            "Build a document and print it to a file on disk as a report.",
        ),
    }
    assert foundational_capabilities(catalogue, ("file",)) == ["file_reader"]


def test_a_domain_with_no_declared_act_offers_nothing(live_catalogue):
    """Silence is not read as an open gate.

    A domain added to the foundational set without saying what act reaches it
    would otherwise fall back to the noun count this test file exists to
    remove.
    """
    assert foundational_capabilities(live_catalogue, ("web",)) != []
    assert foundational_capabilities(live_catalogue, ("nowhere",)) == []


def test_a_question_that_names_a_path_matches_the_declaration_it_fits():
    """Mood alone cannot answer a question that names a place.

    "why is the test failing in <path>" is a question, so the mood reader is
    right to call it one, and the skill that runs a project's tests and
    reports which assertion failed matched every other way. It was reached
    only because its description held the slot meant for the file reader.
    """
    assert request_matches_declaration(
        "why is the test failing in /tmp/ledger", verbs=["test"], objects=["test"]
    )


@pytest.mark.parametrize(
    "message",
    [
        "my code doesn't run anymore",
        "I use python at work",
    ],
)
def test_a_turn_that_names_no_place_is_still_read_for_mood(message):
    """The gate this loosens is what keeps a complaint from dispatching."""
    assert not request_matches_declaration(message, verbs=["run"], objects=["code"])
