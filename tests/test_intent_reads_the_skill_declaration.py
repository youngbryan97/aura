"""A skill is reachable by what it says it does, not by phrases someone listed.

LIVE, 2026-08-19. "run a tiny bit of python and give me the actual number it
printed" dispatched nothing, with ``code_repl`` READY. Its trigger list was::

    run (?:this )?(?:python )?code
    execute (?:this )?(?:python )?(?:code|script)

so "run this code" matched and four ordinary phrasings did not — the modifier
slot is hard-coded as ``(?:this )?``, and ``python`` is written as an optional
modifier of a required ``code``. 37 registered skills have no patterns at all
and were unreachable by intent under every phrasing.

The tests are about the relation, not the sentences: a request is an act, an
object, and the mood that separates asking from mentioning.
"""

from __future__ import annotations

import pytest

from core.intent.declared_capability import (
    declared_vocabulary,
    distinctive_objects,
    request_matches_declaration,
    verb_class_of,
)

#: Declarations as the registry really carries them.
CATALOGUE = {
    "code_repl": "Execute Python code in a real-time, sandboxed REPL",
    "web_search": "Search the web for current information",
    "image_gen": "Generate images from a text prompt using diffusion",
    "desktop_task": "Open applications and click things on the desktop",
    "memory_ops": "Store and recall memories from long term memory",
    "email_adapter": "Send and read email messages",
    "install_package": "Install a python package into the environment",
}


@pytest.fixture(scope="module")
def matcher():
    vocabulary = {n: declared_vocabulary(n, d) for n, d in CATALOGUE.items()}
    objects = distinctive_objects(vocabulary)

    def match(message: str) -> list[str]:
        return [
            name
            for name, (verbs, _declared) in vocabulary.items()
            if request_matches_declaration(
                message, verbs=verbs, objects=objects.get(name, frozenset())
            )
        ]

    return match


@pytest.mark.parametrize(
    "message",
    [
        "run a tiny bit of python and give me the actual number it printed",
        "can you actually run some python for me",
        "execute a quick script and show the output",
        "use your interpreter and tell me what 2**40 is",
        "please compute that expression for real",
        "i want to run some python",
        "run this code",
    ],
)
def test_every_ordinary_way_of_asking_to_execute_reaches_the_executor(matcher, message):
    """The live miss, and the four phrasings around it."""
    assert "code_repl" in matcher(message)


@pytest.mark.parametrize(
    "message,expected",
    [
        ("search the web for the latest on fusion", "web_search"),
        ("look up who won the game last night online", "web_search"),
        ("use the web to check that", "web_search"),
        ("make me an image of a lighthouse", "image_gen"),
        ("open the desktop calculator", "desktop_task"),
        ("install the requests package", "install_package"),
    ],
)
def test_the_same_reading_works_for_every_other_skill(matcher, message, expected):
    """Nothing about this is specific to running code."""
    assert expected in matcher(message)


@pytest.mark.parametrize(
    "message",
    [
        "my code doesn't run anymore",
        "pythons are constrictors, not venomous",
        "I ran a marathon, then wrote some code",
        "the web is full of nonsense these days",
        "my interpreter broke last week",
        "I use python at work",
        "she used to send me emails every day",
        "that image is beautiful",
    ],
)
def test_mentioning_a_thing_is_not_asking_for_it(matcher, message):
    """Mood is the difference, and a false dispatch is the expensive error.

    "my code doesn't run" carries the verb and the object of a request to
    execute code, in one clause. Only where the verb SITS says it is a
    complaint.
    """
    assert matcher(message) == []


def test_a_verb_alone_and_an_object_alone_are_both_insufficient(matcher):
    assert matcher("run") == []
    assert matcher("python") == []


def test_verbs_are_matched_by_act_rather_than_spelling():
    """"run" has to reach a skill that declared "execute"."""
    assert verb_class_of("run") == verb_class_of("execute")
    assert verb_class_of("look") == verb_class_of("search")
    assert verb_class_of("run") != verb_class_of("search")
    assert verb_class_of("banana") == frozenset()


def test_past_tense_is_absent_from_every_verb_class():
    """"I ran the numbers" reports; "run the numbers" asks.

    Keeping narration out of the classes is what stops a story about what
    someone did from dispatching a tool.
    """
    for tense in ("ran", "executed", "searched", "opened", "sent", "installed"):
        assert verb_class_of(tense) == frozenset(), tense


def test_distinctiveness_is_measured_over_the_catalogue_not_chosen():
    """Common words stop selecting on their own as the roster grows."""
    vocabulary = {n: declared_vocabulary(n, d) for n, d in CATALOGUE.items()}
    objects = distinctive_objects(vocabulary)
    assert "code" in objects["code_repl"]
    # A word every skill claims separates nothing, so it stops selecting.
    shared = {f"s{i}": declared_vocabulary(f"s{i}", "Handle the thing") for i in range(7)}
    for kept in distinctive_objects(shared).values():
        assert "thing" not in kept
    # No skill is left with nothing to be selected by.
    for name, kept in objects.items():
        assert kept, f"{name} became unreachable"


def test_a_skill_with_no_trigger_patterns_is_still_reachable(matcher):
    """37 skills had none, so no phrasing could reach them at all."""
    vocabulary = {n: declared_vocabulary(n, d) for n, d in CATALOGUE.items()}
    vocabulary["quantum_lab"] = declared_vocabulary("quantum_lab", "Simulate quantum circuits")
    objects = distinctive_objects(vocabulary)
    verbs, _ = vocabulary["quantum_lab"]
    # "simulate" belongs to no verb class, and a description is a verb phrase,
    # so its first content word is the act.
    assert "simulate" in verbs
    assert request_matches_declaration(
        "simulate a quantum circuit for me", verbs=verbs, objects=objects["quantum_lab"]
    )


@pytest.mark.parametrize(
    "request_text",
    [
        "now read /tmp/x/accounts.py - the test fails. which line is wrong?",
        "ok read the file and tell me",
        "and then run the tests",
        "just open the config",
        "so search the web for that",
    ],
)
def test_a_discourse_word_before_an_imperative_is_still_a_request(request_text: str):
    """"Now read the file" is "read the file" with one word in front.

    LIVE, 2026-08-19: "now read /private/tmp/.../accounts.py - the test fails"
    matched nothing, so the turn was handed no capabilities and a downstream
    fallback chose web_interlocutor — a browser-dialogue skill — to debug a
    Python file.
    """
    from core.intent.declared_capability import looks_like_a_request

    assert looks_like_a_request(request_text)


@pytest.mark.parametrize(
    "conversation",
    [
        "now that you mention it, I agree",
        "so my code doesn't run anymore",
        "well I use python at work",
    ],
)
def test_a_discourse_word_does_not_make_a_statement_a_request(conversation: str):
    from core.intent.declared_capability import looks_like_a_request

    assert not looks_like_a_request(conversation)


@pytest.mark.parametrize(
    "revision",
    [
        "Add one limitation and connect it to the example.",
        "Include another caveat in that explanation.",
        "Expand the second point with a concrete counterexample.",
        "Tie the conclusion back to the opening claim.",
    ],
)
def test_conversational_revisions_name_no_machine_domain(revision: str):
    from core.intent.declared_capability import requested_foundational_domains

    assert requested_foundational_domains(revision) == ()


@pytest.mark.parametrize(
    ("request_text", "expected"),
    [
        ("Read README.md and tell me what it says.", {"file"}),
        ("Inspect /tmp/ledger and run its tests.", {"file", "code"}),
        ("Search the web for current release notes.", {"web"}),
        ("Run some Python and tell me the result.", {"code"}),
        ("Add 41 and 1.", {"code"}),
    ],
)
def test_machine_domains_come_from_the_requests_objects_or_structure(
    request_text: str,
    expected: set[str],
):
    from core.intent.declared_capability import requested_foundational_domains

    assert set(requested_foundational_domains(request_text)) == expected


def test_a_weak_semantic_neighbor_does_not_join_the_best_supported_skill():
    from core.intent.declared_capability import rank_declaration_matches

    declarations = {
        "code_repl": declared_vocabulary(
            "code_repl", "Execute Python code in a sandboxed REPL"
        ),
        "program_dna_equivalence_battery": declared_vocabulary(
            "program_dna_equivalence_battery",
            "Run a program equivalence battery across simulated applications",
        ),
    }
    ranked = rank_declaration_matches(
        "Run some Python and tell me the result.",
        declarations,
        distinctive_objects(declarations),
    )

    assert [name for name, _score in ranked] == ["code_repl"]
