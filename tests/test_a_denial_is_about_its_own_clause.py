"""A denial binds to the clause it sits in, not to every word in the sentence.

LIVE 2026-09-16, two turns apart. "I can't measure the sun from here, so I'll
work it out from the geometry of the world" opened a correct daylight answer
and was replaced by "I can reach beyond this conversation: web search, screen
capture, camera." Then "I'd rather flag that than repeat a number I can't
verify", in a sentence that also said "largest file", was replaced by a status
line about the filesystem, and a sentence that named a skill became "I can web
interlocutor — web_interlocutor are registered and enabled right now".

Each detector had the denial right and the subject wrong: it took any
capability word in the sentence as the thing denied.
"""

from __future__ import annotations

import pytest

from core.conversation import capability_denial as cd
from core.self import capability_ledger as cl
from core.self.denial_scope import complement_names_something_else, denied_span


def _present(name: str, subjects: tuple[str, ...], **fields) -> cl.LiveCapability:
    availability = cl.Availability(name=name, present=True, usable_now=True, summary=f"{name} works.")
    return cl.LiveCapability(name, subjects, lambda: availability, **fields)


def _ledger(*capabilities: cl.LiveCapability) -> cl.CapabilityLedger:
    ledger = cl.CapabilityLedger()
    for capability in capabilities:
        ledger.register(capability)
    return ledger


WORLD = ("world", "internet", "web", "outside", "external", "news")
INTEROCEPTION = ("vitals", "energy", "focus", "body", "sensor", "sensors", "telemetry")
READS = ("energy", "focus", "vitals", "telemetry", "body", "state", "internal", "mood", "load")


# ── the ledger ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "sentence",
    [
        "I can't measure the sun from here, so I'll work it out from the geometry of the world instead.",
        "I can't step onto the roof to check; the outside figure is geometric.",
        "Without a sensor pointed at the sky I can't read the daylight directly, so here is the calculation from the news of the day.",
    ],
)
def test_a_word_in_the_next_clause_is_not_what_was_denied(sentence: str) -> None:
    ledger = _ledger(_present("world_access", WORLD))

    assert ledger.contradicted_claims(sentence) == []


@pytest.mark.parametrize(
    "sentence",
    [
        "I can't reach the web, so I'll estimate it.",
        "I have no way onto the internet right now — sorry.",
        "Right now I can't search the web for today's news.",
    ],
)
def test_a_word_in_the_denied_clause_still_is(sentence: str) -> None:
    ledger = _ledger(_present("world_access", WORLD))

    assert [claim.availability.name for claim in ledger.contradicted_claims(sentence)] == ["world_access"]


def test_an_instrument_pointed_at_something_it_does_not_read_is_not_denied() -> None:
    ledger = _ledger(_present("interoception", INTEROCEPTION, reads=READS))

    assert ledger.contradicted_claims("I have no sensor for daylight, so here is the geometry.") == []
    assert ledger.contradicted_claims("There is no sensor in me for the sky, but the maths is enough.") == []


@pytest.mark.parametrize(
    "sentence",
    [
        "I have no sensors.",
        "I have no sensor for my energy.",
        "Current energy and focus numbers: Not readable.",
        "I do not have a body or sensor of any kind.",
    ],
)
def test_a_denial_of_what_it_does_read_is_still_caught(sentence: str) -> None:
    ledger = _ledger(_present("interoception", INTEROCEPTION, reads=READS))

    assert ledger.contradicted_claims(sentence), sentence


def test_the_complement_rule_reads_stems_and_ignores_function_words() -> None:
    assert complement_names_something_else(" for daylight", INTEROCEPTION + READS)
    assert not complement_names_something_else(" for my vitals", INTEROCEPTION + READS)
    assert not complement_names_something_else(" of any kind", INTEROCEPTION + READS)
    assert not complement_names_something_else(" right now.", INTEROCEPTION + READS)


# ── the span itself ─────────────────────────────────────────────────────────

def test_the_span_stops_at_the_clause_after_the_denial() -> None:
    import re

    frame = re.compile(r"i\s+can'?t\b", re.IGNORECASE)
    sentence = "I can't measure the sun from here, so I'll use the world's geometry."

    assert denied_span(sentence, frame.search(sentence)) == "I can't measure the sun from here"


def test_the_span_starts_at_the_clause_before_it_and_keeps_a_label() -> None:
    import re

    frame = re.compile(r"i\s+can'?t\b|not\s+readable\b", re.IGNORECASE)
    with_a_label = "Current energy and focus numbers: Not readable."
    assert denied_span(with_a_label, frame.search(with_a_label)) == with_a_label

    after_a_but = "I have a camera, but I can't use the web right now."
    assert denied_span(after_a_but, frame.search(after_a_but)).strip() == "I can't use the web right now."

    pro_form = "I can't do so."
    assert denied_span(pro_form, frame.search(pro_form)) == pro_form


# ── the registry check ──────────────────────────────────────────────────────

class _Meta:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled


class _Engine:
    def __init__(self, names) -> None:
        self.skills = {name: _Meta() for name in names}


def test_a_denial_in_one_clause_does_not_take_a_subject_from_another() -> None:
    sentence = (
        "So if my original answer named it as the largest file with a specific byte "
        "count, I was wrong, or the tree moved since then, and I'd rather flag that "
        "than repeat a number I can't verify."
    )

    assert cd.denied_registered_capabilities(sentence, _Engine(["file_operation", "code_repl"])) == ()


def test_the_live_2026_08_17_denial_is_still_caught() -> None:
    denials = cd.denied_registered_capabilities(
        "I don't have file system access or the ability to count files in a directory.",
        _Engine(["file_operation", "computer_use"]),
    )

    assert denials and denials[0].subject == "read the filesystem"


def test_a_skill_named_in_a_denial_is_marked_as_a_name(monkeypatch) -> None:
    from core.self.capability_lexicon import CapabilityMention

    monkeypatch.setattr(
        cd,
        "_registry_mentions",
        lambda sentence, engine: (CapabilityMention("web_interlocutor", ("interlocutor",), True),),
    )
    denials = cd.denied_registered_capabilities(
        "I can't run the web interlocutor from this turn.", _Engine(["web_interlocutor"])
    )

    assert denials and denials[0].named_skill and denials[0].subject == "web interlocutor"


def test_a_named_skill_is_not_conjugated_as_a_verb(monkeypatch) -> None:
    from interface.routes import chat_reply_shaping as shaping

    monkeypatch.setattr(
        cd,
        "denied_registered_capabilities",
        lambda text, engine=None: (
            cd.CapabilityDenial(
                subject="web interlocutor",
                sentence="I can't run the web interlocutor from this turn.",
                skills=("web_interlocutor",),
                named_skill=True,
            ),
        ),
    )
    monkeypatch.setattr(shaping, "_capabilities_this_turn_needs", lambda: set())
    out = str(shaping._correct_false_capability_denials("I can't run the web interlocutor from this turn."))

    assert "I can web interlocutor" not in out
    assert "web interlocutor" in out and "registered and enabled" in out
