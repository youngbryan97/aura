"""Whether the turn asks for something to exist afterwards.

The effect ceiling asks one question before deciding what a turn may reach
for: did the person ask for an effect, or only for an answer? It answered it
with `asks_to_build_software`, whose own docstring states the principle
correctly — "Asking for a page to exist is asking for that effect" — and whose
scope is software.

LIVE, 2026-08-22. "I have to present you to a funding panel in 10 minutes. Six
slides, no fluff..." ran under the self-service ceiling, so every capability
that produces a file was filtered out before it could be offered. The model
was handed code_repl, diagnose_repo and quantum_lab, invented a tool called
`generate_slides`, and wrote one slide of six as prose. It had invented
`create_slides` on the previous attempt. A deck is not software, and asking
for one is exactly as much a request for a thing to exist.

So the question here is the general one. The words below are a floor for the
phrasings somebody thought of; the learned surface settles the ones nobody
did, and every phrasing the floor is sure about teaches it.
"""

from __future__ import annotations

import re

from core.language.learned_matcher import LearnedMatcher as _LearnedMatcher
from core.language.model_features import model_hidden_features as _model_hidden_features

__all__ = ["asks_for_an_artifact", "names_an_artifact"]

#: Things a person asks to be given, that exist after the turn ends.
_ARTIFACT_NOUNS = (
    r"web\s*app|webapp|app|site|website|web\s*page|page|html|game|tool|script|"
    r"program|widget|dashboard|prototype|demo|utility|"
    r"deck|slides?|presentation|pitch|"
    r"report|memo|one[\s-]?pager|onepager|summary|write[\s-]?up|document|doc|"
    r"file|spreadsheet|csv|chart|diagram|plan|checklist|template|draft"
)

#: Asking to be given one.
_ASKS_FOR = re.compile(
    r"\b(?:build|make|create|write|draft|put\s+together|knock\s+up|generate|"
    r"produce|prepare|assemble|give\s+me|send\s+me|i\s+need|i\s+want|"
    r"can\s+you\s+(?:build|make|write|draft|put\s+together|prepare))\b"
    rf"[^.?!]{{0,60}}?\b(?:{_ARTIFACT_NOUNS})\b",
    re.IGNORECASE,
)

#: A bare shape with a count is a request for one: "six slides, no fluff".
_COUNTED_ARTIFACT = re.compile(
    rf"\b\d{{1,2}}[\s-]*(?:{_ARTIFACT_NOUNS})\b"
    rf"|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|twelve)"
    rf"[\s-]*(?:{_ARTIFACT_NOUNS})\b",
    re.IGNORECASE,
)

#: Asking ABOUT one is not asking for one.
_ASKS_ABOUT = re.compile(
    r"\b(?:what\s+is|what's|how\s+does|how\s+do|why\s+is|explain|describe|"
    r"tell\s+me\s+about|what\s+do\s+you\s+think)\b",
    re.IGNORECASE,
)


def names_an_artifact(message: object) -> bool:
    """Whether the request names a thing that would exist afterwards."""
    text = str(message or "")
    if not text.strip():
        return False
    return bool(_ASKS_FOR.search(text) or _COUNTED_ARTIFACT.search(text))


#: Whether the person wants a thing or an answer.
#:
#: The words above are the floor. This is the mechanism: a judgement about
#: what somebody meant belongs to the learned surface, and a list of nouns
#: will always be a list of the nouns somebody thought of.
_WANTS_A_THING = _LearnedMatcher(
    name="artifact_request",
    positives=(
        "Six slides, no fluff: what you are and what you can do.",
        "put together a short report on what you found",
        "make me a deck for the funding panel",
        "build me a little web app for tracking water",
        "can you knock up something I can show them on Thursday",
        "I need something I can send to the team by five",
        "give me a checklist for the move",
    ),
    negatives=(
        "what is a deck?",
        "explain how slides work",
        "how are you feeling today?",
        "who founded Hugging Face?",
        "tell me about Anthropic the company",
        "what do you think about consciousness?",
        "why is that test failing?",
    ),
    features=_model_hidden_features,
)


def _floor_says(text: str) -> bool | None:
    """What the words settle, or None when they settle nothing."""
    try:
        from core.runtime.desktop_objective_intent import asks_to_build_software

        if asks_to_build_software(text):
            return True
    except (ImportError, AttributeError, TypeError, ValueError):
        pass
    if names_an_artifact(text):
        # "What is a deck?" names one and asks for nothing.
        opening = text.strip().split(".", 1)[0]
        return not _ASKS_ABOUT.match(opening.strip())
    if _ASKS_ABOUT.match(text.strip()):
        return False
    return None


def asks_for_an_artifact(message: object) -> bool:
    """Whether this turn asks for something to exist when it is over.

    Software is included, because building software is one way of producing a
    thing; it is not the only way, which is what the narrower reader could not
    say.

    The words settle what they can and teach the learned surface as they go,
    so "knock up something I can show them on Thursday" can reach the same
    answer as "make me a deck" without anyone adding a noun.
    """
    text = str(message or "")
    if not text.strip():
        return False
    settled = _floor_says(text)
    if settled is not None:
        try:
            _WANTS_A_THING.observe(text, holds=settled)
        except (RuntimeError, TypeError, ValueError):
            pass
        return settled
    try:
        learned = _WANTS_A_THING.decide_without_waiting(text)
    except (RuntimeError, TypeError, ValueError):
        learned = None
    return bool(learned)
