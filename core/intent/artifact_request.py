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


def asks_for_an_artifact(message: object) -> bool:
    """Whether this turn asks for something to exist when it is over.

    Software is included, because building software is one way of producing a
    thing; it is not the only way, which is what the narrower reader could not
    say.
    """
    text = str(message or "")
    if not text.strip():
        return False
    try:
        from core.runtime.desktop_objective_intent import asks_to_build_software

        if asks_to_build_software(text):
            return True
    except (ImportError, AttributeError, TypeError, ValueError):
        pass
    if not names_an_artifact(text):
        return False
    # "What is a deck?" names one and asks for nothing.
    opening = text.strip().split(".", 1)[0]
    return not _ASKS_ABOUT.match(opening.strip())
