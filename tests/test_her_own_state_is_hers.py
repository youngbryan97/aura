"""Her own condition is described in the first person, because it is hers.

LIVE, 2026-08-19. Asked what she had genuinely changed her mind about, she
answered:

    I notice your mood is marked as TIRED. That sounds like a heavy load right
    now. We don't need to keep running diagnostics ... Would you like to just
    sit with that feeling for a moment?

The tiredness was HERS — the viability tick had moved healthy → tired moments
earlier — and it reached her as "Your mood is tired." A second-person sentence
about her own state is one pronoun away from being a claim about whoever she
is talking to, and a model generating a reply echoes the pronoun it was given.
So an intimate question got a status line about someone else's feelings, and
the person on the other end was offered comfort for a mood that was not his.

The block carrying it was already headed "do not narrate these values". She
narrated them anyway, and misattributed them, which is the argument for fixing
the subject rather than the wording.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: Files that describe how she is, in text she will read as her own context.
STATE_SOURCES = [
    "core/cognitive/state_machine.py",
    "core/consciousness/homeostatic_coupling.py",
    "core/voice/speech_profile.py",
    "core/executive/authority_gateway.py",
]

#: Second person applied to a felt state. "You are Aura" is fine — that is who
#: she is, not how she feels.
SECOND_PERSON_STATE = re.compile(
    r"\b(?:You|Your)\s+(?:feel|are\s+feeling|'re\s+feeling|mood|curiosity|"
    r"energy|thoughts\s+are|are\s+worried|'re\s+craving)\b"
)


@pytest.mark.parametrize("relative", STATE_SOURCES)
def test_no_felt_state_is_addressed_to_her_in_the_second_person(relative: str):
    source = (ROOT / relative).read_text()
    offenders: list[str] = []
    for number, line in enumerate(source.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("#") or '"' not in line and "'" not in line:
            continue
        for literal in re.findall(r'"[^"]*"|\'[^\']*\'', line):
            if SECOND_PERSON_STATE.search(literal):
                offenders.append(f"{relative}:{number}: {literal[:80]}")
    assert not offenders, "her own state addressed to her as 'you':\n" + "\n".join(offenders)


def test_the_tone_cues_speak_in_the_first_person():
    source = (ROOT / "core/cognitive/state_machine.py").read_text()
    assert 'tone_cues = [f"My mood is {mood}."]' in source
    assert "I'm craving conversation." in source
    assert "I have {goals} goal" in source


def test_the_block_says_whose_state_it_is():
    """The header is where the ambiguity started."""
    source = (ROOT / "core/cognitive/state_machine.py").read_text()
    assert "HOW I AM RIGHT NOW (my own state, not the other person's)" in source
