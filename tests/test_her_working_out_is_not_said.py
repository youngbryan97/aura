"""Her working out of what she was given is not said.

LIVE 8 November 2026, rehearsal 11: her reply on the episode she picked was "Okay so I'm in the middle of a
watching-along session. The user just said ... Looking at the sensory evidence: ... Wait, let me re-read carefully. The
observation says: ...". The turn had been taken for an answer read by code, so when it ran into its budget it was
handed over as it was.
"""
from __future__ import annotations

import inspect

import pytest

from core.utils.an_answer import talks_about_the_asking

LEAKED = ("Okay so I'm in the middle of a watching-along session. The user just said \"Now Potter Puppet Pals.\"\n\n"
          "Looking at the sensory evidence: I searched and found it. Wait, let me re-read carefully. The observation "
          "says: 'Found it'")


@pytest.mark.parametrize("said", [
    "The user just said hello.",
    "Looking at the sensory evidence: a cat on a mat.",
    "Wait, let me re-read. The observation says the page opened.",
])
def test_talk_of_what_she_was_handed_is_commentary_not_an_answer(said):
    assert talks_about_the_asking(said, "")


@pytest.mark.parametrize("said", [
    "The evidence shows the vaccine works.",
    "I picked Neville's Birthday because it is the fuller episode.",
    "What I observed was a puppet facing the curtain.",
])
def test_an_answer_that_mentions_evidence_or_observing_is_an_answer(said):
    assert not talks_about_the_asking(said, "")


def test_a_turn_of_hers_that_is_her_working_out_is_not_said():
    from core.skills import watching_along

    source = inspect.getsource(watching_along.her_turn)
    assert "talks_about_the_asking(words" in source
    assert talks_about_the_asking(LEAKED, "Now Potter Puppet Pals. Pick whichever episode you want.")


def test_a_turn_of_hers_is_not_taken_for_an_answer_code_reads():
    from core.brain import inference_gate_turn_setup

    source = inspect.getsource(inference_gate_turn_setup)
    assert 'and not context.get("her_own_turn")\n                and not self._origin_is_user_facing(' in source
