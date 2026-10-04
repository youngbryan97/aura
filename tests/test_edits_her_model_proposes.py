"""What her model proposes for a program is a list of edits to try, never a fix: only text really there becomes one."""
from __future__ import annotations

import asyncio

import pytest

from core.self_modification.code_that_looks_wrong import applied
from core.self_modification.edits_her_model_proposes import (
    ProposedEdit,
    ProposedEdits,
    edits_from_proposals,
    edits_her_model_proposes,
)

pytestmark = pytest.mark.unit

PROGRAM = """function step(world) {
  world.x += world.speed;
  if (world.x > world.width) world.x = 0;
  return world;
}
"""


def test_a_proposal_naming_text_on_its_line_becomes_an_edit():
    [suspicion] = edits_from_proposals(PROGRAM, ProposedEdits(edits=[ProposedEdit(line=3, old="world.x = 0", new="world.x = world.width", why="it should stop at the edge")]))
    assert suspicion.function == "step" and suspicion.pattern == "proposed by her model"
    assert "world.x = world.width;" in applied(PROGRAM, suspicion.edits)


def test_a_proposal_for_code_that_is_not_there_is_dropped():
    assert edits_from_proposals(PROGRAM, ProposedEdits(edits=[ProposedEdit(line=2, old="world.y -= 1", new="world.y += 1")])) == []


def test_an_empty_old_inserts_before_the_line():
    [suspicion] = edits_from_proposals(PROGRAM, ProposedEdits(edits=[ProposedEdit(line=3, old="", new="  if (world.x < 0) world.x = 0;")]))
    lines = applied(PROGRAM, suspicion.edits).splitlines()
    assert lines[2] == "  if (world.x < 0) world.x = 0;" and lines[3].startswith("  if (world.x > world.width)")


def test_nothing_is_asked_when_nothing_was_seen_wrong():
    class _Never:
        async def generate(self, *_args, **_kwargs):
            raise AssertionError("asked")

    assert asyncio.run(edits_her_model_proposes(PROGRAM, {}, advisor=_Never())) == []


def test_what_the_model_returns_is_read_as_proposals():
    class _Advisor:
        async def generate(self, prompt, **_kwargs):
            assert "seen_wrong" in prompt and "0003:" in prompt
            return ProposedEdits(edits=[ProposedEdit(line=2, old="+=", new="-=", why="it goes the wrong way")])

    [suspicion] = asyncio.run(edits_her_model_proposes(PROGRAM, {"controls": "it goes the wrong way"}, advisor=_Advisor()))
    assert "world.x -= world.speed" in applied(PROGRAM, suspicion.edits)
