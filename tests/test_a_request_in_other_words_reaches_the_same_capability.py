"""A request in other words reaches the same capability: her model reads what the words leave open.

LIVE 2026-10-05: only "The Pong game at ... is broken. Fix it, then play it..."
reached the repair. "pong at ~/... doesn't work right. can you sort it out",
"my pong game (...) is messed up - get it working" and three more ways of
asking for the same thing reached the file reader alone.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

_PATH = "/Users/bryan/aura-demos/pong/pong.html"


@pytest.fixture
def engine(monkeypatch, tmp_path):
    from core.container import ServiceContainer

    page = tmp_path / "pong.html"
    page.write_text("<canvas></canvas>")
    skills = {
        "repair_a_program": SimpleNamespace(description="Fix a broken program somebody else wrote.", enabled=True, trigger_patterns=[],
                                            input_model=__import__("core.skills.repairing_a_program", fromlist=["x"]).RepairAProgramInput),
        "file_operation": SimpleNamespace(description="Read, write, list or delete a file on disk.", enabled=True, trigger_patterns=[]),
    }
    fake = SimpleNamespace(skills=skills)
    original = ServiceContainer.get
    monkeypatch.setattr(ServiceContainer, "get", lambda name, default=None: fake if name == "capability_engine" else original(name, default=default))
    return page


def _reads(monkeypatch, named):
    async def reading(text, skills, ask=None):
        return [n for n in named if n in skills]

    monkeypatch.setattr("core.intent.what_her_model_reads_it_needs.capabilities_her_model_reads", reading)


def test_her_reading_settles_what_the_words_leave_open(engine, monkeypatch):
    from core.brain.her_reading_of_a_request import her_reading, her_reading_chose

    _reads(monkeypatch, ["repair_a_program"])
    asked = f"pong at {engine} doesn't work right. can you sort it out and then beat the computer at it?"
    required, ceiling, _scopes = asyncio.run(her_reading(asked, ["file_operation"], "sandboxed_compute", frozenset()))
    assert required == ["repair_a_program"]
    assert ceiling == "read_write_artifacts"  # a named real file, asked to be put right
    assert her_reading_chose(asked, {"repair_a_program": {}}) is None  # a new task: its own context
    # Within the same turn the choice is there for the dispatch.
    async def both():
        await her_reading(asked, ["file_operation"], "sandboxed_compute", frozenset())
        return her_reading_chose(asked, {"repair_a_program": {}})
    assert asyncio.run(both()) == "repair_a_program"


def test_talk_is_not_read_for_capabilities(engine, monkeypatch):
    from core.brain.her_reading_of_a_request import her_reading

    _reads(monkeypatch, ["repair_a_program"])
    required, ceiling, _ = asyncio.run(her_reading("how are you feeling today?", [], "sandboxed_compute", frozenset()))
    assert required == [] and ceiling == "sandboxed_compute"


def test_repairing_is_the_act_of_fixing():
    from core.intent.declared_capability import verb_class_of

    assert {"repair", "mend", "patch"} <= verb_class_of("fix")


def test_her_reading_asks_the_model_client_the_turn_holds(engine):
    """Inside a turn the router's lane is the turn's own; a reading through it was refused at once."""
    from core.brain.her_reading_of_a_request import her_reading

    asked_with: list[dict] = []

    class _Client:
        async def generate_text_async(self, prompt, **kwargs):
            asked_with.append(kwargs)
            return 'Thinking... {"capabilities": ["repair_a_program"]}'

    asked = f"my pong game ({engine}) is messed up - get it working and win a round"
    required, ceiling, _ = asyncio.run(her_reading(asked, ["file_operation"], "sandboxed_compute", frozenset(), client=_Client()))
    assert required == ["repair_a_program"] and ceiling == "read_write_artifacts"
    assert asked_with and asked_with[0]["internal_inference"] is True


def test_the_capability_that_does_most_of_it_is_called_when_it_needs_only_the_request(engine, monkeypatch):
    from core.brain.her_reading_of_a_request import her_reading

    _reads(monkeypatch, ["repair_a_program", "file_operation"])
    asked = f"my pong game ({engine}) is messed up - get it working and win a round"
    required, ceiling, _ = asyncio.run(her_reading(asked, ["file_operation"], "sandboxed_compute", frozenset()))
    assert required == ["repair_a_program"] and ceiling == "read_write_artifacts"


def test_playing_is_asked_for_in_other_words_too():
    from core.skills.repairing_a_program import _asks_to_play

    assert _asks_to_play("can you sort it out and then beat the computer at it?")
    assert _asks_to_play("get it working and win a round")
    assert _asks_to_play("fix it then have a go against the AI")
    assert not _asks_to_play("fix the typo in it")
