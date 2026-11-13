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


def test_a_reading_that_lands_on_the_reply_itself_asks_for_no_tool(engine, monkeypatch):
    """LIVE 2026-10-05: a word problem was read as needing native_chat, called with no message,
    and "No message provided." was served as her answer."""
    from core.brain.her_reading_of_a_request import her_reading
    from core.container import ServiceContainer
    from core.skills.native_chat import NativeChatSkill

    skills = ServiceContainer.get("capability_engine").skills
    skills["native_chat"] = SimpleNamespace(
        description="Conversational engine with robust dependency resolution.", enabled=True,
        trigger_patterns=[], skill_class=NativeChatSkill,
    )
    _reads(monkeypatch, ["native_chat"])
    asked = "Two trains start 300 km apart and run toward each other; how far does the bird fly?"

    required, ceiling, _scopes = asyncio.run(her_reading(asked, [], "sandboxed_compute", frozenset()))

    assert required == [] and ceiling == "sandboxed_compute"


def test_the_reply_itself_is_not_in_the_catalogue_her_model_reads():
    from core.intent.what_her_model_reads_it_needs import capabilities_her_model_reads
    from core.skills.native_chat import NativeChatSkill

    seen: list[str] = []

    async def ask(prompt):
        seen.append(prompt)
        return SimpleNamespace(capabilities=[])

    skills = {
        "native_chat": SimpleNamespace(description="Conversational engine.", enabled=True, skill_class=NativeChatSkill),
        "file_operation": SimpleNamespace(description="Read, write, list or delete a file on disk.", enabled=True),
    }
    asyncio.run(capabilities_her_model_reads("open the notes file", skills, ask=ask))

    assert "file_operation" in seen[0] and "native_chat" not in seen[0]


def _means(monkeypatch, name):
    monkeypatch.setattr("core.brain.requests_by_meaning.meant_capability", lambda text, skills: name)

    async def asked(_said):
        return True

    monkeypatch.setattr("core.together.taking_it_up.asked_to_do_something", asked)


def test_a_capability_meant_is_reached_whatever_verb_asked_for_it(engine, monkeypatch):
    """LIVE 13 November 2026: "Could you send me a desktop notification that says the tea is ready?" stopped at a verb
    list before her reading by meaning, and she said she could not send notifications, with notify_user ready."""
    from core.brain.her_reading_of_a_request import her_reading
    from core.container import ServiceContainer

    ServiceContainer.get("capability_engine").skills["notify_user"] = SimpleNamespace(
        description="Pushes a native OS desktop notification to the user.", enabled=True, trigger_patterns=[])
    _means(monkeypatch, "notify_user")
    _reads(monkeypatch, [])  # her large model is not asked: the meaning decided
    asked = "Could you send me a desktop notification that says the tea is ready?"
    required, ceiling, scopes = asyncio.run(her_reading(asked, [], "sandboxed_compute", frozenset({"sandboxed_compute"})))
    assert required == ["notify_user"]
    assert ceiling == "external_io" and "external_io" in scopes  # asking for the effect is asking for it


def test_nothing_meant_and_no_order_leaves_the_turn_to_words(engine, monkeypatch):
    from core.brain.her_reading_of_a_request import her_reading

    _means(monkeypatch, None)
    called = []

    async def reading(text, skills, ask=None):
        called.append(text)
        return ["repair_a_program"]

    monkeypatch.setattr("core.intent.what_her_model_reads_it_needs.capabilities_her_model_reads", reading)
    required, ceiling, _ = asyncio.run(her_reading("I had tea with my sister this afternoon.", [], "sandboxed_compute", frozenset()))
    assert required == [] and ceiling == "sandboxed_compute" and called == []


def test_when_meaning_narrows_it_to_a_few_her_model_chooses_among_them(engine, monkeypatch):
    """LIVE 13 November 2026: "Could you put up a desktop notification for me that says time to water the plants?"
    stood notify_user 3.03 out and 0.12 clear of reminder; with no one chosen, nothing was done."""
    from core.brain.her_reading_of_a_request import her_reading
    from core.container import ServiceContainer

    skills = ServiceContainer.get("capability_engine").skills
    skills["notify_user"] = SimpleNamespace(description="Pushes a native OS desktop notification.", enabled=True, trigger_patterns=[])
    skills["reminder"] = SimpleNamespace(description="Sets a reminder for later.", enabled=True, trigger_patterns=[])
    _means(monkeypatch, None)
    monkeypatch.setattr("core.brain.requests_by_meaning.meant_shortlist", lambda text, s: ["notify_user", "reminder"])
    shown = []

    async def reading(text, offered, ask=None):
        shown.append(sorted(offered))
        return ["notify_user"]

    monkeypatch.setattr("core.intent.what_her_model_reads_it_needs.capabilities_her_model_reads", reading)
    asked = "Could you put up a desktop notification for me that says time to water the plants?"
    required, ceiling, scopes = asyncio.run(her_reading(asked, [], "sandboxed_compute", frozenset({"sandboxed_compute"})))
    assert shown == [["notify_user", "reminder"]], "her model chooses among what meaning narrowed it to"
    assert required == ["notify_user"] and ceiling == "external_io"


def test_a_shortlist_is_what_stands_out_from_them_all():
    import numpy as np

    from core.intent.capability_by_meaning import Meanings

    vectors = {"notify": [1.0, 0.0], "remind": [0.97, 0.24], **{f"other{i}": [0.0, 1.0] for i in range(40)}}
    meanings = Meanings(lambda texts: np.array([vectors.get(t.split()[0], [0.0, 1.0]) for t in texts], dtype=np.float32),
                        embed_query=lambda text: np.array([1.0, 0.05], dtype=np.float32))
    skills = {name: SimpleNamespace(description=f"{name} does its own thing here.", enabled=True) for name in vectors}
    assert meanings.which("notify me", skills) is None  # two stand out, neither clearly ahead
    assert meanings.standing_out("notify me", skills) == ["notify", "remind"]
    # One standing out alone is not a tie to settle: nothing is narrowed to.
    lone = Meanings(lambda texts: np.array([vectors.get(t.split()[0], [0.0, 1.0]) if not t.startswith("remind") else [0.0, 1.0]
                                            for t in texts], dtype=np.float32),
                    embed_query=lambda text: np.array([1.0, 0.05], dtype=np.float32))
    assert lone.standing_out("notify me", skills) == []
