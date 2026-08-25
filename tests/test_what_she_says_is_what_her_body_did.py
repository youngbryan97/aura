"""The commentary has to be about the moves she actually made.

Mind and voice being aware of what the body is doing in the moment is the
thing this demonstrates, and it only means something if the two cannot drift
apart. Announcing a decision announces an intention: a keystroke refused for
focus, or delivered to the wrong window, would have been described as a move
she made.

So a move is reported after the body makes it, named by the key that was
actually pressed, in the order it was pressed. A run that presses nothing
says nothing.
"""
from __future__ import annotations

import pytest

from core.skills import screen_pursuit as sp


class _Store:
    def record(self, episode):
        return "ep"

    def resolve(self, episode_id, outcome):
        pass

    def query_consequences(self, action, params=None):
        return []

    def record_outcome(self, action, context, outcome, success):
        pass


@pytest.fixture
def body(monkeypatch):
    """A body whose keypresses can be made to fail."""
    state = {"pressed": [], "said": [], "accepts": True, "text": "board 0"}

    async def read(app_name=""):
        return {"ok": True, "text": state["text"], "layout": [], "bounds": []}

    async def press(key, *, expect_app=""):
        if not state["accepts"]:
            return False
        state["pressed"].append(key)
        state["text"] = f"board {len(state['pressed'])}"
        return True

    async def frontmost(_app):
        return True

    async def identity():
        return {"url": "https://example.test/", "title": "board", "error": ""}

    def said(key, because=""):
        state["said"].append(f"Board: {str(key).capitalize()}")

    monkeypatch.setattr(sp, "read_screen", read)
    monkeypatch.setattr(sp, "press", press)
    monkeypatch.setattr(sp, "_ensure_frontmost", frontmost)
    monkeypatch.setattr(sp, "current_page_identity", identity)
    monkeypatch.setattr(sp, "_say_move", said)

    from core.agency import task_knowledge as tk

    async def no_learning(goal, **kw):
        return tk.TaskKnowledge(goal=goal)

    monkeypatch.setattr(tk, "learn_about", no_learning)
    return state


def _thinks(*replies):
    queue = list(replies)

    async def think(objective, evidence):
        return queue.pop(0) if queue else replies[-1]

    return think


@pytest.mark.asyncio
async def test_every_move_she_makes_is_reported(body):
    await sp.pursue_on_screen(
        goal="raise the number",
        success_when="never happens",
        think=_thinks("up", "left", "down"),
        max_cycles=3,
        max_seconds=10.0,
        narrate=False,
        lived=False,
        spine=_Store(),
        graph=_Store(),
    )
    assert body["pressed"], "she never moved"
    assert len(body["said"]) == len(body["pressed"])


@pytest.mark.asyncio
async def test_the_order_of_the_words_is_the_order_of_the_moves(body):
    await sp.pursue_on_screen(
        goal="raise the number",
        success_when="never happens",
        think=_thinks("up", "left", "down"),
        max_cycles=3,
        max_seconds=10.0,
        narrate=False,
        lived=False,
        spine=_Store(),
        graph=_Store(),
    )
    spoken = [line.split(": ", 1)[1].lower() for line in body["said"]]
    assert spoken == body["pressed"]


@pytest.mark.asyncio
async def test_a_keystroke_that_did_not_land_is_not_described_as_a_move(body):
    """A press refused for focus would otherwise be narrated as a move."""
    body["accepts"] = False
    await sp.pursue_on_screen(
        goal="raise the number",
        success_when="never happens",
        think=_thinks("up", "up", "up"),
        max_cycles=3,
        max_seconds=10.0,
        narrate=False,
        lived=False,
        spine=_Store(),
        graph=_Store(),
    )
    assert body["pressed"] == []
    assert body["said"] == [], "she described a move her body never made"


def test_a_move_is_reported_in_the_shape_a_person_reads():
    from core.skills.screen_pursuit import _say_move

    seen = []

    class _Workspace:
        async def publish(self, **fields):
            seen.append(fields)
            return True

    import core.container as container

    original = container.ServiceContainer.get
    container.ServiceContainer.get = staticmethod(
        lambda name, default=None: _Workspace() if name == "global_workspace" else default
    )
    try:
        import asyncio

        async def run():
            _say_move("up", "the corner holds")
            await asyncio.sleep(0)

        asyncio.run(run())
    finally:
        container.ServiceContainer.get = original

    assert seen, "the move reached nothing"
    assert seen[0]["reason"] == "Board: Up"
    assert seen[0]["payload"]["decision"]["chose"] == "Board: Up"


def test_the_line_carries_her_reasoning_not_just_the_keystroke():
    """Narrating from the body alone describes a twitch.

    What reaches the surface has to be her thinking about the choice, at the
    moment the body carried it out — mind and body in one record, or the two
    are disconnected.
    """
    import asyncio

    from core.agency.deliberate_action import ActionOption, Deliberation, Expectation
    from core.skills.screen_pursuit import _say_move

    seen = []

    class _Workspace:
        async def publish(self, **fields):
            seen.append(fields)
            return True

    import core.container as container

    original = container.ServiceContainer.get
    container.ServiceContainer.get = staticmethod(
        lambda name, default=None: _Workspace() if name == "global_workspace" else default
    )
    try:
        chosen = Deliberation(
            goal="reach 128",
            situation="a board",
            chosen=ActionOption(
                name="up",
                detail="press up",
                expectation=Expectation(describes="the big tile to stay in the corner"),
            ),
            rationale="the corner holds if I go up",
            spoke=True,
        )

        async def run():
            _say_move("up", chosen)
            await asyncio.sleep(0)

        asyncio.run(run())
    finally:
        container.ServiceContainer.get = original

    decision = seen[0]["payload"]["decision"]
    assert decision["chose"] == "Board: Up"
    assert decision["because"] == "the corner holds if I go up"
    assert decision["expected"] == "the big tile to stay in the corner"


def test_a_choice_made_without_language_says_so_in_the_same_line():
    import asyncio

    from core.agency.deliberate_action import ActionOption, Deliberation
    from core.skills.screen_pursuit import _say_move

    seen = []

    class _Workspace:
        async def publish(self, **fields):
            seen.append(fields)
            return True

    import core.container as container

    original = container.ServiceContainer.get
    container.ServiceContainer.get = staticmethod(
        lambda name, default=None: _Workspace() if name == "global_workspace" else default
    )
    try:
        wordless = Deliberation(
            goal="reach 128",
            situation="a board",
            chosen=ActionOption(name="left", detail="press left"),
            rationale="left has not been tried yet",
            spoke=False,
        )

        async def run():
            _say_move("left", wordless)
            await asyncio.sleep(0)

        asyncio.run(run())
    finally:
        container.ServiceContainer.get = original

    assert seen[0]["payload"]["decision"]["spoke"] is False


def test_the_decision_is_not_also_announced_separately():
    """Reported once, at the moment the body acts, or she says it twice."""
    import inspect

    from core.skills import screen_pursuit

    source = inspect.getsource(screen_pursuit.pursue_on_screen)
    assert "announce=False" in source


def test_a_line_reaches_both_the_bubble_and_the_conversation():
    """Two views of the same her. A commentary that only reaches one is
    invisible to whoever is looking at the other."""
    from core.agency.narrator import Narrator

    said_to_bubble = []
    published = []

    class _Presence:
        def offer_utterance(self, line, requested=False):
            said_to_bubble.append((line, requested))
            return True

    class _Orchestrator:
        def _publish_telemetry(self, payload):
            published.append(payload)

    import core.container as container
    import core.perception.ambient_presence as ap

    original_get = container.ServiceContainer.get
    original_presence = ap.get_ambient_presence
    container.ServiceContainer.get = staticmethod(
        lambda name, default=None: _Orchestrator() if name == "orchestrator" else default
    )
    ap.get_ambient_presence = lambda: _Presence()
    try:
        Narrator.say_everywhere("Board: Up — the corner holds")
    finally:
        container.ServiceContainer.get = original_get
        ap.get_ambient_presence = original_presence

    assert said_to_bubble == [("Board: Up — the corner holds", True)]
    assert published and published[0]["type"] == "aura_message"
    assert published[0]["message"] == "Board: Up — the corner holds"
    assert published[0]["metadata"]["narration"] is True


def test_the_conversation_does_not_collapse_repeated_moves():
    """Pressing the same key twice is two things that happened."""
    from pathlib import Path

    source = Path("interface/static/aura.js").read_text()
    # From the handler, not from the helper's own definition.
    handler = source.index("type === 'aura_message'")
    where = source.index("rememberMessageFingerprint(fingerprint)", handler)
    guard = source[handler:where]
    assert "meta.narration" in guard, "a stream of events is being de-duplicated as a reply"
