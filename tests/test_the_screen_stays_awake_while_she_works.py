"""A machine she is working on does not lock itself under her.

She types with keys addressed to a process and looks with a picture of a
window, so from the outside nobody has touched the machine and its idle timer
runs. LIVE 2026-09-17: the screen locked on move 226 of a game she had been
asked to play and was winning, and the run ended with nothing on screen
offering a move.

Held while she works and let go when she stops, so a machine she is not using
locks itself as it should. A person locking their own screen still locks it.
"""

from __future__ import annotations

import pytest

from core.capabilities import keeping_the_screen_awake as awake


class _Holder:
    def __init__(self) -> None:
        self.alive = True
        self.let_go = False

    def poll(self):
        return None if self.alive else 0

    def terminate(self):
        self.let_go = True
        self.alive = False

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self.alive = False


@pytest.fixture(autouse=True)
def _nothing_held(monkeypatch):
    monkeypatch.setattr(awake, "_HOLDING", None, raising=False)
    monkeypatch.setattr(awake, "_RELYING", 0, raising=False)
    yield
    monkeypatch.setattr(awake, "_HOLDING", None, raising=False)
    monkeypatch.setattr(awake, "_RELYING", 0, raising=False)


def test_it_is_held_while_she_works_and_let_go_after(monkeypatch):
    holders: list[_Holder] = []

    def hold(*_a, **_k):
        holders.append(_Holder())
        return holders[-1]

    monkeypatch.setattr(awake, "_take_hold", hold)
    assert awake.it_is_being_kept_awake() is False
    with awake.keeping_it_awake("playing a game") as held:
        assert held is True
        assert awake.it_is_being_kept_awake() is True
    assert awake.it_is_being_kept_awake() is False
    assert holders[0].let_go is True


def test_two_things_working_share_one_hold(monkeypatch):
    holders: list[_Holder] = []
    monkeypatch.setattr(awake, "_take_hold", lambda *_a, **_k: holders.append(_Holder()) or holders[-1])
    with awake.keeping_it_awake("one"):
        with awake.keeping_it_awake("two"):
            assert awake.it_is_being_kept_awake() is True
        # The inner one finishing does not let the display sleep under the outer.
        assert awake.it_is_being_kept_awake() is True
    assert awake.it_is_being_kept_awake() is False
    assert len(holders) == 1


def test_a_machine_that_will_not_hold_it_is_worked_on_anyway(monkeypatch):
    monkeypatch.setattr(awake, "_take_hold", lambda *_a, **_k: None)
    with awake.keeping_it_awake("playing a game") as held:
        assert held is False


def test_the_hold_is_tied_to_this_process():
    """A hold that outlives her is a machine that never sleeps again."""
    import inspect

    source = inspect.getsource(awake._take_hold)
    assert "-w" in source and "os.getpid()" in source


@pytest.mark.asyncio
async def test_a_run_on_screen_holds_it(monkeypatch):
    from screen_pursuit_support import patch_pursuit

    from core.skills import screen_pursuit as sp

    held: list[str] = []

    class _Awake:
        def __init__(self, why: str) -> None:
            self.why = why

        def __enter__(self):
            held.append(self.why)
            return True

        def __exit__(self, *_a):
            held.append("let go")
            return False

    monkeypatch.setattr(awake, "keeping_it_awake", lambda why="": _Awake(why))

    async def read(app_name="", over=None):
        return {"ok": True, "text": "nothing", "layout": [], "grids": []}

    async def yes(*_a, **_k):
        return True

    async def identity():
        return {"url": "", "title": "", "error": ""}

    async def thinks(*_a, **_k):
        return "I will press left."

    patch_pursuit(monkeypatch, "read_screen", read)
    patch_pursuit(monkeypatch, "press", yes)
    patch_pursuit(monkeypatch, "_ensure_frontmost", yes)
    patch_pursuit(monkeypatch, "current_page_identity", identity)
    patch_pursuit(monkeypatch, "_bring_the_thing_back_to_the_front", yes, raising=False)
    await sp.pursue_on_screen(
        goal="do something on screen", success_when="never happens", think=thinks,
        target_app="Some Game", max_cycles=1, max_seconds=10.0,
        narrate=False, lived=False, research=False,
    )
    assert held and "Some Game" in held[0]
    assert held[-1] == "let go"
