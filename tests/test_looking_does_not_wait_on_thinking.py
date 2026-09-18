"""A reading is taken in a process of its own, and nothing depends on it.

Capturing a window and reading it is about an eighth of a second of work. In
Aura's own process the same work took nearly a second, measured live on
2026-09-17 while she played a game: everything else she is doing wants the
interpreter and a reading is mostly Python holding it.

So it happens elsewhere, and every way that can fail — no child, a slow
child, a child that answers with a refusal — leaves her reading it herself.
"""

from __future__ import annotations

import json

import pytest

from core.perception import eyes_of_their_own as eyes


class _Window:
    number = 11
    owner = "Some Game"
    title = "Some Game"


class _Child:
    """A child that answers with whatever it was given, as the real one does."""

    def __init__(self, answers: list[str], *, alive: bool = True) -> None:
        self.answers = list(answers)
        self.asked: list[dict] = []
        self._alive = alive
        self.stdin = self
        self.stdout = self

    # stdin
    def write(self, line: str) -> None:
        self.asked.append(json.loads(line))

    def flush(self) -> None:
        return None

    def close(self) -> None:
        self._alive = False

    # stdout
    def readline(self) -> str:
        return self.answers.pop(0) if self.answers else ""

    def poll(self):
        return None if self._alive else 0

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self._alive = False


@pytest.fixture(autouse=True)
def _no_child_left_behind(monkeypatch):
    monkeypatch.delenv("AURA_EYES_IN_THIS_PROCESS", raising=False)
    monkeypatch.setattr(eyes, "_CHILD", None, raising=False)
    monkeypatch.setattr(eyes, "_GAVE_UP", False, raising=False)
    yield
    monkeypatch.setattr(eyes, "_CHILD", None, raising=False)


def _with(monkeypatch, child: _Child) -> None:
    monkeypatch.setattr(eyes, "_CHILD", child, raising=False)
    monkeypatch.setattr(eyes, "_start_them", lambda: True)


def test_a_reading_comes_back_as_the_reading_it_is(monkeypatch):
    said = {"ok": True, "text": "2 4", "layout": [], "grids": [{"rows": 4, "columns": 4}],
            "_shape": [738, 870], "_settled": True}
    child = _Child([json.dumps(said) + "\n"])
    _with(monkeypatch, child)
    reading = eyes.look_through_them(_Window(), None, True, 1.5)
    assert reading is not None and reading["grids"][0]["rows"] == 4
    assert child.asked[0]["number"] == 11
    assert child.asked[0]["wait_for_stillness"] is True


def test_a_child_that_cannot_read_it_sends_her_back_to_reading_it_herself(monkeypatch):
    child = _Child([json.dumps({"ok": False, "error": "that window is not there any more"}) + "\n"])
    _with(monkeypatch, child)
    assert eyes.look_through_them(_Window()) is None


def test_a_child_that_does_not_answer_is_let_go(monkeypatch):
    child = _Child([])  # readline returns "" and the answer never arrives
    _with(monkeypatch, child)
    monkeypatch.setattr(eyes, "_LONG_ENOUGH_S", 0.05)
    monkeypatch.setattr(eyes, "_TO_START_S", 0.05)
    assert eyes.look_through_them(_Window(), None, False, 0.0) is None


def test_told_to_look_here_she_looks_here(monkeypatch):
    monkeypatch.setenv("AURA_EYES_IN_THIS_PROCESS", "1")
    assert eyes.look_through_them(_Window()) is None


def test_a_child_that_will_not_start_is_not_started_again(monkeypatch):
    tries = {"n": 0}

    def refuses(*_a, **_k):
        tries["n"] += 1
        raise OSError("no")

    monkeypatch.setattr(eyes.subprocess, "Popen", refuses)
    assert eyes.look_through_them(_Window()) is None
    assert eyes.look_through_them(_Window()) is None
    assert tries["n"] == 1


def test_the_child_is_its_own_module_not_the_one_aura_boots():
    """A spawned child re-imports the module its parent was started from."""
    import inspect

    source = inspect.getsource(eyes)
    assert "core.perception.eyes_of_their_own" in source
    assert "get_subprocess_gateway().spawn" in source
    # Nothing here starts a child the way multiprocessing does, which would
    # re-import the module Aura is started from.
    assert "import multiprocessing" not in source
    assert "mp.get_context" not in source


def test_reader_uses_registered_non_model_process_owner(monkeypatch):
    from types import SimpleNamespace
    calls = []

    def spawn(argv, **options):
        calls.append((argv, options))
        return _Child([])

    monkeypatch.setattr(eyes, "get_subprocess_gateway", lambda: SimpleNamespace(spawn=spawn))
    assert eyes._start_them()
    argv, options = calls[0]
    assert argv == [eyes.sys.executable, "-m", "core.perception.eyes_of_their_own"]
    assert options["read_only"] and options["accelerator_capability"] == "none"
    assert options["source"] == "perception.window_reader"
    assert options["env"]["AURA_EYES_IN_THIS_PROCESS"] == "1"
