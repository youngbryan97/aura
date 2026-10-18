"""Her eyes for play in a process of their own: what moves in the picture is worked out where nothing else wants the
interpreter.

Working out what moves in a picture of play (core/perception/what_moves_in_the_picture.py) is most of what a picture
of play costs: measured offline 2026-10-12, 44 of every 50 milliseconds, nearly all of it Python. In Aura's own
process the interpreter is shared with everything else she is doing at the time (her voice, her senses, her
background thought), and only one of them holds it at once: measured live 2026-10-13, her play waited for it a fifth
of the time, and played at 7 to 9 pictures a second where it played at 20 alone. A person's eyes and hands do not
wait for the rest of them to finish a thought.

So the eyes for play are opened in a process of their own, as her eyes for reading windows already are
(core/perception/eyes_of_their_own.py), and started the same way, as a plain subprocess (a spawned one re-imports
whatever module Aura was started from, and that boots Aura). The play loop is handed something that stands in for
them (``SeenElsewhere``): each picture goes over through shared memory, the other process sees it with the same
eyes, and what it saw comes back, with a likeness of what the eyes hold that everyone in this process reads as they
read the eyes themselves: the things, their kinds, the screen's shape. A thing seen in two pictures is one object
here as it is there: its likeness is brought up to date in place, so whoever kept it holds it still.

Nothing is seen differently: the same eyes see the same pictures in the same order. Where the other process cannot
be started, does not answer in time, or fails, the eyes are opened in this process from what had been seen, and play
goes on, only slower. Nothing here decides anything.
"""
from __future__ import annotations

import logging
import os
import pickle
import select
import struct
import subprocess
import sys
import threading
import uuid
import weakref
from typing import Any

import numpy as np

from core.runtime.flags import FlagKind, declare

logger = logging.getLogger("Aura.SeeingPlayElsewhere")

__all__ = ["SeenElsewhere", "eyes_for_play", "short_turns_while_playing", "stop_them"]

_IN_THIS_PROCESS = declare(
    "AURA_PLAY_EYES_IN_THIS_PROCESS",
    kind=FlagKind.BOOL,
    default=False,
    description="Work out what moves in pictures of play in this process instead of in the eyes' own process",
    owner="core/perception/seeing_play_elsewhere.py",
)

#: How long a picture may take over there before the eyes are opened here instead, in seconds; and how long the
#: other process has to come up and answer first.
LONG_ENOUGH_S = 3.0
TO_START_S = 30.0

#: How many things are remembered here by number, the earliest let go first.
KNOWN_AT_MOST = 4096
#: What the eyes hold that no one outside them reads, and that is large: left over there.
_KEPT_THERE = frozenset({"points", "activity", "out_of_sight", "_drawn", "_still", "_first", "_was", "_point_motion",
                         "_backdrop"})
#: What stands for a backdrop here: that there is one is all anyone here asks of it.
_A_BACKDROP = np.zeros((1, 1, 3), np.uint8)

#: How large the memory the other process writes its answers into is, in bytes; an answer larger comes down the pipe.
ANSWERS_AT_MOST = 16 * 1024 * 1024
_ANSWERS: Any = None
_LOCK = threading.RLock()
_CHILD: Any = None
#: Eyes no longer wanted here, told to the other process with the next thing asked of it: a likeness can be let go
#: at any moment, even in the middle of a message, and nothing may be written into one.
_ENDED: list[str] = []
_GAVE_UP = False
_MEMORY: Any = None


def _repo_root() -> str:
    here = os.path.abspath(__file__)
    return os.path.dirname(os.path.dirname(os.path.dirname(here)))


# -- the wire: a length, then what was said ------------------------------------------------------------------------


def _send(stream: Any, message: Any) -> None:
    data = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
    stream.write(struct.pack("!I", len(data)))
    stream.write(data)
    stream.flush()


def _read_exactly(stream: Any, size: int, wait_s: float | None) -> bytes | None:
    chunks, got = [], 0
    fd = stream.fileno() if wait_s is not None else None
    while got < size:
        if fd is not None and not select.select([fd], [], [], wait_s)[0]:
            return None
        chunk = os.read(fd, size - got) if fd is not None else stream.read(size - got)
        if not chunk:
            return None
        chunks.append(chunk)
        got += len(chunk)
    return b"".join(chunks)


def _receive(stream: Any, wait_s: float | None = None) -> Any:
    head = _read_exactly(stream, 4, wait_s)
    if head is None:
        return None
    body = _read_exactly(stream, struct.unpack("!I", head)[0], wait_s)
    return None if body is None else pickle.loads(body)  # noqa: S301 - from her own child, over its own pipe


# -- this side -----------------------------------------------------------------------------------------------------


def _start_them() -> Any:
    """The other process, started if it is not running; None where it cannot be, and then not again."""
    global _CHILD, _GAVE_UP
    if _GAVE_UP:
        return None
    if _CHILD is not None and _CHILD.poll() is None:
        return _CHILD
    try:
        from core.runtime.subprocess_gateway import get_subprocess_gateway

        root = _repo_root()
        environment = dict(os.environ)
        environment["PYTHONPATH"] = root + os.pathsep + environment.get("PYTHONPATH", "")
        environment["AURA_PLAY_EYES_IN_THIS_PROCESS"] = "1"
        environment["AURA_EYES_IN_THIS_PROCESS"] = "1"
        _CHILD = get_subprocess_gateway().spawn(
            [sys.executable, "-m", "core.perception.seeing_play_elsewhere"],
            cwd=root, env=environment, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=False, start_new_session=False, read_only=True, source="perception.play_eyes",
            accelerator_capability="none",
        )
        _CHILD.answered = False
    except Exception as why:  # noqa: BLE001 - any refusal or failure to start leaves the eyes in this process
        _GAVE_UP = True
        logger.info("the eyes for play stay in this process: %s", why)
        return None
    return _CHILD


def _ask(message: Any) -> Any:
    """One message to the other process and its answer; None where there is no answer in time."""
    child = _start_them()
    if child is None:
        return None
    try:
        while _ENDED:
            _send(child.stdin, ("end", _ENDED.pop()))
        _send(child.stdin, message)
        answer = _receive(child.stdout, LONG_ENOUGH_S if child.answered else TO_START_S)
    except (OSError, ValueError, EOFError, pickle.UnpicklingError) as why:
        logger.info("the eyes for play did not answer: %s", why)
        answer = None
    if answer is None:
        stop_them()
        return None
    child.answered = True
    return answer


def _memory_for(size: int) -> Any:
    """Shared memory at least ``size`` bytes long, kept between pictures, and given back when this process ends."""
    global _MEMORY
    import atexit
    from multiprocessing import shared_memory

    if _MEMORY is None or _MEMORY.size < size:
        _let_the_memory_go()
        _MEMORY = shared_memory.SharedMemory(create=True, size=max(size, 4 * 1024 * 1024))
        atexit.register(_let_the_memory_go)
    return _MEMORY


def _let_the_memory_go() -> None:
    global _MEMORY
    memory, _MEMORY = _MEMORY, None
    if memory is None:
        return
    try:
        memory.close()
        memory.unlink()
    except (OSError, FileNotFoundError):
        pass


def _answers_memory() -> Any:
    """The memory the other process writes what it saw into: read here in one go, where an answer down a pipe comes in
    pieces, and each piece waits its turn for the interpreter among whatever else is running here."""
    global _ANSWERS
    import atexit
    from multiprocessing import shared_memory

    if _ANSWERS is None:
        _ANSWERS = shared_memory.SharedMemory(create=True, size=ANSWERS_AT_MOST)
        atexit.register(_let_the_answers_go)
    return _ANSWERS


def _let_the_answers_go() -> None:
    global _ANSWERS
    memory, _ANSWERS = _ANSWERS, None
    if memory is not None:
        try:
            memory.close()
            memory.unlink()
        except (OSError, FileNotFoundError):
            pass


def stop_them() -> None:
    """Let the other process go, and then the memory it was handed pictures in; the next eyes for play are opened in
    this process until it is asked for again."""
    global _CHILD
    child, _CHILD = _CHILD, None
    try:
        if child is not None:
            if child.stdin is not None:
                child.stdin.close()
            child.wait(timeout=2.0)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        try:
            child.kill()
        except OSError:
            pass
    finally:
        _let_the_memory_go()
        _let_the_answers_go()


def _brought_up_to_date(likeness: Any, state: dict[str, Any]) -> None:
    """The likeness of the eyes made what they now hold, each thing kept as the object it was where it is still seen."""
    # A thing's number is never given to another: a thing gone out of sight, or on a screen that went and came back,
    # is the same thing when it is seen again, and here the same object, as it is to the eyes themselves.
    known = likeness.__dict__.setdefault("_known", {})
    for name in ("things", "pending"):
        fresh = state.pop(name, None)
        if fresh is None:
            continue
        kept = {}
        for number, thing in fresh.items():
            before = known.get(number)
            if before is not None and type(before) is type(thing) and hasattr(before, "__dict__"):
                before.__dict__.update(thing.__dict__)
                kept[number] = before
            else:
                kept[number] = known[number] = thing
        likeness.__dict__[name] = kept
    while len(known) > KNOWN_AT_MOST:
        del known[next(iter(known))]
    kinds = state.pop("kinds", None)
    if kinds is not None:
        held = likeness.__dict__.get("kinds") or []
        for i, kind in enumerate(kinds):
            if i < len(held) and type(held[i]) is type(kind) and hasattr(held[i], "__dict__"):
                held[i].__dict__.update(kind.__dict__)
                kinds[i] = held[i]
        likeness.__dict__["kinds"] = kinds
    if state.pop("_has_a_backdrop", False):
        state["_backdrop"] = _A_BACKDROP
    else:
        state["_backdrop"] = None
    likeness.__dict__.update(state)


class SeenElsewhere:
    """Stands in for her eyes for play (``WhatMoves``): each picture is seen in the other process, and what the eyes
    hold is kept here as a likeness. Where that cannot be done, the eyes are opened here, from the likeness."""

    def __init__(self, kinds: list[Any] | None = None) -> None:
        from core.perception.what_moves_in_the_picture import WhatMoves

        self._name = uuid.uuid4().hex
        self._likeness = WhatMoves.__new__(WhatMoves)
        self._here: Any = None
        with _LOCK:
            answer = _ask(("begin", self._name, list(kinds or [])))
        if not isinstance(answer, tuple) or answer[0] != "ready":
            self._here = WhatMoves(kinds=kinds)
            return
        _brought_up_to_date(self._likeness, answer[1])
        weakref.finalize(self, _done_with, self._name)

    @property
    def elsewhere(self) -> bool:
        return self._here is None

    def see(self, picture: np.ndarray, at: float, *, search_hint: Any = None,
            material_retention: Any = None) -> list[dict[str, Any]]:
        kwargs = {'search_hint': search_hint} if search_hint is not None else {}
        if material_retention is not None:
            kwargs['material_retention'] = material_retention
        if self._here is not None:
            return self._here.see(picture, at, **kwargs)
        pixels = np.asarray(picture)
        with _LOCK:
            memory, answers = _memory_for(pixels.nbytes), _answers_memory()
            np.ndarray(pixels.shape, pixels.dtype, buffer=memory.buf)[...] = pixels
            message = ("see", self._name, memory.name, pixels.shape, pixels.dtype.str, at,
                       getattr(picture, "drawing_scene", None), answers.name, answers.size)
            payload = (*message, search_hint, material_retention) if material_retention is not None else (*message, search_hint) if search_hint is not None else message
            answer = _ask(payload)
            if isinstance(answer, tuple) and answer[0] == "written":
                answer = pickle.loads(answers.buf[:answer[1]])  # noqa: S301 - her own child's answer
        if isinstance(answer, tuple) and answer[0] == "seen":
            _brought_up_to_date(self._likeness, answer[2])
            return answer[1]
        why = answer[1] if isinstance(answer, tuple) and len(answer) > 1 else "no answer in time"
        logger.info("the eyes for play are opened here, from what was seen there: %s", why)
        self._here = _eyes_from(self._likeness)
        return self._here.see(picture, at, **kwargs)

    def __getattr__(self, name: str) -> Any:
        holder = self.__dict__.get("_here") or self.__dict__.get("_likeness")
        if holder is None:
            raise AttributeError(name)
        return getattr(holder, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._here if self._here is not None else self._likeness, name, value)


def _eyes_from(likeness: Any) -> Any:
    """Eyes opened here, knowing the kinds the other eyes knew: what moves is found afresh, as on a new screen."""
    from core.perception.what_moves_in_the_picture import WhatMoves

    return WhatMoves(kinds=list(getattr(likeness, "kinds", []) or []))


def _done_with(name: str) -> None:
    _ENDED.append(name)


#: The share of a moment's work in this process spent waiting for the interpreter, from which the eyes for play are
#: better in a process of their own; and how long a moment of work it is measured over, in seconds of her own work.
CONTENDED = 0.15
MEASURED_OVER_S = 0.02


def kept_waiting() -> float:
    """The share of a moment of work in this process spent waiting for the interpreter: a little pure work done, and
    the time it took against the time it was being done. Nothing else running, nought; others holding it, more."""
    import time

    began, worked = time.perf_counter(), time.thread_time()
    count = 0
    while time.thread_time() - worked < MEASURED_OVER_S:
        count += 1
    took = time.perf_counter() - began
    return max(0.0, 1.0 - (time.thread_time() - worked) / took) if took > 0 else 0.0


def separate_classical_eyes_enabled() -> bool:
    """An explicit local-observation setting also excludes transport preparation."""
    return not bool(_IN_THIS_PROCESS.value())


def eyes_for_play(kinds: list[Any] | None = None) -> Any:
    """Her eyes for a stretch of play: in a process of their own where this one keeps her waiting for the interpreter,
    else here. Seen elsewhere, a picture costs a few milliseconds more in passing to and fro, and that is all it costs
    where nothing else wants the interpreter; where other work does, seeing here costs many times over."""
    from core.perception.learned_tracking.selection import (
        learned_points_requested,
        selected_motion_eyes,
    )
    from core.perception.what_moves_in_the_picture import WhatMoves

    if learned_points_requested():
        # Numerical GPU state belongs to the owned observation worker. The
        # existing child declares no accelerator, so it cannot host this model.
        return selected_motion_eyes(kinds=kinds)
    if not separate_classical_eyes_enabled():
        return WhatMoves(kinds=kinds)
    waiting = kept_waiting()
    if waiting < CONTENDED:
        logger.info("the eyes for play are opened here: this process kept her waiting %.0f%% of a moment", 100 * waiting)
        return WhatMoves(kinds=kinds)
    logger.info("the eyes for play are opened elsewhere: this process kept her waiting %.0f%% of a moment", 100 * waiting)
    return SeenElsewhere(kinds=kinds)


# -- the interpreter's turns ---------------------------------------------------------------------------------------

#: How long one thread may hold the interpreter while another waits, in seconds, while she plays as it happens.
TURN_WHILE_PLAYING_S = 0.001
_TURNS: dict[str, Any] = {"playing": 0, "before": None}
_TURNS_LOCK = threading.Lock()


class short_turns_while_playing:  # noqa: N801 - read as what it does, in a with
    """While she plays as it happens, the interpreter changes hands every millisecond instead of every five, and as it
    was after. Whatever else she is doing goes on, in shorter turns; her play, which hands the interpreter back between
    every picture and every press, gets it back sooner each time. Measured offline 2026-10-13 with three other threads
    busy beside play: 99 ms a picture with turns of five milliseconds, 46 with turns of one, 41 alone at most."""

    def __enter__(self) -> short_turns_while_playing:
        with _TURNS_LOCK:
            if _TURNS["playing"] == 0:
                _TURNS["before"] = sys.getswitchinterval()
                sys.setswitchinterval(min(TURN_WHILE_PLAYING_S, _TURNS["before"]))
            _TURNS["playing"] += 1
        return self

    def __exit__(self, *_: Any) -> None:
        with _TURNS_LOCK:
            _TURNS["playing"] = max(0, _TURNS["playing"] - 1)
            if _TURNS["playing"] == 0 and _TURNS["before"] is not None:
                sys.setswitchinterval(_TURNS["before"])
                _TURNS["before"] = None


# -- that side -----------------------------------------------------------------------------------------------------


def _state_of(eyes: Any) -> dict[str, Any]:
    state = {name: value for name, value in vars(eyes).items() if name not in _KEPT_THERE}
    state["_has_a_backdrop"] = getattr(eyes, "_backdrop", None) is not None
    return state


def _serve() -> None:
    """The other process: eyes for each stretch of play it is asked to see for, until its input closes."""
    from multiprocessing import resource_tracker, shared_memory

    from core.perception.what_moves_in_the_picture import WhatMoves, prepare_play_vision

    wire_in, wire_out = sys.stdin.buffer, sys.stdout.buffer
    sys.stdout = sys.stderr                                   # nothing printed may go down the wire
    eyes: dict[str, Any] = {}
    memories: dict[str, Any] = {}
    while True:
        message = _receive(wire_in)
        if message is None:
            return
        kind, name = message[0], message[1]
        try:
            if kind == "begin":
                eyes[name] = WhatMoves(kinds=message[2])
                prepare_play_vision()
                _send(wire_out, ("ready", _state_of(eyes[name])))
            elif kind == "see":
                _, _, memory_name, shape, dtype, at, scene, answers_name, answers_size = message[:9]
                search_hint = message[9] if len(message) >= 10 else None
                material_retention = message[10] if len(message) >= 11 else None
                memory = memories.get(memory_name)
                if memory is None:
                    memory = memories[memory_name] = shared_memory.SharedMemory(name=memory_name)
                    # Attached, not owned: this process must not unlink it when it ends.
                    resource_tracker.unregister(memory._name, "shared_memory")  # noqa: SLF001
                picture = np.ndarray(shape, np.dtype(dtype), buffer=memory.buf).copy()
                if scene is not None:
                    from core.perception.the_drawing_as_objects import described

                    picture = described(picture, scene)
                kwargs = {'search_hint': search_hint} if search_hint is not None else {}
                if material_retention is not None:
                    kwargs['material_retention'] = material_retention
                happened = eyes[name].see(picture, at, **kwargs)
                data = pickle.dumps(("seen", happened, _state_of(eyes[name])), protocol=pickle.HIGHEST_PROTOCOL)
                answers = memories.get(answers_name)
                if answers is None:
                    answers = memories[answers_name] = shared_memory.SharedMemory(name=answers_name)
                    resource_tracker.unregister(answers._name, "shared_memory")  # noqa: SLF001
                if len(data) <= answers_size:
                    answers.buf[:len(data)] = data
                    _send(wire_out, ("written", len(data)))
                else:
                    wire_out.write(struct.pack("!I", len(data)))
                    wire_out.write(data)
                    wire_out.flush()
            elif kind == "end":
                eyes.pop(name, None)
        except Exception as why:  # noqa: BLE001 - any failure is told, and the eyes are opened on the other side
            _send(wire_out, ("failed", f"{type(why).__name__}: {str(why)[:300]}"))


if __name__ == "__main__":
    _serve()
