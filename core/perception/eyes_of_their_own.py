"""Looking at a window in a process of its own, so looking does not wait on thinking.

Taking a picture of a window and reading it costs about an eighth of a second
of work. Inside Aura's own process the same eighth of a second took nearly a
second, measured live on 2026-09-17 while she played a game: her mind, her
voice, her health checks and her background loops all want the interpreter,
and a reading is mostly Python holding it.

So the reading happens somewhere else. The child holds the reader — and with
it what it has learned to recognise in that window — and answers with the
reading. Nothing here decides anything: if the child is missing, slow or
broken, the caller reads in this process instead and is only slower.

Started as a plain subprocess rather than through multiprocessing, because a
spawned child re-imports whatever module the parent was started from, and the
module Aura is started from boots Aura.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
from typing import Any

from core.governance_context import GovernanceViolation
from core.runtime.subprocess_gateway import get_subprocess_gateway

logger = logging.getLogger("Aura.EyesOfTheirOwn")

__all__ = ["look_through_them", "stop_them", "they_are_running"]

#: How long a reading may take over there before she reads it herself, on top
#: of however long she asked it to wait for the window to settle. Several
#: times what a reading costs in a quiet process, so a busy machine does not
#: send her round the slow way for nothing.
_LONG_ENOUGH_S = 2.0

#: How long the child has to come up before she gives up on it for this run.
_TO_START_S = 20.0

_LOCK = threading.Lock()
_CHILD: Any = None
#: Set when the child could not be started, so nothing tries again every look.
_GAVE_UP = False


def _repo_root() -> str:
    here = os.path.abspath(__file__)
    return os.path.dirname(os.path.dirname(os.path.dirname(here)))


def they_are_running() -> bool:
    child = _CHILD
    return bool(child is not None and child.poll() is None)


def _start_them() -> bool:
    """Bring the child up. False when it cannot be, and then not again."""
    global _CHILD, _GAVE_UP
    if _GAVE_UP:
        return False
    if they_are_running():
        return True
    try:
        root = _repo_root()
        environment = dict(os.environ)
        environment["PYTHONPATH"] = root + os.pathsep + environment.get("PYTHONPATH", "")
        environment["AURA_EYES_IN_THIS_PROCESS"] = "1"
        _CHILD = get_subprocess_gateway().spawn(
            [sys.executable, "-m", "core.perception.eyes_of_their_own"],
            cwd=root,
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            start_new_session=False,
            read_only=True,
            source="perception.window_reader",
            accelerator_capability="none",
        )
    except (OSError, ValueError, GovernanceViolation) as why:
        _GAVE_UP = True
        logger.info("looking stays in this process: %s", why)
        return False
    return True


def _let_go(child: Any) -> None:
    """Close a child down, wherever it is called from."""
    if child is None:
        return
    try:
        if child.stdin is not None:
            child.stdin.close()
        child.wait(timeout=2.0)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        try:
            child.kill()
        except OSError:
            pass


def stop_them() -> None:
    """Let the child go. Looking carries on in this process."""
    global _CHILD
    with _LOCK:
        child, _CHILD = _CHILD, None
    _let_go(child)


def _ask(child: Any, job: dict[str, Any], wait_s: float) -> dict[str, Any] | None:
    """One question and its answer, or None when the answer did not come."""
    answer: dict[str, Any] = {}

    def read_it() -> None:
        line = child.stdout.readline() if child.stdout is not None else ""
        if line:
            try:
                answer.update(json.loads(line))
            except ValueError:
                answer["ok"] = False
                answer["error"] = "the answer was not a reading"

    child.stdin.write(json.dumps(job) + "\n")
    child.stdin.flush()
    reader = threading.Thread(target=read_it, daemon=True)
    reader.start()
    reader.join(timeout=wait_s)
    if reader.is_alive() or not answer:
        return None
    return answer


def look_through_them(
    window: Any,
    over: Any = None,
    wait_for_stillness: bool = True,
    still_within_s: float = 1.5,
) -> dict[str, Any] | None:
    """One reading of a window, taken and read in the other process.

    None when they are not available or did not answer in time, which means
    the caller should look for itself.
    """
    if os.getenv("AURA_EYES_IN_THIS_PROCESS", "").strip().lower() in {"1", "true", "yes"}:
        return None
    global _CHILD, _GAVE_UP
    with _LOCK:
        if not _start_them():
            return None
        child = _CHILD
        if child is None:
            return None
        job = {
            "number": int(getattr(window, "number", 0) or 0),
            "owner": str(getattr(window, "owner", "") or ""),
            "over": list(over) if over else None,
            "wait_for_stillness": bool(wait_for_stillness),
            "still_within_s": float(still_within_s),
        }
        waiting = _LONG_ENOUGH_S + float(still_within_s)
        if not getattr(child, "_has_answered", False):
            # The first answer includes starting an interpreter and loading
            # what it reads with.
            waiting = _TO_START_S
        try:
            reading = _ask(child, job, waiting)
        except (BrokenPipeError, OSError, ValueError) as why:
            logger.info("they could not be asked (%s); reading here", why)
            _CHILD = None
            return None
        if reading is None:
            # An answer that never came leaves the pipe holding a reading of a
            # picture that is no longer on the screen, so the child goes. Let
            # go of here rather than through stop_them, which takes this same
            # lock: a lock taken twice by one thread does not come back.
            logger.info("the reading took longer than %.1fs over there; reading here", waiting)
            _CHILD = None
            _let_go(child)
            return None
        child._has_answered = True  # noqa: SLF001 - her own handle
    if reading.get("ok") is False:
        logger.info("they could not read it (%s); reading here", reading.get("error"))
        return None
    return reading


def _stay_out_of_the_dock() -> None:  # pragma: no cover - runs in the other process
    """These eyes are part of her, not an application of their own.

    Anything here that makes the process a Cocoa application would put a
    Python rocket in the Dock while she plays. Marked as an agent first, it
    never gets an icon, as the runtime does for itself in aura_main.
    """
    if sys.platform != "darwin":
        return
    try:
        from Foundation import NSBundle
    except ImportError:
        return
    info = NSBundle.mainBundle().infoDictionary()
    if info is not None and info.get("LSUIElement") is None:
        info["LSUIElement"] = "1"


def _what_it_says(reading: dict[str, Any]) -> tuple:
    """What a reading says, for telling one reading from another."""
    from core.perception.what_the_pixels_show import what_a_reading_says  # noqa: PLC0415

    return what_a_reading_says(reading)


def _serve() -> None:  # pragma: no cover - runs in the other process
    """Read windows for whoever asks, one line of JSON at a time."""
    import time

    _stay_out_of_the_dock()

    from core.capabilities import window_server
    from core.perception import what_the_pixels_show as pixels

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            job = json.loads(line)
        except ValueError:
            continue
        try:
            number = int(job.get("number") or 0)
            owner = str(job.get("owner") or "")
            over = job.get("over")
            window = next((one for one in window_server.windows() if one.number == number), None)
            if window is None:
                print(json.dumps({"ok": False, "error": "that window is not there any more"}), flush=True)
                continue
            began = time.monotonic()

            def take(this_window: Any = window, part: Any = over) -> Any:
                picture = window_server.capture(this_window)
                if picture is None:
                    return None
                return pixels.crop_to(picture, part) if part else picture

            # A thing half moved is not a state, and pixels stopping is not
            # the same as the thing having stopped. A tile easing into place
            # moves less between two captures than compression noise, so the
            # pictures agree while it is still a few pixels short of its
            # square — and the reading puts it in the wrong one. What has to
            # stop changing is what the picture SAYS, and how each place in it
            # looks.
            picture, reading, still = pixels.settled_reading(
                take,
                pixels.looker_for(owner),
                wait=bool(job.get("wait_for_stillness", True)),
                within_s=float(job.get("still_within_s") or 1.5),
                began=began,
            )
            if picture is None or reading is None:
                print(json.dumps({"ok": False, "error": "no picture of that window"}), flush=True)
                continue
            reading["_shape"] = [int(picture.shape[1]), int(picture.shape[0])]
            reading["_settled"] = bool(still)
            reading["_looked_took"] = round(time.monotonic() - began, 3)
            print(json.dumps(reading, default=float), flush=True)
        except Exception as why:  # noqa: BLE001 - the parent reads it instead
            print(json.dumps({"ok": False, "error": f"{type(why).__name__}: {why}"}), flush=True)


if __name__ == "__main__":  # pragma: no cover - the child's own entry
    _serve()
