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

import base64
import binascii
import json
import logging
import math
import os
import subprocess
import sys
import threading
from typing import Any

from core.governance_context import GovernanceViolation
from core.runtime.flags import FlagKind, declare
from core.runtime.lockdep import checked_lock
from core.runtime.subprocess_gateway import get_subprocess_gateway
from core.verify.invariants import invariant

logger = logging.getLogger("Aura.EyesOfTheirOwn")

_EYES_IN_THIS_PROCESS = declare(
    "AURA_EYES_IN_THIS_PROCESS",
    kind=FlagKind.BOOL,
    default=False,
    description="Read windows in this process instead of in the eyes' own process",
    owner="core/perception/eyes_of_their_own.py",
)

__all__ = ["look_through_them", "stop_them", "they_are_running"]

#: How long a reading may take over there before she reads it herself, on top
#: of however long she asked it to wait for the window to settle. Several
#: times what a reading costs in a quiet process, so a busy machine does not
#: send her round the slow way for nothing.
_LONG_ENOUGH_S = 2.0

#: How long the child has to come up before she gives up on it for this run.
_TO_START_S = 20.0

#: The private wire payload is bounded before it is parsed or decoded. Picture
#: bytes are removed from the mapping before an observation reaches a caller.
_MAX_IPC_BYTES = 8 * 1024 * 1024
_PIXEL_ENVELOPE = "_captured_pixels"

_LOCK = checked_lock("perception.eyes_of_their_own")
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
        # not a failure: a child already gone is the state killing it reaches.
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
        if child.stdout is None:
            return
        try:
            line = child.stdout.readline(_MAX_IPC_BYTES + 1)
        except TypeError:
            # Older stream adapters expose only readline(); real process pipes
            # use the bounded read above. Their answers are still size checked.
            line = child.stdout.readline()
        if len(line) > _MAX_IPC_BYTES or len(line.encode("utf-8")) > _MAX_IPC_BYTES:
            return
        if line:
            try:
                decoded = json.loads(line)
                if not isinstance(decoded, dict):
                    raise ValueError("a reading is a mapping")
                answer.update(decoded)
            except (TypeError, ValueError):
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
    if bool(_EYES_IN_THIS_PROCESS.value()):
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
    # What the eyes noticed while reading, said here where it is heard. Their
    # own logging goes nowhere: the other process has no log of its own.
    for line in reading.pop("_noticed", None) or ():
        logger.info("her eyes: %s", str(line)[:300])
    if reading.get("ok") is False:
        logger.info("they could not read it (%s); reading here", reading.get("error"))
        return None
    try:
        return _restore_reading(reading)
    except (TypeError, ValueError) as why:
        logger.info("their captured picture was not usable (%s); reading here", why)
        return None


def _capture_identity(reading: dict[str, Any]) -> tuple[float, str]:
    """Only metadata supplied by the final take can identify its pixels."""
    captured = reading.get("capture_at")
    epoch = reading.get("_capture_epoch")
    if (not isinstance(captured, (int, float)) or isinstance(captured, bool)
            or not math.isfinite(captured) or captured <= 0
            or not isinstance(epoch, str) or not 1 <= len(epoch) <= 128):
        raise ValueError("the captured picture has no bounded capture identity")
    return float(captured), epoch


def _picture_to_send(picture: Any, reading: dict[str, Any]) -> dict[str, Any]:
    """An explicit private envelope for immutable pixels from the final reading."""
    from core.perception.observed_transfer import make_snapshot

    snapshot = make_snapshot(picture)
    if snapshot is None:
        raise ValueError("the captured picture is unsupported")
    captured, epoch = _capture_identity(reading)
    return {
        "version": 1,
        "height": snapshot.height,
        "width": snapshot.width,
        "channels": snapshot.channels,
        "data": base64.b64encode(snapshot.data).decode("ascii"),
        "capture_at": captured,
        "capture_epoch": epoch,
    }


def _restore_reading(reading: dict[str, Any]) -> dict[str, Any]:
    """Consume private wire pixels; public dictionary serialization keeps only words."""
    from core.perception.observed_transfer import MAX_EDGE, PixelSnapshot, ScreenReading

    public = dict(reading)
    if _PIXEL_ENVELOPE not in public:
        # Old readers can still supply words. Their missing pixels remain unknown
        # evidence and never borrow a picture from a previous capture.
        return ScreenReading(public)
    envelope = public.pop(_PIXEL_ENVELOPE)
    fields = {"version", "height", "width", "channels", "data", "capture_at", "capture_epoch"}
    if not isinstance(envelope, dict) or set(envelope) != fields:
        raise ValueError("the private picture envelope has unexpected fields")
    if type(envelope["version"]) is not int or envelope["version"] != 1:
        raise ValueError("the private picture envelope has an unsupported version")
    height, width, channels = (envelope[key] for key in ("height", "width", "channels"))
    if (any(type(value) is not int for value in (height, width, channels))
            or not 1 <= height <= MAX_EDGE or not 1 <= width <= MAX_EDGE or channels != 3):
        raise ValueError("the private picture dimensions are invalid")
    encoded = envelope["data"]
    expected_size = height * width * channels
    if not isinstance(encoded, str) or len(encoded) != 4 * ((expected_size + 2) // 3):
        raise ValueError("the private picture bytes have an invalid length")
    captured, epoch = _capture_identity(public)
    envelope_capture, envelope_epoch = _capture_identity({
        "capture_at": envelope["capture_at"], "_capture_epoch": envelope["capture_epoch"],
    })
    if captured != envelope_capture or epoch != envelope_epoch:
        raise ValueError("the private pixels belong to a different capture")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as why:
        raise ValueError("the private picture is not valid base64") from why
    snapshot = PixelSnapshot(data, height, width, channels)
    result = ScreenReading(public, picture_snapshot=snapshot)
    return result


def _reading_to_send(picture: Any, reading: dict[str, Any], *, window: Any,
                     over: Any, still: bool, wait: bool, looked_took: float) -> dict[str, Any]:
    """Finish every feature and the private evidence from one final captured frame."""
    public = dict(reading)
    public[_PIXEL_ENVELOPE] = _picture_to_send(picture, public)
    public["_shape"] = [int(picture.shape[1]), int(picture.shape[0])]
    public["_settled"] = bool(still)
    public["_looked_took"] = round(looked_took, 3)
    public["surface_id"] = f"window:{window.owner}:{window.number}"
    left, top, wide, tall = window.bounds
    bounds = [left, top, wide, tall]
    if over is not None:
        crop_left, crop_top, crop_right, crop_bottom = over
        bounds = [left + int(crop_left * wide), top + int(crop_top * tall),
                  max(1, int((crop_right - crop_left) * wide)),
                  max(1, int((crop_bottom - crop_top) * tall))]
    public["bounds"] = bounds
    if wait and (still or public.get("_still_but_unread") is True):
        from core.perception.how_a_place_looks import look_of
        from core.perception.keys_drawn_on_screen import keys_drawn
        from core.perception.shapes_that_look_pressable import pressable_shapes

        public["shapes"] = pressable_shapes(picture, apart_from=public.get("layout") or ())
        for region in public["shapes"]:
            region["look"] = look_of(picture, region)
        public["keys_drawn"] = keys_drawn(picture)
    return public


@invariant("perception.child_pixels_keep_capture_custody", scope="perception",
           owner="core/perception/eyes_of_their_own.py", observational=False)
def _child_capture_custody_invariant() -> tuple:
    import numpy as np

    reading = {"text": "visible", "capture_at": 1.0, "_capture_epoch": "one", "_settled": False}
    wire = {**reading, _PIXEL_ENVELOPE: _picture_to_send(np.zeros((2, 3, 3), dtype=np.uint8), reading)}
    restored = _restore_reading(wire)
    assert restored.picture_snapshot is not None, "a child's exact pixels were discarded"
    assert _PIXEL_ENVELOPE not in restored and "data" not in json.loads(json.dumps(restored)), "pixels escaped into public evidence"
    wire["_capture_epoch"] = "two"
    try:
        _restore_reading(wire)
    except ValueError:
        return ()
    raise AssertionError("pixels crossed capture identities")


class _WhatTheyNoticed(logging.Handler):
    """The other process's log lines, kept to go back with the next reading."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - other process
        try:
            self.lines.append(record.getMessage())
            del self.lines[:-20]
        # not a failure: a record whose message will not render is not one this keeps.
        except (TypeError, ValueError):
            return

    def taken(self) -> list[str]:
        lines, self.lines = self.lines, []
        return lines


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
    # not a failure: the module is optional here, and its absence is the answer.
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
    noticed = _WhatTheyNoticed()
    for name in ("Aura.WhatThePixelsShow", "Aura.WindowServer"):
        watched = logging.getLogger(name)
        watched.addHandler(noticed)
        watched.setLevel(logging.INFO)

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
            reading = _reading_to_send(
                picture, reading, window=window, over=over, still=still,
                wait=bool(job.get("wait_for_stillness", True)), looked_took=time.monotonic() - began,
            )
            said = noticed.taken()
            if said:
                reading["_noticed"] = said
            answer = json.dumps(reading, default=float)
            if len(answer.encode("utf-8")) > _MAX_IPC_BYTES:
                raise ValueError("the bounded reading is too large for its private pipe")
            print(answer, flush=True)
        except Exception as why:  # noqa: BLE001 - the parent reads it instead
            print(json.dumps({"ok": False, "error": f"{type(why).__name__}: {why}"}), flush=True)


if __name__ == "__main__":  # pragma: no cover - the child's own entry
    _serve()
