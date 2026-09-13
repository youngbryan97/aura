"""core/senses/sight.py — looking, on purpose, when she is asked to.

There is already a camera path in this runtime and it is not this one. The
interaction-signal lane samples a 320×240 frame every few seconds to know
whether somebody is in front of the machine; it exists to inform presence,
and it is deliberately low-resolution and low-rate because that is all
presence needs. Asking it "how many fingers am I holding up" would answer
from whichever thumbnail happened to be sitting in a rolling buffer, possibly
seconds stale, at a resolution where fingers are a smudge.

Answering a question about *right now* requires looking *right now*. That is
the whole of this module: a request/response round trip to whichever surface
owns a camera, at a resolution a model can actually read, bounded so a
missing or slow client is a clean failure rather than a hung turn.

Three things it refuses to do, each because the alternative is a plausible
sentence with nothing behind it:

**It does not answer from a stale frame.** A capture has a deadline. If no
fresh frame arrives, the failure is recorded and she says she could not look
— which is true, and far better than describing a moment that has passed.

**It does not describe an image to a model that cannot see.** The configured
`vision_model` in `core/config.py` is the text cortex, which cannot read an
image at all; handing it one produces confident fiction. Sight here goes to
`MLXVisionClient` and its genuinely multimodal Qwen2-VL, or it does not
happen.

**It does not pretend the camera is on when it is off.** Camera privacy is a
setting the user controls, and a refusal to look because they turned the
camera off is a fact about their choice, not a malfunction — so it is
reported as exactly that, and she can offer to turn it on rather than
silently failing.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import time
import uuid
from dataclasses import dataclass, field

from core.runtime.errors import record_degradation
from core.runtime.lockdep import LockRank, checked_async_lock

logger = logging.getLogger("Aura.Senses.Sight")

# Successful camera interpretation is perceptual evidence, not a tool
# receipt. Retain only its monotonic age; no pixels or descriptions live here.
_LAST_CAMERA_OBSERVATION_AT = 0.0


def _note_camera_observation() -> None:
    global _LAST_CAMERA_OBSERVATION_AT
    _LAST_CAMERA_OBSERVATION_AT = time.monotonic()


def camera_observation_age_seconds() -> float | None:
    """Seconds since a frame was successfully interpreted, or None."""
    if _LAST_CAMERA_OBSERVATION_AT <= 0.0:
        return None
    return max(0.0, time.monotonic() - _LAST_CAMERA_OBSERVATION_AT)


def _record_canonical_camera_observation(observation: str) -> None:
    """Converge one-shot sight with the runtime's canonical sense state."""

    try:
        from core.runtime.service_registry import get_runtime_service

        engine = get_runtime_service("interaction_signals", default=None)
        recorder = getattr(engine, "record_explicit_vision_observation", None)
        if callable(recorder):
            recorder(observation, observed_at=time.time())
    except _SIGHT_ERRORS as exc:
        record_degradation(
            "senses.sight.canonical_state",
            exc,
            action="kept the successful one-shot observation in turn context",
        )

# How long to wait for a surface to hand back a frame. A browser that has the
# camera warm answers in tens of milliseconds; one that has to start the
# device takes noticeably longer, and past this the honest answer is that she
# could not look in time.
CAPTURE_TIMEOUT_S = 6.0

# What a model needs versus what presence needs. The signal lane's 320×240 is
# plenty for "is somebody there" and useless for counting fingers, reading a
# label, or telling a mug from a glass.
CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720

# A JPEG frame ceiling. Large enough for the resolution above at good quality,
# small enough that a malicious or broken client cannot post a payload that
# matters.
MAX_FRAME_BYTES = 3 * 1024 * 1024

_SIGHT_ERRORS = (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError)


@dataclass(slots=True)
class Frame:
    """One captured moment."""

    data: bytes
    mime_type: str = "image/jpeg"
    width: int = 0
    height: int = 0
    captured_at: float = field(default_factory=lambda: time.time())

    @property
    def age_s(self) -> float:
        return max(0.0, time.time() - self.captured_at)


@dataclass(slots=True)
class Look:
    """What she got when she tried to look, and why if she did not."""

    ok: bool
    answer: str = ""
    frame: Frame | None = None
    cause: str = ""
    detail: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "answer": self.answer,
            "cause": self.cause,
            "detail": self.detail,
            "frame_age_s": round(self.frame.age_s, 2) if self.frame else None,
        }


class CaptureBroker:
    """Round-trips a capture request to whichever surface owns a camera.

    The server cannot reach the webcam directly — the camera belongs to the
    browser, behind a permission the user granted to a page. So a look is a
    request published to the surface and a frame posted back, correlated by
    id. One pending request per id, with a deadline, and no unbounded waiting
    anywhere: a surface that never answers must cost one timed-out turn, not
    a wedged session.
    """

    def __init__(self) -> None:
        self._pending: dict[str, asyncio.Future[Frame]] = {}
        self._one_shot_authorized: set[str] = set()
        # Checked rather than raw: lockdep only sees the locks it wraps, and
        # a lock it cannot see is one that cannot be ordered against the rest
        # — which is how an ABBA deadlock stays invisible until it happens on
        # the live runtime. LEAF because nothing is acquired underneath it.
        self._lock = checked_async_lock("sight.capture_broker", rank=LockRank.LEAF)

    async def request_frame(
        self,
        *,
        timeout_s: float = CAPTURE_TIMEOUT_S,
        explicit_user_consent: bool = False,
    ) -> Frame | None:
        """Ask the surface for a fresh frame. None if none arrived in time."""
        request_id = uuid.uuid4().hex[:16]
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Frame] = loop.create_future()
        async with self._lock:
            self._pending[request_id] = future
            if explicit_user_consent:
                self._one_shot_authorized.add(request_id)

        try:
            if not await self._publish_request(
                request_id,
                explicit_user_consent=explicit_user_consent,
            ):
                return None
            return await asyncio.wait_for(future, timeout=max(0.5, float(timeout_s)))
        except TimeoutError:
            logger.info("capture %s: no frame within %.1fs", request_id, timeout_s)
            return None
        except asyncio.CancelledError:
            raise
        finally:
            async with self._lock:
                self._pending.pop(request_id, None)
                self._one_shot_authorized.discard(request_id)

    async def _publish_request(
        self,
        request_id: str,
        *,
        explicit_user_consent: bool = False,
    ) -> bool:
        """Tell the surface to capture. False when there is no surface."""
        try:
            from core.container import ServiceContainer

            orchestrator = ServiceContainer.get("orchestrator", default=None)
            publish = getattr(orchestrator, "_publish_telemetry", None)
            if publish is None:
                return False
            publish(
                {
                    "type": "camera_capture_request",
                    "request_id": request_id,
                    "width": CAPTURE_WIDTH,
                    "height": CAPTURE_HEIGHT,
                    # This permits one correlated frame only. It does not turn
                    # on ambient camera sensing or mutate the persisted switch.
                    "one_shot_authorized": bool(explicit_user_consent),
                }
            )
            return True
        except _SIGHT_ERRORS as exc:
            record_degradation(
                "senses.sight",
                exc,
                action="could not ask a surface for a camera frame",
                severity="warning",
            )
            return False

    async def one_shot_authorized(self, request_id: str) -> bool:
        """Whether this pending request carries direct per-turn consent."""
        async with self._lock:
            return str(request_id or "") in self._one_shot_authorized

    async def deliver(self, request_id: str, frame: Frame) -> bool:
        """Hand a captured frame to whoever asked for it."""
        async with self._lock:
            future = self._pending.pop(str(request_id or ""), None)
            self._one_shot_authorized.discard(str(request_id or ""))
        if future is None or future.done():
            # Late or unsolicited. Dropping it is correct: the turn that
            # wanted it has already given up and said so.
            return False
        future.set_result(frame)
        return True


_BROKER: CaptureBroker | None = None


def get_capture_broker() -> CaptureBroker:
    global _BROKER
    if _BROKER is None:
        _BROKER = CaptureBroker()
    return _BROKER


def reset_capture_broker_for_test() -> None:
    global _BROKER
    _BROKER = None


def decode_frame(data_url: str) -> Frame | None:
    """Turn a browser data URL into bytes, or None if it is not usable."""
    raw = str(data_url or "")
    if not raw.startswith("data:image/"):
        return None
    try:
        header, _, payload = raw.partition(",")
        if not payload:
            return None
        mime = header[5 : header.index(";")] if ";" in header else "image/jpeg"
        data = base64.b64decode(payload, validate=True)
    except (ValueError, IndexError) as exc:
        record_degradation(
            "senses.sight", exc, action="rejected a malformed camera frame", severity="debug"
        )
        return None
    if not data or len(data) > MAX_FRAME_BYTES:
        return None
    return Frame(data=data, mime_type=mime)


def camera_enabled() -> bool:
    """Whether the user has the camera turned on right now."""
    try:
        from interface.routes.privacy import get_browser_camera_privacy

        return bool(get_browser_camera_privacy().get("enabled", False))
    except _SIGHT_ERRORS as exc:
        record_degradation(
            "senses.sight",
            exc,
            action="could not read the camera privacy setting; assumed off",
            severity="warning",
        )
        return False


def sight_dependency_gap() -> str:
    """What is missing before she can see at all, or "" if nothing is.

    Checked up front rather than discovered as an opaque worker timeout. The
    vision worker is a subprocess; when its imports fail, the parent sees
    "failed to initialize within 30s" and nothing about why — so a missing
    package looks identical to a wedged model, and the operator debugs the
    wrong thing for an hour.

    ``transformers`` 5.x builds its image and video processors on torchvision,
    so a machine with torch but no torchvision loads text models fine and
    cannot construct a vision processor at all. That is a one-line fix and
    she should be able to say so.
    """
    try:
        import importlib.util

        if importlib.util.find_spec("mlx_vlm") is None:
            return "mlx_vlm is not installed, so there is no local vision runtime"
        if importlib.util.find_spec("torchvision") is None:
            return (
                "torchvision is not installed — transformers builds its image "
                "processors on it, so the vision model cannot load without it "
                "(pip install torchvision==0.26.0, which matches the installed torch)"
            )
    except (ImportError, ValueError) as exc:
        return f"the vision runtime could not be checked: {type(exc).__name__}: {exc}"
    return ""


def _sight_prompt(question: str) -> str:
    """What to ask the vision model about the frame.

    Deliberately narrow. The vision model's job here is to *read the image*
    and nothing else — the conversational answer is composed later, by her,
    from this reading plus everything else she knows. Asking a 2B model to
    also be conversational is how a reading turns into a small model's guess
    dressed as an observation, and the tell is that it starts hedging in
    assistant register instead of saying what is in front of it.

    The instruction against guessing matters more than it looks. A vision
    model asked "how many fingers" will produce a number for an empty frame
    if nothing tells it that "I cannot tell" is an allowed answer.
    """
    return (
        "Look at this image and answer the question about it directly and "
        "literally. Describe only what is actually visible. If the thing "
        "being asked about is not visible, or the image is too dark or "
        "blurred to tell, say exactly that instead of guessing.\n\n"
        f"Question: {str(question or 'What is in this image?').strip()}"
    )


async def look(
    question: str,
    *,
    timeout_s: float = CAPTURE_TIMEOUT_S,
    explicit_user_consent: bool = False,
) -> Look:
    """Capture now and answer ``question`` from what is actually in frame.

    Every failure path records facts through ``failure_context`` rather than
    returning a sentence, so the reply is hers. The strings returned here are
    diagnostic detail for that record, not dialogue.
    """
    from core.conversation.failure_context import record_capability_failure

    # Before the camera, before the frame: can she see at all? Capturing a
    # frame she has no way to read wastes the user's time and turns a missing
    # package into a mysterious timeout.
    gap = sight_dependency_gap()
    if gap:
        record_capability_failure(
            "camera",
            intent=f"look through the camera to answer: {question[:120]}",
            cause="not_installed",
            detail=gap,
            still_possible=("everything that does not need eyes",),
        )
        return Look(ok=False, cause="no_vision_runtime", detail=gap)

    if not camera_enabled() and not explicit_user_consent:
        record_capability_failure(
            "camera",
            intent=f"look through the camera to answer: {question[:120]}",
            cause="unauthorized",
            detail="the camera is switched off in privacy settings",
            still_possible=("turning the camera on, if they want you to look",),
        )
        return Look(ok=False, cause="camera_off", detail="camera privacy is off")

    frame = await get_capture_broker().request_frame(
        timeout_s=timeout_s,
        explicit_user_consent=explicit_user_consent,
    )
    if frame is None:
        record_capability_failure(
            "camera",
            intent=f"capture a frame to answer: {question[:120]}",
            cause="timeout",
            detail=(
                f"no surface returned a frame within {timeout_s:.0f}s — the desktop "
                "window may be closed, or the camera may be in use by another app"
            ),
        )
        return Look(ok=False, cause="no_frame", detail="no frame arrived in time")

    try:
        from core.brain.llm.mlx_vision_client import get_vision_client

        # The shared worker, not a fresh one: constructing the client spawns a
        # subprocess that loads a 1.2 GB model, and a look is not worth a
        # second copy of it on a host whose resident 32B already wires ~20 GB.
        client = get_vision_client()
        answer = await client.see_async(
            _sight_prompt(question),
            base64.b64encode(frame.data).decode("ascii"),
        )
    except _SIGHT_ERRORS as exc:
        record_degradation("senses.sight", exc, action="vision model could not read the frame")
        record_capability_failure(
            "camera",
            intent=f"read the captured frame to answer: {question[:120]}",
            cause="failed",
            detail=f"the vision model errored: {type(exc).__name__}: {exc}"[:300],
        )
        return Look(ok=False, frame=frame, cause="vision_failed", detail=str(exc))

    text = str(answer or "").strip()
    if not text:
        record_capability_failure(
            "camera",
            intent=f"read the captured frame to answer: {question[:120]}",
            cause="empty_result",
            detail="the vision model returned nothing for this frame",
        )
        return Look(ok=False, frame=frame, cause="empty", detail="vision returned nothing")

    _record_canonical_camera_observation(text)
    _note_camera_observation()
    logger.info("look: %d bytes -> %d chars", len(frame.data), len(text))
    return Look(ok=True, answer=text, frame=frame)


__all__ = [
    "CAPTURE_HEIGHT",
    "CAPTURE_TIMEOUT_S",
    "CAPTURE_WIDTH",
    "MAX_FRAME_BYTES",
    "CaptureBroker",
    "Frame",
    "Look",
    "camera_enabled",
    "camera_observation_age_seconds",
    "decode_frame",
    "get_capture_broker",
    "look",
    "reset_capture_broker_for_test",
]
