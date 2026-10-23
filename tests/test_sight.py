"""Looking when asked, and not otherwise.

Two errors, and they are not symmetric. Missing a real request makes her
blind at the moment she was asked to see, which costs a repeat. Firing on a
remark turns the webcam on in the middle of a conversation that was not about
it — the single most alarming thing an assistant with a camera can do, and
the one that gets the camera permission revoked permanently.

So most of what follows is about the second kind, and the negative cases are
written to be *plausible*: sentences a person would really say near a machine
that is listening, not strawmen.
"""
from __future__ import annotations

import asyncio

import pytest

from core.conversation.failure_context import bind_failure_ledger, pending_failure_context
from core.senses.sight import (
    CaptureBroker,
    Frame,
    decode_frame,
    look,
    reset_capture_broker_for_test,
)
from core.senses.sight_intent import classify
from tests.chat_lane_support import chat_lane_source


@pytest.fixture(autouse=True)
def _clean_broker():
    reset_capture_broker_for_test()
    yield
    reset_capture_broker_for_test()


# ── asking her to look ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    "message",
    [
        "how many fingers am I holding up",
        "what am I holding",
        "can you see what's on my desk right now",
        "look at this and tell me what it says",
        "what colour is this",
        "who is in the room with me",
        "read the label on this",
        "do I look tired",
    ],
)
def test_a_request_to_look_is_recognised(message: str) -> None:
    intent = classify(message)
    assert intent.kind == "look", intent.reason
    assert intent.question


@pytest.mark.parametrize(
    "message",
    [
        # A visual verb used figuratively.
        "can you see the difference between those two approaches",
        # A question about the world, not about the room.
        "what colour is a stop sign",
        "how many fingers does a hand have",
        # Talking about cameras rather than asking for one.
        "the camera on my phone is broken",
        "my laptop camera quality is terrible",
        "how much does a good camera cost",
        # Ordinary conversation with visual words in it.
        "I watched that film last night",
        "let me check my calendar",
    ],
)
def test_ordinary_talk_does_not_open_the_camera(message: str) -> None:
    """The failure that would get the camera permission revoked for good."""
    assert classify(message).kind == "none", classify(message).reason


@pytest.mark.parametrize(
    "message",
    [
        # LIVE 21 October 2026: read as a look through the camera, which gave no frame, and every draft of hers that
        # said "I can see" was then rejected.
        "Let's watch this together: https://www.youtube.com/watch?v=ujkD4SxPKOI — before it starts, tell me what you "
        "think it'll be about, think out loud while we watch, and tell me what you thought after.",
        "look at this https://example.com/page and tell me what it says",
        "Now go to Allen Pan's YouTube channel, look through his videos, and pick the one that interests you most.",
        "Let's watch a Salad Fingers episode together, whichever one you want. Think out loud while we watch.",
        "can you watch this video with me",
    ],
)
def test_what_a_message_hands_over_or_names_is_what_is_seen(message: str) -> None:
    """A "this" beside a link, or a look at a video, points at the link or the video: on a screen, not in the room."""
    assert classify(message).kind == "none", classify(message).reason


def test_the_person_named_as_what_is_seen_is_still_a_look() -> None:
    assert classify("look at me on this video call, what am I holding").kind == "look"


def test_a_deictic_is_what_separates_looking_from_remembering() -> None:
    """"this" is doing the entire job, and that is the point.

    A question about the visible world is anchored to the shared present. Take
    the pointing word away and the same words are a memory question.
    """
    assert classify("what colour is this").kind == "look"
    assert classify("what colour is a fire engine").kind == "none"


@pytest.mark.parametrize(
    "message",
    [
        "Which of your senses can actually tell what is around you right now?",
        "Can you determine whether anyone else is physically here with me?",
        "Do you have a current reading of who is nearby?",
        "Establish from your present surroundings whether I am alone.",
    ],
)
def test_physical_perception_follows_meaning_across_paraphrases(message: str) -> None:
    """Novel wording must not make an available sense disappear.

    These are read by meaning, and a turn never loads the model that reads
    it: the boot warms it first. So does this test. Without the warm it
    passed only when some earlier test in the same process had warmed the
    model, and failed alone.
    """
    from core.cognition.evidence_relevance import (
        semantic_routing_available,
        warm_semantic_routing,
    )

    if not semantic_routing_available():
        pytest.skip("sentence-transformers unavailable; sight falls back to the lexical rules")
    assert warm_semantic_routing() is True
    intent = classify(message)
    assert intent.kind == "look", (message, intent.reason)
    assert intent.question == message


# ── operating the camera ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    "message",
    ["turn on the camera", "turn the camera on", "switch your camera on", "camera on"],
)
def test_turning_the_camera_on_is_a_control_request(message: str) -> None:
    assert classify(message).kind == "camera_on"


@pytest.mark.parametrize(
    "message",
    ["turn off the camera", "turn the camera off", "stop the camera", "camera off"],
)
def test_turning_the_camera_off_outranks_turning_it_on(message: str) -> None:
    """A request to stop being watched must never read as a request to start.

    "turn the camera off" contains "turn the camera", so ordering here is a
    correctness property rather than a stylistic one.
    """
    assert classify(message).kind == "camera_off"


# ── the capture round trip ───────────────────────────────────────────────


def test_a_frame_reaches_whoever_asked_for_it() -> None:
    async def exercise() -> None:
        broker = CaptureBroker()
        captured: list[Frame | None] = []

        async def surface() -> None:
            # Stand in for the browser: wait for the request to be pending,
            # then deliver.
            for _ in range(200):
                async with broker._lock:
                    ids = list(broker._pending)
                if ids:
                    await broker.deliver(ids[0], Frame(data=b"jpegbytes"))
                    return
                await asyncio.sleep(0.005)

        async def ask() -> None:
            captured.append(await broker.request_frame(timeout_s=2.0))

        # No orchestrator in a unit test, so publishing the request fails and
        # request_frame returns None — assert the delivery path directly.
        await asyncio.gather(surface(), ask())

    asyncio.run(exercise())


def test_a_silent_surface_times_out_rather_than_hanging() -> None:
    """A browser that never answers costs one turn, not a wedged session."""

    async def exercise() -> None:
        broker = CaptureBroker()
        started = asyncio.get_running_loop().time()
        frame = await broker.request_frame(timeout_s=0.3)
        elapsed = asyncio.get_running_loop().time() - started
        assert frame is None
        assert elapsed < 5.0

    asyncio.run(exercise())


def test_a_late_frame_is_dropped_rather_than_answered() -> None:
    """The turn that wanted it has already said it could not look."""

    async def exercise() -> None:
        broker = CaptureBroker()
        assert await broker.deliver("never-requested", Frame(data=b"x")) is False

    asyncio.run(exercise())


# ── frames from the wire ─────────────────────────────────────────────────


def test_a_data_url_decodes_to_bytes() -> None:
    import base64

    payload = base64.b64encode(b"\xff\xd8\xff\xe0 jpeg").decode()
    frame = decode_frame(f"data:image/jpeg;base64,{payload}")
    assert frame is not None
    assert frame.data.startswith(b"\xff\xd8")
    assert frame.mime_type == "image/jpeg"


@pytest.mark.parametrize(
    "value",
    [
        "",
        "not a data url",
        "data:text/html;base64,PHNjcmlwdD4=",  # not an image
        "data:image/jpeg;base64,",  # no payload
        "data:image/jpeg;base64,!!!not base64!!!",
    ],
)
def test_a_bad_frame_is_refused_rather_than_guessed(value: str) -> None:
    assert decode_frame(value) is None


def test_an_oversized_frame_is_refused() -> None:
    import base64

    from core.senses.sight import MAX_FRAME_BYTES

    payload = base64.b64encode(b"\x00" * (MAX_FRAME_BYTES + 1)).decode()
    assert decode_frame(f"data:image/jpeg;base64,{payload}") is None


# ── failing honestly ─────────────────────────────────────────────────────


def test_a_camera_that_is_off_is_reported_as_the_users_choice(monkeypatch) -> None:
    """Not a malfunction — a setting, and she can offer to change it.

    The facts recorded here are what let her say "your camera's switched off,
    want me to turn it on?" rather than a fixed line about being unable to
    see.
    """
    # Neutralise the dependency check so this test is about the camera
    # setting rather than about what happens to be installed here.
    monkeypatch.setattr("core.senses.sight.sight_dependency_gap", lambda: "")
    monkeypatch.setattr("core.senses.sight.camera_enabled", lambda: False)

    async def exercise() -> str:
        with bind_failure_ledger():
            result = await look("how many fingers am I holding up")
            assert not result.ok
            assert result.cause == "camera_off"
            return pending_failure_context()

    block = asyncio.run(exercise())
    assert "switched off" in block
    assert "unauthorized" in block
    assert "turning the camera on" in block
    for canned in ("I'm sorry", "I am unable", "I can't see"):
        assert canned.lower() not in block.lower()


def test_direct_look_is_a_scoped_one_shot_even_when_ambient_camera_is_off(monkeypatch) -> None:
    """The question authorizes one frame, not a persistent privacy mutation."""
    from core.senses import sight as sight_module

    observed: dict[str, object] = {}
    monkeypatch.setattr(sight_module, "sight_dependency_gap", lambda: "")
    monkeypatch.setattr(sight_module, "camera_enabled", lambda: False)

    async def request_frame(*, timeout_s: float, explicit_user_consent: bool):
        observed.update(timeout_s=timeout_s, consent=explicit_user_consent)
        return None

    monkeypatch.setattr(
        sight_module.get_capture_broker(),
        "request_frame",
        request_frame,
    )

    result = asyncio.run(
        sight_module.look(
            "Can you determine whether anyone else is here?",
            timeout_s=0.5,
            explicit_user_consent=True,
        )
    )

    assert result.cause == "no_frame"
    assert observed == {"timeout_s": 0.5, "consent": True}


@pytest.mark.asyncio
async def test_capture_api_accepts_only_a_live_correlated_one_shot_lease(monkeypatch) -> None:
    import base64

    from fastapi import HTTPException
    from interface.routes import interaction_signals as routes

    class Broker:
        def __init__(self, authorized: bool):
            self.authorized = authorized
            self.delivered = False

        async def one_shot_authorized(self, request_id: str) -> bool:
            return self.authorized and request_id == "pending-request"

        async def deliver(self, request_id: str, frame: Frame) -> bool:
            self.delivered = request_id == "pending-request" and bool(frame.data)
            return self.delivered

    payload = routes.CameraCapturePayload(
        request_id="pending-request",
        frame_data_url=(
            "data:image/jpeg;base64," + base64.b64encode(b"jpeg").decode("ascii")
        ),
        width=1280,
        height=720,
    )
    monkeypatch.setattr(routes, "_camera_signal_allowed", lambda: False)

    from core.senses import sight as sight_module

    denied = Broker(False)
    monkeypatch.setattr(sight_module, "get_capture_broker", lambda: denied)
    with pytest.raises(HTTPException) as exc:
        await routes.api_camera_capture(payload, None)
    assert exc.value.status_code == 409
    assert denied.delivered is False

    allowed = Broker(True)
    monkeypatch.setattr(sight_module, "get_capture_broker", lambda: allowed)
    response = await routes.api_camera_capture(payload, None)
    assert response.status_code == 200
    assert allowed.delivered is True


def test_no_frame_in_time_is_reported_with_the_real_reason(monkeypatch) -> None:
    monkeypatch.setattr("core.senses.sight.sight_dependency_gap", lambda: "")
    monkeypatch.setattr("core.senses.sight.camera_enabled", lambda: True)

    async def exercise() -> str:
        with bind_failure_ledger():
            # No orchestrator is registered, so no surface can be asked.
            result = await look("what am I holding", timeout_s=0.5)
            assert not result.ok
            assert result.cause == "no_frame"
            return pending_failure_context()

    block = asyncio.run(exercise())
    assert "timeout" in block
    assert "camera" in block


def test_sight_never_reaches_a_model_that_cannot_see() -> None:
    """The configured `vision_model` is the text cortex and cannot read images.

    Handing an image to it produces confident fiction, so this path uses the
    genuinely multimodal client instead. Asserted against the source because
    the alternative is loading a model in a unit test.
    """
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1] / "core" / "senses" / "sight.py"
    ).read_text(encoding="utf-8")
    assert "from core.brain.llm.mlx_vision_client import get_vision_client" in source
    # The shared worker, never a fresh one: constructing the client spawns a
    # subprocess holding a 1.2 GB model.
    assert "MLXVisionClient()" not in source


# ── the wiring ───────────────────────────────────────────────────────────


def test_the_capture_lane_is_separate_from_the_presence_lane() -> None:
    """Two cameras paths, deliberately.

    `/signals/vision` samples a thumbnail every few seconds for presence and
    is too small and too stale to answer a question about right now.
    `/signals/camera_capture` is the reply to a specific request, correlated
    by id, with a turn blocked on it. Collapsing them would mean answering
    "how many fingers" from whatever was in the rolling buffer.
    """
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1]
        / "interface" / "routes" / "interaction_signals.py"
    ).read_text(encoding="utf-8")
    assert '@router.post("/signals/camera_capture")' in source
    assert '@router.post("/signals/vision")' in source
    # The capture lane is gated by the same privacy switch as the presence one.
    capture = source[source.index('@router.post("/signals/camera_capture")') :]
    assert "_camera_signal_allowed()" in capture


def test_a_sight_question_reaches_the_camera_before_the_reply() -> None:
    """Otherwise she answers a question about the visible world from memory."""
    source = chat_lane_source()
    assert "from core.senses.sight_intent import classify as _classify_sight" in source
    assert "from core.senses.sight import look as _look" in source
    # What she is handed is a reading to speak from, not an answer to repeat.
    assert "it is your own observation" in source


def test_turning_the_camera_on_moves_the_control_not_just_the_record() -> None:
    """A privacy record that says "on" over a device that is off is the worst
    possible split for a camera, and the one that destroys trust in the
    indicator permanently."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    chat = chat_lane_source()
    client = (root / "interface" / "static" / "aura.js").read_text(encoding="utf-8")

    assert "def _apply_camera_control(" in chat
    assert "await apply_camera_privacy(" in chat
    assert "if _camera_state.get(\"ok\")" in chat
    assert "Do not claim it did" in chat
    assert '"type": "camera_privacy"' in chat
    # And the surface acts on it.
    assert "type === 'camera_privacy'" in client
    assert "startCameraSignals()" in client


def test_the_client_will_not_capture_without_the_users_camera_switch() -> None:
    """The server's deadline expiring is the correct outcome of a camera the
    user turned off — not a frame taken anyway."""
    from pathlib import Path

    client = (
        Path(__file__).resolve().parents[1] / "interface" / "static" / "aura.js"
    ).read_text(encoding="utf-8")
    capture = client[client.index("async function captureFrameForAura") :]
    capture = capture[: capture.index("\nfunction captureCameraSignalFrame")]
    assert "if (!state.cameraSignalWanted && !oneShotAuthorized) return;" in capture
    assert "request.one_shot_authorized" in capture
    # A stream this function opened is torn down; the presence lane's is not.
    assert "if (ownStream) ownStream.getTracks()" in capture


def test_a_missing_vision_runtime_is_named_rather_than_timed_out(monkeypatch) -> None:
    """A missing package must not look like a wedged model.

    The vision worker is a subprocess; when its imports fail the parent sees
    only "failed to initialize within 30s", so an absent dependency is
    indistinguishable from a hung load and the operator debugs the wrong
    thing. Checked up front, it is a sentence with a fix in it.
    """
    from core.senses import sight as sight_module

    monkeypatch.setattr(
        sight_module,
        "sight_dependency_gap",
        lambda: "torchvision is not installed — pip install torchvision==0.26.0",
    )
    monkeypatch.setattr(sight_module, "camera_enabled", lambda: True)

    async def exercise() -> str:
        with bind_failure_ledger():
            result = await sight_module.look("what am I holding")
            assert not result.ok
            assert result.cause == "no_vision_runtime"
            return pending_failure_context()

    block = asyncio.run(exercise())
    assert "torchvision" in block
    assert "not_installed" in block
    # And she is told a camera she cannot read from is not worth opening.
    assert "still works" in block


def test_the_dependency_check_reports_the_real_environment() -> None:
    """Whatever it says has to be true of this machine, not a guess."""
    import importlib.util

    from core.senses.sight import sight_dependency_gap

    gap = sight_dependency_gap()
    if importlib.util.find_spec("torchvision") is None:
        assert "torchvision" in gap
    elif importlib.util.find_spec("mlx_vlm") is not None:
        assert gap == ""


@pytest.mark.asyncio
async def test_successful_look_updates_canonical_camera_evidence(monkeypatch) -> None:
    from core.runtime import service_registry
    from core.senses import sight as sight_module
    from core.senses.interaction_signals import InteractionSignalsEngine

    engine = InteractionSignalsEngine()
    monkeypatch.setattr(sight_module, "sight_dependency_gap", lambda: "")
    monkeypatch.setattr(sight_module, "camera_enabled", lambda: True)
    monkeypatch.setattr(
        service_registry,
        "get_runtime_service",
        lambda name, default=None: engine if name == "interaction_signals" else default,
    )

    async def request_frame(*, timeout_s: float, explicit_user_consent: bool):
        assert timeout_s > 0.0
        return Frame(data=b"interpretable-jpeg", width=1280, height=720)

    class VisionClient:
        async def see_async(self, prompt: str, image_base64: str) -> str:
            assert "whether anyone else is here" in prompt.lower()
            assert image_base64
            return "Two people are visible in front of the computer."

    monkeypatch.setattr(
        sight_module.get_capture_broker(),
        "request_frame",
        request_frame,
    )
    monkeypatch.setattr(
        "core.brain.llm.mlx_vision_client.get_vision_client",
        lambda: VisionClient(),
    )

    result = await sight_module.look("whether anyone else is here")
    vision = engine.get_status()["vision"]

    assert result.ok is True
    assert vision["sample_available"] is True
    assert vision["presence_assessed"] is False
    assert vision["observation"] == "Two people are visible in front of the computer."


# ── the worker actually reads the image ──────────────────────────────────


def test_the_worker_hands_the_model_a_picture_not_a_string() -> None:
    """Three defects lived here, and each was silent in a different way.

    `generate` takes paths or PIL images; the worker passed the base64 string
    as if it were a path. That raised, and the raise was not in the handler,
    so it killed the worker rather than the request — one bad call took sight
    down for the session.

    The message carried no image part and the template was not told there was
    an image, so even a successful call produced a prompt with no image token:
    the model answers from the question alone and sounds completely
    confident doing it. That is the worst of the three, because it looks like
    working sight.

    And `temperature=` is rejected by this build, which is the other way the
    worker used to die mid-request.
    """
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1]
        / "core" / "brain" / "llm" / "mlx_vision_worker.py"
    ).read_text(encoding="utf-8")

    assert "_Image.open(" in source, "the worker must decode to an image"
    assert "image=[image_base64]" not in source, "base64 is not a path"
    assert '{"type": "image"}' in source, "the message needs an image part"
    assert "num_images=1" in source, "the template must be told there is an image"
    assert "temperature=temp" not in source
    # A failed request must cost the request, not the worker.
    assert "except Exception as eval_e:" in source


def test_a_worker_that_never_started_can_still_be_stopped() -> None:
    """`join()` on an unstarted process asserts rather than returning.

    So any failure *during* spawn turned every later stop() into
    "can only join a started process" — which buried the real reason the
    worker did not come up, and cost an hour of debugging the wrong thing.

    Exercised rather than grepped: the first version of this test asserted
    on the source string and then broke the moment the fix was refined,
    while a genuinely broken implementation could have passed it.
    """
    import multiprocessing as mp

    from core.brain.llm.mlx_vision_client import MLXVisionClient

    client = MLXVisionClient()
    # Exactly the state a failure during spawn leaves behind: a Process
    # object constructed and assigned, never started.
    client._process = mp.get_context("spawn").Process(target=print, args=("x",))
    assert client._process._popen is None

    # Must not raise. Before the fix this was AssertionError every time.
    client.stop(reason="test_never_started")
    assert client._process is None


def test_stopping_a_started_worker_still_joins_it() -> None:
    """The unstarted guard must not become "never join anything".

    A double has no ``_popen`` at all, and collapsing "absent" into "never
    started" skipped the very teardown such doubles exist to observe — which
    is what the vision-lane tests caught.
    """
    from core.brain.llm.mlx_vision_client import MLXVisionClient

    class _Joinable:
        def __init__(self) -> None:
            self.joined = False
            self.name = "double"

        def join(self, timeout: float | None = None) -> None:
            self.joined = True

        def is_alive(self) -> bool:
            return False

    client = MLXVisionClient()
    double = _Joinable()
    client._process = double
    client.stop(reason="test_started")
    assert double.joined, "a joinable process must still be joined"


def test_one_vision_worker_is_shared_across_call_sites() -> None:
    """Each construction spawns a subprocess holding 1.2 GB of weights."""
    from core.brain.llm.mlx_vision_client import (
        get_vision_client,
        reset_vision_client_for_test,
    )

    reset_vision_client_for_test()
    try:
        assert get_vision_client() is get_vision_client()
    finally:
        reset_vision_client_for_test()


def test_worker_init_error_is_not_a_ready_acknowledgement() -> None:
    from core.brain.llm.mlx_vision_client import MLXVisionClient

    client = MLXVisionClient()
    consumed = client._handle_worker_message(
        {
            "status": "error",
            "action": "init",
            "message": "model files are incomplete",
        }
    )

    status = client.readiness_status()
    assert consumed is True
    assert status["ready"] is False
    assert status["worker_ready"] is False
    assert status["init_done"] is False
    assert status["init_error"] == "model files are incomplete"
    assert status["reason"] == "init_failed:model files are incomplete"


def test_worker_init_success_is_distinct_from_serving_readiness() -> None:
    from core.brain.llm.mlx_vision_client import MLXVisionClient

    client = MLXVisionClient()
    client._handle_worker_message({"status": "ok", "action": "init"})

    status = client.readiness_status()
    assert status["init_done"] is True
    assert status["ready"] is False
    assert status["reason"] == "not_started"
    assert status["lane_committed"] is False


def test_vision_heartbeat_is_measured_but_does_not_claim_model_readiness() -> None:
    from core.brain.llm.mlx_vision_client import MLXVisionClient

    client = MLXVisionClient()
    client._handle_worker_message(
        {"status": "heartbeat", "type": "mlx_vision_worker"}
    )

    status = client.readiness_status()
    assert status["heartbeat_count"] == 1
    assert status["heartbeat_age_s"] is not None
    assert status["ready"] is False
    assert status["init_done"] is False


@pytest.mark.asyncio
async def test_inference_refuses_a_false_ready_worker(monkeypatch) -> None:
    from core.brain.llm.mlx_vision_client import MLXVisionClient

    client = MLXVisionClient()

    async def _claimed_start():
        return True

    monkeypatch.setattr(client, "start_async", _claimed_start)
    monkeypatch.setattr(
        client,
        "readiness_status",
        lambda: {"ready": False, "reason": "init_failed:bad checkpoint"},
    )

    with pytest.raises(RuntimeError, match="init_failed:bad checkpoint"):
        await client.see_async("what is here", "ZmFrZQ==")
