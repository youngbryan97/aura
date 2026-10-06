"""A blocked background worker cannot freeze observation or delivery ownership."""
from __future__ import annotations

import asyncio
import base64
import contextlib
import contextvars
import io
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from PIL import Image


@contextlib.asynccontextmanager
async def _occupied_default_pool():
    loop = asyncio.get_running_loop()
    previous = loop._default_executor
    pool = ThreadPoolExecutor(max_workers=1)
    loop.set_default_executor(pool)
    release, entered = threading.Event(), asyncio.Event()

    def wait():
        loop.call_soon_threadsafe(entered.set)
        release.wait(3)

    pending = loop.run_in_executor(None, wait)
    try:
        await asyncio.wait_for(entered.wait(), 1)
        yield
    finally:
        release.set()
        await pending
        loop._default_executor = previous
        pool.shutdown(wait=True)


@pytest.mark.asyncio
async def test_delivery_admission_renewal_and_finalization_do_not_wait_for_background_work(tmp_path):
    from core.runtime.chat_delivery_journal import (
        AdmissionKind, ChatDeliveryJournal, DeliveryIdentity, DeliveryState, canonical_request_hash,
    )

    journal = ChatDeliveryJournal(tmp_path / "delivery.sqlite3")
    identity = DeliveryIdentity.create(principal="test", session_id="session", idempotency_key="one")
    async with _occupied_default_pool():
        async with asyncio.timeout(1):
            admission = await journal.reserve(identity, canonical_request_hash({"message": "work"}))
            assert admission.kind is AdmissionKind.EXECUTE
            assert await journal.renew(admission)
            await journal.publish_progress(admission, phase="executing", message="Inputs delivered.")
            result = await journal.finalize(admission, state=DeliveryState.COMPLETED,
                                            http_status=200, response={"response": "Verified."})
            assert result.state is DeliveryState.COMPLETED
            assert (await journal.get(identity)).state is DeliveryState.COMPLETED


@pytest.mark.asyncio
async def test_canvas_decoding_continues_without_a_default_worker():
    from core.perception.frames_as_they_are_drawn import CanvasFrames

    encoded = io.BytesIO()
    Image.new("RGB", (100, 100), "white").save(encoded, format="JPEG")
    data = "data:image/jpeg;base64," + base64.b64encode(encoded.getvalue()).decode()
    async def evaluate(*_args):
        return {"pixels": data, "scene": None}
    frames = CanvasFrames(SimpleNamespace(evaluate=evaluate))
    frames._checked = frames._observing = True
    async with _occupied_default_pool():
        picture, _at = await asyncio.wait_for(frames.look({"x": 0, "y": 0, "width": 100, "height": 100}), 1)
        assert picture.shape == (100, 100, 3)


@pytest.mark.asyncio
async def test_interactive_worker_preserves_context_and_recovers_after_pool_shutdown():
    from core.runtime import executors

    marker = contextvars.ContextVar("interactive_test_marker", default="missing")
    marker.set("owner")
    assert await executors.run_interactive_cpu(marker.get) == "owner"
    executors.INTERACTIVE_CPU_POOL.shutdown(wait=False, cancel_futures=True)
    assert await executors.run_interactive_cpu(marker.get) == "owner"
    release = threading.Event()
    try:
        with pytest.raises(TimeoutError):
            await executors.run_interactive_cpu(release.wait, 1, timeout_s=0.01)
    finally:
        release.set()


@pytest.mark.asyncio
async def test_journal_offload_preserves_governance_context():
    from core.runtime.chat_delivery_journal import _journal_io

    marker = contextvars.ContextVar("journal_test_marker", default="missing")
    marker.set("owner")
    assert await _journal_io(marker.get) == "owner"


def test_pool_ownership_has_a_measured_invariant_and_claim():
    from core.runtime.executors import _interactive_worker_invariant
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    assert _interactive_worker_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == "interactive_and_receipt_workers_are_separate")
    assert check.predict(None) is True and any(c.test == check.name for c in suite.claims())
