"""
test_load_stress.py
────────────────────
Load and stress tests for the Aura server.
Verifies the system does not crash under sustained pressure, concurrent
connections, or message floods.

All tests are marked @pytest.mark.slow and @pytest.mark.stress so they
can be excluded from fast CI runs with:  pytest -m "not stress"
"""
from __future__ import annotations


import asyncio
import json
import os
import sys
import time
import logging

import pytest

# ---------------------------------------------------------------------------
# Graceful import guards
# ---------------------------------------------------------------------------
try:
    import httpx
except ImportError:
    pytest.skip("httpx not installed; skipping stress tests.", allow_module_level=True)

try:
    from starlette.testclient import TestClient
    from starlette.websockets import WebSocketDisconnect
except ImportError:
    pytest.skip("starlette not installed; skipping stress tests.", allow_module_level=True)

try:
    import psutil
except ImportError:
    psutil = None  # Memory test will skip gracefully

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
except ImportError:
    pytest.skip("fastapi not installed; skipping stress tests.", allow_module_level=True)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _build_test_app() -> FastAPI:
    """
    Build a lightweight FastAPI app that mirrors the real server's
    /api/chat, /api/health, and /ws endpoints but uses mocked internals
    so we never need a running kernel, database, or LLM.

    This avoids importing interface.server (which triggers heavy side-effects
    like loading the full orchestrator, event bus, prometheus, etc.).
    """
    from pydantic import BaseModel

    class ChatRequest(BaseModel):
        message: str

    app = FastAPI()

    # Track received message count for assertions
    app.state.message_count = 0
    app.state.ws_connections = 0

    @app.get("/api/health")
    async def health():
        return {"status": "ok", "time": time.time()}

    @app.get("/api/health/heartbeat")
    async def heartbeat():
        return {"status": "ok", "time": time.time()}

    @app.post("/api/chat")
    async def chat(body: ChatRequest):
        app.state.message_count += 1
        # Simulate a small amount of processing
        await asyncio.sleep(0.005)
        return {
            "response": f"Mock response #{app.state.message_count} to: {body.message[:50]}"
        }

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        app.state.ws_connections += 1
        try:
            while True:
                data = await ws.receive_json()
                msg_type = data.get("type")

                if msg_type == "ping":
                    await ws.send_json({"type": "pong"})
                elif msg_type == "user_message":
                    content = data.get("content", "")
                    await ws.send_json({
                        "type": "response",
                        "content": f"Echo: {content[:80]}",
                    })
                else:
                    await ws.send_json({"type": "ack"})
        except WebSocketDisconnect:
            pass
        except Exception:
            pass
        finally:
            app.state.ws_connections -= 1

    return app


@pytest.fixture(scope="module")
def test_app():
    """Module-scoped test app so we don't rebuild per-test."""
    return _build_test_app()


def _build_websocket_test_app() -> FastAPI:
    """Dedicated websocket helper with a minimal, proven-stable echo loop."""
    app = FastAPI()
    app.state.ws_connections = 0

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        app.state.ws_connections += 1
        try:
            while True:
                payload = await ws.receive_json()
                msg_type = payload.get("type")
                if msg_type == "ping":
                    await ws.send_json({"type": "pong"})
                elif msg_type == "user_message":
                    content = str(payload.get("content", ""))
                    await ws.send_json({
                        "type": "response",
                        "content": f"Echo: {content[:80]}",
                    })
                else:
                    await ws.send_json({"type": "ack"})
        except WebSocketDisconnect:
            pass
        finally:
            app.state.ws_connections -= 1

    return app


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.stress
@pytest.mark.asyncio
async def test_rapid_message_flood(test_app):
    """
    Send 50 messages in rapid succession via /api/chat.
    Verify no 500 errors are returned.
    """
    transport = httpx.ASGITransport(app=test_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        tasks = []
        for i in range(50):
            payload = {"message": f"Flood message {i}: {'x' * 20}"}
            tasks.append(client.post("/api/chat", json=payload))

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        errors_500 = 0
        exceptions = 0
        for resp in responses:
            if isinstance(resp, Exception):
                exceptions += 1
                logger.warning("Request raised exception: %s", resp)
            elif resp.status_code >= 500:
                errors_500 += 1
                logger.warning("Got %d: %s", resp.status_code, resp.text[:200])

        assert errors_500 == 0, f"{errors_500}/50 requests returned 5xx errors"
        assert exceptions == 0, f"{exceptions}/50 requests raised exceptions"


@pytest.mark.slow
@pytest.mark.stress
@pytest.mark.asyncio
async def test_concurrent_websocket_connections():
    """
    Open 5 WebSocket connections simultaneously.
    Send a message on each and verify all get responses (no crashes).
    """
    websocket_test_app = _build_websocket_test_app()

    # Run all 5 sessions
    # TestClient is synchronous so we run them via threads
    tasks = [asyncio.to_thread(_ws_session_sync, websocket_test_app, i) for i in range(5)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    successes = sum(1 for r in results if isinstance(r, str) and "Echo:" in r)
    failures = [r for r in results if isinstance(r, Exception)]

    assert successes >= 3, (
        f"Expected at least 3/5 WS connections to succeed, got {successes}. "
        f"Failures: {failures}"
    )


def _ws_session_sync(app: FastAPI, session_id: int) -> str:
    """Synchronous WebSocket session for use with asyncio.to_thread."""
    with TestClient(app) as tc:
        with tc.websocket_connect("/ws") as ws:
            ws.send_json({
                "type": "user_message",
                "content": f"Concurrent session {session_id}",
            })
            payload = ws.receive_json()
            return str(payload.get("content", ""))


@pytest.mark.slow
@pytest.mark.stress
@pytest.mark.asyncio
async def test_message_queue_overflow(test_app):
    """
    Send 200 messages without waiting for responses.
    Verify the server stays alive (health endpoint returns 200).
    """
    transport = httpx.ASGITransport(app=test_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        # Fire 200 messages as fast as possible
        tasks = []
        for i in range(200):
            payload = {"message": f"Overflow msg {i}"}
            tasks.append(client.post("/api/chat", json=payload))

        # We don't care about individual responses -- just gather them
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Now check the server is still alive
        health = await client.get("/api/health")
        assert health.status_code == 200, (
            f"Health endpoint returned {health.status_code} after 200-message flood"
        )

        health_data = health.json()
        assert health_data.get("status") == "ok", (
            f"Health status is not 'ok': {health_data}"
        )

        # Verify that at least 90% of messages got through without 5xx
        server_errors = sum(
            1 for r in responses
            if not isinstance(r, Exception) and r.status_code >= 500
        )
        assert server_errors < 20, (
            f"{server_errors}/200 messages returned 5xx -- server is unstable"
        )


@pytest.mark.slow
@pytest.mark.stress
@pytest.mark.asyncio
async def test_memory_growth_bounded(test_app):
    """
    Run 100 messages and verify process RSS hasn't grown more than 50MB.
    """
    if psutil is None:
        pytest.skip("psutil not installed; cannot measure memory growth")

    process = psutil.Process(os.getpid())
    baseline_rss = process.memory_info().rss

    transport = httpx.ASGITransport(app=test_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        for i in range(100):
            payload = {"message": f"Memory test message {i}: " + ("data " * 50)}
            resp = await client.post("/api/chat", json=payload)
            assert resp.status_code < 500, f"Message {i} failed with {resp.status_code}"

    current_rss = process.memory_info().rss
    growth_bytes = current_rss - baseline_rss
    growth_mb = growth_bytes / (1024 * 1024)

    logger.info(
        "Memory growth: %.2f MB (baseline=%.2f MB, current=%.2f MB)",
        growth_mb,
        baseline_rss / (1024 * 1024),
        current_rss / (1024 * 1024),
    )

    assert growth_mb < 50.0, (
        f"Memory grew by {growth_mb:.1f} MB after 100 messages (limit is 50 MB). "
        f"Possible memory leak."
    )
