import json

import pytest

from interface.routes import inner_state


@pytest.mark.asyncio
async def test_await_maybe_resolves_live_async_status():
    async def status_value():
        return {"status": "live"}

    assert await inner_state._await_maybe(status_value()) == {"status": "live"}


@pytest.mark.asyncio
async def test_unity_state_returns_structured_failure_for_runtime_boundary(monkeypatch):
    recorded = []
    calls = []

    def unavailable_surface():
        calls.append("called")
        raise RuntimeError("unity unavailable")

    monkeypatch.setattr(inner_state, "_build_unity_surface", unavailable_surface)
    monkeypatch.setattr(
        inner_state,
        "record_degradation",
        lambda subsystem, error: recorded.append((subsystem, str(error))),
    )

    response = await inner_state.get_unity_state()
    payload = json.loads(response.body)

    assert response.status_code == 500
    assert payload == {"error": "unity unavailable"}
    assert calls == ["called"]
    assert recorded == [("inner_state", "unity unavailable")]
