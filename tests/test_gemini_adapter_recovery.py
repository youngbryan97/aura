import httpx
import pytest

from core.brain.llm import gemini_adapter as gemini_module
from core.brain.llm.gemini_adapter import GeminiAdapter, GeminiProviderUnavailable


class _AllowingLimiter:
    def can_call(self, *_args, **_kwargs):
        return True

    def record_call(self, *_args, **_kwargs):
        return None


@pytest.mark.asyncio
async def test_gemini_stream_timeout_raises_provider_unavailable_for_router_failover(monkeypatch):
    recorded = []
    monkeypatch.setattr(
        gemini_module,
        "_record_gemini_degradation",
        lambda error, **kwargs: recorded.append((error, kwargs)),
    )

    class TimeoutStream:
        async def __aenter__(self):
            raise httpx.TimeoutException("stream stalled")

        async def __aexit__(self, *_args):
            return False

    class TimeoutClient:
        def stream(self, *_args, **_kwargs):
            return TimeoutStream()

    adapter = GeminiAdapter(api_key="test", rate_limiter=_AllowingLimiter(), timeout=1.0)
    monkeypatch.setattr(adapter, "_get_client", lambda: TimeoutClient())

    with pytest.raises(GeminiProviderUnavailable):
        async for _chunk in adapter.generate_text_stream_async("hello"):
            pass

    assert adapter.is_available() is False
    assert recorded[0][1]["action"] == (
        "raised provider-unavailable signal so router can fail over after Gemini stream timeout"
    )


@pytest.mark.asyncio
async def test_gemini_call_returns_failed_result_when_error_handler_fails(monkeypatch):
    recorded = []
    monkeypatch.setattr(
        gemini_module,
        "_record_gemini_degradation",
        lambda error, **kwargs: recorded.append((error, kwargs)),
    )

    class ResponseClient:
        async def post(self, *_args, **_kwargs):
            return httpx.Response(500, content=b"server fault")

    adapter = GeminiAdapter(api_key="test", rate_limiter=_AllowingLimiter())
    monkeypatch.setattr(adapter, "_get_client", lambda: ResponseClient())

    async def broken_error_handler(_response):
        raise RuntimeError("handler broken")

    monkeypatch.setattr(adapter, "_handle_error", broken_error_handler)

    ok, text, metadata = await adapter.call("hello")

    assert ok is False
    assert text == ""
    assert metadata["error"] == "handler broken"
    assert recorded[0][1]["action"] == (
        "returned failed Gemini call result after provider error handler failed"
    )
