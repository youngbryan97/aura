from __future__ import annotations

import pytest

from core.runtime import public_http_transport


class _Gateway:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple, dict]] = []

    async def request_async(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return {"ok": True, "status_code": 200, "content": b"ok"}


@pytest.mark.asyncio
async def test_transport_derives_read_only_and_pins_public_redirects(monkeypatch) -> None:
    gateway = _Gateway()
    monkeypatch.setattr(public_http_transport, "get_network_gateway", lambda: gateway)

    result = await public_http_transport.request_public_http(
        "GET",
        "https://example.com/data",
        source="test.reader",
        max_response_bytes=1024,
    )

    assert result["ok"] is True
    assert gateway.calls == [
        (
            ("GET", "https://example.com/data"),
            {
                "headers": None,
                "data": None,
                "timeout": 30.0,
                "source": "test.reader",
                "read_only": True,
                "max_response_bytes": 1024,
                "public_network_only": True,
            },
        )
    ]


@pytest.mark.asyncio
async def test_mutating_method_cannot_be_mislabelled_read_only(monkeypatch) -> None:
    gateway = _Gateway()
    monkeypatch.setattr(public_http_transport, "get_network_gateway", lambda: gateway)

    await public_http_transport.request_public_http(
        "POST",
        "https://example.com/data",
        data=b"{}",
        source="test.writer",
        max_response_bytes=1024,
    )

    assert gateway.calls[0][1]["read_only"] is False
    assert gateway.calls[0][1]["public_network_only"] is True


@pytest.mark.asyncio
async def test_transport_refuses_unbounded_or_unowned_requests() -> None:
    with pytest.raises(ValueError):
        await public_http_transport.request_public_http(
            "GET",
            "https://example.com",
            source="",
            max_response_bytes=1024,
        )
    with pytest.raises(ValueError):
        await public_http_transport.request_public_http(
            "GET",
            "https://example.com",
            source="test",
            max_response_bytes=32 * 1024 * 1024,
        )
