"""Her lookups go through her network gateway, to public addresses only: its governance and egress privacy boundary
apply to them as to every request she makes."""
from __future__ import annotations

import asyncio

import pytest

pytestmark = pytest.mark.unit


def test_a_search_is_asked_and_read_through_the_gateway(monkeypatch):
    import core.runtime.network_gateway as network_gateway
    from core.skills import looking_it_up

    asked: list[dict] = []

    class Gateway:
        async def request_async(self, method, url, **kwargs):
            asked.append({"method": method, "url": url, **kwargs})
            page = b"<html><body><p>The answer is to turn the valve twice.</p></body></html>"
            return {"status_code": 200, "content": b"results" if "search" in url else page, "url": url, "ok": True}

    monkeypatch.setattr(network_gateway, "get_network_gateway", lambda: Gateway())
    monkeypatch.setattr(looking_it_up, "ENGINES", [
        ("an engine", "https://search.example/search?q={q}", lambda text: [("Valves", "https://valves.example/how")]),
    ])

    pages = asyncio.run(looking_it_up.search_and_read("how to open a stuck valve", 10.0))

    assert pages and pages[0][:2] == ("Valves", "https://valves.example/how")
    assert "turn the valve twice" in pages[0][2]
    assert [a["url"] for a in asked] == ["https://search.example/search?q=how+to+open+a+stuck+valve",
                                         "https://valves.example/how"]
    for request in asked:
        assert request["public_network_only"] is True and request["read_only"] is True
        assert request["headers"] == looking_it_up._AS_A_BROWSER
