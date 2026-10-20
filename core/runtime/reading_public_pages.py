"""Reading public pages through her network gateway, in the shape code written for an HTTP client expects.

Code that reads a page asks ``ThroughHerGateway(...).get(url)`` and gets a reply with ``status_code``, ``text``,
``json()`` and ``url``, as it would from httpx. The request goes through ``core/runtime/network_gateway.py``: its
governance, its egress privacy boundary, public addresses only, every redirect hop checked. Three readers opened their
own httpx clients and skipped all of that (tests/test_egress_lane_ratchet.py, failing on main from 6 October 2026).
"""
from __future__ import annotations

import json
import urllib.parse
from dataclasses import dataclass
from typing import Any

__all__ = ["Reply", "ThroughHerGateway"]


@dataclass(frozen=True)
class Reply:
    """What a page answered."""

    status_code: int
    content: bytes
    url: str

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", errors="replace")

    def json(self) -> Any:
        return json.loads(self.content or b"null")


class ThroughHerGateway:
    """GET requests to public addresses, through her network gateway, saying who is asking in ``headers``."""

    def __init__(self, *, headers: dict[str, str], source: str, seconds: float = 20.0) -> None:
        self.headers, self.source, self.seconds = dict(headers), source, seconds

    async def get(self, url: str, *, params: dict[str, Any] | None = None, seconds: float | None = None) -> Reply:
        from core.runtime.network_gateway import get_network_gateway

        if params:
            url = f"{url}{'&' if '?' in url else '?'}{urllib.parse.urlencode(params)}"
        answer = await get_network_gateway().request_async(
            "GET", url, headers=self.headers, timeout=self.seconds if seconds is None else seconds, read_only=True,
            source=self.source, public_network_only=True)
        if not answer.get("status_code") and answer.get("error"):
            raise OSError(str(answer["error"]))
        return Reply(int(answer.get("status_code") or 0), bytes(answer.get("content") or b""), str(answer.get("url") or url))
