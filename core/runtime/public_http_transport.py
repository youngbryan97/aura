"""Canonical transport owner for bounded requests to the public web."""

from __future__ import annotations

from typing import Any

from core.runtime.network_gateway import get_network_gateway

_METHODS = frozenset({"DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"})
_READ_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
_MAX_RESPONSE_BYTES = 16 * 1024 * 1024


async def request_public_http(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    data: bytes | str | None = None,
    timeout_s: float = 30.0,
    source: str,
    max_response_bytes: int,
) -> dict[str, Any]:
    """Make one governed, redirect-safe, size-bounded public HTTP request.

    The caller cannot label a mutating verb as read-only or opt into private
    address space. NetworkGateway pins each DNS answer and repeats admission
    for every redirect, so preflight resolution cannot be swapped before the
    socket connects.
    """

    verb = str(method or "").strip().upper()
    if verb not in _METHODS:
        raise ValueError(f"unsupported HTTP method: {verb or '<empty>'}")
    owner = str(source or "").strip()
    if not owner:
        raise ValueError("network source identity is required")
    if isinstance(max_response_bytes, bool) or not isinstance(max_response_bytes, int):
        raise TypeError("max_response_bytes must be an integer")
    if not 1 <= max_response_bytes <= _MAX_RESPONSE_BYTES:
        raise ValueError(
            f"max_response_bytes must be between 1 and {_MAX_RESPONSE_BYTES}"
        )

    return await get_network_gateway().request_async(
        verb,
        url,
        headers=headers,
        data=data,
        timeout=timeout_s,
        source=owner,
        read_only=verb in _READ_METHODS,
        max_response_bytes=max_response_bytes,
        public_network_only=True,
    )


__all__ = ["request_public_http"]
