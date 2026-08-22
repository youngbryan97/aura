"""Every HTTPS request was failing in forty-eight milliseconds.

LIVE, 2026-08-22: asked to read a paper at a PubMed Central address, the fetch
dispatched and came straight back with "'_PublicPinnedHTTPSHandler' object has
no attribute '_check_hostname'". The pinned handler overrides https_open and
read an attribute urllib's HTTPSHandler no longer keeps — the SSL context
carries that setting now — so no page anyone named could be read at all.
"""

from __future__ import annotations

import ssl
import urllib.request
from pathlib import Path

from core.runtime.network_gateway import _PublicPinnedHTTPSHandler


def test_the_stdlib_no_longer_keeps_the_attribute_this_read():
    """The reason the override broke, stated as a check rather than a memory."""
    handler = urllib.request.HTTPSHandler(context=ssl.create_default_context())
    assert hasattr(handler, "_context")
    assert not hasattr(handler, "_check_hostname")


def test_the_pinned_handler_does_not_read_it_either():
    handler = _PublicPinnedHTTPSHandler(context=ssl.create_default_context())
    assert hasattr(handler, "_context")
    source = Path("core/runtime/network_gateway.py").read_text(encoding="utf-8")
    block = source[source.index("class _PublicPinnedHTTPSHandler") :]
    block = block[: block.index("def _build_public_pinned_opener")]
    assert "self._check_hostname" not in block


def test_the_pinned_handler_still_pins_and_validates():
    """The point of the override survives the fix."""
    source = Path("core/runtime/network_gateway.py").read_text(encoding="utf-8")
    block = source[source.index("class _PublicPinnedHTTPSHandler") :]
    block = block[: block.index("def _build_public_pinned_opener")]
    assert "_resolve_public_http_address" in block
    assert "_PinnedHTTPSConnection" in block
    assert "context=self._context" in block
