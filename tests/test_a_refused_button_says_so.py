"""A control that does nothing must not look like one that worked.

LIVE, 2026-08-18. Pressing "Reboot Aura" and confirming the dialog did
nothing: /api/reboot is an owner-protected path and answered 401, the window
sat there unchanged, and no message appeared anywhere.

The handler awaited the fetch and never looked at the result. A refusal is a
SUCCESSFUL fetch with ok === false — it throws nothing — so the only branch
that could speak was the catch, whose message says the request failed "before
it reached the server". The case where it REACHED the server and was refused
had no branch at all.

That is indistinguishable, from the outside, from a reboot already under way.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

AURA_JS = Path(__file__).resolve().parents[1] / "interface/static/aura.js"
SOURCE = AURA_JS.read_text(encoding="utf-8")


def _reboot_handler() -> str:
    start = SOURCE.index("const rebootBtn")
    return SOURCE[start : start + 2400]


def test_the_reboot_response_is_inspected():
    handler = _reboot_handler()
    assert re.search(r"=\s*await fetch\('/api/reboot'", handler), (
        "the reboot response is discarded again"
    )
    assert "res.ok" in handler or "response.ok" in handler


def test_a_refusal_is_reported_to_the_person():
    handler = _reboot_handler()
    assert "401" in handler and "403" in handler
    assert "not signed in as the owner" in handler


def test_a_non_refusal_failure_is_also_reported():
    """Any status that is not ok has to say something."""
    handler = _reboot_handler()
    assert re.search(r"not accepted", handler)


def test_a_torn_down_connection_is_not_called_a_failure():
    """A reboot that IS happening kills the connection mid-flight.

    Asserting failure there would be as wrong as asserting success was.
    """
    handler = _reboot_handler()
    assert "may be restarting anyway" in handler


@pytest.mark.parametrize("path", ["/api/reboot"])
def test_the_endpoint_is_still_owner_protected(path):
    """The refusal being reported does not mean the refusal should stop."""
    auth = (AURA_JS.parents[1] / "auth.py").read_text(encoding="utf-8")
    assert path in auth
