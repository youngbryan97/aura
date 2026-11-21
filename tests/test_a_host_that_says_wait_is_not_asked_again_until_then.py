"""A host that has said how long to wait is not asked again until then (core/runtime/what_a_host_has_left.py).

Measured at Wikimedia: seven image topics asked in a tight loop, and the last five came back "no image available"
(core/skills/computer_use.py). Each one asked again of a host that had already said no.
"""
from __future__ import annotations

import email.utils
import io
import urllib.error

import pytest

import core.runtime.network_gateway as network_gateway
import core.runtime.what_a_host_has_left as host_supply

NOW = 1_795_300_000.0


@pytest.fixture(autouse=True)
def _fresh() -> None:
    host_supply.forget_hosts_for_test()


def test_a_refusal_with_a_wait_holds_the_host_until_then() -> None:
    host_supply.heard_from("https://commons.wikimedia.org/w/api.php?x=1", 429, {"Retry-After": "30"}, now=NOW)
    held = host_supply.held_back("https://commons.wikimedia.org/w/api.php?x=2", now=NOW + 10)
    assert held is not None and held["status_code"] == 429 and held["retry_after_s"] == pytest.approx(20)
    assert host_supply.held_back("https://commons.wikimedia.org/other", now=NOW + 31) is None
    # Another host is asked as before.
    assert host_supply.held_back("https://en.wikipedia.org/w/api.php", now=NOW + 10) is None


def test_a_wait_written_as_a_date_is_read() -> None:
    when = email.utils.formatdate(NOW + 120, usegmt=True)
    host_supply.heard_from("https://api.example.org/a", 503, {"retry-after": when}, now=NOW)
    held = host_supply.held_back("https://api.example.org/b", now=NOW + 60)
    assert held is not None and held["retry_after_s"] == pytest.approx(60, abs=1)


def test_none_left_holds_until_the_count_resets() -> None:
    host_supply.heard_from("https://api.github.com/x", 200, {"X-RateLimit-Remaining": "0", "X-RateLimit-Limit": "60",
                                                             "X-RateLimit-Reset": str(int(NOW + 600))}, now=NOW)
    assert host_supply.held_back("https://api.github.com/y", now=NOW + 300) is not None
    assert host_supply.what_each_host_has_left()["api.github.com"].allows == 60


def test_some_left_holds_nothing_and_a_refusal_without_a_wait_holds_nothing() -> None:
    host_supply.heard_from("https://api.example.org/a", 200, {"RateLimit-Remaining": "4", "RateLimit-Reset": "60"},
                           now=NOW)
    assert host_supply.held_back("https://api.example.org/a", now=NOW + 1) is None
    assert host_supply.what_each_host_has_left()["api.example.org"].left == 4
    host_supply.heard_from("https://quiet.example.org/a", 429, {}, now=NOW)
    assert host_supply.held_back("https://quiet.example.org/a", now=NOW + 1) is None


def test_the_gateway_does_not_send_what_the_host_already_refused(monkeypatch) -> None:
    sent: list[str] = []

    def refusing(request, timeout=None):
        sent.append(request.full_url)
        raise urllib.error.HTTPError(request.full_url, 429, "Too Many Requests", {"Retry-After": "120"},
                                     io.BytesIO(b""))

    monkeypatch.setattr(network_gateway.urllib.request, "urlopen", refusing)
    gateway = network_gateway.NetworkGateway()
    first = gateway.request("GET", "https://upload.wikimedia.org/a.png", source="test", read_only=True)
    second = gateway.request("GET", "https://upload.wikimedia.org/b.png", source="test", read_only=True)
    assert first["status_code"] == 429 and second["status_code"] == 429
    assert second.get("held_back") is True
    assert sent == ["https://upload.wikimedia.org/a.png"]
