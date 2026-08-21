"""Fetching a URL: what it will and will not reach.

LIVE GAP, 2026-08-20. Asked to work out how to call an API from its endpoint,
she answered "let's make a request" — because among seventy-five skills none
could make one.
"""

from __future__ import annotations

import asyncio

import pytest

from core.executive.execution_policy import resolve_execution_effect_scope
from core.skills.http_request import HttpRequestSkill, check_url


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:8000/api/skills",
        "http://127.0.0.1:8000/",
        "http://[::1]:8000/",
        "http://169.254.169.254/latest/meta-data/",
        "http://metadata.google.internal/",
    ],
)
def test_this_machine_is_not_the_web(url: str) -> None:
    """The runtime's own API listens on loopback; a link must not reach it."""
    fetched, reason = check_url(url)
    assert not fetched
    assert reason


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.com/x", "/etc/passwd", ""])
def test_only_http_is_fetched(url: str) -> None:
    assert check_url(url)[0] == ""


def test_a_public_url_is_allowed() -> None:
    fetched, reason = check_url("https://example.com/docs")
    assert reason == ""
    assert fetched


def test_a_refused_url_never_reaches_the_network() -> None:
    result = asyncio.run(HttpRequestSkill().execute({"url": "http://127.0.0.1:9/"}))
    assert result["ok"] is False
    assert "own network" in result["error"]


@pytest.mark.asyncio
async def test_request_uses_the_pinned_public_transport_with_query_and_json(
    monkeypatch,
) -> None:
    from core.skills import http_request as module

    calls: list[tuple[tuple, dict]] = []

    async def transport(*args, **kwargs):
        calls.append((args, kwargs))
        return {
            "ok": True,
            "status_code": 201,
            "url": "https://example.com/final",
            "headers": {"Content-Type": "application/json"},
            "content": b'{"accepted":true}',
        }

    monkeypatch.setattr(module, "check_url", lambda url: (url, ""))
    monkeypatch.setattr(module, "request_public_http", transport)

    result = await module.HttpRequestSkill().execute(
        {
            "url": "https://example.com/items?existing=1",
            "method": "POST",
            "params": {"kind": "orca", "tag": ["a", "b"]},
            "json_body": {"enabled": True},
        }
    )

    assert result["ok"] is True
    assert result["json"] == {"accepted": True}
    assert result["content_type"] == "application/json"
    assert calls[0][0] == (
        "POST",
        "https://example.com/items?existing=1&kind=orca&tag=a&tag=b",
    )
    assert calls[0][1]["data"] == b'{"enabled":true}'
    assert calls[0][1]["headers"]["Content-Type"] == "application/json"
    assert calls[0][1]["max_response_bytes"] == module.MAX_BODY_BYTES


@pytest.mark.asyncio
async def test_response_limit_failure_is_reported_as_partial_not_complete(
    monkeypatch,
) -> None:
    from core.skills import http_request as module

    monkeypatch.setattr(module, "check_url", lambda url: (url, ""))

    async def transport(*_args, **_kwargs):
        return {
            "ok": False,
            "status_code": 200,
            "url": "https://example.com/large",
            "headers": {"content-type": "text/plain"},
            "content": b"bounded prefix",
            "error": "response_body_exceeds_limit",
        }

    monkeypatch.setattr(module, "request_public_http", transport)

    result = await module.HttpRequestSkill().execute(
        {"url": "https://example.com/large"}
    )

    assert result["ok"] is False
    assert result["truncated"] is True
    assert result["error"] == "response_body_exceeds_limit"
    assert result["text"] == "bounded prefix"


def test_reading_is_scoped_as_reading_and_writing_is_not() -> None:
    """The point of the per-action table: a GET is not an external effect."""
    assert resolve_execution_effect_scope("http_request", {"method": "GET"}) == "read_only"
    assert resolve_execution_effect_scope("http_request", {"method": "HEAD"}) == "read_only"
    for method in ("POST", "PUT", "PATCH", "DELETE"):
        assert resolve_execution_effect_scope("http_request", {"method": method}) == "external_io"


def test_a_reader_is_scoped_by_its_action_without_a_branch_of_its_own() -> None:
    """The general fallthrough, checked on the skill that motivated it.

    Reading a file outside the workspace used to resolve to state_mutation,
    because the only per-action rule was a workspace-specific special case.
    """
    assert resolve_execution_effect_scope("file_operation", {"action": "read", "path": "/etc/hosts"}) == "read_only"
    assert resolve_execution_effect_scope("file_operation", {"action": "delete", "path": "/etc/hosts"}) == "state_mutation"


def test_the_skill_is_in_the_catalogue() -> None:
    """A skill the catalogue rejects is a skill nobody can be offered."""
    from core.skills.discovery import build_skill_catalog

    names = {declaration.name for declaration in build_skill_catalog().accepted}
    assert "http_request" in names


def test_an_omitted_method_is_the_default_not_the_worst_case() -> None:
    """LIVE, 2026-08-20: the model chose http_request and was refused.

    It sent ``{"url": ...}`` with no method, meaning GET. Scope resolution
    read a field literally named "action", found nothing, and fell back to the
    skill's worst action — so the one call the turn was entitled to make was
    the one it was denied.
    """
    from core.skills.action_scope import (
        action_field_and_default,
        declared_action_name,
        skill_class_named,
    )

    target = skill_class_named("http_request")
    assert action_field_and_default(target) == ("method", "get")
    assert declared_action_name(target, {"url": "https://example.com"}) == "get"
    assert resolve_execution_effect_scope("http_request", {"url": "https://example.com"}) == "read_only"


def test_the_action_field_is_found_by_what_it_allows_not_by_its_name() -> None:
    """file_operation calls it `action`, http_request calls it `method`."""
    from core.skills.action_scope import action_field_and_default, skill_class_named

    assert action_field_and_default(skill_class_named("file_operation"))[0] == "action"
    assert action_field_and_default(skill_class_named("http_request"))[0] == "method"
