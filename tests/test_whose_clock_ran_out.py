"""A timeout inside a skill is not the skill running out of time.

``asyncio.timeout`` raises TimeoutError when it expires, and so does every
bounded wait inside the skill. Catching the type and reporting "skill timed
out" makes them indistinguishable, and they send a reader to different clocks.

Live 2026-08-30: os_automation declares 90s, failed at 35.6s, and the log said
it timed out — so the search went to the skill's budget, the engine's per-request
sizing and the executive constraints, none of which had anything to do with it.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from core.runtime.errors import _raise_site
from core.skills.base_skill import BaseSkill


class _ItsOwnBudget(BaseSkill):
    name = "its own budget"
    description = "sleeps past the budget it declared"
    timeout_seconds = 0.2

    async def execute(self, params, context):  # noqa: D102
        await asyncio.sleep(5)
        return {"ok": True}


class _SomethingInside(BaseSkill):
    name = "something inside"
    description = "has plenty of budget and a short wait within it"
    timeout_seconds = 30.0

    async def execute(self, params, context):  # noqa: D102
        async with asyncio.timeout(0.1):
            await asyncio.sleep(5)
        return {"ok": True}


async def _what_it_said(skill, caplog):
    with caplog.at_level(logging.WARNING):
        await skill.safe_execute({}, {})
    return " ".join(one.getMessage() for one in caplog.records)


@pytest.mark.asyncio
async def test_a_skill_that_ran_out_its_own_budget_says_so(caplog):
    said = await _what_it_said(_ItsOwnBudget(), caplog)
    assert "ran out its own" in said
    assert "INSIDE" not in said


@pytest.mark.asyncio
async def test_a_timeout_within_a_skill_is_not_reported_as_the_skill_timing_out(caplog):
    said = await _what_it_said(_SomethingInside(), caplog)
    assert "INSIDE it" in said
    assert "30.0s budget" in said
    assert "raised at" in said


def test_a_raise_site_names_this_codebase_rather_than_the_standard_library(monkeypatch):
    """A bounded wait expiring raises inside asyncio.timeouts, and naming that
    says only that a timeout was a timeout."""
    import core.runtime.errors as errors

    async def waits_here():
        async with asyncio.timeout(0.01):
            await asyncio.sleep(5)

    try:
        asyncio.run(waits_here())
    except TimeoutError as exc:
        caught = exc

    assert "asyncio" in _raise_site(caught)
    monkeypatch.setattr(
        errors, "_OUR_PACKAGES", frozenset({*errors._OUR_PACKAGES, __name__.split(".")[0]})
    )
    assert "waits_here" in errors._raise_site(caught)


def test_a_raise_site_falls_back_to_the_innermost_frame_it_has():
    """Where nothing of ours is on the stack, the innermost frame is honest."""
    try:
        int("not a number")
    except ValueError as exc:
        assert _raise_site(exc) != "unknown"
