"""A refusal recorded beneath a caller must be visible to that caller.

LIVE, 2026-08-22. The router deferred every endpoint for a tier — 19.1GB free
against a 22GB requirement — recorded "no_endpoint_available_for_tier", and
returned an empty string. The task engine read the registry a moment later,
found nothing, and raised "LLM returned empty or None response". That was
reported as a planning FAILURE, which drove frustration to 1.00 and the
resilience state to strain. A full machine cascaded into a runtime that had
decided it was broken.

The record was made. The reader could not see it: a ContextVar set inside a
child task does not propagate back, because asyncio gives children a copy of
the context, and the router runs beneath its caller.
"""

from __future__ import annotations

import asyncio
import time

from core.brain.llm.deferral_record import (
    explain_empty_generation,
    record_deferral,
    reset_for_test,
    take_deferral,
)


def setup_function() -> None:
    reset_for_test()


def teardown_function() -> None:
    reset_for_test()


def test_a_deferral_recorded_in_a_child_task_reaches_the_parent():
    async def main() -> object:
        started = time.time()

        async def beneath() -> None:
            record_deferral(
                origin="autonomous_task_engine",
                reason="no_endpoint_available_for_tier:tertiary",
            )

        await asyncio.create_task(beneath())
        return take_deferral(origin="autonomous_task_engine", not_before=started)

    entry = asyncio.run(main())
    assert entry is not None
    assert "no_endpoint_available_for_tier" in entry.reason


def test_a_deferral_is_consumed_once():
    started = time.time()
    record_deferral(origin="o", reason="busy")
    assert take_deferral(origin="o", not_before=started) is not None
    assert take_deferral(origin="o", not_before=started) is None


def test_another_origin_does_not_take_it():
    started = time.time()
    record_deferral(origin="one", reason="busy")
    assert take_deferral(origin="two", not_before=started) is None
    assert take_deferral(origin="one", not_before=started) is not None


def test_a_deferral_from_before_the_call_is_not_this_call_s():
    record_deferral(origin="o", reason="busy")
    assert take_deferral(origin="o", not_before=time.time() + 5.0) is None


def test_a_stale_deferral_is_not_offered():
    record_deferral(origin="o", reason="busy")
    assert take_deferral(origin="o", not_before=0.0, now=time.time() + 3600.0) is None


def test_an_empty_generation_can_say_why():
    record_deferral(origin="router", reason="no_endpoint_available_for_tier:tertiary")
    said = explain_empty_generation()
    assert "deferred" in said
    assert "no_endpoint_available_for_tier" in said
