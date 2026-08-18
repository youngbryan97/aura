"""Capability present, content never supplied — the defect this registry closes.

Four incidents, one shape. Each was fixed on its own before the pattern was
obvious:

    "how many .py files in core/introspection"  ->  "There are 3"      (ten)
    "read CONTRIBUTING.md"                      ->  "I tried and failed"
                                                    (nothing ran)
    "what's on my clipboard right now?"         ->  "I can only work with the
                                                     information you provide"
                                                    (it held BUILD-7741-verify)
    "what time is it"                           ->  "my clock says 06:15 and the
                                                     ambient light sensors report
                                                     low illumination"  (at 01:40,
                                                     with no light sensor)

The capability was registered every time and the reading was cheap every time.
Nothing took the reading before the answer was composed, and a model asked
about a fact it does not hold produces something fact-shaped.
"""

from __future__ import annotations

import asyncio

import pytest

import core.brain.observable_registry as registry
from core.brain.observable_grounding import (
    Observable,
    OBSERVABLES,
    observable_blocks,
    register_observable,
)


def _blocks(prompt: str) -> list[str]:
    return asyncio.run(observable_blocks(prompt))


# ── the registry is populated and routes ────────────────────────────────────

def test_the_expected_observables_are_registered() -> None:
    names = set(registry.observable_names())

    assert {"clipboard", "file", "file_count", "corpus", "clock"} <= names


@pytest.mark.parametrize(
    ("prompt", "header"),
    [
        ("read CONTRIBUTING.md", "## FILE YOU WERE ASKED ABOUT"),
        (
            "how many python files are in core/introspection",
            "## DIRECTORY LISTING YOU WERE ASKED ABOUT",
        ),
        ("what time is it?", "## THE CURRENT LOCAL TIME"),
    ],
)
def test_a_question_gets_its_reading(prompt: str, header: str) -> None:
    blocks = _blocks(prompt)

    assert any(block.startswith(header) for block in blocks), blocks


def test_a_conversational_turn_takes_no_readings() -> None:
    """Readers are privacy- and latency-relevant; they are never ambient."""
    assert _blocks("how are you doing today?") == []


def test_the_directory_listing_carries_the_real_count() -> None:
    from pathlib import Path

    truth = len(list((Path(__file__).resolve().parents[2] / "core" / "introspection").glob("*.py")))
    blocks = _blocks("how many python files are in core/introspection")

    assert any(str(truth) in block for block in blocks)


# ── the mechanism's own contract ────────────────────────────────────────────

def test_a_failing_reader_does_not_break_the_turn() -> None:
    async def _boom(_prompt: str) -> str:
        raise OSError("device unavailable")

    register_observable(
        Observable("test_broken", "## BROKEN", lambda _p: True, _boom)
    )
    try:
        assert _blocks("anything at all") == [] or all(
            "## BROKEN" not in block for block in _blocks("anything")
        )
    finally:
        OBSERVABLES[:] = [o for o in OBSERVABLES if o.name != "test_broken"]


def test_a_slow_reader_cannot_hold_the_turn() -> None:
    async def _slow(_prompt: str) -> str:
        await asyncio.sleep(5)
        return "too late"

    register_observable(
        Observable("test_slow", "## SLOW", lambda _p: True, _slow, timeout_s=0.15)
    )
    try:
        import time

        started = time.monotonic()
        blocks = _blocks("anything")
        elapsed = time.monotonic() - started

        assert elapsed < 3.0
        assert all("## SLOW" not in block for block in blocks)
    finally:
        OBSERVABLES[:] = [o for o in OBSERVABLES if o.name != "test_slow"]


def test_a_raising_matcher_is_survivable() -> None:
    def _bad_matcher(_prompt: str) -> bool:
        raise ValueError("matcher exploded")

    async def _never(_prompt: str) -> str:
        return "unreachable"

    register_observable(Observable("test_bad", "## BAD", _bad_matcher, _never))
    try:
        assert all("## BAD" not in block for block in _blocks("anything"))
    finally:
        OBSERVABLES[:] = [o for o in OBSERVABLES if o.name != "test_bad"]


def test_an_empty_reading_produces_no_block() -> None:
    async def _empty(_prompt: str) -> str:
        return "   "

    register_observable(Observable("test_empty", "## EMPTY", lambda _p: True, _empty))
    try:
        assert all("## EMPTY" not in block for block in _blocks("anything"))
    finally:
        OBSERVABLES[:] = [o for o in OBSERVABLES if o.name != "test_empty"]


def test_two_observables_in_one_question_both_answer() -> None:
    blocks = _blocks(
        "what's on my clipboard, and how many python files are in core/introspection?"
    )
    headers = {block.split("\n")[0] for block in blocks}

    assert "## DIRECTORY LISTING YOU WERE ASKED ABOUT" in headers
    assert "## WHAT IS ON THE CLIPBOARD" in headers


def test_registering_the_same_name_replaces_rather_than_duplicates() -> None:
    async def _one(_prompt: str) -> str:
        return "first"

    async def _two(_prompt: str) -> str:
        return "second"

    register_observable(Observable("test_dup", "## DUP", lambda _p: True, _one))
    register_observable(Observable("test_dup", "## DUP", lambda _p: True, _two))
    try:
        dup = [block for block in _blocks("anything") if block.startswith("## DUP")]

        assert len(dup) == 1
        assert "second" in dup[0]
    finally:
        OBSERVABLES[:] = [o for o in OBSERVABLES if o.name != "test_dup"]
