"""A topic says where an event goes. What it carries is a separate claim.

OpenHands has typed event classes across its runtime; CrewAI dispatches typed
events and catches schema mistakes before dispatch. Aura's bus is
`publish(topic: str, data: Any)`, so a consumer's only way to know what
arrives on a topic was to read every producer — and a producer that renamed a
key found out when something downstream stopped working.
"""

from __future__ import annotations

import asyncio

import pytest

from core.runtime.what_an_event_carries import (
    WhatATopicCarries,
    check,
    declare,
    forget_what_was_seen,
    the_declared,
    the_undeclared,
    violations,
)


@pytest.fixture(autouse=True)
def _fresh():
    forget_what_was_seen()
    yield
    forget_what_was_seen()


def test_an_undeclared_topic_passes_and_is_counted():
    """Silence about a topic is not a claim that anything is wrong with it."""

    ok, why = check("nobody.declared.this", {"anything": 1})
    assert ok and not why
    assert the_undeclared().get("nobody.declared.this") == 1


def test_a_declared_topic_is_checked_on_every_key():
    declare(
        WhatATopicCarries(
            topic="a.test.topic",
            version="1",
            requires={"who": str, "how_many": int},
            allows={"why": str},
        )
    )
    assert check("a.test.topic", {"who": "someone", "how_many": 2})[0]
    assert check("a.test.topic", {"who": "someone", "how_many": 2, "why": "because"})[0]

    ok, why = check("a.test.topic", {"who": "someone"})
    assert not ok and "how_many is missing" in why

    ok, why = check("a.test.topic", {"who": 1, "how_many": 2})
    assert not ok and any("who is int" in one for one in why)

    ok, why = check("a.test.topic", {"who": "s", "how_many": 2, "surprise": 1})
    assert not ok and any("surprise is not declared" in one for one in why)


def test_an_open_topic_allows_what_it_did_not_name():
    declare(
        WhatATopicCarries(
            topic="an.open.topic", version="1", requires={"kind": str}, closed=False
        )
    )
    assert check("an.open.topic", {"kind": "a", "whatever": 1})[0]
    assert not check("an.open.topic", {"whatever": 1})[0]


def test_a_declared_topic_carrying_something_that_is_not_a_mapping_is_a_fault():
    declare(WhatATopicCarries(topic="mapping.only", version="1", requires={"a": str}))
    ok, why = check("mapping.only", ["not", "a", "mapping"])
    assert not ok and "declared a mapping" in why[0]


def test_the_shapes_load_themselves_on_the_first_check():
    """Nothing imports the declarations for their own sake, so the check does."""

    check("anything at all", {})
    named = the_declared()
    assert "turn_recorded" in named, sorted(named)
    assert named["turn_recorded"].requires, "the declaration carries no requirements"


def test_the_bus_records_a_violation_and_delivers_the_event_anyway():
    """A bus that raises inside publish turns a schema mistake into a crash."""

    from core.event_bus import get_event_bus

    async def go() -> None:
        bus = get_event_bus()
        # turn_recorded requires turn_id; this payload has none.
        await bus.publish(
            "turn_recorded", {"role": "user", "content": "hi", "session_id": "s"}
        )

    asyncio.run(go())
    caught = [one for one in violations() if one["topic"] == "turn_recorded"]
    assert caught, "the bus did not check a declared topic"
    assert "turn_id is missing" in caught[-1]["reasons"]


def test_violations_are_a_ring_rather_than_a_leak():
    from core.runtime.what_an_event_carries import HOW_MANY_VIOLATIONS_ARE_KEPT

    declare(WhatATopicCarries(topic="a.noisy.topic", version="1", requires={"a": str}))
    for _ in range(HOW_MANY_VIOLATIONS_ARE_KEPT + 20):
        check("a.noisy.topic", {})
    assert len(violations()) == HOW_MANY_VIOLATIONS_ARE_KEPT
