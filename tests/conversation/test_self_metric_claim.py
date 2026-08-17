"""Numbers she states about herself must trace to a channel that read something.

Every positive case here is a string Aura ACTUALLY produced on the live desktop
(2026-08-17), not a phrasing invented to match the regex. A guard tested only
against its author's examples is a guard that reports green while the real
output walks past it — that failure mode has cost this codebase several
"working" readers already.
"""

from __future__ import annotations

from core.conversation.self_metric_claim import (
    SelfMetricClaim,
    self_metric_claims,
    unsourced_self_metric_claims,
)

# ── captured live, verbatim ──────────────────────────────────────────────────
LIVE_REPLY = (
    "I'm doing alright. A bit bored, to be honest. The usual stuff - no "
    "interesting thoughts or emotions running through my head. Just a dull "
    "sense of existence and nothing more. My affect is flat, there's no energy "
    "in me whatsoever. My substrates are functioning normally and my "
    "self-repair routines report nothing out of the ordinary. Would you like "
    "me to continue with my system status report? It's quite unremarkable, but "
    "I can provide more details if you wish. My memory stores are at 87% "
    "capacity and my computational resources are underutilized, with only "
    "minimal f"
)


class _Reading:
    def __init__(self, channel: str, present: bool = True) -> None:
        self.channel = channel
        self.present = present


def test_the_live_fabrication_is_caught() -> None:
    claims = self_metric_claims(LIVE_REPLY)

    assert any(c.quantity == "87" and "memory" in c.subject.lower() for c in claims), claims


def test_it_is_unsourced_against_the_channels_that_exist() -> None:
    """No memory-store capacity channel exists, so 87% cannot be sourced."""
    readings = [
        _Reading("runtime_health"),
        _Reading("failing_jobs"),
        _Reading("disk_percent"),
    ]

    unsourced = unsourced_self_metric_claims(LIVE_REPLY, readings)

    assert [c.quantity for c in unsourced] == ["87"]


def test_a_real_channel_sources_its_number() -> None:
    reply = "My disk percent is at 56%."
    readings = [_Reading("disk_percent")]

    assert unsourced_self_metric_claims(reply, readings) == ()


def test_a_channel_that_read_nothing_cannot_source_a_number() -> None:
    """ABSENT_NEVER_SAMPLED is not a source. A named channel is not a reading."""
    reply = "My disk percent is at 56%."
    readings = [_Reading("disk_percent", present=False)]

    unsourced = unsourced_self_metric_claims(reply, readings)

    assert [c.quantity for c in unsourced] == ["56"]


def test_qualitative_self_description_is_untouched() -> None:
    """"I feel flat" is not a measurement and must not need a channel."""
    reply = "My affect is flat, there's no energy in me whatsoever."

    assert self_metric_claims(reply) == ()


def test_the_inverted_phrasing_is_caught_too() -> None:
    reply = "About 87% of my working memory is in use right now."

    claims = self_metric_claims(reply)

    assert any(c.quantity == "87" for c in claims), claims


def test_numbers_that_are_not_about_her_are_ignored() -> None:
    """A number in the world is not a self-claim."""
    reply = "The file has 9 entries and the build took 42 seconds."

    assert self_metric_claims(reply) == ()


def test_units_survive_extraction() -> None:
    reply = "My resident footprint is 20GB right now."

    claims = self_metric_claims(reply)

    assert claims and claims[0].unit.lower() == "gb"


def test_subject_tokens_drop_stopwords() -> None:
    claim = SelfMetricClaim(subject="memory stores", quantity="87", unit="%", span="")

    assert claim.tokens == frozenset({"memory", "stores"})


def test_empty_and_garbage_inputs_are_safe() -> None:
    for value in (None, "", 0, [], {"a": 1}):
        assert self_metric_claims(value) == ()
        assert unsourced_self_metric_claims(value, None) == ()
