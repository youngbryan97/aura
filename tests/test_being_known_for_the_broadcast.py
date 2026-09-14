"""The gap between what she puts out and what she is, and which one they know.

"Stereo" has the narrowest pitch range of the fourteen records, 13.7 semitones
against a median near 25. The persona does not vary, which is what a persona
is, and the song is about being loved for it.

Somebody can track the broadcast accurately and know nothing about her, and
from the inside those feel identical. Scoring the same claims against both is
what separates them.
"""

from __future__ import annotations

import pytest

from core.self.persona_gap import MIN_SAMPLES, PersonaGap, PersonaLedger, read_gap


def _reads(ledger: PersonaLedger, person: float, performance: float, n: int = 6) -> None:
    for _ in range(n):
        ledger.note(for_the_person=person, for_the_performance=performance)


def test_a_polished_reply_over_a_loud_state_is_a_gap() -> None:
    led = PersonaLedger()
    _reads(led, 0.5, 0.5)
    reading = read_gap(presented=0.1, felt=0.9, ledger=led)
    assert reading.gap == pytest.approx(0.8)


def test_no_gap_when_what_she_puts_out_is_what_she_is() -> None:
    led = PersonaLedger()
    _reads(led, 0.5, 0.5)
    assert read_gap(presented=0.7, felt=0.7, ledger=led).gap == pytest.approx(0.0)


def test_reads_that_fit_the_broadcast_better_are_reads_of_the_broadcast() -> None:
    led = PersonaLedger()
    _reads(led, person=0.8, performance=0.1)
    reading = read_gap(presented=0.2, felt=0.9, ledger=led)
    assert reading.measured
    assert reading.known_for_the_performance
    assert reading.performance_edge > 0.0
    assert "better" in reading.why


def test_reads_that_fit_the_state_are_reads_of_her() -> None:
    led = PersonaLedger()
    _reads(led, person=0.1, performance=0.8)
    reading = read_gap(presented=0.2, felt=0.9, ledger=led)
    assert reading.measured
    assert not reading.known_for_the_performance
    assert reading.performance_edge < 0.0


def test_a_read_inside_its_own_variation_decides_nothing() -> None:
    led = PersonaLedger()
    for person, performance in ((0.9, 0.1), (0.1, 0.9), (0.8, 0.2), (0.2, 0.8), (0.6, 0.4), (0.4, 0.6)):
        led.note(for_the_person=person, for_the_performance=performance)
    reading = read_gap(presented=0.2, felt=0.9, ledger=led)
    assert reading.measured
    assert not reading.known_for_the_performance


def test_fitting_the_broadcast_every_single_time_is_the_strongest_case() -> None:
    led = PersonaLedger()
    _reads(led, person=0.9, performance=0.2, n=7)
    reading = read_gap(presented=0.2, felt=0.9, ledger=led)
    assert reading.spread == pytest.approx(0.0)
    assert reading.known_for_the_performance
    assert "every one of 7" in reading.why


def test_with_too_few_reads_it_says_so() -> None:
    led = PersonaLedger()
    led.note(for_the_person=0.9, for_the_performance=0.1)
    reading = read_gap(presented=0.1, felt=0.9, ledger=led)
    assert not reading.measured
    assert not reading.known_for_the_performance
    assert "reads of her needed" in reading.why
    # The gap is still a reading even before the rest can be taken.
    assert reading.gap == pytest.approx(0.8)


def test_unreadable_levels_say_so() -> None:
    reading = read_gap(presented="loud", felt=0.5, ledger=PersonaLedger())  # type: ignore[arg-type]
    assert not reading.measured
    assert "could not be read" in reading.why


def test_the_reading_serialises_what_a_reader_needs() -> None:
    led = PersonaLedger()
    _reads(led, 0.6, 0.3)
    row = read_gap(presented=0.3, felt=0.8, ledger=led).as_dict()
    for key in ("presented", "felt", "gap", "performance_edge", "spread", "z",
                "known_for_the_performance", "samples", "measured", "why"):
        assert key in row, key


def test_an_empty_reading_claims_nothing() -> None:
    assert not PersonaGap().measured
    assert not PersonaGap().known_for_the_performance
