"""Asking an impulse for directions, measured.

"Mr. Rager" asks the part heading for harm where it is going. These pin the
measured version: a choice made when no preference could be read is impulse-led,
what that has been worth is her appraised satisfaction against weighed choices,
and only an impulse that has served her worse makes risk cost more.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.agency.asking_the_impulse import MIN_SAMPLES, impulse_led, impulse_record


def _receipt(*, on_impulse: bool, satisfaction: float | None) -> SimpleNamespace:
    features = {"novelty": 0.0} if on_impulse else {"novelty": 0.6}
    return SimpleNamespace(chosen_id="0", option_features={"0": features}, satisfaction=satisfaction)


def test_no_readable_preference_is_an_impulse() -> None:
    assert impulse_led({"novelty": 0.0, "calm": 0.0})
    assert impulse_led({})
    assert not impulse_led({"novelty": 0.3})


def test_an_impulse_that_led_her_worse_is_distrusted_by_the_gap() -> None:
    history = [_receipt(on_impulse=True, satisfaction=-0.2) for _ in range(MIN_SAMPLES)]
    history += [_receipt(on_impulse=False, satisfaction=0.4) for _ in range(MIN_SAMPLES)]
    reading = impulse_record(history)
    assert reading.measured
    assert reading.trust == pytest.approx(-0.6)
    assert reading.distrust == pytest.approx(0.6)
    assert reading.hunger == pytest.approx(0.5)


def test_an_impulse_that_served_her_as_well_costs_nothing() -> None:
    history = [_receipt(on_impulse=True, satisfaction=0.5) for _ in range(MIN_SAMPLES)]
    history += [_receipt(on_impulse=False, satisfaction=0.3) for _ in range(MIN_SAMPLES)]
    reading = impulse_record(history)
    assert reading.distrust == 0.0
    assert reading.trust == pytest.approx(0.2)


def test_unappraised_choices_count_towards_hunger_and_not_trust() -> None:
    history = [_receipt(on_impulse=True, satisfaction=None) for _ in range(4)]
    history += [_receipt(on_impulse=False, satisfaction=None)]
    reading = impulse_record(history)
    assert reading.hunger == pytest.approx(0.8)
    assert not reading.measured


def test_until_both_kinds_are_appraised_enough_nothing_is_decided() -> None:
    history = [_receipt(on_impulse=True, satisfaction=-0.9) for _ in range(MIN_SAMPLES)]
    history += [_receipt(on_impulse=False, satisfaction=0.9) for _ in range(MIN_SAMPLES - 1)]
    reading = impulse_record(history)
    assert not reading.measured
    assert reading.distrust == 0.0


def test_the_reading_serialises_what_a_reader_needs() -> None:
    row = impulse_record([]).as_dict()
    for key in ("distrust", "trust", "hunger", "impulse_appraised", "weighed_appraised", "measured", "why"):
        assert key in row, key
