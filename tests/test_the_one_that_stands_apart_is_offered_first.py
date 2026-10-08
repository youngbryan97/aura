"""Of a row of alike buttons, the one whose colour stands apart is offered first, and a lone number is a choice.

LIVE 2026-10-08 a game's level select had one bright level, "1", among
twenty-three locked grey ones. The number was rated a thing to read and the
shapes ran together into one; she clicked the row captions for four minutes.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.language.a_way_on import how_much_it_leads_on
from core.perception.shapes_that_look_pressable import STANDS_OUT, pressable_shapes


def _a_row_of_levels(lit: int) -> np.ndarray:
    picture = np.full((360, 540, 3), (200, 170, 90), dtype=np.uint8)  # a wooden floor
    for n in range(6):
        cx, cy = 120 + n * 60, 180
        yy, xx = np.ogrid[:360, :540]
        disc = (xx - cx) ** 2 + (yy - cy) ** 2 <= 22 ** 2
        picture[disc] = (230, 60, 200) if n == lit else (130, 130, 140)
        picture[(xx - cx) ** 2 + (yy - cy) ** 2 <= 6 ** 2] = (40, 40, 40)  # each one's keyhole or number
    return picture


@pytest.mark.unit
def test_the_lit_one_of_a_row_stands_apart_and_comes_first():
    shapes = pressable_shapes(_a_row_of_levels(lit=0))
    assert shapes and shapes[0]["text"].startswith(STANDS_OUT) and shapes[0].get("stands_apart")
    assert 0.18 <= shapes[0]["center_x"] <= 0.27


@pytest.mark.unit
def test_a_row_all_alike_has_none_apart():
    assert not any(s.get("stands_apart") for s in pressable_shapes(_a_row_of_levels(lit=-1)))


@pytest.mark.unit
def test_a_lone_short_number_is_a_choice_and_the_one_apart_a_likely_way_on():
    assert how_much_it_leads_on("1") > 0.7  # offered where keys do nothing
    assert how_much_it_leads_on("00003900") < 0.7 and how_much_it_leads_on("0") < 0.7
    assert how_much_it_leads_on(f"{STANDS_OUT} 20% across, 50% down") > how_much_it_leads_on("the shape at 20% across, 50% down")
