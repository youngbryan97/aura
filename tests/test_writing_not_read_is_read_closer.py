"""Writing that did not read as words is read again from its own place, larger; and what moved by itself was not sent.

LIVE 2026-10-09 a game's title said START in outlined letters. The reading of
the whole screen made it "sitarit", so she saw no way on; read again by itself,
it is "Start". And the word bobbing on that title was taken for a thing her
press had sent.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from core.language.a_way_on import offers_a_way_on
from core.perception import reading_closer
from core.perception.reading_closer import CLOSER_AT_MOST, read_closer

pytestmark = pytest.mark.unit


def _piece(text: str, x: float = 0.7, y: float = 0.8) -> dict:
    return {"text": text, "x": x, "y": y, "width": 0.12, "height": 0.05, "center_x": x + 0.06, "center_y": y + 0.025}


def _picture(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 255, size=(200, 300, 3), dtype=np.uint8)


def test_a_label_misread_is_read_again_from_its_own_place_and_is_then_a_way_on():
    asked: list[tuple[int, int]] = []

    def closer(patch):
        asked.append(patch.shape[:2])
        return [{"text": "Start"}]

    seen = read_closer(_picture(1), [_piece("sitarit"), _piece("Candy Machine", y=0.2)], closer)
    assert [piece["text"] for piece in seen] == ["Start", "Candy Machine"]
    assert seen[0]["read_closer_from"] == "sitarit"
    assert offers_a_way_on(" ".join(piece["text"] for piece in seen))
    # Larger than the piece's own place: it was enlarged before it was read.
    assert asked and asked[0][0] > 0.05 * 200 * 2


def test_what_a_closer_reading_found_is_remembered_and_one_that_is_not_words_changes_nothing():
    calls: list[int] = []

    def closer(patch):
        calls.append(1)
        return [{"text": "Xqzv"}]

    picture = _picture(2)
    first = read_closer(picture, [_piece("BIEADELS")], closer)
    again = read_closer(picture, [_piece("BIEADELS")], closer)
    assert first[0]["text"] == again[0]["text"] == "BIEADELS"
    assert len(calls) == 1


def test_only_a_few_pieces_are_read_closer_in_one_look_and_never_prose_or_words():
    calls: list[int] = []

    def closer(patch):
        calls.append(1)
        return [{"text": "Next"}]

    pieces = [_piece(f"qwzx{i}", y=i / 12) for i in range(CLOSER_AT_MOST + 3)]
    pieces += [_piece("Play"), _piece("Using tubes and an assortment of offbeat devices, get it in")]
    seen = read_closer(_picture(3), pieces, closer)
    assert len(calls) == CLOSER_AT_MOST
    assert seen[-2]["text"] == "Play" and seen[-1]["text"].startswith("Using tubes")


def test_a_thing_already_going_before_the_press_was_not_sent_by_it():
    from core.agency.playing_by_shots import _where_it_went

    bobbing = SimpleNamespace(number=7, x=200.0, y=160.0, vx=40.0, vy=0.0)
    moves = SimpleNamespace(things={7: bobbing}, shape=(200, 300), see=lambda picture, at: [])
    clock = iter(i * 0.1 for i in range(200))

    async def look():
        return (None, next(clock))

    went, _gained = asyncio.run(_where_it_went(look, moves, 0.0, None, None, going_already=frozenset({7})))
    assert went is None
    clock = iter(i * 0.1 for i in range(200))
    went, _gained = asyncio.run(_where_it_went(look, moves, 0.0, None, None))
    assert went is not None


@pytest.fixture(autouse=True)
def _forget_closer_readings():
    reading_closer._remembered.clear()
    yield
    reading_closer._remembered.clear()
