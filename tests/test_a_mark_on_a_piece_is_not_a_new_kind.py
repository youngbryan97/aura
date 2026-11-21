"""Once the kinds of piece are learned from a board at its start, a mark on a piece is not a kind of its own.

LIVE 20 November 2026 her record of En Croissant's board, which has six kinds of piece, held sixteen: most were looks
taken while the game marked a square. Each was kept as a kind, and the piece wearing it no longer matched the board she
held. The record was also too big to keep, and the store cut the oldest rows off each look.
"""
from __future__ import annotations

import json

import numpy as np
from PIL import Image, ImageDraw

from core.perception.pieces_by_their_look import LOOK_SIDE, Looks, squares_of_one_ground
from tests.fixtures.a_board_application import BoardApplication
from tests.fixtures.a_game_of_pieces_to_learn import CHESS


def _a_board_and_one_piece_marked() -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    app = BoardApplication(CHESS, seed=3)
    start = app.draw()
    square = next(iter(sorted(app.position.cells)))
    x, y = app._centre(square)
    marked = Image.fromarray(start.copy())
    dot = app.square * 0.3
    ImageDraw.Draw(marked).ellipse([x - dot, y - dot, x + dot, y + dot], fill=(90, 90, 90))
    row_seen = CHESS.rows - 1 - square[0]
    return start, np.asarray(marked), (row_seen, square[1])


def test_a_settled_reading_learns_no_kind_from_a_mark() -> None:
    start, marked, where = _a_board_and_one_piece_marked()
    looks = Looks()
    lattice = squares_of_one_ground(start)
    looks.read(start, lattice)
    kinds = len(looks.kinds)
    seen = looks.read(marked, lattice, settled=True)
    assert len(looks.kinds) == kinds
    side, kind = seen.cells[where[0]][where[1]]
    assert kind is None and where in seen.nearest
    # Learning as she goes, the same mark is a new kind: what the settled reading is for.
    looks.read(marked, lattice)
    assert len(looks.kinds) == kinds + 1


def test_looks_are_kept_whole_and_come_back_the_same() -> None:
    start, _marked, _where = _a_board_and_one_piece_marked()
    looks = Looks()
    looks.read(start, squares_of_one_ground(start))
    kept = json.loads(json.dumps(looks.as_memory()))
    assert all(isinstance(kind, str) for kind in kept["kinds"])
    back = Looks.from_memory(kept)
    assert len(back.kinds) == len(looks.kinds)
    assert all(kind.shape == (2 * LOOK_SIDE, LOOK_SIDE) for kind in back.kinds)
    assert max(float(np.abs(a - b).max()) for a, b in zip(back.kinds, looks.kinds)) <= 1 / 255 + 1e-6
    # Kept rows of numbers by an earlier sitting are still read back.
    assert len(Looks.from_memory({"kinds": [k.tolist() for k in looks.kinds]}).kinds) == len(looks.kinds)
