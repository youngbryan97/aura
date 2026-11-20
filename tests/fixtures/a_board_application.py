"""A board application for tests, with rules she is never told: it draws its board, marks the squares a picked-up piece
may go to, tints the last move, refuses moves its rules do not allow, and answers with moves of its own.

It stands in for an application on her screen (core/skills/screen_pursuit_board_of_pieces.py hands her the real one's
pixels and clicks). Its pieces are drawn as plain shapes, shuffled per run, so nothing about how a piece looks says how
it moves.
"""
from __future__ import annotations

import asyncio
import math
import random

import numpy as np
from PIL import Image, ImageDraw

from core.agency.pieces_by_their_kind import FIRST, SECOND, Family, Move, Position, after, moves, outcome
from core.agency.searching_by_kind import search

SHAPES = ("circle", "triangle", "square", "diamond", "cross", "star", "bar", "ring")
LIGHT, DARK = (235, 236, 239), (139, 146, 155)
LAST_LIGHT, LAST_DARK = (205, 210, 120), (170, 175, 95)
PICKED = (120, 170, 120)


def _shape(draw: ImageDraw.ImageDraw, shape: str, x: float, y: float, s: float, fill: tuple, out: tuple) -> None:
    if shape == "circle":
        draw.ellipse([x - s, y - s, x + s, y + s], fill=fill, outline=out, width=3)
    elif shape == "triangle":
        draw.polygon([(x, y - s), (x - s, y + s), (x + s, y + s)], fill=fill, outline=out)
    elif shape == "square":
        draw.rectangle([x - s * 0.8, y - s * 0.8, x + s * 0.8, y + s * 0.8], fill=fill, outline=out, width=3)
    elif shape == "diamond":
        draw.polygon([(x, y - s), (x + s, y), (x, y + s), (x - s, y)], fill=fill, outline=out)
    elif shape == "cross":
        t = s / 3
        draw.polygon([(x - t, y - s), (x + t, y - s), (x + t, y - t), (x + s, y - t), (x + s, y + t), (x + t, y + t),
                      (x + t, y + s), (x - t, y + s), (x - t, y + t), (x - s, y + t), (x - s, y - t), (x - t, y - t)],
                     fill=fill, outline=out)
    elif shape == "star":
        points = []
        for i in range(10):
            radius = s if i % 2 == 0 else s * 0.45
            angle = math.pi / 2 + i * math.pi / 5
            points.append((x + radius * math.cos(angle), y - radius * math.sin(angle)))
        draw.polygon(points, fill=fill, outline=out)
    elif shape == "bar":
        draw.rectangle([x - s, y - s * 0.35, x + s, y + s * 0.35], fill=fill, outline=out, width=3)
    else:
        draw.ellipse([x - s, y - s, x + s, y + s], fill=fill, outline=out, width=3)
        draw.ellipse([x - s * 0.4, y - s * 0.4, x + s * 0.4, y + s * 0.4], fill=out)


class BoardApplication:
    """The application: its position, what is picked up, its last move. Her side is the first, drawn at the bottom."""

    def __init__(self, family: Family, *, she_goes_first: bool = True, square: int = 64, margin: int = 80,
                 answers_in_s: float = 0.05, seed: int = 0) -> None:
        self.family = family
        self.square = square
        self.margin = margin
        self.position = family.the_start()
        if not she_goes_first:
            self.position = Position(cells=self.position.cells, turn=SECOND, unmoved=self.position.unmoved)
        self.picked: tuple[int, int] | None = None
        self.last: set[tuple[int, int]] = set()
        self.answers_in_s = answers_in_s
        shapes = list(SHAPES)
        random.Random(seed).shuffle(shapes)
        self.shape = {kind: shapes[i % len(shapes)] for i, kind in enumerate(sorted(family.kinds))}
        self.wide = family.columns * square + 2 * margin
        self.tall = family.rows * square + 2 * margin
        self.clicks = 0
        self.made: list[Move] = []

    def _centre(self, square: tuple[int, int]) -> tuple[float, float]:
        row, column = square
        return (self.margin + column * self.square + self.square / 2,
                self.margin + (self.family.rows - 1 - row) * self.square + self.square / 2)

    def _square_at(self, x: float, y: float) -> tuple[int, int] | None:
        column = int((x - self.margin) // self.square)
        row_seen = int((y - self.margin) // self.square)
        if 0 <= column < self.family.columns and 0 <= row_seen < self.family.rows:
            return self.family.rows - 1 - row_seen, column
        return None

    def draw(self) -> np.ndarray:
        image = Image.new("RGB", (self.wide, self.tall), (28, 28, 32))
        draw = ImageDraw.Draw(image)
        offers = set()
        if self.picked is not None:
            offers = {m.end for m in moves(self.family, self.position) if m.start == self.picked}
        for row in range(self.family.rows):
            for column in range(self.family.columns):
                x0 = self.margin + column * self.square
                y0 = self.margin + (self.family.rows - 1 - row) * self.square
                light = (row + column) % 2 == 1
                colour = LIGHT if light else DARK
                if (row, column) in self.last:
                    colour = LAST_LIGHT if light else LAST_DARK
                if self.picked == (row, column):
                    colour = PICKED
                draw.rectangle([x0, y0, x0 + self.square - 1, y0 + self.square - 1], fill=colour)
        for square, (side, kind) in self.position.cells.items():
            x, y = self._centre(square)
            fill = (250, 250, 250) if side == FIRST else (60, 55, 50)
            _shape(draw, self.shape[kind], x, y, self.square * 0.34, fill, (20, 20, 20))
        for square in offers:
            x, y = self._centre(square)
            dot = self.square * 0.12
            draw.ellipse([x - dot, y - dot, x + dot, y + dot], fill=(90, 90, 90))
        return np.asarray(image)

    async def look(self) -> np.ndarray:
        await asyncio.sleep(0.005)
        return self.draw()

    async def click(self, x_share: float, y_share: float) -> bool:
        self.clicks += 1
        square = self._square_at(x_share * self.wide, y_share * self.tall)
        if square is None or self.position.turn != FIRST:
            self.picked = None
            return True
        if self.picked is None:
            if square in self.position.cells and self.position.cells[square][0] == FIRST:
                self.picked = square
            return True
        if square == self.picked:
            self.picked = None
            return True
        chosen = [m for m in moves(self.family, self.position) if m.start == self.picked and m.end == square]
        if chosen:
            self.make(chosen[0])
        self.picked = None
        return True

    def make(self, move: Move) -> None:
        self.made.append(move)
        self.last = {move.start, move.end}
        self.position = after(self.family, self.position, move)

    async def answers(self, stop: asyncio.Event) -> None:
        """The other side: it moves, by searching its own rules, whenever it is its turn."""
        while not stop.is_set():
            await asyncio.sleep(0.02)
            if self.position.turn != FIRST and not outcome(self.family, self.position):
                found = search(self.family, self.position, seconds=self.answers_in_s)
                if found.move is not None:
                    self.make(found.move)
