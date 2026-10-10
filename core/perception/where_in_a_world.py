"""Where she is in a world bigger than the view: how far the view going by has taken her, and how far along she is.

A person playing a game that scrolls knows where they are in it without a map:
they have gone two screens to the right since the start, they are further than
they got last time, the way on is the way the world keeps coming from, and the
place behind them they have seen. Nothing on the screen says so; it is the sum
of how the view went by.

So is hers. Each time the view goes by (core/perception/how_the_scenery_goes_by.py)
the layer it went by in, where she is, is the world going past her: the layer
of her own band where she knows which thing is hers, else the layer of the
lowest band that moves, which is the ground in most worlds that scroll. Its
going, a picture at a time, is added up: the view has gone the other way, and
so has she, in the world. How far she has come is counted in views (a view is
the width or height of the picture), and the furthest she has been along the
way the world goes is her progress through it. The places she has been are
kept a view at a time.

Nothing here knows what the world is: a level, a map, a long page.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["WhereInTheWorld"]


@dataclass
class WhereInTheWorld:
    """How far the view has carried her, in working pixels across and down, and the views she has been in."""

    #: Where the view is in the world, in working pixels from where it started (right and down are positive).
    across: float = 0.0
    down: float = 0.0
    #: The size of the view, in working pixels.
    wide: int = 0
    tall: int = 0
    #: The furthest the view has been from where it started, each way.
    furthest: dict[str, float] = field(default_factory=lambda: {"right": 0.0, "left": 0.0, "down": 0.0, "up": 0.0})
    #: The views she has been in, as whole views from the start.
    been: set[tuple[int, int]] = field(default_factory=lambda: {(0, 0)})
    #: Views gone into that she had not been in before, since she began.
    new_views: int = 0

    def saw(self, moved: Any, over: int, tall: int, wide: int, hers: Any = None) -> bool:
        """Add how the view went by over ``over`` pictures (``moved``, a ViewMoved) to where she is. Whether she came
        into a view she had not been in."""
        if moved is None or getattr(moved, "still", True):
            return False
        self.tall, self.wide = tall, wide
        dx, dy = self._the_world_going(moved, tall, wide, hers)
        # The scenery going left is the view going right, and her with it.
        self.across -= dx / max(1, over)
        self.down -= dy / max(1, over)
        self.furthest["right"] = max(self.furthest["right"], self.across)
        self.furthest["left"] = max(self.furthest["left"], -self.across)
        self.furthest["down"] = max(self.furthest["down"], self.down)
        self.furthest["up"] = max(self.furthest["up"], -self.down)
        # Views a whole view apart, centred on where she began: a few pixels' going either way of the start is the view she
        # began in (LIVE 2026-10-10 a still game's jitter of a pixel to the left was counted a new view).
        view = (int((self.across + max(1, wide) / 2) // max(1, wide)), int((self.down + max(1, tall) / 2) // max(1, tall)))
        if view in self.been:
            return False
        self.been.add(view)
        self.new_views += 1
        return True

    @staticmethod
    def _the_world_going(moved: Any, tall: int, wide: int, hers: Any) -> tuple[float, float]:
        """The shift of the layer that is the world where she is: her own band's, else the lowest band that moves."""
        if hers is not None:
            ways = moved.ways_going(hers.y, hers.x, tall, wide)
            if ways:
                across = sorted(dx for dx, _dy in ways)
                down = sorted(dy for _dx, dy in ways)
                return float(across[len(across) // 2]), float(down[len(down) // 2])
        for layers in reversed(moved.across or ()):
            moving = [dx for dx in layers if dx]
            if moving:
                return float(sorted(moving)[len(moving) // 2]), 0.0
        for layers in reversed(moved.down or ()):
            moving = [dy for dy in layers if dy]
            if moving:
                return 0.0, float(sorted(moving)[len(moving) // 2])
        return 0.0, 0.0

    @property
    def the_way_on(self) -> str:
        """The way the world has taken her furthest: where what she has not seen is."""
        way = max(self.furthest, key=self.furthest.__getitem__)
        return way if self.furthest[way] > 0 else ""

    def views_along(self) -> float:
        """How far she has come along the way on, in views."""
        way = self.the_way_on
        if not way:
            return 0.0
        size = self.wide if way in ("right", "left") else self.tall
        return self.furthest[way] / max(1, size)

    def said(self) -> str:
        """Where she is, in words, or "" before she has gone anywhere."""
        along = self.views_along()
        if along < 1.0:
            return ""
        return f"I've come about {round(along)} screen{'s' if round(along) != 1 else ''} {self.the_way_on}."
