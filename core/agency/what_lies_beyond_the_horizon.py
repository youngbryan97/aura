"""What lies beyond the horizon of her search: what a position comes to long after any line she looks at ends.

A search sees as far as it looks and judges where it stops. What changes little in a move and decides games long after
is outside every line it looks at, however deep: a piece that only goes forward and becomes another kind, with nothing
of the other side ahead of it to stop it; such a piece with no quiet move forward; pieces held by others of their side;
ground in the other side's half that one side reaches and the other does not; and what stands beside the piece that
must be kept safe. Her compiled search counts these for each side from the rules she has learned
(rust_extensions/aura_board_search/src/lib.rs, BEYOND); nothing here or there knows a game.

What each is worth cannot be taught by a longer look of hers, which has a horizon of its own. It is taught by how her
games ended: of each position she rehearsed, what her search scored it and what the game came to, and the worths fitted
are what the results show that her search's score did not. Her search's score is held fixed in the fit and only what
lies beyond is fitted, so the fit cannot take back what her other groups earned. It is tried only once it foretells
games held back from it better than her search's score alone, and kept, as every group is, only where it wins its
match (core/agency/what_a_kind_is_worth.py).
"""
from __future__ import annotations

import io
import math
from dataclasses import dataclass, field

import numpy as np

from core.agency.pieces_by_their_kind import Family

__all__ = ["BEYOND", "WhatGamesCameTo", "beyond_at"]

#: The quantities her compiled search counts, in its order.
BEYOND = ("how far an unopposed runner has come", "runners with no quiet move forward", "pieces held by their own",
          "ground held in the other half", "pieces beside the royal one")
#: Positions kept at most, the latest: about three megabytes of her disk.
KEPT_AT_MOST = 100_000
#: How far a fit is held back toward nothing, as rehearsal's are.
HELD_BACK = 1.0
#: One game in this many is held back from the fit, to judge what it foretells.
HELD_OUT_EVERY = 5


def beyond_at(family: Family) -> int:
    """Where what lies beyond the horizon begins in a position's features from her compiled search."""
    kinds = len(family.kinds)
    return 2 * kinds + 2 + kinds * family.rows * family.columns + kinds


def _logistic(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def _loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


@dataclass
class WhatGamesCameTo:
    """Each position she rehearsed: what lies beyond the horizon there (the first side's less the second's), what her
    search scored it for the first side, what the game came to for the first side, and which game it was."""

    beyond: list[np.ndarray] = field(default_factory=list)
    score: list[float] = field(default_factory=list)
    result: list[float] = field(default_factory=list)
    game: list[int] = field(default_factory=list)

    def add(self, played: dict, came_to: list[float], round_: int, at: int) -> None:
        """The positions of one round of her games, where her compiled search counted what lies beyond them."""
        samples = played.get("samples") or []
        scores = played.get("sample_scores") or []
        games = played.get("sample_games") or []
        if len(scores) != len(samples) or len(came_to) != len(samples):
            return
        for n, ((features, _result), score) in enumerate(zip(samples, scores, strict=True)):
            if len(features) < at + len(BEYOND) or abs(float(score)) >= 3000:
                continue
            self.beyond.append(np.asarray(features[at:at + len(BEYOND)], dtype=np.float32))
            self.score.append(float(score))
            self.result.append(float(came_to[n]))
            self.game.append(round_ * 10_000 + (int(games[n]) if n < len(games) else 0))
        extra = len(self.result) - KEPT_AT_MOST
        if extra > 0:
            del self.beyond[:extra], self.score[:extra], self.result[:extra], self.game[:extra]

    def worth(self, *, seed: int = 0) -> tuple[list[float] | None, str]:
        """What each quantity is worth, in the units her search scores in, fitted to how her games ended with her
        search's score held fixed; and why not, where the fit foretells games held back from it no better than her
        search's score alone."""
        from core.agency.what_a_kind_is_worth import calibrated_scale

        if len(self.result) < 50 * len(BEYOND):
            return None, "too few positions to fit"
        x = np.array(self.beyond, dtype=np.float64)
        s = np.array(self.score, dtype=np.float64)
        y = np.array(self.result, dtype=np.float64)
        games = np.array(self.game)
        scale = calibrated_scale(list(s), list(y))
        offset = s / scale
        # Games, not positions, are held back: a game's positions are alike, and one held back with its neighbours
        # in the fit would be foretold by them.
        held = np.isin(games, np.unique(games)[np.random.default_rng(seed).permutation(len(np.unique(games)))
                                               [:max(1, len(np.unique(games)) // HELD_OUT_EVERY)]])
        if held.all() or not held.any():
            return None, "too few games to hold any back"
        theta = self._fitted(x[~held], offset[~held], y[~held])
        if theta is None:
            return None, "positions too alike to fit"
        before = _loss(_logistic(offset[held]), y[held])
        after = _loss(_logistic(offset[held] + x[held] @ theta), y[held])
        if not after < before:
            return None, f"it foretells games held back no better than her search does ({after:.4f} vs {before:.4f})"
        theta = self._fitted(x, offset, y)
        if theta is None:
            return None, "positions too alike to fit"
        return [round(float(t * scale), 2) for t in theta], ""

    @staticmethod
    def _fitted(x: np.ndarray, offset: np.ndarray, y: np.ndarray) -> np.ndarray | None:
        theta = np.zeros(x.shape[1])
        penalty = HELD_BACK * np.eye(x.shape[1])
        for _ in range(30):
            p = _logistic(offset + x @ theta)
            try:
                step = np.linalg.solve((x * (p * (1 - p))[:, None]).T @ x + penalty, x.T @ (p - y) + penalty @ theta)
            # not a failure: positions too alike to fit say nothing yet.
            except np.linalg.LinAlgError:
                return None
            theta -= step
            if float(np.abs(step).max()) < 1e-7:
                break
        return theta if all(math.isfinite(float(t)) for t in theta) else None

    def as_bytes(self) -> bytes:
        """Kept on disk as one compressed file of arrays."""
        out = io.BytesIO()
        np.savez_compressed(out, beyond=np.array(self.beyond, dtype=np.float32).reshape(-1, len(BEYOND)),
                            score=np.array(self.score, dtype=np.float32), result=np.array(self.result, dtype=np.float32),
                            game=np.array(self.game, dtype=np.int64))
        return out.getvalue()

    @classmethod
    def from_bytes(cls, data: bytes) -> WhatGamesCameTo:
        held = np.load(io.BytesIO(data), allow_pickle=False)
        if held["beyond"].ndim != 2 or held["beyond"].shape[1] != len(BEYOND):
            return cls()
        return cls(beyond=[row.astype(np.float32) for row in held["beyond"]], score=[float(v) for v in held["score"]],
                   result=[float(v) for v in held["result"]], game=[int(v) for v in held["game"]])
