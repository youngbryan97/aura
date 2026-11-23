"""Thinking before it comes: where a situation comes back, the thinking it needs is done ahead of it.

Some situations come back. Every game at a board begins at the same place, so its first decisions are met again in
every game, and so are the answers the other side tends to give to them. A person prepares for what comes back: an
opening, a standard situation in a sport, a hard conversation rehearsed before it is had. The thinking is the same
thinking. What preparing changes is when it is done, and so how long it may take: between games there is time no
game gives, and an answer worked out ahead costs nothing on the clock when the situation comes.

So when she is idle she walks the situations she is likeliest to meet next. From the start: her own move, as a long
look finds it; then each answer the other side has given there (kept as they gave them, core/agency/
thinking_on_their_time.py), or, where she has seen none, the answer her own look expects of them; then her move after
each of those, and on. Each answer she works out is kept with the choices it was worked out among. In a game, a
situation she prepared that offers her the same choices is answered from what she worked out; any other is thought
about as before.

LIVE 23 November 2026, on the opening positions of games 26 and 27: her moves as played cost 1,345 hundredths of a
piece by Stockfish held to depth 14, her search at her game pace (2.5 s) 346, and a 30 s look 202.

What she prepares grows as a repertoire does. Each look goes to the situation not yet worked out that the most of her
games reach, by the answers the other side has given on the way, an answer she only expects counted as one they
give. Weighing an expected answer by how often her expectations of them had come true met fewer of the situations she
prepared, and lost more: over 40 games on each of two seeds against Stockfish limited to 2500, 2.1 and 2.3 moves a
game came from what she worked out and she scored 0.300 and 0.287, against 2.6 and 2.5 and 0.412 and 0.438 this way,
and 0.225 and 0.250 without preparing. A move of hers that a game she lost went through is set aside where she
prepares that situation again, as it is in her games (WhatFailedWhere), so the line she lost is replaced by another.
A situation worked out is not looked at again for longer: on 12 opening positions of her games of 22 and 23 November
2026, Stockfish held to depth 18 put her moves at game pace at 211 hundredths of a piece, a 30 s look at 61, and
looks of 60 and 120 s at 215 and 255. Games 26, 27 and 28 were decided by their fifth to tenth moves.

Nothing here knows a game: a situation is a position written out, an answer a move named.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("Aura.ThinkingBeforeItComes")

__all__ = ["LOOKS_FOR_S", "Prepared", "among"]

#: How long she looks at each situation she prepares: the look measured above to choose better than her game pace.
LOOKS_FOR_S = 30.0
#: Situations kept at most: a game's opening meets a few dozen, and the other side's answers branch them.
KEPT_AT_MOST = 2_000
#: How many of the other side's answers in one situation she prepares for: the ones they have given most.
ANSWERS_FOLLOWED = 2


def _key(place: str) -> str:
    return hashlib.sha1(str(place).encode()).hexdigest()[:16]


def among(legal: Any) -> str:
    """The choices a situation offers, as a short name: an answer worked out among these choices is an answer here.

    An answer is kept while the choices it was made among are the choices she has: a move her rules have come to
    allow there, or no longer allow, is a choice the look that made it never weighed. A change to her rules that
    leaves the choices here as they were changes nothing she chose between. Nor does a change to her judgement: a
    30 s look under the judgement she had is still far better than her game pace under the one she has (on 12 opening
    positions, 61 hundredths of a piece against 211), and a judgement she keeps is changed a little at a time."""
    from core.agency.playing_pieces_by_kind import _named

    return hashlib.sha1("|".join(sorted(_named(m) for m in legal)).encode()).hexdigest()[:16]


@dataclass
class Prepared:
    """Her answers, worked out ahead, to the situations she expects to meet: each with the move, how deep the look
    went, what it judged, and the choices it was worked out among."""

    answers: dict[str, dict[str, Any]] = field(default_factory=dict)

    def answer(self, place: str, choices: str) -> dict[str, Any] | None:
        """What she worked out for this situation among these choices (``among``), if she has."""
        held = self.answers.get(_key(place))
        return held if held is not None and held.get("among") == choices else None

    def prepare(self, family: Any, start: Any, weights: Any, *, they_answered: Any = None, went_wrong: Any = None,
                positions: int, plies: int, seconds: float = LOOKS_FOR_S, threads: int = 1, search: Any = None) -> int:
        """Spends up to ``positions`` looks of ``seconds`` on the situations she expects to meet and has not worked
        out, walking from ``start`` through her own answers and the other side's, ``plies`` turns deep at most: each
        look goes to the one the largest share of her games reach, by the answers they have given on the way. A move
        that lost there (``went_wrong``) is set aside. How many it worked out."""
        from core.agency.pieces_by_their_kind import FIRST, after, moves
        from core.agency.playing_pieces_by_kind import _a_place, _named

        if search is None:
            from core.agency.searching_by_kind import search
        done = 0
        while done < positions:
            mine = [item for item in self._walked(family, start, they_answered, went_wrong, plies,
                                                  FIRST, after, moves, _a_place, _named) if not item[4]]
            if not mine:
                break
            # The likeliest first; among equals, the nearest the start.
            position, earlier, ply, place, _looked, _share = max(mine, key=lambda item: (item[5], -item[2]))
            lost_here = went_wrong.already_failed_at(place) if went_wrong is not None else frozenset()
            legal = moves(family, position)
            aside = [m for m in legal if _named(m) in lost_here]
            if self._looked(search, family, position, earlier, weights, seconds, threads,
                            aside if len(aside) < len(legal) else [], place, ply, among(legal)) is None:
                break
            done += 1
        if len(self.answers) > KEPT_AT_MOST:
            for key in sorted(self.answers, key=lambda k: self.answers[k].get("at", 0.0))[:len(self.answers)
                                                                                         - KEPT_AT_MOST]:
                del self.answers[key]
        return done

    def _walked(self, family: Any, start: Any, they_answered: Any, went_wrong: Any,
                plies: int, first: int, after: Any, moves: Any, _a_place: Any, _named: Any
                ) -> list[tuple[Any, list[Any], int, str, float, float]]:
        """Her situations on the way from ``start``: through her answers where she has them, the answers the other
        side has given (or the one she expects), each with how far from the start it is, how long she has looked at it
        among the choices it offers now (nought where she has not, or where her answer is a move that lost there),
        and the share of games that reach it, each answer of theirs weighed by how often they have given it, and an
        answer she only expects as one they give. The walk stops at a situation of hers she has not worked out: what
        follows it depends on her answer."""
        found: list[tuple[Any, list[Any], int, str, float, float]] = []
        walk: list[tuple[Any, list[Any], int, Any, float]] = [(start, [], 0, None, 1.0)]
        seen: set[str] = set()
        while walk and len(seen) < KEPT_AT_MOST:
            position, earlier, ply, expected, share = walk.pop(0)
            place = _a_place(position)
            if ply >= plies or place in seen:
                continue
            seen.add(place)
            legal = moves(family, position)
            if not legal:
                continue
            if position.turn == first:
                held = self.answer(place, among(legal))
                lost_here = went_wrong.already_failed_at(place) if went_wrong is not None else frozenset()
                move = next((m for m in legal if held is not None and _named(m) == held.get("move")
                             and _named(m) not in lost_here), None)
                if move is None:
                    # Not worked out, or worked out as the move a game she lost went through: worked out (again) first.
                    found.append((position, earlier, ply, place, 0.0, share))
                    continue
                found.append((position, earlier, ply, place, float(held.get("seconds", LOOKS_FOR_S)), share))
                walk.append((after(family, position, move), [*earlier, position], ply + 1, held.get("reply"), share))
                continue
            # Their turn: the answers they have given here, most often first; or the one her look expects of them.
            given = dict(getattr(they_answered, "answered", {}).get(_key(place), {})) if they_answered else {}
            total = sum(given.values())
            named = sorted(given, key=lambda m: -given[m])[:ANSWERS_FOLLOWED] or ([expected] if expected else [])
            for name in named:
                move = next((m for m in legal if _named(m) == name), None)
                if move is not None:
                    part = given[name] / total if given else 1.0
                    walk.append((after(family, position, move), [*earlier, position], ply + 1, None, share * part))
        return found

    def _looked(self, search: Any, family: Any, position: Any, earlier: list[Any], weights: Any, seconds: float,
                threads: int, aside: list[Any], place: str, ply: int, choices: str) -> tuple[Any, Any] | None:
        """One look at a situation of hers, kept as her answer to it: the move, and the reply her look expects."""
        from core.agency.playing_pieces_by_kind import _named

        began = time.monotonic()
        found = search(family, position, seconds=seconds, weights=weights, earlier=earlier, threads=threads,
                       **({"set_aside": aside} if aside else {}))
        if found.move is None:
            return None
        move = found.move
        line = list(found.line or [])
        reply = _named(line[1]) if len(line) > 1 and line[0] == move else None
        self.answers[_key(place)] = {"move": _named(move), "depth": int(found.depth), "score": int(found.score),
                                     "among": choices, "reply": reply, "at": time.time(),
                                     "seconds": float(seconds), "ply": int(ply)}
        logger.info("worked out ahead a situation %d turn(s) from the start in %.0f s: %s (looked %d deep)",
                    ply, time.monotonic() - began, _named(move), found.depth)
        return move, reply

    def as_bytes(self) -> bytes:
        return json.dumps(self.answers, separators=(",", ":")).encode()

    @classmethod
    def from_bytes(cls, data: bytes | None) -> Prepared:
        try:
            held = json.loads(data or b"{}")
        # not a failure: nothing kept, or nothing readable, is nothing prepared.
        except (TypeError, ValueError):
            return cls()
        return cls(answers={str(k): dict(v) for k, v in held.items() if isinstance(v, dict)}
                   if isinstance(held, dict) else {})
