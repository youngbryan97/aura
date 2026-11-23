"""Thinking before it comes: where a situation comes back, the thinking it needs is done ahead of it.

Some situations come back. Every game at a board begins at the same place, so its first decisions are met again in
every game, and so are the answers the other side tends to give to them. A person prepares for what comes back: an
opening, a standard situation in a sport, a hard conversation rehearsed before it is had. The thinking is the same
thinking. What preparing changes is when it is done, and so how long it may take: between games there is time no
game gives, and an answer worked out ahead costs nothing on the clock when the situation comes.

So when she is idle she walks the situations she is likeliest to meet next. From the start: her own move, as a long
look finds it; then each answer the other side has given there (kept as they gave them, core/agency/
thinking_on_their_time.py), or, where she has seen none, the answer her own look expects of them; then her move after
each of those, and on. Each answer she works out is kept with the rules and judgement it was worked out under. In a
game, a situation she prepared under the same rules and judgement is answered from what she worked out; any other is
thought about as before.

LIVE 23 November 2026, on the opening positions of games 26 and 27: her moves as played cost 1,345 hundredths of a
piece by Stockfish held to depth 14, her search at her game pace (2.5 s) 346, and a 30 s look 202.

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

__all__ = ["LOOKS_FOR_S", "Prepared", "under"]

#: How long she looks at each situation she prepares: the look measured above to choose better than her game pace.
LOOKS_FOR_S = 30.0
#: Situations kept at most: a game's opening meets a few dozen, and the other side's answers branch them.
KEPT_AT_MOST = 2_000
#: How many of the other side's answers in one situation she prepares for: the ones they have given most.
ANSWERS_FOLLOWED = 2


def _key(place: str) -> str:
    return hashlib.sha1(str(place).encode()).hexdigest()[:16]


def under(family: Any, weights: Any) -> str:
    """The rules and judgement an answer is worked out under, as a short name: either changed is another answer."""
    from core.agency.searching_by_kind import described

    return hashlib.sha1((described(family) + "\0" + weights.as_json()).encode()).hexdigest()[:16]


@dataclass
class Prepared:
    """Her answers, worked out ahead, to the situations she expects to meet: each with the move, how deep the look
    went, what it judged, and the rules and judgement it was worked out under."""

    answers: dict[str, dict[str, Any]] = field(default_factory=dict)

    def answer(self, place: str, worked_out_under: str) -> dict[str, Any] | None:
        """What she worked out for this situation under these rules and this judgement, if she has."""
        held = self.answers.get(_key(place))
        return held if held is not None and held.get("under") == worked_out_under else None

    def prepare(self, family: Any, start: Any, weights: Any, *, they_answered: Any = None, positions: int,
                plies: int, seconds: float = LOOKS_FOR_S, threads: int = 1, search: Any = None) -> int:
        """Works out up to ``positions`` situations she has not prepared under these rules and this judgement, walking
        from ``start`` through her own answers and the other side's, ``plies`` turns deep at most. How many it worked
        out."""
        from core.agency.pieces_by_their_kind import FIRST, after, moves
        from core.agency.playing_pieces_by_kind import _a_place, _named

        if search is None:
            from core.agency.searching_by_kind import search
        worked_out_under = under(family, weights)
        done = 0
        # Breadth first: the situations nearest the start are the ones met in every game.
        walk: list[tuple[Any, list[Any], int, Any]] = [(start, [], 0, None)]
        seen: set[str] = set()
        while walk and done < positions:
            position, earlier, ply, expected = walk.pop(0)
            place = _a_place(position)
            if ply >= plies or place in seen:
                continue
            seen.add(place)
            legal = moves(family, position)
            if not legal:
                continue
            if position.turn == FIRST:
                held = self.answer(place, worked_out_under)
                move = next((m for m in legal if held is not None and _named(m) == held.get("move")), None)
                reply = None
                if move is None:
                    began = time.monotonic()
                    found = search(family, position, seconds=seconds, weights=weights, earlier=earlier,
                                   threads=threads)
                    if found.move is None:
                        continue
                    move = found.move
                    line = list(found.line or [])
                    reply = _named(line[1]) if len(line) > 1 and line[0] == move else None
                    self.answers[_key(place)] = {"move": _named(move), "depth": int(found.depth),
                                                 "score": int(found.score), "under": worked_out_under,
                                                 "reply": reply, "at": time.time()}
                    done += 1
                    logger.info("worked out ahead a situation %d turn(s) from the start in %.0f s: %s (looked %d deep)",
                                ply, time.monotonic() - began, _named(move), found.depth)
                else:
                    reply = held.get("reply")
                walk.append((after(family, position, move), [*earlier, position], ply + 1, reply))
                continue
            # Their turn: the answers they have given here, most often first; or the one her look expects of them.
            given = dict(getattr(they_answered, "answered", {}).get(_key(place), {})) if they_answered else {}
            named = sorted(given, key=lambda m: -given[m])[:ANSWERS_FOLLOWED] or ([expected] if expected else [])
            for name in named:
                move = next((m for m in legal if _named(m) == name), None)
                if move is not None:
                    walk.append((after(family, position, move), [*earlier, position], ply + 1, None))
        if len(self.answers) > KEPT_AT_MOST:
            for key in sorted(self.answers, key=lambda k: self.answers[k].get("at", 0.0))[:len(self.answers)
                                                                                         - KEPT_AT_MOST]:
                del self.answers[key]
        return done

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
