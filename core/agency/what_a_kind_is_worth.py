"""What a kind of piece, its moves and its safety are worth in a game, measured by her own rehearsal in her model of it.

Nobody tells her what a piece is worth. Before she has played, a kind is worth what it can do: the squares it reaches
on an empty board (core/agency/searching_by_kind.py). That is a guess, and a poor one: a piece that reaches few squares
may decide games. So she finds out the way anyone does, by playing: games in her own model of the game (the rules she
has learned), her search against itself, and then which positions went on to be won. Weights are fitted to them by a
logistic fit of the outcome on what a position is made of.

What is fitted grows a group at a time, because a weak player's games say little: first what each kind held is worth,
then whose turn it is and how pressed a royal piece stands, then each kind's moves, then where each kind stands. Each
group is kept only where the weights it gives beat the ones she plays by now, in a paired match in her model with
colours swapped, by more than the match's standard error; a group that does not is tried again when more games are in.
Measured on chess in its test world, 19 November 2026: fitting worth alone scored 0.56 against the first guess, and
fitting moves too on the same games scored 0.43 and 0.28.

What she measures is hers, about this one game, kept with what she learned of it. Nothing is measured for her and
shipped; nothing here knows a game.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.agency.pieces_by_their_kind import Family, Position
from core.agency.searching_by_kind import PER_SQUARE, Weights, described, first_weights, placed

logger = logging.getLogger("Aura.WhatAKindIsWorth")

__all__ = ["GROUPS", "Rehearsal", "a_first_rehearsal", "fitted", "rehearse"]

#: Games a round of rehearsal plays, and the nodes each move is searched to: small, so a round fits in the other
#: side's turn.
GAMES_A_ROUND = 40
NODES_A_MOVE = 1_500
#: Plies at the start of each rehearsal game chosen at random, so the games are not one game played over and over.
RANDOM_PLIES = 8
#: How far a fit is held back toward nothing: a weight is earned by the positions that show it.
HELD_BACK = 1.0
#: How many standard errors above even a new set of weights must score against the old to be kept.
BETTER_BY = 1.0
#: What is fitted, a group at a time, in the order a weak player's games can tell them apart.
GROUPS = ("worth", "turn and safety", "moves", "where each stands")


@dataclass
class Rehearsal:
    """Her rehearsal of one game: the weights she plays by, the positions gathered, which groups she has earned."""

    weights: Weights
    samples_x: list[list[float]] = field(default_factory=list)
    samples_y: list[float] = field(default_factory=list)
    earned: list[str] = field(default_factory=list)
    #: How many plies each rehearsed game ran: what tells her how long a game of this sort goes.
    lengths: list[int] = field(default_factory=list)
    rounds: int = 0
    last: dict[str, Any] = field(default_factory=dict)


def _engine() -> Any | None:
    try:
        import aura_board_search

        return aura_board_search
    except ImportError:
        return None


def _columns(family: Family, groups: list[str]) -> list[int]:
    kinds = len(family.kinds)
    squares = family.rows * family.columns
    columns: list[int] = []
    if "worth" in groups:
        columns += list(range(kinds))
    if "moves" in groups:
        columns += list(range(kinds, 2 * kinds))
    if "turn and safety" in groups:
        columns += [2 * kinds, 2 * kinds + 1]
    if "where each stands" in groups:
        columns += list(range(2 * kinds + 2, 2 * kinds + 2 + kinds * squares))
    return columns


def fitted(family: Family, xs: np.ndarray, ys: np.ndarray, groups: list[str]) -> Weights | None:
    """Weights fitted to positions and how their games ended, on the features of ``groups``: the logistic fit of the
    result on the features, held back toward nothing, scaled so the cheapest kind worth having is worth what the first
    guess's scale puts on reaching four squares."""
    columns = _columns(family, groups)
    if len(ys) < 50 or not columns:
        return None
    chosen = xs[:, columns]
    theta = np.zeros(len(columns))
    penalty = HELD_BACK * np.eye(len(columns))
    for _ in range(30):
        p = 1.0 / (1.0 + np.exp(-np.clip(chosen @ theta, -30, 30)))
        try:
            step = np.linalg.solve((chosen * (p * (1 - p))[:, None]).T @ chosen + penalty,
                                   chosen.T @ (p - ys) + penalty @ theta)
        except np.linalg.LinAlgError:
            return None
        theta -= step
        if float(np.abs(step).max()) < 1e-7:
            break
    full = np.zeros(xs.shape[1])
    full[columns] = theta
    letters = list(family.kinds)
    kinds = len(letters)
    positive = sorted(full[i] for i, letter in enumerate(letters) if full[i] > 0 and not family.kind(letter).royal)
    if not positive:
        return None
    scale = (PER_SQUARE * 4) / positive[0]
    squares = family.rows * family.columns
    table = full[2 * kinds + 2:]
    return Weights(
        worth={letter: round(float(full[i] * scale), 1) for i, letter in enumerate(letters)},
        mobility={letter: round(float(full[kinds + i] * scale), 2) for i, letter in enumerate(letters)},
        squares=({letter: [[round(float(table[i * squares + r * family.columns + c] * scale), 1)
                            for c in range(family.columns)] for r in range(family.rows)]
                  for i, letter in enumerate(letters)} if "where each stands" in groups else {}),
        # The feature counts squares pressed against the first side as less; the search takes the weight off per square.
        royal_pressed=round(float(full[2 * kinds] * scale), 2),
        tempo=round(float(full[2 * kinds + 1] * scale), 1),
    )


def _standing_where_it_stopped(played: dict, weights: Weights, family: Family) -> tuple[list[float], float]:
    """Each game's material where it stopped, first side less second, in what she holds each kind worth; and how widely
    those standings spread, the unit one is read in."""
    letters = list(family.kinds)
    standing = [sum(weights.worth.get(letter, 0.0) * float(count) for letter, count in zip(letters, counts))
                for counts in played.get("final_counts") or []]
    spread = float(np.std(standing)) if len(standing) > 1 else 0.0
    return standing, spread


def _as_a_result(standing: float, spread: float) -> float:
    """A standing read as a result between losing and winning: level is half, a spread ahead about three quarters."""
    return 1.0 / (1.0 + math.exp(-standing / spread))


def rehearse(family: Family, start: Position, rehearsal: Rehearsal, *, games: int = GAMES_A_ROUND,
             nodes: int = NODES_A_MOVE, seed: int = 1) -> Rehearsal:
    """One round: games of her search against itself in her model, their positions added to what she has, weights
    fitted on the groups earned and the next one, and kept only where they beat the weights she plays by now. Blocks
    for the round; the compiled games hold no lock of Python's, so it runs beside her in a thread."""
    engine = _engine()
    if engine is None:
        return rehearsal
    description = described(family)
    begin = placed(start)
    played = engine.self_play(description, begin, rehearsal.weights.as_json(), rehearsal.weights.as_json(),
                              games, nodes, random_plies=RANDOM_PLIES, seed=seed)
    rehearsal.lengths.extend(int(n) for n in played.get("plies", []))
    # A game that stopped without a result (no piece she knows must be kept safe, so none can be lost; or out of moves
    # to play) is judged by how it stood in material when it stopped. Without that, every game of a sort whose ending
    # she has not learned is level, and there is nothing to learn from: LIVE 20-21 November 2026 she played two games
    # against Stockfish and rehearsed in neither.
    standing, spread = _standing_where_it_stopped(played, rehearsal.weights, family)
    games_of = list(played.get("sample_games") or [])
    for index, (features, result) in enumerate(played["samples"]):
        label = float(result)
        if label == 0.5 and spread > 0 and index < len(games_of) and games_of[index] < len(standing):
            label = _as_a_result(standing[games_of[index]], spread)
        rehearsal.samples_x.append(list(features))
        rehearsal.samples_y.append(label)
    rehearsal.rounds += 1
    trying = next((group for group in GROUPS if group not in rehearsal.earned), None)
    groups = [*rehearsal.earned, *([trying] if trying else [])]
    candidate = fitted(family, np.array(rehearsal.samples_x, dtype=np.float64),
                       np.array(rehearsal.samples_y, dtype=np.float64), groups)
    if candidate is None:
        rehearsal.last = {"samples": len(rehearsal.samples_y), "tried": trying, "kept": False,
                          "why": "too few positions to fit"}
        return rehearsal
    match = engine.self_play(description, begin, candidate.as_json(), rehearsal.weights.as_json(), games, nodes,
                             random_plies=RANDOM_PLIES, seed=seed + 7919)
    results = np.array(match["results"], dtype=np.float64)
    standing, spread = _standing_where_it_stopped(match, rehearsal.weights, family)
    if spread > 0 and len(standing) == len(results):
        for game, points in enumerate(results):
            if points == 0.5:
                # The candidate played first in the even games.
                results[game] = _as_a_result(standing[game] if game % 2 == 0 else -standing[game], spread)
    mean = float(results.mean()) if len(results) else 0.5
    error = float(results.std(ddof=1) / math.sqrt(len(results))) if len(results) > 1 else 1.0
    kept = mean - 0.5 > BETTER_BY * max(error, 1e-9)
    rehearsal.last = {"samples": len(rehearsal.samples_y), "tried": trying, "scored": round(mean, 3),
                      "error": round(error, 3), "kept": kept}
    if kept:
        rehearsal.weights = candidate
        if trying and trying not in rehearsal.earned:
            rehearsal.earned.append(trying)
    logger.info("rehearsal round %d: %s", rehearsal.rounds, rehearsal.last)
    return rehearsal


def a_first_rehearsal(family: Family) -> Rehearsal:
    return Rehearsal(weights=first_weights(family))
