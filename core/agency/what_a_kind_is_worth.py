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

Results of weak games teach what a kind is worth and little more. A group they fail to earn twice, once worth is
earned, is taught by her longer look instead: each position she rehearses is also searched twenty times as long, and
what that search finds it worth, read as a chance to win on the scale her own results show, is the label. Her slow
thinking teaches her quick judgement what it would have found. It cannot teach worth itself, which it reads from the
weights it searches with.

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

__all__ = ["GROUPS", "HER_LONGER_LOOK", "RESULTS", "Rehearsal", "a_first_rehearsal", "calibrated_scale", "fitted",
           "rehearse"]

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
GROUPS = ("worth", "turn and safety", "moves", "what is left to be taken", "where each stands")
#: Who teaches her a position: how her games ended, or what a much longer look of hers finds it worth.
RESULTS = "how her games ended"
HER_LONGER_LOOK = "her longer look"
#: Nodes of the longer look that teaches a position once results cannot: twenty times a rehearsal move's.
LONGER_LOOK_NODES = 30_000
#: Tries of a group taught by results, failed, before her longer look teaches instead.
RESULTS_TRIED_AT_MOST = 2


@dataclass
class Rehearsal:
    """Her rehearsal of one game: the weights she plays by, the positions gathered, which groups she has earned."""

    weights: Weights
    samples_x: list[np.ndarray] = field(default_factory=list)
    samples_y: list[float] = field(default_factory=list)
    earned: list[str] = field(default_factory=list)
    #: How many plies each rehearsed game ran: what tells her how long a game of this sort goes.
    lengths: list[int] = field(default_factory=list)
    rounds: int = 0
    last: dict[str, Any] = field(default_factory=dict)
    #: The kinds held to be the one the game keeps safe when these weights were fitted: what a kind is worth is worth
    #: in that game, and none where it is not known. LIVE 21 November 2026 weights fitted while she held the pawns'
    #: kind as the one kept safe valued her rooks at -16.7 and her pawns at -68.4, and were kept for every game after.
    royal: list[str] | None = None
    #: How many times each group has been tried: each try is one more chance for a group that does not help to win a
    #: match by luck, so each asks more of it.
    tries: dict[str, int] = field(default_factory=dict)
    #: Who teaches her positions now (RESULTS or HER_LONGER_LOOK). Results of games between two weak players say what
    #: a kind holds is worth and little more: measured 22 November 2026 on 141,547 positions of hers, where each piece
    #: stands added nothing to how well a fit foretold results it had not seen (log-loss 0.5294 without, 0.5295 with),
    #: and a candidate fitted to them lost its match 0.107 +- 0.048. Taught by what a search twenty times as long found
    #: each position worth, the same group was earned on its first try (0.657 +- 0.071). A longer look cannot teach
    #: what a kind holds is worth, which it reads from the weights it searches with: begun from her first guess, eight
    #: rounds of it earned nothing.
    teacher: str = RESULTS


def _engine() -> Any | None:
    try:
        import aura_board_search

        return aura_board_search
    # not a failure: without the compiled search there is nothing to rehearse in, and a rehearsal is returned as it
    # was given.
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
    if "what is left to be taken" in groups:
        # What she cannot afford to lose (core/cognition/what_she_cannot_afford_to_lose.py): per kind, the pieces
        # each side leaves attacked and undefended.
        start = 2 * kinds + 2 + kinds * squares
        columns += list(range(start, start + kinds))
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
        # not a failure: positions too alike to fit tell her nothing yet, and no candidate is tried this round.
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
    table = full[2 * kinds + 2:2 * kinds + 2 + kinds * squares]
    exposed_at = 2 * kinds + 2 + kinds * squares
    return Weights(
        worth={letter: round(float(full[i] * scale), 1) for i, letter in enumerate(letters)},
        mobility={letter: round(float(full[kinds + i] * scale), 2) for i, letter in enumerate(letters)},
        squares=({letter: [[round(float(table[i * squares + r * family.columns + c] * scale), 1)
                            for c in range(family.columns)] for r in range(family.rows)]
                  for i, letter in enumerate(letters)} if "where each stands" in groups else {}),
        # The feature counts squares pressed against the first side as less; the search takes the weight off per square.
        royal_pressed=round(float(full[2 * kinds] * scale), 2),
        tempo=round(float(full[2 * kinds + 1] * scale), 1),
        # The feature counts the other side's exposed pieces less her own; the search takes the weight off per piece.
        exposed=({letter: round(float(full[exposed_at + i] * scale), 2) for i, letter in enumerate(letters)}
                 if "what is left to be taken" in groups and full.shape[0] >= exposed_at + kinds else {}),
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


def calibrated_scale(scores: list[float], results: list[float]) -> float:
    """The score that reads as about three chances in four to win: the scale on which her search's scores best foretell
    how her games ended. Her own scores, read against her own results; nothing is assumed of what a score means."""
    v = np.clip(np.asarray(scores, dtype=np.float64), -3000, 3000)
    z = np.asarray(results, dtype=np.float64)
    best, best_fit = 200.0, -math.inf
    for scale in np.geomspace(30, 3000, 60):
        p = np.clip(1.0 / (1.0 + np.exp(-v / scale)), 1e-6, 1 - 1e-6)
        fit = float(np.sum(z * np.log(p) + (1 - z) * np.log(1 - p)))
        if fit > best_fit:
            best, best_fit = float(scale), fit
    return best


def _what_her_longer_look_found(played: dict) -> list[float] | None:
    """Each sampled position read as a chance to win from what her longer look found it worth, or None where the
    games carry no such scores (a build of the compiled search from before them)."""
    scores = list(played.get("sample_scores") or [])
    results = [float(result) for _features, result in played.get("samples") or []]
    if not scores or len(scores) != len(results):
        return None
    scale = calibrated_scale(scores, results)
    return [1.0 / (1.0 + math.exp(-max(-3000.0, min(3000.0, float(v))) / scale)) for v in scores]


def _played(engine: Any, *asked: Any, longer: int = 0, **options: Any) -> tuple[dict, bool]:
    """Games of her search in her model, and whether each sampled position was also looked at ``longer`` nodes."""
    if longer:
        try:
            return engine.self_play(*asked, deep_nodes=longer, **options), True
        # not a failure: a build of the compiled search from before longer looks plays the games without them, and
        # the positions are taught by results.
        except TypeError:
            pass
    return engine.self_play(*asked, **options), False


def how_far_above_even(tries: int) -> float:
    """How many standard errors above even a group must score on its ``tries``-th try to be kept. The first try asks
    BETTER_BY; later ones ask more, so that the chance a group that does not help is ever kept stays what one try's
    was, shared out over the tries (Bonferroni). LIVE-like 21 November 2026: a lead-over-what-is-left feature was tried
    seven times at one standard error, kept on the seventh at 0.584 +- 0.074, and then lost to the weights without it,
    0.446 +- 0.040 over 120 games."""
    from statistics import NormalDist

    chance = 1.0 - NormalDist().cdf(BETTER_BY)
    return NormalDist().inv_cdf(1.0 - chance / max(1, tries))


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
    longer = LONGER_LOOK_NODES if rehearsal.teacher == HER_LONGER_LOOK else 0
    played, looked_longer = _played(engine, description, begin, rehearsal.weights.as_json(),
                                    rehearsal.weights.as_json(), games, nodes, longer=longer,
                                    random_plies=RANDOM_PLIES, seed=seed)
    taught = _what_her_longer_look_found(played) if looked_longer else None
    # How long a game of this sort goes, from games that ended with a result: one that stopped on a repetition early, in
    # a model that cannot yet end a game, says nothing of it. LIVE 21 November 2026 such games ran a dozen plies, she
    # expected two or three more moves of her own, and gave each move a third of her clock.
    rehearsal.lengths.extend(int(n) for n, points in zip(played.get("plies", []), played.get("results", []))
                             if float(points) != 0.5)
    # A game that stopped without a result (no piece she knows must be kept safe, so none can be lost; or out of moves
    # to play) is judged by how it stood in material when it stopped. Without that, every game of a sort whose ending
    # she has not learned is level, and there is nothing to learn from: LIVE 20-21 November 2026 she played two games
    # against Stockfish and rehearsed in neither.
    standing, spread = _standing_where_it_stopped(played, rehearsal.weights, family)
    games_of = list(played.get("sample_games") or [])
    for index, (features, result) in enumerate(played["samples"]):
        label = float(result)
        if taught is not None:
            label = taught[index]
        elif label == 0.5 and spread > 0 and index < len(games_of) and games_of[index] < len(standing):
            label = _as_a_result(standing[games_of[index]], spread)
        # Kept as one compact row of numbers, not a list of Python floats: a position's features as floats took 13 KB,
        # and LIVE 21 November 2026 one game's rehearsal gathered 15,820 of them.
        rehearsal.samples_x.append(np.asarray(features, dtype=np.float32))
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
    tries = rehearsal.tries[trying] = rehearsal.tries.get(trying, 0) + 1 if trying else 1
    kept = mean - 0.5 > how_far_above_even(tries) * max(error, 1e-9)
    rehearsal.last = {"samples": len(rehearsal.samples_y), "tried": trying, "scored": round(mean, 3),
                      "error": round(error, 3), "kept": kept}
    if kept:
        rehearsal.weights = candidate
        rehearsal.royal = royal_kinds(family)
        if trying and trying not in rehearsal.earned:
            rehearsal.earned.append(trying)
    elif (rehearsal.teacher == RESULTS and trying and "worth" in rehearsal.earned
          and tries >= RESULTS_TRIED_AT_MOST):
        # What results could teach they have taught: from here her longer look teaches, on positions of its own.
        rehearsal.teacher = HER_LONGER_LOOK
        rehearsal.samples_x, rehearsal.samples_y = [], []
        rehearsal.last["teacher"] = HER_LONGER_LOOK
    logger.info("rehearsal round %d: %s", rehearsal.rounds, rehearsal.last)
    return rehearsal


def a_first_rehearsal(family: Family) -> Rehearsal:
    return Rehearsal(weights=first_weights(family), royal=royal_kinds(family))


def royal_kinds(family: Family) -> list[str]:
    """The kinds the game keeps safe, as she holds the game now."""
    return sorted(letter for letter, kind in family.kinds.items() if kind.royal)
