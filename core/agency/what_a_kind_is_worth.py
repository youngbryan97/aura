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
Each side of the match looks at as many positions a move as its own pace affords in the same time, because a game is
played on a clock: a judgement that slows her search is judged with what it costs.
Measured on chess in its test world, 19 November 2026: fitting worth alone scored 0.56 against the first guess, and
fitting moves too on the same games scored 0.43 and 0.28.

Results of weak games teach what a kind is worth and little more. A group they fail to earn twice, once worth is
earned, is taught by her longer look instead: each position she rehearses is also searched twenty times as long, and
what that search finds it worth, read as a chance to win on the scale her own results show, is the label. Her slow
thinking teaches her quick judgement what it would have found. It cannot teach worth itself, which it reads from the
weights it searches with.

Once every group a weighted sum can hold is earned, what is left is how the pieces stand together: what a piece is
worth where it stands depends on where the others stand. The quiet positions her longer look searched are kept, and a
judgement over which kind stands where, from the side to move, is learned from what her first look missed of what the
longer look found (core/cognition/a_quick_judgement_from_slow_ones.py). It is tried as a group like any other, once it
foretells positions it was not fitted on better than her first look does, and kept only where it wins its match.
Measured 22 November 2026 on 84,299 such positions against Stockfish held to depth 5, 40 games on each of four seeds:
0.447, 0.474, 0.550 and 0.538 with it, 0.412, 0.450, 0.350 and 0.425 without.

What she measures is hers, about this one game, kept with what she learned of it. Nothing is measured for her and
shipped; nothing here knows a game.
"""
from __future__ import annotations

import io
import logging
import math
import time
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from core.agency.pieces_by_their_kind import Family, Position
from core.agency.searching_by_kind import PER_SQUARE, Weights, described, first_weights, placed

logger = logging.getLogger("Aura.WhatAKindIsWorth")

__all__ = ["GROUPS", "HER_LONGER_LOOK", "RESULTS", "TOGETHER", "LookedAtLonger", "Rehearsal", "a_first_rehearsal",
           "calibrated_scale", "fitted", "rehearse", "tried_as"]

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
#: What a weighted sum cannot hold: how the pieces stand together, learned from her longer look.
TOGETHER = "how the pieces stand together"
#: What is fitted, a group at a time, in the order a weak player's games can tell them apart.
GROUPS = ("worth", "turn and safety", "moves", "what is left to be taken", "where each stands", TOGETHER)
#: Who teaches her a position: how her games ended, or what a much longer look of hers finds it worth.
RESULTS = "how her games ended"
HER_LONGER_LOOK = "her longer look"
#: Nodes of the longer look that teaches a position once results cannot: twenty times a rehearsal move's.
LONGER_LOOK_NODES = 30_000
#: Tries of a group taught by results, failed, before her longer look teaches instead.
RESULTS_TRIED_AT_MOST = 2
#: Positions her longer look searched, kept at most, the latest: about seven and a half megabytes of her disk.
LOOKED_AT_LONGER_AT_MOST = 100_000


@dataclass
class LookedAtLonger:
    """The quiet positions her longer look searched: each one's pieces as the side to move sees them, her first
    judgement of it for that side, what the longer look found it worth for that side, and the game it came from. A
    position is quiet where a look at the takes alone leaves her first judgement as it is: what is left to learn there is
    how the pieces stand, not what is about to be taken."""

    #: The kinds in the order a position's parts count them; a kind seen later is added at the end.
    kinds: list[str] = field(default_factory=list)
    squares: int = 0
    columns: int = 0
    parts: list[np.ndarray] = field(default_factory=list)
    glance: list[float] = field(default_factory=list)
    slow: list[float] = field(default_factory=list)
    game: list[int] = field(default_factory=list)
    #: The scale her results showed her longer look's scores on, a round at a time.
    scales: list[float] = field(default_factory=list)

    def add(self, played: dict, round_: int, rows: int, columns: int) -> None:
        """The positions of one round of her games, where her compiled search gave what is needed of them."""
        cells, statics = played.get("sample_cells"), played.get("sample_static")
        quiet, turns, scores = played.get("sample_quiet"), played.get("sample_turns"), played.get("sample_scores")
        order, games = played.get("kind_order"), played.get("sample_games") or []
        if not (cells and statics and quiet and turns and scores and order) or len({len(cells), len(statics),
                                                                                     len(quiet), len(turns),
                                                                                     len(scores)}) != 1:
            return
        squares = rows * columns
        self.squares, self.columns = squares, columns
        for letter in order:
            if letter not in self.kinds:
                self.kinds.append(letter)
        slot = np.array([self.kinds.index(letter) for letter in order], dtype=np.int64)
        results = [float(result) for _features, result in played.get("samples") or []]
        if len(results) == len(scores):
            self.scales.append(calibrated_scale(list(scores), results))
        for n, (seen, glance, settled, turn, score) in enumerate(zip(cells, statics, quiet, turns, scores, strict=True)):
            if not settled or abs(int(score)) >= 3000:
                continue
            index = np.asarray(seen, dtype=np.int64)
            kind, rest = np.divmod(index, 2 * squares)
            self.parts.append(((slot[kind] * 2 * squares) + rest).astype(np.int16))
            self.glance.append(float(glance))
            self.slow.append(float(score) if int(turn) == 0 else -float(score))
            self.game.append(round_ * 10_000 + (int(games[n]) if n < len(games) else 0))
        extra = len(self.parts) - LOOKED_AT_LONGER_AT_MOST
        if extra > 0:
            del self.parts[:extra], self.glance[:extra], self.slow[:extra], self.game[:extra]

    def judgement(self, *, seed: int = 0) -> dict[str, Any] | None:
        """How the pieces stand together, learned from what her first look missed of what her longer look found, as
        weights her compiled search reads; None where nothing learned yet foretells positions it was not fitted on
        better than her first look."""
        from core.cognition.a_quick_judgement_from_slow_ones import learned_from_slow_ones

        if not self.parts or not self.scales or not self.squares:
            return None
        width = max(len(p) for p in self.parts)
        parts = np.full((len(self.parts), width), -1, dtype=np.int64)
        for n, p in enumerate(self.parts):
            parts[n, :len(p)] = p
        learned = learned_from_slow_ones(parts, np.array(self.glance), np.array(self.slow), np.array(self.game),
                                         inputs=len(self.kinds) * 2 * self.squares,
                                         scale=float(np.median(self.scales)), seed=seed)
        if learned is None:
            return None
        rows = learned.first

        def by_kind(theirs: int) -> dict[str, list[list[float]]]:
            return {letter: [[round(float(x), 3) for x in row]
                             for row in rows[(k * 2 + theirs) * self.squares:(k * 2 + theirs + 1) * self.squares]]
                    for k, letter in enumerate(self.kinds)}

        return {"hidden": int(rows.shape[1]), "columns": self.columns, "own": by_kind(0), "theirs": by_kind(1),
                "first_bias": [round(float(x), 3) for x in learned.first_bias],
                "out": [round(float(x), 5) for x in learned.out], "out_bias": round(float(learned.out_bias), 3),
                "held_out": [round(learned.held_out_before, 6), round(learned.held_out_after, 6)]}

    def as_bytes(self) -> bytes:
        """Kept on disk as one compressed file of arrays."""
        width = max((len(p) for p in self.parts), default=0)
        parts = np.full((len(self.parts), width), -1, dtype=np.int16)
        for n, p in enumerate(self.parts):
            parts[n, :len(p)] = p
        out = io.BytesIO()
        np.savez_compressed(out, kinds=np.array(self.kinds), squares=np.array(self.squares),
                            columns=np.array(self.columns), parts=parts,
                            glance=np.array(self.glance, dtype=np.float32), slow=np.array(self.slow, dtype=np.float32),
                            game=np.array(self.game, dtype=np.int64), scales=np.array(self.scales, dtype=np.float32))
        return out.getvalue()

    @classmethod
    def from_bytes(cls, data: bytes) -> LookedAtLonger:
        held = np.load(io.BytesIO(data), allow_pickle=False)
        return cls(kinds=[str(k) for k in held["kinds"]], squares=int(held["squares"]), columns=int(held["columns"]),
                   parts=[row[row >= 0].astype(np.int16) for row in held["parts"]],
                   glance=[float(x) for x in held["glance"]], slow=[float(x) for x in held["slow"]],
                   game=[int(x) for x in held["game"]], scales=[float(x) for x in held["scales"]])


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
    #: The quiet positions her longer look searched, for learning how the pieces stand together.
    looked_at_longer: LookedAtLonger = field(default_factory=LookedAtLonger)
    #: How the last judgement of how the pieces stand together that was tried foretold positions held back from it,
    #: as a part of how her first look foretold them: one before any is tried.
    together_tried_at: float = 1.0


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


def tried_as(group: str, teacher: str) -> str:
    """What a try of ``group`` is counted under: the group, and who taught it where that is not her results. A group
    taught by another teacher is another candidate, and the tries of the first say nothing of it. LIVE 22 November 2026
    a group tried 71 times was kept by none: its candidates taught by her longer look scored up to 0.646 +- 0.072, two
    standard errors above even, against a bar its candidates taught by results had raised to three."""
    return group if teacher == RESULTS else f"{group}, taught by {teacher}"


def _pace(engine: Any, description: str, begin: str, candidate: str, current: str, nodes: int) -> float:
    """How many positions a second her compiled search looks at with the candidate's weights, as a part of how many it
    looks at with the current ones: from the start, each timed twice by turns, the faster of each pair kept, so that
    whatever else the machine is doing counts against neither. A difference no larger than the two timings of one
    side differ by is no difference: on a busy machine, candidates that differed from her weights only in a weighted
    sum, and so ran at her pace, were timed at 0.907 and 0.982 of it."""
    looks = max(1, nodes * 10)
    timed: dict[str, list[float]] = {candidate: [], current: []}
    for _ in range(2):
        for weights in (candidate, current):
            began = time.perf_counter()
            try:
                found = engine.search(description, begin, weights, 60.0, 64, 4, looks)
            # not a failure: an engine that cannot be timed this way is taken to look at as many either way.
            except (AttributeError, TypeError, ValueError):
                return 1.0
            timed[weights].append((time.perf_counter() - began) / max(1, int(found.get("nodes", looks) or looks)))
    if not all(v > 0 and math.isfinite(v) for times in timed.values() for v in times):
        return 1.0
    pace = min(timed[current]) / min(timed[candidate])
    scatter = max(max(times) / min(times) for times in timed.values())
    return 1.0 if abs(math.log(pace)) <= math.log(scatter) else pace


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
    if looked_longer:
        rehearsal.looked_at_longer.add(played, rehearsal.rounds, family.rows, family.columns)
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
    if trying == TOGETHER:
        learned = rehearsal.looked_at_longer.judgement(seed=seed)
        if learned is None:
            rehearsal.last = {"samples": len(rehearsal.looked_at_longer.parts), "tried": trying, "kept": False,
                              "why": "nothing learned yet foretells her longer look better than her first look"}
            return rehearsal
        before, after = learned.get("held_out", [1.0, 1.0])
        foretells = after / before if before > 0 else 1.0
        if foretells >= rehearsal.together_tried_at:
            # Each try asks more of the next, so one no better than the last is not tried: LIVE 22 November 2026 her
            # first round between games tried one learned from 1,891 positions.
            rehearsal.last = {"samples": len(rehearsal.looked_at_longer.parts), "tried": trying, "kept": False,
                              "why": "it foretells her longer look no better than the one tried last",
                              "foretells": round(foretells, 4), "tried_at": round(rehearsal.together_tried_at, 4)}
            return rehearsal
        rehearsal.together_tried_at = foretells
        candidate = replace(rehearsal.weights, learned=learned)
    else:
        candidate = fitted(family, np.array(rehearsal.samples_x, dtype=np.float64),
                           np.array(rehearsal.samples_y, dtype=np.float64), groups)
        if candidate is not None:
            # A refit of what a weighted sum holds keeps how the pieces stand together as it was learned.
            candidate.learned = rehearsal.weights.learned
    if candidate is None:
        rehearsal.last = {"samples": len(rehearsal.samples_y), "tried": trying, "kept": False,
                          "why": "too few positions to fit"}
        return rehearsal
    theirs, mine = candidate.as_json(), rehearsal.weights.as_json()
    # Each side looks at as many positions a move as its pace affords in the same time: a judgement that slows her
    # search is judged with the time it costs, as it is in a game on a clock.
    pace = _pace(engine, description, begin, theirs, mine, nodes)
    try:
        match = engine.self_play(description, begin, theirs, mine, games, max(1, round(nodes * pace)),
                                 random_plies=RANDOM_PLIES, seed=seed + 7919, nodes_per_move_b=nodes)
    # not a failure: a build of the compiled search from before paces gives both sides the same positions.
    except TypeError:
        pace = 1.0
        match = engine.self_play(description, begin, theirs, mine, games, nodes, random_plies=RANDOM_PLIES,
                                 seed=seed + 7919)
    results = np.array(match["results"], dtype=np.float64)
    standing, spread = _standing_where_it_stopped(match, rehearsal.weights, family)
    if spread > 0 and len(standing) == len(results):
        for game, points in enumerate(results):
            if points == 0.5:
                # The candidate played first in the even games.
                results[game] = _as_a_result(standing[game] if game % 2 == 0 else -standing[game], spread)
    mean = float(results.mean()) if len(results) else 0.5
    error = float(results.std(ddof=1) / math.sqrt(len(results))) if len(results) > 1 else 1.0
    tries = 1
    if trying:
        # A refit of the groups she has, with none new to try, is no group's try.
        counted = tried_as(trying, rehearsal.teacher)
        tries = rehearsal.tries[counted] = rehearsal.tries.get(counted, 0) + 1
    kept = mean - 0.5 > how_far_above_even(tries) * max(error, 1e-9)
    rehearsal.last = {"samples": len(rehearsal.samples_y), "tried": trying, "scored": round(mean, 3),
                      "error": round(error, 3), "kept": kept, "pace": round(pace, 3)}
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
