"""Watching someone play: what the player in a recording made their thing do, read as acts she can do, what each was
for, the kinds of thing they went for and kept clear of, and the places along their way.

A recording has no keys in it. The systems that learn from people's play read the actions back out of what changed:
VPT labelled Minecraft video with the actions that must have caused each change (Baker et al. 2022); behavioural cloning
from observation does it with the learner's own inverse model, learned from its own play (Torabi et al. 2018); robots
learning from people's video read the motion of the object, not the hand, and make that motion with their own body
(Im2Flow2Act and ATM, 2024). Here the player's thing is found by how it moves: it starts, stops and turns where nothing
made it (not at a wall, not at a touch), and it lasts while what it touches goes. Its going, picture to picture, is
the act; where her own keys are known (core/agency/which_one_answers_to_her.py), the act is the key of hers that makes
that going.

What each act was for is read the way people read goals off moves, by asking which goal makes it the sensible thing to
do (inverse planning, Baker, Saxe and Tenenbaum 2009): going toward a thing that it then took was going for it; going
away from one that was closing was keeping clear of it. Over a recording that says which kinds a player wants and which
they avoid, before she has touched any of them; and what took the player's thing away says what costs. Along the way,
the places it passed are kept as a coarse look each (checkpoints, as Aytar et al. 2018 made of one YouTube video): how
far along a player's way a picture of hers is.

Nothing here knows a game. Kinds are told apart by look, so what was learned from a recording is matched to the kinds
of her own play by colour and size, as what she keeps of a game is (core/agency/what_she_keeps_of_a_game.py).
"""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["Footage", "Moment", "how_far_along", "read_a_recording", "read_footage", "stance_by_look", "stance_watched"]

#: How near, in the player's thing's own sizes, two things must be to have met, and to matter to the going.
MET_WITHIN, NEAR_WITHIN = 1.2, 6.0
#: How soon after a meeting a thing that went was taken by it, and the player's thing lost to it, in seconds.
TAKEN_WITHIN_S = 0.3
#: The coarse look a checkpoint keeps of a picture, and how often one is kept while nothing new shows.
CHECKPOINT_CELLS = (16, 12)
CHECKPOINT_EVERY_S = 2.0
#: What the moves are called, by the way the going points.
LEFT, RIGHT, UP, DOWN, STILL = "left", "right", "up", "down", "still"


@dataclass
class Moment:
    """A choice the player made: when, what they did, and what it was for where that can be read."""

    at: float
    act: str
    key: str = ""
    toward: int | None = None
    away_from: int | None = None

    def said(self, named: Mapping[int, str]) -> str:
        doing = f"pressed {self.key}" if self.key else f"went {self.act}" if self.act != STILL else "stopped"
        if self.toward is not None:
            return f"{doing} to get the {named.get(self.toward, 'thing')}"
        if self.away_from is not None:
            return f"{doing} out of the way of the {named.get(self.away_from, 'thing')}"
        return doing


@dataclass
class Footage:
    """What a recording of someone playing showed."""

    hers: int | None = None
    looks: dict[int, tuple[tuple[int, int, int], float]] = field(default_factory=dict)
    moments: list[Moment] = field(default_factory=list)
    went_for: Counter = field(default_factory=Counter)
    kept_clear: Counter = field(default_factory=Counter)
    taken: Counter = field(default_factory=Counter)
    lost_to: Counter = field(default_factory=Counter)
    checkpoints: list[np.ndarray] = field(default_factory=list)
    seconds: float = 0.0

    def stances(self) -> dict[int, str]:
        """Each kind the player plainly went for ("meet") or plainly kept clear of ("avoid")."""
        found = {}
        for kind in set(self.went_for) | set(self.kept_clear) | set(self.taken) | set(self.lost_to):
            wanted = self.went_for[kind] + 2 * self.taken[kind]
            unwanted = self.kept_clear[kind] + 3 * self.lost_to[kind]
            if wanted >= 2 and wanted > 2 * unwanted:
                found[kind] = "meet"
            elif unwanted >= 2 and unwanted > 2 * wanted:
                found[kind] = "avoid"
        return found

    def as_memory(self) -> dict[str, Any]:
        """What is kept of it with the game: each kind's stance by its look, and the checkpoints."""
        return {"stances": [[list(self.looks[k][0]), round(float(self.looks[k][1]), 2), stance]
                            for k, stance in self.stances().items() if k in self.looks],
                "hers": ([list(self.looks[self.hers][0]), round(float(self.looks[self.hers][1]), 2)]
                         if self.hers in self.looks else None),
                "checkpoints": [c.astype(int).tolist() for c in self.checkpoints[:60]]}

    def said(self, named: Mapping[int, str] | None = None) -> list[str]:
        named = named or {}
        lines = [f"The player went for the {named.get(k, 'thing')}" if s == "meet" else
                 f"The player kept clear of the {named.get(k, 'thing')}" for k, s in self.stances().items()]
        return lines + [m.said(named) for m in self.moments[:3]]


def _the_way(vx: float, vy: float, size: float) -> str:
    if math.hypot(vx, vy) < 0.5 * size:
        return STILL
    if abs(vx) >= abs(vy):
        return RIGHT if vx > 0 else LEFT
    return DOWN if vy > 0 else UP


def _key_for(vx: float, vy: float, ways: Mapping[str, tuple[float, float]]) -> str:
    """The key of hers whose going points most nearly as this going does."""
    speed = math.hypot(vx, vy)
    if not ways or speed <= 0:
        return ""
    best, score = "", 0.5
    for key, (wx, wy) in ways.items():
        norm = math.hypot(wx, wy)
        if norm > 0 and (cos := (vx * wx + vy * wy) / (speed * norm)) > score:
            best, score = key, cos
    return best


def _coarse(picture: np.ndarray) -> np.ndarray:
    rows, columns = CHECKPOINT_CELLS[1], CHECKPOINT_CELLS[0]
    tall, wide = picture.shape[:2]
    cells = np.zeros((rows, columns, 3), np.float32)
    for r in range(rows):
        for c in range(columns):
            patch = picture[r * tall // rows:(r + 1) * tall // rows, c * wide // columns:(c + 1) * wide // columns]
            cells[r, c] = patch.reshape(-1, patch.shape[-1])[:, :3].mean(axis=0) if patch.size else 0
    return cells


def how_far_along(picture: np.ndarray, checkpoints: list[np.ndarray]) -> int | None:
    """Which of a player's checkpoints a picture of hers looks most like; None where there are none."""
    if not checkpoints:
        return None
    here = _coarse(picture)
    return int(np.argmin([float(np.abs(here - np.asarray(c, np.float32)).mean()) for c in checkpoints]))


def stance_by_look(kept: Mapping[str, Any], colour: tuple[int, int, int], size: float) -> str:
    """The stance a recording showed for a kind that looks like this one; "" where it showed none."""
    for seen_colour, seen_size, stance in kept.get("stances") or []:
        apart = max(abs(int(a) - int(b)) for a, b in zip(colour, seen_colour, strict=False)) / 255.0
        if apart < 45 / 255.0 and max(size, seen_size) / max(1e-6, min(size, seen_size)) < 2.5:
            return str(stance)
    return ""


def stance_watched(watched: Mapping[str, Any], moves: Any, kind: int) -> str:
    """The stance a recording showed for a kind of her own play, matched by its look; "" where it showed none."""
    found = next((k for k in getattr(moves, "kinds", ()) or () if getattr(k, "number", None) == kind), None)
    if found is None or not watched:
        return ""
    return stance_by_look(watched, tuple(int(c) for c in found.colour), float(found.size))


def read_footage(frames: Iterable[tuple[np.ndarray, float]], *, ways: Mapping[str, tuple[float, float]] | None = None,
                 hers_look: tuple[tuple[int, int, int], float] | None = None) -> Footage:
    """What someone's play showed, picture by picture. ``ways`` are her own keys' goings, where known, to read the
    player's acts as hers; ``hers_look`` the look of her own thing in this game, where known, to find the player's."""
    from core.perception.what_moves_in_the_picture import WhatMoves

    moves = WhatMoves()
    footage = Footage()
    tracks: dict[int, list[tuple[float, float, float, float, float]]] = {}
    kinds_of: dict[int, int] = {}
    gone: list[tuple[float, int, int, float, float]] = []
    began = last_checkpoint = None
    for picture, at in frames:
        began = at if began is None else began
        happened = moves.see(picture, at)
        if last_checkpoint is None or at - last_checkpoint >= CHECKPOINT_EVERY_S or any(
                h.get("what") == "new screen" for h in happened):
            footage.checkpoints.append(_coarse(np.asarray(picture)))
            last_checkpoint = at
        for thing in moves.things.values():
            tracks.setdefault(thing.number, []).append((at, thing.x, thing.y, thing.vx, thing.vy, max(thing.w, thing.h)))
            kinds_of[thing.number] = thing.kind
        # What leaves by the picture's edge has gone out of view, not been taken.
        tall, wide = moves.shape
        gone += [(h["at"], h["thing"], h["kind"], h["x"], h["y"]) for h in happened if h.get("what") == "gone"
                 and min(h["x"], wide - h["x"], h["y"], tall - h["y"]) > max(h["width"], h["height"])]
        footage.seconds = at - began
    for kind in moves.kinds:
        footage.looks[kind.number] = (tuple(int(c) for c in kind.colour), float(kind.size))
    player = _the_players_thing(moves, tracks, kinds_of, footage.looks, hers_look)
    if player is None:
        return footage
    footage.hers = kinds_of[player]
    size = max(1.0, float(np.median([step[5] for step in tracks[player]])))
    _read_the_going(footage, tracks, kinds_of, player, size, ways or {}, gone)
    return footage


def _the_players_thing(moves: Any, tracks: Mapping[int, list], kinds_of: Mapping[int, int],
                       looks: Mapping[int, tuple], hers_look: tuple | None) -> int | None:
    """The thing that starts, stops and turns of its own accord and lasts; the one that looks like hers, where that is
    known."""
    if hers_look is not None:
        from core.perception.what_moves_in_the_picture import KIND_ALIKE

        alike = [n for n, kind in kinds_of.items() if kind in looks and max(
            abs(int(a) - int(b)) for a, b in zip(looks[kind][0], hers_look[0], strict=False)) / 255.0 < KIND_ALIKE]
        if alike:
            return max(alike, key=lambda n: len(tracks.get(n, ())))
    best, score = None, 0.0
    for number, track in tracks.items():
        if len(track) < 6:
            continue
        turns = 0
        for (_t0, _x0, _y0, vx0, vy0, _s0), (_t1, x1, y1, vx1, vy1, _s1) in zip(track, track[1:], strict=False):
            change = math.hypot(vx1 - vx0, vy1 - vy0)
            if change > 0 and _the_way(vx0, vy0, 1.0) != _the_way(vx1, vy1, 1.0):
                # A turn at a wall is the wall's; a turn of its own is the player's.
                tall, wide = moves.shape if moves.shape[0] else (1, 1)
                at_a_wall = min(x1, wide - x1, y1, tall - y1) < 4.0
                turns += 0 if at_a_wall else 1
        lasting = len(track)
        if turns * lasting > score:
            best, score = number, turns * lasting
    return best


def _read_the_going(footage: Footage, tracks: Mapping[int, list], kinds_of: Mapping[int, int], player: int,
                    size: float, ways: Mapping[str, tuple[float, float]], gone: list[tuple]) -> None:
    """The player's acts, what each was for, and what was taken and what took the player's thing."""
    track = tracks[player]
    radius = NEAR_WITHIN * size
    last_way = STILL
    for at, x, y, vx, vy, _size in track:
        way = _the_way(vx, vy, size)
        near = [(n, t) for n, t in ((n, _at(tracks[n], at)) for n in tracks if n != player) if t is not None
                and math.hypot(t[0] - x, t[1] - y) <= radius]
        for number, (tx, ty, tvx, tvy) in near:
            closing = (tvx * (x - tx) + tvy * (y - ty)) > 0
            toward = (vx * (tx - x) + vy * (ty - y)) / max(1e-6, math.hypot(tx - x, ty - y))
            kind = kinds_of[number]
            if toward > size and _taken_later(gone, number, at):
                footage.went_for[kind] += 1
            elif closing and toward < -size:
                footage.kept_clear[kind] += 1
        if way != last_way:
            moment = Moment(at, way, key=_key_for(vx, vy, ways))
            ahead = [(n, t) for n, t in near if (vx * (t[0] - x) + vy * (t[1] - y)) > 0]
            behind = [(n, t) for n, t in near if (t[2] * (x - t[0]) + t[3] * (y - t[1])) > 0]
            took = [n for n, _t in ahead if _taken_later(gone, n, at)]
            if took:
                moment.toward = kinds_of[took[0]]
            elif behind and way != STILL:
                moment.away_from = kinds_of[behind[0][0]]
            footage.moments.append(moment)
            last_way = way
    for at, number, kind, gx, gy in gone:
        here = _at(track, at)
        if number == player:
            others = [kinds_of[n] for n in tracks if n != player and (t := _at(tracks[n], at)) is not None
                      and here is not None and math.hypot(t[0] - here[0], t[1] - here[1]) <= MET_WITHIN * size]
            for other in others[:1]:
                footage.lost_to[other] += 1
        elif here is not None and math.hypot(gx - here[0], gy - here[1]) <= MET_WITHIN * size:
            footage.taken[kind] += 1


def _at(track: list[tuple[float, ...]], at: float) -> tuple[float, float, float, float] | None:
    """Where a track had its thing at ``at``, and how it was going; None where it was not followed then."""
    before = [step for step in track if step[0] <= at]
    if not before or at - before[-1][0] > 0.25:
        return None
    _t, x, y, vx, vy, _size = before[-1]
    return x, y, vx, vy


def _taken_later(gone: list[tuple], number: int, at: float, within: float = 3.0) -> bool:
    return any(n == number and at <= t <= at + within for t, n, *_rest in gone)


def read_a_recording(path: str, *, per_second: float = 10.0, wide: int = 480, most_s: float = 600.0,
                     ways: Mapping[str, tuple[float, float]] | None = None,
                     hers_look: tuple[tuple[int, int, int], float] | None = None) -> Footage:
    """A recording on disk read as footage: its pictures taken ``per_second`` at ``wide`` pixels across, up to
    ``most_s`` seconds, by ffmpeg."""
    import subprocess

    from core.runtime.subprocess_gateway import get_subprocess_gateway

    probe = get_subprocess_gateway().run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                          "stream=width,height", "-of", "csv=p=0", path],
                                         timeout=30, read_only=True, source="watching_someone_play.ffprobe",
                                         accelerator_capability="none")
    width, height = (int(v) for v in probe.stdout.strip().split(",")[:2])
    tall = max(2, int(round(height * wide / width / 2)) * 2)
    # Read-only: ffmpeg decodes the recording to its output and writes nowhere.
    process = get_subprocess_gateway().spawn(["ffmpeg", "-v", "error", "-t", str(most_s), "-i", path, "-vf",
                                              f"fps={per_second},scale={wide}:{tall}", "-f", "rawvideo", "-pix_fmt",
                                              "rgb24", "-"],
                                             stdout=subprocess.PIPE, text=False, read_only=True,
                                             source="watching_someone_play.ffmpeg", accelerator_capability="none")

    def frames() -> Iterable[tuple[np.ndarray, float]]:
        size, index = wide * tall * 3, 0
        assert process.stdout is not None
        while chunk := process.stdout.read(size):
            if len(chunk) < size:
                break
            yield np.frombuffer(chunk, np.uint8).reshape(tall, wide, 3), index / per_second
            index += 1

    try:
        return read_footage(frames(), ways=ways, hers_look=hers_look)
    finally:
        process.kill()
        process.wait(timeout=10)
