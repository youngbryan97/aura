#!/usr/bin/env python3
"""Put her in camera worlds she has never seen and count how often she does what she is asked.

SIMA 2 is measured on tasks grouped by skill, in games it trained on and in
games it never saw, against success checks that do not depend on the agent's
own account of itself (arXiv 2512.04797, §3.4). This is the same kind of
measurement for her camera loop, in generated worlds, so it runs anywhere and
reruns the same.

Every world is different in the ways that break an agent tuned on one: what
the walking key is (w, up or i), which way the mouse turns the camera and how
far a point turns it, where the things are and what they are called, and
which key each thing's prompt names. She is told none of it; each world
starts with her learning her body.

    navigation      go to a thing in view, and arrive
    using           go to a thing and do what its prompt says; the world answers
    finding         go to a thing that starts behind her
    asking          two things answer to the name; she must ask, not walk

    python tools/measure_in_camera_worlds.py --worlds 10
"""

from __future__ import annotations

import argparse
import asyncio
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.agency.what_hands_do import Chunk  # noqa: E402
from core.skills.in_a_world_through_a_camera import (  # noqa: E402
    ACameraWorld,
    go_to,
    learn_the_body,
)

VIEW = (240, 640)
FIELD = 90.0
PX_PER_DEGREE = VIEW[1] / FIELD
PACE = 3.0
REACH = 2.0
RADIUS = 15.0
NAMES = ("door", "chest", "well", "lamp", "barrel", "gate", "statue", "crate", "altar", "anvil")
COLOURS = ("red", "blue", "green", "gold")
WALKING = (("w", "s"), ("up", "down"), ("i", "k"))
PROMPT_KEYS = ("e", "f", "r", "space")


@dataclass
class Thing:
    name: str
    x: float
    y: float
    key: str
    used: bool = False


@dataclass
class AGeneratedWorld:
    """A round room with things in it, seen through a camera."""

    seed: int
    things: list[Thing] = field(default_factory=list)
    walks: str = "w"
    backs: str = "s"
    turn: float = 0.25
    facing: float = 0.0
    x: float = 0.0
    y: float = 0.0
    walked: bool = False

    def __post_init__(self) -> None:
        roll = random.Random(self.seed)
        self.walks, self.backs = roll.choice(WALKING)
        self.turn = roll.choice((-1, 1)) * roll.uniform(0.15, 0.4)
        wall = np.random.default_rng(self.seed)
        self.panorama = np.kron(wall.random((30, 160)), np.ones((12, 16))) * 255.0

    def place(self, name: str, bearing: float, far: float, key: str) -> Thing:
        thing = Thing(name, far * math.sin(math.radians(bearing)), far * math.cos(math.radians(bearing)), key)
        self.things.append(thing)
        return thing

    def _zoom(self) -> float:
        ux, uy = math.sin(math.radians(self.facing)), math.cos(math.radians(self.facing))
        along = self.x * ux + self.y * uy
        ahead = -along + math.sqrt(max(0.0, along * along - (self.x ** 2 + self.y ** 2 - RADIUS ** 2)))
        return min(20.0, max(0.5, RADIUS / max(ahead, 0.5)))

    def frame(self) -> np.ndarray:
        high, wide = VIEW
        zoom = self._zoom()
        crop_h, crop_w = int(high / zoom), int(wide / zoom)
        left = int(round(self.facing * PX_PER_DEGREE)) + (wide - crop_w) // 2
        top = (self.panorama.shape[0] - crop_h) // 2
        rows = np.clip(top + np.linspace(0, crop_h - 1, high).round().astype(int), 0, self.panorama.shape[0] - 1)
        cols = (left + np.linspace(0, crop_w - 1, wide).round().astype(int)) % self.panorama.shape[1]
        return self.panorama[np.ix_(rows, cols)]

    def _bearing(self, thing: Thing) -> tuple[float, float]:
        dx, dy = thing.x - self.x, thing.y - self.y
        off = (math.degrees(math.atan2(dx, dy)) - self.facing + 180.0) % 360.0 - 180.0
        return off, math.hypot(dx, dy)

    def layout(self) -> list[dict]:
        seen = []
        for thing in self.things:
            off, far = self._bearing(thing)
            if abs(off) < FIELD / 2:
                high = min(0.9, 1.0 / max(far, 0.1))
                seen.append({"text": thing.name.title(), "center_x": 0.5 + off / FIELD,
                             "center_y": 0.5, "width": high * 0.5, "height": high})
                if far < REACH and abs(off) < 12 and not thing.used:
                    said = "Press SPACE" if thing.key == "space" else f"Press {thing.key.upper()}"
                    seen.append({"text": f"{said} to use the {thing.name}", "center_x": 0.5,
                                 "center_y": 0.85, "width": 0.3, "height": 0.03})
            if thing.used:
                seen.append({"text": f"The {thing.name} was used", "center_x": 0.2,
                             "center_y": 0.1, "width": 0.25, "height": 0.03})
        return seen

    def look(self):
        return self.frame(), self.layout()

    async def play(self, chunk: Chunk) -> None:
        for slot in chunk.slots:
            self.facing += self.turn * slot.moved[0]
            ahead = 1.0 if self.walks in slot.held else -1.0 if self.backs in slot.held else 0.0
            if ahead:
                step = ahead * PACE * chunk.slot_s
                self.x += step * math.sin(math.radians(self.facing))
                self.y += step * math.cos(math.radians(self.facing))
            for thing in self.things:
                off, far = self._bearing(thing)
                if thing.key in slot.held and far < REACH and abs(off) < 12:
                    thing.used = True


def a_world_for(category: str, seed: int) -> tuple[AGeneratedWorld, str]:
    """A world and the name of the thing she is sent to, set up for one category."""
    roll = random.Random(f"{category}-{seed}")
    world = AGeneratedWorld(seed)
    names = roll.sample(NAMES, 3)
    if category == "asking":
        colours = roll.sample(COLOURS, 2)
        world.place(f"{colours[0]} {names[0]}", roll.uniform(-35, -8), roll.uniform(6, 11), roll.choice(PROMPT_KEYS))
        world.place(f"{colours[1]} {names[0]}", roll.uniform(8, 35), roll.uniform(6, 11), roll.choice(PROMPT_KEYS))
        return world, names[0]
    ahead = roll.uniform(-35, 35) if category != "finding" else roll.uniform(120, 240)
    world.place(names[0], ahead, roll.uniform(6, 11), roll.choice(PROMPT_KEYS))
    for other in names[1:]:
        world.place(other, roll.uniform(-180, 180), roll.uniform(6, 12), roll.choice(PROMPT_KEYS))
    return world, names[0]


def wilson(worked: int, tried: int) -> tuple[float, float]:
    """The 95% Wilson interval for a success rate."""
    if not tried:
        return 0.0, 0.0
    z, p = 1.96, worked / tried
    middle = (p + z * z / (2 * tried)) / (1 + z * z / tried)
    half = z * math.sqrt(p * (1 - p) / tried + z * z / (4 * tried * tried)) / (1 + z * z / tried)
    return max(0.0, middle - half), min(1.0, middle + half)


async def one(category: str, seed: int) -> bool:
    world, named = a_world_for(category, seed)
    loop = ACameraWorld(look=world.look, play=world.play)
    body = await learn_the_body(loop, keys=("w", "s", "up", "down", "i", "k"), slot_s=0.2)
    trip = await go_to(loop, named, body, slot_s=0.2, most_chunks=200)
    thing = next(thing for thing in world.things if named in thing.name)
    if category == "asking":
        return trip.chunks == 0 and "things answer to" in trip.ended
    if category == "navigation":
        off, far = world._bearing(thing)
        return far < REACH * 1.5 and abs(off) < FIELD / 4
    return thing.used and bool(trip.answered) and "nothing on screen" not in trip.answered


async def measure(worlds: int) -> None:
    print(f"{worlds} generated worlds per category, each with its own keys, mouse and layout\n")
    print(f"{'category':<12} {'did it':>8}  {'95% interval':>14}")
    for category in ("navigation", "using", "finding", "asking"):
        results = [await one(category, seed) for seed in range(worlds)]
        worked = sum(results)
        low, high = wilson(worked, worlds)
        print(f"{category:<12} {worked:>3}/{worlds:<4}  {low:>6.0%} - {high:>4.0%}")


def main() -> int:
    ask = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ask.add_argument("--worlds", type=int, default=10)
    said = ask.parse_args()
    asyncio.run(measure(said.worlds))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
