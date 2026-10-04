#!/usr/bin/env python3
"""How well she plays worlds that do not wait, against a player pressing keys at random.

Each world in tests/fixtures/worlds_that_move is a small canvas game served to
a headless browser. She is given what any player gets: pictures of the canvas,
keys and clicks, and the list of keys a player is usually told about. The
game's own tally (``window.__world``) is read only to say how it went.

    python tools/measure_playing_as_it_happens.py paddle catch --seconds 60
    python tools/measure_playing_as_it_happens.py --held-out --seeds 1 2 3

Worlds are split in two. Work on the play stack is done against the first
set; the second is run only once the code is fixed, and a change made for a
failure there is checked against every world before it ships.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

WORLDS = ROOT / "tests" / "fixtures" / "worlds_that_move"
WORKED_ON = ["paddle", "catch"]
HELD_OUT = ["dodge", "shooter", "collect", "gallery"]
TOLD_KEYS = ["up", "down", "left", "right", "space"]
_KEY = {"up": "ArrowUp", "down": "ArrowDown", "left": "ArrowLeft", "right": "ArrowRight", "space": "Space"}


class _Page:
    """Eyes and hands on one canvas in a page."""

    def __init__(self, page, box):
        self.page, self.box = page, box
        self.clip = {"x": box[0], "y": box[1], "width": box[2], "height": box[3]}

    async def look(self):
        import cv2
        import numpy as np

        try:
            data = await self.page.screenshot(clip=self.clip, type="jpeg", quality=80)
        except Exception:  # noqa: BLE001 - a closed page ends the run
            return None
        at = time.monotonic()
        picture = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        return picture[:, :, ::-1], at

    async def down(self, key):
        await self.page.keyboard.down(_KEY.get(key, key))

    async def up(self, key):
        await self.page.keyboard.up(_KEY.get(key, key))

    async def tap(self, key):
        await self.page.keyboard.press(_KEY.get(key, key))

    async def click(self, x, y):
        await self.page.mouse.click(self.box[0] + x * self.box[2], self.box[1] + y * self.box[3])

    async def point(self, x, y):
        await self.page.mouse.move(self.box[0] + x * self.box[2], self.box[1] + y * self.box[3])


async def _random_player(eyes, seconds):
    began = time.monotonic()
    held = ""
    while time.monotonic() - began < seconds:
        if held:
            await eyes.up(held)
        held = random.choice(TOLD_KEYS + [""])
        if held:
            await eyes.down(held)
        if random.random() < 0.3:
            await eyes.click(random.random(), random.random())
        await asyncio.sleep(0.25)
    if held:
        await eyes.up(held)
    return {"ended": "out of time"}


async def _one(browser, world, seed, seconds, player):
    from core.agency.playing_as_it_happens import play_as_it_happens
    from core.perception.what_the_pixels_show import recognize_text

    page = await browser.new_page(viewport={"width": 800, "height": 600})
    await page.goto(f"file://{WORLDS / (world + '.html')}?start=1&seed={seed}")
    await asyncio.sleep(0.3)
    box = await page.evaluate(
        "(() => { const r = document.querySelector('canvas').getBoundingClientRect();"
        " return [r.left, r.top, r.width, r.height]; })()"
    )
    eyes = _Page(page, box)
    # What a player reads on the title screen before playing: the game's rules.
    from core.agency.playing_as_it_happens import controls_named_in

    rules = " ".join(await page.evaluate("__world.game.rules"))
    keys, pointer_first = controls_named_in(rules)
    began = time.monotonic()
    stretches, keep = [], {}
    while time.monotonic() - began < seconds:
        left = seconds - (time.monotonic() - began)
        if player == "her":
            stretch = await play_as_it_happens(
                eyes.look, eyes, keys=keys, seconds=left, read_words=recognize_text, keep=keep,
                pointer_first=pointer_first, told=rules,
            )
        else:
            stretch = await _random_player(eyes, left)
        stretches.append(stretch)
        state = await page.evaluate("__world.state")
        if state != "play":
            await page.evaluate("__world.begin()")
    tally = await page.evaluate(
        "(() => { const w = __world; return {state: w.state, score: w.score, lives: w.lives,"
        " losses: w.losses || 0, games: w.games, ended: w.ended, mine: w.mine, theirs: w.theirs,"
        " returns: w.returns, caught: w.caught}; })()"
    )
    await page.close()
    return {"world": world, "seed": seed, "player": player, "tally": tally, "stretches": stretches}


async def main(argv):
    from playwright.async_api import async_playwright

    parser = argparse.ArgumentParser()
    parser.add_argument("worlds", nargs="*")
    parser.add_argument("--held-out", action="store_true")
    parser.add_argument("--seeds", nargs="*", type=int, default=[1])
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--players", nargs="*", default=["her", "random"])
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)
    worlds = args.worlds or (HELD_OUT if args.held_out else WORKED_ON)
    results = []
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        for world in worlds:
            for seed in args.seeds:
                for player in args.players:
                    result = await _one(browser, world, seed, args.seconds, player)
                    results.append(result)
                    last = result["stretches"][-1] if result["stretches"] else {}
                    print(json.dumps({
                        "world": world, "seed": seed, "player": player, "tally": result["tally"],
                        "stretches": len(result["stretches"]),
                        "learned": last.get("learned"), "hers": last.get("hers"),
                        "fires": last.get("fires"), "fps": last.get("pictures_a_second"),
                        "said": [line for s in result["stretches"] for line in s.get("said", [])][:8],
                    }), flush=True)
        await browser.close()
    return results


def _write(results, out):
    if out:
        Path(out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    _out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else ""
    _write(asyncio.run(main(sys.argv[1:])), _out)
