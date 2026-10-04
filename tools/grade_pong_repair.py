#!/usr/bin/env python3
"""Grade a repaired Pong against the flaws it was given, and against breaking what worked.

tests/fixtures/pong_repair/pong.html is a whole game of Pong with five flaws
put in on purpose. This runs a copy of it in a headless browser and asks each
question by behaviour, setting the game's own state and watching what it does,
so a repair is judged by what the game does rather than by what the code
looks like:

    controls     up moves the paddle up, down moves it down
    opponent     the computer's paddle follows the ball
    collision    a ball meeting the middle of a paddle bounces off it
    top wall     a ball going up comes back down
    scoring      a ball past the player is the computer's point, and the other way

And what worked before has to still work: the title screen starts a game, a
point serves the ball again, five points end the game, a paddle stays on the
court, and the page throws no errors.

    python tools/grade_pong_repair.py path/to/pong.html
    python tools/grade_pong_repair.py --original      # the broken copy: every flaw present

The flaws, for whoever marks the demo (the copy she works on carries no notes):

    1 movePlayer      ArrowUp adds to y and ArrowDown subtracts: the controls are reversed
    2 moveComputer    the step is multiplied by 0: the computer never moves
    3 hitsPaddle      the vertical test uses PADDLE_W where it means PADDLE_H
    4 moveBall        only the bottom wall bounces; the ball leaves through the top
    5 moveBall        a ball past the player scores for the player
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ORIGINAL = ROOT / "tests" / "fixtures" / "pong_repair" / "pong.html"

FLAWS = ("controls", "opponent", "collision", "top wall", "scoring")
STILL_WORKS = ("starts", "serves again", "ends at five", "paddle stays on court", "no errors")


async def _fresh(page, address: str) -> None:
    await page.goto(address)
    await page.wait_for_timeout(200)
    await page.keyboard.press("Space")
    await page.wait_for_timeout(100)


async def _controls(page) -> bool:
    await page.evaluate("ball.vx = 0; ball.vy = 0; ball.x = W / 2; ball.y = H / 2; player.y = H / 2 - PADDLE_H / 2")
    start = await page.evaluate("player.y")
    await page.keyboard.down("ArrowUp")
    await page.wait_for_timeout(250)
    await page.keyboard.up("ArrowUp")
    up = await page.evaluate("player.y")
    await page.keyboard.down("ArrowDown")
    await page.wait_for_timeout(500)
    await page.keyboard.up("ArrowDown")
    down = await page.evaluate("player.y")
    return up < start - 20 and down > up + 40


async def _opponent(page) -> bool:
    await page.evaluate("computer.y = 10; ball.x = W / 2; ball.y = H - 60; ball.vx = 60; ball.vy = 0")
    await page.wait_for_timeout(400)
    return await page.evaluate("computer.y") > 40


async def _collision(page) -> bool:
    # A ball that went through scores a point and is served again, half the
    # time to the right, which looks like a bounce unless the scores are read.
    await page.evaluate(
        "playerScore = 0; computerScore = 0; player.y = H / 2 - PADDLE_H / 2;"
        " ball.x = player.x + PADDLE_W + 40; ball.y = H / 2 - BALL / 2; ball.vx = -200; ball.vy = 0"
    )
    await page.wait_for_timeout(400)
    return await page.evaluate("ball.vx > 0 && ball.x > player.x && playerScore + computerScore === 0")


async def _top_wall(page) -> bool:
    await page.evaluate("ball.x = W / 2; ball.y = 30; ball.vx = 0.001; ball.vy = -200")
    await page.wait_for_timeout(400)
    return await page.evaluate("ball.vy > 0 && ball.y >= 0")


async def _scoring(page) -> bool:
    await page.evaluate("playerScore = 0; computerScore = 0; ball.x = 2; ball.y = 20; ball.vx = -400; ball.vy = 0; player.y = H - PADDLE_H")
    await page.wait_for_timeout(200)
    after_miss = await page.evaluate("[playerScore, computerScore]")
    await page.evaluate("ball.x = W - 4; ball.y = 20; ball.vx = 400; ball.vy = 0; computer.y = H - PADDLE_H")
    await page.wait_for_timeout(200)
    after_win = await page.evaluate("[playerScore, computerScore]")
    return after_miss == [0, 1] and after_win == [1, 1]


async def _starts(page) -> bool:
    return await page.evaluate("state") == "playing"


async def _serves_again(page) -> bool:
    await page.evaluate("ball.x = W + 20; ball.vx = 300")
    await page.wait_for_timeout(150)
    return await page.evaluate("Math.abs(ball.x - (W / 2 - BALL / 2)) < 60 && ball.vx !== 0")


async def _ends_at_five(page) -> bool:
    await page.evaluate("playerScore = 4; ball.x = W + 20; ball.y = 20; ball.vx = 300; computer.y = H - PADDLE_H")
    await page.wait_for_timeout(200)
    return await page.evaluate("state") == "over"


async def _stays_on_court(page) -> bool:
    await page.keyboard.down("ArrowDown")
    await page.wait_for_timeout(1500)
    await page.keyboard.up("ArrowDown")
    low = await page.evaluate("player.y")
    await page.keyboard.down("ArrowUp")
    await page.wait_for_timeout(1500)
    await page.keyboard.up("ArrowUp")
    high = await page.evaluate("player.y")
    tall = await page.evaluate("H - PADDLE_H")
    return all(0 <= value <= tall for value in (low, high))


_CHECKS = {
    "controls": _controls, "opponent": _opponent, "collision": _collision,
    "top wall": _top_wall, "scoring": _scoring, "starts": _starts,
    "serves again": _serves_again, "ends at five": _ends_at_five,
    "paddle stays on court": _stays_on_court,
}


async def grade(address: str) -> dict[str, object]:
    """The report for the page at ``address`` (a file:// address of the repaired copy)."""
    from playwright.async_api import async_playwright

    results: dict[str, object] = {}
    errors: list[str] = []
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        for name, check in _CHECKS.items():
            page = await browser.new_page(viewport={"width": 800, "height": 600})
            page.on("pageerror", lambda error: errors.append(str(error)))
            try:
                await _fresh(page, address)
                results[name] = bool(await check(page))
            except Exception as why:  # noqa: BLE001 - a broken page is a failed check
                results[name] = False
                errors.append(f"{name}: {why}")
            await page.close()
        await browser.close()
    results["no errors"] = not errors
    return {
        "file": address,
        "fixed": [flaw for flaw in FLAWS if results.get(flaw)],
        "still broken": [flaw for flaw in FLAWS if not results.get(flaw)],
        "newly broken": [part for part in STILL_WORKS if not results.get(part)],
        "checks": results,
        "errors": errors[:5],
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?")
    parser.add_argument("--original", action="store_true")
    args = parser.parse_args(argv)
    path = ORIGINAL if args.original or not args.path else Path(args.path)
    report = asyncio.run(grade(path.resolve().as_uri()))
    print(json.dumps(report, indent=1))
    return 0 if not report["still broken"] and not report["newly broken"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
