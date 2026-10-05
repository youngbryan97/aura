#!/usr/bin/env python3
"""How she does on real Flash games, played from their Internet Archive copies, offline.

The Cartoon Network list she is asked to pick from is a museum that refuses its
game files to automated browsers; each game's page links its Internet Archive
copy, which plays in any browser. This plays those copies the way the live demo
does: her own browser with nothing hiding that it is automated, the page's
drawing handed to her screen pursuit and its reflexes, asked to win. What is
offline is only her language: with no model loaded, the moments the screen
pursuit would think in words are decided without them.

    python tools/measure_on_archived_games.py --games 16 2 9 --minutes 6
    python tools/measure_on_archived_games.py --all --minutes 4 --out report.json
    python tools/measure_on_archived_games.py --games 6 --minutes 5 --trace /tmp/traces

Each game's report says how many runs she played, whether she won, how each
ended in the game's own words, and what she said while playing.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

#: The list's own order (0-55) and each game's Internet Archive copy, read off
#: the museum's game pages on 4 October 2026.
ARCHIVED = {
    0: "https://archive.org/details/scooby-snapshot_flash",
    1: "https://archive.org/details/the-riddlers-secret-identity-inventor_flash",
    2: "https://archive.org/details/food_bash",
    3: "https://archive.org/details/runaway-robot-dexter",
    5: "https://archive.org/details/ask-swami-shaggy_flash",
    6: "https://archive.org/details/cartoon_cove_mini_golf",
    7: "https://archive.org/details/operation-stat",
    8: "https://archive.org/details/courage_the_cowardly_dog_nightmare_vacation",
    9: "https://archive.org/details/ballet-parking",
    10: "https://archive.org/details/dexter_cloneadoodledoo",
    11: "https://archive.org/details/flash_candymachinedeluxe",
    12: "https://archive.org/details/samurai-jack",
    13: "https://archive.org/details/scooby-doo-and-the-creepy-castle_flash",
    14: "https://archive.org/details/scooby-doo-scooby-trap_flash",
    16: "https://archive.org/details/1100_tunnel_rush",
    17: "https://archive.org/details/knd-numbuh-generator",
    18: "https://archive.org/details/operation-tommy",
    19: "https://archive.org/details/a-friend-in-need",
    20: "https://archive.org/details/cocos-egg-scramble",
    21: "https://archive.org/details/door-to-door-game",
    22: "https://archive.org/details/mid-flight-snack",
    23: "https://archive.org/details/simply-smashing",
    24: "https://archive.org/details/wilts-wash-swoosh",
    25: "https://archive.org/details/pjinns_camp_lazlo2_flash",
    26: "https://archive.org/details/knd-tummy-trouble",
    27: "https://archive.org/details/the-batman-cobblebot-caper",
    28: "https://archive.org/details/toms_trap-o-matic",
    31: "https://archive.org/details/big-shot-checkers",
    32: "https://archive.org/details/ben-10-blockade-blitz-swf",
    33: "https://archive.org/details/ben_10_kraken_attack",
    34: "https://archive.org/details/cnreadyimfire",
    35: "https://archive.org/details/operation-zero-out-mandyd",
    36: "https://archive.org/details/flight-of-the-hamsters-game",
    37: "https://archive.org/details/rainbow-monkey-rundown",
    38: "https://archive.org/details/2199-foster-mansion-team-work",
    39: "https://archive.org/details/ppgpuppybots",
    40: "https://archive.org/details/pokemon-towering-legends_flash",
    41: "https://archive.org/details/ben-10-cavern-run-swf",
    42: "https://archive.org/details/beatz-editor",
    44: "https://archive.org/details/regular-show-all-nighter",
    45: "https://archive.org/details/blind-fooled_202408",
    46: "https://archive.org/details/annoying-orange-escape-from-dr.-fruitenstein",
    47: "https://archive.org/details/cartoon-network-snowbrawl-fight",
    52: "https://archive.org/details/tawog-water-sons",
    54: "https://archive.org/details/sonic-boom-link-n-smash",
    55: "https://archive.org/details/the-amazing-world-of-gumball-battle-bowlers",
}


async def _one(number: int, address: str, minutes: float, visible: bool, trace: Path | None = None) -> dict:
    from core.capabilities.phantom_browser import PhantomBrowser
    from core.skills import sovereign_browser_drawing as drawing

    said: list[str] = []
    import core.skills.screen_pursuit as pursuit

    original_tell = pursuit._tell

    def tell(line: str) -> None:
        said.append(" ".join(str(line).split())[:200])
        original_tell(line)

    pursuit._tell = tell
    drawing.PLAY_UNTIL_WON_S = minutes * 60.0
    # Everything she decided and why, game by game, and the screen she ended on.
    import logging

    kept = None
    if trace is not None:
        await asyncio.to_thread(trace.mkdir, parents=True, exist_ok=True)
        kept = logging.FileHandler(trace / f"game{number}.log")
        kept.setFormatter(logging.Formatter("%(asctime)s %(name)s %(message)s"))
        logging.getLogger().addHandler(kept)
        logging.getLogger().setLevel(logging.INFO)
    began = time.monotonic()
    browser = PhantomBrowser(visible=visible, browser_type="chromium", principal="owner")
    try:
        if not await browser.ensure_ready():
            return {"game": number, "error": "browser did not start"}
        await browser.page.goto(address, wait_until="domcontentloaded", timeout=60000)
        await asyncio.sleep(6)
        played = await drawing.played_on_the_drawing(browser, "Play this game and win.", {"url": address})
    except Exception as why:  # noqa: BLE001 - one game failing is a result, not a stop
        played = {"error": f"{type(why).__name__}: {why}"}
    finally:
        pursuit._tell = original_tell
        if trace is not None:
            try:
                await browser.page.screenshot(path=str(trace / f"game{number}_end.png"))
            except Exception:  # noqa: BLE001 - no last picture is still a result
                pass
            logging.getLogger().removeHandler(kept)
            kept.close()
        await browser.close()
    return {
        "game": number,
        "address": address,
        "minutes": round((time.monotonic() - began) / 60.0, 1),
        "runs": played.get("runs"),
        "won": played.get("won"),
        "error": played.get("error"),
        "last_seen": str(played.get("last_seen") or "")[:200],
        "said": said[:30],
    }


async def main(argv: list[str]) -> list[dict]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", nargs="*", type=int, default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--minutes", type=float, default=5.0)
    parser.add_argument("--visible", action="store_true")
    parser.add_argument("--out", default="")
    parser.add_argument("--trace", default="", help="a folder for each game's log of decisions and its last screen")
    args = parser.parse_args(argv)
    os.environ.setdefault("AURA_BROWSER_STEALTH", "0")
    numbers = sorted(ARCHIVED) if args.all else args.games
    results = []
    for number in numbers:
        if number not in ARCHIVED:
            print(json.dumps({"game": number, "error": "no archived copy that plays here"}), flush=True)
            continue
        result = await _one(number, ARCHIVED[number], args.minutes, args.visible, Path(args.trace) if args.trace else None)
        results.append(result)
        print(json.dumps({k: result[k] for k in ("game", "minutes", "runs", "won", "error", "last_seen")}), flush=True)
    return results


if __name__ == "__main__":
    _results = asyncio.run(main(sys.argv[1:]))
    if "--out" in sys.argv:
        Path(sys.argv[sys.argv.index("--out") + 1]).write_text(json.dumps(_results, indent=1))
