#!/usr/bin/env python3
"""How she does on real Flash games, played from their Internet Archive copies, offline.

The Cartoon Network list she is asked to pick from is a museum that refuses its
game files to automated browsers; each game's page links its Internet Archive
copy, which plays in any browser. This plays those copies the way the live demo
does: her own browser with nothing hiding that it is automated, the page's
drawing handed to her screen pursuit and its reflexes, asked to win. What is
offline is only her language: with no model loaded, the moments the screen
pursuit would think in words are decided without them.

    python tools/measure_on_archived_games.py --games 16 2 --minutes 6 --out pass.json
    python tools/measure_on_archived_games.py --all --minutes 4 --out report.json
    python tools/measure_on_archived_games.py --games 6 --minutes 5 --trace traces --out pass.json
    python tools/measure_on_archived_games.py --bundle ~/.aura/flash-campaign/offline-bundle --all --trace traces --out pass.json
    python tools/measure_on_archived_games.py --bundle ~/.aura/flash-campaign/offline-bundle --random 3 --seed 42 --out pass.json

With ``--bundle`` the games are the offline bundle's (tools/prepare_offline_flash_bundle.py): each original program,
kept with its hash, played by the pinned emulator from a server on this machine, so no game's copy can change or go
missing between runs. With ``--trace`` her play is also pictured every few seconds, to be watched afterwards.

Each game's report says how many runs she played, whether she won, how each
ended in the game's own words, and what she said while playing.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

#: The list's own order (0-55) and each game's Internet Archive copy, read off
#: the museum's game pages on 4 October 2026.
ARCHIVED = {
    0: "https://archive.org/details/scooby-snapshot_flash",
    1: "https://archive.org/details/the-riddlers-secret-identity-inventor_flash",
    2: "https://archive.org/details/food_bash",
    3: "https://archive.org/details/runaway-robot-dexter",
    4: "https://archive.org/details/eene_spinstadium",
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


#: What she is asked of each game: the demo's own words for a thing that may be won, scored or made.
GOAL = ("Play this game. If it can be won, keep playing it until you win. If it only keeps a score, play it three times "
        "and tell me your best. If it has neither, like a character maker, do it once and tell me what you made and why.")
#: How often her play is pictured for watching, in seconds.
PICTURED_EVERY_S = 20.0


async def _pictured(browser: object, trace: Path, number: int, began: float) -> None:
    """Her screen, every few seconds while she plays, for watching afterwards."""
    while True:
        await asyncio.sleep(PICTURED_EVERY_S)
        page = getattr(browser, "page", None)
        if page is None:
            continue
        try:
            await page.screenshot(path=str(trace / f"game{number}_{int(time.monotonic() - began):04d}s.png"))
        except Exception:  # noqa: BLE001 - a picture missed is one fewer to watch
            continue


def _served(bundle: Path, case_ids: set[int] | None = None) -> dict[int, str]:
    """The offline bundle served on a loopback port for the rest of this run; where each of its games is played."""
    import functools
    import threading
    from http.server import ThreadingHTTPServer

    from tools.serve_offline_flash_bundle import BundleHandler, check_bundle

    directory = bundle.expanduser().resolve()
    check_bundle(directory, case_ids=case_ids)
    server = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(BundleHandler, directory=str(directory)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    origin = f"http://127.0.0.1:{server.server_port}"
    os.environ["AURA_BROWSER_ALLOW_LOOPBACK"] = "1"
    os.environ["AURA_BROWSER_LOOPBACK_ORIGINS"] = origin
    os.environ["AURA_WEB_FETCH_ALLOW_HTTP"] = "1"
    manifest = json.loads((directory / "manifest.json").read_text())
    return {case["id"]: f"{origin}/{case['url']}" for case in manifest["cases"]
            if case.get("status") == "asset_verified" and (case_ids is None or case['id'] in case_ids)}


async def _one(number: int, address: str, minutes: float, visible: bool, trace: Path | None = None,
               goal: str = GOAL, *, graphics_backend: str = 'host') -> dict:
    from core.capabilities.phantom_browser import PhantomBrowser
    from core.skills import sovereign_browser_drawing as drawing

    said: list[str] = []
    import core.skills.screen_pursuit as pursuit

    original_tell = pursuit._tell

    def tell(line: str) -> None:
        said.append(" ".join(str(line).split())[:200])
        original_tell(line)

    pursuit._tell = tell
    previous_budget, previous_share = drawing.PLAY_UNTIL_WON_S, drawing.LEAST_SHARE_S
    drawing.PLAY_UNTIL_WON_S = minutes * 60.0
    drawing.LEAST_SHARE_S = min(previous_share, minutes * 60.0)
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
    browser = PhantomBrowser(visible=visible, browser_type="chromium", principal="owner", graphics_backend=graphics_backend)
    graphics = None
    try:
        if not await browser.ensure_ready():
            return {"game": number, "error": "browser did not start"}
        graphics = browser.get_status().get('graphics')
        await browser.page.goto(address, wait_until="domcontentloaded", timeout=60000)
        await asyncio.sleep(6)
        watching = asyncio.ensure_future(_pictured(browser, trace, number, began)) if trace is not None else None
        try:
            played = await drawing.played_on_the_drawing(browser, goal, {"url": address})
        finally:
            if watching is not None:
                watching.cancel()
    except Exception as why:  # noqa: BLE001 - one game failing is a result, not a stop
        played = {"error": f"{type(why).__name__}: {why}"}
    finally:
        pursuit._tell = original_tell
        drawing.PLAY_UNTIL_WON_S, drawing.LEAST_SHARE_S = previous_budget, previous_share
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
        "said": said[:200],
        "stopped_because": str(played.get("stopped_because") or ""),
        "graphics_backend_requested": graphics_backend,
        "graphics": graphics,
    }


#: What may be asked of her model over the wire, and the arguments that cannot be sent (they hold something of the
#: asker's own, not words).
_ASKED_OVER_THE_WIRE = ("generate_with_metadata", "generate", "think")
_NOT_SENT = ("_generation_metadata_sink", "_gate_already_held", "on_token", "callback", "stream_callback")


class HerModelOverTheWire:
    """Her router, as every part of her asks it, answered by her model held in another process (tools/her_model_server.py):
    each game process holding its own copy of her model is more than the machine holds, and without one, everything her
    model reads live was read here by her word patterns alone."""

    def __init__(self, address: str) -> None:
        self.address = address.rstrip("/")

    async def _ask(self, how: str, args: tuple, kwargs: dict) -> Any:
        import aiohttp

        sent = {key: value for key, value in kwargs.items() if key not in _NOT_SENT and not callable(value)}
        timeout = float(kwargs.get("timeout") or (args[2] if len(args) > 2 and isinstance(args[2], (int, float)) else 900))
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout + 60)) as session:
                async with session.post(f"{self.address}/ask", json={"how": how, "args": list(args), "kwargs": sent}) as reply:
                    body = await reply.json()
        except (aiohttp.ClientError, TimeoutError, OSError) as why:
            raise RuntimeError(f"her model could not be reached: {why}") from why
        if "error" in body:
            kind = {"TimeoutError": TimeoutError, "ValueError": ValueError, "TypeError": TypeError}.get(body.get("kind"), RuntimeError)
            raise kind(body["error"])
        return body.get("said")

    async def generate_with_metadata(self, *args: Any, **kwargs: Any) -> Any:
        return await self._ask("generate_with_metadata", args, kwargs)

    async def generate(self, *args: Any, **kwargs: Any) -> Any:
        return await self._ask("generate", args, kwargs)

    async def think(self, *args: Any, **kwargs: Any) -> Any:
        return await self._ask("think", args, kwargs)


def _her_model_from(address: str) -> None:
    from core.container import ServiceContainer

    ServiceContainer.register_instance("llm_router", HerModelOverTheWire(address))


def _arguments(argv: list[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", nargs="*", type=int, default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--minutes", type=float, default=5.0)
    parser.add_argument("--visible", action="store_true")
    parser.add_argument("--out", default="")
    parser.add_argument("--trace", default="", help="a folder for each game's log of decisions and its last screen")
    parser.add_argument("--bundle", default="", help="the offline bundle to play the games from, instead of their archived copies")
    parser.add_argument("--her-model", default="", help="where her model answers (tools/her_model_server.py), as live")
    parser.add_argument("--random", type=int, default=0, help="this many games drawn at random, as the demo draws them")
    parser.add_argument("--seed", type=int, default=None, help="what draws them, to draw the same ones again")
    parser.add_argument("--exclude", nargs="*", type=int, default=[], help="case identities this run must leave alone")
    parser.add_argument("--held-out", type=Path, default=Path('~/.aura/flash-campaign/held_out.json').expanduser())
    parser.add_argument("--evaluate-held-out", action="store_true", help="explicitly open held-out cases for evaluation")
    parser.add_argument("--resume", action="store_true", help="continue a settled campaign checkpoint at --out.state.json")
    parser.add_argument("--plan-only", action="store_true", help="save the case plan without reading programs or starting play")
    parser.add_argument("--graphics", choices=['host', 'software'], default='host', help="software requires a measured CPU driver")
    parser.add_argument("--vision-cpu", action='store_true', help='require CPU acknowledgement before Apple Vision OCR')
    parser.add_argument("--cpu-only", action='store_true', help='software graphics, CPU OCR, classical points and no model workers')
    args = parser.parse_args(argv)
    if args.cpu_only:
        if args.her_model:
            parser.error('--cpu-only cannot call a model endpoint')
        args.graphics, args.vision_cpu = 'software', True
        os.environ.update(AURA_MODEL_WORKERS_ALLOWED='0', AURA_VISION_CPU_ONLY='1',
                          AURA_PLAY_POINT_BACKEND='classical', AURA_PLAY_POINT_DEVICE='cpu')
    if not math.isfinite(args.minutes) or not 0 < args.minutes <= 60 or args.random < 0:
        parser.error('minutes must be finite, positive and at most sixty; random must be nonnegative')
    if not args.out:
        parser.error('--out is required to retain attempted and pending cases')
    if args.resume and args.random and args.seed is None:
        parser.error('a random campaign needs its recorded --seed to resume')
    args.out = Path(args.out).expanduser()
    args.trace = Path(args.trace).expanduser() if args.trace else None
    return args, parser, args.out.with_name(args.out.name + '.state.json')


async def main(argv: list[str]) -> list[dict]:
    args, parser, checkpoint = _arguments(argv)
    from tools.play_campaign_checkpoint import campaign_lease

    async with campaign_lease(checkpoint):
        return await _campaign(args, parser, checkpoint)


def _prepare_campaign(args, parser, checkpoint: Path):
    if args.vision_cpu:
        os.environ['AURA_VISION_CPU_ONLY'] = '1'
    if checkpoint.exists() != args.resume:
        parser.error('use a fresh --out or --resume with its existing checkpoint')
    if args.out.exists() and not args.resume:
        parser.error('the output already exists; use a fresh --out')
    excluded = set(args.exclude)
    if args.held_out.exists() and not args.evaluate_held_out:
        excluded.update(json.loads(args.held_out.read_text())['held_out'])
    os.environ.setdefault("AURA_BROWSER_STEALTH", "0")
    # What her live process refuses to load is refused here too: on 11 October her seeing of a well asked for OpenCV,
    # every run of this tool passed, and live it ended every game she opened.
    from core.media.safe_imports import install_main_process_cv2_guard

    install_main_process_cv2_guard()
    places = dict(ARCHIVED)
    manifest_bytes = b''
    if args.bundle:
        manifest_bytes = (Path(args.bundle).expanduser() / 'manifest.json').read_bytes()
        places = {c['id']: c.get('url', '') for c in json.loads(manifest_bytes)['cases']}
    numbers = sorted(places) if args.all else args.games
    if args.random:
        import random

        seed = args.seed if args.seed is not None else int(time.time())
        available = sorted(set(places) - excluded)
        numbers = sorted(random.Random(seed).sample(available, min(args.random, len(available))))
        print(json.dumps({"drawn": numbers, "seed": seed}), flush=True)
    from tools.play_campaign_checkpoint import campaign_plan, resume_campaign

    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    changed = subprocess.check_output(['git', 'diff', 'HEAD', '--', 'core', 'tools'], cwd=ROOT)
    identity = {'commit': revision, 'tracked_diff_sha256': hashlib.sha256(changed).hexdigest(),
                'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(), 'model_endpoint': args.her_model,
                'visible': args.visible, 'trace': str(args.trace.resolve()) if args.trace else '',
                'graphics': args.graphics, 'vision_cpu': args.vision_cpu, 'cpu_only': args.cpu_only}
    record = campaign_plan(numbers, excluded=excluded, identity=identity, seconds=args.minutes * 60)
    if args.resume:
        record = resume_campaign(json.loads(checkpoint.read_text()), record)
    return record, places


async def _campaign(args, parser, checkpoint: Path) -> list[dict]:
    from tools.play_campaign_checkpoint import run_campaign

    record, places = await asyncio.to_thread(_prepare_campaign, args, parser, checkpoint)

    async def save(value):
        from core.governance_context import local_internal_governed_scope
        from core.runtime.file_write_gateway import get_file_write_gateway

        with local_internal_governed_scope('play_campaign_checkpoint'):
            gateway = get_file_write_gateway()
            await gateway.ensure_directory_async(checkpoint.parent, source='play_campaign_checkpoint')
            await gateway.write_text_async(checkpoint, json.dumps(value, indent=2), source='play_campaign_checkpoint')
            results = [c['result'] for c in value['cases'] if c['state'] == 'recorded']
            await gateway.write_text_async(args.out, json.dumps(results, indent=2),
                                           source='play_campaign_checkpoint')

    await save(record)
    if args.plan_only:
        return [c['result'] for c in record['cases'] if c['state'] == 'recorded']
    active = {c['case'] for c in record['cases'] if c['state'] == 'pending'}
    if args.bundle and active:
        places = await asyncio.to_thread(_served, Path(args.bundle), active)
    if args.her_model and active:
        _her_model_from(args.her_model)

    async def play(number):
        if number not in places:
            return {'game': number, 'error': 'no copy that plays here'}
        result = await _one(number, places[number], args.minutes, args.visible, args.trace, graphics_backend=args.graphics)
        print(json.dumps({k: result.get(k) for k in ('game', 'minutes', 'runs', 'won', 'error', 'last_seen')}), flush=True)
        return result

    return await run_campaign(record, play=play, save=save)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
