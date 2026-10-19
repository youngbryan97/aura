#!/usr/bin/env python3
"""How well she builds a program asked for by name, measured on programs fixed before any of it was built.

The programs are well known, of many kinds, and each can live in a browser page. They were written down on 19
October 2026, before the engine's libraries had built anything, and split with seed 20261019 into a development set,
which the engine may be worked on against, and a held-out set, whose builds are only ever counted
(docs/design-docs/BUILDING_ANY_PROGRAM.md). Each build is her model's, through the engine as it runs in her chat
(core/rebuilding/rebuilding_a_program.py), in one of two arms: with the document code we wrote, or hers alone
(AURA_REBUILD_WITH_OUR_PARTS).

For each program: the features her model listed, how many work by their checks, the libraries planned and how many
of their tests held, how many came from her store, the minutes, and where it was kept.

Seen before the split: Microsoft Paint and Microsoft Excel were read with core/rebuilding/what_is_known_of_it.py
while it was written, and the reading was mended by what it found for them. Paint fell in the held-out set; held-out
results are given with it and without it.

Usage:
    measure_rebuilding.py --split                       # print the two sets
    measure_rebuilding.py --set development --arm hers --out DIR [--only NAME ...] [--minutes N]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SEED = 20261019
HELD_OUT = 24
#: Read while the engine's reading was written, before the split (see above).
SEEN_BEFORE_THE_SPLIT = ("Microsoft Paint", "Microsoft Excel")
#: Written 19 October 2026 before anything was built: programs a person knows by name, each of which can be a page.
PROGRAMS = (
    "Microsoft Paint", "Notepad", "WordPad", "Microsoft Excel", "VisiCalc", "Lotus 1-2-3", "Windows Calculator",
    "Microsoft Minesweeper", "Microsoft Solitaire", "FreeCell", "Tetris", "Snake (video game genre)", "Pong", "2048 (video game)",
    "Sudoku", "Conway's Game of Life", "Fractint", "Winamp", "Microsoft PowerPoint", "Microsoft Visio", "draw.io",
    "XMind", "Trello", "Todoist", "Sticky Notes", "Anki (software)", "Desmos", "Mavis Beacon Teaches Typing",
    "Aseprite", "Inkscape", "MuseScore", "GarageBand", "Audacity", "Typora", "Notion (productivity software)",
    "Microsoft Outlook", "Apple Calendar", "Google Keep", "KeePass", "Zork", "Wordle", "Mahjong solitaire",
    "Breakout (video game)", "Space Invaders", "Asteroids (video game)", "Lights Out (game)", "15 puzzle",
    "Tic-tac-toe", "Connect Four", "Reversi", "Checkers", "Battleship (game)", "Hangman (game)", "Etch A Sketch",
    "LibreOffice Calc", "Apple Numbers", "Microsoft Word", "TextEdit", "Gnumeric", "Kanban board",
)


def split(programs: tuple[str, ...] = PROGRAMS, seed: int = SEED) -> dict[str, list[str]]:
    """The development set and the held-out set, by seed."""
    drawn = sorted(programs)
    random.Random(seed).shuffle(drawn)
    return {"held_out": sorted(drawn[:HELD_OUT]), "development": sorted(drawn[HELD_OUT:])}


def measured(rebuilt: Any, seconds: float, folder: Path | None) -> dict[str, Any]:
    """What one build came to."""
    built = getattr(rebuilt, "built", None)
    outcomes = list(getattr(built, "outcomes", []) or [])
    libraries = []
    if folder is not None and (folder / "libraries.json").exists():
        libraries = json.loads((folder / "libraries.json").read_text("utf-8"))
    return {
        "program": rebuilt.program, "minutes": round(seconds / 60, 1), "folder": str(folder or ""),
        "features": len(outcomes), "working": sum(1 for o in outcomes if o.kept),
        "libraries": [{"name": lib["plan"]["name"], "tests": len(lib["plan"]["tests"]), "held": len(lib["held"]),
                       "from_her_store": lib["from_her_store"]} for lib in libraries],
        "why_not": getattr(rebuilt, "why_not", "") or "",
    }


async def _one(program: str, where: Path, minutes: float) -> dict[str, Any]:
    from core.rebuilding.her_model import ask_her_model
    from core.rebuilding.rebuilding_a_program import rebuild

    began = time.monotonic()
    rebuilt = await rebuild(program, ask_her_model, where, deadline_s=minutes * 60)
    folder = getattr(getattr(rebuilt, "built", None), "path", None)
    return measured(rebuilt, time.monotonic() - began, Path(folder).parent if folder else None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", action="store_true")
    parser.add_argument("--set", choices=("development", "held_out"))
    parser.add_argument("--arm", choices=("ours", "hers"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--only", nargs="*", default=[])
    parser.add_argument("--minutes", type=float, default=60.0, help="the most one build may take")
    args = parser.parse_args()
    sets = split()
    if args.split or not args.set:
        print(json.dumps(sets, indent=1))
        return 0
    os.environ["AURA_REBUILD_WITH_OUR_PARTS"] = "1" if args.arm == "ours" else "0"
    out = args.out / args.arm
    out.mkdir(parents=True, exist_ok=True)
    for program in args.only or sets[args.set]:
        record = out / f"{program.replace('/', '-')}.json"
        if record.exists():
            continue
        row = asyncio.run(_one(program, out / "builds", args.minutes))
        record.write_text(json.dumps({**row, "arm": args.arm, "set": args.set}, indent=1))
        print(json.dumps(row), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
