#!/usr/bin/env python3
"""Put the broken Pong back where the demo asks her to mend it, as it was before any repair.

    python tools/reset_pong_demo.py            # ~/aura-demos/pong/pong.html
    python tools/reset_pong_demo.py --to DIR

The copy is the fixture in tests/fixtures/pong_repair/pong.html, five flaws and
all. A repair leaves pong.html mended and pong.html.before-repair beside it;
this removes both and puts the broken one back. The request to type in her
chat is printed at the end.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BROKEN = ROOT / "tests" / "fixtures" / "pong_repair" / "pong.html"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--to", default=str(Path.home() / "aura-demos" / "pong"))
    args = parser.parse_args(argv)
    folder = Path(args.to).expanduser()
    folder.mkdir(parents=True, exist_ok=True)
    for left in (folder / "pong.html.before-repair",):
        left.unlink(missing_ok=True)
    shutil.copyfile(BROKEN, folder / "pong.html")
    print(f"Broken Pong put back at {folder / 'pong.html'}")
    print("Type in her chat:")
    print(f"  The Pong game at {folder / 'pong.html'} is broken. Fix it, then play it against the computer until you win.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
