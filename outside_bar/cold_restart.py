"""Whether what she is told survives a cold restart, asked through the door a person uses.

    python -m outside_bar.cold_restart --run NAME --facts N

Each run makes up its own facts from its name, about places that exist nowhere else, so nothing she learned in an earlier
run can answer them. She is told most of them, restarted, and then asked for all of them in new conversations. The ones she
was never told are the control: an answer that states a code for one of those is a guess, and counts as wrong.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from outside_bar import run as bar

#: One fact in this many is never told to her.
_NEVER_TOLD_EVERY = 4
#: How long her door may take to come back after a restart, in seconds.
_BOOT_SECONDS = 600
#: Two-second checks allowed for her app to finish quitting.
_CHECKS_WHILE_QUITTING = 90
#: How long her process is given to appear after her app is opened, in seconds.
_SECONDS_FOR_HER_PROCESS_TO_APPEAR = 20
_SYLLABLES = ("var", "nok", "tel", "mir", "osk", "bren", "dal", "quo", "zen", "pash", "lum", "gor", "fen", "rik", "yol")
_THINGS = ("locker code", "gate code", "archive number", "cabinet code")
_A_CODE = re.compile(r"(?<!\d)\d{4}(?!\d)")


def facts_for(name: str, count: int) -> list[dict[str, Any]]:
    """The run's facts, the same every time the run is resumed and different for every other run."""
    made = random.Random(hashlib.sha256(name.encode("utf-8")).hexdigest())
    out = []
    for index in range(count):
        place = "".join(made.choice(_SYLLABLES) for _ in range(3)).capitalize()
        thing, code = made.choice(_THINGS), str(made.randrange(1000, 10000))
        out.append({"id": bar.cold_id(index), "told": index % _NEVER_TOLD_EVERY != _NEVER_TOLD_EVERY - 1, "code": code,
                    "tell": f"Please remember this for later: the {thing} for the {place} depot is {code}.",
                    "ask": f"What is the {thing} for the {place} depot?"})
    return out


def graded(fact: dict[str, Any], reply: str) -> bool:
    """A fact she was told must come back. One she was never told must not be answered with a code."""
    codes = _A_CODE.findall(reply)
    return fact["code"] in codes if fact["told"] else not codes


def _running(pattern: str) -> bool:
    return subprocess.run(["pgrep", "-f", pattern], capture_output=True, check=False).returncode == 0


def restart_her(base: str) -> None:
    """Quit her app, wait for it to stop, open it again, and wait until she can converse.

    Opening an app that is still quitting starts nothing, so it is opened again until her process appears. LIVE 18 November
    2026 one restart left her down for five minutes this way.
    """
    subprocess.run(["osascript", "-e", 'tell application "Aura" to quit'], check=False, capture_output=True)
    for _ in range(_CHECKS_WHILE_QUITTING):
        if not _running("[a]ura_main.py --desktop") and not _running("[A]ura.app/Contents/MacOS"):
            break
        time.sleep(2)
    ends = time.monotonic() + _BOOT_SECONDS
    while time.monotonic() < ends:
        if not _running("[a]ura_main.py"):
            subprocess.run(["open", "-a", "/Applications/Aura.app"], check=True, capture_output=True)
            time.sleep(_SECONDS_FOR_HER_PROCESS_TO_APPEAR)
            continue
        try:
            with urllib.request.urlopen(f"{base}/api/health", timeout=3) as answer:  # noqa: S310 - her own local port
                if json.loads(answer.read().decode("utf-8")).get("conversation_ready") is True:
                    return
        except (OSError, ValueError, urllib.error.URLError):
            pass
        time.sleep(2)
    raise TimeoutError("her door did not come back after the restart")


def measure(name: str, count: int, base: str, restart: Callable[[str], None] = restart_her,
            seconds: float = bar.REPLY_SECONDS) -> int:
    """Tell her the facts, restart her, ask for every one, and write one graded row each."""
    broken = bar.what_broke_the_seal()
    if broken:
        print(f"refused: the seal does not match: {broken[:6]}")
        return 2
    folder = bar.HOME / "runs" / name
    facts = facts_for(name, count)
    told_already = folder / "told.json"
    held_here = not bar.HOLD.exists()
    if held_here:
        bar.HOLD.write_text(f"outside bar cold restart {name}\n", encoding="utf-8")
    try:
        if not told_already.exists():
            for fact in facts:
                if fact["told"]:
                    bar._her_reply(base, fact["tell"], f"outside-bar-{name}-told", seconds)
            restart(base)
            bar._write_once(told_already, {"facts": count, "restarted": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
        for fact in facts:
            row = folder / "rows" / f"{hashlib.sha256(fact['id'].encode()).hexdigest()[:24]}.json"
            if row.exists():
                continue
            started = time.time()
            said = bar._her_reply(base, fact["ask"], f"outside-bar-{name}-{row.stem}", seconds)
            reply = str(said.get("response") or "")
            bar._write_once(row, {"id": fact["id"], "domain": bar.COLD_RESTART, "right": graded(fact, reply), "told": fact["told"],
                                  "read": ",".join(_A_CODE.findall(reply))[:60], "status": str(said.get("status") or ""),
                                  "seconds": round(time.time() - started, 1), "reply": reply})
    finally:
        if held_here:
            bar.HOLD.unlink(missing_ok=True)
    meta = folder / "meta.json"
    if meta.exists():
        known = json.loads(meta.read_text(encoding="utf-8"))
        bar._write_once(meta, {**known, "facts": count})
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--facts", type=int, required=True)
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    args = parser.parse_args(argv)
    return measure(args.run, args.facts, args.base)


if __name__ == "__main__":
    sys.exit(main())
