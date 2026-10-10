"""What she took in while she played, source by source, and what of it reached what she did: read from her own log.

    python tools/what_she_took_in.py --since 13:40 [--until 14:30] [--log ~/.aura/logs/aura_json.log]

Times are local to this machine. The log is split at each thing she picked by a rule (a game of a demo, one of
several things asked for), and for each it says:

- what came in: the screen's words read, her eyes' looks (and how many saw nothing), what she noticed, her guide,
  her reading of the place and what it rested on, each stock she took (asked of, heard, kept), the web's pages,
  what others wrote, and her model's answers and the asks left out for want of time;
- what reached her play: which thing she found was hers and by what, the keys found to move her, what she learned
  to keep clear of, bars and counters she read, the gains and losses of each stretch, and how each run ended.

It reads what the log says and nothing else: a source with nothing here said nothing, or was never asked.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

#: Each thing looked for in the log: its name, the logger it comes from (a part of the name), and the words it opens with.
TAKEN_IN = (
    ("the screen's words", "ScreenPursuit", "saying out loud: 'It says:"),
    ("her eyes looked", "HerEyes", "her eyes took"),
    ("her eyes saw nothing in the place", "WhatIsInAPlace", "what is in this place, to her eyes: nothing"),
    ("her eyes placed things", "WhatIsInAPlace", "what is in this place, to her eyes: ", "nothing"),
    ("her eyes' answer kept nothing", "WhatIsInAPlace", "her eyes answered what is in this place"),
    ("she noticed", "Noticing", "noticing:"),
    ("her guide", "ScreenPursuit", "saying out loud: 'My guide to"),
    ("her reading of the place", "WhatThisPlaceIs", "what this place is ("),
    ("stock taken", "TakingStock", "taking stock ("),
    ("her model left out of stock", "TakingStock", "my model needs about"),
    ("the web read", "LookingItUp", "looked up "),
    ("what others wrote", "WhatOthersWrote", "what others wrote of"),
    ("her model asked (answer clock)", "InferenceGate", "[ANSWER CLOCK]"),
)
USED = (
    ("her thing, by her eyes", "she supposed", "That looks like me:"),
    ("her thing, by her keys", "found", "That's me:"),
    ("what she has left", "bars", "it's what I have left"),
    ("counters said", "counters", "saying while playing: ", "That"),
    ("her plan", "plan", "What I'm after:"),
)


def _local(stamp: str) -> datetime:
    return datetime.fromisoformat(stamp.replace("Z", "+00:00")).astimezone()


def _lines(log: Path, since: datetime, until: datetime | None):
    with open(log, "rb") as f:
        f.seek(max(0, os.path.getsize(log) - 400_000_000))
        for raw in f:
            try:
                d = json.loads(raw)
            except ValueError:
                continue
            stamp = d.get("timestamp")
            if not stamp:
                continue
            at = _local(stamp)
            if at < since or (until is not None and at > until):
                continue
            yield at, str(d.get("logger") or ""), str(d.get("event") or d.get("message") or "")


def taken_in(log: Path, since: datetime, until: datetime | None) -> dict[str, dict]:
    games: dict[str, dict] = {}
    current = "before the first pick"
    for at, logger, event in _lines(log, since, until):
        picked = re.search(r"picked by the rule: .*?-> (.+?) \(https?://", event)
        if picked:
            current = f"{at:%H:%M} {picked.group(1)}"
        game = games.setdefault(current, {"in": Counter(), "said": defaultdict(list), "stretches": [], "ends": []})
        for name, from_logger, opens, *unless in TAKEN_IN:
            if from_logger in logger and opens in event and not any(u in event for u in unless):
                game["in"][name] += 1
                if len(game["said"][name]) < 3:
                    game["said"][name].append(event[:240])
        for name, _kind, opens, *unless in USED:
            if any(u in event for u in unless):
                continue
            if opens in event and "saying out loud" not in event or (opens in event and name == "her plan"):
                game["in"][name] += 1
                if len(game["said"][name]) < 4:
                    game["said"][name].append(event[event.find(opens):][:200])
        if "a stretch played as it happened:" in event:
            game["stretches"].append(event.split(":", 1)[1].strip()[:400])
        if "the run is over:" in event or "Round " in event and ("won" in event or "lost" in event):
            game["ends"].append(event[:160])
    return games


def said(games: dict[str, dict]) -> str:
    out = []
    for game, seen in games.items():
        out.append(f"\n== {game}")
        for name, *_rest in TAKEN_IN:
            if seen["in"][name]:
                out.append(f"  in   {name:36s} {seen['in'][name]:4d}   e.g. {seen['said'][name][0][:150]}")
        missing = [name for name, *_rest in TAKEN_IN if not seen["in"][name]]
        if missing:
            out.append(f"  none from: {', '.join(missing)}")
        for name, *_rest in USED:
            if seen["in"][name]:
                out.append(f"  used {name:36s} {seen['in'][name]:4d}   e.g. {seen['said'][name][0][:150]}")
        for stretch in seen["stretches"][-3:]:
            out.append(f"  play {stretch[:260]}")
        for end in seen["ends"][-3:]:
            out.append(f"  end  {end}")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--since", required=True, help="local time HH:MM, today")
    parser.add_argument("--until", default="", help="local time HH:MM, today")
    parser.add_argument("--log", default=str(Path.home() / ".aura/logs/aura_json.log"))
    args = parser.parse_args()
    today = datetime.now().astimezone()

    def at(hhmm: str) -> datetime:
        hour, minute = (int(v) for v in hhmm.split(":"))
        return today.replace(hour=hour, minute=minute, second=0, microsecond=0)

    games = taken_in(Path(args.log).expanduser(), at(args.since), at(args.until) if args.until else None)
    print(said(games))


if __name__ == "__main__":
    main()
