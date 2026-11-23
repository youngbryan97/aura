"""Receipts for the board games she plays against an engine, and which of them a claim may count.

Not part of her. A claim about how she plays is only as good as what it was measured against: which engine, which
build of it, how many threads and how much table it had, whether its strength was held back, and what clock the game
was played on. This tool writes that down for each game she has played, from her own record of the game and from the
engine's settings and binary, and keeps it beside the game.

It also keeps games used to judge a change apart from games used to make one. Before a game is played, its purpose is
declared: "evaluation" (it may be counted for a claim, and nothing is tuned on it) or "tuning". A game's receipt
carries the last purpose declared before it ended; a game with none declared is counted as tuning.

Usage:
    python tools/board_game_receipts.py declare evaluation|tuning [--note TEXT]
    python tools/board_game_receipts.py receipt --engine BINARY [--settings JSON_FILE] [--log LOG]...
    python tools/board_game_receipts.py list

Files (append-only, under the repo's data/board_games/): purposes.jsonl, receipts.jsonl.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
from datetime import UTC, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEPT = ROOT / "data" / "board_games"
PURPOSES = ("evaluation", "tuning")
#: Her record of each game, as her pursuit logs it (core/skills/screen_pursuit_board_of_pieces.py).
PLAYED = "a board of pieces played: "
#: How long an engine is given to name itself and its options.
ASKS_FOR_S = 10.0


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _append(name: str, row: dict) -> None:
    KEPT.mkdir(parents=True, exist_ok=True)
    with open(KEPT / name, "a", encoding="utf-8") as kept:
        kept.write(json.dumps(row, sort_keys=True) + "\n")


def _rows(name: str) -> list[dict]:
    path = KEPT / name
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def declare(purpose: str, note: str) -> dict:
    row = {"declared_at": _now(), "purpose": purpose, "note": note}
    _append("purposes.jsonl", row)
    return row


def games_in(logs: list[Path]) -> list[dict]:
    """Every game her pursuit recorded in these logs (her JSON log, a line an event), with when the record was
    written, oldest first."""
    found = []
    for log in logs:
        if not log.exists():
            continue
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            if PLAYED not in line:
                continue
            try:
                entry = json.loads(line)
                record = ast.literal_eval(str(entry["event"]).split(PLAYED, 1)[1])
            # not a failure: a line cut short in the log is no record of a game.
            except (ValueError, SyntaxError, KeyError, IndexError):
                continue
            if not isinstance(record, dict) or record.get("moves", 0) <= 0:
                continue
            at = datetime.fromisoformat(str(entry["timestamp"])).astimezone(UTC)
            found.append({"ended_at": at.isoformat(timespec="seconds"), "record": record})
    return sorted(found, key=lambda game: game["ended_at"])


def what_the_engine_says(binary: Path) -> dict:
    """The engine's name for itself and its options with their defaults, as it gives them when asked."""
    asked = subprocess.run([str(binary)], input="uci\nquit\n", capture_output=True, text=True, timeout=ASKS_FOR_S)
    name, options = "", {}
    for line in asked.stdout.splitlines():
        if line.startswith("id name "):
            name = line[len("id name "):].strip()
        elif line.startswith("option name "):
            option = re.match(r"option name (.+?) type (\S+)(?: default (\S*))?", line)
            if option:
                options[option.group(1)] = option.group(3)
    return {"name": name, "defaults": options,
            "sha256": hashlib.sha256(binary.read_bytes()).hexdigest()}


def receipt(engine: Path, settings: Path | None, logs: list[Path]) -> dict | None:
    games = games_in(logs)
    if not games:
        return None
    game = games[-1]
    declared = [row for row in _rows("purposes.jsonl") if row["declared_at"] <= game["ended_at"]]
    held = json.loads(settings.read_text(encoding="utf-8")) if settings is not None and settings.exists() else None
    record = game["record"]
    row = {
        "written_at": _now(), "ended_at": game["ended_at"],
        "purpose": declared[-1]["purpose"] if declared else "tuning",
        "declared": declared[-1] if declared else None,
        "engine": {"binary": str(engine), **what_the_engine_says(engine)},
        "settings": held,
        "the_window_when_it_began": record.get("the_window_when_it_began", ""),
        "ended": record.get("ended"), "read_from_the_window": record.get("read_from_the_window", ""),
        "moves": record.get("the_moves", []), "first_was_hers": record.get("first_was_hers"),
    }
    if any(kept["ended_at"] == row["ended_at"] for kept in _rows("receipts.jsonl")):
        return None
    _append("receipts.jsonl", row)
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="what", required=True)
    told = sub.add_parser("declare")
    told.add_argument("purpose", choices=PURPOSES)
    told.add_argument("--note", default="")
    made = sub.add_parser("receipt")
    made.add_argument("--engine", type=Path, required=True)
    made.add_argument("--settings", type=Path)
    made.add_argument("--log", type=Path, action="append",
                      default=None, help="her JSON logs to read games from (default: ~/.aura/logs/aura_json.log*)")
    sub.add_parser("list")
    asked = parser.parse_args(argv)
    if asked.what == "declare":
        print(json.dumps(declare(asked.purpose, asked.note)))
    elif asked.what == "receipt":
        logs = asked.log or sorted((Path.home() / ".aura" / "logs").glob("aura_json.log*"))
        row = receipt(asked.engine, asked.settings, logs)
        print(json.dumps(row, indent=1) if row else "no game without a receipt")
    else:
        for row in _rows("receipts.jsonl"):
            print(row["ended_at"], row["purpose"], row["ended"], len(row["moves"]), row["engine"]["name"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
