"""Ask her the frozen problems through the door a person uses, grade the replies, and compare two runs.

    python -m outside_bar.run check
    python -m outside_bar.run ask --run NAME --each N [--part panel|pilot]
    python -m outside_bar.run score --run NAME
    python -m outside_bar.run baseline --run NAME
    python -m outside_bar.run compare --baseline NAME --candidate NAME
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from outside_bar import planbench
from outside_bar.grade import BENCH, grade
from outside_bar.panel import DATA, DOMAINS, HERE, PANEL, tasks

HOME = Path(os.environ.get("AURA_OUTSIDE_BAR_HOME") or "~/.aura/outside-bar").expanduser()
SEAL = HERE / "SEAL.json"
#: Her own landing waits while this file exists (core/agency/her_goal_at_work.py).
HOLD = Path(os.environ.get("AURA_OUTSIDE_BAR_HOLD") or "~/.aura/measuring.hold").expanduser()
#: How long one reply may take, in seconds. Past this the problem counts as unanswered.
REPLY_SECONDS = 1800
#: How long to wait between asks of the delivery route, in seconds: the Retry-After her route sends.
DELIVERY_POLL_SECONDS = 1.0
#: The domain outside_bar/cold_restart.py adds to a run, and every domain two runs are compared on.
COLD_RESTART = "cold_restart"
JUDGED = (*DOMAINS, COLD_RESTART)
#: The chance of a loss this lopsided by luck that counts as a regression, the convention G09 was judged by.
ALPHA = 0.05


def cold_id(index: int) -> str:
    return f"{COLD_RESTART}:{index}"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sealed_files() -> dict[str, Path]:
    """Every file the result depends on: this harness, the benchmarks' data and MATH-500's grader."""
    files = {f"outside_bar/{path.name}": path for path in sorted(HERE.iterdir())
             if path.is_file() and path.suffix in (".py", ".json", ".md") and path.name != SEAL.name}
    files.update({f"data/{name}": path for name, path in DATA.items()})
    frozen = json.loads(PANEL.read_text(encoding="utf-8"))["domains"]
    for domain in ("blocksworld", "mystery_blocksworld"):
        setting = planbench.config(domain)
        files[f"data/{domain}.yaml"] = planbench.ROOT / "configs" / f"{domain}.yaml"
        files[f"data/{domain}.pddl"] = planbench.ROOT / "instances" / setting["domain_file"]
        for task_id in [*frozen[domain]["panel"], *frozen[domain]["pilot"]]:
            number = int(task_id.rsplit(":", 1)[1])
            files[f"data/{domain}/{number}"] = (planbench.ROOT / "instances" / setting["instance_dir"]
                                                / setting["instances_template"].format(number))
    files.update({f"grader/{name}": BENCH / "math500/grading" / name for name in ("grader.py", "math_normalize.py")})
    return files


def what_broke_the_seal() -> list[str]:
    """The files that differ from the seal; empty when every one matches."""
    held = json.loads(SEAL.read_text(encoding="utf-8"))["files"]
    now = {name: _digest(path) for name, path in sealed_files().items() if path.exists()}
    return sorted(name for name in {*held, *now} if held.get(name) != now.get(name))


def seal_id() -> str:
    return _digest(SEAL)


def _write_once(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def append_to_ledger(entry: dict[str, Any]) -> dict[str, Any]:
    """One line, carrying the hash of the line before it, so an edit to an earlier line shows."""
    ledger = HOME / "ledger.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    lines = ledger.read_text(encoding="utf-8").splitlines() if ledger.exists() else []
    previous = hashlib.sha256(lines[-1].encode("utf-8")).hexdigest() if lines else ""
    entry = {**entry, "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "seal": seal_id(), "previous": previous}
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True, ensure_ascii=False) + "\n")
    return entry


def the_ledger_holds() -> bool:
    ledger = HOME / "ledger.jsonl"
    lines = ledger.read_text(encoding="utf-8").splitlines() if ledger.exists() else []
    return all(json.loads(line)["previous"] == hashlib.sha256(lines[index - 1].encode("utf-8")).hexdigest()
               for index, line in enumerate(lines) if index) and (not lines or json.loads(lines[0])["previous"] == "")


def _opened(request: urllib.request.Request, seconds: float) -> tuple[int, dict[str, Any]]:
    """The status and JSON body of one request; a 4xx or 5xx carries her answer as much as a 200 does."""
    try:
        with urllib.request.urlopen(request, timeout=seconds) as answer:  # noqa: S310 - her own local port
            return answer.status, json.loads(answer.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as refused:
        try:
            return refused.code, json.loads(refused.read().decode("utf-8") or "{}")
        except ValueError:
            return refused.code, {}


def _her_reply(base: str, message: str, session: str, seconds: float) -> dict[str, Any]:
    """Her finished reply. A turn still running answers 202, and her window then asks the delivery route until it ends."""
    ends = time.monotonic() + seconds
    status, said = _opened(urllib.request.Request(
        f"{base}/api/chat", data=json.dumps({"message": message, "session_id": session}).encode(),
        headers={"Content-Type": "application/json"}), seconds)
    key = str(said.get("idempotency_key") or "")
    while status == 202 and key:
        if time.monotonic() >= ends:
            raise TimeoutError
        time.sleep(DELIVERY_POLL_SECONDS)
        query = urllib.parse.urlencode({"session_id": session})
        status, polled = _opened(urllib.request.Request(f"{base}/api/chat/delivery/{urllib.parse.quote(key)}?{query}"),
                                 max(1.0, ends - time.monotonic()))
        said = polled.get("result") if isinstance(polled.get("result"), dict) else polled
    return said


def source_commit(root: Path) -> str:
    done = subprocess.run(["git", "-C", str(root), "-c", "core.fsmonitor=false", "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=False)
    return done.stdout.strip()


def ask(name: str, each: int, part: str, base: str, source: Path, seconds: float = REPLY_SECONDS) -> int:
    broken = what_broke_the_seal()
    if broken:
        print(f"refused: the seal does not match: {broken[:6]}")
        return 2
    folder = HOME / "runs" / name
    meta = folder / "meta.json"
    if not meta.exists():
        _write_once(meta, {"name": name, "each": each, "part": part, "seal": seal_id(), "commit": source_commit(source),
                           "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    held_here = not HOLD.exists()
    if held_here:
        HOLD.write_text(f"outside bar run {name}\n", encoding="utf-8")
    try:
        for task in tasks(each, part):
            row = folder / "rows" / f"{hashlib.sha256(task['id'].encode()).hexdigest()[:24]}.json"
            if row.exists():
                continue
            started = time.time()
            try:
                said = _her_reply(base, task["request"], f"outside-bar-{name}-{row.stem}", seconds)
                reply, status = str(said.get("response") or ""), str(said.get("status") or "")
            except TimeoutError:
                reply, status = "", "no reply in time"
            except urllib.error.URLError as exc:
                if isinstance(exc.reason, TimeoutError):
                    reply, status = "", "no reply in time"
                else:
                    print(f"stopped: her door did not answer: {exc}")
                    return 3
            right, read = grade(task["domain"], reply, task["truth"]) if reply else (False, "no reply")
            _write_once(row, {"id": task["id"], "domain": task["domain"], "right": right, "read": read, "status": status,
                              "seconds": round(time.time() - started, 1), "reply": reply})
            print(f"{task['id']} right={right} {time.time() - started:.0f}s", flush=True)
    finally:
        if held_here:
            HOLD.unlink(missing_ok=True)
    append_to_ledger({"kind": "run", "run": name, "score": score(name), "commit": json.loads(meta.read_text())["commit"]})
    return 0


def rows_of(name: str) -> dict[str, dict[str, Any]]:
    return {row["id"]: row for row in (json.loads(path.read_text(encoding="utf-8"))
                                       for path in sorted((HOME / "runs" / name / "rows").glob("*.json")))}


def score(name: str) -> dict[str, dict[str, int]]:
    rows = rows_of(name).values()
    return {domain: {"right": sum(r["right"] for r in rows if r["domain"] == domain),
                     "of": sum(r["domain"] == domain for r in rows)} for domain in JUDGED}


def chance_of(losses: int, gains: int) -> float:
    """The exact chance of at least this many losses among the problems that changed, were each change a coin toss."""
    changed = losses + gains
    return sum(math.comb(changed, k) for k in range(losses, changed + 1)) / 2 ** changed if changed else 1.0


def compared(baseline: dict[str, dict[str, Any]], candidate: dict[str, dict[str, Any]],
             domains: Sequence[str] = JUDGED) -> dict[str, Any]:
    """HELD, REGRESSED or INCOMPLETE. A domain is judged at ALPHA shared across the domains, so four chances to lose
    do not add up to one false alarm in five."""
    if set(baseline) != set(candidate) or not baseline:
        return {"verdict": "INCOMPLETE", "missing": sorted(set(baseline) ^ set(candidate))[:20]}
    each: dict[str, Any] = {}
    for domain in domains:
        shared = [key for key, row in baseline.items() if row["domain"] == domain]
        losses = sum(baseline[key]["right"] and not candidate[key]["right"] for key in shared)
        gains = sum(candidate[key]["right"] and not baseline[key]["right"] for key in shared)
        each[domain] = {"losses": losses, "gains": gains, "chance": chance_of(losses, gains),
                        "regressed": chance_of(losses, gains) < ALPHA / len(domains)}
    losses, gains = sum(d["losses"] for d in each.values()), sum(d["gains"] for d in each.values())
    regressed = chance_of(losses, gains) < ALPHA or any(d["regressed"] for d in each.values())
    return {"verdict": "REGRESSED" if regressed else "HELD", "losses": losses, "gains": gains,
            "chance": chance_of(losses, gains), "domains": each}


def compare(baseline: str, candidate: str) -> int:
    seals = {json.loads((HOME / "runs" / name / "meta.json").read_text())["seal"] for name in (baseline, candidate)}
    if len(seals) != 1:
        print("refused: the two runs were not made under one seal")
        return 2
    result = compared(rows_of(baseline), rows_of(candidate))
    entry = append_to_ledger({"kind": "compare", "baseline": baseline, "candidate": candidate, **result})
    print(json.dumps(entry, indent=1, ensure_ascii=False))
    return {"HELD": 0, "REGRESSED": 1}.get(result["verdict"], 2)


def baseline(name: str) -> int:
    """Name a finished run as what later runs are compared with."""
    meta = json.loads((HOME / "runs" / name / "meta.json").read_text(encoding="utf-8"))
    wanted = {task["id"] for task in tasks(meta["each"], meta["part"])} | {cold_id(i) for i in range(int(meta.get("facts") or 0))}
    if meta["part"] != "panel" or not meta.get("facts") or meta["seal"] != seal_id() or wanted != set(rows_of(name)):
        print("refused: a baseline is a whole panel run with its cold restart, made under this seal")
        return 2
    print(json.dumps(append_to_ledger({"kind": "baseline", "run": name, "commit": meta["commit"], "score": score(name)}),
                     indent=1))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check")
    asking = commands.add_parser("ask")
    asking.add_argument("--run", required=True)
    asking.add_argument("--each", type=int, required=True)
    asking.add_argument("--part", choices=("panel", "pilot"), default="panel")
    asking.add_argument("--base", default="http://127.0.0.1:8000")
    asking.add_argument("--source", type=Path, default=Path("~/.aura/live-source").expanduser())
    commands.add_parser("score").add_argument("--run", required=True)
    commands.add_parser("baseline").add_argument("--run", required=True)
    comparing = commands.add_parser("compare")
    comparing.add_argument("--baseline", required=True)
    comparing.add_argument("--candidate", required=True)
    args = parser.parse_args(argv)
    if args.command == "check":
        broken = what_broke_the_seal()
        print(json.dumps({"seal": seal_id(), "broken": broken, "ledger_holds": the_ledger_holds()}, indent=1))
        return 1 if broken or not the_ledger_holds() else 0
    if args.command == "ask":
        return ask(args.run, args.each, args.part, args.base, args.source)
    if args.command == "score":
        print(json.dumps(score(args.run), indent=1))
        return 0
    if args.command == "baseline":
        return baseline(args.run)
    return compare(args.baseline, args.candidate)


if __name__ == "__main__":
    sys.exit(main())
