#!/usr/bin/env python3
"""tools/longevity/run_gauntlet.py

Longevity gauntlet runner — drive Aura for 24h / 72h / 7d / 30d windows
and emit a public artifact bundle for each. The runner is *resumable*:
each run is keyed by a UUID and produces a per-tick row in a JSONL trace
file under ``~/.aura/data/longevity/<run_id>/``.

Profiles
--------
24h_no_user       — pure idle, no user input
72h_mixed         — scripted user pulses + idle gaps
7d_with_failures  — adds chaos: subprocess kill, network drop, model load
                    failure, memory pressure
30d_summary       — long-window run with daily continuity-hash snapshots

Artifacts produced per run
--------------------------
  events.jsonl                — every tick's snapshot
  receipts.jsonl              — durable copy of action receipts
  resource.csv                — cpu/ram/disk over time
  goals_outcome.csv           — goals started / completed / abandoned
  identity_continuity.jsonl   — periodic continuity-hash captures
  summary.md                  — human-readable run summary

Usage:
    python tools/longevity/run_gauntlet.py --profile 24h_no_user
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.runtime.resource_observation import (  # noqa: E402
    ResourceObserver,
    get_resource_observer,
)

logger = logging.getLogger("Aura.Longevity")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


_PROFILES: dict[str, dict[str, Any]] = {
    "24h_no_user": {"duration_s": 24 * 3600, "user_pulse_s": 0, "chaos": False},
    "72h_mixed": {"duration_s": 72 * 3600, "user_pulse_s": 1800, "chaos": False},
    "7d_with_failures": {"duration_s": 7 * 24 * 3600, "user_pulse_s": 3600, "chaos": True},
    "30d_summary": {"duration_s": 30 * 24 * 3600, "user_pulse_s": 0, "chaos": False, "snapshot_only": True},
}

_RECOVERABLE_SNAPSHOT_ERRORS = (
    ImportError,
    AttributeError,
    RuntimeError,
    TypeError,
    ValueError,
    OSError,
)


def _append_text(path: Path, line: str) -> None:
    from tools.longevity.soak_output import append_line

    append_line(path, line)


def _append_resource_row(path: Path, row: list[Any]) -> None:
    from tools.longevity.soak_output import append_row

    append_row(path, row)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    rows.append(json.loads(line))
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue
    except FileNotFoundError:
        return []
    return rows


def _write_summary(run_dir: Path, *, run_id: str, profile_name: str, duration_s: float, rows: list[dict[str, Any]]) -> None:
    with (run_dir / "summary.md").open("w", encoding="utf-8") as fh:
        fh.write(f"# longevity run {run_id}\n\n")
        fh.write(f"profile: `{profile_name}`\n")
        fh.write(f"duration: {duration_s}s\n")
        fh.write(f"snapshots: {len(rows)}\n")
        if rows:
            cpu = [float(r.get("cpu_pct") or 0.0) for r in rows]
            ram = [float(r.get("ram_pct") or 0.0) for r in rows]
            fh.write(f"cpu: min={min(cpu):.1f} max={max(cpu):.1f}\n")
            fh.write(f"ram: min={min(ram):.1f} max={max(ram):.1f}\n")
            unique_hashes = sorted({r.get("continuity_hash", "?") for r in rows})
            fh.write(f"unique continuity hashes: {len(unique_hashes)}\n")

        pulses = _read_jsonl(run_dir / "user_pulse.jsonl")
        fh.write("\n## User pulses\n\n")
        if not pulses:
            fh.write("none scheduled.\n")
        elif all(p.get("kind") == "undriven_pulse" for p in pulses):
            fh.write(
                f"{len(pulses)} UNDRIVEN pulses — no --chat-url was given, so "
                "nothing was asked of the runtime. This run says nothing about "
                "behaviour under load and must not be cited as if it did.\n"
            )
        else:
            answered = [p for p in pulses if p.get("ok")]
            correct = [p for p in pulses if p.get("correct")]
            latencies = sorted(
                float(p.get("latency_s") or 0.0) for p in answered
            )
            fh.write(f"pulses: {len(pulses)}\n")
            fh.write(f"answered: {len(answered)}\n")
            # Correctness over ATTEMPTS, not over answers. A run where most
            # pulses never got a reply must not report high accuracy on the
            # handful that did.
            fh.write(f"correct: {len(correct)} ({len(correct) / len(pulses):.1%} of attempts)\n")
            if latencies:
                mid = latencies[len(latencies) // 2]
                fh.write(f"latency p50: {mid:.2f}s  max: {latencies[-1]:.2f}s\n")
                # The documented failure mode is a latency WALL, not a bad
                # average: 11s -> 25s -> 105s, dead by turn 20. First versus
                # last is what shows it.
                first = float(answered[0].get("latency_s") or 0.0)
                last = float(answered[-1].get("latency_s") or 0.0)
                fh.write(f"latency first: {first:.2f}s  last: {last:.2f}s\n")
                if first > 0 and last > first * 3:
                    fh.write(
                        f"LATENCY WALL: last pulse was {last / first:.1f}x the first.\n"
                    )
            lanes = sorted({str(p.get("lane") or p.get("model") or "") for p in answered} - {""})
            if lanes:
                fh.write(f"lanes that answered: {', '.join(lanes)}\n")


async def _tick_snapshot(
    run_dir: Path,
    *,
    observer: ResourceObserver | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {"when": time.time()}
    observer = observer or get_resource_observer()
    compute = observer.compute()
    memory = observer.memory()
    disk = observer.disk("/")
    out["resource_observation"] = observer.provenance.to_dict()
    out["cpu_pct"] = compute.cpu_percent if compute.available else 100.0
    out["ram_pct"] = memory.percent if memory.available else 100.0
    out["disk_pct"] = disk.percent if disk.available else 100.0
    if not (compute.available and memory.available and disk.available):
        out["resource_probe"] = "observation_unavailable"
    try:
        from core.identity.self_object import get_self

        snap = get_self().snapshot()
        out["continuity_hash"] = snap.continuity_hash
        out["viability"] = snap.viability_state
        out["active_goals"] = len(snap.active_goals)
        out["active_tokens"] = snap.active_capability_tokens
    except _RECOVERABLE_SNAPSHOT_ERRORS as exc:
        out["self_error"] = str(exc)
    try:
        from core.agency.agency_orchestrator import get_receipt_log

        out["receipts_recent"] = len(get_receipt_log().recent(limit=200))
    except _RECOVERABLE_SNAPSHOT_ERRORS as exc:
        out["receipt_error"] = str(exc)
    await asyncio.to_thread(_append_text, run_dir / "events.jsonl", json.dumps(out, default=str) + "\n")
    await asyncio.to_thread(
        _append_resource_row,
        run_dir / "resource.csv",
        [out.get("when"), out.get("cpu_pct"), out.get("ram_pct"), out.get("disk_pct")],
    )
    return out


#: Pulses with checkable answers. A soak that only proves Aura REPLIED after 60
#: hours proves the process is alive, which the resource sampler already says.
#: What matters is whether she is still right, and these are chosen so the
#: grader is exact and needs no model: no ambiguity, no partial credit, no
#: judgement call at hour 60 that nobody is awake to make.
_VERIFIABLE_PULSES: tuple[tuple[str, str], ...] = (
    ("What is 17 multiplied by 23? Reply with the number only.", "391"),
    ("How many days are in a leap year? Reply with the number only.", "366"),
    ("What is the chemical symbol for iron? Reply with the symbol only.", "Fe"),
    ("Spell the word 'rhythm' backwards. Reply with the word only.", "mhtyhr"),
    ("What is 144 divided by 12? Reply with the number only.", "12"),
)


async def _fire_verifiable_pulse(
    run_dir: Path,
    *,
    chat_url: str,
    index: int,
    timeout_s: float,
) -> dict[str, Any]:
    """Ask one checkable question and record what came back.

    Every outcome is recorded, including the ones that are tempting to drop: an
    unreachable endpoint, a timeout, a wrong answer. A soak whose trace only
    contains successes cannot distinguish 72 hours of health from 72 hours of
    silence, which is exactly what the previous hook produced.
    """
    prompt, expected = _VERIFIABLE_PULSES[index % len(_VERIFIABLE_PULSES)]
    record: dict[str, Any] = {
        "when": time.time(),
        "kind": "verifiable_user_pulse",
        "prompt": prompt,
        "expected": expected,
    }
    started = time.monotonic()
    try:
        # Raw httpx, deliberately, and flagged as such by governance lint. This
        # driver is a CLIENT of the runtime, not part of it — it exists to
        # exercise Aura from outside exactly as a user's browser does, and a
        # user's browser does not route through Aura's own network gateway.
        # Driving the soak through the runtime's governed egress would make the
        # measurement depend on the subsystem under measurement.
        import httpx

        async with httpx.AsyncClient(timeout=timeout_s) as client:
            response = await client.post(chat_url, json={"message": prompt})
        record["status_code"] = response.status_code
        body = response.json() if response.content else {}
        answer = str(
            body.get("response") or body.get("message") or body.get("text") or ""
        )
        record["answer"] = answer[:500]
        record["ok"] = response.status_code == 200
        # Substring, deliberately: the pulse asks for the bare value but a
        # runtime is entitled to be conversational, and grading that as wrong
        # would report a style difference as a capability collapse.
        record["correct"] = expected.lower() in answer.lower()
        # Lane provenance when the runtime offers it. A reply from the reflex
        # rung at hour 60 is a different event from a reply from the cortex,
        # and a trace that cannot tell them apart cannot say what survived.
        for key in ("lane", "model", "source", "provenance"):
            if key in body:
                record[key] = body[key]
    except Exception as exc:  # noqa: BLE001 — every outcome must be recorded
        # Broad on purpose, and this is the one place in this file where that
        # is right. The whole value of the pulse is that its outcome lands in
        # the trace, and httpx raises from its own hierarchy
        # (httpx.ConnectError is not an OSError) — enumerating a third-party
        # library's exception tree here would mean an unreachable endpoint
        # CRASHES a 72-hour run at hour 3 instead of recording "unreachable"
        # and continuing. Measured: the first version did exactly that.
        record["ok"] = False
        record["correct"] = False
        record["error"] = f"{type(exc).__name__}: {exc}"[:300]
    record["latency_s"] = round(time.monotonic() - started, 3)
    await asyncio.to_thread(
        _append_text, run_dir / "user_pulse.jsonl", json.dumps(record, default=str) + "\n"
    )
    return record


async def _maybe_fire_user(
    run_dir: Path,
    *,
    chat_url: str = "",
    index: int = 0,
    timeout_s: float = 120.0,
) -> dict[str, Any]:
    """Drive one real turn when a chat endpoint is configured.

    Without --chat-url this records an EXPLICITLY undriven pulse rather than a
    synthetic one that reads like a real interaction. The distinction is the
    whole point: the previous hook wrote `{"kind": "scripted_user_pulse"}`
    whether or not anything had been asked, so a trace full of them was
    indistinguishable from a trace of a runtime nobody ever contacted.
    """
    if not chat_url:
        record = {
            "when": time.time(),
            "kind": "undriven_pulse",
            "ok": False,
            "correct": False,
            "reason": "no --chat-url configured; nothing was asked of the runtime",
        }
        await asyncio.to_thread(
            _append_text, run_dir / "user_pulse.jsonl", json.dumps(record) + "\n"
        )
        return record
    return await _fire_verifiable_pulse(
        run_dir, chat_url=chat_url, index=index, timeout_s=timeout_s
    )


async def _maybe_inject_chaos(run_dir: Path) -> None:
    from tools.chaos.injector import inject_random_fault

    fault = await inject_random_fault()
    await asyncio.to_thread(
        _append_text,
        run_dir / "chaos.jsonl",
        json.dumps({"when": time.time(), "fault": fault}) + "\n",
    )


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True, choices=list(_PROFILES.keys()))
    parser.add_argument("--tick-s", type=float, default=30.0, help="seconds between snapshots")
    parser.add_argument(
        "--chat-url",
        default="",
        help=(
            "chat endpoint to drive real verifiable turns against, e.g. "
            "http://127.0.0.1:8000/api/chat. WITHOUT this the user-pulse "
            "profiles record undriven pulses and prove nothing about load."
        ),
    )
    parser.add_argument(
        "--pulse-timeout-s", type=float, default=120.0,
        help="per-pulse timeout; a slow reply is a finding, not a hang",
    )
    args = parser.parse_args()
    profile = _PROFILES[args.profile]
    run_id = f"longevity-{args.profile}-{uuid.uuid4().hex[:8]}"
    from tools.longevity.soak_output import resolve_output_dir

    run_dir = resolve_output_dir(
        str(Path.home() / ".aura" / "data" / "longevity" / run_id)
    )
    logger.info("longevity run_id=%s profile=%s dir=%s duration_s=%s", run_id, args.profile, run_dir, profile["duration_s"])

    started = time.time()
    last_user_pulse = 0.0
    last_chaos = 0.0
    pulse_index = 0
    pulses: list[dict[str, Any]] = []
    if profile["user_pulse_s"] and not args.chat_url:
        logger.warning(
            "profile %s schedules user pulses but no --chat-url was given: "
            "pulses will be recorded as UNDRIVEN and this run cannot support "
            "any claim about behaviour under load.",
            args.profile,
        )
    while time.time() - started <= profile["duration_s"]:
        now = time.time()
        elapsed = now - started
        await _tick_snapshot(run_dir)
        if profile["user_pulse_s"] and (now - last_user_pulse) > profile["user_pulse_s"]:
            pulse = await _maybe_fire_user(
                run_dir,
                chat_url=args.chat_url,
                index=pulse_index,
                timeout_s=args.pulse_timeout_s,
            )
            pulse_index += 1
            pulses.append(pulse)
            last_user_pulse = now
        if profile.get("chaos") and (now - last_chaos) > 600.0:
            await _maybe_inject_chaos(run_dir)
            last_chaos = now
        remaining = max(0.0, profile["duration_s"] - elapsed)
        await asyncio.sleep(min(args.tick_s, remaining))

    rows = await asyncio.to_thread(_read_jsonl, run_dir / "events.jsonl")
    await asyncio.to_thread(
        _write_summary,
        run_dir,
        run_id=run_id,
        profile_name=args.profile,
        duration_s=profile["duration_s"],
        rows=rows,
    )
    logger.info("longevity complete: %s", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
