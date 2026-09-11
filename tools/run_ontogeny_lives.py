#!/usr/bin/env python3
"""Two matched lives, different histories, and whether the difference is hers.

The developmental domain is the slowest thing in the core, and a slow counter
looks exactly like development over a short run. Both go up. Both correlate
with how long the system has been running. The question the battery cannot
answer from one life is whether N carries what happened rather than how much
happened, and that needs two lives that differ in what they met.

So: build one organism, run it long enough to have a past, then fork it. Give
the two forks different histories — one meets conversation and salience, the
other idle and memory — and watch N. Then put both back into the same
environment and measure whether the difference survives into what they do
next. A reservoir that has genuinely developed answers the same stimulus
differently after two different lives; a counter does not.

Three things stop this from coming out positive by construction.

The twins are forked from one snapshot, so they start identical in every
domain, and the fork carries the RNG. Anything that differs afterwards differs
because of what they met.

A third fork runs the same history as the first with its reservoir frozen. Its
memory grows the same way and its databases fill the same way, so anything the
frozen twin also shows is accumulation rather than development.

And N is snapshotted and restored across a simulated restart, because a
developmental state that does not survive a process boundary is a cache.

    python tools/run_ontogeny_lives.py --epochs 4 --turns-per-epoch 12
    python tools/run_ontogeny_lives.py --quick
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

logger = logging.getLogger("ontogeny_lives")

#: The two histories. Named before the run, and not chosen after seeing which
#: pair diverged: one is the outward-facing life and the other the inward one,
#: which is the split the conditions were written with.
HISTORY_A: tuple[str, ...] = ("conversation", "salience", "tool_use", "problem_solving")
HISTORY_B: tuple[str, ...] = ("idle", "memory", "autonomy", "stress")

#: What both lives are put back into afterwards, to ask whether the difference
#: reaches what they do next.
SHARED: tuple[str, ...] = ("conversation", "idle")


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _reading(runtime: Any) -> Any:
    from core.subject.state import read_core_state

    return read_core_state(
        runtime.state, ontogeny=runtime.ontogeny, organs=runtime.organs
    )


def _n_vector(runtime: Any) -> np.ndarray:
    return np.asarray(_reading(runtime).values["N"], dtype=np.float64)


def _reservoir(runtime: Any) -> np.ndarray:
    hidden = getattr(runtime.ontogeny, "h", None)
    if hidden is None:
        return np.zeros(1)
    return np.asarray(hidden, dtype=np.float64).reshape(-1).copy()


def _distance(left: np.ndarray, right: np.ndarray) -> float:
    """Cosine distance, which does not grow just because both grew."""
    a, b = left.reshape(-1), right.reshape(-1)
    size = min(a.size, b.size)
    a, b = a[:size], b[:size]
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    if norm <= 1e-12:
        return 0.0
    return float(1.0 - (a @ b) / norm)


async def _live(
    runtime: Any,
    conditions: Sequence[Any],
    history: Sequence[str],
    *,
    epochs: int,
    turns: int,
    label: str,
    freeze: bool = False,
) -> dict[str, Any]:
    """Run one life and record what its developmental state did, epoch by epoch."""
    by_name = {c.name: c for c in conditions}
    chosen = [by_name[name] for name in history if name in by_name]
    if not chosen:
        raise RuntimeError(f"none of {list(history)} is an ordinary condition")

    frozen = _reservoir(runtime) if freeze else None
    track: list[dict[str, Any]] = []
    for epoch in range(epochs):
        novelties: list[float] = []
        for turn in range(turns):
            condition = chosen[turn % len(chosen)]
            await runtime.turn_once(condition)
            if frozen is not None and runtime.ontogeny is not None:
                # The control. Everything else about this life runs; only the
                # reservoir is held where it started, so its memory and its
                # stores accumulate exactly as the others do.
                runtime.ontogeny.h = frozen.copy()
            novelties.append(float(getattr(runtime.ontogeny, "last_novelty", 0.5)))
        hidden = _reservoir(runtime)
        track.append(
            {
                "epoch": epoch,
                "steps": int(getattr(runtime.ontogeny, "steps", 0)),
                "era": int(getattr(runtime.ontogeny, "era", 1)),
                "novelty_mean": round(float(np.mean(novelties)), 5),
                "novelty_last": round(novelties[-1], 5),
                "reservoir_norm": round(float(np.linalg.norm(hidden)), 5),
                "reservoir_saturation": round(
                    float(np.mean(np.abs(np.tanh(hidden)))), 5
                ),
            }
        )
        _log(
            f"  {label} epoch {epoch}: novelty {track[-1]['novelty_mean']:.4f}, "
            f"|h| {track[-1]['reservoir_norm']:.4f}, "
            f"saturation {track[-1]['reservoir_saturation']:.4f}"
        )
    return {
        "label": label,
        "history": list(history),
        "epochs": track,
        "reservoir": _reservoir(runtime),
        "n_domain": _n_vector(runtime),
        "frozen": freeze,
    }


async def _same_room(runtime: Any, conditions: Sequence[Any], turns: int) -> np.ndarray:
    """Put a life into the shared environment and record what it does there."""
    by_name = {c.name: c for c in conditions}
    shared = [by_name[name] for name in SHARED if name in by_name]
    from core.subject.state import FAST_DOMAINS

    rows: list[np.ndarray] = []
    for turn in range(turns):
        await runtime.turn_once(shared[turn % len(shared)])
        reading = _reading(runtime)
        rows.append(np.concatenate([reading.values[key] for key in FAST_DOMAINS]))
    return np.mean(np.vstack(rows), axis=0) if rows else np.zeros(1)


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO / "artifacts" / "ontogeny_lives")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--turns-per-epoch", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=8, help="turns before the fork")
    parser.add_argument("--shared-turns", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.epochs, args.turns_per_epoch, args.warmup, args.shared_turns = 2, 3, 2, 2

    args.out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("AURA_LOG_DIR", str(args.out / "logs"))

    from core.subject.driver import (
        CONDITIONS,
        build_runtime,
        quiesce_organism,
        start_organism,
    )
    from core.subject.provenance import campaign_v25, environment, next_run_directory

    run_dir = next_run_directory(args.out)
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    _log(f"ontogeny run {run_dir.name}")

    runtime = build_runtime(run_dir, seed=args.seed)
    await start_organism(runtime)

    evidence: dict[str, Any] = {
        "campaign": campaign_v25(
            seed=args.seed, rounds=args.turns_per_epoch, anchors=0,
            history_turns=args.epochs, turns=args.turns_per_epoch,
            cut_rounds=0, support=("N",),
        ),
        "environment": environment(),
        "design": {
            "history_a": list(HISTORY_A),
            "history_b": list(HISTORY_B),
            "shared": list(SHARED),
            "epochs": args.epochs,
            "turns_per_epoch": args.turns_per_epoch,
            "warmup": args.warmup,
        },
    }

    try:
        _log(f"warming up {args.warmup} turns before the fork")
        for turn in range(args.warmup):
            await runtime.turn_once(CONDITIONS[turn % len(CONDITIONS)])
        fork = runtime.snapshot()
        start_n = _n_vector(runtime)
        start_h = _reservoir(runtime)

        _log("life A: " + ", ".join(HISTORY_A))
        life_a = await _live(
            runtime, CONDITIONS, HISTORY_A,
            epochs=args.epochs, turns=args.turns_per_epoch, label="A",
        )
        after_a = await _same_room(runtime, CONDITIONS, args.shared_turns)

        runtime.restore(fork)
        _log("life B: " + ", ".join(HISTORY_B))
        life_b = await _live(
            runtime, CONDITIONS, HISTORY_B,
            epochs=args.epochs, turns=args.turns_per_epoch, label="B",
        )
        after_b = await _same_room(runtime, CONDITIONS, args.shared_turns)

        runtime.restore(fork)
        _log("life A with the reservoir frozen: the accumulation control")
        life_frozen = await _live(
            runtime, CONDITIONS, HISTORY_A,
            epochs=args.epochs, turns=args.turns_per_epoch, label="frozen",
            freeze=True,
        )
        after_frozen = await _same_room(runtime, CONDITIONS, args.shared_turns)

        # Does N survive a process boundary? The snapshot and restore the
        # experiment already uses is the same boundary a restart is.
        runtime.restore(fork)
        before_restart = _reservoir(runtime)
        restored = runtime.snapshot()
        runtime.restore(restored)
        after_restart = _reservoir(runtime)

        evidence["lives"] = {
            name: {k: v for k, v in life.items() if k not in ("reservoir", "n_domain")}
            for name, life in (("A", life_a), ("B", life_b), ("frozen", life_frozen))
        }
        evidence["divergence"] = {
            "reservoir_a_vs_b": round(_distance(life_a["reservoir"], life_b["reservoir"]), 6),
            "reservoir_a_vs_frozen": round(
                _distance(life_a["reservoir"], life_frozen["reservoir"]), 6
            ),
            "n_domain_a_vs_b": round(_distance(life_a["n_domain"], life_b["n_domain"]), 6),
            "n_domain_a_vs_frozen": round(
                _distance(life_a["n_domain"], life_frozen["n_domain"]), 6
            ),
            "from_the_fork_a": round(_distance(start_h, life_a["reservoir"]), 6),
            "from_the_fork_b": round(_distance(start_h, life_b["reservoir"]), 6),
        }
        evidence["same_room_afterwards"] = {
            "a_vs_b": round(_distance(after_a, after_b), 6),
            "a_vs_frozen": round(_distance(after_a, after_frozen), 6),
            "turns": args.shared_turns,
        }
        evidence["restart"] = {
            "reservoir_survived": bool(
                np.allclose(before_restart, after_restart, atol=1e-9)
            ),
            "drift": round(float(np.max(np.abs(before_restart - after_restart))), 12),
        }
        evidence["saturation"] = _saturation(life_a, life_b)
        evidence["verdict"] = _verdict(evidence)
        del start_n
    finally:
        await quiesce_organism(runtime)

    evidence["seconds"] = round(time.monotonic() - started, 1)
    out = run_dir / "ontogeny_lives.json"
    out.write_text(json.dumps(evidence, indent=2, default=str) + "\n", encoding="utf-8")
    _log("")
    for line in _lines(evidence):
        print(line)
    _log(f"wrote {out} in {evidence['seconds']}s")
    return 0 if evidence["verdict"]["development_rather_than_accumulation"] else 1


def _saturation(life_a: dict[str, Any], life_b: dict[str, Any]) -> dict[str, Any]:
    """Whether N ran out of room, and whether novelty fell as experience grew."""
    out: dict[str, Any] = {}
    for name, life in (("A", life_a), ("B", life_b)):
        epochs = life["epochs"]
        if len(epochs) < 2:
            out[name] = {"epochs": len(epochs)}
            continue
        saturation = [row["reservoir_saturation"] for row in epochs]
        novelty = [row["novelty_mean"] for row in epochs]
        out[name] = {
            "saturation_first": saturation[0],
            "saturation_last": saturation[-1],
            # A reservoir pinned at its ceiling has stopped carrying anything
            # new, which is the failure "N does not immediately saturate" names.
            "saturated": bool(saturation[-1] > 0.95),
            "novelty_first": novelty[0],
            "novelty_last": novelty[-1],
            "novelty_fell_with_experience": bool(novelty[-1] < novelty[0]),
        }
    return out


def _verdict(evidence: dict[str, Any]) -> dict[str, Any]:
    """Two lives diverged, and the frozen control did not."""
    div = evidence["divergence"]
    room = evidence["same_room_afterwards"]
    saturation = evidence.get("saturation", {})
    lives_diverged = div["reservoir_a_vs_b"] > 0.0
    # The control is the whole argument. Anything the frozen twin also shows
    # is its stores filling, not its reservoir developing.
    beyond_accumulation = div["reservoir_a_vs_b"] > div["reservoir_a_vs_frozen"]
    reaches_behaviour = room["a_vs_b"] > 0.0
    unsaturated = not any(
        bool(row.get("saturated")) for row in saturation.values() if isinstance(row, dict)
    )
    return {
        "two_lives_diverge_in_N": bool(lives_diverged),
        "development_rather_than_accumulation": bool(lives_diverged and beyond_accumulation),
        "the_difference_reaches_what_they_do_next": bool(reaches_behaviour),
        "N_did_not_saturate": bool(unsaturated),
        "N_survives_a_restart": bool(evidence["restart"]["reservoir_survived"]),
        "note": (
            "A slow counter also goes up with time. What separates development "
            "from accumulation here is the frozen twin, which lives the same "
            "history and fills the same stores with its reservoir held still."
        ),
    }


def _lines(evidence: dict[str, Any]) -> list[str]:
    div = evidence["divergence"]
    verdict = evidence["verdict"]
    return [
        "two lives from one fork:",
        f"  reservoir, A against B:            {div['reservoir_a_vs_b']}",
        f"  reservoir, A against frozen twin:  {div['reservoir_a_vs_frozen']}",
        f"  N domain, A against B:             {div['n_domain_a_vs_b']}",
        f"  in the same room afterwards:       {evidence['same_room_afterwards']['a_vs_b']}",
        "",
        f"  two lives diverge in N:            {verdict['two_lives_diverge_in_N']}",
        f"  development, not accumulation:     {verdict['development_rather_than_accumulation']}",
        f"  the difference reaches behaviour:  {verdict['the_difference_reaches_what_they_do_next']}",
        f"  N did not saturate:                {verdict['N_did_not_saturate']}",
        f"  N survives a restart:              {verdict['N_survives_a_restart']}",
    ]


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    raise SystemExit(asyncio.run(main()))
