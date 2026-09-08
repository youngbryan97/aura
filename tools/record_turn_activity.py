#!/usr/bin/env python3
"""Record the phase pipeline running a turn, so the ring has a chance to close.

The coalition test asks whether influence runs from interoception through
affect, the workspace, higher-order monitoring, the self model and planning to
action and back. Every recording made so far answered no, and the recordings
were the reason. They drove each station by calling its readers — a getter, a
status, a snapshot — which fires cells inside a station and hands nothing to
the next one. A recording in which no station can influence another cannot
show one influencing another, and reporting that as a negative result about
the architecture would have been reporting the workload.

This drives the mechanism by which the stations are actually coupled: the
kernel's phase pipeline, running over one shared ``AuraState``. Affect writes
the state that motivation reads; integration writes what routing reads;
response writes what review reads. That is the coupling, and it only exists
while a turn is being taken.

The model is a stub that answers deterministically. That is not a simulation of
her thinking — the phases are the real phases and the state is the real state
— it is the one part that cannot run offline without a resident 32B, and
holding it constant is what makes conditions comparable rather than a
measurement of decoding noise.

Each objective is a condition. Within a condition the whole ring runs, so a
between-station effect is possible; across conditions the same ring runs on
different content, so a link that carries in all of them is a different finding
from one that carries in one.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

os.environ.setdefault("AURA_TESTING", "1")

#: What she is asked, and the name the condition gets. Chosen to differ in kind
#: rather than in wording: a greeting, a question about her own state, a request
#: that would act on the world, a recall, an inference, and an idle tick with no
#: objective at all. If the effective graphs of these come out identical, the
#: phases are not reading the objective.
OBJECTIVES: tuple[tuple[str, str], ...] = (
    ("greeting", "Hello, how are you?"),
    ("self_report", "What are you feeling right now, and how sure are you?"),
    ("task", "Write a file called notes.txt holding the plan for today."),
    ("recall", "What did we talk about before this?"),
    ("inference", "Every A is a B, and some B are C. Does some A being C follow?"),
    ("idle", ""),
)


class _DeterministicPhaseLLM:
    """One answer, always. Held constant so conditions differ by objective only."""

    async def think(self, prompt: str, **_kwargs: Any) -> str:
        return "Verified continuity summary: the phase pipeline is executing deterministically."

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        return await self.think(prompt, **kwargs)

    async def classify(self, _prompt: str) -> str:
        return "CHAT"

    async def embed(self, _text: str) -> list[float]:
        return [0.0] * 8


def _build_kernel(tmpdir: Path) -> Any:
    from core.kernel.aura_kernel import AuraKernel, KernelConfig
    from core.state.state_repository import StateRepository

    vault = StateRepository(db_path=str(tmpdir / "turn_recording.db"), is_vault_owner=True)
    kernel = AuraKernel(config=KernelConfig(), vault=vault)
    kernel._setup_phases()
    kernel._initialize_organs()
    kernel.organs["llm"] = SimpleNamespace(get_instance=lambda: _DeterministicPhaseLLM())
    return kernel


async def _one_turn(kernel: Any, objective: str, deadline: float) -> dict[str, Any]:
    """Run every phase once over one state. A phase that raises is counted."""
    from core.state.aura_state import AuraState

    state = AuraState.default()
    state.cognition.current_objective = objective
    ran = 0
    failed: dict[str, str] = {}
    for phase in kernel._phases:
        if time.monotonic() > deadline:
            break
        name = phase.__class__.__name__
        try:
            result = await asyncio.wait_for(
                phase.execute(state, objective=objective), timeout=20.0
            )
            ran += 1
            if result is not None:
                state = result
        except BaseException as exc:  # noqa: BLE001 - a phase that fails is a datum
            failed[name] = f"{type(exc).__name__}: {exc}"[:160]
    return {"phases_ran": ran, "phases_failed": len(failed), "failures": failed}


async def _drive(args: argparse.Namespace, recorder: Any) -> list[dict[str, Any]]:
    import tempfile

    log: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as raw:
        kernel = _build_kernel(Path(raw))
        deadline = time.monotonic() + args.budget
        for round_index in range(args.rounds):
            for condition, objective in OBJECTIVES:
                if time.monotonic() > deadline:
                    break
                recorder.set_condition(condition)
                started = time.monotonic()
                outcome = await _one_turn(kernel, objective, deadline)
                entry = {
                    "round": round_index,
                    "condition": condition,
                    "seconds": round(time.monotonic() - started, 2),
                    **outcome,
                }
                log.append(entry)
                print(
                    json.dumps({k: v for k, v in entry.items() if k != "failures"}),
                    flush=True,
                )
        # One failure report, not one per turn: the same phases fail every time
        # and printing them each round buries the useful line.
        seen: dict[str, str] = {}
        for entry in log:
            seen.update(entry.get("failures", {}))
        if seen:
            print(json.dumps({"phases_that_never_ran": seen}, indent=2), flush=True)
    return log


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=float, default=300.0, help="seconds of driving")
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--frame-seconds", type=float, default=0.05)
    parser.add_argument("--out", type=Path, default=REPO / "artifacts" / "connectome" / "turn")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("AURA_LOG_DIR", str(args.out / "logs"))

    from core.connectome.activity import ActivityRecorder, RecorderConfig

    recorder = ActivityRecorder(
        REPO,
        RecorderConfig(
            frame_seconds=args.frame_seconds,
            capture_edges=True,
            max_wall_seconds=args.budget + 300,
            max_frames=32_768,
        ),
    )
    recorder.start(OBJECTIVES[0][0])
    try:
        log = asyncio.run(_drive(args, recorder))
    finally:
        trace = recorder.stop()

    import numpy as np

    np.savez_compressed(args.out / "activity.npz", spikes=trace.matrix())
    (args.out / "activity_manifest.json").write_text(
        json.dumps(
            {
                "uids": list(trace.uids),
                "conditions": list(trace.conditions),
                "frame_seconds": trace.frame_seconds,
                "summary": trace.summary(),
                "attrs": trace.attrs,
                "log": log,
            },
            indent=2,
        )
    )
    (args.out / "observed_edges.json").write_text(
        json.dumps(
            {
                "counts": {
                    f"{pre}>{post}": count
                    for (pre, post), count in recorder.observed.counts.items()
                },
                "summary": recorder.observed.summary(),
            },
            indent=2,
        )
    )
    print(
        json.dumps(
            {"trace": trace.summary(), "observed": recorder.observed.summary()},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
