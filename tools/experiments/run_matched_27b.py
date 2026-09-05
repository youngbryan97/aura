#!/usr/bin/env python3
"""Run the matched experiment between Aura's 27B and the base 27B it came from.

The comparison item 9 asks for: frozen Aura against the base model, equal
compute, tokens, tools, time and information, on externally authored tasks,
with an ablation ladder.

Two things about how it runs, both forced by the machine rather than chosen.
The models are 15GB and 14GB and the host has about 17GB free, so they are
loaded one at a time and each arm finishes before the next begins. And every
run is bounded — a task cap, a token cap per answer, and a wall clock — because
an unbounded sweep on this host is how the live instance gets starved.

    python tools/experiments/run_matched_27b.py --tasks 40 --minutes 30

Tasks come from a file of externally authored problems, one JSON object per
line with ``id``, ``prompt`` and ``answer``. The harness refuses anything whose
author is not external, so a task set written here will score nothing, which
is the point.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from collections.abc import Callable  # noqa: E402

from core.evals.matched_experiment import (  # noqa: E402
    Arm,
    Budget,
    Spend,
    Task,
    run_matched,
)

MODELS = pathlib.Path.home() / ".aura" / "models"
BASE = MODELS / "Qwen3.8-27B-4bit-3e6447f082e8"
AURA = MODELS / "Aura-Qwen3.8-27B-persona-crsm-7f6a2e83f73f5eef9d15"


def load_tasks(path: pathlib.Path, limit: int) -> list[Task]:
    tasks: list[Task] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or len(tasks) >= limit:
                continue
            record = json.loads(line)
            gold = str(record["answer"]).strip()
            tasks.append(
                Task(
                    task_id=str(record["id"]),
                    prompt=str(record["prompt"]),
                    grade=(lambda answer, want=gold: want.lower() in answer.lower()),
                    author=str(record.get("author", "external_human")),
                    source=str(record.get("source", str(path))),
                    seen_before=bool(record.get("seen_before", False)),
                )
            )
    return tasks


def mlx_arm(
    name: str, path: pathlib.Path, *, adds: tuple[str, ...] = ()
) -> tuple[Arm, Callable[[], None]]:
    """One arm backed by a model loaded on first use and freed after the run."""
    state: dict[str, object] = {}

    def run(prompt: str, budget: Budget) -> tuple[str, Spend]:
        if "model" not in state:
            from mlx_lm import load

            print(f"  loading {path.name} ...", flush=True)
            model, tokenizer = load(str(path))
            state["model"], state["tokenizer"] = model, tokenizer
        from mlx_lm import generate

        started = time.perf_counter()
        text = generate(
            state["model"],
            state["tokenizer"],
            prompt=prompt,
            max_tokens=budget.tokens,
            verbose=False,
        )
        elapsed = time.perf_counter() - started
        tokens = len(state["tokenizer"].encode(text))
        return text, Spend(tokens=tokens, seconds=elapsed, tool_calls=0)

    def free() -> None:
        state.clear()
        import gc

        gc.collect()

    # Returned beside the arm rather than attached to it: Arm is frozen,
    # which is the right thing for an experiment design and means the
    # unload handle has to travel separately.
    return Arm(name=name, run=run, adds=adds), free


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=int, default=40)
    parser.add_argument("--minutes", type=float, default=30.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument(
        "--task-file",
        type=pathlib.Path,
        default=ROOT / "data" / "experiments" / "external_tasks.jsonl",
    )
    parser.add_argument("--out", type=pathlib.Path, default=None)
    args = parser.parse_args()

    for path in (BASE, AURA):
        if not path.is_dir():
            print(f"❌ model not found: {path}")
            return 1
    if not args.task_file.is_file():
        print(f"❌ no task file at {args.task_file}")
        print("   One JSON object per line: {\"id\", \"prompt\", \"answer\", \"author\"}")
        return 1

    tasks = load_tasks(args.task_file, args.tasks)
    print(f"📋 {len(tasks)} tasks from {args.task_file}")

    budget = Budget(
        tokens=args.max_tokens,
        seconds=args.minutes * 60.0 / max(1, len(tasks) * 2),
        tool_calls=0,
    )
    print(f"⚖️  matched allowance per task: {budget.to_dict()}")

    built = [
        mlx_arm("base_27b", BASE),
        mlx_arm("aura_27b", AURA, adds=("persona", "crsm adaptation")),
    ]
    arms = [arm for arm, _free in built]

    started = time.time()
    report = run_matched(arms, tasks, budget=budget)
    for _arm, free in built:
        free()

    print(f"\n⏱  {time.time() - started:.1f}s")
    print(f"🔒 seal {report.seal}")
    for result in report.arms:
        print(f"  {result.arm:12s} {result.to_dict()}")
    for rung in report.ladder:
        print(f"  {rung.arm} over {rung.over}: {rung.to_dict()}")
    print(f"\n🧾 {report.verdict}")

    out = args.out or (
        ROOT / "docs" / "evidence" / "matched_experiment" / f"run_{int(started)}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
    print(f"📄 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
