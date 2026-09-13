#!/usr/bin/env python3
"""Two arms that differ by recurrent depth, and nothing else.

The depth gate has never had an arm to read. The two reports on disk carry the
same responses digest, the same accuracy and no declared depth, seventy
seconds apart — one run written twice, because ``tools/heldout_eval.py`` has
no depth option and could not have varied the thing the filenames claim to
compare.

This varies it. ``apply_recurrent_depth`` patches the model's forward pass to
re-run a middle band of layers, and the worker's surface contract sets
``inner._recurrent_depth_runtime_loops`` per request — so one patched model
serves both arms and the only difference between them is that integer. That
is the exact quantity ``user_surface_recurrent_ceiling`` governs, so an
improvement here is an improvement in the thing the ceiling would be raised
for.

Decoding is greedy and the battery is the sealed held-out set, regenerated
from (version, seed, size), so a stranger can re-run this and get the same
tasks. Each arm writes the report shape the gate reads: the declared depth,
the accuracy, the battery, and a digest of the raw responses beside it.

**The substrate is whatever --model names.** A depth that helps one checkpoint
says nothing about another, which is why the receipt the gate writes is named
by descriptor.

    python tools/run_recurrent_depth_arms.py --model <path> --size 40
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from core.learning.heldout_battery import (  # noqa: E402
    BatterySpec,
    generate_battery,
    grade_response,
)

#: Greedy, so two runs at the same depth are the same run. The arms have to
#: differ by the depth and by nothing else, and a sampler is the easiest way
#: to lose that.
TEMPERATURE = 0.0
MAX_TOKENS = 256


def _inner(model: Any) -> Any:
    from core.runtime.model_layers import resolve_model_layers

    view = resolve_model_layers(model)
    return view.owner if view is not None else None


def _ask(model, tokenizer, prompt: str) -> str:
    import mlx_lm
    from mlx_lm.sample_utils import make_sampler

    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    return str(
        mlx_lm.generate(
            model,
            tokenizer,
            prompt=text,
            max_tokens=MAX_TOKENS,
            verbose=False,
            sampler=make_sampler(temp=TEMPERATURE),
        )
    )


def _run_arm(model, tokenizer, tasks, loops: int, out: Path) -> dict[str, Any]:
    """One pass of the battery at one depth, written where the gate reads."""
    inner = _inner(model)
    if inner is None or not getattr(inner, "_recurrent_depth_config", None):
        raise RuntimeError(
            "the loaded model carries no recurrent depth: an arm at any loop "
            "count would be the same forward pass, which is the defect this "
            "harness exists to stop repeating"
        )
    inner._recurrent_depth_runtime_loops = int(loops)

    started = time.time()
    rows: list[dict[str, Any]] = []
    correct = 0
    for task in tasks:
        response = _ask(model, tokenizer, task.prompt)
        ok = grade_response(task, response)
        correct += 1 if ok else 0
        rows.append(
            {
                "task_id": task.task_id,
                "domain": task.domain,
                "correct": bool(ok),
                "response": response,
            }
        )

    responses = out.with_suffix("").with_suffix(".responses.jsonl")
    responses.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    report = {
        "schema": "aura.recurrent_depth.arm.v1",
        "tool": "run_recurrent_depth_arms",
        # The name the gate reads the depth from, and the reason these two
        # files are an arm each rather than one run written twice.
        "recurrent_loops": int(loops),
        "recurrent_depth_config": dict(inner._recurrent_depth_config),
        "accuracy": round(correct / len(tasks), 4) if tasks else 0.0,
        "result": {"correct": correct, "total": len(tasks)},
        "battery": {"size": len(tasks)},
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "elapsed_s": round(time.time() - started, 1),
        "responses_sha256": hashlib.sha256(responses.read_bytes()).hexdigest(),
        "created_at": time.time(),
    }
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"  loops={loops}  {correct}/{len(tasks)} = {report['accuracy']:.4f}"
        f"  ({report['elapsed_s']:.0f}s)  -> {out.name}",
        flush=True,
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size", type=int, default=40)
    parser.add_argument("--loops", type=int, nargs=2, default=[1, 2])
    parser.add_argument(
        "--out-dir", type=Path, default=REPO / "artifacts" / "recurrent_depth"
    )
    arguments = parser.parse_args()

    tasks = generate_battery(BatterySpec(seed=arguments.seed, size=arguments.size))
    print(f"battery: {len(tasks)} tasks, seed {arguments.seed}", flush=True)

    from core.brain.llm.recurrent_depth import apply_recurrent_depth
    from core.runtime.model_lane_control import standalone_model_lane

    with standalone_model_lane(
        owner_id=f"recurrent-depth-arms:{Path(arguments.model).name}",
        model_path=str(arguments.model),
        purpose="evaluation",
        preemptible=False,
        metadata={"tool": "run_recurrent_depth_arms"},
    ):
        from mlx_lm import load

        print(f"loading {Path(arguments.model).name}", flush=True)
        model, tokenizer = load(str(arguments.model))

        # Patched ONCE, at the deeper count, so both arms run the identical
        # forward pass and differ only in the runtime loop integer the worker
        # sets per request. Patching per arm would make the arms two different
        # models and the comparison meaningless.
        if not apply_recurrent_depth(model, n_loops=max(arguments.loops)):
            print("recurrent depth could not be applied to this model", file=sys.stderr)
            return 2

        arguments.out_dir.mkdir(parents=True, exist_ok=True)
        reports = [
            _run_arm(
                model,
                tokenizer,
                tasks,
                loops,
                arguments.out_dir / f"arm_loops{loops}.json",
            )
            for loops in arguments.loops
        ]

    if reports[0]["responses_sha256"] == reports[1]["responses_sha256"]:
        print(
            "\nthe two arms produced identical responses: the loop count did "
            "not reach the forward pass, and this is not a comparison",
            file=sys.stderr,
        )
        return 2
    print("\nthe arms differ. Run the gate on them:")
    for loops in arguments.loops:
        print(f"  artifacts/recurrent_depth/arm_loops{loops}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
