#!/usr/bin/env python
"""Accuracy gate for CP227 intrinsic recurrence — does the CE crossover convert?

CP227's held-out verdict was cross-entropy ordering (+0.005 nats). This tool
asks the question that actually decides whether that means anything: on
held-out tasks, decoded through the SAME intrinsic recurrent forward the
adapter trained under, does *exact-answer accuracy* rise with recurrent
depth, and does the trained adapter beat the untrained path and vanilla?

Arms (one wrapped model, no reloads — toggle the trained deltas on/off):
  * on@d   — trained adapter, intrinsic forward at depth d
  * off@d  — deltas zeroed (untrained intrinsic path); off@1 == vanilla
The decisive blocks (on@4, on@1, off@1) run FIRST and a partial report is
written after every block, so a wall-clock stop still answers "does trained
depth-4 beat vanilla" honestly.

Faithful by construction: it imports attach_adapters / trainable_parameters
from the trainer, so the model is wrapped identically to training. Bounded,
partial-safe, detached-friendly (run it under tools/run_detached_step.py).

  AURA_LOG_DIR=~/.aura/lab-logs .venv/bin/python tools/eval_intrinsic_accuracy.py \
      --model training/fused-model/Aura-32B-crsm-closeout-jul1-20260701-215118 \
      --adapter artifacts/closeout/latent_cortex/cp227_intrinsic_training \
      --out artifacts/closeout/latent_cortex/cp227_intrinsic_training/accuracy_gate.json \
      --calibrate 2      # measures timing, prints a real ETA, then exits

MEMORY SAFETY: 32B runs require the live app DOWN.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

EVAL_SCHEMA = "aura.intrinsic_accuracy_gate.v1"
_JSON_OBJ_RE = re.compile(r"\{[^{}]*\}")


def _expected_payload(task) -> dict:
    raw = task.answer.split("FINAL_ANSWER:", 1)[1].strip()
    return json.loads(raw)


def _graded_correct(generated: str, expected: dict) -> bool:
    """Tolerant on form, strict on value: first JSON object must equal expected."""
    match = _JSON_OBJ_RE.search(generated or "")
    if not match:
        return False
    try:
        return json.loads(match.group(0)) == expected
    except (ValueError, TypeError):
        return False


def _decode(
    model,
    tokenizer,
    prompt_ids,
    plan,
    *,
    max_tokens: int,
    eos: int | None,
    activation_totals: dict[str, int] | None = None,
):
    """Greedy decode through the intrinsic recurrent forward (non-cached).

    Answers are short JSON objects, so a re-forward-per-token loop is bounded
    and avoids per-iteration recurrent-cache subtlety. Stops at EOS, at a
    parseable JSON object, or the token budget.

    The forward runs inside ``recurrence_adapter_scope`` because
    ``ScopedLoRALinear`` is dark outside it: the first gate run decoded both
    arms with the adapter never applied, so on@d equaled off@d exactly and
    the verdict measured base-vs-base. ``activation_totals`` accumulates the
    scope's call/position counts so the caller can PROVE the treatment fired
    instead of assuming it did.
    """
    import mlx.core as mx

    from core.brain.llm.latent_cortex.recurrence_adapter import (
        recurrence_adapter_scope,
    )
    from core.learning.intrinsic_recurrence import recurrent_logits

    tokens = list(prompt_ids)
    produced: list[int] = []
    for _ in range(int(max_tokens)):
        with recurrence_adapter_scope(start=None, stop=None) as activation:
            logits = recurrent_logits(model, mx.array([tokens]), plan)
        if activation_totals is not None:
            activation_totals["calls"] = (
                activation_totals.get("calls", 0) + activation.calls
            )
            activation_totals["adapted_positions"] = (
                activation_totals.get("adapted_positions", 0)
                + activation.adapted_positions
            )
            # Per-site identity, not just the aggregate. A block that was
            # wrapped and never fired is invisible in `calls` -- the same
            # class of blindness that voided the first gate run, one level
            # down.
            fired = activation_totals.setdefault("applied_sites", {})
            for site, count in activation.applied_sites.items():
                fired[site] = fired.get(site, 0) + count
            blocks = activation_totals.setdefault("applied_blocks", {})
            for block, count in activation.applied_blocks.items():
                blocks[str(block)] = blocks.get(str(block), 0) + count
        nxt = int(mx.argmax(logits[0, -1, :]))
        if eos is not None and nxt == eos:
            break
        produced.append(nxt)
        tokens.append(nxt)
        text = tokenizer.decode(produced)
        if _JSON_OBJ_RE.search(text):  # answer complete
            break
    return tokenizer.decode(produced)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True, help="dir holding adapters.safetensors")
    parser.add_argument("--out", required=True)
    parser.add_argument("--families", default="khop,modular,register_trace")
    parser.add_argument("--task-depth", type=_positive_int, default=8)
    parser.add_argument("--per-cell", type=_positive_int, default=16)
    parser.add_argument("--depths", default="1,2,4")
    parser.add_argument("--prelude-end", type=int, default=16)
    parser.add_argument("--coda-start", type=int, default=48)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-targets", default="o_proj,v_proj")
    parser.add_argument("--eval-seed", type=int, default=424242)
    parser.add_argument("--train-seed", type=int, default=20260721)
    parser.add_argument("--max-answer-tokens", type=_positive_int, default=24)
    parser.add_argument("--max-minutes", type=float, default=120.0)
    parser.add_argument("--memory-fraction", type=float, default=0.55)
    parser.add_argument(
        "--calibrate", type=int, default=0,
        help="run this many tasks across all arms, print timing + ETA, then exit",
    )
    args = parser.parse_args()

    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten
    from mlx_lm import load

    from core.learning import recurrence_curriculum as curriculum
    from core.learning.intrinsic_recurrence_objective import IntrinsicTrainingSpec
    from core.runtime.mlx_memory_guard import mlx_memory_envelope
    from core.runtime.model_lane_control import standalone_model_lane
    from tools.train_intrinsic_recurrence import attach_adapters, trainable_parameters

    families = [f.strip() for f in args.families.split(",") if f.strip()]
    depths = sorted({int(v) for v in args.depths.split(",") if v.strip()} | {1})
    targets = tuple(t.strip() for t in args.lora_targets.split(",") if t.strip())
    bridge = "\n\nFINAL_ANSWER: "
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + args.max_minutes * 60.0

    spec = IntrinsicTrainingSpec(
        prelude_end=args.prelude_end,
        coda_start=args.coda_start,
        depths=tuple(depths),
    )

    # Held-out eval tasks: a seed disjoint from training's (20260721 and its
    # +9973 holdout), filtered against a regenerated train set as belt-and-
    # suspenders so a decoded win cannot be memorization.
    eval_tasks = curriculum.task_battery(
        families, [args.task_depth], args.per_cell, seed=args.eval_seed
    )
    train_prompts = {
        t.prompt
        for t in curriculum.task_battery(families, [args.task_depth], 64, seed=args.train_seed)
    }
    eval_tasks = [t for t in eval_tasks if t.prompt not in train_prompts]
    if args.calibrate:
        eval_tasks = eval_tasks[: args.calibrate]

    report: dict = {
        "schema": EVAL_SCHEMA,
        "model": args.model,
        "adapter": args.adapter,
        "families": families,
        "task_depth": args.task_depth,
        "depths": depths,
        "eval_seed": args.eval_seed,
        "n_tasks": len(eval_tasks),
        "bridge": bridge,
        "started_at": time.time(),
        "accuracy": {},          # "on@4" -> {"correct": k, "n": n, "acc": x}
        "block_seconds": {},
        "partial": True,
    }

    def write_report():
        out_path.write_text(json.dumps(report, indent=1, sort_keys=True))

    with standalone_model_lane(
        owner_id=f"eval-intrinsic-accuracy:{out_path.name}",
        model_path=args.model,
        purpose="evaluation",
        preemptible=False,
        metadata={"tool": "eval_intrinsic_accuracy", "operator_launched": True},
    ), mlx_memory_envelope(fraction=args.memory_fraction) as envelope:
        print(f"loading {args.model} …", flush=True)
        model, tokenizer = load(args.model)
        eos = getattr(tokenizer, "eos_token_id", None)
        wiring = attach_adapters(
            model, spec, rank=args.lora_rank, targets=targets, depth_conditioned=True
        )
        expected_sites = list(wiring.get("adapted_sites") or [])
        report["adapter_wiring"] = dict(wiring)

        loaded = mx.load(str(Path(args.adapter) / "adapters.safetensors"))
        # names in the file are the trainable (lora/delta) params
        on_params = dict(loaded)
        off_params = {k: mx.zeros_like(v) for k, v in loaded.items()}
        trainable_names = set(dict(tree_flatten(trainable_parameters(model))))
        matched = trainable_names & set(on_params)
        report["adapter_param_match"] = {
            "adapter_params": len(on_params),
            "model_trainable": len(trainable_names),
            "matched": len(matched),
        }
        if not matched:
            report["error"] = "adapter params do not match model trainable params"
            write_report()
            print("FATAL: adapter/model param mismatch — wrong wrapping", flush=True)
            return 2

        def set_arm(params):
            model.update(tree_unflatten(list(params.items())))

        # Decisive blocks first so a wall-stop still answers the core question.
        block_order = [("on", 4), ("on", 1), ("off", 1), ("on", 2), ("off", 4), ("off", 2)]
        block_order = [(a, d) for (a, d) in block_order if d in depths]

        current_arm = None
        for arm, depth in block_order:
            if time.monotonic() > deadline:
                report["stop_reason"] = "wall_clock"
                break
            if arm != current_arm:
                set_arm(on_params if arm == "on" else off_params)
                current_arm = arm
            plan = spec.plan_at(depth)
            key = f"{arm}@{depth}"
            correct = 0
            n = 0
            activation_totals: dict[str, int] = {}
            block_start = time.monotonic()
            for task in eval_tasks:
                if time.monotonic() > deadline:
                    break
                prompt_ids = tokenizer.encode(
                    tokenizer.apply_chat_template(
                        [{"role": "user", "content": task.prompt}],
                        add_generation_prompt=True,
                        tokenize=False,
                    ) + bridge
                )
                gen = _decode(
                    model, tokenizer, prompt_ids, plan,
                    max_tokens=args.max_answer_tokens, eos=eos,
                    activation_totals=activation_totals,
                )
                try:
                    ok = _graded_correct(gen, _expected_payload(task))
                except (ValueError, KeyError, IndexError):
                    ok = False
                correct += int(ok)
                n += 1
                if envelope is not None:
                    envelope.reclaim(force=True)
            report["accuracy"][key] = {
                "correct": correct, "n": n,
                "acc": round(correct / n, 4) if n else None,
            }
            report["block_seconds"][key] = round(time.monotonic() - block_start, 1)
            # Treatment-fired proof. The first gate run decoded every arm
            # with the adapter dark (no scope entered) and reported a
            # base-vs-base comparison as a verdict; this makes that failure
            # loud instead of silent. Zero adapter calls in ANY decoded
            # block means the instrument is broken, not that the answer is
            # "no effect" — the off arm carries zeroed factors precisely so
            # it can run under the same scope.
            report.setdefault("adapter_activation", {})[key] = dict(activation_totals)
            if n and not activation_totals.get("calls"):
                report["error"] = f"adapter scope never fired during {key}"
                report["stop_reason"] = "instrument_dark_adapter"
                write_report()
                print(
                    f"FATAL: adapter scope never fired during {key} — "
                    "the arms would compare base against base",
                    flush=True,
                )
                return 2
            # A non-zero call count is not proof the treatment ran. It is
            # proof that at least one projection ran. If any wrapped site
            # stayed dark, the arms differ by a fraction of the adapter and
            # the comparison measures something nobody designed.
            unfired = sorted(
                set(expected_sites) - set(activation_totals.get("applied_sites") or {})
            )
            report.setdefault("unfired_sites", {})[key] = unfired
            if n and expected_sites and unfired:
                report["error"] = (
                    f"{len(unfired)} of {len(expected_sites)} adapted sites "
                    f"never fired during {key}"
                )
                report["stop_reason"] = "instrument_partial_adapter"
                write_report()
                print(
                    f"FATAL: {len(unfired)} of {len(expected_sites)} adapted "
                    f"sites never fired during {key} — the treatment is partial; "
                    f"first dark site: {unfired[0]}",
                    flush=True,
                )
                return 2
            write_report()
            print(
                f"[{key}] acc={correct}/{n} "
                f"({report['block_seconds'][key]}s"
                f"{', ' + str(round(report['block_seconds'][key]/max(n,1),2)) + 's/task' if n else ''})",
                flush=True,
            )
            if args.calibrate and n:
                per_task = report["block_seconds"][key] / n
                full_n = len([t for t in curriculum.task_battery(families, [args.task_depth], args.per_cell, seed=args.eval_seed) if t.prompt not in train_prompts])
                blocks = len(block_order)
                eta_min = per_task * full_n * blocks / 60.0
                report.setdefault("calibration", {})[key] = {
                    "per_task_s": round(per_task, 2),
                    "projected_full_min": round(eta_min, 1),
                }

    # Verdict — only when the decisive arms completed.
    acc = report["accuracy"]

    def a(k):
        row = acc.get(k) or {}
        return row.get("acc")

    verdict = {}
    if a("off@1") is not None and a(f"on@{max(depths)}") is not None:
        verdict["trained_depth_beats_vanilla"] = a(f"on@{max(depths)}") > a("off@1")
    if a("on@1") is not None and a(f"on@{max(depths)}") is not None:
        verdict["depth_helps_accuracy"] = a(f"on@{max(depths)}") > a("on@1")
    for d in depths:
        if a(f"on@{d}") is not None and a(f"off@{d}") is not None:
            verdict.setdefault("training_helps_by_depth", {})[str(d)] = a(f"on@{d}") > a(f"off@{d}")
    report["verdict"] = verdict
    report["partial"] = report.get("stop_reason") == "wall_clock"
    report["finished_at"] = time.time()
    write_report()
    print(f"verdict: {json.dumps(verdict)}", flush=True)
    print(f"{out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
