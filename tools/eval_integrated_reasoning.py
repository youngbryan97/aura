#!/usr/bin/env python3
"""Run the CP238 model-local retrieval/depth diagnostic.

This uses planted fixture retrieval and direct intrinsic recurrence. It does
not exercise Aura's ordinary desktop pipeline or measure live retrieval.
The scalar-prefill generation protocol is retained for historical comparison;
new reports identify the versioned public-answer and paired-task grader.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from core.learning.integrated_reasoning_eval import (  # noqa: E402
    FixtureRetrieval,
    assert_base_recall_guard,
    build_knowledge_tasks,
    run_factorial,
)
from core.learning.intrinsic_recurrence import (  # noqa: E402
    RecurrentDepthPlan,
    make_recurrent_caches,
    recurrent_hidden_states,
)
from core.runtime.mlx_memory_guard import mlx_memory_envelope  # noqa: E402

INTEGRATED_RUN_SCHEMA = "aura.integrated_reasoning_run.v2"
BRIDGE = "\n\nFINAL_ANSWER: "


def _head_logits(model, hidden):
    if getattr(model, "lm_head", None) is not None:
        return model.lm_head(hidden)
    return model.model.embed_tokens.as_linear(hidden)


def make_solver(model, tokenizer, *, prelude_end, coda_start, max_tokens, envelope):
    """A solver that reasons with retrieved context at a chosen depth.

    ``context`` passages are prepended to the prompt (retrieval-on) or empty
    (retrieval-off). ``depth`` sets the intrinsic recurrence iterations, so
    depth is applied to the answer's own computation.
    """
    import mlx.core as mx

    from core.brain.llm.chat_format import split_native_thinking_generation
    from core.brain.llm.latent_cortex.answer_contract import is_contract_complete

    def solve(prompt: str, context: list[str], depth: int) -> str:
        blocks = []
        if context:
            blocks.append("Known facts:\n" + "\n".join(f"- {c}" for c in context))
        blocks.append(prompt)
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": "\n\n".join(blocks)}],
            add_generation_prompt=True, tokenize=False,
        )
        native_thinking = rendered.rstrip().endswith("<think>")
        ids = tokenizer.encode(rendered + BRIDGE)
        plan = RecurrentDepthPlan(
            prelude_end=prelude_end, coda_start=coda_start,
            iterations=depth, renormalize=True,
        )
        caches = make_recurrent_caches(model, plan)
        hidden, _ = recurrent_hidden_states(model, mx.array([ids]), plan, caches=caches)
        token = int(mx.argmax(_head_logits(model, hidden)[0, -1]))
        eos = tokenizer.eos_token_id
        generated: list[int] = []
        public = ""
        for step in range(max_tokens):
            if token == eos:
                break
            generated.append(token)
            raw = tokenizer.decode(generated, skip_special_tokens=False)
            channels = split_native_thinking_generation(
                raw, native_thinking=native_thinking or raw.lstrip().startswith("<think>"))
            public = channels.surface if channels.boundary_closed else ""
            if channels.boundary_closed and (
                "}" in public or is_contract_complete(public) or "\n" in public.lstrip()
            ):
                break
            hidden, _ = recurrent_hidden_states(
                model, mx.array([[token]]), plan, caches=caches
            )
            token = int(mx.argmax(_head_logits(model, hidden)[0, -1]))
            if envelope is not None and step % 16 == 15:
                envelope.reclaim(force=True)
        if envelope is not None:
            envelope.reclaim(force=True)
        return public

    return solve


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", default="", help="optional adapter dir")
    parser.add_argument("--out", required=True)
    parser.add_argument("--families", default="transitive_chain,conflicting_sources")
    parser.add_argument("--hops", default="2,4")
    parser.add_argument("--per-cell", type=int, default=8)
    parser.add_argument("--depths", default="1,2,4")
    parser.add_argument("--prelude-end", type=int, default=16)
    parser.add_argument("--coda-start", type=int, default=48)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--memory-fraction", type=float, default=0.5)
    args = parser.parse_args()
    output = Path(args.out).expanduser().absolute()
    if output.exists():
        raise FileExistsError(output)

    families = [f.strip() for f in args.families.split(",") if f.strip()]
    hops = [int(h) for h in args.hops.split(",") if h.strip()]
    depths = tuple(int(d) for d in args.depths.split(",") if d.strip())
    tasks = build_knowledge_tasks(
        families=families, hops=hops, per_cell=args.per_cell, seed=args.seed
    )
    source = FixtureRetrieval()
    for task in tasks:
        source.plant(task)
    print(f"[tasks] {len(tasks)} knowledge-gated tasks", flush=True)
    started = time.time()

    from mlx_lm import load
    from core.runtime.model_lane_control import standalone_model_lane

    with standalone_model_lane(
        owner_id=f"eval-integrated-reasoning:{Path(args.out).name}",
        model_path=args.model,
        purpose="evaluation",
        preemptible=False,
        metadata={"tool": "eval_integrated_reasoning", "operator_launched": True},
    ), mlx_memory_envelope(fraction=args.memory_fraction) as envelope:
        print(f"[envelope] {envelope.to_receipt()}", flush=True)
        model, tokenizer = load(args.model)
        if args.adapter:
            adapter_file = Path(args.adapter) / "grpo_adapters.safetensors"
            if adapter_file.exists():
                model.load_weights(str(adapter_file), strict=False)
                print(f"[adapter] loaded {adapter_file}", flush=True)
        solver = make_solver(
            model, tokenizer,
            prelude_end=args.prelude_end, coda_start=args.coda_start,
            max_tokens=args.max_tokens, envelope=envelope,
        )

        # Keep baseline successes in the cohort so regressions remain measurable.
        guard = assert_base_recall_guard(tasks, solver, depth=max(depths))
        print(f"[baseline] correct_without_retrieval={len(guard['answered_from_memory'])}", flush=True)
        report = run_factorial(tasks, source, solver, depths=depths)

    report.update({
        "factorial_schema": report["schema"],
        "schema": INTEGRATED_RUN_SCHEMA,
        "model": args.model,
        "adapter": args.adapter or None,
        "base_recall_guard": guard,
        "tasks_after_guard": len(tasks),
        "task_selection_policy": "all_declared_tasks_no_observed_outcome_filter",
        "elapsed_minutes": round((time.time() - started) / 60.0, 2),
    })
    from core.governance_context import local_internal_governed_scope
    from core.runtime.atomic_writer import atomic_write_bytes_if_absent

    with local_internal_governed_scope("integrated-reasoning-evaluation"):
        if not atomic_write_bytes_if_absent(output,
                (json.dumps(report, indent=2, allow_nan=False) + "\n").encode("utf-8"), mode=0o400):
            raise FileExistsError(output)
    v = report["verdicts"]
    print(
        f"[verdict] retrieval_causal={v['retrieval_is_causal']} "
        f"recurrence_helps={v['recurrence_helps']} "
        f"both_required={v['both_required']}",
        flush=True,
    )
    print(f"[accuracy] {report['accuracy']}", flush=True)
    print(f"[receipt] {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
