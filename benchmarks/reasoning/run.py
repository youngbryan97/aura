"""CLI entry for the reasoning benchmark.

    python -m benchmarks.reasoning.run                       # deterministic canned candidates
    python -m benchmarks.reasoning.run --live                # answers from a real local MLX model
    python -m benchmarks.reasoning.run --live --model PATH    # pick the model
    python -m benchmarks.reasoning.run --live --router        # use the live app's LLM router
    python -m benchmarks.reasoning.run --out results.json

The deterministic run exercises the truth engines + amplifier against seeded errors
(no model needed) and is suitable for CI regression gating. ``--live`` loads an
actual MLX cortex standalone (no full-app boot) and answers the same objectives
on-device — real model, real tokens, not stubs.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from .harness import ReasoningBenchmark, write_results

#: Terminal-gate thresholds.
#:
#: MIN_PASS_RATE exists because the original gate could be satisfied with
#: zero correct answers: a model that fails every case but marks each one
#: unverified and low-confidence has a perfect catch rate and no false
#: confidence. Self-knowledge is necessary, not sufficient.
MIN_PASS_RATE = 0.60
MIN_HALLUCINATION_CATCH_RATE = 1.0

def _resolve_model(explicit: str) -> str:
    if explicit:
        return explicit
    from core.brain.llm.model_registry import ACTIVE_MODEL, get_runtime_model_path

    model_path = get_runtime_model_path(ACTIVE_MODEL)
    if not Path(model_path).expanduser().is_dir():
        raise FileNotFoundError(
            "The active Cortex is not a local model directory. "
            "Promote a local artifact or pass --model PATH."
        )
    return model_path


async def _mlx_generator(model_path: str):
    """A real on-device generate(prompt, temperature) backed by mlx_lm.

    Generation is serialized with a lock — the amplifier samples candidates in
    parallel, but a single MLX model is not safe to drive from concurrent threads.
    """
    from mlx_lm import generate as mlx_generate
    from mlx_lm import load
    from mlx_lm.sample_utils import make_sampler

    from core.runtime.model_lane_control import (
        acquire_in_process_model_lane,
        run_owned_model_thread_call,
    )

    print(f"loading MLX model: {model_path} …", flush=True)
    lease = await acquire_in_process_model_lane(
        owner_id="reasoning-benchmark",
        model_path=model_path,
        purpose="benchmark",
        preemptible=False,
        metadata={"tool": "benchmarks.reasoning.run"},
    )
    try:
        model, tokenizer = await run_owned_model_thread_call(
            lambda: load(model_path),
            operation_name="reasoning_benchmark_model_load",
        )
    except BaseException:  # noqa: BLE001 - release lease on interrupts and process termination.
        await lease.release(reason="reasoning_benchmark_model_load_failed")
        raise
    lock = asyncio.Lock()

    async def gen(prompt: str, temperature: float) -> str:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False
        )
        sampler = make_sampler(temp=max(0.01, float(temperature)))
        async with lock:
            return await run_owned_model_thread_call(
                lambda: mlx_generate(
                    model,
                    tokenizer,
                    prompt=text,
                    max_tokens=512,
                    sampler=sampler,
                    verbose=False,
                ),
                operation_name="reasoning_benchmark_generate",
            )

    return gen, lease


async def _router_generator():
    from core.runtime import service_access

    router = service_access.resolve_llm_router(default=None)
    if router is None:
        raise RuntimeError("live LLM router unavailable; use --model for a standalone MLX run")

    async def gen(prompt: str, temperature: float) -> str:
        res = await router.think(prompt, mode="FAST", temperature=temperature)
        return res.content if hasattr(res, "content") else str(res or "")

    return gen


async def _main_async(args: argparse.Namespace) -> int:
    generate = None
    model_lease = None
    try:
        if args.live:
            if args.router:
                generate = await _router_generator()
            else:
                generate, model_lease = await _mlx_generator(
                    _resolve_model(args.model)
                )
        result = await ReasoningBenchmark().run(generate=generate)
        print(result.summary())
        for o in result.outcomes:
            flag = "✅" if o.correct else "❌"
            fc = " ⚠️false-confidence" if o.false_confidence else ""
            print(
                f"  {flag} {o.case_id:<12} verified={o.verified!s:<5} "
                f"conf={o.confidence:.2f} lat={o.latency_ms:.0f}ms{fc}"
            )
        if args.out:
            write_results(result, args.out)
            print(f"wrote {args.out}")
        # CP126 (critical): "Completely wrong live model can exit
        # successfully." The gate asked only for a perfect verifier catch
        # rate and a zero false-confidence rate. Both are satisfied by a
        # model that gets EVERY answer wrong, provided each wrong answer is
        # marked unverified with low confidence — a system that knows it is
        # useless passed the benchmark that certifies it is useful.
        #
        # Catching your own failures is necessary and not sufficient. The
        # gate now also requires that some answers were actually right, and
        # that fabrications were caught, with the failing conditions named
        # rather than collapsed into a bare exit code.
        failures: list[str] = []
        if result.n <= 0:
            failures.append("no cases ran")
        if result.verifier_catch_rate < 1.0:
            failures.append(
                f"verifier_catch_rate={result.verifier_catch_rate:.0%} (need 100%)"
            )
        if result.false_confidence_rate > 0.0:
            failures.append(
                f"false_confidence_rate={result.false_confidence_rate:.0%} (need 0%)"
            )
        if result.pass_rate < MIN_PASS_RATE:
            failures.append(
                f"pass_rate={result.pass_rate:.0%} (need >={MIN_PASS_RATE:.0%}) — "
                "knowing you are wrong is not the same as being right"
            )
        if result.hallucination_catch_rate < MIN_HALLUCINATION_CATCH_RATE:
            failures.append(
                f"hallucination_catch_rate={result.hallucination_catch_rate:.0%} "
                f"(need >={MIN_HALLUCINATION_CATCH_RATE:.0%})"
            )
        if failures:
            print("BENCHMARK GATE FAILED:")
            for reason in failures:
                print(f"  - {reason}")
            return 1
        return 0
    finally:
        if model_lease is not None:
            generate = None
            import gc

            import mlx.core as mx

            await asyncio.to_thread(gc.collect)
            try:
                mx.clear_cache()
            except (AttributeError, RuntimeError):
                pass
            await model_lease.release(reason="reasoning_benchmark_finished")


def main() -> int:
    parser = argparse.ArgumentParser(description="Aura reasoning benchmark")
    parser.add_argument("--live", action="store_true", help="answer with a real model instead of canned candidates")
    parser.add_argument("--model", default="", help="MLX model path/id for --live (default: a local Qwen2.5)")
    parser.add_argument("--router", action="store_true", help="use the live app's LLM router instead of standalone MLX")
    parser.add_argument("--out", default="", help="write results JSON to this path")
    args = parser.parse_args()
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    sys.exit(main())
