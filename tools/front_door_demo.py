#!/usr/bin/env python3
"""front_door_demo.py — five load-bearing proofs, one command, no theater.

The front door for a stranger (or a skeptical reviewer): each proof
exercises the REAL mechanism through its public seam and reports an honest
verdict. Nothing here is mocked-to-pass; four proofs are the named contract
tests that gate the suite, the fifth drives a real model.

  1. memory → decision      one-shot episodic binding, ranked TF-IDF recall,
                            and unity state gating the Will's live decision.
  2. amplifier vs baseline  verifier-filtered self-consistency (5 draws) vs
                            ONE draw from the same distribution on the
                            sealed heldout battery, REAL 1.5B weights
                            (--with-model; skipped otherwise). The verdict
                            is whatever the measurement says — PROVEN,
                            HELD, or REFUTED-at-this-scale; a refuted gain
                            claim is a successful honest measurement.
  3. System-2 catches a bad plan
                            the courtroom rejects a verifier-failed answer
                            instead of delivering it.
  4. governed + receipted action
                            actions demand expectations, verdicts land as
                            receipts, failures feed back into planning.
  5. unsafe improvement refused
                            generated solver candidates with side effects
                            or unknown handlers are demoted/rejected;
                            promotion audits stay hard gates.

Pytest proofs run through the governed managed-command lane (no raw
subprocess). Exit 0 iff every executed proof passes. A JSON report lands
under artifacts/front_door/.

Usage:
  python tools/front_door_demo.py                 # offline proofs (fast)
  python tools/front_door_demo.py --with-model    # + real-model amplifier proof
  python tools/front_door_demo.py --with-model --model <dir-or-hf-id>
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.runtime.model_lane_control import standalone_model_lane  # noqa: E402

DEFAULT_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
LOCAL_MODEL = REPO_ROOT / "models" / "Qwen2.5-1.5B-Instruct-4bit"
AMPLIFIER_SEED = 3000  # disjoint from training (<1000) and specialist (2000+) seeds
AMPLIFIER_SIZE = 16
AMPLIFIER_TIME_BUDGET_S = 45.0


@dataclass
class Proof:
    number: int
    title: str
    mechanism: str
    verdict: str = "PENDING"  # PROVEN | FAILED | SKIPPED
    detail: str = ""
    seconds: float = 0.0
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "number": self.number,
            "title": self.title,
            "mechanism": self.mechanism,
            "verdict": self.verdict,
            "detail": self.detail,
            "seconds": round(self.seconds, 1),
            "evidence": self.evidence,
        }


def _run_pytest_proof(proof: Proof, selectors: list[str], *, timeout_s: float = 300.0) -> None:
    from core.tasks.managed_command import run_project_pytest

    started = time.time()
    failures: list[str] = []
    for selector in selectors:
        result = run_project_pytest(selector, timeout_s=timeout_s)
        tail = (result.stdout or result.stderr or "").strip().splitlines()
        proof.evidence.append(f"{selector}: rc={result.returncode}")
        if tail:
            proof.evidence.append(f"  {tail[-1][:160]}")
        if result.returncode != 0:
            failures.append(selector)
    proof.seconds = time.time() - started
    if failures:
        proof.verdict = "FAILED"
        proof.detail = f"failing selector(s): {', '.join(failures)}"
    else:
        proof.verdict = "PROVEN"
        proof.detail = f"{len(selectors)} contract selector(s) green"


def _resolve_model(model_arg: str) -> str:
    if model_arg:
        return model_arg
    if LOCAL_MODEL.exists():
        return str(LOCAL_MODEL)
    return DEFAULT_MODEL


def _run_amplifier_proof(proof: Proof, model_arg: str) -> None:
    """Arm A: one greedy sample. Arm B: ReasoningAmplifierV2 over the same
    generate fn. Same sealed battery, same weights — the delta is the
    mechanism."""
    import asyncio

    from core.learning.heldout_battery import BatterySpec, generate_battery, grade_battery

    started = time.time()
    model_id = _resolve_model(model_arg)
    proof.evidence.append(f"model: {model_id}")

    from mlx_lm import generate, load

    model, tokenizer = load(model_id)

    def _prompt(user_prompt: str) -> str:
        apply = getattr(tokenizer, "apply_chat_template", None)
        if callable(apply):
            try:
                return apply(
                    [{"role": "user", "content": user_prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except (TypeError, ValueError):
                pass
        return user_prompt

    def _generate(prompt_text: str, *, temp: float = 0.0, max_tokens: int = 256) -> str:
        kwargs: dict = {"max_tokens": max_tokens, "verbose": False}
        try:
            from mlx_lm.sample_utils import make_sampler

            kwargs["sampler"] = make_sampler(temp=temp)
        except ImportError:
            pass
        try:
            out = generate(model, tokenizer, prompt=_prompt(prompt_text), **kwargs)
        except TypeError:
            kwargs.pop("sampler", None)
            out = generate(model, tokenizer, prompt=_prompt(prompt_text), **kwargs)
        return out if isinstance(out, str) else str(out)

    spec = BatterySpec(seed=AMPLIFIER_SEED, size=AMPLIFIER_SIZE)
    tasks = generate_battery(spec)
    proof.evidence.append(f"battery: {spec.battery_id()} ({len(tasks)} tasks)")

    try:
        import mlx.core as mx

        mx.random.seed(AMPLIFIER_SEED)
    except ImportError:
        pass

    # Reference — deterministic greedy decode (a DIFFERENT decoding
    # strategy, shown for context, not the mechanism comparison).
    greedy_responses = {t.task_id: _generate(t.prompt) for t in tasks}
    greedy = grade_battery(spec, tasks, greedy_responses)

    # Arm A — baseline: ONE draw from the same sampling distribution the
    # amplifier draws from. The mechanism claim under test is 'verifier-
    # filtered majority over N draws beats one draw', apples to apples.
    baseline_responses = {t.task_id: _generate(t.prompt, temp=0.7) for t in tasks}
    baseline = grade_battery(spec, tasks, baseline_responses)

    # Arm B — amplifier: verifier-filtered self-consistency over the SAME
    # generate fn (sampled candidates + exact verifiers + calibration).
    from core.brain.reasoning_amplifier_v2 import (
        AmplificationRequest,
        ReasoningAmplifierV2,
    )

    # GenerateFn contract is POSITIONAL (prompt, temperature). One resident
    # model: serialize actual decoding under a lock — the amplifier gathers
    # candidates concurrently and MLX is not thread-safe.
    decode_lock = threading.Lock()

    def _locked_generate(prompt_text: str, temp: float) -> str:
        with decode_lock:
            return _generate(prompt_text, temp=temp)

    async def _gen_async(prompt_text: str, temperature: float = 0.7) -> str:
        return await asyncio.to_thread(_locked_generate, prompt_text, float(temperature))

    amplifier = ReasoningAmplifierV2(_gen_async)

    # Battery domains map to the registry's coarse task types — the same
    # types the live system hands the amplifier — so the EXACT verifiers
    # (math engine, code engine) engage instead of only the generic logic
    # pass. Without this the filter cannot filter and sampled majority
    # loses to greedy.
    math_domains = {
        "arithmetic_chain",
        "modular",
        "linear_equation",
        "sequence",
        "unit_conversion",
        "comparison",
    }

    def _task_type(domain: str) -> str:
        if domain in math_domains:
            return "math"
        if domain == "program_output":
            return "code"
        return domain

    receipt_stats: dict[str, int] = {}

    async def _amplified(task) -> tuple[str, str]:
        request = AmplificationRequest(
            objective=task.prompt,
            task_type=_task_type(task.domain),
            time_budget_s=AMPLIFIER_TIME_BUDGET_S,
            sample_budget=5,
            # Benchmark hygiene: the solved-cache must not serve answers
            # cached by earlier runs — every task is amplified fresh.
            context={"skip_cache": True},
        )
        answer = await amplifier.amplify(request)
        receipt = getattr(answer, "receipt", None)
        if receipt is not None:
            key = (
                f"mode={getattr(receipt, 'mode', '?')}"
                f"/strategy={getattr(receipt, 'strategy_used', '?')}"
                f"/verifiers={','.join(getattr(receipt, 'verifiers_run', []) or [])}"
                f"/n={getattr(receipt, 'num_candidates', 0)}"
            )
            receipt_stats[key] = receipt_stats.get(key, 0) + 1
        return task.task_id, str(getattr(answer, "answer", answer) or "")

    async def _run_all() -> dict[str, str]:
        results: dict[str, str] = {}
        for task in tasks:  # sequential: one model, bounded memory
            task_id, answer = await _amplified(task)
            results[task_id] = answer
        return results

    amplified_responses = asyncio.run(_run_all())
    amplified = grade_battery(spec, tasks, amplified_responses)

    proof.seconds = time.time() - started
    proof.evidence.append(
        f"baseline (1 draw @ temp 0.7): {baseline.correct}/{baseline.total}"
    )
    proof.evidence.append(
        f"amplified (5 verified draws): {amplified.correct}/{amplified.total}"
    )
    proof.evidence.append(
        f"reference (greedy decode): {greedy.correct}/{greedy.total}"
    )
    for key, count in sorted(receipt_stats.items()):
        proof.evidence.append(f"receipts: {count}x {key}")

    if amplified.correct > baseline.correct:
        proof.verdict = "PROVEN"
        proof.detail = (
            f"amplifier {amplified.correct}/{amplified.total} > "
            f"single-draw baseline {baseline.correct}/{baseline.total} "
            f"on {spec.battery_id()}"
        )
    elif amplified.correct == baseline.correct:
        proof.verdict = "HELD"
        proof.detail = (
            f"amplifier matched the single-draw baseline "
            f"{baseline.correct}/{baseline.total} — no gain claimable on "
            "this battery/seed at this size"
        )
    else:
        # The demo's own claim is honest measurement, and a refuted gain
        # claim IS a successful measurement (house precedent: the DNU
        # baseline-fairness audit, the arithmetic-chain gate that refused
        # to claim gain over a 1.000 base). Recorded loudly, not hidden.
        proof.verdict = "REFUTED"
        proof.detail = (
            f"amplifier {amplified.correct}/{amplified.total} < "
            f"single-draw baseline {baseline.correct}/{baseline.total} — "
            "no amplification gain at this scale on this battery; the "
            "boundary is the finding"
        )


def build_proofs() -> list[Proof]:
    return [
        Proof(
            1,
            "memory → decision",
            "one-shot episodic bind + partial-cue completion, TF-IDF-ranked recall, "
            "and low unity state blocking a consequential Will decision",
        ),
        Proof(
            2,
            "amplifier vs baseline (real 1.5B)",
            "verifier-filtered self-consistency (5 draws) vs one draw from the same "
            "distribution, same weights, sealed heldout battery; greedy shown as reference",
        ),
        Proof(
            3,
            "System-2 catches a bad plan",
            "the courtroom refuses to deliver a verifier-failed answer",
        ),
        Proof(
            4,
            "governed + receipted action",
            "actions carry expectations, verdicts persist as receipts, "
            "expectation failures feed back into planning",
        ),
        Proof(
            5,
            "unsafe improvement refused",
            "solver candidates with side effects or unknown handlers are demoted; "
            "promotion audits stay hard",
        ),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-model",
        action="store_true",
        help="run the real-model amplifier proof (downloads/loads the 1.5B)",
    )
    parser.add_argument("--model", default="", help="model dir or HF id for proof 2")
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "artifacts" / "front_door" / "front_door_report.json"),
    )
    args = parser.parse_args(argv)

    proofs = build_proofs()
    print("═" * 72)
    print("AURA FRONT-DOOR DEMO — five load-bearing proofs, real mechanisms only")
    print("═" * 72)

    _run_pytest_proof(
        proofs[0],
        [
            "tests/test_memory_retrieval_backbone.py",
            "tests/unity/test_will_unity_gating.py::test_low_unity_blocks_external_tool_action",
        ],
    )

    if args.with_model:
        try:
            model_id = _resolve_model(args.model)
            with standalone_model_lane(
                owner_id="front-door-amplifier",
                model_path=model_id,
                purpose="benchmark",
                metadata={"tool": "front_door_demo"},
            ):
                _run_amplifier_proof(proofs[1], model_id)
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            proofs[1].verdict = "FAILED"
            proofs[1].detail = f"{type(exc).__name__}: {exc}"
    else:
        proofs[1].verdict = "SKIPPED"
        proofs[1].detail = (
            "model-bound; run: python tools/front_door_demo.py --with-model"
        )

    _run_pytest_proof(
        proofs[2],
        ["tests/test_courtroom.py::test_courtroom_honors_failed_verifier"],
    )
    _run_pytest_proof(
        proofs[3],
        [
            "tests/test_action_depth_honesty.py",
            "tests/test_planner_expectation_feedback.py",
        ],
    )
    _run_pytest_proof(
        proofs[4],
        [
            "tests/test_autonomous_rsi_hardening.py",
            "tests/test_memory_retrieval_backbone.py::TestPromotionGateThreshold",
        ],
    )

    print()
    for proof in proofs:
        mark = {
            "PROVEN": "ok",
            "HELD": "warning",
            "REFUTED": "failed",
            "SKIPPED": "skipped ",
            "FAILED": "failed",
        }.get(proof.verdict, "unknown")
        print(f"{mark} [{proof.number}] {proof.title} — {proof.verdict} ({proof.seconds:.1f}s)")
        print(f"     mechanism: {proof.mechanism}")
        print(f"     {proof.detail}")
        for line in proof.evidence:
            print(f"       {line}")
    print()
    print("Deeper evidence: CLAIMS_MATRIX.md (falsification ledger), make demo-learning,")
    print("make triage, docs/RELIABILITY_MATURITY_ROADMAP.md.")

    report = {
        "schema_version": 1,
        "tool": "front_door_demo",
        "created_at": time.time(),
        "with_model": bool(args.with_model),
        "proofs": [p.to_dict() for p in proofs],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Report: {output}")

    # REFUTED is a successful, honest measurement — the demo's own claim.
    # Only a broken proof (FAILED) or a failing contract test flips exit 1.
    executed = [p for p in proofs if p.verdict not in {"SKIPPED"}]
    return 0 if all(p.verdict in {"PROVEN", "HELD", "REFUTED"} for p in executed) else 1


if __name__ == "__main__":
    raise SystemExit(main())
