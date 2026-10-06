#!/usr/bin/env python3
"""Her freely decoded public answers, with and without what her program reader found.

G05 asks whether an internal gain reaches the answer a person reads. The
internal gain here is G03's frozen reader (v12): it turns a request into a
typed program, and the executor runs it. Three arms, each a free, greedy
decode by her own 27B through her own chat template, native thinking on, at
the medium reasoning effort she serves with:

* ``ordinary``: the request alone;
* ``assisted``: the request, then the reader's program and its executed
  values as runtime evidence, the way her runtime gives any turn evidence;
* ``sham``: the same evidence form taken from another request in the same
  stratum, so a model that copies evidence blindly is caught.

The evidence is data, not instructions: steps, values and a result, with no
word about how to answer. The answer is the last exact integer in her public
reply (``parse_integral_numeric_claim``); her private reasoning is never
read. Every decode's tokens, termination and wall time are kept so compute is
reported per arm, and an answer that stopped at the token budget is counted
as such rather than forgiven.

Usage:
    run_g05_public_answers.py --features DIR --output DIR [--limit N]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ARMS = ("ordinary", "assisted", "sham")


def _value(value: Any) -> str:
    return "[" + ", ".join(map(str, value)) + "]" if isinstance(value, tuple) else str(value)


def reading(program: Any, inputs: tuple, values: list) -> str:
    """The reader's program as steps over the request's own values."""
    n_inputs = len(inputs)
    lines = ["Semantic program reader:"]
    for ordinal, instruction in enumerate(program.instructions):
        args = ", ".join(
            _value(inputs[a]) if a < n_inputs else f"r{a - n_inputs + 1}" for a in instruction.args
        )
        lines.append(f"  r{ordinal + 1} = {instruction.op}({args}) = {_value(values[n_inputs + ordinal])}")
    lines.append(f"  result: {_value(values[-1])}")
    return "\n".join(lines)


def reader_evidence(reader: Any, item: Any) -> dict[str, Any]:
    """What the frozen reader makes of one request, chosen without execution."""
    from tools.run_g04_transfer import decode

    outcome, failure = decode(reader, item)
    if outcome is None or outcome.ir is None:
        return {"text": "Semantic program reader: no reading of this request.", "answer": None,
                "failure": failure or (getattr(outcome, "refusal", "") or "refused")}
    program = outcome.ir.to_program()
    values = list(item.public_inputs)
    try:
        for instruction in program.instructions:
            single = type(program)(n_inputs=len(values), instructions=(type(instruction)(
                instruction.op, instruction.args),))
            values.append(single.run(tuple(values)))
    except (ArithmeticError, RuntimeError, TypeError, ValueError, IndexError) as exc:
        return {"text": "Semantic program reader: its program does not run on these values.",
                "answer": None, "failure": f"execution:{type(exc).__name__}"}
    return {"text": reading(program, tuple(item.public_inputs), values), "answer": values[-1],
            "failure": ""}


def messages(arm: str, request: str, evidence: str | None) -> list[dict[str, Any]]:
    from core.utils.injected_blocks import RUNTIME_EVIDENCE_ROLE, stamp_grounding

    turn = [{"role": "user", "content": request}]
    if arm == "ordinary":
        return turn
    block = stamp_grounding({"role": "system", "content": evidence,
                             "metadata": {"type": "semantic_program_reading"}})
    block["role"] = RUNTIME_EVIDENCE_ROLE
    return [*turn, block]


def decode_public(model: Any, tokenizer: Any, conversation: list, *, max_tokens: int) -> dict[str, Any]:
    import mlx.core as mx
    from mlx_lm import stream_generate

    from core.brain.llm.chat_format import render_chat_template, split_native_thinking_generation

    prompt = render_chat_template(tokenizer, conversation, add_generation_prompt=True,
                                  enable_thinking=True, reasoning_effort="medium")
    tokens = [int(t) for t in tokenizer.encode(prompt, add_special_tokens=False)]
    pieces, generated, finish = [], 0, "token_limit"
    began = time.monotonic()
    for response in stream_generate(model, tokenizer, tokens, max_tokens=max_tokens,
                                    sampler=lambda logits: mx.argmax(logits, axis=-1)):
        pieces.append(str(response.text or ""))
        if response.finish_reason != "stop":
            generated += 1
        if response.finish_reason:
            finish = str(response.finish_reason)
    raw = "".join(pieces)
    channels = split_native_thinking_generation(raw, native_thinking=True)
    return {
        "public_text": channels.surface,
        "raw_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "prompt_tokens": len(tokens),
        "generated_tokens": generated,
        "termination": finish if channels.boundary_closed else "native_thinking_incomplete",
        "seconds": round(time.monotonic() - began, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path,
                        default=Path("~/.aura/models/Aura-Qwen3.8-27B-persona-crsm-7f6a2e83f73f5eef9d15"))
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--limit", type=int, default=0, help="the first N tasks only (a pilot)")
    parser.add_argument("--order-seed", type=int, default=0)
    parser.add_argument("--arm-order", type=Path, help="a plan's arm order by task id")
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load

    from core.learning.semantic_operation_peaks import peak_recognition_transducer_from_dict
    from core.learning.semantic_program_campaign import training_examples_from_feature_bundle
    from core.learning.semantic_program_compositional_transducer import (
        compositional_semantic_program_transducer_from_dict as restore,
    )
    from core.learning.semantic_program_feature_materialization import (
        load_standard_semantic_feature_bundle,
        rebuild_semantic_feature_selection,
    )
    from core.learning.semantic_program_ordinary_baseline import parse_integral_numeric_claim
    from core.runtime.model_lane_control import standalone_model_lane
    from tools import g04_transfer_protocol as protocol
    from tools.run_semantic_peak_recognition import _write_once

    incumbent = restore(json.loads(protocol.INCUMBENT.read_text(encoding="utf-8")))
    reader = peak_recognition_transducer_from_dict(
        json.loads(protocol.CANDIDATE.read_text(encoding="utf-8")), restore_base=restore)
    features = args.features.expanduser()
    bundle = load_standard_semantic_feature_bundle(features)
    from dataclasses import replace

    items = training_examples_from_feature_bundle(
        bundle, required_splits=frozenset(str(e.metadata["split"]) for e in bundle.examples))
    items = [replace(i, ir=replace(i.ir, model_basis_receipt_sha256=incumbent.model_basis_sha256))
             for i in items]
    _config, examples = rebuild_semantic_feature_selection(bundle.manifest)
    by_sha = {hashlib.sha256(e.source_text.encode("utf-8")).hexdigest(): e for e in examples}
    tasks = sorted(items, key=lambda item: item.ir.source_text_sha256)
    if args.limit:
        tasks = tasks[: args.limit]
    output = args.output.expanduser()

    evidence: dict[str, dict[str, Any]] = {}
    for item in tasks:
        path = output / "evidence" / f"{item.ir.source_text_sha256}.json"
        evidence[item.ir.source_text_sha256] = (
            json.loads(path.read_text()) if path.exists()
            else _write_once(path, reader_evidence(reader, item))
        )
    # The sham is the next request's reading within the same stratum.
    sham: dict[str, str] = {}
    by_stratum: dict[str, list[str]] = {}
    for item in tasks:
        by_stratum.setdefault(item.construction_id.split(":")[0], []).append(item.ir.source_text_sha256)
    for shas in by_stratum.values():
        for index, sha in enumerate(shas):
            sham[sha] = evidence[shas[(index + 1) % len(shas)]]["text"]

    planned = json.loads(args.arm_order.read_text()) if args.arm_order else {}
    rng = random.Random(args.order_seed)
    model_path = args.model.expanduser().resolve(strict=True)
    with standalone_model_lane(owner_id=f"g05-public-answers:{output.name}", model_path=str(model_path),
                               purpose="evaluation", preemptible=False, require_exclusive=True,
                               allow_owner_eviction=True, metadata={"tool": Path(__file__).name}):
        model, tokenizer = load(str(model_path))
        try:
            for count, item in enumerate(tasks, 1):
                sha = item.ir.source_text_sha256
                example = by_sha[sha]
                expected = example.program.run(example.inputs)
                order = planned.get(example.example_id) or rng.sample(ARMS, len(ARMS))
                for arm in order:
                    path = output / "rows" / arm / f"{sha}.json"
                    if path.exists():
                        continue
                    shown = {"ordinary": None, "assisted": evidence[sha]["text"], "sham": sham[sha]}[arm]
                    decoded = decode_public(model, tokenizer, messages(arm, example.source_text, shown),
                                            max_tokens=args.max_tokens)
                    parsed = parse_integral_numeric_claim(decoded["public_text"])
                    _write_once(path, {
                        "task_id": example.example_id, "source_sha256": sha, "arm": arm,
                        "construction_id": example.construction_id, "expected": expected,
                        "evidence_sha256": hashlib.sha256((shown or "").encode()).hexdigest(),
                        "reader_answer": evidence[sha]["answer"] if arm == "assisted" else None,
                        "parsed_integer": parsed, "answer_exact": parsed == expected, **decoded,
                    })
                print(json.dumps({"done": count, "of": len(tasks)}), flush=True)
        finally:
            del model, tokenizer
            mx.synchronize()
            mx.clear_cache()

    summary: dict[str, Any] = {"tasks": len(tasks), "arms": {}}
    for arm in ARMS:
        rows = [json.loads(p.read_text()) for p in sorted((output / "rows" / arm).glob("*.json"))]
        summary["arms"][arm] = {
            "decodes": len(rows),
            "exact": sum(r["answer_exact"] for r in rows),
            "terminations": {t: sum(r["termination"] == t for r in rows)
                             for t in sorted({r["termination"] for r in rows})},
            "generated_tokens": sum(r["generated_tokens"] for r in rows),
            "seconds": round(sum(r["seconds"] for r in rows), 1),
        }
    summary["reader_correct"] = sum(
        evidence[i.ir.source_text_sha256]["answer"] == by_sha[i.ir.source_text_sha256].program.run(
            by_sha[i.ir.source_text_sha256].inputs) for i in tasks)
    (output / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
