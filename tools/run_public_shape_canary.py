#!/usr/bin/env python3
"""Measure resident public output shapes separately from exact task answers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
os.environ.setdefault("AURA_LOG_DIR", "/tmp/aura-public-shape-canary")


def load_cases(path):
    cases = json.loads(path.read_text())
    if not isinstance(cases, list) or not cases:
        raise ValueError("canary requires a nonempty case list")
    ids = set()
    for case in cases:
        if not isinstance(case, dict) or set(case) != {"id", "prompt", "shape", "expected"}:
            raise ValueError("each case requires id, prompt, shape and expected")
        if not isinstance(case["id"], str) or not case["id"] or case["id"] in ids:
            raise ValueError("case ids must be nonempty and unique")
        ids.add(case["id"])
        if not isinstance(case["prompt"], str) or not case["prompt"].strip():
            raise ValueError("case prompt must be nonempty")
        if case["shape"] not in {"any", "object", "array"}:
            raise ValueError("unsupported shape")
        if case["shape"] == "object" and not isinstance(case["expected"], dict):
            raise ValueError("object case requires an object expectation")
        if case["shape"] == "array" and not isinstance(case["expected"], list):
            raise ValueError("array case requires an array expectation")
        json.dumps(case, allow_nan=False)
    return cases


def _reject_constant(value):
    raise ValueError(f"nonfinite JSON constant: {value}")


def assess(case, decoded, processor):
    parsed = None
    parse_error = ""
    try:
        parsed = json.loads(decoded.text, parse_constant=_reject_constant)
    except (ValueError, TypeError) as exc:
        parse_error = str(exc)
    shape_ok = not parse_error and (
        case["shape"] == "any"
        or case["shape"] == "object" and isinstance(parsed, dict)
        or case["shape"] == "array" and isinstance(parsed, list)
    )
    exact = bool(shape_ok and json.dumps(parsed, sort_keys=True, ensure_ascii=True) ==
                 json.dumps(case["expected"], sort_keys=True, ensure_ascii=True))
    complete = bool(decoded.boundary_closed and decoded.stop_reason == "eos" and decoded.text)
    enforcing = bool(processor.state["enforcing"] and processor.state["refused"] == 0)
    return {"id": case["id"], "public_text": decoded.text, "channel": decoded.receipt(),
            "parse_error": parse_error, "shape_valid": bool(shape_ok),
            "exact_expected_json": exact, "public_complete": complete,
            "processor_enforcing": enforcing, "processor_refusals": processor.state["refused"],
            "passed": bool(complete and shape_ok and exact and enforcing)}


def write_result(path, payload):
    from core.governance_context import local_internal_governed_scope
    from core.runtime.file_write_gateway import get_file_write_gateway

    with local_internal_governed_scope("evaluation.public_shape_canary", domain="file_write"):
        gateway = get_file_write_gateway()
        gateway.ensure_directory(path.parent, source="evaluation.public_shape_canary")
        gateway.write_text(path, json.dumps(payload, sort_keys=True, allow_nan=False) + "\n",
                           source="evaluation.public_shape_canary")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=3072)
    args = parser.parse_args(argv)
    if args.max_tokens < 1 or args.out.exists():
        parser.error("use a positive allocation and a new output path")
    cases = load_cases(args.cases)
    from core.brain.llm.model_registry import get_active_cortex_spec
    from core.runtime.model_lane_control import standalone_model_lane

    spec = get_active_cortex_spec(force_refresh=True)
    if spec is None:
        raise ValueError("no verified resident model")
    sources = [Path(__file__).resolve(), REPO / "core/brain/llm/public_channel_decode.py",
               REPO / "core/brain/llm/a_shape_the_decoder_enforces.py"]
    payload = {"schema": "aura.public_shape_canary.v1", "qualification": False,
               "model_path": str(spec.model_path), "descriptor_sha256": str(spec.descriptor_sha256),
               "cases": cases, "max_tokens": args.max_tokens,
               "source_sha256": {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sources}, "complete": False, "results": []}
    write_result(args.out, payload)
    with standalone_model_lane(owner_id="public-shape-canary", model_path=str(spec.model_path),
                               purpose="evaluation", preemptible=False,
                               metadata={"tool": "run_public_shape_canary"}):
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.sample_utils import make_sampler
        from core.brain.llm.public_channel_decode import decode_public_sample
        from core.brain.llm.a_shape_the_decoder_enforces import enforce_json
        from core.brain.llm.a_bounded_private_channel import _the_token_that_closes_it

        print(f"loading {spec.model_path}", flush=True)
        model, tokenizer = load(str(spec.model_path))
        started = time.monotonic()
        for case in cases:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": case["prompt"]}], tokenize=False, add_generation_prompt=True)
            native = prompt.rstrip().endswith("<think>")
            closing = _the_token_that_closes_it(tokenizer) if native else None
            if native and closing is None:
                raise ValueError("native private boundary is unsupported")
            processor = enforce_json(tokenizer, after_token=closing, require=case["shape"])
            if processor is None:
                raise ValueError("shape processor unavailable")
            print(f"starting {case['id']}", flush=True)
            def progress(count):
                if count % 64 == 0:
                    print(f"{case['id']} generated_tokens={count}", flush=True)
            decoded = decode_public_sample(model, tokenizer, prompt, max_tokens=args.max_tokens,
                                           sampler=make_sampler(temp=0), progress=progress,
                                           logits_processors=[processor])
            result = assess(case, decoded, processor)
            payload["results"].append(result)
            payload["decode_seconds"] = time.monotonic() - started
            write_result(args.out, payload)
            print(json.dumps(result, sort_keys=True), flush=True)
        del model
        mx.clear_cache()
    payload["complete"] = True
    payload["passed"] = all(row["passed"] for row in payload["results"])
    write_result(args.out, payload)
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
