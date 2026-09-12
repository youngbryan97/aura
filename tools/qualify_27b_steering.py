#!/usr/bin/env python3
"""tools/qualify_27b_steering.py — the evidence a captured vector has not earned.

Capture grants nothing. A regenerated vector that exists, is the right width and
came out of the right checkpoint has still not been shown to do anything, and
that is exactly the state that looks safest and is not. Four things have to hold
before the steering component of a migration contract can move off `deferred`:

  extraction_bound_to_active_descriptor   every vector carries this checkpoint
  causal_ab_vs_matched_noop               steering changes the words, and beats
                                          a norm-matched random direction
  lesion_removes_the_effect               opposing substrate states separate,
                                          and silence sits between them
  no_regression_on_the_control_prompts    forced-choice accuracy is unmoved

The middle three are what `core.consciousness.fusion_probe` already measures,
so they are measured through the same hooks a served turn would use rather than
through a second implementation that could agree with itself.

The hooks are installed directly here. The serving gate refuses to attach them
while the contract says deferred, and the contract cannot stop saying deferred
until somebody measures them, so a measurement path that the gate does not
own is the only way out of that circle. Nothing here serves anything.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

os.environ.setdefault("AURA_LOG_DIR", "/tmp/aura_steering_qualify")

DEFAULT_PLAN = REPO / "artifacts/migration/27b/recovery/steering_plan.json"
DEFAULT_VECTORS = REPO / "training/vectors/cortex-52d313c2"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--vectors", type=Path, default=DEFAULT_VECTORS)
    parser.add_argument("--alphas", default="0.1,0.2")
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO / "artifacts/migration/27b/recovery/steering_evidence.json",
    )
    arguments = parser.parse_args(argv)

    import numpy as np

    plan = json.loads(arguments.plan.read_text(encoding="utf-8"))
    model_path = Path(plan["model_path"])
    target_layers = [int(row["index"]) for row in plan["target_layers"]]
    hidden = int(plan["hidden_size"])

    from core.brain.llm.model_registry import get_active_cortex_spec
    from core.learning.steering_regeneration import authority_errors, may_serve

    spec = get_active_cortex_spec(force_refresh=True)
    if spec is None:
        print("no active cortex", file=sys.stderr)
        return 1
    descriptor = str(spec.descriptor_sha256)

    # ── 1. extraction_bound_to_active_descriptor ──────────────────────────
    files = sorted(arguments.vectors.glob("*_layer*.npz"))
    bound = 0
    stray: list[str] = []
    for path in files:
        payload = np.load(path, allow_pickle=True)
        if str(payload["model_descriptor_sha256"]) == descriptor:
            bound += 1
        else:
            stray.append(path.name)
    extraction_bound = bool(files) and not stray and bound == len(files)
    print(
        f"extraction  {bound} of {len(files)} vectors carry {descriptor[:16]}"
        + (f"; stray: {', '.join(stray[:4])}" if stray else ""),
        flush=True,
    )

    os.environ["AURA_STEERING_DIR"] = str(arguments.vectors)

    # One 27B at a time. The live instance holds ~20GB wired on a 64GB host,
    # so a second checkpoint loaded beside it is what takes the machine down
    # rather than what measures it. The lane is the thing that knows.
    from core.runtime.model_lane_control import standalone_model_lane

    with standalone_model_lane(
        owner_id=f"caa-qualify:{model_path.name}",
        model_path=str(model_path),
        purpose="evaluation",
        preemptible=False,
        metadata={"tool": "qualify_27b_steering"},
    ):
        import mlx.core as mx  # noqa: F401  (imported for its side effect on load)
        from mlx_lm import load

        started = time.time()
        print("loading the checkpoint", flush=True)
        model, tokenizer = load(str(model_path))

        from core.brain.llm.decoder_topology import resolve_language_model
        from core.consciousness.affective_steering import (
            AffectiveSteeringHook,
            SteeringVectorLibrary,
        )

        decoder = resolve_language_model(model)
        blocks = list(getattr(decoder, "layers", None) or getattr(decoder.model, "layers", []))

        library = SteeringVectorLibrary(
            cache_dir=arguments.vectors,
            source_dirs=[arguments.vectors],
            expected_model_identity={"descriptor_sha256": descriptor},
            allow_derivation=False,
        )
        by_layer = library.load_or_derive(model, tokenizer, target_layers, hidden)
        hooks = []
        for layer_index in target_layers:
            vectors = by_layer.get(layer_index) or {}
            if not vectors:
                continue
            hook = AffectiveSteeringHook(blocks[layer_index], layer_index, vectors)
            hook.install()
            hooks.append(hook)
        print(
            f"installed   {len(hooks)} hooks over "
            f"{sum(len(v) for v in by_layer.values())} vectors in "
            f"{time.time() - started:.1f}s",
            flush=True,
        )
        if not hooks:
            print("no hooks installed; nothing to measure", file=sys.stderr)
            return 1

        def set_alpha(value: float) -> None:
            for hook in hooks:
                hook._alpha = float(value)

        from core.consciousness.fusion_probe import measure_fusion

        alphas = tuple(float(a) for a in str(arguments.alphas).split(",") if a.strip())
        certificates = measure_fusion(
            model,
            tokenizer,
            hooks,
            set_alpha,
            model_identity=descriptor,
            model_name=str(model_path),
            alphas=alphas,
            steps=int(arguments.steps),
            runner="tools/qualify_27b_steering.main",
        )
        for certificate in certificates:
            print(
                f"alpha {certificate.alpha:<5} arrives={certificate.arrives} "
                f"beats_noise={certificate.beats_noise} "
                f"carries_content={certificate.carries_content} "
                f"costs_nothing={certificate.costs_nothing} "
                f"| words change on {certificate.prompts_that_change}/{certificate.prompts}, "
                f"states separate {certificate.state_separation:.3f}",
                flush=True,
            )

        holding = [c for c in certificates if c.holds]
        best = min(holding, key=lambda c: c.alpha) if holding else None

        evidence = {
            "schema": "aura.steering.regeneration_evidence.v1",
            "descriptor_fingerprint": plan.get("descriptor_fingerprint"),
            "model_descriptor_sha256": descriptor,
            "measured_at": time.time(),
            "vectors": len(files),
            "hooks": len(hooks),
            "extraction_bound_to_active_descriptor": extraction_bound,
            "causal_ab_vs_matched_noop": bool(best and best.arrives and best.beats_noise),
            "lesion_removes_the_effect": bool(best and best.carries_content),
            "no_regression_on_the_control_prompts": bool(best and best.costs_nothing),
            "alpha": float(best.alpha) if best else 0.0,
            "certificates": [
                {
                    "alpha": c.alpha,
                    "arrives": c.arrives,
                    "beats_noise": c.beats_noise,
                    "carries_content": c.carries_content,
                    "costs_nothing": c.costs_nothing,
                    "prompts": c.prompts,
                    "prompts_that_change": c.prompts_that_change,
                    "state_separation": c.state_separation,
                    "holds": c.holds,
                }
                for c in certificates
            ],
        }
        arguments.out.parent.mkdir(parents=True, exist_ok=True)
        arguments.out.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")

        outstanding = authority_errors(plan, evidence)
        print(f"\nwrote {arguments.out}")
        if may_serve(plan, evidence):
            print("every requirement holds; the steering component can be qualified")
            return 0
        for error in outstanding:
            print(f"  outstanding   {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
