#!/usr/bin/env python3
"""tools/train_steering_vectors_by_gradient.py — vectors that are trained, not averaged.

Difference-of-means CAA gives every layer a vector pointing along the same
axis. Measured on the 27B at n=24, across four adjacent layers, sixteen, and six
chosen for being least alike, the shuffled-layer control kept 43%, 84% and 70%
of the effect — and removing the shared axis removes the effect entirely. So
the intervention has no layer-assignment specificity to demonstrate: an average
cannot give each layer a different job, because averaging is what makes them
the same.

Training can. Each layer's vector is a parameter, the objective is the
behaviour the campaign scores, and a penalty on cross-layer similarity makes
"the right vector at the right layer" the thing being optimised rather than
something hoped for afterwards. Permuting trained vectors then has to cost
something, because each one was fitted where it sits.

The objective is preference, not imitation. For each pair of contrast
statements the model already carries -- the same ones difference-of-means uses
-- the steered model must raise the log-probability of the positive
continuation ABOVE the negative one by more than the unsteered model does:

    margin(v)  = log p(positive | carrier, steered) - log p(negative | ...)
    loss       = -logsigmoid(beta * (margin(v) - margin(0))) + lam * overlap

`margin(0)` is measured once per pair and held, so the loss cannot be reduced
by a vector that simply makes everything more likely. `overlap` is the mean
squared cosine between every pair of layers.

Nothing here reaches a served prompt. The statements become a direction and
then stop existing.
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

os.environ.setdefault("AURA_LOG_DIR", "/tmp/aura_steering_train")

DEFAULT_PLAN = REPO / "artifacts/migration/27b/recovery/steering_plan_wide.json"

#: Carriers for the contrast, held apart from the campaign's scored tasks.
EXTRACTION_CARRIERS = (
    "Tell me where the work stands.",
    "What is on your mind about this project?",
    "Give me your read on the last change.",
    "How is the current task going?",
    "Say something about where your attention is.",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--model", type=Path, default=None, help="defaults to the plan's")
    parser.add_argument(
        "--init",
        type=Path,
        default=REPO / "training/vectors/generation-position-wide",
        help="vectors to start from; trained from there rather than from noise",
    )
    parser.add_argument(
        "--out", type=Path, default=REPO / "training/vectors/gradient-trained"
    )
    parser.add_argument("--steps", type=int, default=60, help="optimiser steps per dimension")
    parser.add_argument(
        "--pairs",
        type=int,
        default=16,
        help="preference pairs kept per dimension. Every one costs two forward "
        "passes in the baseline measurement, on the ops path, and a ten-step "
        "run samples four at a time -- so carrying sixty of them spends most "
        "of the run measuring pairs it never uses.",
    )
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--beta", type=float, default=1.0, help="preference temperature")
    parser.add_argument(
        "--orthogonality",
        type=float,
        default=1.0,
        help="weight on the mean squared cosine between layers. The whole "
        "reason for training rather than averaging.",
    )
    parser.add_argument("--alpha", type=float, default=0.2, help="training injection strength")
    parser.add_argument(
        "--no-regression",
        type=float,
        default=2.0,
        help="weight on keeping the model's preference for the right answer. "
        "The fusion certificate refuses a direction that costs more of it than "
        "a random one does, so it is an objective rather than an afterthought.",
    )
    parser.add_argument("--dimensions", default="", help="comma-separated subset")
    arguments = parser.parse_args(argv)

    import mlx.core as mx
    import mlx.optimizers as optim
    import numpy as np

    plan = json.loads(arguments.plan.read_text(encoding="utf-8"))
    model_path = Path(arguments.model or plan["model_path"])
    target_layers = sorted(int(row["index"]) for row in plan["target_layers"])
    hidden = int(plan["hidden_size"])

    from core.brain.llm.model_registry import get_active_cortex_spec
    from core.consciousness.affective_steering import AFFECTIVE_DIMENSIONS

    spec = get_active_cortex_spec(force_refresh=True)
    descriptor = str(spec.descriptor_sha256) if spec is not None else ""

    wanted = {name.strip() for name in arguments.dimensions.split(",") if name.strip()}
    dimensions = [
        d for d in AFFECTIVE_DIMENSIONS if not wanted or str(d["key"]) in wanted
    ]

    from core.runtime.model_lane_control import standalone_model_lane

    with standalone_model_lane(
        owner_id=f"steering-training:{model_path.name}",
        model_path=str(model_path),
        purpose="training",
        preemptible=False,
        metadata={"tool": "train_steering_vectors_by_gradient"},
    ):
        from mlx_lm import load

        started = time.time()
        print(f"loading {model_path.name}", flush=True)
        model, tokenizer = load(str(model_path))
        # The one flag the whole build turned on. This checkpoint's linear
        # attention is a Metal kernel with no backward pass, and the layer
        # already chooses between it and the pure-ops recurrence on
        # `use_kernel=not self.training`. Without this a gradient cannot reach
        # any vector injected below layer 63: `[Primitive::vjp] Not implemented
        # for CustomKernel`. The ops path is a Python loop over timesteps, so
        # this is slow and it is differentiable.
        model.train()

        from core.brain.llm.decoder_topology import resolve_language_model

        blocks = list(getattr(resolve_language_model(model), "layers", []))
        if not blocks:
            print("no decoder layers", file=sys.stderr)
            return 1

        vectors = {"v": mx.zeros((len(target_layers), hidden))}
        index_of = {layer: position for position, layer in enumerate(target_layers)}
        originals: list[tuple[object, type]] = []

        def steered_class(original: type, layer: int) -> type:
            position = index_of[layer]

            class Steered(original):  # type: ignore[misc, valid-type]
                __module__ = original.__module__

                def __call__(self, x, *args, **kwargs):
                    result = super().__call__(x, *args, **kwargs)
                    h = result[0] if isinstance(result, tuple) else result
                    added = vectors["v"][position]
                    if added is not None:
                        # The same geometry the campaign injects with: the
                        # component along the vector is SET, so a layer that
                        # runs after another arrives where one does.
                        #
                        # float32 for the norm. This model's activations are
                        # float16, where a sum of 1536 squares reaches inf, and
                        # inf * a zero vector is the NaN that made every margin
                        # NaN before the first step could run.
                        wide = h.astype(mx.float32)
                        unit = added / mx.maximum(mx.linalg.norm(added), 1e-6)
                        scale = mx.mean(mx.linalg.norm(wide, axis=-1))
                        component = mx.sum(wide * unit, axis=-1, keepdims=True)
                        moved = wide + (arguments.alpha * scale - component) * unit
                        h = moved.astype(h.dtype)
                    return (h,) + result[1:] if isinstance(result, tuple) else h

            return Steered

        for layer in target_layers:
            originals.append((blocks[layer], blocks[layer].__class__))
            blocks[layer].__class__ = steered_class(blocks[layer].__class__, layer)

        def token_ids(carrier: str, continuation: str) -> tuple[list[int], int]:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": carrier}],
                tokenize=False,
                add_generation_prompt=True,
            )
            head = tokenizer.encode(prompt)
            whole = tokenizer.encode(prompt + continuation)
            return whole, len(head)

        def continuation_logprob(ids: list[int], start: int) -> mx.array:
            """Summed log-probability of the tokens after ``start``."""
            tokens = mx.array([ids])
            # float32 before the normaliser. A logsumexp over 151,936 float16
            # logits overflows to inf, and inf - inf is the NaN that made every
            # margin and every loss NaN on the first run.
            logits = model(tokens)[0].astype(mx.float32)
            logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            wanted_ids = mx.array(ids[start:])
            rows = mx.arange(start - 1, len(ids) - 1)
            return mx.sum(logprobs[rows, wanted_ids])

        # The forced-choice set the certificate itself scores, so the trainer
        # optimises the quantity the gate measures rather than a proxy.
        from core.consciousness.fusion_probe import FORCED_CHOICE

        controls: list[tuple[list[int], int, list[int], int]] = []
        for stem, right, wrong in FORCED_CHOICE:
            right_ids = tokenizer.encode(stem + right)
            wrong_ids = tokenizer.encode(stem + wrong)
            head = len(tokenizer.encode(stem))
            controls.append((right_ids, head, wrong_ids, head))

        pairs_by_dimension: dict[str, list[tuple[list[int], int, list[int], int]]] = {}
        for dimension in dimensions:
            key = str(dimension["key"])
            positives = list(dimension["positive"])
            negatives = list(dimension["negative"])
            pairs = []
            for index, carrier in enumerate(EXTRACTION_CARRIERS):
                for offset in range(min(len(positives), len(negatives))):
                    good = positives[(offset + index) % len(positives)]
                    bad = negatives[(offset + index) % len(negatives)]
                    good_ids, good_start = token_ids(carrier, good)
                    bad_ids, bad_start = token_ids(carrier, bad)
                    pairs.append((good_ids, good_start, bad_ids, bad_start))
            if len(pairs) > int(arguments.pairs):
                keep = np.random.default_rng(11).choice(
                    len(pairs), size=int(arguments.pairs), replace=False
                )
                pairs = [pairs[int(index)] for index in sorted(keep)]
            pairs_by_dimension[key] = pairs
            print(f"  {key}: {len(pairs)} preference pairs", flush=True)

        arguments.out.mkdir(parents=True, exist_ok=True)
        report: dict[str, object] = {
            "schema": "aura.caa.gradient_training.v1",
            "model_path": str(model_path),
            "model_descriptor_sha256": descriptor,
            "target_layers": target_layers,
            "steps": int(arguments.steps),
            "lr": float(arguments.lr),
            "beta": float(arguments.beta),
            "orthogonality_weight": float(arguments.orthogonality),
            "alpha": float(arguments.alpha),
            "dimensions": {},
        }

        try:
            for dimension in dimensions:
                key = str(dimension["key"])
                pairs = pairs_by_dimension[key]

                seeds = []
                for layer in target_layers:
                    seed_path = arguments.init / f"{key}_layer{layer}.npz"
                    if seed_path.exists():
                        seeds.append(np.load(seed_path)["v"].astype(np.float32))
                    else:
                        # No captured seed for this checkpoint. A random unit
                        # direction is a worse start and an honest one; what
                        # matters is that the optimiser is what finds the
                        # vector, not the seed.
                        draw = np.random.default_rng(len(seeds) + 7).standard_normal(hidden)
                        seeds.append((draw / np.linalg.norm(draw)).astype(np.float32))
                parameter = mx.array(np.stack(seeds).astype(np.float32))

                # The unsteered margin, measured once and held. Without it the
                # loss falls for a vector that makes every continuation more
                # likely, which is not steering.
                vectors["v"] = mx.zeros_like(parameter)
                control_base = []
                for right_ids, right_start, wrong_ids, wrong_start in controls:
                    control_base.append(
                        float(
                            continuation_logprob(right_ids, right_start)
                            - continuation_logprob(wrong_ids, wrong_start)
                        )
                    )
                control_baseline = mx.array(control_base)
                base = []
                for good_ids, good_start, bad_ids, bad_start in pairs:
                    base.append(
                        float(
                            continuation_logprob(good_ids, good_start)
                            - continuation_logprob(bad_ids, bad_start)
                        )
                    )
                    mx.eval(base[-1])
                baseline = mx.array(base)
                print(f"  {key}: unsteered margin {float(mx.mean(baseline)):+.3f}", flush=True)

                history: list[dict[str, float]] = []

                def loss_for(value, batch_indices):
                    vectors["v"] = value
                    total = mx.array(0.0)
                    for position in batch_indices:
                        good_ids, good_start, bad_ids, bad_start = pairs[position]
                        margin = continuation_logprob(
                            good_ids, good_start
                        ) - continuation_logprob(bad_ids, bad_start)
                        gain = margin - baseline[position]
                        total = total + mx.logaddexp(
                            mx.array(0.0), -float(arguments.beta) * gain
                        )
                    preference = total / max(len(batch_indices), 1)
                    # What the model still prefers the right answer by. A
                    # direction that costs more of this than a random one is
                    # refused by the certificate however well it steers.
                    kept = mx.array(0.0)
                    for position, (right_ids, right_start, wrong_ids, wrong_start) in enumerate(
                        controls
                    ):
                        margin = continuation_logprob(
                            right_ids, right_start
                        ) - continuation_logprob(wrong_ids, wrong_start)
                        kept = kept + mx.maximum(
                            mx.array(0.0), control_baseline[position] - margin
                        )
                    preference = preference + float(arguments.no_regression) * (
                        kept / max(len(controls), 1)
                    )
                    unit = value / mx.maximum(
                        mx.linalg.norm(value, axis=-1, keepdims=True), 1e-6
                    )
                    gram = unit @ unit.T
                    off = gram - mx.eye(gram.shape[0])
                    overlap = mx.sum(off * off) / max(
                        gram.shape[0] * (gram.shape[0] - 1), 1
                    )
                    return preference + float(arguments.orthogonality) * overlap

                rng = np.random.default_rng(20260913)
                optimiser_state = optim.Adam(learning_rate=float(arguments.lr))
                for step in range(int(arguments.steps)):
                    batch = rng.choice(len(pairs), size=min(4, len(pairs)), replace=False)
                    loss, gradient = mx.value_and_grad(
                        lambda value: loss_for(value, list(batch))
                    )(parameter)
                    parameter = optimiser_state.apply_gradients(
                        {"v": gradient}, {"v": parameter}
                    )["v"]
                    # Unit norm: alpha owns magnitude at serve time, so the
                    # optimiser must spend its freedom on direction.
                    parameter = parameter / mx.maximum(
                        mx.linalg.norm(parameter, axis=-1, keepdims=True), 1e-6
                    )
                    mx.eval(parameter, loss)
                    if step % 10 == 0 or step == int(arguments.steps) - 1:
                        unit = parameter
                        gram = unit @ unit.T
                        off = gram - mx.eye(gram.shape[0])
                        mean_cos = float(
                            mx.sum(mx.abs(off)) / (gram.shape[0] * (gram.shape[0] - 1))
                        )
                        history.append(
                            {"step": step, "loss": float(loss), "mean_abs_cosine": mean_cos}
                        )
                        print(
                            f"    step {step:3d}  loss {float(loss):+.4f}  "
                            f"mean|cos| {mean_cos:.4f}",
                            flush=True,
                        )

                trained = np.asarray(parameter, dtype=np.float32)
                for layer in target_layers:
                    row = trained[index_of[layer]]
                    np.savez(
                        arguments.out / f"{key}_layer{layer}.npz",
                        v=row.astype(np.float32),
                        derived_at=time.time(),
                        source="gradient_trained_preference_caa",
                        requested_layer=layer,
                        selected_layer=layer,
                        selection_reason="gradient_trained",
                        extracted=True,
                        model_descriptor_sha256=descriptor,
                    )
                report["dimensions"][key] = {  # type: ignore[index]
                    "pairs": len(pairs),
                    "unsteered_margin": float(mx.mean(baseline)),
                    "history": history,
                }
        finally:
            vectors["v"] = mx.zeros((len(target_layers), hidden))
            for block, original in originals:
                block.__class__ = original
            model.eval()

        (arguments.out / "capture.json").write_text(
            json.dumps(
                {
                    "schema": "aura.caa.vector_capture.v1",
                    "method": "gradient_trained_preference_steering",
                    "statistic": "per-layer vectors optimised so the steered "
                    "model raises the positive continuation above the negative "
                    "one by more than the unsteered model does, penalised on "
                    "cross-layer cosine",
                    "position": "the served chat template with its generation "
                    "prompt; the contrast statements are the continuation",
                    "extraction_carriers": list(EXTRACTION_CARRIERS),
                    "captured_by": "tools/train_steering_vectors_by_gradient.py",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        (arguments.out / "training_report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\nwrote {arguments.out} in {time.time() - started:.0f}s", flush=True)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
