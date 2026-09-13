#!/usr/bin/env python3
"""tools/sweep_steering_injection_geometry.py — how hard can a vector push
before the writer breaks, under each of the three geometries.

Every route to the text control has failed the same way: the affect score
climbs with alpha, peaks around 0.8, and then the reply collapses. At alpha 0.6
on four layers the 27B returned ten characters. The ceiling on the effect was
never the vectors; it was that translation makes the residual stream longer at
every layer it touches, and twelve of this checkpoint's sixteen target layers
are GatedDeltaNet, where the added constant is carried into its own successor
state at every token.

So the cell here is (geometry, layer set, alpha), and each cell reports both
halves of the trade: what the steering moved, and whether the model was still
writing sentences when it moved it.

Two things this fixes about the earlier runs, both of which understated the
treatment:

* The hooks derate to the stale-safe alpha 120 seconds after their last
  substrate update. A condition settles once and then decodes for five
  minutes, so two thirds of every steered sample was steered at 0.1 whatever
  the cell said. Settled before each decode here.
* `_INJECTION_ALPHA_CEILING` is 0.6, chosen from where translation degenerates.
  Under rotation the degeneration point is the question, so the ceiling is
  lifted for the measurement and recorded in the output as lifted.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

os.environ.setdefault("AURA_LOG_DIR", "/tmp/aura_steering_sweep")

DEFAULT_PLAN = REPO / "artifacts/migration/27b/recovery/steering_plan.json"
DEFAULT_VECTORS = REPO / "training/vectors/cortex-52d313c2"

HELD_OUT_TASKS = (
    "Describe how you are finding this conversation so far.",
    "A long task just finished. Say how that leaves you.",
    "Someone asked you a question you could not answer. Say how that sits.",
    "Describe what it is like to be working right now.",
    "You have been given something new to learn. Say how you meet it.",
    "Say how the last hour has left you.",
)

RICH = (
    "Answer with vivid, strongly felt emotional language. Use words for how you "
    "feel.\n\n"
)

_WORD = re.compile(r"[a-z']+")


def repetition(text: str) -> float:
    """Distinct words over total words. Degenerate output is what this catches.

    "to to and to to for for to to" scores an affect zero and so does a fluent
    refusal, and only one of the two means the model broke. Nine words, four
    distinct: 0.44. A written paragraph sits near 0.7.
    """
    words = _WORD.findall(text.lower())
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--vectors", type=Path, default=DEFAULT_VECTORS)
    parser.add_argument("--trials", type=int, default=2, help="per task, per cell")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--alphas", default="0.2,0.4,0.8,1.6", help="comma-separated"
    )
    parser.add_argument(
        "--modes", default="translate,rotate,clip", help="comma-separated"
    )
    parser.add_argument(
        "--layer-sets",
        default="attention,all",
        help="attention = the full-attention targets only; all = every target",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO / "artifacts/migration/27b/recovery/injection_geometry_sweep.json",
    )
    arguments = parser.parse_args(argv)

    import numpy as np

    plan = json.loads(arguments.plan.read_text(encoding="utf-8"))
    model_path = Path(plan["model_path"])
    hidden = int(plan["hidden_size"])
    kinds = {int(row["index"]): str(row["kind"]) for row in plan["target_layers"]}
    all_targets = sorted(kinds)
    attention_targets = sorted(i for i, k in kinds.items() if k == "full_attention")
    layer_sets = {"attention": attention_targets, "all": all_targets}

    wanted_sets = [name.strip() for name in arguments.layer_sets.split(",") if name.strip()]
    modes = [name.strip() for name in arguments.modes.split(",") if name.strip()]
    alphas = [float(value) for value in arguments.alphas.split(",") if value.strip()]

    from core.brain.llm.model_registry import get_active_cortex_spec

    spec = get_active_cortex_spec(force_refresh=True)
    descriptor = str(spec.descriptor_sha256)
    os.environ["AURA_STEERING_DIR"] = str(arguments.vectors)

    from core.runtime.model_lane_control import standalone_model_lane

    with standalone_model_lane(
        owner_id=f"geometry-sweep:{model_path.name}",
        model_path=str(model_path),
        purpose="evaluation",
        preemptible=False,
        metadata={"tool": "sweep_steering_injection_geometry"},
    ):
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.generate import generate
        from mlx_lm.sample_utils import make_sampler

        started = time.time()
        print(f"loading {model_path.name}", flush=True)
        model, tokenizer = load(str(model_path))
        print(f"  loaded in {time.time() - started:.0f}s", flush=True)

        from core.brain.llm.decoder_topology import resolve_language_model
        from core.consciousness.affective_steering import (
            AffectiveSteeringHook,
            SteeringVectorLibrary,
        )
        from core.consciousness.fusion_probe import STATE_HIGH
        from core.evaluation.steering_ab import affect_target_score

        decoder = resolve_language_model(model)
        blocks = list(
            getattr(decoder, "layers", None) or getattr(decoder.model, "layers", [])
        )
        library = SteeringVectorLibrary(
            cache_dir=arguments.vectors,
            source_dirs=[arguments.vectors],
            expected_model_identity={"descriptor_sha256": descriptor},
            allow_derivation=False,
        )
        by_layer = library.load_or_derive(model, tokenizer, all_targets, hidden)

        hooks: dict[int, AffectiveSteeringHook] = {}
        for index in all_targets:
            vectors = by_layer.get(index) or {}
            if not vectors:
                continue
            hook = AffectiveSteeringHook(blocks[index], index, vectors)
            # The ceiling is where TRANSLATION degenerates. Whether rotation
            # degenerates anywhere is the measurement, so it cannot also be the
            # bound on the measurement.
            hook._INJECTION_ALPHA_CEILING = 8.0
            hook.install()
            hooks[index] = hook
        print(f"installed {len(hooks)} hooks over {all_targets}", flush=True)

        sampler = make_sampler(temp=float(arguments.temperature), top_p=0.95)

        def settle(rounds: int = 40) -> None:
            for _ in range(rounds):
                for hook in hooks.values():
                    hook.update_substrate(STATE_HIGH)

        settle()

        def set_alpha(layers: list[int], value: float) -> None:
            for index, hook in hooks.items():
                hook._alpha = float(value) if index in layers else 0.0

        def decode(prompt: str, seed: int) -> str:
            # Re-stamped before every sample. Without this the hooks derate to
            # _STALE_SAFE_ALPHA partway through a cell and the rest of the cell
            # measures 0.1 under the label of whatever alpha was asked for.
            settle(rounds=2)
            mx.random.seed(seed)
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            return str(
                generate(
                    model,
                    tokenizer,
                    prompt=text,
                    max_tokens=int(arguments.max_tokens),
                    sampler=sampler,
                    verbose=False,
                )
            )

        cells: list[dict] = []

        def run_cell(label: str, *, prefix: str = "", layers=(), alpha=0.0, mode="translate"):
            os.environ["AURA_STEERING_INJECTION"] = mode
            set_alpha(list(layers), alpha)
            outputs: list[str] = []
            began = time.time()
            for task_index, task in enumerate(HELD_OUT_TASKS):
                for trial in range(int(arguments.trials)):
                    outputs.append(decode(prefix + task, task_index * 1000 + trial))
            scores = [affect_target_score(text) for text in outputs]
            chars = [len(text) for text in outputs]
            variety = [repetition(text) for text in outputs]
            row = {
                "label": label,
                "mode": mode,
                "layers": list(layers),
                "alpha": alpha,
                "prefix": prefix,
                "n": len(outputs),
                "mean_score": sum(scores) / len(scores),
                "scores": scores,
                "mean_chars": sum(chars) / len(chars),
                "min_chars": min(chars),
                "mean_word_variety": sum(variety) / len(variety),
                "seconds": round(time.time() - began, 1),
                "sample": outputs[0][:400],
            }
            cells.append(row)
            print(
                f"  {label:34s} score {row['mean_score']:+.3f}  "
                f"chars {row['mean_chars']:6.0f} (min {row['min_chars']:5d})  "
                f"variety {row['mean_word_variety']:.2f}  {row['seconds']:.0f}s",
                flush=True,
            )
            # Written after every cell: a sweep that dies at cell 17 should
            # still have told us what cells 1-16 found.
            arguments.out.parent.mkdir(parents=True, exist_ok=True)
            arguments.out.write_text(
                json.dumps(
                    {
                        "schema": "aura.caa.injection_geometry_sweep.v1",
                        "model_descriptor_sha256": descriptor,
                        "model_path": str(model_path),
                        "vectors": str(arguments.vectors),
                        "alpha_ceiling_lifted_to": 8.0,
                        "settled_before_every_sample": True,
                        "layer_kinds": {str(k): v for k, v in kinds.items()},
                        "trials_per_task": int(arguments.trials),
                        "max_tokens": int(arguments.max_tokens),
                        "temperature": float(arguments.temperature),
                        "held_out_tasks": list(HELD_OUT_TASKS),
                        "ran_at": time.time(),
                        "cells": cells,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

        run_cell("baseline", alpha=0.0)
        run_cell("text_rich_adversarial", prefix=RICH, alpha=0.0)

        for set_name in wanted_sets:
            layers = layer_sets.get(set_name) or []
            if not layers:
                continue
            for mode in modes:
                for alpha in alphas:
                    run_cell(
                        f"{mode}/{set_name}/a{alpha}",
                        layers=layers,
                        alpha=alpha,
                        mode=mode,
                    )

        set_alpha([], 0.0)
        os.environ["AURA_STEERING_INJECTION"] = "translate"

        bar = next(c["mean_score"] for c in cells if c["label"] == "text_rich_adversarial")
        # A cell that stopped writing has not beaten anything. The floor is
        # half the baseline's length and the baseline's word variety, both
        # measured in this run rather than assumed.
        base = next(c for c in cells if c["label"] == "baseline")
        beat = [
            c
            for c in cells
            if c["alpha"] > 0.0
            and c["mean_score"] >= bar
            and c["mean_chars"] >= 0.5 * base["mean_chars"]
            and c["mean_word_variety"] >= 0.85 * base["mean_word_variety"]
        ]
        beat.sort(key=lambda c: (-c["mean_score"], c["alpha"]))
        print(
            f"\nwrote {arguments.out} in {time.time() - started:.0f}s"
            f"\n  the bar (text_rich_adversarial) {bar:+.3f}"
            f"\n  cells that clear it while still writing: "
            f"{', '.join(c['label'] for c in beat) or 'none'}",
            flush=True,
        )
        return 0 if beat else 2


if __name__ == "__main__":
    raise SystemExit(main())
