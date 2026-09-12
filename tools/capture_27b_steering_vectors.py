#!/usr/bin/env python3
"""tools/capture_27b_steering_vectors.py — CAA vectors for the resident checkpoint.

Her affective steering is causal: a vector added to the residual stream at
chosen layers changes the tokens. That is exactly why it cannot survive a model
swap. The 45 vectors retained from the 32B are 5120 wide and so is this
checkpoint's residual stream, so every one of them loads and every one of them
names a direction in a space that no longer exists. The migration contract says
so itself, with the steering component marked deferred, and a deferred steering
component is why no hooks install, why her substrate cannot reach the resident
model's forward pass, and why a fusion certificate over that channel would be a
measurement of nothing.

This captures the replacements. One forward pass per prompt with every target
layer hooked at once, rather than one pass per layer, because the difference of
means at layer 30 and the difference of means at layer 40 are two readings of
the same run.

Capture grants nothing. The plan's four evidence requirements are a separate
job, and `steering_regeneration.may_serve` fails closed without them.

    tools/capture_27b_steering_vectors.py
    tools/capture_27b_steering_vectors.py --plan artifacts/.../steering_plan.json
"""

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

os.environ.setdefault("AURA_LOG_DIR", "/tmp/aura_steering_capture")

DEFAULT_PLAN = REPO / "artifacts/migration/27b/recovery/steering_plan.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO / "training/vectors",
        help="where the per-layer .npz files land",
    )
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args(argv)

    import numpy as np

    plan = json.loads(arguments.plan.read_text(encoding="utf-8"))
    model_path = Path(plan["model_path"])
    layers_wanted = [int(row["index"]) for row in plan["target_layers"]]
    kind_by_layer = {int(row["index"]): str(row["kind"]) for row in plan["target_layers"]}
    hidden = int(plan["hidden_size"])

    from core.brain.llm.model_registry import get_active_cortex_spec
    from core.consciousness.affective_steering import AFFECTIVE_DIMENSIONS

    spec = get_active_cortex_spec(force_refresh=True)
    if spec is None:
        print("no active cortex; nothing to bind a capture to", file=sys.stderr)
        return 1
    descriptor = str(spec.descriptor_sha256)
    # Two identifiers name this checkpoint and they are not the same string.
    # The plan fingerprints the weights; the registry keeps its own descriptor,
    # and the descriptor is the one a cached vector is checked against when the
    # steering engine decides whether to load it. So the capture binds on the
    # path, carries both, and stamps each file with the registry's.
    if Path(str(getattr(spec, "model_path", ""))).resolve() != model_path.resolve():
        print(
            f"the plan targets {model_path} and the active cortex is "
            f"{getattr(spec, 'model_path', None)}; refusing to bind a capture to "
            "a checkpoint the plan did not describe",
            file=sys.stderr,
        )
        return 1

    pairs = sum(len(d["positive"]) + len(d["negative"]) for d in AFFECTIVE_DIMENSIONS)
    print(
        f"checkpoint      {model_path.name}\n"
        f"descriptor      {descriptor[:16]}\n"
        f"target layers   {len(layers_wanted)} "
        f"({sum(1 for k in kind_by_layer.values() if k == 'full_attention')} attention, "
        f"{sum(1 for k in kind_by_layer.values() if k != 'full_attention')} linear)\n"
        f"dimensions      {', '.join(d['key'] for d in AFFECTIVE_DIMENSIONS)}\n"
        f"forward passes  {pairs} (one per prompt, every layer read from each)\n"
        f"vectors out     {len(layers_wanted) * len(AFFECTIVE_DIMENSIONS)}",
        flush=True,
    )
    if arguments.dry_run:
        return 0

    import mlx.core as mx
    from mlx_lm import load

    started = time.time()
    print("loading the checkpoint", flush=True)
    model, tokenizer = load(str(model_path))
    print(f"loaded in {time.time() - started:.1f}s", flush=True)

    from core.brain.llm.decoder_topology import resolve_language_model
    decoder = resolve_language_model(model)
    blocks = list(getattr(decoder, "layers", None) or getattr(decoder.model, "layers", []))
    if not blocks:
        print("the loaded object publishes no decoder layers", file=sys.stderr)
        return 1
    if max(layers_wanted) >= len(blocks):
        print(
            f"the plan names layer {max(layers_wanted)} and the checkpoint has "
            f"{len(blocks)}",
            file=sys.stderr,
        )
        return 1

    captured: dict[int, object] = {}

    # One subclass per target block, each closing over its own index, so a
    # single pass reads every layer the plan asks for. Reading them one layer
    # at a time would run the same prompt sixteen times for sixteen numbers
    # that a single run already contains.
    for index in layers_wanted:
        block = blocks[index]
        base = block.__class__

        def make(base_class, layer_index):
            class Capturing(base_class):
                def __call__(self, x, *args, **kwargs):
                    out = super().__call__(x, *args, **kwargs)
                    hidden_states = out[0] if isinstance(out, tuple) else out
                    if hidden_states is not None:
                        captured[layer_index] = hidden_states[0, -1, :].astype(mx.float32)
                    return out
            return Capturing

        block.__class__ = make(base, index)

    def read_prompt(text: str):
        captured.clear()
        tokens = tokenizer.encode(text)
        ids = getattr(tokens, "input_ids", tokens)
        try:
            out = model(mx.array([ids]))
            mx.eval(out, *[v for v in captured.values() if v is not None])
        except (RuntimeError, ValueError, TypeError) as exc:
            print(f"  prompt discarded: {type(exc).__name__}: {exc}", flush=True)
            return None
        return {
            index: np.array(value, dtype=np.float32)
            for index, value in captured.items()
            if value is not None
        }

    arguments.out.mkdir(parents=True, exist_ok=True)
    config_path = model_path / "config.json"
    config_sha = _sha256_file(config_path) if config_path.exists() else ""
    written = 0
    for dimension in AFFECTIVE_DIMENSIONS:
        key = str(dimension["key"])
        sums: dict[str, dict] = {"positive": {}, "negative": {}}
        counts = {"positive": 0, "negative": 0}
        for side in ("positive", "negative"):
            for prompt in dimension[side]:
                reading = read_prompt(str(prompt))
                if not reading:
                    continue
                counts[side] += 1
                for index, vector in reading.items():
                    running = sums[side].get(index)
                    sums[side][index] = vector if running is None else running + vector
        if not counts["positive"] or not counts["negative"]:
            print(f"  {key}: no usable prompts on one side; skipped", flush=True)
            continue
        for index in layers_wanted:
            positive = sums["positive"].get(index)
            negative = sums["negative"].get(index)
            if positive is None or negative is None:
                continue
            vector = (positive / counts["positive"]) - (negative / counts["negative"])
            if vector.shape[0] != hidden:
                print(
                    f"  {key} layer {index}: width {vector.shape[0]} against "
                    f"{hidden}; skipped",
                    flush=True,
                )
                continue
            out = arguments.out / f"{key}_layer{index}.npz"
            np.savez(
                out,
                v=vector.astype(np.float32),
                source="extracted_caa",
                extracted=True,
                dimension=key,
                layer=index,
                layer_kind=kind_by_layer.get(index, ""),
                model=str(model_path),
                model_path=str(model_path),
                model_config_sha256=config_sha,
                model_config_path=str(config_path),
                model_descriptor_sha256=descriptor,
                plan_descriptor_fingerprint=str(plan["descriptor_fingerprint"]),
                positive_prompts=counts["positive"],
                negative_prompts=counts["negative"],
                derived_at=time.time(),
            )
            written += 1
        print(
            f"  {key}: {counts['positive']} positive, {counts['negative']} negative, "
            f"{len(layers_wanted)} layers",
            flush=True,
        )

    print(
        f"\nwrote {written} vectors to {arguments.out} in "
        f"{time.time() - started:.1f}s, bound to {descriptor[:16]}\n"
        "capture grants nothing: the four evidence requirements are a separate run",
        flush=True,
    )
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
