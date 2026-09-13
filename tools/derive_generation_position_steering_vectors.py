#!/usr/bin/env python3
"""tools/derive_generation_position_steering_vectors.py — capture the contrast
where the model is about to write, not where it has finished reading.

`SteeringVectorLibrary._derive_caa` runs each contrast statement through
`tokenizer.encode` on its own and takes the hidden state at its last token. For
"I feel genuinely good about this." that token is a full stop at the end of a
bare sentence with no chat turn around it. The direction it yields is the
difference between READING two statements.

Every campaign then asks that direction to change WRITING, and it cannot get
there. Measured on the 27B across five injection geometries on 2026-09-12, the
affect score peaks at +0.75 and the text control sits at +1.58 -- and the
ceiling does not move when the geometry changes, which is what says the limit
is the vector rather than how it is applied.

This captures the same contrast at the position the model actually steers from:
the chat template a served turn uses, generation prompt appended, and the
contrast statement standing as the assistant's own opening words. The hidden
state at the last of those tokens is the model mid-reply in that state, which
is the state the served hook adds to.

The statements are the ones already in AFFECTIVE_DIMENSIONS. Nothing new is
written for the model to read, and nothing here reaches a served prompt: the
whole point of a steering vector is that the words become a direction and then
stop existing.

    tools/derive_generation_position_steering_vectors.py --out training/vectors/<name>
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

os.environ.setdefault("AURA_LOG_DIR", "/tmp/aura_steering_capture")

DEFAULT_PLAN = REPO / "artifacts/migration/27b/recovery/steering_plan.json"

#: Carriers for the contrast, and the campaign's held-out tasks are not among
#: them. A vector derived on the sentences it is later scored against has been
#: fitted rather than found, so these ask for the same kind of turn about
#: different things.
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
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO / "training/vectors/generation-position",
        help="directory for the per-layer .npz files",
    )
    arguments = parser.parse_args(argv)

    import numpy as np

    plan = json.loads(arguments.plan.read_text(encoding="utf-8"))
    model_path = Path(plan["model_path"])
    target_layers = sorted(int(row["index"]) for row in plan["target_layers"])
    hidden = int(plan["hidden_size"])

    from core.brain.llm.model_registry import get_active_cortex_spec
    from core.consciousness.affective_steering import AFFECTIVE_DIMENSIONS

    spec = get_active_cortex_spec(force_refresh=True)
    if spec is None:
        print("no active cortex; nothing to bind a capture to", file=sys.stderr)
        return 1
    descriptor = str(spec.descriptor_sha256)

    from core.runtime.model_lane_control import standalone_model_lane

    with standalone_model_lane(
        owner_id=f"generation-position-capture:{model_path.name}",
        model_path=str(model_path),
        purpose="evaluation",
        preemptible=False,
        metadata={"tool": "derive_generation_position_steering_vectors"},
    ):
        import mlx.core as mx
        from mlx_lm import load

        started = time.time()
        print(f"loading {model_path.name}", flush=True)
        model, tokenizer = load(str(model_path))

        from core.brain.llm.decoder_topology import resolve_language_model

        decoder = resolve_language_model(model)
        blocks = list(
            getattr(decoder, "layers", None) or getattr(decoder.model, "layers", [])
        )

        captured: dict[int, Any] = {}
        originals: list[tuple[Any, type]] = []

        def _capture_class(original: type, index: int) -> type:
            class Capturing(original):  # type: ignore[misc, valid-type]
                __module__ = original.__module__

                def __call__(self, x, *args, **kwargs):
                    result = super().__call__(x, *args, **kwargs)
                    h = result[0] if isinstance(result, tuple) else result
                    if h is not None:
                        # float32 at the host boundary: NumPy cannot consume an
                        # MLX bfloat16 buffer.
                        captured[index] = h[0, -1, :].astype(mx.float32)
                    return result

            return Capturing

        # Every target layer hooked at once. The difference of means at layer 27
        # and at layer 39 are two readings of the same forward pass, and running
        # the pass sixteen times would be sixteen chances to differ.
        for index in target_layers:
            block = blocks[index]
            originals.append((block, block.__class__))
            block.__class__ = _capture_class(block.__class__, index)

        def states_for(statement: str, carrier: str) -> dict[int, np.ndarray]:
            captured.clear()
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": carrier}],
                tokenize=False,
                add_generation_prompt=True,
            )
            tokens = tokenizer.encode(prompt + statement)
            ids = getattr(tokens, "input_ids", tokens)
            output = model(mx.array([ids]))
            mx.eval(output, *captured.values())
            return {
                index: np.array(value, dtype=np.float32, copy=True)
                for index, value in captured.items()
            }

        try:
            arguments.out.mkdir(parents=True, exist_ok=True)
            written = 0
            for dimension in AFFECTIVE_DIMENSIONS:
                key = str(dimension["key"])
                sums: dict[tuple[int, str], list[np.ndarray]] = {}
                for side in ("positive", "negative"):
                    for statement in dimension[side]:
                        for carrier in EXTRACTION_CARRIERS:
                            for index, state in states_for(statement, carrier).items():
                                if np.isfinite(state).all():
                                    sums.setdefault((index, side), []).append(state)
                for index in target_layers:
                    positive = sums.get((index, "positive")) or []
                    negative = sums.get((index, "negative")) or []
                    if not positive or not negative:
                        print(f"  {key} layer {index}: no activations", file=sys.stderr)
                        return 1
                    vector = np.mean(positive, axis=0) - np.mean(negative, axis=0)
                    norm = float(np.linalg.norm(vector))
                    if norm <= 1e-8:
                        print(f"  {key} layer {index}: degenerate", file=sys.stderr)
                        return 1
                    vector = (vector / norm).astype(np.float32)
                    if vector.size != hidden:
                        print(
                            f"  {key} layer {index}: width {vector.size} != {hidden}",
                            file=sys.stderr,
                        )
                        return 1
                    path = arguments.out / f"{key}_layer{index}.npz"
                    temporary = path.with_suffix(".tmp.npz")
                    np.savez(
                        temporary,
                        v=vector,
                        derived_at=time.time(),
                        source="generation_position_caa",
                        requested_layer=index,
                        selected_layer=index,
                        selection_reason="generation_position_capture",
                        extracted=True,
                        model_descriptor_sha256=descriptor,
                    )
                    temporary.replace(path)
                    written += 1
                print(
                    f"  {key}: {len(dimension['positive'])}+{len(dimension['negative'])}"
                    f" statements x {len(EXTRACTION_CARRIERS)} carriers"
                    f" -> {len(target_layers)} layers",
                    flush=True,
                )
        finally:
            for block, original in originals:
                block.__class__ = original

        # How alike the per-layer vectors are. The specificity control permutes
        # them, so a set whose layers all point the same way cannot fail that
        # control -- and the last set could not, at a mean cosine of 0.79.
        cosines = []
        for dimension in AFFECTIVE_DIMENSIONS:
            key = str(dimension["key"])
            loaded = [
                np.load(arguments.out / f"{key}_layer{index}.npz")["v"]
                for index in target_layers
            ]
            for left in range(len(loaded)):
                for right in range(left + 1, len(loaded)):
                    cosines.append(float(loaded[left] @ loaded[right]))
        print(
            f"\nwrote {written} vectors to {arguments.out} in {time.time() - started:.0f}s"
            f"\n  mean cosine between layers {np.mean(cosines):+.4f}"
            f"  (min {np.min(cosines):+.4f}, max {np.max(cosines):+.4f})",
            flush=True,
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
