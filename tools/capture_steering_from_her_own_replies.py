#!/usr/bin/env python3
"""Steering directions from her own replies, by the state she was in when she wrote them.

    python tools/capture_steering_from_her_own_replies.py --out ~/.aura/rlc-evidence/her-own-steering-20261018
    python tools/capture_steering_from_her_own_replies.py --out DIR --select-only     # no model; the pairs and counts

The vectors she has were captured from sentences written for the purpose ("I
feel genuinely good about this."). On 13 October the persona 27B steered by them
met four of the fusion certificate's five criteria and failed the fifth, her
voice: a reply about fledged wrens came out as "The wren brood has successfully
completed its operational cycle". A direction taken from authored sentences
carries their register.

This takes the direction from what she has said. The endogenous pair recorder
(core/brain/llm/endogenous_pair_recorder.py) keeps every reply with the state
she held when she began it. For each steering dimension whose state reading
varies in that record, her replies from its top quarter are set against her
replies from its bottom quarter, paired by length so that the contrast is not
how much she wrote. Her latest fifth of replies, by when she gave them, is held
out (the recorder's prompt hash is one value for every row, so it cannot
separate prompts). Each reply is read as her own assistant turn in her chat
template, and the hidden states over the reply's tokens are averaged at the
steering plan's layers. The direction is the difference of the two means on the
training pairs; it is kept for a layer only when it separates the held-out
pairs better than the same direction with their labels shuffled.

Capture grants nothing. Installing the vectors and running the fusion
certificate (tools/measure_fusion_channel.py) are separate steps.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DEFAULT_PAIRS = REPO / "data/endogenous_language/pairs.jsonl"
DEFAULT_PLAN = REPO / "artifacts/migration/27b/recovery/steering_plan.json"
#: Steering dimension -> the state reading it is taken from (core/brain/llm/endogenous_state.py).
DIMENSIONS: Mapping[str, str] = {
    "valence_positive": "affect.valence",
    "arousal": "affect.arousal",
    "curiosity": "affect.curiosity",
    "energy": "affect.mobilization",
    "frustration": "affect.stress",
}
#: The latest one in this many replies is held out: fit on what she said before, check on what she said after.
HELD_OUT_ONE_IN = 5


def _quantile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = q * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _length_matched(high: list[dict], low: list[dict]) -> list[tuple[dict, dict]]:
    """Pair each high reply with the unused low reply nearest its length."""
    remaining = sorted(low, key=lambda r: len(r["text"]))
    pairs: list[tuple[dict, dict]] = []
    for row in sorted(high, key=lambda r: len(r["text"])):
        if not remaining:
            break
        best = min(range(len(remaining)), key=lambda i: abs(len(remaining[i]["text"]) - len(row["text"])))
        pairs.append((row, remaining.pop(best)))
    return pairs


def contrasts(rows: Iterable[Mapping[str, Any]], layout: Sequence[str], *, model: str
              ) -> dict[str, dict[str, Any]]:
    """For each dimension: train and held-out (high, low) reply pairs, or why there are none."""
    index = {name: i for i, name in enumerate(layout)}
    seen: set[str] = set()
    kept: list[dict] = []
    for row in rows:
        text = str(row.get("text") or "").strip()
        if str(row.get("model") or "") != model or len(text.split()) < 2 or text in seen:
            continue
        seen.add(text)
        kept.append({"text": text, "z": list(row.get("z") or []), "t": float(row.get("t") or 0.0)})
    cut = _quantile([r["t"] for r in kept], 1.0 - 1.0 / HELD_OUT_ONE_IN)
    for r in kept:
        r["held_out"] = r["t"] > cut
    out: dict[str, dict[str, Any]] = {}
    for dimension, reading in DIMENSIONS.items():
        column = index.get(reading)
        if column is None:
            out[dimension] = {"reason": f"{reading} is not in the state layout"}
            continue
        values = [r["z"][column] for r in kept if column < len(r["z"])]
        top, bottom = _quantile(values, 0.75), _quantile(values, 0.25)
        if not top > bottom:
            out[dimension] = {"reason": f"{reading} does not vary in her record (quartiles {bottom:.3f} to {top:.3f})"}
            continue
        high = [r for r in kept if column < len(r["z"]) and r["z"][column] >= top]
        low = [r for r in kept if column < len(r["z"]) and r["z"][column] <= bottom]
        train = _length_matched([r for r in high if not r["held_out"]], [r for r in low if not r["held_out"]])
        held = _length_matched([r for r in high if r["held_out"]], [r for r in low if r["held_out"]])
        out[dimension] = {"reading": reading, "quartiles": [round(bottom, 4), round(top, 4)],
                          "train": [(a["text"], b["text"]) for a, b in train],
                          "held_out": [(a["text"], b["text"]) for a, b in held]}
    return out


def _auroc(positive: Sequence[float], negative: Sequence[float]) -> float:
    wins = sum((p > n) + 0.5 * (p == n) for p in positive for n in negative)
    return wins / max(1, len(positive) * len(negative))


def capture(selection: Mapping[str, Mapping[str, Any]], *, model_path: Path, layers: Sequence[int],
            out: Path, descriptor: str) -> dict[str, Any]:
    """Run each reply through her model and write the directions that separate held-out replies."""
    import mlx.core as mx
    import numpy as np
    from mlx_lm import load

    from core.brain.llm.decoder_topology import resolve_language_model

    model, tokenizer = load(str(model_path))
    decoder = resolve_language_model(model)
    blocks = list(getattr(decoder, "layers", None) or getattr(decoder.model, "layers", []))
    captured: dict[int, Any] = {}
    span: dict[str, int] = {"start": 0}
    for layer in layers:
        base = blocks[layer].__class__

        def make(base_class: type, layer_index: int) -> type:
            class Capturing(base_class):  # type: ignore[misc, valid-type]
                def __call__(self, x: Any, *args: Any, **kwargs: Any) -> Any:
                    result = super().__call__(x, *args, **kwargs)
                    hidden = result[0] if isinstance(result, tuple) else result
                    captured[layer_index] = mx.mean(hidden[0, span["start"]:, :].astype(mx.float32), axis=0)
                    return result
            return Capturing

        blocks[layer].__class__ = make(base, layer)

    def read(reply: str) -> dict[int, Any]:
        head = tokenizer.apply_chat_template([{"role": "user", "content": ""}], add_generation_prompt=True,
                                             tokenize=True)
        ids = list(head) + list(tokenizer.encode(reply, add_special_tokens=False))
        span["start"] = len(head)
        captured.clear()
        mx.eval(model(mx.array([ids])), *captured.values())
        return {layer: np.array(value, dtype=np.float32) for layer, value in captured.items()}

    out.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"model_path": str(model_path), "model_descriptor_sha256": descriptor,
                              "layers": list(layers), "dimensions": {}}
    for dimension, chosen in selection.items():
        if "train" not in chosen or not chosen["train"] or not chosen["held_out"]:
            report["dimensions"][dimension] = {"kept_layers": [], "reason": chosen.get("reason", "too few pairs")}
            continue
        train = [(read(a), read(b)) for a, b in chosen["train"]]
        held = [(read(a), read(b)) for a, b in chosen["held_out"]]
        kept_layers = []
        per_layer = {}
        for layer in layers:
            vector = np.mean([a[layer] - b[layer] for a, b in train], axis=0)
            unit = vector / max(float(np.linalg.norm(vector)), 1e-9)
            centre = np.mean([x[layer] for pair in held for x in pair], axis=0)
            high = [float((a[layer] - centre) @ unit) for a, _b in held]
            low = [float((b[layer] - centre) @ unit) for _a, b in held]
            observed = _auroc(high, low)
            # One-sided Mann-Whitney U. A permutation test of 200 shuffles cannot report p below 1/201,
            # above 0.05 split over the plan's layers, so on 18 October it kept no layer although
            # held-out separation reached 0.74.
            from scipy.stats import mannwhitneyu

            p_value = float(mannwhitneyu(high, low, alternative="greater").pvalue)
            kept = p_value < 0.05 / len(layers)
            per_layer[layer] = {"held_out_auroc": round(observed, 4), "p_one_sided": p_value, "kept": kept}
            if kept:
                kept_layers.append(layer)
            # Every layer's direction is written, kept or not, so a later decision never needs the
            # model run again; "kept" says which the held-out test admits.
            np.savez(out / f"{dimension}_layer{layer}{'' if kept else '.not_kept'}.npz",
                     v=vector.astype(np.float32), source="her_own_replies", extracted=True, dimension=dimension,
                     layer=layer, model=str(model_path), model_path=str(model_path),
                     model_descriptor_sha256=descriptor, train_pairs=len(train), held_out_pairs=len(held),
                     held_out_auroc=observed, p_one_sided=p_value, derived_at=time.time())
        report["dimensions"][dimension] = {"reading": chosen["reading"], "quartiles": chosen["quartiles"],
                                           "train_pairs": len(train), "held_out_pairs": len(held),
                                           "per_layer": per_layer, "kept_layers": kept_layers}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--select-only", action="store_true")
    args = parser.parse_args()

    from core.brain.llm.endogenous_state import describe_layout

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    model_path = Path(plan["model_path"])
    layout = [d["name"] for d in describe_layout()]
    rows = (json.loads(line) for line in args.pairs.read_text(encoding="utf-8").splitlines() if line.strip())
    selection = contrasts(rows, layout, model=model_path.name)
    summary = {d: ({"train": len(s["train"]), "held_out": len(s["held_out"]), "quartiles": s["quartiles"]}
                   if "train" in s else s) for d, s in selection.items()}
    print(json.dumps(summary, indent=2))
    args.out.expanduser().mkdir(parents=True, exist_ok=True)
    (args.out.expanduser() / "selection.json").write_text(json.dumps(selection, indent=1))
    if args.select_only:
        return 0
    from core.brain.llm.model_registry import get_active_cortex_spec

    spec = get_active_cortex_spec(force_refresh=True)
    descriptor = str(getattr(spec, "descriptor_sha256", "") or "")
    report = capture(selection, model_path=model_path, layers=[int(r["index"]) for r in plan["target_layers"]],
                     out=args.out.expanduser(), descriptor=descriptor)
    print(json.dumps({d: r.get("kept_layers") for d, r in report["dimensions"].items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
