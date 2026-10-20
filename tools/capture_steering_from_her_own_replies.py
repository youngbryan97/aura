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
its generation was steered by ("s", slot for slot as the hooks read it). Each
dimension is keyed on its own slot, as the hook weighs it, and on only the part
of that weight the other dimensions' weights do not explain, fitted on the
training replies: the partial effect, so that a state that moves several slots
at once is not credited to each of them in full. Her replies from the top
quarter of that remainder are set against her replies from its bottom quarter,
paired by length so that the contrast is not how much she wrote. Her latest
fifth of replies, by when she gave them, is held out. Each reply is read as her
own assistant turn in her chat template, and the hidden states over the reply's
tokens are averaged at the steering plan's layers. The direction is the
difference of the two means on the training pairs. It is kept for a layer only
when it separates the held-out pairs (one-sided Mann-Whitney, Bonferroni over
the layers) and separates them better than any other dimension's direction at
that layer does.

The first capture, on 18 October, keyed each dimension on the affect engine's
reading in "z". Those readings move together in her record (valence, arousal
and mobilization correlate 0.96 to 1.00; mobilization is 0.65 of arousal
whenever valence is positive), and they are not what scales a vector: the hooks
read her fused valence and arousal and the substrate's frustration, curiosity
and energy neurons. The four directions it kept pointed the same way, cosine
0.95 to 1.00, so positive valence and frustration would have steered her alike.

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

def steered_dimensions() -> dict[str, tuple[int, str]]:
    """Each steering dimension's slot in the recorded steering state, and how its hook weighs it."""
    from core.consciousness.affective_steering import AFFECTIVE_DIMENSIONS

    return {str(d["key"]): (int(d["substrate_idx"]), str(d["substrate_fn"])) for d in AFFECTIVE_DIMENSIONS}

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


def _resolution() -> float:
    """The smallest change in a hook's weight the record can show.

    The recorder keeps each slot to STEERING_DECIMALS places and a hook's weight moves at most twice as fast as
    its slot (core/consciousness/mood_weight.py), so a spread no wider than this is rounding, not her state.
    """
    from core.brain.llm.endogenous_pair_recorder import STEERING_DECIMALS

    return 2.0 * 10.0 ** -STEERING_DECIMALS


def _remainder(weights: Any, column: int, fit_rows: Any) -> tuple[Any, float]:
    """One dimension's weight less what the other dimensions' weights predict of it, and the share they explain.

    Least squares with an intercept, fitted on ``fit_rows`` only, so the held-out replies are not used to decide
    their own labels.
    """
    import numpy as np

    target = weights[:, column]
    others = np.delete(weights, column, axis=1)
    design = np.column_stack([np.ones(len(weights)), others])
    coefficients, *_ = np.linalg.lstsq(design[fit_rows], target[fit_rows], rcond=None)
    remainder = target - design @ coefficients
    spread = float(np.var(target[fit_rows]))
    explained = 1.0 - float(np.var(remainder[fit_rows])) / spread if spread > 0 else 0.0
    return remainder, explained


def contrasts(rows: Iterable[Mapping[str, Any]], dimensions: Mapping[str, tuple[int, str]], *, model: str
              ) -> dict[str, dict[str, Any]]:
    """For each dimension: train and held-out (high, low) reply pairs, or why there are none."""
    import numpy as np

    from core.consciousness.mood_weight import signed_weight

    needed = max(slot for slot, _fn in dimensions.values()) + 1
    seen: set[str] = set()
    kept: list[dict] = []
    without = 0
    for row in rows:
        text = str(row.get("text") or "").strip()
        if str(row.get("model") or "") != model or len(text.split()) < 2 or text in seen:
            continue
        steering = list(row.get("s") or [])
        if len(steering) < needed:
            without += 1
            continue
        seen.add(text)
        kept.append({"text": text, "s": steering, "t": float(row.get("t") or 0.0)})
    names = list(dimensions)
    if len(kept) < 2:
        reason = (f"{len(kept)} of her replies carry the state they were steered by "
                  f"({without} were recorded before it was kept)")
        return {name: {"reason": reason} for name in names}
    cut = _quantile([r["t"] for r in kept], 1.0 - 1.0 / HELD_OUT_ONE_IN)
    held_out = np.array([r["t"] > cut for r in kept])
    weights = np.array([[signed_weight(r["s"][slot], fn) for slot, fn in dimensions.values()] for r in kept])
    out: dict[str, dict[str, Any]] = {}
    for column, name in enumerate(names):
        remainder, explained = _remainder(weights, column, ~held_out)
        fitted = [float(v) for v, h in zip(remainder, held_out, strict=True) if not h]
        top, bottom = _quantile(fitted, 0.75), _quantile(fitted, 0.25)
        if not top - bottom > _resolution():
            out[name] = {"reason": f"slot {dimensions[name][0]} does not vary in her record once the other "
                                   f"dimensions are known (the others explain {explained:.3f} of it)"}
            continue
        high = [r for r, v in zip(kept, remainder, strict=True) if v >= top]
        low = [r for r, v in zip(kept, remainder, strict=True) if v <= bottom]
        train = _length_matched([r for r in high if r["t"] <= cut], [r for r in low if r["t"] <= cut])
        held = _length_matched([r for r in high if r["t"] > cut], [r for r in low if r["t"] > cut])
        out[name] = {"slot": dimensions[name][0], "explained_by_the_others": round(explained, 4),
                     "quartiles": [round(bottom, 4), round(top, 4)],
                     "train": [(a["text"], b["text"]) for a, b in train],
                     "held_out": [(a["text"], b["text"]) for a, b in held]}
    return out


def admitted(observed: float, p_value: float, rivals: Mapping[str, float], *, layers: int) -> tuple[bool, bool]:
    """Whether a layer's direction is its dimension's own, and whether it is kept.

    Kept when it separates the held-out pairs at p < 0.05 split over the layers, and separates them better than
    every other dimension's direction does.
    """
    specific = all(observed > rival for rival in rivals.values())
    return specific, p_value < 0.05 / layers and specific


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

    cache: dict[str, dict[int, Any]] = {}

    def read(reply: str) -> dict[int, Any]:
        if reply in cache:
            return cache[reply]
        head = tokenizer.apply_chat_template([{"role": "user", "content": ""}], add_generation_prompt=True,
                                             tokenize=True)
        ids = list(head) + list(tokenizer.encode(reply, add_special_tokens=False))
        span["start"] = len(head)
        captured.clear()
        mx.eval(model(mx.array([ids])), *captured.values())
        cache[reply] = {layer: np.array(value, dtype=np.float32) for layer, value in captured.items()}
        return cache[reply]

    out.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"model_path": str(model_path), "model_descriptor_sha256": descriptor,
                              "layers": list(layers), "dimensions": {}, "cosines": {}}
    usable = {d: c for d, c in selection.items() if c.get("train") and c.get("held_out")}
    for dimension, chosen in selection.items():
        if dimension not in usable:
            report["dimensions"][dimension] = {"kept_layers": [], "reason": chosen.get("reason", "too few pairs")}
    train = {d: [(read(a), read(b)) for a, b in c["train"]] for d, c in usable.items()}
    held = {d: [(read(a), read(b)) for a, b in c["held_out"]] for d, c in usable.items()}
    vectors = {d: {layer: np.mean([a[layer] - b[layer] for a, b in train[d]], axis=0) for layer in layers}
               for d in usable}

    def separation(direction: Any, pairs: list, layer: int) -> tuple[list[float], list[float]]:
        unit = direction / max(float(np.linalg.norm(direction)), 1e-9)
        centre = np.mean([x[layer] for pair in pairs for x in pair], axis=0)
        return ([float((a[layer] - centre) @ unit) for a, _b in pairs],
                [float((b[layer] - centre) @ unit) for _a, b in pairs])

    from scipy.stats import mannwhitneyu

    for dimension, chosen in usable.items():
        kept_layers = []
        per_layer = {}
        for layer in layers:
            vector = vectors[dimension][layer]
            high, low = separation(vector, held[dimension], layer)
            observed = _auroc(high, low)
            # One-sided Mann-Whitney U. A permutation test of 200 shuffles cannot report p below 1/201,
            # above 0.05 split over the plan's layers, so on 18 October it kept no layer although
            # held-out separation reached 0.74.
            p_value = float(mannwhitneyu(high, low, alternative="greater").pvalue)
            # The other dimensions' directions on this dimension's held-out pairs. A direction they
            # separate as well is not this dimension's own.
            rivals = {other: round(_auroc(*separation(vectors[other][layer], held[dimension], layer)), 4)
                      for other in usable if other != dimension}
            specific, kept = admitted(observed, p_value, rivals, layers=len(layers))
            per_layer[layer] = {"held_out_auroc": round(observed, 4), "p_one_sided": p_value,
                                "others_on_these_pairs": rivals, "specific": specific, "kept": kept}
            if kept:
                kept_layers.append(layer)
            # Every layer's direction is written, kept or not, so a later decision never needs the
            # model run again; "kept" says which the held-out tests admit.
            np.savez(out / f"{dimension}_layer{layer}{'' if kept else '.not_kept'}.npz",
                     v=vector.astype(np.float32), source="her_own_replies", extracted=True, dimension=dimension,
                     layer=layer, model=str(model_path), model_path=str(model_path),
                     model_descriptor_sha256=descriptor, train_pairs=len(train[dimension]),
                     held_out_pairs=len(held[dimension]), held_out_auroc=observed, p_one_sided=p_value,
                     derived_at=time.time())
        report["dimensions"][dimension] = {"slot": chosen["slot"], "quartiles": chosen["quartiles"],
                                           "explained_by_the_others": chosen["explained_by_the_others"],
                                           "train_pairs": len(train[dimension]),
                                           "held_out_pairs": len(held[dimension]),
                                           "per_layer": per_layer, "kept_layers": kept_layers}
    for layer in layers:
        names = sorted(usable)
        report["cosines"][str(layer)] = {
            f"{a}~{b}": round(float(vectors[a][layer] @ vectors[b][layer]
                                    / max(float(np.linalg.norm(vectors[a][layer]) * np.linalg.norm(vectors[b][layer])), 1e-9)), 4)
            for i, a in enumerate(names) for b in names[i + 1:]}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--select-only", action="store_true")
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    model_path = Path(plan["model_path"])
    rows = (json.loads(line) for line in args.pairs.read_text(encoding="utf-8").splitlines() if line.strip())
    selection = contrasts(rows, steered_dimensions(), model=model_path.name)
    summary = {d: ({"train": len(s["train"]), "held_out": len(s["held_out"]), "quartiles": s["quartiles"],
                    "explained_by_the_others": s["explained_by_the_others"]}
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
