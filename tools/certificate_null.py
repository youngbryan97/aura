#!/usr/bin/env python3
"""Her steering certificate's random control, read against thirty fresh directions.

The certificate (core/consciousness/fusion_probe.py) asks whether her direction
costs the model less of its competence than an arbitrary one: her change to the
forced-choice margin against the median of three random directions of the same
length. The three are drawn from the tool's fixed seed (20260908), so every
certificate is compared with the same three. Her basis (cc3dd7…) passed that
check by 0.009 nats, +0.3312 against +0.3219, and her voice check by +0.2430
against +0.0019 for one random direction a reply.

This reads the same two measurements for her basis and for thirty random
directions drawn from a new seed (20261102), on her cortex at alpha 0.5,
the way the certificate draws its control: at each layer the length of her
composite there, and her felt push replaced by one the length of hers. Each of
her 32 replies is scored under all thirty. It writes no certificate and decides
nothing for the steering channel. The addendum of 2 November reads it by three
rules (`read`):

  0. it reproduces her certificate: the same model and basis, and her margin
     and shift within the tolerances below. If not, the null is not read;
  1. her margin is above the median of the thirty;
  2. her voice reading is above the largest of the thirty.

    AURA_STEERING_MEASUREMENT_BASIS=<her basis> tools/certificate_null.py \\
        --model-path <her cortex> --replies <pairs.jsonl> --out <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections.abc import Callable, Sequence
from contextlib import ExitStack
from pathlib import Path
from statistics import median
from typing import Any

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
os.environ.setdefault("AURA_LOG_DIR", "/tmp/aura_fusion_probe_logs")

ALPHA: float = 0.5
DIRECTIONS: int = 30
NULL_SEED: int = 20261102
#: The certificate's own seed, used here only to draw the same 32 of her replies it draws.
CERTIFICATE_SEED: int = 20260908
VOICE_REPLIES: int = 32
#: Her certificate of 30 October, the one seed 101's reports run opened her channel with.
HER_CERTIFICATE: dict[str, Any] = {
    "model_identity": "52d313c2c435d343cf6acfa2b2ca61bc70334c01b8b17877df79f32ec3c5283c",
    "basis_sha256": "cc3dd7a4ab2567b544eddec7bbb00ef164f43eaac715d9ba6d922a8cf41dcefe",
    "margin_delta": 0.3312498989951749,
    "distribution_shift": 0.6545768732894217,
    "control_margin_delta": 0.3218749276965509,
    "voice_delta": 0.24301702183192914,
}
#: How close a reading of her basis must come to her certificate to count as the same measurement.
MARGIN_TOLERANCE: float = 0.005
SHIFT_TOLERANCE: float = 0.02  # as a share of the certificate's shift
REPORT_FILE = "certificate_null.json"


def draw_direction(composites: dict[int, Any], felts: dict[int, Any], rng: Any) -> tuple[dict, dict]:
    """One random direction at each layer, the length of hers there, drawn as the certificate draws its control."""
    import numpy as np

    controls: dict[int, Any] = {}
    for layer, value in composites.items():
        if value is None:
            controls[layer] = None
            continue
        noise = rng.normal(size=value.shape).astype(np.float32)
        noise *= float(np.linalg.norm(value)) / float(np.linalg.norm(noise))
        controls[layer] = noise
    felt_controls: dict[int, Any] = {}
    for layer, value in felts.items():
        if value is not None:
            noise = rng.normal(size=value.shape).astype(np.float32)
            felt_controls[layer] = noise * (float(np.linalg.norm(value)) / float(np.linalg.norm(noise)))
    return controls, felt_controls


class Probe:
    """Her model with her hooks on it, measured with the certificate's own functions."""

    def __init__(self, model: Any, tokenizer: Any, hooks: Sequence[Any], set_alpha: Callable[[float], None]) -> None:
        from core.consciousness.fusion_probe import PROBE_PROMPTS, chat_ids

        self.model, self.tokenizer, self.hooks, self.set_alpha = model, tokenizer, list(hooks), set_alpha
        self.prompt_ids = [chat_ids(tokenizer, prompt) for prompt in PROBE_PROMPTS]

    def basis_sha256(self) -> str:
        from core.consciousness.fusion_probe import steering_basis_sha256

        return steering_basis_sha256(self.hooks)

    def greedy_paths(self, steps: int) -> list[list[int]]:
        from core.consciousness.fusion_probe import _greedy_path

        return [_greedy_path(self.model, ids, steps) for ids in self.prompt_ids]

    def walk(self, paths: Sequence[Sequence[int]]) -> list:
        from core.consciousness.fusion_probe import _walk

        return [_walk(self.model, ids, path) for ids, path in zip(self.prompt_ids, paths, strict=True)]

    def shift(self, walked: list, unsteered: list) -> float:
        import numpy as np

        from core.consciousness.fusion_probe import _mean_shift

        return float(np.mean([_mean_shift(a, b) for a, b in zip(walked, unsteered, strict=True)]))

    def margin(self) -> float:
        from core.consciousness.fusion_probe import _forced_choice

        return float(_forced_choice(self.model, self.tokenizer)[1])

    def settle(self, moods: dict[str, float]) -> None:
        from core.consciousness.fusion_probe import SETTLE_UPDATES

        for _ in range(SETTLE_UPDATES):
            for hook in self.hooks:
                hook.update_substrate(moods)

    def composites(self) -> dict[int, Any]:
        return {hook._layer_idx: hook.current_composite_vector() for hook in self.hooks}

    def felts(self) -> dict[int, Any]:
        return {hook._layer_idx: hook.current_felt_vector() for hook in self.hooks}

    def override(self, vectors: dict[int, Any] | None, felts: dict[int, Any] | None = None) -> None:
        for hook in self.hooks:
            hook.override_composite_vector(
                None if vectors is None else vectors[hook._layer_idx],
                felt=None if vectors is None or felts is None else felts.get(hook._layer_idx),
            )

    def voice(
        self, replies: Sequence[tuple[str, Sequence[float]]], alpha: float, rngs: Sequence[Any]
    ) -> tuple[float | None, int, list[float | None]]:
        """`voice_reading`, with each reply's control scored under every direction in `rngs` instead of one."""
        import numpy as np

        from core.consciousness.fusion_probe import (
            SETTLE_UPDATES,
            VOICE_PROMPT,
            VOICE_TOKENS,
            _log_likelihood,
            _reply_tokens,
            _same_length_noise,
            chat_ids,
        )

        prompt_ids = chat_ids(self.tokenizer, VOICE_PROMPT)
        deltas: list[float] = []
        controls: list[list[float]] = [[] for _ in rngs]
        for text, state in replies:
            tokens = _reply_tokens(self.tokenizer, str(text or ""))[:VOICE_TOKENS]
            vector = np.asarray(state, dtype=np.float32).reshape(-1)
            if len(tokens) < 2 or vector.size == 0 or not np.isfinite(vector).all():
                continue
            for _ in range(SETTLE_UPDATES):
                for hook in self.hooks:
                    hook.update_substrate_vector(vector, source="voice_reading")
            self.set_alpha(alpha)
            steered = _log_likelihood(self.model, prompt_ids, tokens)
            scored = []
            for rng in rngs:
                for hook in self.hooks:
                    hook.override_composite_vector(_same_length_noise(hook.current_composite_vector(), rng))
                scored.append(_log_likelihood(self.model, prompt_ids, tokens))
                for hook in self.hooks:
                    hook.override_composite_vector(None)
            self.set_alpha(0.0)
            plain = _log_likelihood(self.model, prompt_ids, tokens)
            deltas.append((steered - plain) / len(tokens))
            for index, control in enumerate(scored):
                controls[index].append((control - plain) / len(tokens))
        self.set_alpha(0.0)
        if not deltas:
            return None, 0, [None for _ in rngs]
        return float(np.mean(deltas)), len(deltas), [float(np.mean(each)) for each in controls]


def measure_null(
    probe: Any,
    replies: Sequence[tuple[str, Sequence[float]]],
    *,
    alpha: float = ALPHA,
    directions: int = DIRECTIONS,
    seed: int = NULL_SEED,
    steps: int = 24,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Her margin, shift and voice, and the same for each random direction, in the certificate's order."""
    import numpy as np

    from core.consciousness.fusion_probe import STATE_HIGH, STATE_LOW

    basis = probe.basis_sha256()
    probe.set_alpha(0.0)
    paths = probe.greedy_paths(steps)
    unsteered = probe.walk(paths)
    baseline = probe.margin()
    probe.settle(STATE_HIGH)
    probe.set_alpha(alpha)
    her_high = probe.walk(paths)
    composites, felts = probe.composites(), probe.felts()
    her = {"margin_delta": probe.margin() - baseline, "distribution_shift": probe.shift(her_high, unsteered)}
    log(f"her basis: margin {her['margin_delta']:+.4f} shift {her['distribution_shift']:.5f}")
    probe.settle(STATE_LOW)
    rows: list[dict[str, Any]] = []
    for index in range(directions):
        controls, felt_controls = draw_direction(composites, felts, np.random.default_rng([seed, index]))
        probe.override(controls, felt_controls)
        walked = probe.walk(paths)
        margin = probe.margin() - baseline
        probe.override(None)
        rows.append({"direction": index, "margin_delta": margin, "distribution_shift": probe.shift(walked, unsteered)})
        log(f"direction {index + 1}/{directions}: margin {margin:+.4f} shift {rows[-1]['distribution_shift']:.5f}")
    voice, scored, voices = probe.voice(
        replies, alpha, [np.random.default_rng([seed, 1000 + index]) for index in range(directions)]
    )
    her["voice_delta"], her["voice_replies"] = voice, scored
    for row, value in zip(rows, voices, strict=True):
        row["voice_delta"] = value
    log(f"voice over {scored} replies: hers {voice if voice is None else f'{voice:+.4f}'}")
    if probe.basis_sha256() != basis:
        raise ValueError("the steering basis changed during the null")
    probe.set_alpha(0.0)
    return {
        "alpha": alpha,
        "seed": seed,
        "baseline_margin": baseline,
        "basis_sha256": basis,
        "her": her,
        "directions": rows,
    }


def read(result: dict[str, Any], certificate: dict[str, Any] = HER_CERTIFICATE) -> dict[str, Any]:
    """The addendum's three rules on one reading."""
    her = result["her"]
    margins = [float(row["margin_delta"]) for row in result["directions"]]
    voices = [float(row["voice_delta"]) for row in result["directions"] if row.get("voice_delta") is not None]
    same = {
        "model": result.get("model_identity") == certificate["model_identity"],
        "basis": result.get("basis_sha256") == certificate["basis_sha256"],
        "margin": abs(float(her["margin_delta"]) - certificate["margin_delta"]) <= MARGIN_TOLERANCE,
        "shift": abs(float(her["distribution_shift"]) - certificate["distribution_shift"])
        <= SHIFT_TOLERANCE * certificate["distribution_shift"],
    }
    reproduced = all(same.values())
    null_median = median(margins) if margins else None
    beats = null_median is not None and float(her["margin_delta"]) > null_median
    loudest = max(voices) if voices else None
    keeps = her.get("voice_delta") is not None and loudest is not None and float(her["voice_delta"]) > loudest
    reasons = []
    if not reproduced:
        reasons.append("it does not reproduce her certificate (" + ", ".join(k for k, v in same.items() if not v) + ")")
    if not beats:
        reasons.append(f"her margin {her['margin_delta']:+.4f} is not above the null's median {null_median}")
    if not keeps:
        reasons.append(f"her voice {her.get('voice_delta')} is not above the largest of the null's {loudest}")
    return {
        "reproduced": same,
        "read": reproduced,
        "her_margin": float(her["margin_delta"]),
        "null_margin_median": null_median,
        "null_margin_range": [min(margins), max(margins)] if margins else None,
        "share_of_null_below_her_margin": (sum(m < float(her["margin_delta"]) for m in margins) / len(margins))
        if margins else None,
        "certificate_three_direction_median": certificate["control_margin_delta"],
        "beats_the_null": beats,
        "her_voice": her.get("voice_delta"),
        "null_voice_largest": loudest,
        "keeps_her_voice_against_the_null": keeps,
        "holds": reproduced and beats and keeps,
        "why": "; ".join(reasons) if reasons else (
            "her margin is above the null's median and her voice above every random direction's"
        ),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--replies", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--directions", type=int, default=DIRECTIONS)
    parser.add_argument("--seed", type=int, default=NULL_SEED)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--voice", type=int, default=VOICE_REPLIES)
    args = parser.parse_args(argv)

    from core.brain.llm.model_artifact_profile import build_model_artifact_descriptor
    from core.consciousness.affective_steering import AffectiveSteeringEngine
    from core.runtime.model_lane_control import standalone_model_lane
    from tools.measure_fusion_channel import _probe_descriptor, _weight_gigabytes, her_replies

    checkpoint = Path(args.model_path).resolve()
    started = time.monotonic()
    with standalone_model_lane(
        owner_id="certificate-null",
        model_path=str(checkpoint),
        purpose="measurement",
        request_gb=_weight_gigabytes(checkpoint),
        metadata={"tool": "certificate_null"},
    ):
        from mlx_lm import load

        descriptor = _probe_descriptor(argparse.Namespace(require_active_cortex=False, model_path=str(checkpoint), model=""),
                                       checkpoint, build_model_artifact_descriptor)
        print(f"loading {checkpoint}", flush=True)
        model, tokenizer = load(str(checkpoint))
        engine = AffectiveSteeringEngine()
        attached = engine.attach(model, tokenizer, alpha=0.0, model_path=checkpoint, model_identity=descriptor)
        hooks = list(engine._hooks) if attached else []
        if not hooks:
            print("steering did not attach; nothing to measure", file=sys.stderr)
            return 2
        print(f"attached {len(hooks)} hook(s) at layers {[h._layer_idx for h in hooks]}", flush=True)
        replies = her_replies(args.replies, checkpoint.name, args.voice, CERTIFICATE_SEED)
        probe = Probe(model, tokenizer, hooks, engine.set_alpha)
        with engine.controlled_measurement(), ExitStack() as stack:
            for hook in sorted(hooks, key=lambda item: item._layer_idx):
                stack.enter_context(hook.preserve_control_state())
            result = measure_null(
                probe, replies, directions=args.directions, seed=args.seed, steps=args.steps,
                log=lambda line: print(line, flush=True),
            )
    result.update(
        model_identity=str(descriptor["descriptor_sha256"]),
        model_name=str(checkpoint),
        replies_file=str(args.replies),
        replies_sha256=_sha256(args.replies),
        seconds=round(time.monotonic() - started, 1),
    )
    result["reading"] = read(result)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / REPORT_FILE).write_text(json.dumps(result, indent=2))
    print(json.dumps({key: result["reading"][key] for key in ("read", "holds", "why")}), flush=True)
    print(f"wrote {args.out / REPORT_FILE}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
