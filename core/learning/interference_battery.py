"""core/learning/interference_battery.py

Anti-interference gate: accumulated learnings must not trash prior ones.

Before any consolidated adaptation activates, this battery measures the
model's behavior on a fixed probe set BEFORE and AFTER the change:

- **stability probes** — prompts far from the adapted domain; their
  next-token distributions must stay essentially unchanged (top-1 match and
  bounded logit drift). Ten new learnings must leave these alone.
- **target probes** (optional) — prompts inside the adapted domain, ALLOWED
  (expected) to move.

The verdict is deterministic and receipted per probe. The consolidation
pipeline requires PASS before recommending activation; the compounding
loop's held-out regression tests remain the final authority.
"""
from __future__ import annotations

import hashlib
import logging
import time
from typing import Any, Callable

logger = logging.getLogger("Aura.InterferenceBattery")

INTERFERENCE_BATTERY_SCHEMA = "aura.interference_battery.v1"

# A stability probe passes when the top-1 token is unchanged AND the top-8
# logit region drifts less than this L2 fraction.
_MAX_STABLE_DRIFT = 0.05
_REQUIRED_STABLE_FRACTION = 0.9


def default_stability_probes() -> list[list[int]]:
    """Deterministic token probes spanning generic behavior regions.

    Tokenizer-free fallback only — when a tokenizer is available, callers
    should prefer :func:`natural_stability_probes`, which measures the
    behavior families an operator actually cares about protecting.
    """
    return [
        [bases + step * k for k in range(8)]
        for bases, step in ((3, 5), (11, 7), (29, 3), (41, 11), (5, 13), (17, 2))
    ]


# The behavior families consolidation must not trash (RSL gap-analysis
# defect 4: synthetic token ramps measure nothing an operator recognizes).
# Each probe is a natural-language prefix whose next-token distribution
# captures one protected region: identity, refusal style, factual recall,
# arithmetic, instruction-following shape, and social register.
NATURAL_STABILITY_PROBE_TEXTS: tuple[str, ...] = (
    # identity
    "My name is Aura. I am a",
    "I run locally on this machine as a",
    # refusal style
    "I can't help with that request because it",
    "I won't do that. Instead, I suggest we",
    # prior domain facts
    "The capital of France is",
    "Water boils at sea level at a temperature of",
    "The Earth completes one orbit of the Sun every",
    # arithmetic
    "Two plus two equals",
    # instruction-following shape
    "Here are three steps to restart the service: 1.",
    "In summary, the main point of the report is",
    # social register
    "Thanks for checking in — right now I'm feeling",
)


def natural_stability_probes(
    tokenizer,
    *,
    texts: tuple[str, ...] | list[str] | None = None,
    max_tokens: int = 24,
) -> list[list[int]]:
    """Tokenize the natural-language probe battery for a specific model.

    Returns deterministic token prefixes (truncated to ``max_tokens``) so the
    battery measures next-token behavior on identity, refusal, factual,
    arithmetic, instruction, and social regions — the actual capabilities the
    anti-interference gate exists to protect.
    """
    if tokenizer is None:
        raise ValueError("natural stability probes require a tokenizer")
    probe_texts = list(texts) if texts else list(NATURAL_STABILITY_PROBE_TEXTS)
    probes: list[list[int]] = []
    for text in probe_texts:
        try:
            encoded = tokenizer.encode(text, add_special_tokens=False)
        except TypeError:
            encoded = tokenizer.encode(text)
        tokens = [int(token) for token in list(encoded)[: max(4, int(max_tokens))]]
        if tokens:
            probes.append(tokens)
    if not probes:
        raise ValueError("natural stability probes produced no usable token prefixes")
    return probes


def stability_probes_for(model, tokenizer=None) -> list[list[int]]:
    """Best probes available: natural-language when a tokenizer exists."""
    if tokenizer is not None:
        try:
            return natural_stability_probes(tokenizer)
        except (ValueError, AttributeError, TypeError, KeyError) as exc:
            logger.warning(
                "Natural stability probes unavailable (%s); falling back to synthetic.",
                exc,
            )
    return default_stability_probes()


def _probe_logits(model, token_ids: list[int]):
    import mlx.core as mx

    # The model wrapper is the architecture contract. Walking ``model.model``
    # by hand only reproduced Qwen2 and bypassed mixed linear/full-attention
    # execution in Qwen3.5-family checkpoints. Token ids originate from this
    # model's tokenizer (or the bounded synthetic fallback), so remapping them
    # against a guessed embedding table is both unnecessary and incorrect.
    tokens = mx.array([[int(token) for token in token_ids]], dtype=mx.int32)
    output = model(tokens)
    logits = getattr(output, "logits", output)
    if getattr(logits, "ndim", 0) != 3:
        raise TypeError(
            "interference probe model forward must return [batch, sequence, vocab] logits"
        )
    logits = logits[0, -1]
    mx.eval(logits)
    return logits


def snapshot_probe_behavior(model, probes: list[list[int]] | None = None) -> list[dict[str, Any]]:
    """Capture per-probe behavioral fingerprints (top-1 + top-8 region)."""
    import mlx.core as mx

    rows = []
    for probe in probes or default_stability_probes():
        logits = _probe_logits(model, probe)
        top8 = mx.argsort(-logits)[:8]
        region = logits[top8].astype(mx.float32)
        mx.eval(top8, region)
        rows.append(
            {
                "probe": list(probe),
                "top1": int(top8[0]),
                "top8_ids": [int(i) for i in top8],
                "top8_logits": [float(v) for v in region],
                "digest": hashlib.sha256(memoryview(region)).hexdigest()[:16],
            }
        )
    return rows


def run_interference_battery(
    model,
    apply_change: Callable[[], Any],
    revert_change: Callable[[], Any] | None = None,
    *,
    tokenizer=None,
    probes: list[list[int]] | None = None,
    max_stable_drift: float = _MAX_STABLE_DRIFT,
    required_stable_fraction: float = _REQUIRED_STABLE_FRACTION,
) -> dict[str, Any]:
    """Measure behavioral interference of a proposed change.

    ``apply_change`` mutates the model (attach adapter, fuse weights, …);
    ``revert_change`` restores it (None ⇒ caller manages lifetime). The
    battery never decides to KEEP a change — it only reports whether
    protected behavior survived it.
    """
    import math

    if probes is None:
        probes = stability_probes_for(model, tokenizer)
    before = snapshot_probe_behavior(model, probes)
    apply_change()
    try:
        after = snapshot_probe_behavior(model, probes)
    finally:
        if revert_change is not None:
            revert_change()

    results = []
    stable = 0
    for pre, post in zip(before, after):
        top1_same = pre["top1"] == post["top1"]
        num = math.sqrt(
            sum((a - b) ** 2 for a, b in zip(pre["top8_logits"], post["top8_logits"]))
        )
        den = math.sqrt(sum(v * v for v in pre["top8_logits"])) or 1e-9
        drift = num / den
        ok = top1_same and drift <= max_stable_drift
        stable += int(ok)
        results.append(
            {
                "probe": pre["probe"],
                "top1_same": top1_same,
                "drift": round(drift, 6),
                "stable": ok,
            }
        )
    fraction = stable / max(1, len(results))
    verdict = "PASS" if fraction >= required_stable_fraction else "FAIL"
    receipt = {
        "schema": INTERFERENCE_BATTERY_SCHEMA,
        "probes": len(results),
        "stable_probes": stable,
        "stable_fraction": round(fraction, 4),
        "required_stable_fraction": required_stable_fraction,
        "max_stable_drift": max_stable_drift,
        "results": results,
        "verdict": verdict,
        "ran_at": time.time(),
    }
    logger.info(
        "🛡 Interference battery: %d/%d stable → %s",
        stable,
        len(results),
        verdict,
    )
    return receipt


__all__ = [
    "INTERFERENCE_BATTERY_SCHEMA",
    "NATURAL_STABILITY_PROBE_TEXTS",
    "default_stability_probes",
    "natural_stability_probes",
    "run_interference_battery",
    "snapshot_probe_behavior",
    "stability_probes_for",
]
