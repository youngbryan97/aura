"""core/brain/latent_bridge.py

Latent-Space Bridge
===================
Substrate math directly modulates inference parameters in real time, instead
of being translated into prompt strings. The bridge converts the live
substrate signal (vitality, phi, free energy, neurochemicals, viability)
into concrete, measurable changes to the LLM call:

  * temperature        — scaled by acetylcholine (sharp focus → low temp)
  * top_p              — narrowed by phi (high integration → tighter sampling)
  * top_k              — narrowed by serotonin (regulated → fewer alts)
  * max_tokens budget  — gated by vitality (fatigue → shorter output)
  * repetition_penalty — boosted by frustration / failed-loop count
  * presence_penalty   — boosted by curiosity (lower repeats → more novelty)
  * stop_sequences     — appended when viability is degraded (early stop)
  * sampling_seed      — reseeded on dream/sleep transitions

The bridge is consumed by:
  * MLX inference path (``core/brain/llm/mlx_client.py`` reads
    ``current_inference_params()`` before each generation)
  * Cloud fallback (``core/brain/cloud_client.py``)
  * Brainstem fast path

This is the structural alternative to "tell the LLM in the prompt that
Aura feels tired" — the prompt does not change; the *sampling* changes.
The LoRA personality is the thing being sampled; the bridge controls how
the LoRA is sampled.

Activation steering hook
------------------------
``activation_offsets()`` returns the residual-stream offset vectors keyed
by transformer layer. The MLX side is responsible for adding these
offsets to the appropriate hidden states during generation; the bridge
only computes the offsets from substrate state. If MLX cannot accept
activation steering on a given build, the bridge degrades gracefully —
the sampling-parameter modulation continues to work.
"""
from __future__ import annotations


import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("Aura.LatentBridge")


# ---------------------------------------------------------------------------
# Live substrate read-through
# ---------------------------------------------------------------------------


def _safe_get(eng: Any, attr: str, default: float) -> float:
    try:
        v = getattr(eng, attr, None)
        if callable(v):
            v = v()
        if isinstance(v, dict):
            return float(v.get(attr.replace("get_", ""), default) or default)
        return float(v) if v is not None else default
    except Exception:
        return default


def _read_substrate() -> Dict[str, float]:
    out: Dict[str, float] = {
        "vitality": 0.7,
        "phi": 0.0,
        "free_energy": 0.5,
        "acetylcholine": 0.5,
        "serotonin": 0.5,
        "norepinephrine": 0.5,
        "cortisol": 0.3,
        "frustration": 0.0,
        "curiosity": 0.5,
        "valence": 0.0,
        "arousal": 0.5,
    }
    try:
        from core.container import ServiceContainer
        homeo = ServiceContainer.get("homeostasis_engine", default=None) or ServiceContainer.get("homeostatic_engine", default=None)
        if homeo is not None:
            out["vitality"] = _safe_get(homeo, "vitality", out["vitality"])
        phi_engine = ServiceContainer.get("phi_core", default=None)
        if phi_engine is not None:
            out["phi"] = _safe_get(phi_engine, "phi_s", out["phi"])
        fe_engine = ServiceContainer.get("free_energy_engine", default=None)
        if fe_engine is not None and getattr(fe_engine, "current", None) is not None:
            cur = fe_engine.current
            out["free_energy"] = float(getattr(cur, "free_energy", out["free_energy"]) or out["free_energy"])
            out["valence"] = float(getattr(cur, "valence", 0.0) or 0.0)
            out["arousal"] = float(getattr(cur, "arousal", 0.5) or 0.5)
        nc = ServiceContainer.get("neurochemical_regulator", default=None)
        if nc is not None and hasattr(nc, "snapshot"):
            d = nc.snapshot() or {}
            for k in ("acetylcholine", "serotonin", "norepinephrine", "cortisol"):
                if k in d:
                    out[k] = float(d[k])
        affect = ServiceContainer.get("affect_engine", default=None)
        if affect is not None and hasattr(affect, "snapshot"):
            d = affect.snapshot() or {}
            for k in ("frustration", "curiosity"):
                if k in d:
                    out[k] = float(d[k])
    except Exception as exc:
        logger.debug("latent_bridge substrate read failed: %s", exc)
    return out


# ---------------------------------------------------------------------------
# Mapping rules — explicit, tunable, falsifiable
# ---------------------------------------------------------------------------


@dataclass
class InferenceParams:
    temperature: float
    top_p: float
    top_k: int
    max_tokens: int
    repetition_penalty: float
    presence_penalty: float
    extra_stop_sequences: List[str] = field(default_factory=list)
    seed: Optional[int] = None
    layer_offsets: Dict[int, List[float]] = field(default_factory=dict)
    rationale: List[str] = field(default_factory=list)

    def merge_with_origin(self, base: Dict[str, Any]) -> Dict[str, Any]:
        """Apply the substrate-derived params on top of a base param dict
        (the params the inference gate would have used). Substrate values
        take precedence for the keys it controls; the rest pass through.
        """
        out = dict(base or {})
        out["temperature"] = self.temperature
        out["top_p"] = self.top_p
        out["top_k"] = self.top_k
        # max_tokens is *capped* downward, never raised, by the bridge.
        out["max_tokens"] = min(int(out.get("max_tokens", self.max_tokens) or self.max_tokens), self.max_tokens)
        out["repetition_penalty"] = self.repetition_penalty
        out["presence_penalty"] = self.presence_penalty
        if self.extra_stop_sequences:
            stops = list(out.get("stop_sequences") or [])
            stops.extend(self.extra_stop_sequences)
            out["stop_sequences"] = stops
        if self.seed is not None:
            out["seed"] = self.seed
        return out


def compute_inference_params(
    *,
    base_max_tokens: int = 768,
    base_temperature: float = 0.7,
    foreground: bool = True,
) -> InferenceParams:
    """Compute the live inference params from substrate state."""

    s = _read_substrate()
    rationale: List[str] = []

    # ─── temperature ────────────────────────────────────────────────────
    # acetylcholine is sharp attention → lower temp. Cortisol high → narrow
    # the search to avoid wandering. Curiosity high → permit exploration.
    temp = base_temperature
    temp -= 0.20 * (s["acetylcholine"] - 0.5)
    temp -= 0.15 * (s["cortisol"] - 0.3)
    temp += 0.20 * (s["curiosity"] - 0.5)
    temp_ceiling = 0.92 if foreground else 1.00
    temp = max(0.15, min(temp_ceiling, temp))
    rationale.append(
        f"temp={temp:.2f} (ach={s['acetylcholine']:.2f}, corts={s['cortisol']:.2f}, curio={s['curiosity']:.2f})"
    )

    # ─── top_p ─────────────────────────────────────────────────────────
    # High phi → narrower (more decisive sampling). Frustration narrows too.
    top_p = 0.95
    top_p -= 0.20 * max(0.0, s["phi"])  # phi can be 0..1+
    top_p -= 0.10 * s["frustration"]
    top_p -= 0.05 * max(0.0, s["arousal"] - 0.65)
    top_p = max(0.55, min(0.99, top_p))
    rationale.append(f"top_p={top_p:.2f} (phi={s['phi']:.2f}, frust={s['frustration']:.2f})")

    # ─── top_k ─────────────────────────────────────────────────────────
    # Serotonin high → regulated → fewer alternatives. Norepinephrine high
    # → more alert → wider scan.
    top_k = 60
    top_k -= int(20 * (s["serotonin"] - 0.5))
    top_k += int(15 * (s["norepinephrine"] - 0.5))
    top_k = max(20, min(120, top_k))
    rationale.append(f"top_k={top_k} (sero={s['serotonin']:.2f}, ne={s['norepinephrine']:.2f})")

    # ─── max_tokens budget ─────────────────────────────────────────────
    # Vitality drop literally shortens the output. This is the structural
    # version of "tell the model to be brief when tired".
    vitality_factor = max(0.35, min(1.0, s["vitality"]))
    cap = max(1, int(base_max_tokens * vitality_factor))
    # Fatigue from cortisol clips further
    cap = max(1, int(cap * max(0.55, 1.0 - 0.4 * max(0.0, s["cortisol"] - 0.5))))
    rationale.append(f"max_tokens={cap} (vitality={s['vitality']:.2f}, cap={cap})")

    # ─── repetition penalty ────────────────────────────────────────────
    loop_pressure = (
        0.22 * max(0.0, s["arousal"] - 0.60)
        + 0.18 * max(0.0, s["free_energy"] - 0.55)
        + 0.14 * max(0.0, temp - 0.80)
    )
    rep = 1.10 + 0.20 * s["frustration"] + loop_pressure
    rep_floor = 1.10 if foreground else 1.06
    rep = max(rep_floor, min(1.45, rep))
    rationale.append(
        f"rep_penalty={rep:.2f} (frust={s['frustration']:.2f}, arousal={s['arousal']:.2f}, fe={s['free_energy']:.2f})"
    )

    # ─── presence penalty ─────────────────────────────────────────────
    pres = 0.0 + 0.30 * s["curiosity"]
    pres = max(0.0, min(0.8, pres))
    rationale.append(f"presence={pres:.2f} (curio={s['curiosity']:.2f})")

    # ─── early-stop sequences when degraded ────────────────────────────
    extra_stops: List[str] = []
    try:
        from core.organism.viability import get_viability, ViabilityState
        v = get_viability().state
        if v in (ViabilityState.STARVED, ViabilityState.DEGRADED, ViabilityState.INJURED, ViabilityState.RECOVERING):
            extra_stops = ["\n\n##", "\n---\n"]
            rationale.append(f"early-stop appended (viability={v.value})")
    except Exception:
        pass

    # ─── activation steering offsets ──────────────────────────────────
    # Map (valence, arousal, dominant emotion) into per-layer residual
    # offsets. The MLX side may or may not consume these; if not, no harm
    # done. The vectors are deterministic given (s) so the same affect
    # produces the same nudge across runs.
    layer_offsets: Dict[int, List[float]] = {}
    for layer in (8, 16, 24):  # representative layers — MLX side maps these
        # 8-dim offset is a small, fixed-size descriptor that the MLX
        # side reshapes into a hidden-dim direction via a learned mapping.
        offset = [
            s["valence"],
            s["arousal"] - 0.5,
            s["acetylcholine"] - 0.5,
            s["serotonin"] - 0.5,
            s["norepinephrine"] - 0.5,
            s["cortisol"] - 0.3,
            s["curiosity"] - 0.5,
            s["frustration"],
        ]
        layer_offsets[layer] = offset

    return InferenceParams(
        temperature=temp,
        top_p=top_p,
        top_k=top_k,
        max_tokens=cap,
        repetition_penalty=rep,
        presence_penalty=pres,
        extra_stop_sequences=extra_stops,
        seed=None,
        layer_offsets=layer_offsets,
        rationale=rationale,
    )


def current_inference_params(*, base_max_tokens: int = 768, base_temperature: float = 0.7, foreground: bool = True) -> Dict[str, Any]:
    """Convenience wrapper used by inference clients. Returns a plain dict
    so callers don't take a hard import dependency on this module's
    dataclasses.
    """
    p = compute_inference_params(
        base_max_tokens=base_max_tokens,
        base_temperature=base_temperature,
        foreground=foreground,
    )
    return p.merge_with_origin({"max_tokens": base_max_tokens})


def activation_offsets() -> Dict[int, List[float]]:
    """Return the per-layer residual-stream offsets for activation steering.
    The MLX side reads this just before token generation.
    """
    return compute_inference_params().layer_offsets


__all__ = [
    "InferenceParams",
    "compute_inference_params",
    "current_inference_params",
    "activation_offsets",
]
