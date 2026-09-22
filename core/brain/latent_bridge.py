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
import time
from dataclasses import dataclass, field
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.LatentBridge")

DEFAULT_SUBSTRATE_STATE: dict[str, float] = {
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
    "active_uncertainty": 0.0,
    "active_tool_pressure": 0.0,
    "active_error_pressure": 0.0,
    "active_reduce_load": 0.0,
    "active_seek_information": 0.0,
    "organismal_coherence": 0.6,
    "causal_verification_need": 0.0,
    "causal_governance_pressure": 0.0,
    "causal_metabolic_budget": 0.7,
    "sentience_candidate_strength": 0.0,
}


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
    except (OSError, ConnectionError, TimeoutError):
        return default


@dataclass(frozen=True)
class SubstrateReading:
    """Substrate values, and how much of the substrate actually answered.

    CP126 (high): "Partial and total substrate failures collapse to healthy
    defaults. Missing services and many errors return biologically plausible
    defaults... the output carries no degraded flag, missing-channel list,
    uncertainty, freshness, or receipt."

    The defaults are plausible on purpose — a caller needs a number to sample
    with. The problem was that a dead substrate produced the same dict as a
    calm one, so "she is settled" and "nothing is reporting" were the same
    reading, and inference parameters were derived from the difference.
    """

    values: dict[str, float]
    sourced: frozenset[str]
    read_at: float

    @property
    def defaulted(self) -> tuple[str, ...]:
        return tuple(sorted(set(DEFAULT_SUBSTRATE_STATE) - self.sourced))

    @property
    def coverage(self) -> float:
        total = len(DEFAULT_SUBSTRATE_STATE)
        return (len(self.sourced & set(DEFAULT_SUBSTRATE_STATE)) / total) if total else 0.0

    @property
    def degraded(self) -> bool:
        """True when most of the substrate did not answer.

        Half is the line because the mapping rules below read from several
        channels at once; below that the parameters are mostly derived from
        constants, which is a different claim from "measured and calm".
        """
        return self.coverage < 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "aura.substrate_reading.v1",
            "coverage": round(self.coverage, 3),
            "degraded": self.degraded,
            "sourced": sorted(self.sourced),
            "defaulted": list(self.defaulted),
            "read_at": self.read_at,
        }


def _read_substrate_detailed() -> SubstrateReading:
    """Read every channel independently, recording which ones answered.

    Each channel gets its own guard. The previous single try/except around
    the whole body meant the FIRST failing service silently skipped every
    read after it — one unavailable homeostasis engine blanked phi, free
    energy, neurochemistry, affect, active inference and the causal vector,
    and the result was still returned as a normal substrate state.
    """
    out: dict[str, float] = dict(DEFAULT_SUBSTRATE_STATE)
    sourced: set[str] = set()

    def _channel(keys: tuple[str, ...], reader: Any) -> None:
        try:
            if reader():
                sourced.update(keys)
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, KeyError) as exc:
            record_degradation(
                "latent_bridge",
                exc,
                severity="info",
                action=f"defaulted substrate channel(s) {','.join(keys)} after read failed",
                enforce_failure_policy=False,
            )

    try:
        from core.container import ServiceContainer
    except ImportError as exc:
        record_degradation(
            "latent_bridge",
            exc,
            severity="warning",
            action="returned a fully defaulted substrate reading; no channel was measured",
        )
        return SubstrateReading(out, frozenset(), time.time())

    def _homeo() -> bool:
        homeo = ServiceContainer.get("homeostasis_engine", default=None) or ServiceContainer.get("homeostatic_engine", default=None)
        if homeo is None:
            return False
        out["vitality"] = _safe_get(homeo, "vitality", out["vitality"])
        return True

    def _phi() -> bool:
        phi_engine = ServiceContainer.get("phi_core", default=None)
        if phi_engine is None:
            return False
        out["phi"] = _safe_get(phi_engine, "phi_s", out["phi"])
        return True

    def _free_energy() -> bool:
        fe_engine = ServiceContainer.get("free_energy_engine", default=None)
        if fe_engine is None or getattr(fe_engine, "current", None) is None:
            return False
        cur = fe_engine.current
        out["free_energy"] = float(getattr(cur, "free_energy", out["free_energy"]) or out["free_energy"])
        out["valence"] = float(getattr(cur, "valence", 0.0) or 0.0)
        out["arousal"] = float(getattr(cur, "arousal", 0.5) or 0.5)
        return True

    def _neuro() -> bool:
        nc = ServiceContainer.get("neurochemical_regulator", default=None)
        if nc is None or not hasattr(nc, "snapshot"):
            return False
        d = nc.snapshot() or {}
        seen = False
        for k in ("acetylcholine", "serotonin", "norepinephrine", "cortisol"):
            if k in d:
                out[k] = float(d[k])
                seen = True
        return seen

    def _affect() -> bool:
        affect = ServiceContainer.get("affect_engine", default=None)
        if affect is None or not hasattr(affect, "snapshot"):
            return False
        d = affect.snapshot() or {}
        seen = False
        for k in ("frustration", "curiosity"):
            if k in d:
                out[k] = float(d[k])
                seen = True
        return seen

    def _active() -> bool:
        advisor = ServiceContainer.get("spiking_active_inference", default=None)
        if advisor is None or not hasattr(advisor, "snapshot"):
            return False
        active = advisor.snapshot() or {}
        routing = active.get("routing_bias") or {}
        features = active.get("features") or {}
        if isinstance(routing, dict):
            out["active_reduce_load"] = 1.0 if routing.get("reduce_load") else 0.0
            out["active_seek_information"] = 1.0 if routing.get("seek_information") else 0.0
        if isinstance(features, dict):
            out["active_tool_pressure"] = float(features.get("tool_pressure", 0.0) or 0.0)
            out["active_error_pressure"] = float(features.get("error_pressure", 0.0) or 0.0)
        out["active_uncertainty"] = float(active.get("uncertainty", 0.0) or 0.0)
        return True

    def _causal() -> bool:
        being_runtime = ServiceContainer.get("being_runtime", default=None)
        causal_vector = getattr(being_runtime, "_last_causal_self_vector", None)
        if causal_vector is None or not hasattr(causal_vector, "value"):
            return False
        out["organismal_coherence"] = float(causal_vector.value("organismal_coherence", out["organismal_coherence"]))
        out["causal_verification_need"] = float(causal_vector.value("verification_need", out["causal_verification_need"]))
        out["causal_governance_pressure"] = float(causal_vector.value("governance_pressure", out["causal_governance_pressure"]))
        out["causal_metabolic_budget"] = float(causal_vector.value("metabolic_budget", out["causal_metabolic_budget"]))
        out["sentience_candidate_strength"] = float(causal_vector.value("sentience_candidate_strength", out["sentience_candidate_strength"]))
        return True

    _channel(("vitality",), _homeo)
    _channel(("phi",), _phi)
    _channel(("free_energy", "valence", "arousal"), _free_energy)
    _channel(("acetylcholine", "serotonin", "norepinephrine", "cortisol"), _neuro)
    _channel(("frustration", "curiosity"), _affect)
    _channel(
        ("active_reduce_load", "active_seek_information", "active_tool_pressure",
         "active_error_pressure", "active_uncertainty"),
        _active,
    )
    _channel(
        ("organismal_coherence", "causal_verification_need", "causal_governance_pressure",
         "causal_metabolic_budget", "sentience_candidate_strength"),
        _causal,
    )
    return SubstrateReading(out, frozenset(sourced), time.time())


def _read_substrate_legacy() -> dict[str, float]:
    out: dict[str, float] = dict(DEFAULT_SUBSTRATE_STATE)
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
        advisor = ServiceContainer.get("spiking_active_inference", default=None)
        if advisor is not None and hasattr(advisor, "snapshot"):
            active = advisor.snapshot() or {}
            routing = active.get("routing_bias") or {}
            features = active.get("features") or {}
            if isinstance(routing, dict):
                out["active_reduce_load"] = 1.0 if routing.get("reduce_load") else 0.0
                out["active_seek_information"] = 1.0 if routing.get("seek_information") else 0.0
            if isinstance(features, dict):
                out["active_tool_pressure"] = float(features.get("tool_pressure", 0.0) or 0.0)
                out["active_error_pressure"] = float(features.get("error_pressure", 0.0) or 0.0)
            out["active_uncertainty"] = float(active.get("uncertainty", 0.0) or 0.0)
        being_runtime = ServiceContainer.get("being_runtime", default=None)
        causal_vector = getattr(being_runtime, "_last_causal_self_vector", None)
        if causal_vector is not None and hasattr(causal_vector, "value"):
            out["organismal_coherence"] = float(
                causal_vector.value("organismal_coherence", out["organismal_coherence"])
            )
            out["causal_verification_need"] = float(
                causal_vector.value("verification_need", out["causal_verification_need"])
            )
            out["causal_governance_pressure"] = float(
                causal_vector.value("governance_pressure", out["causal_governance_pressure"])
            )
            out["causal_metabolic_budget"] = float(
                causal_vector.value("metabolic_budget", out["causal_metabolic_budget"])
            )
            out["sentience_candidate_strength"] = float(
                causal_vector.value("sentience_candidate_strength", out["sentience_candidate_strength"])
            )
    except (ImportError, AttributeError, RuntimeError) as exc:
        record_degradation('latent_bridge', exc)
        logger.debug("latent_bridge substrate read failed: %s", exc)
    return out


def _read_substrate() -> dict[str, float]:
    """The substrate values, for callers that only want numbers.

    Provenance is available through _read_substrate_detailed(); this keeps
    the historical signature so every existing caller is unaffected.
    """
    return _read_substrate_detailed().values


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
    extra_stop_sequences: list[str] = field(default_factory=list)
    seed: int | None = None
    layer_offsets: dict[int, list[float]] = field(default_factory=dict)
    rationale: list[str] = field(default_factory=list)

    def merge_with_origin(self, base: dict[str, Any]) -> dict[str, Any]:
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
    base_max_tokens: int = 1536,
    base_temperature: float = 0.7,
    foreground: bool = True,
    lane: str = "speech",
) -> InferenceParams:
    """Compute the live inference params from substrate state."""

    s = dict(DEFAULT_SUBSTRATE_STATE)
    try:
        s.update(_read_substrate() or {})
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation("latent_bridge", exc)
        logger.debug("latent_bridge defaulted substrate state after read failure: %s", exc)
    rationale: list[str] = []

    # ─── temperature ────────────────────────────────────────────────────
    # acetylcholine is sharp attention → lower temp. Cortisol high → narrow
    # the search to avoid wandering. Curiosity high → permit exploration.
    temp = base_temperature
    temp -= 0.20 * (s["acetylcholine"] - 0.5)
    temp -= 0.15 * (s["cortisol"] - 0.3)
    temp -= 0.10 * s["active_uncertainty"]
    temp -= 0.08 * s["active_error_pressure"]
    temp -= 0.10 * s["causal_verification_need"]
    temp -= 0.08 * max(0.0, 0.55 - s["organismal_coherence"])
    temp += 0.20 * (s["curiosity"] - 0.5)
    temp += 0.04 * s["active_seek_information"]
    temp_ceiling = 0.85 if foreground else 0.90
    temp = max(0.15, min(temp_ceiling, temp))

    # Anti-trap governor: sustained floor-pinned temperature with
    # non-improving distress is a closed feedback loop (safety clamp
    # starving the very variance needed to self-heal). The guard observes
    # every computation, opens a bounded exploration escape when trapped,
    # and gives repair/ideation lanes an unconditional exploration floor.
    try:
        from core.brain.affective_antitrap import get_affective_trap_guard
        temp, _antitrap_note = get_affective_trap_guard().observe_and_adjust(
            temp, s, lane=lane,
        )
        if _antitrap_note:
            rationale.append(_antitrap_note)
    except (ImportError, AttributeError, RuntimeError) as exc:
        logger.debug("Anti-trap guard unavailable: %s", exc)
    rationale.append(
        f"temp={temp:.2f} (ach={s['acetylcholine']:.2f}, corts={s['cortisol']:.2f}, "
        f"curio={s['curiosity']:.2f}, active_uncert={s['active_uncertainty']:.2f}, "
        f"cvw={s['organismal_coherence']:.2f}, verify={s['causal_verification_need']:.2f})"
    )

    # ─── top_p ─────────────────────────────────────────────────────────
    # High phi → narrower (more decisive sampling). Frustration narrows too.
    top_p = 0.95
    top_p -= 0.20 * max(0.0, s["phi"])  # phi can be 0..1+
    top_p -= 0.10 * s["frustration"]
    top_p -= 0.08 * s["active_uncertainty"]
    top_p -= 0.06 * s["active_error_pressure"]
    top_p -= 0.05 * max(0.0, s["arousal"] - 0.65)
    top_p -= 0.08 * s["causal_governance_pressure"]
    top_p -= 0.07 * s["causal_verification_need"]
    top_p = max(0.55, min(0.99, top_p))
    rationale.append(
        f"top_p={top_p:.2f} (phi={s['phi']:.2f}, frust={s['frustration']:.2f}, "
        f"active_error={s['active_error_pressure']:.2f}, governance={s['causal_governance_pressure']:.2f})"
    )

    # ─── top_k ─────────────────────────────────────────────────────────
    # Serotonin high → regulated → fewer alternatives. Norepinephrine high
    # → more alert → wider scan.
    top_k = 60
    top_k -= int(20 * (s["serotonin"] - 0.5))
    top_k += int(15 * (s["norepinephrine"] - 0.5))
    top_k = max(20, min(120, top_k))
    rationale.append(f"top_k={top_k} (sero={s['serotonin']:.2f}, ne={s['norepinephrine']:.2f})")

    # ─── max_tokens budget ─────────────────────────────────────────────
    # Vitality and load still change how much compute the substrate spends,
    # but foreground speech needs enough room to finish a coherent thought.
    # Multiplying every homeostatic reduction used to collapse a requested
    # 972-token desktop turn to ~189 tokens, guaranteeing clipped prose and a
    # costly retry storm. Critical memory admission is enforced before this
    # bridge; this floor only prevents non-critical affective state from
    # making the speech organ structurally incapable of completing a reply.
    vitality_factor = max(0.35, min(1.0, s["vitality"]))
    cap = max(1, int(base_max_tokens * vitality_factor))
    # Fatigue from cortisol clips further
    cap = max(1, int(cap * max(0.55, 1.0 - 0.4 * max(0.0, s["cortisol"] - 0.5))))
    if s["active_reduce_load"] > 0.0:
        cap = max(1, int(cap * 0.62))
    cap = max(1, int(cap * max(0.45, s["causal_metabolic_budget"])))
    if s["organismal_coherence"] < 0.45:
        cap = max(1, int(cap * 0.82))
    if foreground and base_max_tokens >= 512:
        completion_floor_factor = 0.55
        if s["active_reduce_load"] > 0.0:
            completion_floor_factor = min(completion_floor_factor, 0.45)
        if s["organismal_coherence"] < 0.45:
            completion_floor_factor = min(completion_floor_factor, 0.48)
        completion_floor = min(
            base_max_tokens,
            max(384, int(base_max_tokens * completion_floor_factor)),
        )
        cap = max(cap, completion_floor)
    rationale.append(f"max_tokens={cap} (vitality={s['vitality']:.2f}, cap={cap})")

    # ─── repetition penalty ────────────────────────────────────────────
    loop_pressure = (
        0.12 * max(0.0, s["arousal"] - 0.60)
        + 0.10 * max(0.0, s["free_energy"] - 0.55)
        + 0.10 * max(0.0, temp - 0.80)
        + 0.04 * s["active_uncertainty"]
        + 0.06 * s["active_error_pressure"]
        + 0.05 * s["causal_verification_need"]
        + 0.04 * max(0.0, 0.55 - s["organismal_coherence"])
    )
    rep = 1.05 + 0.05 * s["frustration"] + loop_pressure
    rep_floor = 1.05 if foreground else 1.02
    rep = max(rep_floor, min(1.15, rep))
    rationale.append(
        f"rep_penalty={rep:.2f} (frust={s['frustration']:.2f}, arousal={s['arousal']:.2f}, fe={s['free_energy']:.2f})"
    )

    # ─── presence penalty ─────────────────────────────────────────────
    pres = (
        0.0
        + 0.30 * s["curiosity"]
        + 0.08 * s["active_tool_pressure"]
        + 0.05 * max(0.0, s["organismal_coherence"] - 0.55)
        - 0.08 * s["causal_governance_pressure"]
    )
    pres = max(0.0, min(0.8, pres))
    rationale.append(
        f"presence={pres:.2f} (curio={s['curiosity']:.2f}, cvw={s['organismal_coherence']:.2f})"
    )

    # ─── early-stop sequences when degraded ────────────────────────────
    extra_stops: list[str] = []
    try:
        from core.organism.viability import ViabilityState, get_viability
        v = get_viability().state
        if v in (ViabilityState.STARVED, ViabilityState.DEGRADED, ViabilityState.INJURED, ViabilityState.RECOVERING):
            extra_stops = ["\n\n##", "\n---\n"]
            rationale.append(f"early-stop appended (viability={v.value})")
    except (ImportError, AttributeError, RuntimeError) as exc:
        logger.debug(
            "%s unavailable (%s: %s); no viability-driven early stop is appended",
            "ViabilityState",
            type(exc).__name__,
            exc,
        )

    # ─── activation steering offsets ──────────────────────────────────
    # Map (valence, arousal, dominant emotion) into per-layer residual
    # offsets. The MLX side may or may not consume these; if not, no harm
    # done. The vectors are deterministic given (s) so the same affect
    # produces the same nudge across runs.
    layer_offsets: dict[int, list[float]] = {}
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


def current_inference_params(*, base_max_tokens: int = 1536, base_temperature: float = 0.7, foreground: bool = True) -> dict[str, Any]:
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


def activation_offsets() -> dict[int, list[float]]:
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
