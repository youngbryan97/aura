"""core/learning/latent_adapter_distillation.py

The missing car of the consolidation train: proposal → durable adapter →
anti-interference + held-out gates → activation → proven rollback.

`latent_consolidation` aggregates mechanically-clean episode fast weights
into governed PROPOSALS; until now the train stopped there ("the one item
that turns parity into improving" — RSL gap analysis). This module finishes
the loop without inventing a second training pipeline: episode ΔW=UVᵀ
candidates are already weight-space objects, so distillation is a provenance
-checked low-rank MERGE (rank concatenation computing the candidate MEAN
delta), the anti-interference battery (natural-language probes) plus a
mandatory sealed held-out regression check gates activation. Activation uses
the same module-swap seam as episodic fast weights, and rollback restores the
original modules with probe-equality proof.

Nothing here mutates stored checkpoint files. Durable adapters live beside
the model as governed artifacts; the compounding loop remains the authority
for fusing anything into published weights.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.learning.heldout_battery import BatterySpec
from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.LatentAdapterDistillation")

DURABLE_ADAPTER_SCHEMA = "aura.latent_durable_adapter.v1"
CONSOLIDATION_TRAIN_SCHEMA = "aura.latent_consolidation_train.v1"

_TARGET_ATTRS = {
    "o_proj": ("self_attn", "o_proj"),
    "down_proj": ("mlp", "down_proj"),
}


def _linear_dims(module) -> tuple[int, int]:
    """(out_features, in_features) for Linear or QuantizedLinear."""
    weight = module.weight
    if hasattr(module, "scales"):  # QuantizedLinear packs weights
        bits = int(getattr(module, "bits", 4))
        return int(weight.shape[0]), int(weight.shape[1]) * (32 // bits)
    return int(weight.shape[0]), int(weight.shape[1])
# Held-out accuracy may drop at most this much after activation.
_HELDOUT_REGRESSION_TOLERANCE = 0.02


class DurableDeltaLinear:
    """y = base(x) + s·((x Vᵀ) Uᵀ) with U, V loaded from a durable artifact."""

    def __init__(self, base, u_factor, v_factor, scale: float, tag: str) -> None:
        import mlx.core as mx

        self.base = base
        self.U = mx.array(u_factor)
        self.V = mx.array(v_factor)
        self.scale = float(scale)
        self.tag = tag
        mx.eval(self.U, self.V)

    def __call__(self, x):
        delta = (x @ self.V.T) @ self.U.T
        return self.base(x) + self.scale * delta


@dataclass
class DurableAdapterHandle:
    layer_index: int
    parent: Any
    attr: str
    original: Any
    wrapper: DurableDeltaLinear


@dataclass
class ActiveDurableAdapter:
    """One activated adapter with everything needed for a proven rollback."""

    adapter_id: str
    handles: list[DurableAdapterHandle] = field(default_factory=list)
    baseline_probe_rows: list[dict[str, Any]] = field(default_factory=list)

    @property
    def active(self) -> bool:
        return bool(self.handles)


def _load_candidate_arrays(candidate_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np

    evidence = json.loads((candidate_dir / "evidence.json").read_text(encoding="utf-8"))
    with np.load(io.BytesIO((candidate_dir / "delta_weights.npz").read_bytes())) as data:
        arrays = {name: np.array(data[name]) for name in data.files}
    return evidence, arrays


def distill_proposal_to_adapter(
    proposal: dict[str, Any],
    *,
    adapter_dir: Path | str,
) -> dict[str, Any]:
    """Merge a proposal's candidate deltas into one durable adapter artifact.

    The merge computes the candidate MEAN delta per layer by rank
    concatenation: U' = [U₁ … Uₙ], V' = [V₁/n … Vₙ/n], so U'V' = Σ UᵢVᵢ/n
    exactly — no approximation, full provenance. Refuses on mixed targets,
    mixed checkpoint fingerprints, or unreadable candidates.
    """
    import numpy as np

    candidates = [
        record
        for record in proposal.get("candidates", [])
        if isinstance(record, dict) and record.get("valid")
    ]
    if len(candidates) < 2:
        return {"ok": False, "reason": "not_enough_valid_candidates"}

    per_layer_u: dict[int, list[Any]] = {}
    per_layer_v: dict[int, list[Any]] = {}
    targets: set[str] = set()
    fingerprints: set[str] = set()
    scales: set[float] = set()
    episode_ids: list[str] = []
    for record in candidates:
        candidate_dir = Path(record["path"])
        try:
            evidence, arrays = _load_candidate_arrays(candidate_dir)
        except (OSError, ValueError, KeyError) as exc:
            return {
                "ok": False,
                "reason": f"candidate_unreadable:{candidate_dir.name}:{type(exc).__name__}",
            }
        targets.add(str(evidence.get("target") or ""))
        inner = evidence.get("evidence")
        inner = inner if isinstance(inner, dict) else {}
        fingerprints.add(str(inner.get("checkpoint_fingerprint") or ""))
        scales.add(float(evidence.get("scale") or 1.0))
        episode_ids.append(str(evidence.get("episode_id") or candidate_dir.name))
        for name, value in arrays.items():
            if not name.startswith("layer") or "_" not in name:
                return {"ok": False, "reason": f"malformed_array_name:{name}"}
            layer_text, kind = name.split("_", 1)
            layer_index = int(layer_text.removeprefix("layer"))
            if kind == "U":
                per_layer_u.setdefault(layer_index, []).append(value)
            elif kind == "V":
                per_layer_v.setdefault(layer_index, []).append(value)
    if len(targets) != 1:
        return {"ok": False, "reason": f"mixed_targets:{sorted(targets)}"}
    if len(fingerprints) != 1:
        return {"ok": False, "reason": "mixed_checkpoint_fingerprints"}
    if not next(iter(fingerprints)):
        return {"ok": False, "reason": "checkpoint_fingerprint_missing"}
    if set(per_layer_u) != set(per_layer_v):
        return {"ok": False, "reason": "layer_uv_mismatch"}
    # One model ⇒ one geometry. Every U must share out_features and every V
    # in_features, across candidates AND layers — the leaked tiny-model
    # candidates (hidden 64) crashing a 32B attach is the failure this
    # refuses at the merge, before any model is touched.
    out_dims = {u.shape[0] for us in per_layer_u.values() for u in us}
    in_dims = {v.shape[1] for vs in per_layer_v.values() for v in vs}
    if len(out_dims) != 1 or len(in_dims) != 1:
        return {
            "ok": False,
            "reason": (
                f"candidate_dimension_mismatch:out={sorted(out_dims)}:in={sorted(in_dims)}"
            ),
        }

    n = len(candidates)
    merged: dict[str, Any] = {}
    for layer_index in sorted(per_layer_u):
        us, vs = per_layer_u[layer_index], per_layer_v[layer_index]
        # Dividing by n (not len(us)) is the exact mean over ALL candidates:
        # an episode that never touched this layer contributed zero delta.
        merged[f"layer{layer_index}_U"] = np.concatenate(us, axis=1)
        merged[f"layer{layer_index}_V"] = np.concatenate(
            [v / float(n) for v in vs], axis=0
        )

    buffer = io.BytesIO()
    np.savez(buffer, **merged)
    delta_payload = buffer.getvalue()
    adapter_id = f"latent_{proposal.get('domain', 'general')}_{int(time.time())}"
    manifest = {
        "schema": DURABLE_ADAPTER_SCHEMA,
        "adapter_id": adapter_id,
        "domain": str(proposal.get("domain") or "general"),
        "target": next(iter(targets)),
        "scale": max(scales),
        "checkpoint_fingerprint": next(iter(fingerprints)),
        "source_episode_ids": episode_ids,
        "candidate_count": n,
        "layers": sorted(per_layer_u),
        "delta_sha256": hashlib.sha256(delta_payload).hexdigest(),
        "created_at": time.time(),
        "activation_state": "distilled",
    }

    target_dir = Path(adapter_dir).expanduser() / adapter_id
    try:
        from core.governance_context import local_internal_governed_scope
        from core.runtime.file_write_gateway import get_file_write_gateway

        gateway = get_file_write_gateway()
        with local_internal_governed_scope("latent_adapter_distillation"):
            gateway.ensure_directory(target_dir, source="latent_adapter_distillation")
            gateway.write_bytes(
                target_dir / "delta_weights.npz",
                delta_payload,
                source="latent_adapter_distillation",
            )
            gateway.write_text(
                target_dir / "manifest.json",
                json.dumps(manifest, indent=1, sort_keys=True),
                source="latent_adapter_distillation",
            )
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        record_degradation(
            "latent_adapter_distillation",
            exc,
            action="refused durable adapter after artifact persist failed",
        )
        return {"ok": False, "reason": f"persist_failed:{type(exc).__name__}"}
    return {"ok": True, "adapter_dir": str(target_dir), "manifest": manifest}


def _attach(model, adapter_dir: Path) -> ActiveDurableAdapter:
    import numpy as np

    manifest = json.loads((adapter_dir / "manifest.json").read_text(encoding="utf-8"))
    with np.load(io.BytesIO((adapter_dir / "delta_weights.npz").read_bytes())) as data:
        arrays = {name: np.array(data[name]) for name in data.files}
    parent_attr, leaf_attr = _TARGET_ATTRS[manifest["target"]]
    inner = model.model
    active = ActiveDurableAdapter(adapter_id=manifest["adapter_id"])
    try:
        for layer_index in manifest["layers"]:
            layer = inner.layers[int(layer_index)]
            parent = getattr(layer, parent_attr)
            original = getattr(parent, leaf_attr)
            u = arrays[f"layer{layer_index}_U"]
            v = arrays[f"layer{layer_index}_V"]
            out_features, in_features = _linear_dims(original)
            if u.shape[0] != out_features or v.shape[1] != in_features:
                raise ValueError(
                    "adapter_model_dimension_mismatch: adapter "
                    f"({u.shape[0]}x{v.shape[1]}) vs module "
                    f"({out_features}x{in_features}) at layer {layer_index}"
                )
            wrapper = DurableDeltaLinear(
                original,
                arrays[f"layer{layer_index}_U"],
                arrays[f"layer{layer_index}_V"],
                scale=float(manifest.get("scale") or 1.0),
                tag=f"{manifest['adapter_id']}:{layer_index}",
            )
            setattr(parent, leaf_attr, wrapper)
            active.handles.append(
                DurableAdapterHandle(
                    layer_index=int(layer_index),
                    parent=parent,
                    attr=leaf_attr,
                    original=original,
                    wrapper=wrapper,
                )
            )
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        _detach(active)
        raise
    return active


def _detach(active: ActiveDurableAdapter) -> int:
    restored = 0
    for handle in reversed(active.handles):
        if getattr(handle.parent, handle.attr) is handle.original:
            continue
        setattr(handle.parent, handle.attr, handle.original)
        restored += 1
    active.handles = []
    return restored


def rollback_adapter(model, active: ActiveDurableAdapter) -> dict[str, Any]:
    """Detach and PROVE restoration against the activation-time baseline."""
    from core.learning.interference_battery import snapshot_probe_behavior

    restored = _detach(active)
    proof_rows = snapshot_probe_behavior(
        model,
        [row["probe"] for row in active.baseline_probe_rows] or None,
    )
    identical = (
        bool(active.baseline_probe_rows)
        and len(proof_rows) == len(active.baseline_probe_rows)
        and all(
            before["digest"] == after["digest"]
            for before, after in zip(
                active.baseline_probe_rows, proof_rows, strict=True
            )
        )
    )
    return {
        "restored_layers": restored,
        "rollback_proven": identical,
    }


def run_consolidation_train(
    proposal: dict[str, Any],
    model,
    *,
    adapter_dir: Path | str,
    tokenizer=None,
    heldout_solver: Callable[[Any, tuple[tuple[str, str], ...]], Mapping[str, str]]
    | None = None,
    heldout_spec: BatterySpec | None = None,
    heldout_evaluator_id: str = "",
) -> dict[str, Any]:
    """The complete durable-learning cycle for one proposal.

    validate held-out promotion contract → distill → interference battery
    (activation candidate applied and reverted while measuring) → paired
    sealed held-out regression → activate →
    return the live handle + a full receipt. On any gate failure the model
    is left untouched and the receipt says exactly why.
    """
    from core.learning.interference_battery import (
        run_interference_battery,
        snapshot_probe_behavior,
        stability_probes_for,
    )

    receipt: dict[str, Any] = {
        "schema": CONSOLIDATION_TRAIN_SCHEMA,
        "domain": str(proposal.get("domain") or "general"),
        "started_at": time.time(),
        "activated": False,
    }
    if heldout_solver is None:
        receipt["refusal_reason"] = "heldout_promotion_contract_missing"
        return receipt
    if not isinstance(heldout_spec, BatterySpec) or heldout_spec.size <= 0:
        receipt["refusal_reason"] = "heldout_promotion_spec_invalid"
        return receipt
    evaluator_id = str(heldout_evaluator_id or "").strip()
    if not evaluator_id:
        receipt["refusal_reason"] = "heldout_promotion_evaluator_missing"
        return receipt

    distilled = distill_proposal_to_adapter(proposal, adapter_dir=adapter_dir)
    receipt["distillation"] = {
        key: value for key, value in distilled.items() if key != "manifest"
    }
    if not distilled.get("ok"):
        receipt["refusal_reason"] = f"distillation:{distilled.get('reason')}"
        return receipt
    receipt["adapter_id"] = distilled["manifest"]["adapter_id"]
    adapter_path = Path(distilled["adapter_dir"])

    probes = stability_probes_for(model, tokenizer)
    trial = ActiveDurableAdapter(adapter_id=distilled["manifest"]["adapter_id"])

    def apply_change():
        nonlocal trial
        trial = _attach(model, adapter_path)

    try:
        battery = run_interference_battery(
            model,
            apply_change,
            lambda: _detach(trial),
            probes=probes,
        )
    except (ValueError, KeyError, IndexError, RuntimeError, TypeError) as exc:
        # Attach-time refusals (dimension mismatch, malformed artifact) leave
        # the model untouched (_attach unwinds transactionally) and become an
        # honest refusal, never a crashed train.
        _detach(trial)
        record_degradation(
            "latent_adapter_distillation",
            exc,
            action="refused adapter activation after attach-time validation failed",
        )
        receipt["refusal_reason"] = f"attach_failed:{exc}"
        return receipt
    receipt["interference_battery"] = {
        "verdict": battery["verdict"],
        "stable_fraction": battery["stable_fraction"],
        "probes": battery["probes"],
    }
    if battery["verdict"] != "PASS":
        receipt["refusal_reason"] = "interference_battery_failed"
        return receipt

    from core.learning.heldout_battery import (
        evaluate_sealed_responses,
        generate_battery,
    )
    from core.learning.permanent_distillation import PASS
    from core.learning.permanent_distillation_gates import frontier_regression_gate

    tasks = generate_battery(heldout_spec)
    prompts = tuple((task.task_id, task.prompt) for task in tasks)
    try:
        before = evaluate_sealed_responses(
            heldout_spec,
            heldout_solver(model, prompts),
            evaluator_id=evaluator_id,
        )
        holdout_trial = _attach(model, adapter_path)
        try:
            after = evaluate_sealed_responses(
                heldout_spec,
                heldout_solver(model, prompts),
                evaluator_id=evaluator_id,
            )
        finally:
            _detach(holdout_trial)
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation(
            "latent_adapter_distillation",
            exc,
            action="refused durable adapter after held-out promotion evidence failed",
        )
        receipt["refusal_reason"] = f"heldout_evaluation_failed:{exc}"
        return receipt

    heldout_gate = frontier_regression_gate(
        before=before.result,
        after=after.result,
        max_drop=_HELDOUT_REGRESSION_TOLERANCE,
    )
    receipt["heldout"] = {
        "schema": "aura.latent_adapter.heldout_promotion.v1",
        "battery": before.manifest,
        "evaluator_id": evaluator_id,
        "before": before.to_dict(),
        "after": after.to_dict(),
        "gate": heldout_gate,
    }
    if heldout_gate["verdict"] != PASS:
        receipt["refusal_reason"] = "heldout_regression"
        return receipt

    baseline_rows = snapshot_probe_behavior(model, probes)
    active = _attach(model, adapter_path)
    active.baseline_probe_rows = baseline_rows
    receipt["activated"] = True
    receipt["activated_layers"] = [handle.layer_index for handle in active.handles]
    receipt["active_adapter"] = active
    logger.info(
        "Durable latent adapter %s activated on %d layers (battery %s)",
        active.adapter_id,
        len(active.handles),
        battery["verdict"],
    )
    return receipt


__all__ = [
    "ActiveDurableAdapter",
    "CONSOLIDATION_TRAIN_SCHEMA",
    "DURABLE_ADAPTER_SCHEMA",
    "DurableDeltaLinear",
    "distill_proposal_to_adapter",
    "rollback_adapter",
    "run_consolidation_train",
]
