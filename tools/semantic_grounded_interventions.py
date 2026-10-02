"""Reversible matched lesions inside a caller-owned grounded decoder."""

from __future__ import annotations

import hashlib
import math
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

MODES = ("intact", "adapter_off", "relation_off", "both_off")


def intervention_contract():
    return {"schema": "aura.grounded_evidence_intervention_contract.v1",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "modes": MODES, "adapter_lesion": "exact_outer_residual_gate_zero",
        "relation_lesion": "zero_learned_evidence_weight",
        "source_and_candidates_unchanged": True, "caller_owns_exclusive_model_lane": True,
        "qualification_evidence": False, "serving_authority": False}


def _sites(owner):
    from mlx_lm.tuner.lora import LoRALinear
    from tools.semantic_native_adapter_layers import SemanticAdapterLinear
    from tools.semantic_native_adapters import native_adapter_keys_by_layer, native_adapter_scale

    layers = owner.suffix.layers
    if len(layers) != owner.plan["suffix_layers"]:
        raise ValueError("intervention suffix depth differs from fitted artifact")
    keys = native_adapter_keys_by_layer(SimpleNamespace(layers=layers), owner.plan)
    contract = owner.plan["adapter_contract"]
    sites, identities = [], set()
    for index, (layer, names) in enumerate(zip(layers, keys, strict=True)):
        modules = dict(layer.named_modules())
        for name in names:
            module = modules[name]
            if not isinstance(module, (LoRALinear, SemanticAdapterLinear)):
                raise ValueError("intervention site is not an installed native adapter")
            scale = native_adapter_scale(owner.plan, module)
            rank = module.lora_budget_rank if getattr(module, "kind", "lora") in {"square", "dense"} else module.lora_a.shape[-1]
            if (id(module) in identities or rank != contract["layer_ranks"][index]
                    or getattr(module, "kind", "lora") != contract["layer_kinds"][index]
                    or module.scale != scale or not math.isfinite(scale)):
                raise ValueError("intervention adapter site differs from fitted arithmetic")
            identities.add(id(module))
            sites.append((f"layers.{index}.{name}", module, scale))
    return tuple(sites)


def _parameters(owner):
    from mlx.utils import tree_flatten

    return {f"{prefix}.{name}": value for prefix, module in (
        ("suffix", owner.suffix), ("pointer", owner.engine.pointer))
        for name, value in tree_flatten(module.parameters())}


def _restore_parameters(owner, parameters):
    from mlx.utils import tree_unflatten

    for prefix, module in (("suffix", owner.suffix), ("pointer", owner.engine.pointer)):
        module.update(tree_unflatten([(name[len(prefix) + 1:], value)
            for name, value in parameters.items() if name.startswith(prefix + ".")]))
    current = _parameters(owner)
    if set(current) != set(parameters) or any(current[name] is not value for name, value in parameters.items()):
        owner._grounded_intervention_tainted = True
        raise ValueError("intervention cannot restore fitted parameter ownership; reload required")


@contextmanager
def grounded_evidence_intervention(owner, mode):
    from tools.semantic_grounded_development_archive import digest

    if (mode not in MODES or getattr(owner, "_grounded_intervention_active", False)
            or getattr(owner, "_grounded_intervention_tainted", False)):
        raise ValueError("intervention requires a declared mode and exclusive decoder ownership")
    if owner.verification.get("learned_checkpoint_selected") is not True or owner.verification["selected_step"] <= 0:
        raise ValueError("intervention requires a verified positive-step artifact")
    sites = _sites(owner)
    weight = owner.engine.evidence_weight
    if type(weight) not in (int, float) or not math.isfinite(weight) or weight <= 0:
        raise ValueError("intervention requires an intact positive learned-evidence weight")
    parameters = _parameters(owner)
    adapter_multiplier = 0. if mode in {"adapter_off", "both_off"} else 1.
    relation_multiplier = 0. if mode in {"relation_off", "both_off"} else 1.
    receipt = {"schema": "aura.grounded_evidence_intervention.v1", "contract": intervention_contract(),
        "mode": mode, "fit_receipt_sha256": owner.verification["fit_receipt_sha256"],
        "weights_sha256": owner.verification["weights_sha256"], "selected_step": owner.verification["selected_step"],
        "adapter_multiplier": adapter_multiplier, "relation_multiplier": relation_multiplier,
        "sites": [{"name": name, "kind": getattr(module, "kind", "lora"),
            "fitted_scale": scale, "applied_scale": scale * adapter_multiplier}
            for name, module, scale in sites], "intact_evidence_weight": weight,
        "applied_evidence_weight": weight * relation_multiplier, "parameter_arrays": len(parameters),
        "restored": False, "parameter_objects_unchanged": False,
        "qualification_evidence": False, "serving_authority": False}
    owner._grounded_intervention_active = True
    try:
        for _name, module, scale in sites:
            module.scale = scale * adapter_multiplier
        owner.engine.evidence_weight = weight * relation_multiplier
        yield receipt
    finally:
        applied_unchanged = (owner.engine.evidence_weight == weight * relation_multiplier
            and all(module.scale == scale * adapter_multiplier for _name, module, scale in sites))
        for _name, module, scale in sites:
            module.scale = scale
        owner.engine.evidence_weight = weight
        owner._grounded_intervention_active = False
        current = _parameters(owner)
        unchanged = set(parameters) == set(current) and all(current[name] is value for name, value in parameters.items())
        if not unchanged:
            try:
                _restore_parameters(owner, parameters)
            except Exception:
                owner._grounded_intervention_tainted = True
                raise
        receipt.update(restored=True, parameter_objects_unchanged=unchanged)
        receipt["receipt_sha256"] = digest(receipt)
        if not applied_unchanged or not unchanged:
            raise ValueError("intervention changed fitted parameters or applied arithmetic during decode")


class IntervenedGroundedDecoder:
    def __init__(self, decoder, *, mode):
        if mode not in MODES:
            raise ValueError("undeclared grounded intervention")
        self.decoder, self.mode, self.last_receipt = decoder, mode, None

    def decode(self, **public):
        self.last_receipt = None
        if set(public) != {"source_token_ids", "hidden_states", "public_inputs",
                "source_text_sha256", "model_basis_sha256", "search_time_limit_s"}:
            raise ValueError("intervened decoder accepts only the unchanged public request")
        with grounded_evidence_intervention(self.decoder.owner, self.mode) as intervention:
            result = self.decoder.decode(**public)
        if not isinstance(self.decoder.last_receipt, dict):
            raise ValueError("intervened decode has no measured selection receipt")
        self.last_receipt = {**self.decoder.last_receipt, "evidence_intervention": intervention}
        return result
