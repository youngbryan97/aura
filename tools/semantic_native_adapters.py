"""Share the declared adapter capacity and scaling across training and replay."""

from __future__ import annotations

import math

ADAPTER_SITES = {
    "binding_v1": ("self_attn.q_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.down_proj"),
    "attention_mlp_v1": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                         "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"),
    "native_topology_v1": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                           "linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.in_proj_b",
                           "linear_attn.in_proj_a", "linear_attn.out_proj",
                           "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"),
}
SCALING_POLICIES = ("direct_v1", "alpha_over_rank_v1", "alpha_over_sqrt_rank_v1")
ADAPTER_KINDS = ("lora", "silu", "product", "routed", "dora", "square", "dense")
ADAPTER_IMPLEMENTATION_PATHS = ("tools/semantic_native_adapters.py",
                                "tools/semantic_native_adapter_layers.py")


def adapter_contract(*, rank, layers, sites="binding_v1", scaling="direct_v1", alpha=16.,
                     kind="lora", layer_ranks=None, experts=1, layer_kinds=None):
    layer_kinds = None if layer_kinds is None else list(layer_kinds)
    if (type(rank) is not int or rank < 1 or type(layers) is not int or layers < 1
            or sites not in ADAPTER_SITES or scaling not in SCALING_POLICIES
            or type(alpha) not in {int, float} or not math.isfinite(alpha) or alpha <= 0
            or kind not in ADAPTER_KINDS or type(experts) is not int or experts < 1
            or (kind != "routed" and (layer_kinds is None or "routed" not in layer_kinds) and experts != 1)):
        raise ValueError("invalid native adapter capacity")
    if layer_ranks is not None:
        layer_ranks = list(layer_ranks)
        if len(layer_ranks) != layers or any(type(value) is not int or value < 1 for value in layer_ranks):
            raise ValueError("native per-layer rank schedule differs from suffix depth")
    if layer_kinds is not None and (len(layer_kinds) != layers or any(value not in ADAPTER_KINDS for value in layer_kinds)):
        raise ValueError("native per-layer function classes differ from suffix depth")
    divisor = rank if scaling == "alpha_over_rank_v1" else math.sqrt(rank) if scaling == "alpha_over_sqrt_rank_v1" else 1.
    contract = {"schema": "aura.native_adapter_contract.v1", "rank": rank, "suffix_layers": layers,
            "sites": sites, "keys": list(ADAPTER_SITES[sites]), "scaling": scaling,
            "alpha": float(alpha), "effective_scale": alpha / divisor,
            "kind": kind, "backbone_frozen": True}
    if kind != "lora" or layer_ranks is not None or layer_kinds is not None or sites == "native_topology_v1":
        contract.update(schema="aura.native_adapter_contract.v2", experts=experts,
                        layer_ranks=layer_ranks or [rank] * layers,
                        mergeable=all(value == "lora" for value in (layer_kinds or [kind])),
                        baseline_lesion="outer_residual_gate_v1")
        if layer_kinds is not None:
            contract["layer_kinds"] = layer_kinds
        if any(value in {"square", "dense"} for value in (layer_kinds or [kind])):
            contract["rank_semantics"] = "parameter_budget_proxy_not_update_rank"
        if sites == "native_topology_v1":
            contract["site_resolution"] = "exactly_one_attention_topology_and_complete_dense_mlp_v1"
        contract["layer_effective_scales"] = [
            alpha / (value if scaling == "alpha_over_rank_v1" else
                     math.sqrt(value) if scaling == "alpha_over_sqrt_rank_v1" else 1.)
            for value in contract["layer_ranks"]]
    return contract


def adapter_config_from_plan(plan):
    contract = plan.get("adapter_contract")
    if contract is None:
        return {"rank": plan["rank"], "scale": 16., "dropout": 0., "keys": plan["adapter_keys"]}
    if not isinstance(contract, dict):
        raise ValueError("native adapter contract differs")
    extra = ({"kind": contract.get("kind"), "layer_ranks": contract.get("layer_ranks"),
              "experts": contract.get("experts"), "layer_kinds": contract.get("layer_kinds")}
             if contract.get("schema") == "aura.native_adapter_contract.v2" else {})
    expected = adapter_contract(rank=contract.get("rank"), layers=contract.get("suffix_layers"),
        sites=contract.get("sites"), scaling=contract.get("scaling"), alpha=contract.get("alpha"), **extra)
    if (expected != contract or contract["rank"] != plan["rank"]
            or contract["suffix_layers"] != plan["suffix_layers"] or contract["keys"] != plan["adapter_keys"]):
        raise ValueError("native adapter plan geometry differs")
    return {"rank": contract["rank"], "scale": contract["effective_scale"], "dropout": 0.,
            "keys": contract["keys"]}


def native_adapter_keys_by_layer(model, plan):
    """Resolve declared site roles, not silently skip missing projections."""
    config = adapter_config_from_plan(plan)
    layers = plan["suffix_layers"]
    if type(layers) is not int or not 1 <= layers <= len(model.layers):
        raise ValueError("native adapter depth exceeds the model")
    resolved = []
    for layer in model.layers[-layers:]:
        modules = dict(layer.named_modules())
        if plan.get("adapter_contract", {}).get("sites") == "native_topology_v1":
            dense = tuple(key for key in config["keys"] if key.startswith("self_attn."))
            linear = tuple(key for key in config["keys"] if key.startswith("linear_attn."))
            mlp = tuple(key for key in config["keys"] if key.startswith("mlp."))
            available = [keys for keys in (dense, linear) if all(key in modules for key in keys)]
            if len(available) != 1 or any(key not in modules for key in mlp):
                raise ValueError("native suffix has an unsupported or ambiguous attention/MLP topology")
            keys = (*available[0], *mlp)
        else:
            keys = tuple(config["keys"])
            if any(key not in modules for key in keys):
                raise ValueError("requested native adapter sites are absent from a suffix layer")
        resolved.append(keys)
    return tuple(resolved)


def install_native_adapters(model, plan):
    from mlx_lm.tuner.utils import linear_to_lora_layers

    config = adapter_config_from_plan(plan)
    layers = plan["suffix_layers"]
    resolved = native_adapter_keys_by_layer(model, plan)
    contract = plan.get("adapter_contract", {})
    if contract.get("schema") != "aura.native_adapter_contract.v2":
        linear_to_lora_layers(model, layers, config)
        return
    import mlx.nn as nn
    from mlx.utils import tree_unflatten
    from mlx_lm.tuner.lora import LoRALinear

    from tools.semantic_native_adapter_layers import SemanticAdapterLinear

    selected = model.layers[-layers:]
    # Validate all sites before mutating even the first layer.
    for layer, keys in zip(selected, resolved, strict=True):
        modules = dict(layer.named_modules())
        if any(not isinstance(modules[key], (nn.Linear, nn.QuantizedLinear)) for key in keys):
            raise ValueError("nonlinear native adapter requires supported linear sites")
    kinds = contract.get("layer_kinds", [contract["kind"]] * layers)
    for layer, keys, rank, scale, kind in zip(selected, resolved, contract["layer_ranks"],
            contract["layer_effective_scales"], kinds, strict=True):
        modules = dict(layer.named_modules())
        replacements = []
        for key in keys:
            kwargs = dict(r=rank, scale=scale, dropout=config["dropout"])
            if kind == "lora":
                adapted = LoRALinear.from_base(modules[key], **kwargs)
            else:
                adapted = SemanticAdapterLinear.from_base(modules[key], kind=kind,
                                                         experts=contract["experts"] if kind == "routed" else 1, **kwargs)
            replacements.append((key, adapted))
        layer.update_modules(tree_unflatten(replacements))


def native_adapter_scale(plan, module=None):
    config = adapter_config_from_plan(plan)
    contract = plan.get("adapter_contract", {})
    if module is not None and contract.get("schema") == "aura.native_adapter_contract.v2":
        rank = (module.lora_budget_rank if getattr(module, "kind", "lora") in {"square", "dense"}
                else module.lora_a.shape[-1])
        if rank not in contract["layer_ranks"]:
            raise ValueError("native module rank differs from the fitted contract")
        return contract["layer_effective_scales"][contract["layer_ranks"].index(rank)]
    return config["scale"]


def native_optimizer(plan):
    import mlx.optimizers as optim

    rate, decay = plan["learning_rate"], plan["weight_decay"]
    ratio = plan.get("adapter_b_learning_rate_ratio", 1.)
    if type(ratio) not in {int, float} or not math.isfinite(ratio) or ratio <= 0:
        raise ValueError("native adapter learning-rate ratio must be positive and finite")
    if ratio != 1. and plan.get("adapter_contract", {}).get("kind") in {"square", "dense"}:
        raise ValueError("B-factor rate ratio does not apply to a single-matrix adapter")
    if ratio == 1.:
        return optim.AdamW(learning_rate=rate, weight_decay=decay)
    return optim.MultiOptimizer([
        optim.AdamW(learning_rate=rate * ratio, weight_decay=decay),
        optim.AdamW(learning_rate=rate, weight_decay=decay),
    ], [lambda name, _value: name.rsplit(".", 1)[-1] == "lora_b"])


def native_router_regularized_objective(model, objective, *, weight):
    """Balance every actual routed call, not a fabricated proxy activation."""
    if type(weight) not in {int, float} or not math.isfinite(weight) or weight < 0:
        raise ValueError("native router regularization must be finite and nonnegative")
    if weight == 0:
        return objective()
    from tools.semantic_native_adapter_layers import SemanticAdapterLinear

    modules = [module for _name, module in model.named_modules()
               if isinstance(module, SemanticAdapterLinear) and module.kind == "routed"]
    if not modules:
        raise ValueError("native router regularization has no routed modules")
    for module in modules:
        module._router_balance_terms = []
        module._record_router_balance = True
    try:
        loss = objective()
        terms = [term for module in modules for term in module._router_balance_terms]
        if not terms:
            raise ValueError("native router regularization observed no routed calls")
        return loss + weight * sum(terms) / len(terms)
    finally:
        for module in modules:
            module._record_router_balance = False
            module._router_balance_terms = []


def native_adapter_parameter_estimate(config, contract):
    """Declared Qwen geometry, excluding unmeasured activation residency.

    Unknown topologies return None, never zero. Five float32 copies include
    weights, gradients, Adam moments and the preserved baseline adapter state;
    they do not include backbone, activations, cached states or temporaries.
    """
    geometry = config.get("text_config", config)
    if geometry.get("model_type") not in {"qwen2", "qwen3", "qwen3_5_text"}:
        return None
    width, intermediate = geometry["hidden_size"], geometry["intermediate_size"]
    depth = geometry["num_hidden_layers"]
    if contract["suffix_layers"] >= depth:
        raise ValueError("native adapters require a nonempty frozen prefix")
    head_dim = geometry.get("head_dim", width // geometry["num_attention_heads"])
    q_width = geometry["num_attention_heads"] * head_dim
    kv_width = geometry["num_key_value_heads"] * head_dim
    query_multiplier = 2 if geometry["model_type"] == "qwen3_5_text" else 1
    dense = {"self_attn.q_proj": (width, q_width * query_multiplier),
             "self_attn.k_proj": (width, kv_width), "self_attn.v_proj": (width, kv_width),
             "self_attn.o_proj": (q_width, width)}
    mlp = {"mlp.gate_proj": (width, intermediate), "mlp.up_proj": (width, intermediate),
           "mlp.down_proj": (intermediate, width)}
    if geometry.get("model_type") == "qwen3_5_text":
        if geometry.get("num_experts", 0):
            return None
        key = geometry["linear_num_key_heads"] * geometry["linear_key_head_dim"]
        value = geometry["linear_num_value_heads"] * geometry["linear_value_head_dim"]
        heads = geometry["linear_num_value_heads"]
        linear = {"linear_attn.in_proj_qkv": (width, 2 * key + value),
                  "linear_attn.in_proj_z": (width, value), "linear_attn.in_proj_b": (width, heads),
                  "linear_attn.in_proj_a": (width, heads), "linear_attn.out_proj": (value, width)}
        layer_types = geometry.get("layer_types") or [
            "full_attention" if (index + 1) % geometry["full_attention_interval"] == 0 else "linear_attention"
            for index in range(depth)]
    else:
        linear, layer_types = {}, ["full_attention"] * depth
    if len(layer_types) != depth or any(kind not in {"full_attention", "linear_attention"} for kind in layer_types):
        return None
    ranks = contract.get("layer_ranks", [contract["rank"]] * contract["suffix_layers"])
    rows = []
    for index, rank in zip(range(depth - contract["suffix_layers"], depth), ranks, strict=True):
        modules = {**(dense if layer_types[index] == "full_attention" else linear), **mlp}
        keys = tuple(modules) if contract["sites"] == "native_topology_v1" else tuple(contract["keys"])
        if any(key not in modules for key in keys):
            raise ValueError("declared adapter site is absent from its native layer type")
        count = 0
        for key in keys:
            input_dim, output_dim = modules[key]
            basic = rank * (input_dim + output_dim)
            kind = contract.get("layer_kinds", [contract["kind"]] * contract["suffix_layers"])[index - depth + contract["suffix_layers"]]
            if kind == "dense":
                count += input_dim * output_dim
            elif kind == "square":
                count += max(1, min(input_dim, output_dim, math.isqrt(basic))) ** 2
            elif kind == "routed":
                count += contract["experts"] * (basic + input_dim)
            else:
                count += basic + (input_dim * rank if kind == "product" else output_dim if kind == "dora" else 0)
        rows.append({"layer": index, "type": layer_types[index], "rank": rank,
                     "sites": list(keys), "trainable_parameters": count})
    count = sum(row["trainable_parameters"] for row in rows)
    return {"schema": "aura.native_adapter_memory_projection.v1", "layers": rows,
            "trainable_parameters": count, "five_float32_copies_bytes": count * 20,
            "activation_bytes": None, "backbone_bytes": None, "fit_is_guaranteed": False}
