"""Joint native/pointer source fitting with exact model and bounded shard custody."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import time
import traceback
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path


class NativeCaptureBank(Mapping):
    def __init__(self, store, templates):
        self.store, self.templates = store, templates

    def __iter__(self):
        return iter(self.templates)

    def __len__(self):
        return len(self.templates)

    def __getitem__(self, identity):
        from core.learning.semantic_grounded_binding_engine import NativeGroundedCapture
        return NativeGroundedCapture(hidden=self.store[identity, 0, 0], **self.templates[identity])


def native_capture_template(item, evidence, depths):
    """Use public proposed operation/spans, not teacher argument registers."""
    instructions = item.ir.instructions
    anchors = (item.register_definition_spans or
               (*item.ir.input_spans, *(instruction.operation_span for instruction in instructions)))
    operations, mentions = {}, {}
    row = 0
    for instruction in instructions:
        for span in instruction.argument_spans:
            role = evidence.roles[row]
            operations[role.identity], mentions[role.identity] = instruction.operation_span, span
            row += 1
    if row != len(evidence.roles) or len(anchors) != len(evidence.context.referents):
        raise ValueError("native source spans differ from grounded role/candidate identities")
    return dict(source_id=evidence.source_id, context=evidence.context, roles=evidence.roles,
        depths=tuple(depths), operation_spans=operations, mention_spans=mentions,
        candidate_spans={record.key: span for record, span in zip(evidence.context.referents, anchors, strict=True)},
        adjacency=evidence.adjacency, relations=evidence.relations)


def fit_native_grounded_sources(training, calibration, source_items, directory, *, spec, rank=32, layers=8,
                               max_tokens=512, cache_bytes=512 * 1024 ** 2, seed=20260930,
                               relation_width=128, rounds=2, fit_options=None, resume=False,
                               prepare_only=False, adapter_options=None, source_basis=None):
    """Load one authorized model; recompute actual suffix states in gradients.

    Prefixes are immutable complete source-only sequences, sharded on disk.
    This function runs source fitting, never held scoring or live activation.
    """
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm import load

    from core.governance_context import local_internal_governed_scope
    from core.learning.frozen_decoder_prefix import FrozenDecoderPrefix, NativeDecoderSuffix
    from core.learning.frozen_state_store import FrozenStateStore
    from core.learning.semantic_grounded_binding_engine import (
        fit_grounded_binding,
        implementation_receipt,
        validate_grounded_fit_inputs,
    )
    from core.learning.semantic_native_program import source_text_from_tokens
    from core.learning.semantic_relational_pointer import RelationalBindingPointer
    from core.runtime.file_write_gateway import get_file_write_gateway
    from core.runtime.mlx_memory_guard import mlx_memory_envelope
    from core.runtime.model_lane_control import standalone_model_lane
    from tools.probe_semantic_native_prefix_branches import installed_arithmetic_basis
    from tools.semantic_native_adapters import (
        adapter_contract,
        install_native_adapters,
        native_adapter_parameter_estimate,
    )
    from tools.train_semantic_native_program import require_native_cortex_spec

    directory, fit_options = Path(directory), dict(fit_options or {})
    training, calibration, source_items = tuple(training), tuple(calibration), tuple(source_items)
    validate_grounded_fit_inputs(training, calibration, **fit_options)
    adapter_options = dict(adapter_options or {})
    source_basis = dict(source_basis or {})
    max_seconds = fit_options.get("max_seconds", 1800.)
    if (any(type(value) is not int or value < 1 for value in (rank, layers, max_tokens, cache_bytes))
            or not math.isfinite(max_seconds) or not 0 < max_seconds <= 14400
            or type(seed) is not int or not 0 <= seed < 2 ** 32
            or type(prepare_only) is not bool or prepare_only and resume
            or not set(adapter_options) <= {"kind", "layer_ranks", "layer_kinds", "experts", "scaling", "alpha"}):
        raise ValueError("native grounded fit needs bounded source geometry")
    if (not set(source_basis) <= {"parent_sha256", "source_report_sha256", "folds_sha256", "bank_plan_sha256"}
            or any(not isinstance(value, str) or len(value) != 64
                   or any(character not in "0123456789abcdef" for character in value)
                   for value in source_basis.values())):
        raise ValueError("native grounded source basis needs exact artifact digests")
    examples = {item.evidence.source_id: item for item in (*training, *calibration)}
    items = {item.ir.source_text_sha256: item for item in source_items}
    if (not training or not calibration or len(examples) != len(training) + len(calibration)
            or set(items) != set(examples) or any(item.split != "train" for item in items.values())
            or type(resume) is not bool or directory.exists() and not resume
            or resume and not (directory / "resume.json").is_file()
            or any(not item.ir.source_token_ids or len(item.ir.source_token_ids) > max_tokens for item in items.values())):
        raise ValueError("native grounded source custody, token bound or fresh fit directory differs")
    adaptation = adapter_contract(rank=rank, layers=layers, sites="native_topology_v1",
        **{"scaling": "alpha_over_sqrt_rank_v1", "alpha": float(rank), **adapter_options})
    arithmetic = installed_arithmetic_basis()
    if arithmetic.get("MLX_ENABLE_TF32") != "0":
        raise ValueError("native grounded fitting requires MLX_ENABLE_TF32=0 at process launch")
    if require_native_cortex_spec().descriptor_sha256 != spec.descriptor_sha256:
        raise ValueError("native grounded descriptor changed before preparation")
    sources = []
    for identity, item in sorted(items.items()):
        template = native_capture_template(item, examples[identity].evidence, tuple(range(layers)))
        spans = {name: sorted((key, span.start, span.end) for key, span in template[name].items())
                 for name in ("operation_spans", "mention_spans", "candidate_spans")}
        for name in ("operation_spans", "mention_spans", "candidate_spans"):
            for span in template[name].values():
                span.validate_bound(len(item.ir.source_token_ids))
        example = examples[identity]
        sources.append({"source_id": identity, "source_tokens_sha256": hashlib.sha256(
            json.dumps(item.ir.source_token_ids).encode()).hexdigest(), "spans": spans,
            "source_observation": example.evidence.receipt(), "positives": example.positives,
            "graphs": example.graphs, "positive_graphs": example.positive_graphs,
            "environment": example.environment,
            "retention_scores": example.retention_scores.tolist() if example.retention_scores is not None else None})
    plan = {"rank": rank, "suffix_layers": layers, "adapter_keys": adaptation["keys"],
        "adapter_contract": adaptation, "model_descriptor_sha256": spec.descriptor_sha256,
        "model_path": str(spec.model_path), "pointer_sha256": spec.pointer_sha256,
        "source_token_only": True, "implementation": implementation_receipt(), "seed": seed,
        "installed_arithmetic": arithmetic, "precision": "native", "prefix_strategy": "full",
        "suffix_layer_execution": "differentiable_native_ops_v1",
        "fit_ids": sorted(item.evidence.source_id for item in training),
        "calibration_ids": sorted(item.evidence.source_id for item in calibration),
        "max_tokens": max_tokens, "prefix_cache_bytes": cache_bytes,
        "relation_width": relation_width, "rounds": rounds,
        "source_basis": source_basis,
        "source_supervision_sha256": hashlib.sha256(json.dumps(sources, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
        "fit_options": {**fit_options, "equivariance_pairs": [asdict(pair) for pair in fit_options.get("equivariance_pairs", ())]}}
    plan = json.loads(json.dumps(plan, allow_nan=False))
    plan_hash = hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    config = json.loads((spec.model_path / "config.json").read_text())
    projection = native_adapter_parameter_estimate(config, adaptation)
    if projection is None:
        raise ValueError("native grounded adapter topology has no measured parameter projection")
    geometry = config.get("text_config", config)
    pointer_geometry = RelationalBindingPointer(geometry["hidden_size"], depths=layers,
        relation_width=relation_width, rounds=rounds)
    pointer_parameters = sum(value.size for _, value in tree_flatten(pointer_geometry.trainable_parameters()))
    nuisance_parameters = ((pointer_geometry.feature_blocks * relation_width + 1)
        * len({item.environment for item in training}) if fit_options.get("domain_reversal", 0.) else 0)
    trainable_bytes = projection["five_float32_copies_bytes"] + 20 * (pointer_parameters + nuisance_parameters)
    del pointer_geometry
    custody = directory.parent / (directory.name + "-native-custody")
    if custody.exists():
        if json.loads((custody / "plan.json").read_bytes()) != {**plan, "plan_sha256": plan_hash}:
            raise ValueError("native grounded restart plan or source supervision differs")
        if (custody / "completion.json").exists():
            raise ValueError("native grounded fit is already complete; do not repeat it")
    else:
        with local_internal_governed_scope("grounded_native_fit_plan", domain="file_write"):
            if not get_file_write_gateway().write_bytes_if_absent(custody / "plan.json",
                json.dumps({**plan, "plan_sha256": plan_hash}, indent=2).encode(),
                source="grounded_native_fit_plan", mode=0o400):
                raise FileExistsError(custody / "plan.json")
    if prepare_only:
        body = {"schema": "aura.grounded_native_preparation.v1", "plan_sha256": plan_hash,
            "adapter_projection": projection, "pointer_parameters": pointer_parameters,
            "nuisance_parameters": nuisance_parameters,
            "trainable_five_float32_copies_bytes": trainable_bytes,
            "activation_memory_is_measured": False,
            "fit_source_count": len(training), "calibration_source_count": len(calibration),
            "maximum_source_tokens": max(len(item.ir.source_token_ids) for item in items.values()),
            "model_weights_loaded": False, "held_sources_scored": False,
            "semantic_success": None, "qualification_evidence": False, "serving_authority": False}
        return None, {**body, "receipt_sha256": hashlib.sha256(json.dumps(body, sort_keys=True,
            separators=(",", ":")).encode()).hexdigest()}
    started = time.monotonic()
    def execute(envelope):
        if (require_native_cortex_spec().descriptor_sha256 != spec.descriptor_sha256
                or installed_arithmetic_basis() != arithmetic):
            raise ValueError("native grounded descriptor changed before model acquisition")
        model, tokenizer = load(str(spec.model_path))
        model.freeze()
        model.eval()
        prefix = FrozenDecoderPrefix(model, split_at=len(model.layers) - layers)
        suffix = NativeDecoderSuffix(model, split_at=len(model.layers) - layers)
        mx.random.seed(seed)
        install_native_adapters(model, plan)
        # Hybrid delta-net inference kernels do not implement backward passes.
        for layer in suffix.layers:
            layer.train()
        observed = sum(value.size for _, value in tree_flatten(suffix.trainable_parameters()))
        if observed != projection["trainable_parameters"] or any("lora_" not in key for key, _ in tree_flatten(model.trainable_parameters())):
            raise ValueError("native grounded adapter ownership differs from its projected sites")
        if mx.get_active_memory() + trainable_bytes + cache_bytes > envelope.memory_bytes:
            raise MemoryError("native grounded fixed residency exceeds the host envelope")
        sequence_digests = {(identity, 0, 0): hashlib.sha256(json.dumps(item.ir.source_token_ids).encode()).hexdigest()
                            for identity, item in items.items()}
        store = (FrozenStateStore.open_existing(custody / "prefixes", plan_sha256=plan_hash,
                    max_resident_bytes=cache_bytes, sequence_digests=sequence_digests) if resume else
                 FrozenStateStore(custody / "prefixes", plan_sha256=plan_hash, max_resident_bytes=cache_bytes))
        templates = {}
        for identity, item in sorted(items.items()):
            if time.monotonic() - started >= max_seconds:
                raise TimeoutError("native grounded acquisition reached its declared bound")
            source_text_from_tokens(item, tokenizer)
            if resume:
                hidden = store[identity, 0, 0]
            else:
                tokens = mx.array([item.ir.source_token_ids], dtype=mx.int32)
                hidden = prefix.capture(tokens)
                store.write_source(identity, {(identity, 0, 0): hidden},
                    sequence_digests={(identity, 0, 0): sequence_digests[identity, 0, 0]})
            templates[identity] = native_capture_template(item, examples[identity].evidence, tuple(range(layers)))
            print(json.dumps({"stage": "grounded_prefix_captured", "source": identity,
                              "source_tokens": len(item.ir.source_token_ids), "active_bytes": mx.get_active_memory()}), flush=True)
        remaining = max_seconds - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("native grounded acquisition exhausted its complete fit allowance")
        pointer = RelationalBindingPointer(hidden.shape[-1], depths=layers, relation_width=relation_width, rounds=rounds)
        engine, report = fit_grounded_binding(pointer, training, calibration, directory,
            native_suffix=suffix, native_captures=NativeCaptureBank(store, templates), native_contract=plan,
            **{**fit_options, "max_seconds": remaining, "resume": resume})
        if (require_native_cortex_spec().descriptor_sha256 != spec.descriptor_sha256
                or implementation_receipt() != plan["implementation"] or installed_arithmetic_basis() != arithmetic):
            raise ValueError("native grounded model or implementation changed during source fitting")
        receipt = {"schema": "aura.grounded_native_acquisition.v1", "plan_sha256": plan_hash,
            "fit_receipt_sha256": report["receipt_sha256"], "adapter_projection": projection,
            "observed_adapter_parameters": observed, "state_store": store.receipt(),
            "memory_envelope": envelope.to_receipt(), "peak_memory_bytes": mx.get_peak_memory(),
            "held_sources_scored": False, "serving_authority": False}
        return report, receipt

    if resume and (directory / "report.json").exists():
        from core.learning.semantic_grounded_binding_engine import (
            NativeGroundedCapture,
            verify_grounded_fit_checkpoint,
        )
        report = verify_grounded_fit_checkpoint(directory)
        if (report.get("native_contract") != plan
                or require_native_cortex_spec().descriptor_sha256 != spec.descriptor_sha256):
            raise ValueError("completed native fit recovery changed its model or source contract")
        supervision = {split: [{"source_id": item.evidence.source_id, "positives": item.positives,
            "graphs": item.graphs, "positive_graphs": item.positive_graphs, "environment": item.environment,
            "retention_scores": item.retention_scores.tolist() if item.retention_scores is not None else None}
            for item in examples] for split, examples in (("training", training), ("calibration", calibration))}
        if json.loads(json.dumps(supervision)) != report["supervision"]:
            raise ValueError("completed native fit recovery changed its source supervision")
        selected = mx.load(str(directory / "selected.safetensors"))
        observed = sum(value.size for key, value in selected.items() if key.startswith("native_suffix."))
        if (observed != projection["trainable_parameters"]
                or any("lora_" not in key for key in selected if key.startswith("native_suffix."))):
            raise ValueError("completed native fit recovery changed its adapter inventory")
        sequence_digests = {(identity, 0, 0): hashlib.sha256(json.dumps(item.ir.source_token_ids).encode()).hexdigest()
                            for identity, item in items.items()}
        store = FrozenStateStore.open_existing(custody / "prefixes", plan_sha256=plan_hash,
            max_resident_bytes=cache_bytes, sequence_digests=sequence_digests)
        captures = [NativeGroundedCapture(hidden=store[identity, 0, 0],
            **native_capture_template(items[identity], examples[identity].evidence, tuple(range(layers)))).receipt()
            for identity in sorted(items)]
        if json.loads(json.dumps(captures)) != report["native_captures"]:
            raise ValueError("completed native fit recovery changed its retained prefixes or public spans")
        receipt = {"schema": "aura.grounded_native_acquisition.v1", "plan_sha256": plan_hash,
            "fit_receipt_sha256": report["receipt_sha256"], "adapter_projection": projection,
            "observed_adapter_parameters": observed, "state_store": store.receipt(),
            "memory_envelope": None, "peak_memory_bytes": None,
            "completion_recovery": "verified_saved_fit_without_model_loading",
            "held_sources_scored": False, "serving_authority": False}
        del store, captures, selected
        gc.collect()
        mx.clear_cache()
    else:
        with (standalone_model_lane(owner_id=f"grounded-native:{directory.name}", model_path=str(spec.model_path),
                purpose="training", preemptible=False, require_exclusive=True, allow_owner_eviction=False,
                metadata={"tool": "fit_semantic_grounded_binding", "production_effect": False}),
              mlx_memory_envelope(fraction=.80) as envelope):
            try:
                report, receipt = execute(envelope)
            except BaseException as error:
                # Completed traceback frames can retain the entire native model.
                traceback.clear_frames(error.__traceback__)
                raise
            finally:
                gc.collect()
                mx.clear_cache()
    with local_internal_governed_scope("grounded_native_fit_receipt", domain="file_write"):
        if not get_file_write_gateway().write_bytes_if_absent(custody / "completion.json",
            json.dumps(receipt, indent=2).encode(), source="grounded_native_fit_receipt", mode=0o400):
            raise FileExistsError(custody / "completion.json")
    return None, report
