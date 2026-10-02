"""Source-fitted relational evidence and the existing joint binding solver.

Representation acquisition must supply observed, aligned states. Neither
training labels nor construction families are available to inference. This
module grants no live-model serving authority.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map, tree_unflatten

from core.governance_context import local_internal_governed_scope
from core.learning.semantic_context_binding import bind_context_roles, candidate_cost
from core.learning.semantic_relational_pointer import (
    RelationalBindingPointer,
    pointer_choice_loss,
    pointer_context_costs,
    pointer_role_margin_loss,
    semantic_role_features,
)
from core.runtime.file_read_gateway import open_stable_readonly_binary
from core.runtime.file_write_gateway import get_file_write_gateway

IMPLEMENTATION_PATHS = (
    "core/brain/llm/decoder_topology.py",
    "core/learning/procedure_induction.py",
    "core/learning/semantic_program_floor.py",
    "core/learning/semantic_native_fit_sampling.py",
    "core/learning/semantic_grounded_binding_engine.py",
    "core/learning/semantic_native_operation_field.py",
    "core/learning/semantic_grounded_program_objective.py",
    "core/learning/semantic_conditioned_relations.py",
    "core/learning/semantic_relational_pointer.py",
    "core/learning/semantic_binding_invariance.py",
    "core/learning/semantic_context_binding.py",
    "core/learning/semantic_argument_optimization.py",
    "core/learning/semantic_argument_chart.py",
    "core/learning/semantic_native_program.py",
    "core/learning/semantic_grounded_binding_acquisition.py",
    "core/learning/semantic_grounded_chart_bridge.py",
    "core/learning/semantic_semasiographic.py",
    "core/learning/semantic_temporal_constraints.py",
    "core/learning/semantic_symbol_inquiry.py",
    "core/learning/semantic_diagram_workspace.py",
    "core/learning/semantic_discourse_smoothing.py",
    "core/learning/semantic_program_transducer_fitting.py",
    "core/learning/semantic_program_compositional_transducer.py",
    "core/learning/frozen_decoder_prefix.py",
    "core/learning/frozen_state_store.py",
    "tools/semantic_native_adapters.py",
    "tools/semantic_native_adapter_layers.py",
    "tools/semantic_grounded_native_fit.py",
    "tools/semantic_grounded_program_pool.py",
    "tools/semantic_grounded_native_decode.py",
    "tools/semantic_native_execution.py",
    "tools/probe_semantic_native_prefix_branches.py",
)


def implementation_receipt():
    root = Path(__file__).resolve().parents[2]
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in IMPLEMENTATION_PATHS}


def _write_binding_artifact(path, payload):
    path = Path(path)
    if (path.name not in {"selected.safetensors", "training-nuisance.safetensors", "report.json", "resume.json"}
            and re.fullmatch(r"checkpoint-(0|[1-9][0-9]*)\.safetensors", path.name) is None
            and re.fullmatch(r"resume-[a-f0-9]{64}\.safetensors", path.name) is None):
        raise ValueError("grounded fit artifact is outside its fixed schema namespace")
    if not (path.parent / "fit-owner.json").is_file():
        raise ValueError("grounded fit artifact has no claimed fresh run namespace")
    with local_internal_governed_scope("grounded_binding_source_fit", domain="file_write"):
        get_file_write_gateway().write_bytes(path, payload, source="grounded_binding_source_fit")


def _save_grounded_restart(directory, state, model, optimizer, selected):
    """Publish one complete optimizer generation before moving its pointer."""
    tensors = {"model/" + key: value for key, value in tree_flatten(model.trainable_parameters())}
    tensors.update({"optimizer/" + key: value for key, value in tree_flatten(optimizer.state)})
    tensors.update({"selected/" + key: value for key, value in selected.items()})
    tensors.update({"rng/" + str(index): value for index, value in enumerate(mx.random.state)})
    mx.eval(tensors)
    if any(not mx.all(mx.isfinite(value)).item() for value in tensors.values()):
        raise ValueError("grounded restart contains nonfinite state")
    body = {**state, "schema": "aura.grounded_optimizer_restart.v1",
            "inventory": {key: [list(value.shape), str(value.dtype)] for key, value in tensors.items()}}
    stream = io.BytesIO()
    mx.save_safetensors(stream, tensors, metadata={"state": json.dumps(body, sort_keys=True, allow_nan=False)})
    payload = stream.getvalue()
    sha = hashlib.sha256(payload).hexdigest()
    name = f"resume-{sha}.safetensors"
    path = directory / name
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError("grounded restart generation collision")
    else:
        _write_binding_artifact(path, payload)
    _write_binding_artifact(directory / "resume.json", json.dumps({
        "schema": "aura.grounded_restart_pointer.v1", "file": name, "sha256": sha,
        "identity": state["identity"], "step": state["step"]}, sort_keys=True).encode())


def _load_grounded_restart(directory, identity, model, optimizer, artifact):
    def read(path, bound):
        with open_stable_readonly_binary(path, max_bytes=bound) as (handle, observed):
            payload = handle.read(bound + 1)
            if len(payload) != observed.size or len(payload) > bound:
                raise ValueError("grounded restart file changed or exceeded its bound")
            return payload

    pointer = json.loads(read(directory / "resume.json", 16384))
    if (set(pointer) != {"schema", "file", "sha256", "identity", "step"}
            or pointer["schema"] != "aura.grounded_restart_pointer.v1"
            or pointer["identity"] != identity
            or re.fullmatch(r"[a-f0-9]{64}", pointer["sha256"]) is None
            or pointer["file"] != f"resume-{pointer['sha256']}.safetensors"):
        raise ValueError("grounded restart pointer or fit identity differs")
    expected_partitions = {
        "model": dict(tree_flatten(model.trainable_parameters())),
        "optimizer": dict(tree_flatten(optimizer.state)),
        "selected": dict(tree_flatten(artifact.trainable_parameters())),
        "rng": {str(index): value for index, value in enumerate(mx.random.state)},
    }
    payload_bound = sum(value.nbytes for partition in expected_partitions.values()
                        for value in partition.values()) + 8 * 1024 ** 2
    payload = read(directory / pointer["file"], payload_bound)
    if hashlib.sha256(payload).hexdigest() != pointer["sha256"]:
        raise ValueError("grounded restart generation checksum differs")
    tensors, metadata = mx.load(io.BytesIO(payload), format="safetensors", return_metadata=True)
    state = json.loads(metadata["state"])
    if (state.get("schema") != "aura.grounded_optimizer_restart.v1" or state.get("identity") != identity
            or state.get("step") != pointer["step"] or type(state["step"]) is not int
            or set(tensors) != set(state["inventory"])
            or any([list(value.shape), str(value.dtype)] != state["inventory"][key]
                   or not mx.all(mx.isfinite(value)).item() for key, value in tensors.items())):
        raise ValueError("grounded restart state or tensor inventory differs")
    partitions = {name: {key[len(name) + 1:]: value for key, value in tensors.items()
                        if key.startswith(name + "/")} for name in ("model", "optimizer", "selected", "rng")}
    if sum(map(len, partitions.values())) != len(tensors):
        raise ValueError("grounded restart has undeclared tensor ownership")
    for name, expected in expected_partitions.items():
        actual = partitions[name]
        if (set(actual) != set(expected)
                or any(actual[key].shape != value.shape or actual[key].dtype != value.dtype
                       for key, value in expected.items())):
            raise ValueError("grounded restart model or optimizer geometry differs")
    if int(partitions["optimizer"]["step"].item()) != state["step"]:
        raise ValueError("grounded restart optimizer update count differs")
    model.update(tree_unflatten(partitions["model"]))
    optimizer.state = tree_unflatten(partitions["optimizer"])
    mx.random.state = [partitions["rng"][str(index)] for index in range(len(partitions["rng"]))]
    return state, partitions["selected"], pointer


def verify_grounded_fit_checkpoint(directory):
    """Check saved source custody and selection without allocating a backbone.

    This checks artifact integrity. It makes no language-correctness claim.
    """
    directory = Path(directory)
    def read(path, bound):
        with open_stable_readonly_binary(path, max_bytes=bound) as (handle, identity):
            payload = handle.read(bound + 1)
            if len(payload) != identity.size or len(payload) > bound:
                raise ValueError("grounded fit artifact read exceeded its bound")
            return payload

    report = json.loads(read(directory / "report.json", 64 * 1024 ** 2))
    body = {key: value for key, value in report.items() if key != "receipt_sha256"}
    if (report.get("schema") != "aura.grounded_binding_fit.v1"
            or hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
               != report.get("receipt_sha256")
            or report.get("implementation") != implementation_receipt()
            or report.get("semantic_success") is not None or report.get("serving_authority") is not False
            or report.get("inference_family_labels") is not False):
        raise ValueError("grounded fit report integrity or authority differs")
    owner = json.loads(read(directory / "fit-owner.json", 1024 ** 2))
    observation_ids = {name: [item["source_id"] for item in report["observations"][name]]
                       for name in ("training", "calibration")}
    if (owner.get("schema") != "aura.grounded_binding_owner.v2"
            or owner.get("identity") != report["fit_identity"]
            or owner.get("fit_ids") != observation_ids["training"]
            or owner.get("calibration_ids") != observation_ids["calibration"]
            or any(not values or len(values) != len(set(values)) for values in observation_ids.values())
            or set(observation_ids["training"]) & set(observation_ids["calibration"])
            or any([item["source_id"] for item in report["supervision"][name]] != values
                   for name, values in observation_ids.items())
            or hashlib.sha256(json.dumps({"observations": report["observations"],
                "supervision": report["supervision"]}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
               != report["source_receipt_sha256"]):
        raise ValueError("grounded fit source custody differs")
    if report.get("operation_field_contract") is not None:
        from core.learning.semantic_native_operation_field import NativeOperationField
        contract = report["operation_field_contract"]
        NativeOperationField.from_contract(contract)
        sources = report.get("operation_supervision", [])
        if (report.get("native_contract", {}).get("operation_field_contract") != contract
                or report["native_contract"].get("operation_supervision") != sources
                or [row["source_id"] for row in sources] != sorted(
                    observation_ids["training"] + observation_ids["calibration"])
                or hashlib.sha256(json.dumps(sources, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                    != report.get("operation_supervision_sha256")
                or not report.get("operation_presence_training")
                or report.get("whole_program_calibration_proven") is not False
                or not math.isfinite(report.get("operation_weight", math.nan))
                or report["operation_weight"] <= 0):
            raise ValueError("grounded operation supervision or native custody differs")
    if report.get("program_objective_contract") is not None:
        contract = report["program_objective_contract"]
        if (report.get("operation_field_contract") is None
                or report.get("native_contract", {}).get("program_objective_contract") != contract
                or contract.get("schema") != "aura.grounded_complete_program_objective.v1"
                or contract.get("source_ids") != sorted(observation_ids["training"] + observation_ids["calibration"])
                or re.fullmatch(r"[a-f0-9]{64}", contract.get("source_pool_sha256", "")) is None
                or contract.get("whole_program_calibration_proven") is not False
                or contract.get("all_programs_covered") is not False
                or not math.isfinite(contract.get("weight", math.nan)) or contract["weight"] <= 0
                or report.get("complete_program_contrast_training") is not True):
            raise ValueError("grounded complete-program source custody differs")
    pointer = json.loads(read(directory / "resume.json", 16384))
    if (pointer.get("schema") != "aura.grounded_restart_pointer.v1"
            or pointer.get("identity") != report["fit_identity"] or pointer.get("step") != report["steps"]
            or re.fullmatch(r"[a-f0-9]{64}", pointer.get("sha256", "")) is None
            or pointer.get("file") != f"resume-{pointer['sha256']}.safetensors"):
        raise ValueError("grounded fit terminal restart identity differs")
    generation = read(directory / pointer["file"], 4 * 1024 ** 3)
    if hashlib.sha256(generation).hexdigest() != pointer["sha256"]:
        raise ValueError("grounded fit terminal generation checksum differs")
    tensors, metadata = mx.load(io.BytesIO(generation), format="safetensors", return_metadata=True)
    state = json.loads(metadata["state"])
    if (state.get("schema") != "aura.grounded_optimizer_restart.v1"
            or state.get("identity") != report["fit_identity"] or state.get("step") != report["steps"]
            or state.get("history") != report["history"]
            or state.get("best") != [report["source_calibration_loss"], report["selected_step"]]
            or tensors["optimizer/step"].item() != report["steps"]
            or set(tensors) != set(state["inventory"])
            or any([list(value.shape), str(value.dtype)] != state["inventory"][key]
                   or not mx.all(mx.isfinite(value)).item() for key, value in tensors.items())):
        raise ValueError("grounded fit terminal optimizer or selection differs")
    weights = read(directory / "selected.safetensors", 1024 ** 3)
    if hashlib.sha256(weights).hexdigest() != report["weights_sha256"]:
        raise ValueError("grounded fit selected weights checksum differs")
    selected = mx.load(io.BytesIO(weights), format="safetensors")
    saved = {key[len("selected/"):]: value for key, value in tensors.items() if key.startswith("selected/")}
    checkpoint = mx.load(io.BytesIO(read(directory / f"checkpoint-{report['selected_step']}.safetensors",
                                        1024 ** 3)), format="safetensors")
    for actual in (saved, checkpoint):
        if (set(actual) != set(selected) or any(actual[key].dtype != value.dtype
                or actual[key].shape != value.shape or not mx.array_equal(actual[key], value).item()
                for key, value in selected.items())):
            raise ValueError("grounded fit selected checkpoint differs from terminal generation")
    return report


@dataclass(frozen=True)
class GroundedBindingEvidence:
    source_id: str
    context: object
    roles: tuple
    operations: dict
    mentions: dict
    candidates: dict
    adjacency: object = None
    relations: tuple = ()

    def arrays(self):
        records = self.context.referents
        if (not isinstance(self.source_id, str) or not self.source_id or not records or not self.roles
                or set(self.operations) != {role.identity for role in self.roles}
                or set(self.mentions) != set(self.operations)
                or set(self.candidates) != {record.key for record in records}
                or len({role.identity for role in self.roles}) != len(self.roles)):
            raise ValueError("grounded binding needs complete source-qualified evidence")
        values = (mx.stack([self.operations[role.identity] for role in self.roles]),
                  mx.stack([self.mentions[role.identity] for role in self.roles]),
                  mx.stack([self.candidates[record.key] for record in records]))
        if any(not mx.all(mx.isfinite(value)).item() for value in values):
            raise ValueError("grounded binding needs finite observed states")
        allowed = mx.array([[self.context.eligible(role, record) for record in records]
                            for role in self.roles], dtype=mx.bool_)
        return values, allowed

    def receipt(self):
        values, allowed = self.arrays()
        arrays = {str(index): value for index, value in enumerate(values)}
        arrays["allowed"] = allowed
        if self.adjacency is not None:
            arrays["adjacency"] = self.adjacency
        digest = hashlib.sha256()
        for name, value in sorted(arrays.items()):
            mx.eval(value)
            digest.update(name.encode())
            digest.update(str((value.dtype, value.shape)).encode())
            digest.update(memoryview(value).tobytes())
        return {"source_id": self.source_id, "observations_sha256": digest.hexdigest(),
                "roles": [{"identity": role.identity, "role": role.role, "type": role.type_name,
                           "scope": role.scope, "tick": role.tick, "referents": role.referents,
                           "agreement": dict(role.agreement), "embedding": role.embedding,
                           "unbound_cost": role.unbound_cost} for role in self.roles],
                "candidates": [{"key": record.key, "source": record.source, "type": record.type_name,
                                "scope": record.scope, "available_from": record.available_from,
                                "available_until": record.available_until, "aliases": record.aliases,
                                "attributes": dict(record.attributes), "embedding": record.embedding}
                               for record in self.context.referents],
                "type_parents": dict(self.context.type_parents), "edges": sorted(self.context.edges),
                "relations": [{"left": relation.left, "right": relation.right,
                               "relation": relation.relation, "violation_cost": relation.violation_cost}
                              for relation in self.relations]}


def capture_grounded_binding_evidence(source_id, prefix, suffix, tokens, context, roles, *,
                                     depths, operation_spans, mention_spans, candidate_spans,
                                     adjacency=None, relations=()):
    """Capture aligned native layers, retaining occurrence and source identity.

    Spans are explicit token offsets from the caller's proposal/acquisition
    layer, not target labels or inferred aliases. Capture does not discover
    spans, establish their semantic correctness or grant model-lane ownership.
    """
    roles = tuple(roles)
    if tokens.ndim != 2 or tokens.shape[0] != 1:
        raise ValueError("grounded binding capture needs one complete token sequence")
    if (set(operation_spans) != {role.identity for role in roles}
            or set(mention_spans) != set(operation_spans)
            or set(candidate_spans) != {record.key for record in context.referents}):
        raise ValueError("grounded binding capture needs complete aligned spans")
    for spans in (operation_spans, mention_spans, candidate_spans):
        for span in spans.values():
            span.validate_bound(tokens.shape[1])
    states = suffix.layer_states(prefix.capture(tokens), tuple(depths))
    def capture(spans):
        return {key: mx.stack([mx.mean(value[0, span.start:span.end], axis=0) for value in states])
                for key, span in spans.items()}
    evidence = GroundedBindingEvidence(source_id, context, roles, capture(operation_spans),
        capture(mention_spans), capture(candidate_spans), adjacency, tuple(relations))
    evidence.arrays()
    return evidence


def project_grounded_evidence(evidence, projection):
    if projection is None:
        return evidence
    def project(states):
        basis = mx.array(projection.basis.tolist())
        values = {}
        for key, value in states.items():
            if value.shape[-1] != basis.shape[0]:
                raise ValueError("grounded projection width differs from observed native states")
            typed_basis = basis.astype(value.dtype)
            values[key] = value - (value @ typed_basis) @ typed_basis.T
        return values
    return replace(evidence, operations=project(evidence.operations), mentions=project(evidence.mentions),
                   candidates=project(evidence.candidates))


@dataclass(frozen=True)
class NativeGroundedCapture:
    """One immutable frozen prefix with public, occurrence-aligned spans.

    Suffix states are recomputed inside the gradient, so the pointer loss can
    train the actual adapter sites. Capturing the prefix and owning the model
    lane remain caller responsibilities. No target graph discovers spans.
    """
    source_id: str
    hidden: object
    context: object
    roles: tuple
    depths: tuple
    operation_spans: dict
    mention_spans: dict
    candidate_spans: dict
    adjacency: object = None
    relations: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "hidden", mx.array(self.hidden))
        for name in ("operation_spans", "mention_spans", "candidate_spans"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))
        for name in ("roles", "depths", "relations"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if self.adjacency is not None:
            object.__setattr__(self, "adjacency", mx.array(self.adjacency))

    def validate(self):
        if (not isinstance(self.source_id, str) or not self.source_id or self.hidden.ndim != 3
                or self.hidden.shape[0] != 1 or min(self.hidden.shape) < 1
                or not mx.all(mx.isfinite(self.hidden)).item()
                or set(self.operation_spans) != {role.identity for role in self.roles}
                or set(self.mention_spans) != set(self.operation_spans)
                or set(self.candidate_spans) != {record.key for record in self.context.referents}):
            raise ValueError("native grounded capture needs complete observed prefix and aligned spans")
        for spans in (self.operation_spans, self.mention_spans, self.candidate_spans):
            for span in spans.values():
                span.validate_bound(self.hidden.shape[1])

    def capture(self, suffix, *, states=None):
        self.validate()
        if states is None:
            states = suffix.layer_states(mx.stop_gradient(self.hidden), self.depths)
        if (len(states) != len(self.depths)
                or any(value.shape != self.hidden.shape for value in states)):
            raise ValueError("native binding capture changed the aligned suffix states")
        def observe(spans):
            return {key: mx.stack([mx.mean(value[0, span.start:span.end], axis=0) for value in states])
                    for key, span in spans.items()}
        return GroundedBindingEvidence(self.source_id, self.context, self.roles,
            observe(self.operation_spans), observe(self.mention_spans), observe(self.candidate_spans),
            self.adjacency, self.relations)

    def receipt(self):
        self.validate()
        mx.eval(self.hidden)
        return {"source_id": self.source_id, "hidden_shape": self.hidden.shape,
                "hidden_dtype": str(self.hidden.dtype),
                "prefix_sha256": hashlib.sha256(memoryview(self.hidden).tobytes()).hexdigest(),
                "depths": self.depths,
                "operation_spans": sorted((key, span.start, span.end) for key, span in self.operation_spans.items()),
                "mention_spans": sorted((key, span.start, span.end) for key, span in self.mention_spans.items()),
                "candidate_spans": sorted((key, span.start, span.end) for key, span in self.candidate_spans.items())}


@dataclass(frozen=True)
class GroundedBindingSupervision:
    evidence: GroundedBindingEvidence
    positives: tuple[tuple[tuple[str, str], ...], ...]
    # Whole graph alternatives are source-witnessed, not independently chosen
    # slot labels. Multiple acceptable complete graphs remain positive.
    graphs: tuple[tuple[tuple[str, str], ...], ...] = ()
    positive_graphs: tuple[int, ...] = ()
    environment: str = "source"
    retention_scores: object = None

    def indices(self):
        values, allowed = self.evidence.arrays()
        keys = {record.key: index for index, record in enumerate(self.evidence.context.referents)}
        if len(self.positives) != len(self.evidence.roles):
            raise ValueError("binding source supervision roles differ")
        indices = []
        for row, positives in enumerate(self.positives):
            if (not positives or len(set(positives)) != len(positives)
                    or any(key not in keys or not allowed[row, keys[key]].item() for key in positives)):
                raise ValueError("binding source supervision names inadmissible identities")
            indices.append(tuple(keys[key] for key in positives))
        if bool(self.graphs) != bool(self.positive_graphs):
            raise ValueError("binding graph alternatives need witnessed positives")
        graph_indices = []
        for graph in self.graphs:
            if len(graph) != len(indices) or any(key not in keys for key in graph):
                raise ValueError("binding graph identities differ from observed candidates")
            if any(not allowed[row, keys[key]].item() for row, key in enumerate(graph)):
                raise ValueError("binding graph includes an inadmissible source")
            assignment = dict(zip((role.identity for role in self.evidence.roles), graph, strict=True))
            for relation in self.evidence.relations:
                if (relation.left not in assignment or relation.right not in assignment
                        or relation.cost(self.evidence.context, assignment[relation.left],
                                         assignment[relation.right]) is None):
                    raise ValueError("binding graph violates its declared relational constraints")
            graph_indices.append(tuple(keys[key] for key in graph))
        if (len(set(self.positive_graphs)) != len(self.positive_graphs)
                or any(type(index) is not int or not 0 <= index < len(self.graphs)
                       for index in self.positive_graphs)
                or not isinstance(self.environment, str) or not self.environment):
            raise ValueError("binding graph supervision or environment differs")
        for index in self.positive_graphs:
            if any(key not in self.positives[row] for row, key in enumerate(self.graphs[index])):
                raise ValueError("positive complete graph conflicts with source-positive roles")
        if self.retention_scores is not None:
            if (self.retention_scores.shape != allowed.shape
                    or not mx.all(mx.isfinite(self.retention_scores)).item()
                    or any(not mx.any(allowed[row]).item() for row in range(len(indices)))):
                raise ValueError("retention needs complete finite source baseline evidence")
            baseline = mx.where(allowed, self.retention_scores, -mx.inf)
            if any(int(mx.argmax(baseline[row]).item()) not in positive
                   for row, positive in enumerate(indices)):
                raise ValueError("only source-witnessed correct baselines may constrain retention")
        return values, allowed, tuple(indices), tuple(graph_indices)


def grounded_source_loss(pointer, example, *, retention_weight=0., stationarity_weight=0., role_margin=0.):
    from core.learning.semantic_native_program import native_choice_loss

    values, allowed, positives, graphs = example.indices()
    scores = pointer(*values, adjacency=example.evidence.adjacency, allowed=allowed,
                     role_features=semantic_role_features(example.evidence.roles))
    distances = mx.array([[candidate_cost(example.evidence.context, role, record, observed_cost=0.)
        if example.evidence.context.eligible(role, record) else 0.
        for record in example.evidence.context.referents] for role in example.evidence.roles])
    scores = scores - distances
    loss = pointer_choice_loss(scores, positives)
    if role_margin:
        loss = loss + pointer_role_margin_loss(scores, positives, margin=role_margin)
    if stationarity_weight:
        from core.learning.semantic_binding_invariance import binding_choice_scale_penalty
        loss = loss + stationarity_weight * mx.mean(mx.stack([
            binding_choice_scale_penalty(scores[row], positive) for row, positive in enumerate(positives)]))
    if retention_weight and example.retention_scores is not None:
        baseline = mx.where(allowed, example.retention_scores, -mx.inf)
        probability = mx.stop_gradient(mx.softmax(baseline, axis=-1))
        log_probability = scores - mx.logsumexp(scores, axis=-1, keepdims=True)
        # Inadmissible entries carry zero probability, not 0 * -inf.
        retained = -mx.sum(probability * mx.where(allowed, log_probability, 0.), axis=-1)
        loss = loss + retention_weight * mx.mean(retained)
    if graphs:
        graph_scores = []
        for graph, keys in zip(graphs, example.graphs, strict=True):
            assignment = dict(zip((role.identity for role in example.evidence.roles), keys, strict=True))
            penalty = sum(relation.cost(example.evidence.context, assignment[relation.left],
                          assignment[relation.right]) for relation in example.evidence.relations)
            graph_scores.append(mx.sum(mx.stack([scores[row, column]
                for row, column in enumerate(graph)])) - penalty)
        graph_scores = mx.stack(graph_scores)
        loss = loss + (native_choice_loss(graph_scores, example.positive_graphs)
                       if len(graphs) > 1 else graph_scores[0] * 0.)
    return loss


@dataclass(frozen=True)
class GroundedEquivariancePair:
    left: str
    right: str
    # Explicit semantic correspondences, not nearest-neighbor guesses.
    role_pairs: tuple[tuple[str, str], ...]
    candidate_pairs: tuple[tuple[tuple[str, str], tuple[str, str]], ...]

    def orders(self, examples):
        if self.left == self.right or self.left not in examples or self.right not in examples:
            raise ValueError("binding equivariance needs two distinct fit-only sources")
        left, right = examples[self.left].evidence, examples[self.right].evidence
        left_roles = [role.identity for role in left.roles]
        right_roles = [role.identity for role in right.roles]
        left_keys = [record.key for record in left.context.referents]
        right_keys = [record.key for record in right.context.referents]
        if (len(self.role_pairs) != len(left_roles) or len(left_roles) != len(right_roles)
                or len(self.candidate_pairs) != len(left_keys) or len(left_keys) != len(right_keys)
                or {pair[0] for pair in self.role_pairs} != set(left_roles)
                or {pair[1] for pair in self.role_pairs} != set(right_roles)
                or {pair[0] for pair in self.candidate_pairs} != set(left_keys)
                or {pair[1] for pair in self.candidate_pairs} != set(right_keys)):
            raise ValueError("binding equivariance requires witnessed role and identity bijections")
        role_map, key_map = dict(self.role_pairs), dict(self.candidate_pairs)
        return (tuple(right_roles.index(role_map[role]) for role in left_roles),
                tuple(right_keys.index(key_map[key]) for key in left_keys))


def grounded_equivariance_loss(pointer, pair, examples):
    role_order, candidate_order = pair.orders(examples)
    left, right = examples[pair.left].evidence, examples[pair.right].evidence
    left_values, left_allowed = left.arrays()
    right_values, right_allowed = right.arrays()
    role_order, candidate_order = mx.array(role_order), mx.array(candidate_order)
    if not mx.array_equal(left_allowed, right_allowed[role_order][:, candidate_order]).item():
        raise ValueError("meaning-preserving binding pairs must preserve candidate admission")
    if not mx.all(mx.any(left_allowed, axis=-1)).item():
        raise ValueError("binding equivariance has an unsupported role")
    left_scores = pointer(*left_values, adjacency=left.adjacency, allowed=left_allowed,
                          role_features=semantic_role_features(left.roles))
    right_scores = pointer(*right_values, adjacency=right.adjacency, allowed=right_allowed,
                           role_features=semantic_role_features(right.roles))[role_order][:, candidate_order]
    left_log = left_scores - mx.logsumexp(left_scores, axis=-1, keepdims=True)
    right_log = right_scores - mx.logsumexp(right_scores, axis=-1, keepdims=True)
    difference = mx.where(left_allowed, left_log, 0.) - mx.where(left_allowed, right_log, 0.)
    return .5 * mx.mean(mx.sum((mx.softmax(left_scores) - mx.softmax(right_scores)) * difference, axis=-1))


class GroundedBindingEngine:
    def __init__(self, pointer, *, evidence_weight=1., nuisance_projection=None, native_suffix=None,
                 operation_field=None):
        if not math.isfinite(evidence_weight) or evidence_weight < 0:
            raise ValueError("grounded binding weight must be finite and nonnegative")
        self.pointer, self.evidence_weight = pointer, evidence_weight
        self.nuisance_projection = nuisance_projection
        self.native_suffix = native_suffix
        self.operation_field = operation_field

    def costs(self, evidence, *, baseline_costs=None):
        evidence = project_grounded_evidence(evidence, self.nuisance_projection)
        evidence.arrays()
        learned = pointer_context_costs(self.pointer, evidence.context, evidence.roles,
            evidence.operations, evidence.mentions, evidence.candidates, adjacency=evidence.adjacency)
        if baseline_costs is None:
            return {key: self.evidence_weight * value for key, value in learned.items()}
        if set(baseline_costs) != set(learned) or any(not math.isfinite(value)
                                                    for value in baseline_costs.values()):
            raise ValueError("grounded baseline evidence must cover exactly admitted candidates")
        return {key: baseline_costs[key] + self.evidence_weight * value for key, value in learned.items()}

    def resolve(self, evidence, *, baseline_costs=None, **options):
        return bind_context_roles(evidence.context, evidence.roles,
            costs=self.costs(evidence, baseline_costs=baseline_costs),
            relations=evidence.relations, **options)

    def binding_receipt(self, evidence, result, *, baseline_costs=None):
        """Expose actual slot/source evidence without inventing component scores.

        Hard eligibility, learned energy, observed-vector distance and baseline
        energy remain separate. Margins report competition, not correctness.
        """
        costs = self.costs(evidence, baseline_costs=baseline_costs)
        selected = dict(result.bindings)
        rows = []
        for role in evidence.roles:
            alternatives = []
            for record in evidence.context.referents:
                address = role.identity, record.key
                admitted = evidence.context.eligible(role, record)
                observed = costs.get(address)
                distance = candidate_cost(evidence.context, role, record, observed_cost=0.) if admitted else None
                alternatives.append({"candidate_id": record.key, "source": record.source,
                    "candidate_type": record.type_name, "admitted": admitted,
                    "learned_plus_baseline_energy": observed, "context_distance": distance,
                    "selected": selected.get(role.identity) == record.key})
            rows.append({"role_instance": role.identity, "role_id": role.role, "required_type": role.type_name,
                "scope": role.scope, "tick": role.tick, "selected_source": selected.get(role.identity),
                "null_cost": role.unbound_cost, "alternatives": alternatives})
        return {"schema": "aura.grounded_role_receipt.v1", "source_id": evidence.source_id,
            "observation_receipt": evidence.receipt(), "status": result.status,
            "margin": result.margin, "margin_is_probability": False, "roles": rows}

    def resolve_native(self, capture, *, baseline_costs=None, **options):
        if self.native_suffix is None:
            raise ValueError("grounded engine has no authorized attached native suffix")
        return self.resolve(capture.capture(self.native_suffix), baseline_costs=baseline_costs, **options)

    def revise_diagrams(self, proposals, **options):
        from core.learning.semantic_diagram_workspace import revise_diagram_workspace

        return revise_diagram_workspace(self, proposals, **options)

    def discourse_costs(self, candidates, factors, role_positions, **options):
        from core.learning.semantic_discourse_smoothing import infer_discourse_references

        posterior = infer_discourse_references(candidates, factors, **options)
        return posterior.binding_costs(role_positions), posterior

    def resolve_discourse(self, evidence, candidates, factors, role_positions, **options):
        from core.learning.semantic_discourse_smoothing import resolve_discourse_binding

        return resolve_discourse_binding(self, evidence, candidates, factors, role_positions, **options)

    def resolve_chart(self, chart, evidence, register_keys, *, baseline_costs=None, **options):
        if sum(len(node) for node in chart.options) != len(evidence.roles):
            raise ValueError("grounded evidence roles differ from chart slots")
        start, roles = 0, []
        for node in chart.options:
            roles.append(evidence.roles[start:start + len(node)])
            start += len(node)
        return chart.solve_grounded(evidence.context, tuple(roles), register_keys,
            costs=self.costs(evidence, baseline_costs=baseline_costs),
            relations=evidence.relations, **options)

    @classmethod
    def load(cls, directory, *, native_suffix=None, native_contract=None):
        directory = Path(directory)
        report = json.loads((directory / "report.json").read_text())
        if report.get("schema") != "aura.grounded_binding_fit.v1":
            raise ValueError("unknown grounded binding fit")
        expected_receipt = report.pop("receipt_sha256", None)
        if hashlib.sha256(json.dumps(report, sort_keys=True, separators=(",", ":")).encode()).hexdigest() != expected_receipt:
            raise ValueError("grounded binding fit receipt differs")
        if report.get("implementation") != implementation_receipt():
            raise ValueError("grounded binding implementation changed since fitting")
        contract = report["pointer_contract"]
        pointer = RelationalBindingPointer(contract["hidden_width"], depths=contract["depths"],
            relation_width=contract["relation_width"], rounds=contract["rounds"],
            role_queries=contract.get("schema") == "aura.relational_binding_pointer.v3")
        if pointer.to_contract() != contract:
            raise ValueError("grounded binding pointer contract differs")
        weights = directory / "selected.safetensors"
        if hashlib.sha256(weights.read_bytes()).hexdigest() != report["weights_sha256"]:
            raise ValueError("grounded binding weights differ from selected source checkpoint")
        if report.get("native_contract") is not None:
            if native_suffix is None or native_contract != report["native_contract"]:
                raise ValueError("joint grounded binding needs the exact caller-owned native suffix contract")
            artifact = nn.Module()
            artifact.pointer, artifact.native_suffix = pointer, native_suffix
            operation_field = None
            if report.get("operation_field_contract") is not None:
                from core.learning.semantic_native_operation_field import NativeOperationField
                operation_field = NativeOperationField.from_contract(report["operation_field_contract"])
                artifact.operation_field = operation_field
            # Only trainable adapters and the pointer are in this artifact.
            loaded = mx.load(str(weights))
            expected = dict(tree_flatten(artifact.trainable_parameters()))
            if set(loaded) != set(expected) or any(loaded[key].shape != expected[key].shape for key in loaded):
                raise ValueError("joint grounded binding state escaped the declared trainable sites")
            if any(not mx.all(mx.isfinite(value)).item() for value in loaded.values()):
                raise ValueError("joint grounded binding checkpoint is nonfinite")
            artifact.load_weights(str(weights), strict=False)
        else:
            operation_field = None
            if report.get("operation_field_contract") is not None:
                raise ValueError("native operation field has no suffix custody")
            if native_suffix is not None or native_contract is not None:
                raise ValueError("cached grounded fit has no native adapter custody")
            pointer.load_weights(str(weights), strict=True)
        if any(not mx.all(mx.isfinite(value)).item() for _, value in tree_flatten(pointer.parameters())):
            raise ValueError("grounded binding weights are nonfinite")
        from core.learning.semantic_binding_invariance import BindingNuisanceProjection
        projection = (BindingNuisanceProjection.from_dict(report["nuisance_projection"])
                      if report.get("nuisance_projection") is not None else None)
        return cls(pointer, nuisance_projection=projection, native_suffix=native_suffix,
                   operation_field=operation_field)


def validate_grounded_fit_inputs(training, calibration, *, steps=128,
                         learning_rate=.001, save_every=16, max_seconds=1800., group_eta=0.,
                         retention_weight=0., stationarity_weight=0., nuisance_projection=None,
                         domain_reversal=0., equivariance_pairs=(), equivariance_weight=1.,
                         role_margin=0., training_schedule=None, sampling_receipt=None):
    """Reject impossible source contracts without acquiring native model weights."""
    training, calibration = tuple(training), tuple(calibration)
    train_ids = [item.evidence.source_id for item in training]
    calibration_ids = [item.evidence.source_id for item in calibration]
    if (not training or not calibration or set(train_ids) & set(calibration_ids)
            or len(set(train_ids)) != len(train_ids) or len(set(calibration_ids)) != len(calibration_ids)
            or any(type(value) is not int or value < 1 for value in (steps, save_every))
            or steps % save_every or not math.isfinite(max_seconds) or not 0 < max_seconds <= 14400
            or not math.isfinite(learning_rate) or learning_rate <= 0
            or not math.isfinite(group_eta) or not 0 <= group_eta <= 1
            or not math.isfinite(retention_weight) or retention_weight < 0
            or not math.isfinite(stationarity_weight) or stationarity_weight < 0
            or not math.isfinite(domain_reversal) or not 0 <= domain_reversal <= 1
            or not math.isfinite(equivariance_weight) or equivariance_weight < 0):
        raise ValueError("grounded binding fit needs disjoint sources and bounded complete checkpoints")
    if not math.isfinite(role_margin) or role_margin < 0:
        raise ValueError("grounded role margin must be finite and nonnegative")
    schedule = tuple(training_schedule) if training_schedule is not None else tuple(
        train_ids[step % len(train_ids)] for step in range(steps))
    if len(schedule) != steps or not set(schedule) <= set(train_ids):
        raise ValueError("grounded training schedule must contain exactly fit-only updates")
    schedule_sha = hashlib.sha256(json.dumps(schedule, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest()
    if sampling_receipt is not None:
        body = dict(sampling_receipt)
        expected = body.pop("receipt_sha256", None)
        if (training_schedule is None or body.get("schedule_sha256") != schedule_sha
                or body.get("primary_updates") != steps or body.get("eligible_sources") != len(train_ids)
                or body.get("primary_sources") != len(set(schedule))
                or body.get("held_labels_used") is not False
                or hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"),
                    ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest() != expected):
            raise ValueError("grounded sampling receipt differs from its actual updates")
    if nuisance_projection is not None and (not set(nuisance_projection.training_ids) <= set(train_ids)
            or set(nuisance_projection.training_ids) & set(calibration_ids)):
        raise ValueError("binding nuisance projection consumed a non-fit source")
    examples = {item.evidence.source_id: item for item in training}
    for item in (*training, *calibration):
        if not isinstance(item.environment, str) or not item.environment:
            raise ValueError("grounded source environment must be explicit")
        item.indices()
    for pair in equivariance_pairs:
        pair.orders(examples)
    if domain_reversal and len({item.environment for item in training}) < 2:
        raise ValueError("domain-adversarial binding needs multiple source environments")
    return schedule, schedule_sha


def fit_grounded_binding(pointer, training, calibration, directory, *, steps=128,
                         learning_rate=.001, save_every=16, max_seconds=1800., group_eta=0.,
                         retention_weight=0., stationarity_weight=0., nuisance_projection=None,
                         domain_reversal=0., equivariance_pairs=(), equivariance_weight=1.,
                         native_suffix=None, native_captures=None, native_contract=None, role_margin=0.,
                         training_schedule=None, sampling_receipt=None, resume=False,
                         operation_field=None, operation_supervision=None, operation_weight=1.,
                         program_supervision=None, program_weight=1.):
    """Fit observed identities; select only on disjoint source calibration.

    Environment labels affect training weights, never pointer features. Group
    weighting does not guarantee unseen-environment invariance.
    """
    import mlx.optimizers as optim

    started = time.monotonic()
    training, calibration = tuple(training), tuple(calibration)
    train_ids = [item.evidence.source_id for item in training]
    calibration_ids = [item.evidence.source_id for item in calibration]
    schedule, schedule_sha = validate_grounded_fit_inputs(training, calibration, steps=steps,
        learning_rate=learning_rate, save_every=save_every, max_seconds=max_seconds, group_eta=group_eta,
        retention_weight=retention_weight, stationarity_weight=stationarity_weight,
        nuisance_projection=nuisance_projection, domain_reversal=domain_reversal,
        equivariance_pairs=equivariance_pairs, equivariance_weight=equivariance_weight,
        role_margin=role_margin, training_schedule=training_schedule, sampling_receipt=sampling_receipt)
    deadline = started + max_seconds
    def check_bound():
        if time.monotonic() >= deadline:
            raise TimeoutError("grounded binding fit reached its declared resource bound")
    if nuisance_projection is not None:
        if (not set(nuisance_projection.training_ids) <= set(train_ids)
                or set(nuisance_projection.training_ids) & set(calibration_ids)):
            raise ValueError("binding nuisance projection consumed a non-fit source")
        training = tuple(replace(item, evidence=project_grounded_evidence(item.evidence, nuisance_projection))
                         for item in training)
        calibration = tuple(replace(item, evidence=project_grounded_evidence(item.evidence, nuisance_projection))
                            for item in calibration)
    examples = {item.evidence.source_id: item for item in training}
    all_examples = {item.evidence.source_id: item for item in (*training, *calibration)}
    operation_sources = None
    if operation_field is not None:
        if (native_suffix is None or not isinstance(operation_supervision, Mapping)
                or set(operation_supervision) != set(all_examples)
                or not math.isfinite(operation_weight) or operation_weight <= 0
                or (operation_field.hidden_width, operation_field.depths)
                    != (pointer.hidden_width, pointer.depths)):
            raise ValueError("operation field needs exact joint-native source custody and positive weight")
        for identity, supervision in operation_supervision.items():
            if (supervision.source_id != identity
                    or supervision.token_count != native_captures[identity].hidden.shape[1]):
                raise ValueError("operation source supervision differs from native observations")
            supervision.indices(operation_field)
        operation_sources = [operation_supervision[key].receipt() for key in sorted(all_examples)]
    elif operation_supervision is not None:
        raise ValueError("operation source supervision requires an attached field")
    program_contract = None
    if program_supervision is not None:
        from core.learning.semantic_grounded_program_objective import program_objective_contract
        if (operation_field is None or not isinstance(program_supervision, Mapping)
                or set(program_supervision) != set(all_examples)):
            raise ValueError("complete-program objective needs exact joint-native source custody")
        for identity, program in program_supervision.items():
            if program.source_id != identity:
                raise ValueError("complete-program pool changed its source identity")
            program.validate(operation_field, operation_supervision[identity].token_count)
        program_contract = program_objective_contract(program_supervision, program_weight)
        if native_contract.get("program_objective_contract") != program_contract:
            raise ValueError("complete-program objective differs from native custody")
    for item in all_examples.values():
        check_bound()
        if not isinstance(item.environment, str) or not item.environment:
            raise ValueError("grounded source environment must be explicit")
        item.indices()
    initial_native_observations = {}
    if native_suffix is not None:
        if (not isinstance(native_contract, dict) or not native_contract
                or not isinstance(native_captures, Mapping) or set(native_captures) != set(all_examples)
                or not tree_flatten(native_suffix.trainable_parameters())
                or any("lora_" not in name for name, _value in tree_flatten(native_suffix.trainable_parameters()))):
            raise ValueError("joint grounded fit needs exact native captures, contract and adapter-only suffix")
        for identity, capture in native_captures.items():
            check_bound()
            capture.validate()
            current = capture.capture(native_suffix)
            old = all_examples[identity].evidence
            if (current.source_id != identity or current.context != old.context
                    or current.roles != old.roles or current.relations != old.relations):
                raise ValueError("native grounded capture changed public source identities or roles")
            replace(all_examples[identity], evidence=current).indices()
            initial_native_observations[identity] = project_grounded_evidence(current, nuisance_projection).receipt()
            print(json.dumps({"stage": "grounded_initial_source_observed", "source": identity,
                              "completed": len(initial_native_observations), "population": len(all_examples)}), flush=True)
            del current
    elif native_captures is not None or native_contract is not None:
        raise ValueError("native capture custody requires an attached suffix")
    pairs_by_source = {identity: [] for identity in train_ids}
    for pair in equivariance_pairs:
        pair.orders(examples)
        pairs_by_source[pair.left].append(pair)
        pairs_by_source[pair.right].append(pair)
    directory = Path(directory)
    environments = sorted({item.environment for item in training})
    if domain_reversal and len(environments) < 2:
        raise ValueError("domain-adversarial binding needs multiple source environments")
    if type(resume) is not bool:
        raise ValueError("grounded resume must be explicitly boolean")
    if directory.exists() and not resume:
        raise FileExistsError(directory)
    if resume and not directory.is_dir():
        raise ValueError("grounded resume needs an existing complete generation")
    if resume and (directory / "report.json").exists():
        raise ValueError("grounded fit is already complete; do not repeat it")
    optimizer = optim.Adam(learning_rate=learning_rate)
    model = nn.Module()
    model.pointer = pointer
    artifact = nn.Module()
    artifact.pointer = pointer
    if native_suffix is not None:
        model.native_suffix = artifact.native_suffix = native_suffix
    else:
        artifact = pointer
    if operation_field is not None:
        model.operation_field = artifact.operation_field = operation_field
    if domain_reversal:
        model.nuisance = nn.Linear(pointer.feature_blocks * pointer.relation_width, len(environments))
    optimizer.init(model.trainable_parameters())
    counts = {key: sum(item.environment == key for item in training) for key in environments}
    log_weights = {key: 0. for key in environments}
    def current_example(item, owner=None):
        if native_suffix is None:
            return item
        suffix = native_suffix if owner is None else owner.native_suffix
        evidence = native_captures[item.evidence.source_id].capture(suffix)
        return replace(item, evidence=project_grounded_evidence(evidence, nuisance_projection))

    def source_loss(item, owner=None):
        actual_pointer = pointer if owner is None else owner.pointer
        if operation_field is None:
            return grounded_source_loss(actual_pointer, current_example(item, owner),
                retention_weight=retention_weight, stationarity_weight=stationarity_weight,
                role_margin=role_margin)
        suffix = native_suffix if owner is None else owner.native_suffix
        field = operation_field if owner is None else owner.operation_field
        capture = native_captures[item.evidence.source_id]
        states = suffix.layer_states(mx.stop_gradient(capture.hidden), capture.depths)
        current = replace(item, evidence=project_grounded_evidence(
            capture.capture(suffix, states=states), nuisance_projection))
        depth_states = mx.stack(states, axis=2)[0]
        operation_loss = field.source_loss(depth_states,
            operation_supervision[item.evidence.source_id])
        program_loss = (program_supervision[item.evidence.source_id].source_loss(actual_pointer, field,
            depth_states, nuisance_projection=nuisance_projection) if program_supervision is not None else mx.array(0.))
        return (grounded_source_loss(actual_pointer, current, retention_weight=retention_weight,
            stationarity_weight=stationarity_weight, role_margin=role_margin)
            + operation_weight * operation_loss + program_weight * program_loss)

    def save_weights(path):
        stream = io.BytesIO()
        mx.save_safetensors(stream, dict(tree_flatten(artifact.trainable_parameters())))
        _write_binding_artifact(path, stream.getvalue())

    def measure():
        losses = []
        for example in calibration:
            check_bound()
            value = source_loss(example).item()
            if not math.isfinite(value):
                raise ValueError("grounded binding calibration is nonfinite")
            losses.append(value)
        return sum(losses) / len(losses)

    observations = {split: [initial_native_observations[item.evidence.source_id] if native_suffix is not None
                            else item.evidence.receipt() for item in examples]
                    for split, examples in (("training", training), ("calibration", calibration))}
    supervision = {split: [{"source_id": item.evidence.source_id, "positives": item.positives,
                            "graphs": item.graphs, "positive_graphs": item.positive_graphs,
                            "environment": item.environment,
                            "retention_scores": (item.retention_scores.tolist()
                                if item.retention_scores is not None else None)} for item in examples]
                   for split, examples in (("training", training), ("calibration", calibration))}
    source_receipt = hashlib.sha256(json.dumps({"observations": observations, "supervision": supervision},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    identity_body = {"source_receipt": source_receipt, "pointer": pointer.to_contract(),
        "implementation": implementation_receipt(), "native_contract": native_contract,
        "steps": steps, "save_every": save_every, "schedule": schedule,
        "learning_rate": learning_rate, "group_eta": group_eta, "domain_reversal": domain_reversal,
        "retention_weight": retention_weight, "stationarity_weight": stationarity_weight,
        "role_margin": role_margin, "equivariance_weight": equivariance_weight,
        "equivariance_pairs": [{"left": pair.left, "right": pair.right,
            "role_pairs": pair.role_pairs, "candidate_pairs": pair.candidate_pairs} for pair in equivariance_pairs],
        "nuisance_projection": nuisance_projection.to_dict() if nuisance_projection is not None else None}
    if operation_field is not None:
        identity_body.update(operation_field_contract=operation_field.to_contract(),
            operation_supervision=operation_sources, operation_weight=operation_weight)
    if program_contract is not None:
        identity_body["program_objective_contract"] = program_contract
    fit_identity = hashlib.sha256(json.dumps(identity_body, sort_keys=True, separators=(",", ":"),
                                            allow_nan=False).encode()).hexdigest()
    owner = {"schema": "aura.grounded_binding_owner.v2", "fit_ids": train_ids,
             "calibration_ids": calibration_ids, "identity": fit_identity}
    if resume:
        if json.loads((directory / "fit-owner.json").read_bytes()) != owner:
            raise ValueError("grounded resume owner or source evidence differs")
        state, selected, resumed_pointer = _load_grounded_restart(directory, fit_identity, model, optimizer, artifact)
        start_step, history = state["step"], state["history"]
        best, log_weights = tuple(state["best"]), state["log_weights"]
        if (not 0 <= start_step <= steps or start_step % save_every
                or set(log_weights) != set(environments)
                or any(not math.isfinite(v) for v in log_weights.values())
                or type(best[1]) is not int or not 0 <= best[1] <= start_step
                or not math.isfinite(best[0])
                or [row["step"] for row in history] != list(range(save_every, start_step + 1, save_every))
                or any(not math.isfinite(row["calibration_loss"]) for row in history)):
            raise ValueError("grounded restart progress differs from declared updates")
        stream = io.BytesIO()
        mx.save_safetensors(stream, selected)
        _write_binding_artifact(directory / "selected.safetensors", stream.getvalue())
    else:
        with local_internal_governed_scope("grounded_binding_source_fit", domain="file_write"):
            if not get_file_write_gateway().write_bytes_if_absent(directory / "fit-owner.json",
                json.dumps(owner).encode(), source="grounded_binding_source_fit", mode=0o400):
                raise FileExistsError(directory)
        best = (measure(), 0)
        save_weights(directory / "checkpoint-0.safetensors")
        save_weights(directory / "selected.safetensors")
        start_step, history, resumed_pointer = 0, [], None

    def restart_generation(step):
        selected = mx.load(str(directory / "selected.safetensors"))
        _save_grounded_restart(directory, {"identity": fit_identity, "step": step, "history": history,
            "best": best, "log_weights": log_weights}, model, optimizer, selected)
        print(json.dumps({"stage": "grounded_restart_saved", "step": step,
                          "fit_identity": fit_identity}), flush=True)

    if not resume:
        restart_generation(0)
    for step in range(start_step + 1, steps + 1):
        check_bound()
        example = examples[schedule[step - 1]]
        mass = mx.softmax(mx.array([log_weights[key] for key in environments]))
        group_mass = float(mass[environments.index(example.environment)].item())
        # Equal group mass compensates unequal example counts, not identity.
        weight = group_mass * len(training) / counts[example.environment]
        source_risk = source_loss(example).item() if group_eta else None
        def objective(owner, example=example, weight=weight):
            loss = weight * source_loss(example, owner)
            pairs = pairs_by_source[example.evidence.source_id]
            if pairs and equivariance_weight:
                loss = loss + equivariance_weight * mx.mean(mx.stack([
                    grounded_equivariance_loss(owner.pointer, pair,
                        {identity: current_example(examples[identity], owner) for identity in (pair.left, pair.right)})
                    for pair in pairs]))
            if domain_reversal:
                from core.learning.semantic_binding_invariance import reverse_nuisance_gradient
                current = current_example(example, owner)
                values, allowed = current.evidence.arrays()
                feature = owner.pointer.relation_features(*values, adjacency=current.evidence.adjacency,
                    role_features=semantic_role_features(current.evidence.roles))
                feature = mx.sum(mx.where(allowed[..., None], feature, 0.), axis=(0, 1)) / mx.sum(allowed)
                logits = owner.nuisance(reverse_nuisance_gradient(feature, domain_reversal))
                loss = loss + mx.mean(nn.losses.cross_entropy(logits[None],
                    mx.array([environments.index(example.environment)], dtype=mx.int32)))
            return loss
        loss, gradients = nn.value_and_grad(model, objective)(model)
        norm = mx.sqrt(sum(mx.sum(value ** 2) for _, value in tree_flatten(gradients)))
        if not mx.isfinite(loss).item() or not mx.isfinite(norm).item():
            raise ValueError("grounded binding fit produced nonfinite computation")
        gradients = tree_map(lambda value, norm=norm: value / mx.maximum(norm, 1.), gradients)
        optimizer.update(model, gradients)
        mx.eval(model.parameters(), optimizer.state)
        if group_eta:
            log_weights[example.environment] += group_eta * source_risk
            maximum = max(log_weights.values())
            log_weights = {key: value - maximum for key, value in log_weights.items()}
        if step % save_every == 0:
            calibration_loss = measure()
            history.append({"step": step, "calibration_loss": calibration_loss})
            save_weights(directory / f"checkpoint-{step}.safetensors")
            if (calibration_loss, step) < best:
                best = calibration_loss, step
                save_weights(directory / "selected.safetensors")
            restart_generation(step)
    artifact.load_weights(str(directory / "selected.safetensors"), strict=native_suffix is None)
    weights = (directory / "selected.safetensors").read_bytes()
    nuisance_hash = None
    if domain_reversal:
        stream = io.BytesIO()
        mx.save_safetensors(stream, dict(tree_flatten(model.nuisance.parameters())))
        _write_binding_artifact(directory / "training-nuisance.safetensors", stream.getvalue())
        nuisance_hash = hashlib.sha256((directory / "training-nuisance.safetensors").read_bytes()).hexdigest()
    report = {"schema": "aura.grounded_binding_fit.v1", "pointer_contract": pointer.to_contract(),
        "implementation": implementation_receipt(),
        "fit_identity": fit_identity, "resume_from_step": start_step,
        "resumed_generation": resumed_pointer,
        "selected_step": best[1], "source_calibration_loss": best[0], "steps": steps,
        "training_schedule": schedule, "training_schedule_sha256": schedule_sha,
        "sampling_receipt": sampling_receipt,
        "primary_sources_visited": len(set(schedule)),
        "complete_primary_epoch": set(schedule) == set(train_ids),
        "learning_rate": learning_rate, "group_eta": group_eta,
        "retention_weight": retention_weight, "history": history,
        "stationarity_weight": stationarity_weight,
        "role_margin": role_margin,
        "domain_reversal": domain_reversal, "training_nuisance_sha256": nuisance_hash,
        "equivariance_weight": equivariance_weight,
        "equivariance_pairs": [{"left": pair.left, "right": pair.right,
                                "role_pairs": pair.role_pairs, "candidate_pairs": pair.candidate_pairs}
                               for pair in equivariance_pairs],
        "nuisance_projection": nuisance_projection.to_dict() if nuisance_projection is not None else None,
        "source_receipt_sha256": source_receipt, "observations": observations,
        "supervision": supervision, "weights_sha256": hashlib.sha256(weights).hexdigest(),
        "semantic_success": None, "serving_authority": False,
        "inference_family_labels": False, "selection": "minimum_disjoint_source_calibration_loss"}
    if native_suffix is not None:
        report["native_contract"] = native_contract
        report["native_captures"] = [native_captures[key].receipt() for key in sorted(native_captures)]
        report["joint_native_adapter_training"] = True
    if operation_field is not None:
        report.update(operation_field_contract=operation_field.to_contract(),
            operation_supervision=operation_sources, operation_weight=operation_weight,
            operation_supervision_sha256=hashlib.sha256(json.dumps(operation_sources,
                sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            operation_presence_training=True, whole_program_calibration_proven=False)
    if program_contract is not None:
        report.update(program_objective_contract=program_contract, complete_program_contrast_training=True)
    report["receipt_sha256"] = hashlib.sha256(json.dumps(report,
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    _write_binding_artifact(directory / "report.json", (json.dumps(report, indent=2) + "\n").encode())
    return GroundedBindingEngine(pointer, nuisance_projection=nuisance_projection, native_suffix=native_suffix,
                                operation_field=operation_field), report
