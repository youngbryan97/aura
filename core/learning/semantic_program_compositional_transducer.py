"""Compose local neural meanings into a typed semantic program graph.

The v6 shared transducer learned several geometries at once, but decoded them
through a whole-request step-count classifier and heads named by absolute step.
That made an unlabelled family name recoverable from the hidden state and made
the geometry follow the family.  A leave-family-out probe exposed the result:
input grounding transferred perfectly while every structural decision failed.

This transducer has no geometry classifier and no step-indexed learned head.
It learns two reusable kinds of local evidence:

* which spans name operations and which primitive each span means;
* which spans are arguments of an operation and which earlier definition they
  refer to.

An exact operation chart and bounded typed graph search compose those atoms
into a connected, acyclic SSA program.  Step count is therefore the number of
operation nodes supported by the request, not a template selected from the
whole-request embedding.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Final

import numpy as np

from core.learning.semantic_input_grounding import (
    SemanticInputGroundingContract,
    semantic_input_grounding_contract_from_dict,
)
from core.learning.semantic_program_floor import semantic_primitive_type_signature
from core.learning.semantic_program_ir import (
    SemanticIRInstruction,
    SemanticProgramIR,
    SemanticValue,
    TokenSpan,
    normalize_semantic_value,
)
from core.learning.semantic_program_shared_transducer import (
    _channel_span,
    _geometry,
    _geometry_name,
    _normalized_weights,
)
from core.learning.semantic_program_transducer import (
    LinearClassifierHead,
    LinearPointerHead,
    MultiViewClassifierHead,
    SemanticTransducerTrainingExample,
    SemanticTransductionOutcome,
    _fit_classifier,
    _hidden_array,
    _joint_pointer_assignment,
    _operation_feature,
)

from .semantic_program_transducer_fitting import (
    _ARGUMENT_BEAM,  # noqa: F401
    _ARGUMENT_CANDIDATES,  # noqa: F401
    _ARGUMENT_MENTIONS_PER_DEFINITION,  # noqa: F401
    _ARGUMENT_PROPOSAL_SCALES,  # noqa: F401
    _DIRECTIONAL_RELATION_PARTS,  # noqa: F401
    _LEGACY_DEFINITION_CANDIDATE_STRATEGY,
    _LOCAL_ARGUMENT_CANDIDATES,  # noqa: F401
    _LOCAL_DEFINITION_CANDIDATE_STRATEGY,
    _LOCAL_DEFINITION_CANDIDATES,  # noqa: F401
    _OPERATION_CANDIDATES,  # noqa: F401
    _OPERATION_CHART_BEAM,
    _OPERATION_MODE,
    _OPERATION_PENALTY_POINTS,  # noqa: F401
    _POINTER_HARD_NEGATIVES,
    _RELATION_TISSUE_BATCH_SIZE,
    _RELATION_TISSUE_EPOCHS,
    _RELATION_TISSUE_GRADIENT_CLIP,
    _RELATION_TISSUE_LEARNING_RATE,
    _RELATION_TISSUE_RANK,  # noqa: F401
    _RELATION_TISSUE_SEED,
    _RELATION_TISSUE_SELECTION_INTERVAL,
    _RELATION_TISSUE_WEIGHT_DECAY,
    _STABLE_REGISTER_TABLE_STRATEGY,
    COMPOSITIONAL_SEMANTIC_TRANSDUCER_SCHEMA,
    DirectionalRelationHead,
    LinearArgumentRoleHead,
    RegisterUseContract,
    _all_semantic_spans,
    _argument_proposal_rows,  # noqa: F401
    _argument_proposals_by_operation,  # noqa: F401
    _assign_typed_arguments,
    _best_nonoverlapping_node_charts,  # noqa: F401
    _best_nonoverlapping_nodes,
    _best_penalized_operation_chart,  # noqa: F401
    _definition_span_candidates,  # noqa: F401
    _directional_relation_feature,  # noqa: F401
    _fit_argument_proposal_heads,
    _fit_argument_role_heads,
    _fit_directional_relation_head,
    _fit_low_rank_relation_tissue,
    _fit_register_use_contract,
    _fit_shared_pointer,
    _input_type,  # noqa: F401
    _log_sigmoid,  # noqa: F401
    _mention_invariant_relation_evidence,  # noqa: F401
    _operation_chart_candidates,
    _operation_nodes,
    _operation_order,  # noqa: F401
    _OperationNode,
    _overlap,
    _register_definition_candidates,  # noqa: F401
    _register_definition_spans,
    _relation_decision_batch,  # noqa: F401
    _relation_tissue_logits,  # noqa: F401
    _relation_tissue_metrics,  # noqa: F401
    _RelationDecisionBatch,  # noqa: F401
    _select_argument_proposal_scale,
    _select_definition_pointer_scale,
    _select_operation_length_penalty,
    _shared_pointer_training_indices,  # noqa: F401
    _TypedArgumentAssignment,  # noqa: F401
)

_V13_COMPOSITIONAL_SEMANTIC_TRANSDUCER_SCHEMA: Final = "aura.semantic_program_transducer.v13"
_V13_COMPOSITIONAL_SEMANTIC_RECEIPT_SCHEMA: Final = (
    "aura.semantic_program_transducer_receipt.v13"
)
_V14_COMPOSITIONAL_SEMANTIC_TRANSDUCER_SCHEMA: Final = "aura.semantic_program_transducer.v14"
_V14_COMPOSITIONAL_SEMANTIC_RECEIPT_SCHEMA: Final = (
    "aura.semantic_program_transducer_receipt.v14"
)
_V15_COMPOSITIONAL_SEMANTIC_TRANSDUCER_SCHEMA: Final = "aura.semantic_program_transducer.v15"
_V15_COMPOSITIONAL_SEMANTIC_RECEIPT_SCHEMA: Final = (
    "aura.semantic_program_transducer_receipt.v15"
)
COMPOSITIONAL_SEMANTIC_RECEIPT_SCHEMA: Final = "aura.semantic_program_transducer_receipt.v16"
_RELATION_CHANNEL: Final = "middle_causal_hidden"
_MAX_STEPS: Final = 16
_MAX_INPUTS: Final = 8
_LOCAL_ARGUMENT_CANDIDATE_STRATEGY: Final = "per_operation_clause_quota_v1"


@dataclass(frozen=True, slots=True)
class SemanticGeometryEnvelope:
    """Separate measured training support from bounded structural inference."""

    observed: tuple[tuple[int, int], ...]
    observed_max_inputs: int
    observed_max_steps: int
    hard_max_inputs: int = _MAX_INPUTS
    hard_max_steps: int = _MAX_STEPS

    def __post_init__(self) -> None:
        if (
            not self.observed
            or any(
                type(inputs) is not int
                or type(steps) is not int
                or not 1 <= inputs <= self.hard_max_inputs
                or not 1 <= steps <= self.hard_max_steps
                for inputs, steps in self.observed
            )
            or self.observed_max_inputs != max(inputs for inputs, _steps in self.observed)
            or self.observed_max_steps != max(steps for _inputs, steps in self.observed)
        ):
            raise ValueError("semantic geometry support is invalid")

    def inference_step_limit(self, input_count: int) -> int | None:
        """Extend shared local heads at the rate measured across training geometries."""

        if type(input_count) is not int or not 1 <= input_count <= self.hard_max_inputs:
            return None
        if input_count <= self.observed_max_inputs:
            return self.observed_max_steps
        by_inputs: dict[int, int] = {}
        for inputs, steps in self.observed:
            by_inputs[inputs] = max(steps, by_inputs.get(inputs, 0))
        ordered = sorted(by_inputs.items())
        growth = max(
            (
                max(0, math.ceil((right_steps - left_steps) / (right_inputs - left_inputs)))
                for (left_inputs, left_steps), (right_inputs, right_steps) in zip(
                    ordered,
                    ordered[1:],
                    strict=False,
                )
                if right_inputs > left_inputs
            ),
            default=0,
        )
        return min(
            self.hard_max_steps,
            self.observed_max_steps
            + growth * (input_count - self.observed_max_inputs),
        )


def _semantic_geometry_envelope(
    receipt: Mapping[str, Any],
) -> SemanticGeometryEnvelope | None:
    rows = receipt.get("observed_geometry_support")
    if not isinstance(rows, list) or not rows:
        return None
    observed: list[tuple[int, int]] = []
    try:
        for row in rows:
            if not isinstance(row, Mapping):
                return None
            fields = {
                name: int(value)
                for name, value in (
                    part.split(":", 1)
                    for part in str(row.get("geometry", "")).split("|")
                )
            }
            if set(fields) != {"inputs", "steps"}:
                return None
            observed.append((fields["inputs"], fields["steps"]))
        return SemanticGeometryEnvelope(
            observed=tuple(sorted(set(observed))),
            observed_max_inputs=max(inputs for inputs, _steps in observed),
            observed_max_steps=max(steps for _inputs, steps in observed),
        )
    except (TypeError, ValueError):
        return None


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
















def _has_symbolic_register_definitions(item: SemanticTransducerTrainingExample) -> bool:
    """Return whether every register has an identity span separate from its payload."""

    if not item.register_definition_spans:
        return False
    definitions = _register_definition_spans(item)
    return all(
        not _overlap(definition, payload)
        for definition, payload in zip(
            definitions[: item.ir.n_inputs],
            item.ir.input_spans,
            strict=True,
        )
    ) and all(
        not _overlap(definition, instruction.operation_span)
        for definition, instruction in zip(
            definitions[item.ir.n_inputs :],
            item.ir.instructions,
            strict=True,
        )
    )




























def _operation_chart(
    nodes: Sequence[_OperationNode],
    *,
    max_steps: int,
    length_penalty: float,
) -> tuple[_OperationNode, ...]:
    candidates = [
        (score - length_penalty * count, selected)
        for count in range(1, max_steps + 1)
        for score, selected in (_best_nonoverlapping_nodes(nodes, count),)
        if math.isfinite(score)
    ]
    if not candidates:
        return ()
    return max(
        candidates,
        key=lambda item: (
            item[0],
            -len(item[1]),
            tuple((-node.span.start, -node.span.end) for node in item[1]),
        ),
    )[1]


































@dataclass(frozen=True, slots=True)
class CompositionalSemanticProgramTransducer:
    """One local-atom chart decoder with no family or geometry router."""

    hidden_size: int
    model_basis_sha256: str
    hidden_channels: tuple[str, ...]
    hidden_channel_widths: tuple[int, ...]
    input_grounding: SemanticInputGroundingContract
    operation_pointer: LinearPointerHead
    argument_pointer: LinearPointerHead
    definition_pointer: LinearPointerHead
    operation_head: MultiViewClassifierHead
    argument_role_heads: tuple[LinearArgumentRoleHead, ...]
    argument_proposal_heads: tuple[LinearArgumentRoleHead, ...]
    definition_relation_head: DirectionalRelationHead
    definition_candidate_strategy: str
    max_steps: int
    max_inputs: int
    max_span_tokens: int
    max_definition_span_tokens: int
    max_argument_span_tokens_by_type: dict[str, int]
    register_use_contract: RegisterUseContract
    operation_chart_beam: int
    operation_length_penalty: float
    argument_role_scale: float
    argument_proposal_scale: float
    definition_relation_scale: float
    argument_pointer_scale: float
    allow_computed_dependencies: bool
    training_receipt: dict[str, Any]
    schema: str = COMPOSITIONAL_SEMANTIC_TRANSDUCER_SCHEMA

    def __post_init__(self) -> None:
        receipt = json.loads(_canonical_bytes(self.training_receipt))
        geometry_envelope = _semantic_geometry_envelope(receipt)
        relation_fit = receipt.get("relation_tissue_fit")
        relation_selection = (
            relation_fit.get("validation_selection") if isinstance(relation_fit, Mapping) else None
        )
        relation_start, relation_end = _channel_span(
            _RELATION_CHANNEL,
            hidden_channels=self.hidden_channels,
            hidden_channel_widths=self.hidden_channel_widths,
        )
        coefficient = self._coefficient_body()
        argument_bounds = dict(self.max_argument_span_tokens_by_type)
        body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
        schema_contracts = {
            _V13_COMPOSITIONAL_SEMANTIC_TRANSDUCER_SCHEMA: (
                _V13_COMPOSITIONAL_SEMANTIC_RECEIPT_SCHEMA,
                _LEGACY_DEFINITION_CANDIDATE_STRATEGY,
            ),
            _V14_COMPOSITIONAL_SEMANTIC_TRANSDUCER_SCHEMA: (
                _V14_COMPOSITIONAL_SEMANTIC_RECEIPT_SCHEMA,
                _LOCAL_DEFINITION_CANDIDATE_STRATEGY,
            ),
            _V15_COMPOSITIONAL_SEMANTIC_TRANSDUCER_SCHEMA: (
                _V15_COMPOSITIONAL_SEMANTIC_RECEIPT_SCHEMA,
                _STABLE_REGISTER_TABLE_STRATEGY,
            ),
            COMPOSITIONAL_SEMANTIC_TRANSDUCER_SCHEMA: (
                COMPOSITIONAL_SEMANTIC_RECEIPT_SCHEMA,
                _STABLE_REGISTER_TABLE_STRATEGY,
            ),
        }
        expected_receipt_schema, expected_candidate_strategy = schema_contracts.get(
            self.schema,
            (None, None),
        )
        if (
            self.schema not in schema_contracts
            or type(self.hidden_size) is not int
            or self.hidden_size < 1
            or not _is_sha256(self.model_basis_sha256)
            or not self.hidden_channels
            or len(self.hidden_channels) != len(self.hidden_channel_widths)
            or sum(self.hidden_channel_widths) != self.hidden_size
            or self.operation_pointer.width != self.hidden_size
            or self.argument_pointer.width != self.hidden_size
            or self.definition_pointer.width != self.hidden_size
            or self.operation_head.modes != (_OPERATION_MODE,)
            or not self.argument_role_heads
            or len(self.argument_proposal_heads) != len(self.argument_role_heads)
            or any(
                head.channel_width != relation_end - relation_start
                for head in (*self.argument_role_heads, *self.argument_proposal_heads)
            )
            or self.definition_relation_head.channel_width != relation_end - relation_start
            or type(self.max_steps) is not int
            or not 1 <= self.max_steps <= _MAX_STEPS
            or type(self.max_inputs) is not int
            or not 1 <= self.max_inputs <= _MAX_INPUTS
            or geometry_envelope is None
            or geometry_envelope.observed_max_inputs != self.max_inputs
            or geometry_envelope.observed_max_steps != self.max_steps
            or type(self.max_span_tokens) is not int
            or self.max_span_tokens < 1
            or type(self.max_definition_span_tokens) is not int
            or self.max_definition_span_tokens < self.max_span_tokens
            or set(argument_bounds) != {"integer", "integer_sequence"}
            or any(type(value) is not int or value < 1 for value in argument_bounds.values())
            or any(
                not math.isfinite(value) or value <= 0
                for value in (
                    self.argument_role_scale,
                    self.definition_relation_scale,
                    self.argument_pointer_scale,
                )
            )
            or not math.isfinite(self.argument_proposal_scale)
            or self.argument_proposal_scale < 0
            or not math.isfinite(self.operation_length_penalty)
            or type(self.allow_computed_dependencies) is not bool
            or receipt.get("schema") != expected_receipt_schema
            or self.definition_candidate_strategy != expected_candidate_strategy
            or (
                self.schema != _V13_COMPOSITIONAL_SEMANTIC_TRANSDUCER_SCHEMA
                and receipt.get("definition_candidate_strategy")
                != self.definition_candidate_strategy
            )
            or receipt.get("receipt_sha256") != _sha(body)
            or receipt.get("model_basis_sha256") != self.model_basis_sha256
            or receipt.get("input_grounding_sha256") != self.input_grounding.contract_sha256
            or receipt.get("coefficient_sha256") != _sha(coefficient)
            or receipt.get("global_geometry_classifier_present") is not False
            or receipt.get("step_indexed_heads_present") is not False
            or receipt.get("argument_span_bounds") != argument_bounds
            or receipt.get("definition_span_bound") != self.max_definition_span_tokens
            or type(self.operation_chart_beam) is not int
            or not 1 <= self.operation_chart_beam <= _OPERATION_CHART_BEAM
            or receipt.get("operation_chart_beam") != self.operation_chart_beam
            or receipt.get("register_use_contract") != self.register_use_contract.to_dict()
            or receipt.get("argument_search_strategy", "legacy_global_v1")
            not in {"legacy_global_v1", "prefix_feasible_v1"}
            or (
                self.schema
                in {
                    _V15_COMPOSITIONAL_SEMANTIC_TRANSDUCER_SCHEMA,
                    COMPOSITIONAL_SEMANTIC_TRANSDUCER_SCHEMA,
                }
                and (
                    not isinstance(receipt.get("identity_definition_supervision"), Mapping)
                    or receipt["identity_definition_supervision"].get("selection")
                    != "payload_and_operation_disjoint_register_spans_v1"
                    or type(
                        receipt["identity_definition_supervision"].get("training_examples")
                    )
                    is not int
                    or receipt["identity_definition_supervision"]["training_examples"] < 1
                    or type(
                        receipt["identity_definition_supervision"].get("validation_examples")
                    )
                    is not int
                    or receipt["identity_definition_supervision"]["validation_examples"] < 1
                    or type(
                        receipt["identity_definition_supervision"].get(
                            "fallback_to_all_examples"
                        )
                    )
                    is not bool
                )
            )
            or (
                self.schema == COMPOSITIONAL_SEMANTIC_TRANSDUCER_SCHEMA
                and receipt.get("argument_candidate_strategy")
                != _LOCAL_ARGUMENT_CANDIDATE_STRATEGY
            )
            or not isinstance(relation_fit, Mapping)
            or relation_fit.get("algorithm") != "minibatch_adamw_cross_entropy_v1"
            or relation_fit.get("selection_objective") != "minimum_validation_cross_entropy"
            or relation_fit.get("rank") != self.definition_relation_head.query_projection.shape[1]
            or relation_fit.get("seed") != _RELATION_TISSUE_SEED
            or receipt.get("relation_score_contract") != "mention_invariant_conditional_tissue_v1"
            or receipt.get("argument_role_contract") != "semantic_and_pointer_proposal_product_v1"
            or not isinstance(receipt.get("argument_proposal_fit"), Mapping)
            or receipt["argument_proposal_fit"].get("hard_negative_limit")
            != _POINTER_HARD_NEGATIVES
            or receipt["argument_proposal_fit"].get("scale_selection_objective")
            != "minimum_validation_cross_entropy"
            or not isinstance(receipt["argument_proposal_fit"].get("scale_selection"), list)
            or sum(
                isinstance(row, Mapping) and row.get("selected") is True
                for row in receipt["argument_proposal_fit"]["scale_selection"]
            )
            != 1
            or not any(
                isinstance(row, Mapping)
                and row.get("selected") is True
                and row.get("proposal_scale") == self.argument_proposal_scale
                for row in receipt["argument_proposal_fit"]["scale_selection"]
            )
            or type(receipt["argument_proposal_fit"].get("positive_rows")) is not int
            or receipt["argument_proposal_fit"]["positive_rows"] < 1
            or type(receipt["argument_proposal_fit"].get("pointer_hard_negative_rows")) is not int
            or receipt["argument_proposal_fit"]["pointer_hard_negative_rows"] < 1
            or not isinstance(relation_selection, list)
            or sum(
                isinstance(row, Mapping) and row.get("selected") is True
                for row in relation_selection
            )
            != 1
            or any(
                receipt.get(field) is not False
                for field in (
                    "expected_answers_available",
                    "verifier_traces_available",
                    "generated_compiler_text_available",
                    "correctness_authority",
                )
            )
        ):
            raise ValueError("compositional semantic transducer envelope is invalid")
        object.__setattr__(self, "training_receipt", receipt)
        object.__setattr__(self, "max_argument_span_tokens_by_type", argument_bounds)

    @property
    def geometry_envelope(self) -> SemanticGeometryEnvelope:
        envelope = _semantic_geometry_envelope(self.training_receipt)
        if envelope is None:  # __post_init__ proves this unreachable after construction.
            raise RuntimeError("semantic geometry support disappeared")
        return envelope

    def inference_step_limit(self, input_count: int) -> int | None:
        return self.geometry_envelope.inference_step_limit(input_count)

    def _coefficient_body(self) -> dict[str, Any]:
        return {
            "operation_pointer": self.operation_pointer.to_dict(),
            "argument_pointer": self.argument_pointer.to_dict(),
            "definition_pointer": self.definition_pointer.to_dict(),
            "operation_head": self.operation_head.to_dict(),
            "argument_role_heads": [head.to_dict() for head in self.argument_role_heads],
            "argument_proposal_heads": [head.to_dict() for head in self.argument_proposal_heads],
            "definition_relation_head": self.definition_relation_head.to_dict(),
            "operation_length_penalty": self.operation_length_penalty,
            "argument_role_scale": self.argument_role_scale,
            "argument_proposal_scale": self.argument_proposal_scale,
            "definition_relation_scale": self.definition_relation_scale,
            "argument_pointer_scale": self.argument_pointer_scale,
            "allow_computed_dependencies": self.allow_computed_dependencies,
            "register_use_contract": self.register_use_contract.to_dict(),
        }

    @property
    def receipt_sha256(self) -> str:
        return str(self.training_receipt["receipt_sha256"])

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "hidden_size": self.hidden_size,
            "model_basis_sha256": self.model_basis_sha256,
            "hidden_channels": list(self.hidden_channels),
            "hidden_channel_widths": list(self.hidden_channel_widths),
            "input_grounding": self.input_grounding.to_dict(),
            "max_steps": self.max_steps,
            "max_inputs": self.max_inputs,
            "max_span_tokens": self.max_span_tokens,
            "max_definition_span_tokens": self.max_definition_span_tokens,
            "max_argument_span_tokens_by_type": self.max_argument_span_tokens_by_type,
            "operation_chart_beam": self.operation_chart_beam,
            **self._coefficient_body(),
            "training_receipt": self.training_receipt,
        }
        if self.schema != _V13_COMPOSITIONAL_SEMANTIC_TRANSDUCER_SCHEMA:
            payload["definition_candidate_strategy"] = self.definition_candidate_strategy
        return payload

    def _with_coefficients(self, **changes: Any) -> CompositionalSemanticProgramTransducer:
        values = {
            "operation_pointer": changes.get("operation_pointer", self.operation_pointer),
            "argument_pointer": changes.get("argument_pointer", self.argument_pointer),
            "definition_pointer": changes.get("definition_pointer", self.definition_pointer),
            "operation_head": changes.get("operation_head", self.operation_head),
            "argument_role_heads": changes.get("argument_role_heads", self.argument_role_heads),
            "argument_proposal_heads": changes.get(
                "argument_proposal_heads", self.argument_proposal_heads
            ),
            "definition_relation_head": changes.get(
                "definition_relation_head", self.definition_relation_head
            ),
            "operation_length_penalty": changes.get(
                "operation_length_penalty", self.operation_length_penalty
            ),
            "argument_role_scale": changes.get("argument_role_scale", self.argument_role_scale),
            "argument_proposal_scale": changes.get(
                "argument_proposal_scale", self.argument_proposal_scale
            ),
            "definition_relation_scale": changes.get(
                "definition_relation_scale", self.definition_relation_scale
            ),
            "argument_pointer_scale": changes.get(
                "argument_pointer_scale", self.argument_pointer_scale
            ),
            "allow_computed_dependencies": changes.get(
                "allow_computed_dependencies", self.allow_computed_dependencies
            ),
            "register_use_contract": changes.get(
                "register_use_contract", self.register_use_contract
            ),
        }
        coefficient = {
            "operation_pointer": values["operation_pointer"].to_dict(),
            "argument_pointer": values["argument_pointer"].to_dict(),
            "definition_pointer": values["definition_pointer"].to_dict(),
            "operation_head": values["operation_head"].to_dict(),
            "argument_role_heads": [head.to_dict() for head in values["argument_role_heads"]],
            "argument_proposal_heads": [
                head.to_dict() for head in values["argument_proposal_heads"]
            ],
            "definition_relation_head": values["definition_relation_head"].to_dict(),
            "operation_length_penalty": values["operation_length_penalty"],
            "argument_role_scale": values["argument_role_scale"],
            "argument_proposal_scale": values["argument_proposal_scale"],
            "definition_relation_scale": values["definition_relation_scale"],
            "argument_pointer_scale": values["argument_pointer_scale"],
            "allow_computed_dependencies": values["allow_computed_dependencies"],
            "register_use_contract": values["register_use_contract"].to_dict(),
        }
        body = {
            key: value for key, value in self.training_receipt.items() if key != "receipt_sha256"
        }
        body["coefficient_sha256"] = _sha(coefficient)
        body["register_use_contract"] = values["register_use_contract"].to_dict()
        return replace(
            self,
            **values,
            training_receipt={**body, "receipt_sha256": _sha(body)},
        )

    def coefficient_lesion(self) -> CompositionalSemanticProgramTransducer:
        zero_pointer = lambda head: LinearPointerHead(  # noqa: E731
            np.zeros_like(head.start_weight),
            head.start_bias,
            np.zeros_like(head.end_weight),
            head.end_bias,
        )
        operation_component = self.operation_head.heads[0]
        return self._with_coefficients(
            operation_pointer=zero_pointer(self.operation_pointer),
            argument_pointer=zero_pointer(self.argument_pointer),
            definition_pointer=zero_pointer(self.definition_pointer),
            operation_head=MultiViewClassifierHead(
                self.operation_head.modes,
                (
                    LinearClassifierHead(
                        operation_component.labels,
                        np.zeros_like(operation_component.weight),
                        operation_component.bias,
                    ),
                ),
            ),
            argument_role_heads=tuple(
                LinearArgumentRoleHead(np.zeros_like(head.weight), head.bias)
                for head in self.argument_role_heads
            ),
            argument_proposal_heads=tuple(
                LinearArgumentRoleHead(np.zeros_like(head.weight), head.bias)
                for head in self.argument_proposal_heads
            ),
            definition_relation_head=DirectionalRelationHead(
                np.zeros_like(self.definition_relation_head.weight),
                self.definition_relation_head.bias,
                self.definition_relation_head.pointer_scale,
                np.zeros_like(self.definition_relation_head.query_projection),
                np.zeros_like(self.definition_relation_head.definition_projection),
            ),
            register_use_contract=RegisterUseContract(
                0,
                self.max_steps * len(self.argument_role_heads),
                0,
                self.max_steps * len(self.argument_role_heads),
                False,
            ),
        )

    def relation_lesion(self) -> CompositionalSemanticProgramTransducer:
        zero_definition_pointer = LinearPointerHead(
            np.zeros_like(self.definition_pointer.start_weight),
            self.definition_pointer.start_bias,
            np.zeros_like(self.definition_pointer.end_weight),
            self.definition_pointer.end_bias,
        )
        return self._with_coefficients(
            definition_pointer=zero_definition_pointer,
            argument_role_heads=tuple(
                LinearArgumentRoleHead(np.zeros_like(head.weight), head.bias)
                for head in self.argument_role_heads
            ),
            argument_proposal_heads=tuple(
                LinearArgumentRoleHead(np.zeros_like(head.weight), head.bias)
                for head in self.argument_proposal_heads
            ),
            definition_relation_head=DirectionalRelationHead(
                np.zeros_like(self.definition_relation_head.weight),
                self.definition_relation_head.bias,
                self.definition_relation_head.pointer_scale,
                np.zeros_like(self.definition_relation_head.query_projection),
                np.zeros_like(self.definition_relation_head.definition_projection),
            ),
        )

    def relation_tissue_lesion(self) -> CompositionalSemanticProgramTransducer:
        """Remove only the cross-feature relation tissue learned by v13."""

        head = self.definition_relation_head
        return self._with_coefficients(
            definition_relation_head=DirectionalRelationHead(
                head.weight,
                head.bias,
                head.pointer_scale,
                np.zeros_like(head.query_projection),
                np.zeros_like(head.definition_projection),
            )
        )

    def argument_proposal_lesion(self) -> CompositionalSemanticProgramTransducer:
        """Remove only evidence learned from runtime pointer proposals."""

        return self._with_coefficients(
            argument_proposal_heads=tuple(
                LinearArgumentRoleHead(np.zeros_like(head.weight), 0.0)
                for head in self.argument_proposal_heads
            )
        )

    def dependency_lesion(self) -> CompositionalSemanticProgramTransducer:
        return self._with_coefficients(allow_computed_dependencies=False)

    def chart_beam_lesion(self) -> CompositionalSemanticProgramTransducer:
        """Keep every learned coefficient but consult only the first chart."""

        body = {
            key: value for key, value in self.training_receipt.items() if key != "receipt_sha256"
        }
        body["operation_chart_beam"] = 1
        return replace(
            self,
            operation_chart_beam=1,
            training_receipt={**body, "receipt_sha256": _sha(body)},
        )

    def with_prefix_feasible_arguments(self) -> CompositionalSemanticProgramTransducer:
        """Create a separately identified search candidate without refitting tissue."""

        body = {
            key: value for key, value in self.training_receipt.items() if key != "receipt_sha256"
        }
        body["argument_search_strategy"] = "prefix_feasible_v1"
        return replace(self, training_receipt={**body, "receipt_sha256": _sha(body)})

    def register_use_lesion(self) -> CompositionalSemanticProgramTransducer:
        """Remove only the source-learned graph-use bounds."""

        return self._with_coefficients(
            register_use_contract=RegisterUseContract(
                0,
                self.max_steps * len(self.argument_role_heads),
                0,
                self.max_steps * len(self.argument_role_heads),
                False,
            )
        )

    def decode(
        self,
        *,
        source_token_ids: Sequence[int],
        hidden_states: Any,
        public_inputs: Sequence[SemanticValue],
        source_text_sha256: str,
        model_basis_sha256: str,
    ) -> SemanticTransductionOutcome:
        if model_basis_sha256 != self.model_basis_sha256:
            return SemanticTransductionOutcome(None, "model_basis_mismatch", {}, {})
        if not _is_sha256(source_text_sha256):
            return SemanticTransductionOutcome(None, "source_identity_invalid", {}, {})
        try:
            inputs = tuple(normalize_semantic_value(value) for value in public_inputs)
            hidden = _hidden_array(hidden_states, expected_width=self.hidden_size)
        except ValueError as exc:
            return SemanticTransductionOutcome(None, str(exc), {}, {})
        tokens = tuple(source_token_ids)
        if hidden.shape[0] != len(tokens):
            return SemanticTransductionOutcome(None, "token_hidden_length_mismatch", {}, {})
        inference_max_steps = self.inference_step_limit(len(inputs))
        if inference_max_steps is None:
            return SemanticTransductionOutcome(None, "public_input_count_unsupported", {}, {})
        input_banks: list[tuple[tuple[TokenSpan, float], ...]] = []
        argument_pointer_scores = self.argument_pointer.score_sequence(hidden)
        for index, value in enumerate(inputs):
            spans = self.input_grounding.candidate_spans(tokens, value)
            if not spans:
                return SemanticTransductionOutcome(
                    None,
                    f"input_value_not_grounded:{index}",
                    {},
                    {},
                )
            input_banks.append(
                tuple((span, argument_pointer_scores.score_span(span)) for span in spans)
            )
        grounded = _joint_pointer_assignment(tuple(input_banks), ordered=False)
        if grounded is None:
            return SemanticTransductionOutcome(None, "input_pointer_assignment_failed", {}, {})
        input_spans, input_scores = grounded
        nodes = _operation_nodes(
            pointer=self.operation_pointer,
            classifier=self.operation_head,
            hidden=hidden,
            input_spans=input_spans,
            max_span_tokens=self.max_span_tokens,
            hidden_channels=self.hidden_channels,
            hidden_channel_widths=self.hidden_channel_widths,
        )
        charts = _operation_chart_candidates(
            nodes,
            max_steps=inference_max_steps,
            length_penalty=self.operation_length_penalty,
            limit=self.operation_chart_beam,
        )
        if not charts:
            return SemanticTransductionOutcome(None, "operation_chart_empty", {}, {})
        assigned = next(
            (
                candidate
                for selected in charts
                for candidate in (
                    _assign_typed_arguments(
                        model=self,
                        hidden=hidden,
                        inputs=inputs,
                        input_spans=input_spans,
                        operation_nodes=selected,
                        argument_pointer_scores=argument_pointer_scores,
                    ),
                )
                if candidate is not None
            ),
            None,
        )
        if assigned is None:
            return SemanticTransductionOutcome(None, "typed_argument_chart_empty", {}, {})
        selected = assigned.operation_nodes
        arguments = assigned.arguments
        argument_spans = assigned.argument_spans
        instructions = tuple(
            SemanticIRInstruction(
                op=node.operation,
                args=arguments[step],
                operation_span=node.span,
                argument_spans=argument_spans[step],
                depends_on=tuple(
                    sorted(
                        argument - len(inputs)
                        for argument in set(arguments[step])
                        if argument >= len(inputs)
                    )
                ),
            )
            for step, node in enumerate(selected)
        )
        try:
            ir = SemanticProgramIR(
                source_token_ids=tokens,
                source_text_sha256=source_text_sha256,
                input_spans=input_spans,
                instructions=instructions,
                report_value=len(inputs) + len(instructions) - 1,
                model_basis_receipt_sha256=model_basis_sha256,
                transducer_receipt_sha256=self.receipt_sha256,
            )
        except ValueError as exc:
            return SemanticTransductionOutcome(None, f"ir_rejected:{exc}", {}, {})
        pointer_scores = {
            **{f"input:{index}": score for index, score in enumerate(input_scores)},
            **{f"operation:{index}": node.pointer_score for index, node in enumerate(selected)},
            "argument_graph_total": assigned.score,
            "argument_graph_mean": assigned.score
            / sum(len(values) for values in assigned.arguments),
        }
        confidences = {f"operation:{index}": node.confidence for index, node in enumerate(selected)}
        confidences["argument_graph_runner_up_available"] = float(
            assigned.runner_up_score is not None
        )
        confidences["argument_graph_margin"] = (
            assigned.score - assigned.runner_up_score
            if assigned.runner_up_score is not None
            else 0.0
        )
        return SemanticTransductionOutcome(ir, "", pointer_scores, confidences)




def fit_compositional_semantic_program_transducer(
    examples: Sequence[SemanticTransducerTrainingExample],
    *,
    input_grounding: SemanticInputGroundingContract,
) -> CompositionalSemanticProgramTransducer:
    """Fit local atom heads and calibrate only the source-side operation chart."""

    training = tuple(item for item in examples if item.split == "train")
    validation = tuple(item for item in examples if item.split == "validation")
    if not training or not validation:
        raise ValueError("compositional semantic fit needs train and validation examples")
    bases = {item.ir.model_basis_receipt_sha256 for item in training}
    tokenizers = {item.tokenizer_identity_sha256 for item in examples}
    channel_geometries = {(item.hidden_channels, item.hidden_channel_widths) for item in training}
    geometries = Counter(_geometry(item) for item in training)
    if (
        len(bases) != 1
        or len(channel_geometries) != 1
        or tokenizers != {input_grounding.tokenizer_identity_sha256}
        or len(geometries) < 2
    ):
        raise ValueError("compositional semantic neural bases or geometries differ")
    hidden_channels, hidden_channel_widths = next(iter(channel_geometries))
    hidden_size = sum(hidden_channel_widths)
    max_steps = max(len(item.ir.instructions) for item in training)
    max_inputs = max(item.ir.n_inputs for item in training)
    max_arity = max(
        len(instruction.args) for item in training for instruction in item.ir.instructions
    )
    max_span_tokens = max(
        span.end - span.start for item in training for span in _all_semantic_spans(item)
    )
    max_definition_span_tokens = max(
        span.end - span.start for item in training for span in _register_definition_spans(item)
    )
    max_argument_span_tokens_by_type = {
        "integer": 1,
        "integer_sequence": 1,
    }
    for item in training:
        for instruction in item.ir.instructions:
            signature = semantic_primitive_type_signature(instruction.op)
            if signature is None:
                raise ValueError(f"compositional primitive has no floor type: {instruction.op}")
            argument_types, _result_type = signature
            if len(argument_types) != len(instruction.argument_spans):
                raise ValueError("compositional primitive arity differs from its floor type")
            for argument_type, span in zip(
                argument_types,
                instruction.argument_spans,
                strict=True,
            ):
                max_argument_span_tokens_by_type[argument_type] = max(
                    max_argument_span_tokens_by_type[argument_type],
                    span.end - span.start,
                )
    operation_pointer = _fit_shared_pointer(
        training,
        spans=lambda item: tuple(
            instruction.operation_span for instruction in item.ir.instructions
        ),
    )
    argument_pointer = _fit_shared_pointer(
        training,
        spans=lambda item: tuple(
            span for instruction in item.ir.instructions for span in instruction.argument_spans
        ),
    )
    identity_training = tuple(item for item in training if _has_symbolic_register_definitions(item))
    identity_validation = tuple(
        item for item in validation if _has_symbolic_register_definitions(item)
    )
    identity_supervision_fallback = not identity_training or not identity_validation
    if identity_supervision_fallback:
        identity_training = training
        identity_validation = validation
    definition_pointer = _fit_shared_pointer(
        identity_training,
        spans=_register_definition_spans,
    )
    register_use_contract = _fit_register_use_contract(training)
    operation_rows = tuple(
        (item, instruction) for item in training for instruction in item.ir.instructions
    )
    operation_head = MultiViewClassifierHead(
        (_OPERATION_MODE,),
        (
            _fit_classifier(
                np.stack(
                    [
                        _operation_feature(
                            item.hidden_states,
                            instruction.operation_span,
                            mode=_OPERATION_MODE,
                            hidden_channels=hidden_channels,
                            hidden_channel_widths=hidden_channel_widths,
                        )
                        for item, instruction in operation_rows
                    ]
                ),
                [instruction.op for item, instruction in operation_rows],
                sample_weight=_normalized_weights(
                    [1.0 / geometries[_geometry(item)] for item, _instruction in operation_rows]
                ),
            ),
        ),
    )
    argument_role_heads = _fit_argument_role_heads(
        training,
        max_arity=max_arity,
        hidden_channels=hidden_channels,
        hidden_channel_widths=hidden_channel_widths,
    )
    argument_proposal_heads, argument_proposal_fit = _fit_argument_proposal_heads(
        training,
        argument_pointer=argument_pointer,
        max_arity=max_arity,
        max_span_tokens=max_span_tokens,
        max_argument_span_tokens_by_type=max_argument_span_tokens_by_type,
        hidden_channels=hidden_channels,
        hidden_channel_widths=hidden_channel_widths,
    )
    argument_proposal_scale, argument_proposal_scale_rows = _select_argument_proposal_scale(
        validation,
        argument_pointer=argument_pointer,
        semantic_heads=argument_role_heads,
        proposal_heads=argument_proposal_heads,
        max_span_tokens=max_span_tokens,
        max_argument_span_tokens_by_type=max_argument_span_tokens_by_type,
        hidden_channels=hidden_channels,
        hidden_channel_widths=hidden_channel_widths,
    )
    identity_geometries = Counter(_geometry(item) for item in identity_training)
    item_weights = {
        id(item): 1.0 / identity_geometries[_geometry(item)] for item in identity_training
    }
    relation_weight, relation_bias = _fit_directional_relation_head(
        identity_training,
        item_weights=item_weights,
        hidden_channels=hidden_channels,
        hidden_channel_widths=hidden_channel_widths,
    )
    (
        relation_query_projection,
        relation_definition_projection,
        relation_tissue_rows,
    ) = _fit_low_rank_relation_tissue(
        identity_training,
        identity_validation,
        relation_weight=relation_weight,
        relation_bias=relation_bias,
        hidden_channels=hidden_channels,
        hidden_channel_widths=hidden_channel_widths,
    )
    uncalibrated_relation_head = DirectionalRelationHead(
        relation_weight,
        relation_bias,
        0.0,
        relation_query_projection,
        relation_definition_projection,
    )
    definition_pointer_scale, definition_pointer_rows = _select_definition_pointer_scale(
        validation,
        relation_head=uncalibrated_relation_head,
        definition_pointer=definition_pointer,
        max_definition_span_tokens=max_definition_span_tokens,
        definition_candidate_strategy=_STABLE_REGISTER_TABLE_STRATEGY,
        hidden_channels=hidden_channels,
        hidden_channel_widths=hidden_channel_widths,
    )
    definition_relation_head = DirectionalRelationHead(
        relation_weight,
        relation_bias,
        definition_pointer_scale,
        relation_query_projection,
        relation_definition_projection,
    )
    operation_length_penalty, penalty_rows = _select_operation_length_penalty(
        validation,
        pointer=operation_pointer,
        classifier=operation_head,
        max_steps=max_steps,
        max_span_tokens=max_span_tokens,
        hidden_channels=hidden_channels,
        hidden_channel_widths=hidden_channel_widths,
    )
    coefficient_body = {
        "operation_pointer": operation_pointer.to_dict(),
        "argument_pointer": argument_pointer.to_dict(),
        "definition_pointer": definition_pointer.to_dict(),
        "operation_head": operation_head.to_dict(),
        "argument_role_heads": [head.to_dict() for head in argument_role_heads],
        "argument_proposal_heads": [head.to_dict() for head in argument_proposal_heads],
        "definition_relation_head": definition_relation_head.to_dict(),
        "operation_length_penalty": operation_length_penalty,
        "argument_role_scale": 1.0,
        "argument_proposal_scale": argument_proposal_scale,
        "definition_relation_scale": 1.0,
        "argument_pointer_scale": 0.5,
        "allow_computed_dependencies": True,
        "register_use_contract": register_use_contract.to_dict(),
    }
    body = {
        "schema": COMPOSITIONAL_SEMANTIC_RECEIPT_SCHEMA,
        "model_basis_sha256": next(iter(bases)),
        "input_grounding_sha256": input_grounding.contract_sha256,
        "training_example_count": len(training),
        "validation_example_count": len(validation),
        "training_example_ids_sha256": _sha(
            sorted(item.ir.source_text_sha256 for item in training)
        ),
        "validation_example_ids_sha256": _sha(
            sorted(item.ir.source_text_sha256 for item in validation)
        ),
        "observed_geometry_support": [
            {"geometry": _geometry_name(geometry), "example_count": count}
            for geometry, count in sorted(geometries.items())
        ],
        "primitive_support": sorted(
            {instruction.op for item in training for instruction in item.ir.instructions}
        ),
        "operation_length_penalty_selection": penalty_rows,
        "chart_decoder": "stable_register_table_probability_kbest_interval_dag_v4",
        "operation_chart_beam": _OPERATION_CHART_BEAM,
        "argument_span_bounds": max_argument_span_tokens_by_type,
        "definition_span_bound": max_definition_span_tokens,
        "definition_candidate_strategy": _STABLE_REGISTER_TABLE_STRATEGY,
        "argument_candidate_strategy": _LOCAL_ARGUMENT_CANDIDATE_STRATEGY,
        "identity_definition_supervision": {
            "selection": "payload_and_operation_disjoint_register_spans_v1",
            "training_examples": len(identity_training),
            "validation_examples": len(identity_validation),
            "fallback_to_all_examples": identity_supervision_fallback,
        },
        "definition_pointer_scale_selection": definition_pointer_rows,
        "relation_tissue_fit": {
            "algorithm": "minibatch_adamw_cross_entropy_v1",
            "selection_objective": "minimum_validation_cross_entropy",
            "rank": int(relation_query_projection.shape[1]),
            "seed": _RELATION_TISSUE_SEED,
            "epochs": _RELATION_TISSUE_EPOCHS,
            "batch_size": _RELATION_TISSUE_BATCH_SIZE,
            "selection_interval": _RELATION_TISSUE_SELECTION_INTERVAL,
            "learning_rate": _RELATION_TISSUE_LEARNING_RATE,
            "weight_decay": _RELATION_TISSUE_WEIGHT_DECAY,
            "gradient_clip": _RELATION_TISSUE_GRADIENT_CLIP,
            "validation_selection": relation_tissue_rows,
        },
        "relation_score_contract": "mention_invariant_conditional_tissue_v1",
        "argument_role_contract": "semantic_and_pointer_proposal_product_v1",
        "argument_proposal_fit": {
            "hard_negative_limit": _POINTER_HARD_NEGATIVES,
            "scale_selection_objective": "minimum_validation_cross_entropy",
            "scale_selection": argument_proposal_scale_rows,
            **argument_proposal_fit,
        },
        "register_use_contract": register_use_contract.to_dict(),
        "global_geometry_classifier_present": False,
        "step_indexed_heads_present": False,
        "family_router_present": False,
        "expected_answers_available": False,
        "verifier_traces_available": False,
        "generated_compiler_text_available": False,
        "correctness_authority": False,
        "coefficient_sha256": _sha(coefficient_body),
    }
    return CompositionalSemanticProgramTransducer(
        hidden_size=hidden_size,
        model_basis_sha256=next(iter(bases)),
        hidden_channels=hidden_channels,
        hidden_channel_widths=hidden_channel_widths,
        input_grounding=input_grounding,
        operation_pointer=operation_pointer,
        argument_pointer=argument_pointer,
        definition_pointer=definition_pointer,
        operation_head=operation_head,
        argument_role_heads=argument_role_heads,
        argument_proposal_heads=argument_proposal_heads,
        definition_relation_head=definition_relation_head,
        definition_candidate_strategy=_STABLE_REGISTER_TABLE_STRATEGY,
        max_steps=max_steps,
        max_inputs=max_inputs,
        max_span_tokens=max_span_tokens,
        max_definition_span_tokens=max_definition_span_tokens,
        max_argument_span_tokens_by_type=max_argument_span_tokens_by_type,
        register_use_contract=register_use_contract,
        operation_chart_beam=_OPERATION_CHART_BEAM,
        operation_length_penalty=operation_length_penalty,
        argument_role_scale=1.0,
        argument_proposal_scale=argument_proposal_scale,
        definition_relation_scale=1.0,
        argument_pointer_scale=0.5,
        allow_computed_dependencies=True,
        training_receipt={**body, "receipt_sha256": _sha(body)},
    )


def refit_compositional_register_identity(
    model: CompositionalSemanticProgramTransducer,
    examples: Sequence[SemanticTransducerTrainingExample],
) -> CompositionalSemanticProgramTransducer:
    """Refit only register identity localization and relation tissue.

    Operation recognition, argument proposal, graph constraints, and input
    grounding are intentionally inherited from the parent receipt. This keeps
    a register-identity repair from silently changing unrelated capabilities.
    """

    training = tuple(
        item
        for item in examples
        if item.split == "train" and _has_symbolic_register_definitions(item)
    )
    validation = tuple(
        item
        for item in examples
        if item.split == "validation" and _has_symbolic_register_definitions(item)
    )
    if not training or not validation:
        raise ValueError("register identity refit needs symbolic train and validation examples")
    if len(training) + len(validation) != sum(
        item.split in {"train", "validation"} for item in examples
    ):
        raise ValueError("register identity refit contains non-symbolic supervision")
    if (
        {item.ir.model_basis_receipt_sha256 for item in (*training, *validation)}
        != {model.model_basis_sha256}
        or {item.tokenizer_identity_sha256 for item in examples}
        != {model.input_grounding.tokenizer_identity_sha256}
        or {
            (item.hidden_channels, item.hidden_channel_widths)
            for item in (*training, *validation)
        }
        != {(model.hidden_channels, model.hidden_channel_widths)}
    ):
        raise ValueError("register identity refit neural basis differs from its parent")

    definition_pointer = _fit_shared_pointer(
        training,
        spans=_register_definition_spans,
    )
    geometries = Counter(_geometry(item) for item in training)
    item_weights = {id(item): 1.0 / geometries[_geometry(item)] for item in training}
    relation_weight, relation_bias = _fit_directional_relation_head(
        training,
        item_weights=item_weights,
        hidden_channels=model.hidden_channels,
        hidden_channel_widths=model.hidden_channel_widths,
    )
    query_projection, definition_projection, relation_rows = _fit_low_rank_relation_tissue(
        training,
        validation,
        relation_weight=relation_weight,
        relation_bias=relation_bias,
        hidden_channels=model.hidden_channels,
        hidden_channel_widths=model.hidden_channel_widths,
    )
    uncalibrated = DirectionalRelationHead(
        relation_weight,
        relation_bias,
        0.0,
        query_projection,
        definition_projection,
    )
    pointer_scale, pointer_rows = _select_definition_pointer_scale(
        validation,
        relation_head=uncalibrated,
        definition_pointer=definition_pointer,
        max_definition_span_tokens=model.max_definition_span_tokens,
        definition_candidate_strategy=_STABLE_REGISTER_TABLE_STRATEGY,
        hidden_channels=model.hidden_channels,
        hidden_channel_widths=model.hidden_channel_widths,
    )
    definition_relation_head = DirectionalRelationHead(
        relation_weight,
        relation_bias,
        pointer_scale,
        query_projection,
        definition_projection,
    )
    coefficient = model._coefficient_body()
    coefficient["definition_pointer"] = definition_pointer.to_dict()
    coefficient["definition_relation_head"] = definition_relation_head.to_dict()
    body = {
        key: value for key, value in model.training_receipt.items() if key != "receipt_sha256"
    }
    body.update(
        {
            "schema": COMPOSITIONAL_SEMANTIC_RECEIPT_SCHEMA,
            "chart_decoder": "stable_register_table_probability_kbest_interval_dag_v4",
            "definition_candidate_strategy": _STABLE_REGISTER_TABLE_STRATEGY,
            "argument_candidate_strategy": _LOCAL_ARGUMENT_CANDIDATE_STRATEGY,
            "definition_pointer_scale_selection": pointer_rows,
            "identity_definition_supervision": {
                "selection": "payload_and_operation_disjoint_register_spans_v1",
                "training_examples": len(training),
                "validation_examples": len(validation),
                "fallback_to_all_examples": False,
            },
            "identity_refit": {
                "schema": "aura.semantic_program_register_identity_refit.v1",
                "parent_transducer_receipt_sha256": model.receipt_sha256,
                "training_example_ids_sha256": _sha(
                    sorted(item.ir.source_text_sha256 for item in training)
                ),
                "validation_example_ids_sha256": _sha(
                    sorted(item.ir.source_text_sha256 for item in validation)
                ),
                "inherited_coefficient_groups": [
                    "operation_pointer",
                    "argument_pointer",
                    "operation_head",
                    "argument_role_heads",
                    "argument_proposal_heads",
                    "operation_length_penalty",
                    "register_use_contract",
                ],
                "refit_coefficient_groups": [
                    "definition_pointer",
                    "definition_relation_head",
                ],
            },
            "relation_tissue_fit": {
                "algorithm": "minibatch_adamw_cross_entropy_v1",
                "selection_objective": "minimum_validation_cross_entropy",
                "rank": int(query_projection.shape[1]),
                "seed": _RELATION_TISSUE_SEED,
                "epochs": _RELATION_TISSUE_EPOCHS,
                "batch_size": _RELATION_TISSUE_BATCH_SIZE,
                "selection_interval": _RELATION_TISSUE_SELECTION_INTERVAL,
                "learning_rate": _RELATION_TISSUE_LEARNING_RATE,
                "weight_decay": _RELATION_TISSUE_WEIGHT_DECAY,
                "gradient_clip": _RELATION_TISSUE_GRADIENT_CLIP,
                "validation_selection": relation_rows,
            },
            "coefficient_sha256": _sha(coefficient),
        }
    )
    return replace(
        model,
        schema=COMPOSITIONAL_SEMANTIC_TRANSDUCER_SCHEMA,
        definition_pointer=definition_pointer,
        definition_relation_head=definition_relation_head,
        definition_candidate_strategy=_STABLE_REGISTER_TABLE_STRATEGY,
        training_receipt={**body, "receipt_sha256": _sha(body)},
    )


def _pointer_from_dict(value: Any) -> LinearPointerHead:
    if not isinstance(value, Mapping):
        raise ValueError("compositional pointer payload is invalid")
    return LinearPointerHead(
        np.asarray(value["start_weight"], dtype=np.float32),
        float(value["start_bias"]),
        np.asarray(value["end_weight"], dtype=np.float32),
        float(value["end_bias"]),
    )


def compositional_semantic_program_transducer_from_dict(
    payload: Any,
) -> CompositionalSemanticProgramTransducer:
    """Reload one immutable transducer without fitting or calibration."""

    if not isinstance(payload, Mapping):
        raise ValueError("compositional semantic transducer payload is invalid")
    operation = payload.get("operation_head")
    if (
        not isinstance(operation, Mapping)
        or operation.get("schema") != "aura.semantic_program_multiview_classifier.v1"
        or operation.get("modes") != [_OPERATION_MODE]
        or not isinstance(operation.get("heads"), list)
        or len(operation["heads"]) != 1
    ):
        raise ValueError("compositional operation head payload is invalid")
    raw_head = operation["heads"][0]
    operation_head = MultiViewClassifierHead(
        (_OPERATION_MODE,),
        (
            LinearClassifierHead(
                tuple(raw_head["labels"]),
                np.asarray(raw_head["weight"], dtype=np.float32),
                np.asarray(raw_head["bias"], dtype=np.float32),
            ),
        ),
    )
    relation = payload["definition_relation_head"]
    schema = str(payload["schema"])
    definition_candidate_strategy = str(
        payload.get(
            "definition_candidate_strategy",
            _LEGACY_DEFINITION_CANDIDATE_STRATEGY,
        )
    )
    return CompositionalSemanticProgramTransducer(
        hidden_size=int(payload["hidden_size"]),
        model_basis_sha256=str(payload["model_basis_sha256"]),
        hidden_channels=tuple(payload["hidden_channels"]),
        hidden_channel_widths=tuple(int(value) for value in payload["hidden_channel_widths"]),
        input_grounding=semantic_input_grounding_contract_from_dict(payload["input_grounding"]),
        operation_pointer=_pointer_from_dict(payload["operation_pointer"]),
        argument_pointer=_pointer_from_dict(payload["argument_pointer"]),
        definition_pointer=_pointer_from_dict(payload["definition_pointer"]),
        operation_head=operation_head,
        argument_role_heads=tuple(
            LinearArgumentRoleHead(
                np.asarray(value["weight"], dtype=np.float32),
                float(value["bias"]),
            )
            for value in payload["argument_role_heads"]
        ),
        argument_proposal_heads=tuple(
            LinearArgumentRoleHead(
                np.asarray(value["weight"], dtype=np.float32),
                float(value["bias"]),
            )
            for value in payload["argument_proposal_heads"]
        ),
        definition_relation_head=DirectionalRelationHead(
            np.asarray(relation["weight"], dtype=np.float32),
            float(relation["bias"]),
            float(relation["pointer_scale"]),
            np.asarray(relation["query_projection"], dtype=np.float32),
            np.asarray(relation["definition_projection"], dtype=np.float32),
        ),
        definition_candidate_strategy=definition_candidate_strategy,
        max_steps=int(payload["max_steps"]),
        max_inputs=int(payload["max_inputs"]),
        max_span_tokens=int(payload["max_span_tokens"]),
        max_definition_span_tokens=int(payload["max_definition_span_tokens"]),
        max_argument_span_tokens_by_type={
            str(key): int(value)
            for key, value in payload["max_argument_span_tokens_by_type"].items()
        },
        register_use_contract=RegisterUseContract(
            **{str(key): value for key, value in payload["register_use_contract"].items()}
        ),
        operation_chart_beam=int(
            payload.get(
                "operation_chart_beam",
                payload["training_receipt"]["operation_chart_beam"],
            )
        ),
        operation_length_penalty=float(payload["operation_length_penalty"]),
        argument_role_scale=float(payload["argument_role_scale"]),
        argument_proposal_scale=float(payload["argument_proposal_scale"]),
        definition_relation_scale=float(payload["definition_relation_scale"]),
        argument_pointer_scale=float(payload["argument_pointer_scale"]),
        allow_computed_dependencies=bool(payload["allow_computed_dependencies"]),
        training_receipt=dict(payload["training_receipt"]),
        schema=schema,
    )


__all__ = [
    "COMPOSITIONAL_SEMANTIC_RECEIPT_SCHEMA",
    "COMPOSITIONAL_SEMANTIC_TRANSDUCER_SCHEMA",
    "CompositionalSemanticProgramTransducer",
    "DirectionalRelationHead",
    "LinearArgumentRoleHead",
    "RegisterUseContract",
    "compositional_semantic_program_transducer_from_dict",
    "fit_compositional_semantic_program_transducer",
    "refit_compositional_register_identity",
]
