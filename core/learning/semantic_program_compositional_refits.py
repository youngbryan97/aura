"""Every refit the compositional transducer can be rebuilt through.

Lifted whole out of `semantic_program_compositional_transducer`, which had
grown past the 2000-line ceiling. Every name taken from that module is
imported at CALL time: it imports this one to re-export these five, and a
test that patches a name on it has to reach the code that reads it.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.learning.semantic_program_transducer import (
        SemanticTransducerTrainingExample,
    )

    from .semantic_program_compositional_transducer import (
        CompositionalSemanticProgramTransducer,
    )


def refit_compositional_operation_pointer(
    model: CompositionalSemanticProgramTransducer,
    examples: Sequence[SemanticTransducerTrainingExample],
) -> CompositionalSemanticProgramTransducer:
    """Refit shared operation boundaries without contradictory target labels."""
    from .semantic_program_compositional_transducer import (
        _fit_shared_pointer,
        _select_operation_length_penalty,
        _sha,
        replace,
    )

    training = tuple(item for item in examples if item.split == "train")
    validation = tuple(item for item in examples if item.split == "validation")
    if not training or not validation:
        raise ValueError("operation pointer refit needs train and validation examples")
    selected = (*training, *validation)
    if (
        {item.ir.model_basis_receipt_sha256 for item in selected} != {model.model_basis_sha256}
        or {item.tokenizer_identity_sha256 for item in selected}
        != {model.input_grounding.tokenizer_identity_sha256}
        or {(item.hidden_channels, item.hidden_channel_widths) for item in selected}
        != {(model.hidden_channels, model.hidden_channel_widths)}
    ):
        raise ValueError("operation pointer refit neural basis differs from its parent")
    train_ids = {item.ir.source_text_sha256 for item in training}
    validation_ids = {item.ir.source_text_sha256 for item in validation}
    if train_ids & validation_ids:
        raise ValueError("operation pointer refit train and validation overlap")
    pointer = _fit_shared_pointer(
        training,
        spans=lambda item: tuple(x.operation_span for x in item.ir.instructions),
    )
    penalty, calibration = _select_operation_length_penalty(
        validation,
        pointer=pointer,
        classifier=model.operation_head,
        max_steps=model.max_steps,
        max_span_tokens=model.max_span_tokens,
        hidden_channels=model.hidden_channels,
        hidden_channel_widths=model.hidden_channel_widths,
    )
    coefficient = model._coefficient_body()
    coefficient["operation_pointer"] = pointer.to_dict()
    coefficient["operation_length_penalty"] = penalty
    body = {key: value for key, value in model.training_receipt.items() if key != "receipt_sha256"}
    body["coefficient_sha256"] = _sha(coefficient)
    body["operation_pointer_refit"] = {
        "schema": "aura.semantic_program_operation_pointer_refit.v1",
        "parent_transducer_receipt_sha256": model.receipt_sha256,
        "negative_label_policy": "exclude_all_same_head_positive_boundaries_v1",
        "training_example_ids_sha256": _sha(sorted(train_ids)),
        "validation_example_ids_sha256": _sha(sorted(validation_ids)),
        "training_examples": len(training),
        "validation_examples": len(validation),
        "test_examples_used": 0,
        "calibration": calibration,
        "serving_authority": False,
    }
    return replace(
        model,
        operation_pointer=pointer,
        operation_length_penalty=penalty,
        training_receipt={**body, "receipt_sha256": _sha(body)},
    )

def refit_compositional_definition_pointer(
    model: CompositionalSemanticProgramTransducer,
    examples: Sequence[SemanticTransducerTrainingExample],
) -> CompositionalSemanticProgramTransducer:
    """Fit definition boundaries across both anchor and symbolic supervision."""
    from .semantic_program_compositional_transducer import (
        Counter,
        _fit_shared_pointer,
        _register_definition_spans,
        _sha,
        replace,
    )

    training = tuple(item for item in examples if item.split == "train")
    validation = tuple(item for item in examples if item.split == "validation")
    if not training or not validation:
        raise ValueError("definition pointer refit needs train and validation examples")
    selected = (*training, *validation)
    if (
        {item.ir.model_basis_receipt_sha256 for item in selected} != {model.model_basis_sha256}
        or {item.tokenizer_identity_sha256 for item in selected}
        != {model.input_grounding.tokenizer_identity_sha256}
        or {(item.hidden_channels, item.hidden_channel_widths) for item in selected}
        != {(model.hidden_channels, model.hidden_channel_widths)}
    ):
        raise ValueError("definition pointer refit neural basis differs from its parent")
    train_ids = {item.ir.source_text_sha256 for item in training}
    validation_ids = {item.ir.source_text_sha256 for item in validation}
    if train_ids & validation_ids or len(train_ids) != len(training) or len(validation_ids) != len(validation):
        raise ValueError("definition pointer refit source examples duplicate or overlap")
    pointer = _fit_shared_pointer(training, spans=_register_definition_spans)
    coefficient = model._coefficient_body()
    coefficient["definition_pointer"] = pointer.to_dict()
    body = {key: value for key, value in model.training_receipt.items() if key != "receipt_sha256"}
    body["coefficient_sha256"] = _sha(coefficient)
    body["definition_pointer_refit"] = {
        "schema": "aura.semantic_program_definition_pointer_refit.v1",
        "parent_transducer_receipt_sha256": model.receipt_sha256,
        "negative_label_policy": "exclude_all_same_head_positive_boundaries_v1",
        "supervision_selection": "all_source_register_definitions_v1",
        "training_example_ids_sha256": _sha(sorted(train_ids)),
        "validation_example_ids_sha256": _sha(sorted(validation_ids)),
        "training_examples": len(training),
        "validation_examples": len(validation),
        "training_definition_targets_sha256": _sha(sorted(
            (
                item.ir.source_text_sha256,
                item.register_definition_origin,
                [[span.start, span.end] for span in _register_definition_spans(item)],
            )
            for item in training
        )),
        "training_definition_origins": dict(Counter(
            item.register_definition_origin for item in training
        )),
        "test_examples_used": 0,
        "validation_examples_used_for_fitting": 0,
        "relation_coefficients_and_scale_preserved": True,
        "serving_authority": False,
    }
    return replace(
        model, definition_pointer=pointer,
        training_receipt={**body, "receipt_sha256": _sha(body)},
    )

def refit_compositional_argument_proposals(
    model: CompositionalSemanticProgramTransducer,
    examples: Sequence[SemanticTransducerTrainingExample],
    *,
    refit_pointer: bool = False,
) -> CompositionalSemanticProgramTransducer:
    """Refit proposal scoring, optionally rebuilding its source-trained pointer."""
    from .semantic_program_compositional_transducer import (
        _fit_argument_proposal_heads,
        _fit_shared_pointer,
        _select_argument_proposal_scale,
        _sha,
        replace,
    )


    training = tuple(item for item in examples if item.split == "train")
    validation = tuple(item for item in examples if item.split == "validation")
    if not training or not validation:
        raise ValueError("argument proposal refit needs train and validation examples")
    selected = (*training, *validation)
    if (
        {item.ir.model_basis_receipt_sha256 for item in selected} != {model.model_basis_sha256}
        or {item.tokenizer_identity_sha256 for item in selected}
        != {model.input_grounding.tokenizer_identity_sha256}
        or {(item.hidden_channels, item.hidden_channel_widths) for item in selected}
        != {(model.hidden_channels, model.hidden_channel_widths)}
    ):
        raise ValueError("argument proposal refit neural basis differs from its parent")
    train_ids = {item.ir.source_text_sha256 for item in training}
    validation_ids = {item.ir.source_text_sha256 for item in validation}
    if train_ids & validation_ids:
        raise ValueError("argument proposal refit train and validation overlap")
    pointer = model.argument_pointer
    if refit_pointer:
        pointer = _fit_shared_pointer(
            training,
            spans=lambda item: tuple(
                span for instruction in item.ir.instructions
                for span in instruction.argument_spans
            ),
        )
    heads, fit = _fit_argument_proposal_heads(
        training,
        argument_pointer=pointer,
        max_arity=len(model.argument_role_heads),
        max_span_tokens=model.max_span_tokens,
        max_argument_span_tokens_by_type=model.max_argument_span_tokens_by_type,
        hidden_channels=model.hidden_channels,
        hidden_channel_widths=model.hidden_channel_widths,
    )
    scale, calibration = _select_argument_proposal_scale(
        validation,
        argument_pointer=pointer,
        semantic_heads=model.argument_role_heads,
        proposal_heads=heads,
        max_span_tokens=model.max_span_tokens,
        max_argument_span_tokens_by_type=model.max_argument_span_tokens_by_type,
        hidden_channels=model.hidden_channels,
        hidden_channel_widths=model.hidden_channel_widths,
    )
    coefficient = model._coefficient_body()
    coefficient["argument_pointer"] = pointer.to_dict()
    coefficient["argument_proposal_heads"] = [head.to_dict() for head in heads]
    coefficient["argument_proposal_scale"] = scale
    body = {key: value for key, value in model.training_receipt.items() if key != "receipt_sha256"}
    body["coefficient_sha256"] = _sha(coefficient)
    body["argument_proposal_fit"] = {
        **dict(model.training_receipt["argument_proposal_fit"]),
        **fit,
        "scale_selection": calibration,
    }
    body["argument_proposal_refit"] = {
        "schema": "aura.semantic_program_argument_proposal_refit.v1",
        "parent_transducer_receipt_sha256": model.receipt_sha256,
        "candidate_source": "runtime_per_operation_clause_proposals_v1",
        "training_example_ids_sha256": _sha(sorted(train_ids)),
        "validation_example_ids_sha256": _sha(sorted(validation_ids)),
        "training_examples": len(training),
        "validation_examples": len(validation),
        "test_examples_used": 0,
        "fit": fit,
        "calibration": calibration,
        "serving_authority": False,
    }
    if refit_pointer:
        body["argument_pointer_refit"] = {
            "schema": "aura.semantic_program_argument_pointer_refit.v1",
            "parent_transducer_receipt_sha256": model.receipt_sha256,
            "negative_label_policy": "exclude_all_same_head_positive_boundaries_v1",
            "training_example_ids_sha256": _sha(sorted(train_ids)),
            "validation_example_ids_sha256": _sha(sorted(validation_ids)),
            "training_examples": len(training),
            "validation_examples": len(validation),
            "test_examples_used": 0,
            "dependent_proposal_heads_refitted": True,
            "serving_authority": False,
        }
    return replace(
        model,
        argument_pointer=pointer,
        argument_proposal_heads=heads,
        argument_proposal_scale=scale,
        training_receipt={**body, "receipt_sha256": _sha(body)},
    )

def refit_compositional_argument_rankings(
    model: CompositionalSemanticProgramTransducer,
    examples: Sequence[SemanticTransducerTrainingExample],
    *,
    preserve_coreferent_mentions: bool=False,
    use_runtime_operation_views: bool=False,
    runtime_operation_view_charts: int=0,
    runtime_mention_margin: bool=False,
    progress: Any=None,
) -> CompositionalSemanticProgramTransducer:
    """Fit source-only argument choices while preserving other learned modules."""
    from core.learning.semantic_argument_ranking import fit_pairwise_argument_weight

    from .semantic_program_compositional_transducer import (
        LinearArgumentRoleHead,
        _argument_proposal_rows,
        _sha,
        np,
        replace,
    )

    training = tuple(item for item in examples if item.split == "train")
    validation = tuple(item for item in examples if item.split == "validation")
    if not training or not validation:
        raise ValueError("argument ranking refit needs train and validation examples")
    selected = (*training, *validation)
    if (
        {item.ir.model_basis_receipt_sha256 for item in selected} != {model.model_basis_sha256}
        or {item.tokenizer_identity_sha256 for item in selected}
        != {model.input_grounding.tokenizer_identity_sha256}
        or {(item.hidden_channels, item.hidden_channel_widths) for item in selected}
        != {(model.hidden_channels, model.hidden_channel_widths)}
    ):
        raise ValueError("argument ranking refit neural basis differs from its parent")
    train_ids = [item.ir.source_text_sha256 for item in training]
    validation_ids = [item.ir.source_text_sha256 for item in validation]
    if (
        len(set(train_ids)) != len(train_ids)
        or len(set(validation_ids)) != len(validation_ids)
        or set(train_ids) & set(validation_ids)
    ):
        raise ValueError("argument ranking source splits duplicate or overlap")
    if type(use_runtime_operation_views) is not bool:
        raise ValueError("runtime operation views must be a boolean")
    if type(runtime_operation_view_charts) is not int or not 0 <= runtime_operation_view_charts <= 64:
        raise ValueError("runtime operation view charts must be inside [0, 64]")
    if runtime_operation_view_charts and not use_runtime_operation_views:
        raise ValueError("runtime operation chart views require runtime operation views")
    if type(runtime_mention_margin) is not bool:
        raise ValueError("runtime mention margin must be a boolean")
    fitting_examples = training
    runtime_view_receipt = None
    if use_runtime_operation_views:
        from core.learning.semantic_runtime_argument_views import runtime_argument_training_views

        fitting_examples, runtime_view_receipt = runtime_argument_training_views(
            model, training, max_operation_charts=runtime_operation_view_charts,
            progress=progress,
        )
    heads, fits = [], []
    for position, (role, proposal) in enumerate(zip(
        model.argument_role_heads, model.argument_proposal_heads, strict=True
    )):
        fixed_scores = [] if runtime_mention_margin else None
        features, labels, weights, _, _ = _argument_proposal_rows(
            fitting_examples, argument_pointer=model.argument_pointer, position=position,
            max_span_tokens=model.max_span_tokens,
            max_argument_span_tokens_by_type=model.max_argument_span_tokens_by_type,
            hidden_channels=model.hidden_channels,
            hidden_channel_widths=model.hidden_channel_widths,
            include_semantic_negatives=True,
            preserve_coreferent_mentions=preserve_coreferent_mentions,
            full_runtime_mentions=runtime_mention_margin,
            fixed_pointer_scores=fixed_scores,
            pointer_scale=model.argument_pointer_scale,
            factorized_features=runtime_mention_margin,
            balance_source_views=bool(runtime_operation_view_charts),
        )
        weight, fit = fit_pairwise_argument_weight(
            features, labels, weights,
            initial_weight=model.argument_role_scale * role.weight
            + model.argument_proposal_scale * proposal.weight,
            fixed_scores=None if fixed_scores is None else np.asarray(fixed_scores),
        )
        del features
        # Keep the proposal module and its calibration fixed. The role module
        # carries the residual needed for their combined log odds to equal the ranker.
        heads.append(LinearArgumentRoleHead(
            (weight - model.argument_proposal_scale * proposal.weight) / model.argument_role_scale,
            -model.argument_proposal_scale * proposal.bias / model.argument_role_scale,
        ))
        fits.append(fit)
    candidate = model._with_coefficients(argument_role_heads=tuple(heads))
    body = {key: value for key, value in candidate.training_receipt.items() if key != "receipt_sha256"}
    body["argument_score_strategy"] = "conditional_log_odds_v1"
    body["argument_ranking_refit"] = {
        "schema": "aura.semantic_argument_ranking_refit.v1",
        "parent_transducer_receipt_sha256": model.receipt_sha256,
        "negative_source": (
            "runtime_pointer_and_noncoreferent_source_spans_v2"
            if preserve_coreferent_mentions
            else "runtime_pointer_shortlist_and_source_semantic_spans_v1"
        ),
        "coefficient_parameterization": "combined_ranker_as_role_residual_v1",
        "role_head_alone_is_calibrated_probability": False,
        "training_examples": len(training),
        "validation_examples": len(validation),
        "training_example_ids_sha256": _sha(sorted(train_ids)),
        "validation_example_ids_sha256": _sha(sorted(validation_ids)),
        "validation_used_for_fit": False,
        "test_examples_used": 0,
        "fits": fits,
        "serving_authority": False,
    }
    if runtime_view_receipt is not None:
        body["argument_ranking_refit"]["runtime_operation_views"] = runtime_view_receipt
    if runtime_mention_margin:
        body["argument_ranking_refit"]["mention_objective"] = "runtime_pointer_margin_v1"
        body["argument_ranking_refit"]["negative_limit"] = None
        body["argument_ranking_refit"]["fixed_pointer_scale"] = model.argument_pointer_scale
        body["argument_ranking_refit"]["graph_relation_terms_fitted"] = False
    return replace(candidate, training_receipt={**body, "receipt_sha256": _sha(body)})

def refit_compositional_register_identity(
    model: CompositionalSemanticProgramTransducer,
    examples: Sequence[SemanticTransducerTrainingExample],
) -> CompositionalSemanticProgramTransducer:
    """Refit only register identity localization and relation tissue.

    Operation recognition, argument proposal, graph constraints, and input
    grounding are intentionally inherited from the parent receipt. This keeps
    a register-identity repair from silently changing unrelated capabilities.
    """
    from .semantic_program_compositional_transducer import (
        _LOCAL_ARGUMENT_CANDIDATE_STRATEGY,
        _RELATION_TISSUE_BATCH_SIZE,
        _RELATION_TISSUE_EPOCHS,
        _RELATION_TISSUE_GRADIENT_CLIP,
        _RELATION_TISSUE_LEARNING_RATE,
        _RELATION_TISSUE_SEED,
        _RELATION_TISSUE_SELECTION_INTERVAL,
        _RELATION_TISSUE_WEIGHT_DECAY,
        _STABLE_REGISTER_TABLE_STRATEGY,
        COMPOSITIONAL_SEMANTIC_RECEIPT_SCHEMA,
        COMPOSITIONAL_SEMANTIC_TRANSDUCER_SCHEMA,
        Counter,
        DirectionalRelationHead,
        _fit_directional_relation_head,
        _fit_low_rank_relation_tissue,
        _fit_shared_pointer,
        _geometry,
        _has_symbolic_register_definitions,
        _register_definition_spans,
        _select_definition_pointer_scale,
        _sha,
        replace,
    )


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
                "pointer_negative_label_policy": "exclude_all_same_head_positive_boundaries_v1",
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
