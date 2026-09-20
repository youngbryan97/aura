"""Small, split-bound mechanism trials before expensive semantic refits."""

from collections import defaultdict, deque

from core.learning.semantic_graph_constraints import fit_complete_graph_constraints
from core.learning.semantic_graph_counterexamples import (
    compare_program_meanings,
    counterfactual_inputs,
)
from core.learning.semantic_joint_graph_learning import (
    align_source_input_registers,
    mine_runtime_graph_contrast,
    mine_source_binding_constraint,
    score_annotated_graph,
    source_operation_constraints,
    source_operation_pointer_constraints,
    source_operation_supervision,
)
from core.learning.semantic_program_campaign import _sha
from core.learning.semantic_program_shared_transducer import _geometry
from core.learning.semantic_runtime_graph_retention import mine_runtime_graph_constraints


def select_trial_examples(examples, *, split, count):
    """Select by source identity across geometries, never by measured correctness."""
    if split not in {"train", "validation"} or type(count) is not int or count < 1:
        raise ValueError("trial selection needs a source split and positive count")
    grouped = defaultdict(list)
    for item in examples:
        if item.split == split:
            grouped[_geometry(item)].append(item)
    if not grouped:
        raise ValueError(f"trial split is empty: {split}")
    groups = deque(deque(sorted(rows, key=lambda item: item.ir.source_text_sha256))
                   for _, rows in sorted(grouped.items()))
    selected = []
    while groups and len(selected) < count:
        group = groups.popleft()
        selected.append(group.popleft())
        if group:
            groups.append(group)
    return tuple(selected)


def _observe(model, item):
    outcome = model.decode(source_token_ids=item.ir.source_token_ids, hidden_states=item.hidden_states,
        public_inputs=item.public_inputs, source_text_sha256=item.ir.source_text_sha256,
        model_basis_sha256=item.ir.model_basis_receipt_sha256)
    row = {"source_text_sha256": item.ir.source_text_sha256, "split": item.split,
           "geometry": _geometry(item), "accepted": outcome.ir is not None, "refusal": outcome.refusal,
           "source_grounding_aligned": None, "annotated_graph_feasible": None,
           "semantic_status": "unmeasured"}
    if outcome.ir is None:
        return {**row, "semantic_status": "decode_refused"}
    try:
        instructions, _ = align_source_input_registers(item, outcome.ir.input_spans)
    except ValueError as exc:
        return {**row, "source_grounding_aligned": False, "failure": str(exc)}
    row["source_grounding_aligned"] = True
    from core.learning.semantic_argument_optimization import ArgumentOptimizationIncompleteError
    try:
        target = score_annotated_graph(model, item, instructions, outcome.ir.input_spans)
        selected = score_annotated_graph(model, item, outcome.ir.instructions, outcome.ir.input_spans)
    except ArgumentOptimizationIncompleteError as exc:
        return {**row, "failure": str(exc)}
    # Scoring with annotated operations does not establish runtime search coverage.
    row["annotated_graph_feasible"] = target is not None
    if target is None or selected is None:
        return row
    comparison = compare_program_meanings(target["program"], selected["program"],
                                         counterfactual_inputs(item.public_inputs))
    return {**row, "semantic_status": comparison["status"], "comparison": comparison,
            "selected_program_sha256": selected["program"].sha(),
            "target_program_sha256": target["program"].sha()}


def _constraint_group_summary(groups, fit):
    """Attribute measured fitting deficits without interpreting missing data as zero."""
    before, after = fit.get("initial_margins"), fit.get("stored_margins")
    if before is None or after is None:
        return None
    if len(before) != len(after) or sorted(i for indices in groups.values() for i in indices) != list(range(len(before))):
        raise ValueError("constraint attribution does not partition the measured fit")
    target = fit["required_margin"]
    return {name: {"count": len(indices),
            "initial_wrong_or_tied": sum(before[i] <= 0 for i in indices),
            "stored_wrong_or_tied": sum(after[i] <= 0 for i in indices),
            "initial_squared_deficit": sum(max(target - before[i], 0.) ** 2 for i in indices),
            "stored_squared_deficit": sum(max(target - after[i], 0.) ** 2 for i in indices)}
            for name, indices in groups.items()}


def run_semantic_graph_trial(model, examples, *, training_count=8, validation_count=8,
                             training_pool_count=None, steps=20, max_charts=32, progress=None,
                             objective="squared_deficit", operation_retention_count=None,
                             learn_operation_pointer=False, update_rule="working_face",
                             boundary_policy="supervised", learn_operations=True,
                             relation_metric="coefficient_euclidean"):
    """Fit only selected source rows and independently replay both small cohorts.

    This returns no deployable candidate. Validation rows never enter mining
    or fitting, and this development trial cannot establish fresh transfer.
    """
    ids = [item.ir.source_text_sha256 for item in examples]
    if len(set(ids)) != len(ids):
        raise ValueError("trial source identities overlap")
    if training_pool_count is None:
        training_pool_count = training_count
    if type(training_pool_count) is not int or training_pool_count < training_count:
        raise ValueError("training pool must cover the requested training cohort")
    training_pool = select_trial_examples(examples, split="train", count=training_pool_count)
    validation = select_trial_examples(examples, split="validation", count=validation_count)
    decoder_policy = model.training_receipt.get("operation_assignment_policy", "first_feasible_v1")
    if decoder_policy not in {"first_feasible_v1", "joint_factor_score_v2"}:
        raise ValueError("runtime graph trial requires a supported selection policy")
    pool_observations = []
    for item in training_pool:
        pool_observations.append(_observe(model, item))
        if progress:
            progress({"stage": "trial_training_scan", "completed": len(pool_observations),
                      "semantic_status": pool_observations[-1]["semantic_status"]})
    # Select witnessed training failures before retention controls. No
    # validation outcome participates in this acquisition decision.
    selected = sorted(range(len(training_pool)),
                      key=lambda i: pool_observations[i]["semantic_status"] not in
                      {"different", "decode_refused"})[:training_count]
    training = tuple(training_pool[i] for i in selected)
    before = [pool_observations[i] for i in selected]
    for item in validation:
        before.append(_observe(model, item))
        if progress:
            progress({"stage": "trial_before", "completed": len(before), "row": before[-1]})
    operation_training = (training if operation_retention_count is None else
                          select_trial_examples(examples, split="train", count=operation_retention_count))
    if not {item.ir.source_text_sha256 for item in training}.issubset(
            {item.ir.source_text_sha256 for item in operation_training}):
        raise ValueError("operation retention must include the graph training cohort")
    operation_supervision = source_operation_supervision(model, operation_training)
    constraints, groups = [], defaultdict(list)

    def retain(name, rows):
        rows = tuple(rows)
        groups[name].extend(range(len(constraints), len(constraints) + len(rows)))
        constraints.extend(rows)

    retain("operation_labels", source_operation_constraints(model, operation_supervision))
    if learn_operation_pointer:
        # Pointer retention follows the full source-retention cohort, just as
        # operation-label retention does. The fitted candidate must not learn
        # boundaries from only the small witnessed-error cohort.
        retain("operation_boundaries", source_operation_pointer_constraints(model, operation_training,
                                                                             policy=boundary_policy))
    records = []
    for item in training:
        # The winning joint graph need not occur in the first operation charts.
        selected_contrast, selected_record = mine_runtime_graph_contrast(
            model, item, learn_arguments=True, learn_operation_pointer=learn_operation_pointer)
        if selected_contrast is not None:
            retain("selected_runtime_errors", (selected_contrast,))
        rows, record = mine_runtime_graph_constraints(model, item, max_charts=max_charts, learn_arguments=True,
                                                      learn_operation_pointer=learn_operation_pointer)
        record["selected_decode"] = selected_record
        records.append(record)
        retain("runtime_graph_competitors", rows)
        binding, binding_record = mine_source_binding_constraint(
            model, item, learn_arguments=True,
        )
        record["binding_constraint"] = binding_record
        if binding is not None:
            retain("source_binding_competitors", (binding,))
        if progress:
            progress({"stage": "trial_mining", "completed": len(records), "pairs": len(constraints), "row": record})
    candidate, fit = fit_complete_graph_constraints(model, tuple(constraints),
        scale=model.definition_relation_scale, steps=steps, adaptive_step=True, progress=progress,
        objective=objective, learn_operation_pointer=learn_operation_pointer, update_rule=update_rule,
        learn_operations=learn_operations, relation_metric=relation_metric)
    after = []
    for item in (*training, *validation):
        after.append(_observe(candidate, item))
        if progress:
            progress({"stage": "trial_after", "completed": len(after), "row": after[-1]})
    summaries = {}
    for split in ("train", "validation"):
        pairs = [(a, b) for a, b in zip(before, after, strict=True) if a["split"] == split]
        summaries[split] = {"total": len(pairs),
            "before_equivalent": sum(a["semantic_status"] == "equivalent" for a, _ in pairs),
            "after_equivalent": sum(b["semantic_status"] == "equivalent" for _, b in pairs),
            "gains": sum(a["semantic_status"] != "equivalent" and b["semantic_status"] == "equivalent" for a, b in pairs),
            "regressions": sum(a["semantic_status"] == "equivalent" and b["semantic_status"] != "equivalent" for a, b in pairs),
            "unmeasured_after": sum(b["semantic_status"] in {"unmeasured", "unknown"} for _, b in pairs),
            "decode_refusals_before": sum(a["semantic_status"] == "decode_refused" for a, _ in pairs),
            "decode_refusals_after": sum(b["semantic_status"] == "decode_refused" for _, b in pairs)}
    blockers = []
    if any(row["semantic_status"] == "decode_refused" for row in after):
        blockers.append("autonomous_decode_refused")
    if any(row["accepted"] and (not row["source_grounding_aligned"] or
                               not row["annotated_graph_feasible"]) for row in (*before, *after)):
        blockers.append("grounding_or_annotated_graph_feasibility")
    if any(row["semantic_status"] in {"unmeasured", "unknown"} for row in (*before, *after)):
        blockers.append("semantic_verification_unmeasured")
    if any(row.get("status") not in {"counterexamples", "no_witnessed_competitor"} for row in records):
        blockers.append("runtime_constraint_mining_unavailable")
    if any(row["selected_decode"]["status"] not in {"equivalent", "counterexample"} for row in records):
        blockers.append("selected_decode_constraint_unavailable")
    if any(not row["operation_search_complete"] for row in records):
        blockers.append("operation_retention_search_incomplete")
    if any(row.get("status") in {"search_incomplete", "alternative_search_incomplete", "equivalence_unresolved"}
           for record in records for row in record["charts"]):
        blockers.append("argument_retention_search_unresolved")
    if fit["stored_wrong_or_tied"]:
        blockers.append("retained_constraints_unsatisfied")
    if any(row["regressions"] for row in summaries.values()):
        blockers.append("autonomous_decode_regression")
    if not any(row["gains"] for row in summaries.values()):
        blockers.append("no_measured_semantic_improvement")
    body = {"schema": "aura.semantic_graph_trial.v3", "parent": model.receipt_sha256,
            "serving_authority": False, "promotion_allowed": False, "fresh_transfer_claim": False,
            "selection_policy": "source_hash_geometry_round_robin_v1",
            "decoder_selection_policy": decoder_policy,
            "selection_objective": "runtime_policy_aligned_v1",
            "training_acquisition_policy": "witnessed_training_failures_then_retention_v1",
            "training_pool_observations": pool_observations,
            "training_sources": [item.ir.source_text_sha256 for item in training],
            "operation_retention_sources": [item.ir.source_text_sha256 for item in operation_training],
            "validation_sources": [item.ir.source_text_sha256 for item in validation],
            "validation_used_for_fit": False, "test_examples_used": 0,
            "boundary_policy": boundary_policy,
            "before": before, "after": after, "mining": records, "fit": fit,
            "constraint_groups": _constraint_group_summary(groups, fit),
            "summaries": summaries, "larger_development_run_ready": not blockers, "blockers": blockers}
    return {**body, "receipt_sha256": _sha(body)}
