"""What is wrong with the evidence a latent receipt carries.

Size, the action trace, the counterfactual and the fast weights. Each returns
the list of what failed rather than a verdict, for the same reason the contract
checks do: a caller that is told only "invalid" cannot say what to fix.
"""
from __future__ import annotations

import hashlib
from typing import Any


class _ChecksTheReceiptEvidence:
    """Lifted whole from LatentCortexService; see latent_cortex_service.py."""

    @staticmethod
    def _receipt_size_errors(receipt: Any) -> list[str]:
        """Refuse a receipt too large or too deep to validate safely.

        CP126 09f2fbcf: decision lists, score/loss/gradient trails, step
        sizes, compaction objects, runtime identity, progress and stage
        timings had no aggregate count or size budget before the facade
        copied and iterated them. A worker response could therefore spend the
        facade's memory and CPU — the validator was a denial-of-service
        surface for the process it was protecting.
        """
        # Imported here rather than at module level: the module these
        # came from imports this one to build the class. A call-time
        # import also still sees a test's patch of the original.
        from .latent_cortex_service import (
            _MAX_RECEIPT_DEPTH,
            _MAX_RECEIPT_ITEMS,
            _MAX_RECEIPT_KEYS,
        )

        if not isinstance(receipt, dict):
            return []
        errors: list[str] = []
        if len(receipt) > _MAX_RECEIPT_KEYS:
            errors.append("receipt_key_count_exceeds_budget")

        total_items = 0
        stack: list[tuple[Any, int]] = [(receipt, 0)]
        while stack:
            node, depth = stack.pop()
            if depth > _MAX_RECEIPT_DEPTH:
                errors.append("receipt_nesting_exceeds_budget")
                break
            if isinstance(node, dict):
                children = list(node.values())
            elif isinstance(node, (list, tuple)):
                children = list(node)
            else:
                continue
            total_items += len(children)
            if total_items > _MAX_RECEIPT_ITEMS:
                errors.append("receipt_size_exceeds_budget")
                break
            for child in children:
                if isinstance(child, (dict, list, tuple)):
                    stack.append((child, depth + 1))
        return errors

    @staticmethod
    def _receipt_action_trace_errors(answer_replacement_private, config, errors, expected_objective, output_text, output_tokens, raw_action_trace, receipt, resource_accounting):
        """Body lifted verbatim out of ``LatentCortexService._receipt_contract_errors``.

        Moved by tools/extract_seam.py, which refuses to write unless the
        relocated body diffs clean against the original. The seam was
        9 names in, 0 out, 0 early return(s), 0 awaits.
        """
        from .latent_cortex_service import (
            logger,
        )

        if isinstance(raw_action_trace, list) and raw_action_trace:
            validating_context_focus = False
            try:
                from core.brain.llm.latent_cortex.cognitive_operators import (
                    validate_operator_receipt,
                )

                neural_actions = {
                    "decompose",
                    "blind_resolve",
                    "branch",
                    "search_memory",
                    "retrieve_evidence",
                    "simulate",
                    "falsify",
                    "check_assumption",
                    "regenerate_from_prefix",
                    "formalize",
                }
                raw_operator_trace = receipt.get("cognitive_operator_trace")
                if not isinstance(raw_operator_trace, list) or not raw_operator_trace:
                    raise ValueError("cognitive operator trace is absent")
                operator_rows = [validate_operator_receipt(row) for row in raw_operator_trace]
                expected_operator_work: dict[str, dict[str, int]] = {}
                for row in operator_rows:
                    operation_name = f"cognitive_operator:{row['operator']}"
                    expected = expected_operator_work.setdefault(
                        operation_name,
                        {
                            "tensor_element_reads": 0,
                            "tensor_element_writes": 0,
                            "tensor_scalar_ops": 0,
                            "host_scalar_ops": 0,
                        },
                    )
                    accounting = row["tensor_accounting"]
                    expected["tensor_element_reads"] += accounting["element_reads"]
                    expected["tensor_element_writes"] += accounting["element_writes"]
                    expected["tensor_scalar_ops"] += accounting["tensor_scalar_ops"]
                    expected["host_scalar_ops"] += accounting["commitment_host_ops"]
                operations = (
                    resource_accounting.get("operations", {})
                    if resource_accounting is not None
                    else {}
                )
                observed_operator_names = {
                    name for name in operations if name.startswith("cognitive_operator:")
                }
                if observed_operator_names != set(expected_operator_work):
                    raise ValueError("cognitive operator resource coverage differs")
                for operation_name, expected in expected_operator_work.items():
                    operation = operations.get(operation_name)
                    if not isinstance(operation, dict) or any(
                        operation.get(name) != value for name, value in expected.items()
                    ):
                        raise ValueError("cognitive operator resource totals differ")
                    if any(operation.get(name) != 0 for name in operation if name not in expected):
                        raise ValueError("cognitive operator resource kind differs")
                by_step: dict[int, list[dict[str, Any]]] = {}
                for row in operator_rows:
                    by_step.setdefault(row["action_step"], []).append(row)
                action_rows_by_step: dict[int, dict[str, Any]] = {}
                for action_row in raw_action_trace:
                    if not isinstance(action_row, dict):
                        raise ValueError("cognitive action trace row is invalid")
                    transition = action_row.get("transition")
                    signal = action_row.get("state_signal")
                    step = transition.get("step_index") if isinstance(transition, dict) else None
                    if (
                        type(step) is not int
                        or step < 0
                        or step in action_rows_by_step
                        or not isinstance(signal, dict)
                    ):
                        raise ValueError("cognitive action trace row is incomplete")
                    action_rows_by_step[step] = action_row
                if set(by_step) - set(action_rows_by_step):
                    raise ValueError("cognitive operator step is orphaned")
                for step, action_row in sorted(action_rows_by_step.items()):
                    transition = action_row["transition"]
                    signal = action_row["state_signal"]
                    action = transition.get("action")
                    rows = by_step.get(step, [])
                    if action not in neural_actions:
                        if rows:
                            raise ValueError("structural action claimed neural operators")
                        continue
                    active_branches = signal.get("active_branches")
                    if (
                        type(active_branches) is not int
                        or active_branches <= 0
                        or len(rows) != active_branches
                        or {row["action"] for row in rows} != {action}
                        or {row["action_step"] for row in rows} != {step}
                        or len({row["branch_index"] for row in rows}) != len(rows)
                        or len({row["operator"] for row in rows}) != len(rows)
                    ):
                        raise ValueError("cognitive operator coverage is invalid")
                from core.brain.llm.latent_cortex.context_focus import (
                    CONTEXT_FOCUS_ACTIONS,
                    validate_context_focus_receipt,
                )
                from core.brain.llm.latent_cortex.epistemic_state import (
                    OperationKind,
                )

                validating_context_focus = True
                raw_focus_trace = receipt.get("context_focus_trace")
                if not isinstance(raw_focus_trace, list):
                    raise ValueError("context focus trace is invalid")
                cognitive_slots = receipt.get("cognitive_slots")
                if not isinstance(cognitive_slots, list):
                    raise ValueError("cognitive slot inventory is invalid")
                focus_rows = [
                    validate_context_focus_receipt(
                        row,
                        cognitive_slots=cognitive_slots,
                    )
                    for row in raw_focus_trace
                ]
                expected_focus_work: dict[str, dict[str, int]] = {}
                for row in focus_rows:
                    operation_name = f"context_focus:{row['action']}"
                    expected = expected_focus_work.setdefault(
                        operation_name,
                        {
                            "tensor_element_reads": 0,
                            "tensor_element_writes": 0,
                            "tensor_scalar_ops": 0,
                            "host_scalar_ops": 0,
                        },
                    )
                    accounting = row["tensor_accounting"]
                    expected["tensor_element_reads"] += accounting["element_reads"]
                    expected["tensor_element_writes"] += accounting["element_writes"]
                    expected["tensor_scalar_ops"] += accounting["tensor_scalar_ops"]
                    expected["host_scalar_ops"] += accounting["commitment_host_ops"]
                observed_focus_names = {
                    name for name in operations if name.startswith("context_focus:")
                }
                if observed_focus_names != set(expected_focus_work):
                    raise ValueError("context focus resource coverage differs")
                for operation_name, expected in expected_focus_work.items():
                    operation = operations.get(operation_name)
                    if not isinstance(operation, dict) or any(
                        operation.get(name) != amount
                        for name, amount in expected.items()
                    ):
                        raise ValueError("context focus resource totals differ")
                    if any(
                        operation.get(name) != 0
                        for name in operation
                        if name not in expected
                    ):
                        raise ValueError("context focus resource kind differs")
                focus_by_step: dict[int, list[dict[str, Any]]] = {}
                for row in focus_rows:
                    focus_by_step.setdefault(row["action_step"], []).append(row)
                if set(focus_by_step) - set(action_rows_by_step):
                    raise ValueError("context focus step is orphaned")
                operator_by_step_branch = {
                    (row["action_step"], row["branch_index"]): row
                    for row in operator_rows
                }
                for step, action_row in sorted(action_rows_by_step.items()):
                    transition = action_row["transition"]
                    signal = action_row["state_signal"]
                    action = OperationKind(transition["action"])
                    rows = focus_by_step.get(step, [])
                    if action not in CONTEXT_FOCUS_ACTIONS:
                        if rows:
                            raise ValueError(
                                "non-context action claimed context focus"
                            )
                        continue
                    active_branches = signal["active_branches"]
                    if (
                        len(rows) != active_branches
                        or {row["action"] for row in rows} != {action.value}
                        or len({row["branch_index"] for row in rows}) != len(rows)
                    ):
                        raise ValueError("context focus coverage is invalid")
                    for row in rows:
                        operator_row = operator_by_step_branch.get(
                            (step, row["branch_index"])
                        )
                        if (
                            operator_row is None
                            or operator_row["action"] != action.value
                            or operator_row["input_sha256"]
                            != row["output_sha256"]
                        ):
                            raise ValueError(
                                "context focus did not feed the cognitive operator"
                            )
            except (ImportError, TypeError, ValueError):
                errors.append(
                    "context_focus_execution_unproven"
                    if validating_context_focus
                    else "cognitive_operator_execution_unproven"
                )
            try:
                from core.brain.llm.latent_cortex.structural_diversity import (
                    validate_structural_diversity_receipt,
                )

                validate_structural_diversity_receipt(
                    receipt.get("structural_diversity"),
                    n_branches=int(receipt.get("n_branches")),
                    cognitive_slots=receipt.get("cognitive_slots"),
                    operator_trace=receipt.get("cognitive_operator_trace"),
                    action_trace=raw_action_trace,
                    branch_isolation=receipt.get("branch_isolation"),
                )
            except (ImportError, TypeError, ValueError):
                errors.append("structural_diversity_unproven")
            try:
                from core.brain.llm.latent_cortex.disagreement_graph import (
                    validate_disagreement_graph_receipt,
                )

                validate_disagreement_graph_receipt(
                    receipt.get("disagreement_graph"),
                    n_branches=int(receipt.get("n_branches")),
                    operator_trace=receipt.get("cognitive_operator_trace"),
                    action_trace=raw_action_trace,
                    structural_diversity=receipt.get("structural_diversity"),
                    blind_review=receipt.get("blind_review"),
                )
            except (ImportError, TypeError, ValueError):
                errors.append("disagreement_graph_unproven")
            try:
                from core.brain.llm.latent_cortex.diagnostic_action_selector import (
                    validate_diagnostic_action_selector_receipt,
                )

                validate_diagnostic_action_selector_receipt(
                    receipt.get("diagnostic_action_selection"),
                    disagreement_graph=receipt.get("disagreement_graph"),
                    value_policy=receipt.get("value_of_computation"),
                    action_trace=raw_action_trace,
                )
            except (ImportError, TypeError, ValueError):
                errors.append("diagnostic_action_selection_unproven")
            try:
                from core.brain.llm.latent_cortex.local_repair import (
                    validate_local_repair_receipt,
                )

                validate_local_repair_receipt(
                    receipt.get("local_repair"),
                    disagreement_graph=receipt.get("disagreement_graph"),
                    diagnostic_selection=receipt.get(
                        "diagnostic_action_selection"
                    ),
                )
            except (ImportError, KeyError, TypeError, ValueError):
                errors.append("local_repair_unproven")
            try:
                from core.brain.llm.latent_cortex.answer_replacement import (
                    validate_answer_replacement_receipt,
                )
                from core.brain.llm.latent_cortex.worker_handler import (
                    config_from_job,
                )

                executed_config = config_from_job(config)
                from core.brain.llm.latent_cortex.answer_replacement import (
                    MAX_REPLACEMENT_OUTPUT_TOKENS,
                )

                replacement_output_limit = min(
                    MAX_REPLACEMENT_OUTPUT_TOKENS,
                    int(executed_config.decode_max_tokens)
                    + (
                        int(executed_config.decode_contract_grace_tokens)
                        if executed_config.decode_contract == "final_answer_v1"
                        else 48
                    ),
                )
                incumbent_floor_declined_abstention = bool(
                    executed_config.decode_incumbent_policy == "vanilla_incumbent"
                    and isinstance(receipt.get("answer_replacement"), dict)
                    and receipt["answer_replacement"].get("decision") == "abstain"
                    and "confidence_bound_abstention_declined_under_incumbent"
                    in (receipt.get("warnings") or [])
                )
                validate_answer_replacement_receipt(
                    receipt.get("answer_replacement"),
                    disagreement_graph=receipt.get("disagreement_graph"),
                    diagnostic_selection=receipt.get(
                        "diagnostic_action_selection"
                    ),
                    local_repair=receipt.get("local_repair"),
                    private_evidence=answer_replacement_private,
                    expected_objective=expected_objective,
                    expected_selected_branch=int(receipt.get("selected_branch")),
                    # Must match the engine exactly. The engine no longer
                    # gates promotion on decode_incumbent_policy -- that
                    # coupling made the floor and the gain mutually exclusive
                    # -- so a validator still expecting it would reject every
                    # receipt the engine now produces as unproven.
                    expected_enabled=executed_config.answer_replacement_enabled,
                    expected_objective_program_enabled=(
                        executed_config.objective_program_enabled
                    ),
                    expected_margin=executed_config.answer_replacement_margin,
                    expected_max_output_tokens=replacement_output_limit,
                    expected_output_text=(
                        None
                        if incumbent_floor_declined_abstention
                        else output_text if isinstance(output_text, str) else None
                    ),
                    expected_output_tokens=(
                        None
                        if incumbent_floor_declined_abstention
                        else output_tokens if isinstance(output_tokens, list) else None
                    ),
                )
                if incumbent_floor_declined_abstention:
                    private_baseline = (
                        answer_replacement_private
                        if isinstance(answer_replacement_private, dict)
                        else {}
                    )
                    if (
                        private_baseline.get("baseline_text") != output_text
                        or private_baseline.get("baseline_tokens") != output_tokens
                    ):
                        raise ValueError(
                            "incumbent floor output differs from the bound baseline"
                        )
            except (ImportError, KeyError, TypeError, ValueError) as exc:
                logger.warning(
                    "Answer replacement contract validation failed: %s",
                    str(exc)[:400],
                )
                errors.append("answer_replacement_unproven")
            # Hidden benchmark answers are deliberately unavailable in the
            # serving trust domain. A research-oracle receipt is valid only in
            # the frozen reconciliation harness and must never cross into a
            # live worker response as output authority.
            if receipt.get("research_oracle_arbitration"):
                errors.append("research_oracle_output_forbidden_in_service")
            try:
                from core.brain.llm.latent_cortex.correlated_support import (
                    validate_correlated_support_receipt,
                )

                validate_correlated_support_receipt(
                    receipt.get("correlated_support"),
                    structural_diversity=receipt.get("structural_diversity"),
                    correlation_evidence=config.get("branch_correlation_evidence"),
                )
            except (ImportError, TypeError, ValueError):
                errors.append("correlated_support_unproven")
        elif resource_accounting is not None and any(
            name.startswith(("cognitive_operator:", "context_focus:"))
            for name in resource_accounting.get("operations", {})
        ):
            errors.append("cognitive_operator_execution_unproven")

    @staticmethod
    def _receipt_counterfactual_errors(config, errors, receipt, verified_counterfactual, verified_generation):
        """Body lifted verbatim out of ``LatentCortexService._receipt_contract_errors``.

        Moved by tools/extract_seam.py, which refuses to write unless the
        relocated body diffs clean against the original. The seam was
        5 names in, 0 out, 0 early return(s), 0 awaits.
        """
        if any(
            receipt.get(field)
            for field in (
                "branch_contract",
                "contract_repair",
                "post_adaptation_candidate",
                "blind_review",
                "decoy_verification",
            )
        ):
            contract_repair_expected = bool(receipt.get("contract_repair")) or (
                config.get("decode_contract") == "final_answer_v1"
                or config.get("verifier_probe_contract") == "final_answer_v1"
            )
            if contract_repair_expected:
                try:
                    from core.brain.llm.latent_cortex.contract_repair import (
                        validate_contract_repair_receipt,
                    )
                    from core.brain.llm.latent_cortex.worker_handler import (
                        config_from_job,
                    )

                    executed_config = config_from_job(config)
                    contract_repair = validate_contract_repair_receipt(
                        receipt.get("contract_repair")
                    )
                    expected_requests = (
                        executed_config.local_repair_max_attempts
                        if executed_config.local_repair_enabled
                        else 0
                    )
                    if (
                        contract_repair["max_requests"] != expected_requests
                        or contract_repair["max_tokens"]
                        != executed_config.local_repair_max_tokens
                    ):
                        raise ValueError("contract repair policy differs")
                except (ImportError, TypeError, ValueError):
                    errors.append("contract_repair_unproven")
            if receipt.get("post_adaptation_candidate"):
                try:
                    from core.brain.llm.latent_cortex.post_adaptation_candidate import (
                        validate_post_adaptation_candidate_receipt,
                    )

                    post_adaptation = validate_post_adaptation_candidate_receipt(
                        receipt.get("post_adaptation_candidate")
                    )
                    if post_adaptation["selected_branch"] != receipt.get(
                        "selected_branch"
                    ):
                        raise ValueError(
                            "post-adaptation candidate branch differs"
                        )
                except (ImportError, TypeError, ValueError):
                    errors.append("post_adaptation_candidate_unproven")
            try:
                from core.brain.llm.latent_cortex.blind_review import (
                    validate_blind_review_receipt,
                    validate_decoy_review_receipt,
                )

                decoy = validate_decoy_review_receipt(
                    receipt.get("decoy_verification"),
                    blind_receipt=receipt.get("blind_review"),
                    episode_id=receipt.get("episode_id"),
                    objective_sha256=receipt.get("input_tokens_sha256"),
                )
                review_selected_branch = receipt.get("selected_branch")
                if isinstance(verified_counterfactual, dict):
                    review_selected_branch = verified_counterfactual[
                        "source_selected_branch"
                    ]
                elif (
                    isinstance(verified_generation, dict)
                    and verified_generation.get("selection_effect")
                    == "winner_replaced"
                ):
                    review_selected_branch = verified_generation["vetoed_branch"]
                validate_blind_review_receipt(
                    receipt.get("blind_review"),
                    n_branches=int(receipt.get("n_branches")),
                    branch_scores=receipt.get("branch_scores"),
                    isolation_receipt=receipt.get("branch_isolation"),
                    objective_sha256=receipt.get("input_tokens_sha256"),
                    episode_id=receipt.get("episode_id"),
                    selected_branch=review_selected_branch,
                    decoy_receipt=receipt.get("decoy_verification"),
                )
                honest_flags = receipt.get("honest_flags")
                if not isinstance(honest_flags, list):
                    raise ValueError("decoy-review honest flags are invalid")
                if (
                    decoy["selection_admitted"] is False
                    and "branch_verifier_decoy_calibration_failed" not in honest_flags
                ):
                    raise ValueError("decoy selection rejection was not disclosed")
            except (ImportError, TypeError, ValueError):
                errors.append("blind_or_decoy_branch_review_unproven")

    @staticmethod
    def _receipt_fast_weight_errors(config, errors, expected_worker_identity, finite_number_list, nonnegative_int, output_text, output_tokens, positive_int, receipt, resource_accounting):
        """Body lifted verbatim out of ``LatentCortexService._receipt_contract_errors``.

        Moved by tools/extract_seam.py, which refuses to write unless the
        relocated body diffs clean against the original. The seam was
        10 names in, 0 out, 0 early return(s), 0 awaits.
        """
        from .latent_cortex_service import (
            _integrity_verdict,
        )

        if config.get("fast_weights") is True:
            learning: dict[str, Any] | None = None
            try:
                from core.brain.llm.latent_cortex.fast_weight_learning import (
                    token_sequence_sha256,
                    validate_fast_weight_learning_receipt,
                )

                learning = validate_fast_weight_learning_receipt(
                    receipt.get("fast_weight_learning"),
                    expected_episode_id=str(receipt.get("episode_id") or ""),
                    expected_input_tokens_sha256=str(
                        receipt.get("input_tokens_sha256") or ""
                    ),
                )
                pseudo_label = learning["admission"].get(
                    "pseudo_label_admission"
                )
                structural_diversity = receipt.get(
                    "structural_diversity"
                )
                if (
                    isinstance(pseudo_label, dict)
                    and pseudo_label
                    and (
                        not isinstance(structural_diversity, dict)
                        or pseudo_label.get(
                            "structural_diversity_sha256"
                        )
                        != structural_diversity.get("receipt_sha256")
                        or pseudo_label.get(
                            "structural_diversity_certified"
                        )
                        is not (
                            structural_diversity.get("certified") is True
                        )
                    )
                ):
                    raise ValueError(
                        "fast-weight pseudo-label structural binding differs"
                    )
                test_time_training = learning["controls"].get(
                    "test_time_training"
                )
                matched_compute = (
                    test_time_training.get("matched_compute")
                    if isinstance(test_time_training, dict)
                    else None
                )
                if isinstance(matched_compute, dict) and matched_compute:
                    operations = (
                        resource_accounting.get("operations", {})
                        if isinstance(resource_accounting, dict)
                        else {}
                    )
                    arm_resource_valid = True
                    for arm_name in ("treatment", "sham"):
                        arm = matched_compute.get(arm_name)
                        if not isinstance(arm, dict):
                            arm_resource_valid = False
                            break
                        gradients = arm.get("backward_evaluations")
                        line_searches = arm.get(
                            "line_search_evaluations"
                        )
                        layer_apps = arm.get("layer_apps")
                        denominator = (
                            3 * gradients + line_searches
                            if type(gradients) is int
                            and type(line_searches) is int
                            else 0
                        )
                        if (
                            denominator <= 0
                            or type(layer_apps) is not int
                            or layer_apps <= 0
                            or layer_apps % denominator
                        ):
                            arm_resource_valid = False
                            break
                        forward_layer_apps = layer_apps // denominator
                        gradient_operation = operations.get(
                            f"fast_weight_{arm_name}_gradient"
                        )
                        line_operation = operations.get(
                            f"fast_weight_{arm_name}_line_search"
                        )
                        if (
                            not isinstance(gradient_operation, dict)
                            or not isinstance(line_operation, dict)
                            or gradient_operation.get(
                                "transformer_layer_apps"
                            )
                            != gradients * forward_layer_apps
                            or line_operation.get(
                                "transformer_layer_apps"
                            )
                            != line_searches * forward_layer_apps
                        ):
                            arm_resource_valid = False
                            break
                    if not arm_resource_valid:
                        errors.append(
                            "fast_weight_matched_compute_resource_unproven"
                        )
                if not isinstance(output_tokens, list) or not isinstance(
                    output_text,
                    str,
                ):
                    raise ValueError("fast-weight output binding is unavailable")
                final_binding = learning["final_answer"]
                if (
                    final_binding["tokens_sha256"]
                    != token_sequence_sha256(output_tokens)
                    or final_binding["text_sha256"]
                    != hashlib.sha256(output_text.encode("utf-8")).hexdigest()
                    or final_binding["token_count"] != len(output_tokens)
                ):
                    raise ValueError("fast-weight final answer binding differs")
            except (ImportError, TypeError, ValueError):
                errors.append("fast_weight_learning_receipt_unproven")
            # Preserve low-level diagnostics even when the unified envelope
            # is missing or malformed. A bad outer proof must not hide the
            # concrete optimizer defect that made it bad.
            if learning is None and receipt.get("fast_weights_applied") is True:
                loss_trail = receipt.get("fast_weight_loss_trail")
                gradient_trail = receipt.get(
                    "fast_weight_gradient_norm_trail"
                )
                step_sizes = receipt.get("fast_weight_accepted_step_sizes")
                accepted_steps = int(
                    receipt.get("fast_weight_optimized_steps") or 0
                )
                if accepted_steps <= 0:
                    errors.append(
                        "fast_weight_optimization_no_accepted_steps"
                    )
                if receipt.get("fast_weight_optimizer") != (
                    "rms_normalized_sgd_backtracking_v1"
                ):
                    errors.append("fast_weight_optimizer_unproven")
                if (
                    not finite_number_list(loss_trail)
                    or len(loss_trail) != accepted_steps + 1
                    or any(
                        later >= earlier
                        for earlier, later in zip(
                            loss_trail,
                            loss_trail[1:],
                            strict=False,
                        )
                    )
                ):
                    errors.append("fast_weight_loss_descent_unproven")
                if (
                    not finite_number_list(gradient_trail)
                    or len(gradient_trail)
                    != receipt.get("fast_weight_optimization_attempts", 0)
                    or any(float(value) <= 0.0 for value in gradient_trail)
                ):
                    errors.append("fast_weight_gradient_evidence_invalid")
                if (
                    not finite_number_list(step_sizes)
                    or len(step_sizes) != accepted_steps
                    or any(float(value) <= 0.0 for value in step_sizes)
                ):
                    errors.append("fast_weight_step_evidence_invalid")
                if not nonnegative_int(
                    receipt,
                    "fast_weight_line_search_backtracks",
                ):
                    errors.append("fast_weight_line_search_evidence_invalid")
            if learning is not None:
                disposition = learning["disposition"]
                not_admitted = (
                    disposition
                    == "not_admitted_high_confidence_evidence_absent"
                )
                if not_admitted:
                    # An ineligible episode must have left the weights alone.
                    # Absence and denial are DIFFERENT failures: a receipt that
                    # never mentions the attach attempt has not proved the
                    # model was untouched, but it has not confessed to
                    # touching it either, and one error name for both sends
                    # whoever reads it looking for a mutation that never
                    # happened.
                    mutation_fields = (
                        ("fast_weights_applied", False),
                        ("fast_weights_attach_attempted", False),
                        ("fast_weights_layers", 0),
                        ("fast_weight_optimization_attempts", 0),
                        ("fast_weight_optimized_steps", 0),
                        ("fast_weight_rejected_steps", 0),
                    )
                    absent = [
                        field for field, _ in mutation_fields if field not in receipt
                    ]
                    if absent:
                        errors.append(
                            "fast_weight_ineligible_episode_receipt_incomplete:"
                            + ",".join(sorted(absent))
                        )
                    if any(
                        field in receipt and receipt[field] != quiet
                        for field, quiet in mutation_fields
                    ):
                        errors.append("fast_weight_ineligible_episode_mutated_model")
                else:
                    if receipt.get("fast_weights_applied") is not True:
                        errors.append("fast_weights_not_applied")
                    erase_verdict = _integrity_verdict(
                        receipt,
                        "fast_weights_erased",
                        expected_worker_identity=expected_worker_identity,
                    )
                    if erase_verdict == "refuted":
                        errors.append("fast_weight_erase_refuted")
                    elif erase_verdict != "proven":
                        errors.append("fast_weight_erase_unproven")
                    if not positive_int(receipt, "fast_weights_layers"):
                        errors.append("fast_weights_no_layers")
                    if not positive_int(
                        receipt,
                        "fast_weight_optimization_attempts",
                    ):
                        errors.append("fast_weight_optimization_not_attempted")
                    if not nonnegative_int(
                        receipt,
                        "fast_weight_optimized_steps",
                    ):
                        errors.append("fast_weight_accepted_count_invalid")
                    if not nonnegative_int(
                        receipt,
                        "fast_weight_rejected_steps",
                    ):
                        errors.append("fast_weight_rejection_count_invalid")
                    elif (
                        positive_int(
                            receipt,
                            "fast_weight_optimization_attempts",
                        )
                        and nonnegative_int(
                            receipt,
                            "fast_weight_optimized_steps",
                        )
                        and receipt["fast_weight_optimization_attempts"]
                        != receipt["fast_weight_optimized_steps"]
                        + receipt["fast_weight_rejected_steps"]
                    ):
                        errors.append(
                            "fast_weight_optimization_accounting_mismatch"
                        )
                    if receipt.get("fast_weight_budget_exhausted") is not False:
                        errors.append(
                            "fast_weight_optimization_budget_exhausted"
                        )
                    loss_trail = receipt.get("fast_weight_loss_trail")
                    gradient_trail = receipt.get(
                        "fast_weight_gradient_norm_trail"
                    )
                    step_sizes = receipt.get(
                        "fast_weight_accepted_step_sizes"
                    )
                    if receipt.get("fast_weight_optimizer") != (
                        "rms_normalized_sgd_backtracking_v1"
                    ):
                        errors.append("fast_weight_optimizer_unproven")
                    accepted_steps = int(
                        receipt.get("fast_weight_optimized_steps") or 0
                    )
                    if (
                        not finite_number_list(loss_trail)
                        or (
                            accepted_steps > 0
                            and len(loss_trail) != accepted_steps + 1
                        )
                        or any(
                            later >= earlier
                            for earlier, later in zip(
                                loss_trail,
                                loss_trail[1:],
                                strict=False,
                            )
                        )
                    ):
                        errors.append("fast_weight_loss_descent_unproven")
                    if (
                        not finite_number_list(gradient_trail)
                        or len(gradient_trail)
                        != receipt.get("fast_weight_optimization_attempts", 0)
                        or any(float(value) <= 0.0 for value in gradient_trail)
                    ):
                        errors.append("fast_weight_gradient_evidence_invalid")
                    if (
                        not finite_number_list(step_sizes)
                        or len(step_sizes) != accepted_steps
                        or any(float(value) <= 0.0 for value in step_sizes)
                    ):
                        errors.append("fast_weight_step_evidence_invalid")
                    if not nonnegative_int(
                        receipt,
                        "fast_weight_line_search_backtracks",
                    ):
                        errors.append(
                            "fast_weight_line_search_evidence_invalid"
                        )
                    if (
                        disposition == "accepted_causal_improvement"
                        and accepted_steps <= 0
                    ):
                        errors.append(
                            "fast_weight_causal_improvement_without_step"
                        )
