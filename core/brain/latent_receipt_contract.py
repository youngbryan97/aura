"""What is wrong with the contract a latent receipt claims to meet.

A receipt that is merely "invalid" tells a caller nothing it can act on, so
every check here returns the list of fields that failed rather than a verdict.
This is the contract itself; the trace, counterfactual and fast-weight checks
are in latent_receipt_evidence.py.
"""
from __future__ import annotations

import hashlib
import math
from typing import Any

from core.brain.llm.latent_cortex.branch_exchange import eligible_exchange_steps


class _ChecksTheReceiptContract:
    """Lifted whole from LatentCortexService; see latent_cortex_service.py."""

    def _safe_receipt_contract_errors(self, *args: Any, **kwargs: Any) -> list[str]:
        """Run the contract validator without letting it raise.

        Every failure mode of a malformed receipt has to arrive as a contract
        error, because that is what the caller was promised (CP126 94593618).
        """
        # Imported here rather than at module level: the module these
        # came from imports this one to build the class. A call-time
        # import also still sees a test's patch of the original.
        from .latent_cortex_service import (
            logger,
        )

        # Size first: an oversized receipt must be refused BEFORE the
        # validator walks it, or the bound protects nothing.
        size_errors = self._receipt_size_errors(args[0] if args else None)
        if size_errors:
            return size_errors
        try:
            return self._receipt_contract_errors(*args, **kwargs)
        except (
            ArithmeticError,
            AttributeError,
            IndexError,
            KeyError,
            LookupError,
            RecursionError,
            TypeError,
            ValueError,
        ) as exc:
            logger.warning("Receipt contract validation raised: %s", exc)
            return [f"receipt_contract_validator_error:{type(exc).__name__}"]

    @staticmethod
    def _receipt_contract_errors(
        receipt: Any,
        config: dict[str, Any],
        runtime_controls: dict[str, Any] | None = None,
        expected_worker_identity: dict[str, Any] | None = None,
        output_tokens: Any = ...,
        expected_domain: str = "general",
        output_text: Any = ...,
        answer_replacement_private: Any = None,
        expected_objective: str = "",
        expected_request_payload_sha256: str = "",
        allocated_budget: dict[str, Any] | None = None,
    ) -> list[str]:
        from .latent_cortex_service import (
            LatentCortexService,
            _check_the_exchange_count_contract,
            _check_the_latent_optimiser_contract,
            _integrity_verdict,
            _recurrence_halt_reason,
        )

        if not isinstance(receipt, dict):
            return ["receipt_not_mapping"]
        errors: list[str] = []

        # LIVE DEFECT, 2026-08-03. Three proofs — terminal_disposition,
        # answer_replacement and fast_weight_learning — each bind the receipt
        # to the answer's TOKENS, and each raises without them. When the
        # transport dropped the token list, all three failed together and the
        # receipt said "terminal_disposition_unproven,answer_replacement_
        # unproven,fast_weight_learning_receipt_unproven" on every single turn:
        # three symptoms of one missing input, none of them naming it. Say the
        # actual condition once, so the next person reads the cause instead of
        # three consequences.
        if output_tokens is not ... and not isinstance(output_tokens, list):
            errors.append("output_tokens_unavailable")

        def positive_int(mapping: dict[str, Any], key: str) -> bool:
            return type(mapping.get(key)) is int and mapping[key] > 0

        def nonnegative_int(mapping: dict[str, Any], key: str) -> bool:
            return type(mapping.get(key)) is int and mapping[key] >= 0

        def finite_number_list(value: Any) -> bool:
            return isinstance(value, list) and all(
                not isinstance(item, bool)
                and isinstance(item, (int, float))
                and math.isfinite(float(item))
                for item in value
            )

        def finite_number(value: Any) -> bool:
            return (
                not isinstance(value, bool)
                and isinstance(value, (int, float))
                and math.isfinite(float(value))
            )

        def verifier_arbitration_valid(
            arbitration: Any,
            *,
            attempts: int,
            accepted_steps: int,
        ) -> bool:
            """Independently replay one non-regression arbitration receipt."""
            if not isinstance(arbitration, dict):
                return False
            score_accepts = arbitration.get("score_improvement_accepts")
            proxy_accepts = arbitration.get("proxy_nonregression_accepts")
            plateau_accepts = arbitration.get("plateau_exploration_accepts")
            plateau_rollbacks = arbitration.get("plateau_rollbacks")
            strict_committed = arbitration.get("strict_improvement_committed")
            score_source = arbitration.get("score_source")
            commit_policy = arbitration.get("commit_policy")
            decisions = arbitration.get("decisions")
            score_trail = arbitration.get("score_trail")
            tolerance = arbitration.get("score_tolerance")
            proxy_scale = arbitration.get("proxy_tolerance_scale")
            if (
                arbitration.get("policy") != "task_score_nonregression_with_proxy_descent_v1"
                or arbitration.get("baseline_source")
                not in {"caller_reused_verified_branch", "decoded_state_probe"}
                or type(score_accepts) is not int
                or score_accepts < 0
                or type(proxy_accepts) is not int
                or proxy_accepts < 0
                or type(plateau_accepts) is not int
                or plateau_accepts < 0
                or type(plateau_rollbacks) is not int
                or not 0 <= plateau_rollbacks <= plateau_accepts
                or type(strict_committed) is not bool
                or score_source
                not in {
                    "unspecified",
                    "semantic_candidate_score_for_latent_search_only_v1",
                }
                or commit_policy
                not in {
                    "immediate",
                    "strict_task_improvement_after_plateau_search_v1",
                }
                or score_accepts + proxy_accepts != accepted_steps
                or not finite_number(tolerance)
                or not 0.0 <= float(tolerance) <= 1e-3
                or not finite_number(proxy_scale)
                or not 0.0 < float(proxy_scale) <= 1e-3
                or not isinstance(decisions, list)
                or len(decisions) != attempts
                or not finite_number_list(score_trail)
                or len(score_trail) != len(decisions) + 1
            ):
                return False

            score_tolerance = float(tolerance)
            proxy_tolerance_scale = float(proxy_scale)
            receipt_epsilon = 2e-12
            observed_score_accepts = 0
            observed_proxy_accepts = 0
            observed_plateau_accepts = 0
            observed_plateau_rollbacks = 0
            allowed_decisions = {
                "accepted_task_score_improvement",
                "accepted_task_score_nonregression_with_proxy_descent",
                "rejected_task_score_regression",
                "rejected_proxy_non_descent",
                "rejected_nonfinite_task_score",
                "rejected_nonfinite_proxy_loss",
            }
            for index, row in enumerate(decisions):
                if not isinstance(row, dict) or row.get("proposal") != index:
                    return False
                decision = row.get("decision")
                if decision not in allowed_decisions:
                    return False
                baseline = row.get("baseline_score")
                current_proxy = row.get("current_proxy_loss")
                required_delta = row.get("proxy_required_delta")
                if (
                    not finite_number(baseline)
                    or not finite_number(current_proxy)
                    or not finite_number(required_delta)
                    or float(required_delta) < 0.0
                    or not math.isclose(
                        float(baseline),
                        float(score_trail[index]),
                        rel_tol=0.0,
                        abs_tol=receipt_epsilon,
                    )
                    or not math.isclose(
                        float(required_delta),
                        proxy_tolerance_scale * max(1.0, abs(float(current_proxy))),
                        rel_tol=1e-6,
                        abs_tol=receipt_epsilon,
                    )
                ):
                    return False
                baseline_score = float(baseline)
                next_score = float(score_trail[index + 1])
                raw_candidate = row.get("candidate_score")
                candidate_proxy = row.get("candidate_proxy_loss")
                if decision == "rejected_nonfinite_task_score":
                    if raw_candidate != "nonfinite" or not math.isclose(
                        next_score,
                        baseline_score,
                        rel_tol=0.0,
                        abs_tol=receipt_epsilon,
                    ):
                        return False
                    continue
                if not finite_number(raw_candidate):
                    return False
                candidate_score = float(raw_candidate)
                score_improved = (
                    candidate_score > baseline_score + score_tolerance + receipt_epsilon
                )
                score_nonregressing = (
                    candidate_score >= baseline_score - score_tolerance - receipt_epsilon
                )
                proxy_finite = finite_number(candidate_proxy)
                proxy_improved = bool(
                    proxy_finite
                    and float(candidate_proxy)
                    < float(current_proxy) - float(required_delta) - receipt_epsilon
                )
                if decision == "rejected_nonfinite_proxy_loss":
                    if proxy_finite or not math.isclose(
                        next_score,
                        baseline_score,
                        rel_tol=0.0,
                        abs_tol=receipt_epsilon,
                    ):
                        return False
                elif decision == "accepted_task_score_improvement":
                    if (
                        not proxy_finite
                        or not score_improved
                        or not math.isclose(
                            next_score,
                            candidate_score,
                            rel_tol=0.0,
                            abs_tol=receipt_epsilon,
                        )
                    ):
                        return False
                    if commit_policy == "strict_task_improvement_after_plateau_search_v1" and row.get(
                        "commit_disposition"
                    ) != "committed_to_best_strict_improvement":
                        return False
                    observed_score_accepts += 1
                elif decision == ("accepted_task_score_nonregression_with_proxy_descent"):
                    if (
                        score_improved
                        or not score_nonregressing
                        or not proxy_improved
                        or not math.isclose(
                            next_score,
                            max(baseline_score, candidate_score),
                            rel_tol=0.0,
                            abs_tol=receipt_epsilon,
                        )
                    ):
                        return False
                    observed_plateau_accepts += 1
                    disposition = row.get("commit_disposition")
                    if commit_policy == "strict_task_improvement_after_plateau_search_v1":
                        if disposition == "committed_to_best_strict_improvement":
                            observed_proxy_accepts += 1
                        elif disposition == "rolled_back_plateau_without_later_task_gain":
                            observed_plateau_rollbacks += 1
                        else:
                            return False
                    elif disposition is not None:
                        return False
                    else:
                        observed_proxy_accepts += 1
                elif decision == "rejected_task_score_regression":
                    if score_nonregressing or not math.isclose(
                        next_score,
                        baseline_score,
                        rel_tol=0.0,
                        abs_tol=receipt_epsilon,
                    ):
                        return False
                elif decision == "rejected_proxy_non_descent":
                    if (
                        not score_nonregressing
                        or score_improved
                        or not proxy_finite
                        or proxy_improved
                        or not math.isclose(
                            next_score,
                            baseline_score,
                            rel_tol=0.0,
                            abs_tol=receipt_epsilon,
                        )
                    ):
                        return False
            return (
                observed_score_accepts == score_accepts
                and observed_proxy_accepts == proxy_accepts
                and observed_plateau_accepts == plateau_accepts
                and observed_plateau_rollbacks == plateau_rollbacks
                and strict_committed == bool(score_accepts)
                and (
                    commit_policy != "strict_task_improvement_after_plateau_search_v1"
                    or score_source
                    == "semantic_candidate_score_for_latent_search_only_v1"
                )
            )

        def sha256(value: Any) -> bool:
            return (
                isinstance(value, str)
                and len(value) == 64
                and all(character in "0123456789abcdef" for character in value)
            )

        def git_oid(value: Any) -> bool:
            return (
                isinstance(value, str)
                and len(value) in {40, 64}
                and all(character in "0123456789abcdef" for character in value)
            )

        resource_accounting: dict[str, Any] | None = None
        information_accounting: dict[str, Any] | None = None
        budget_receipt = receipt.get("budget")
        try:
            from core.brain.llm.latent_cortex.resource_accounting import (
                validate_information_receipt,
                validate_resource_receipt,
            )

            if not isinstance(budget_receipt, dict):
                raise ValueError("episode budget receipt is absent")
            resource_accounting = validate_resource_receipt(
                budget_receipt.get("resource_accounting")
            )
            information_accounting = validate_information_receipt(
                budget_receipt.get("information_accounting")
            )
            if resource_accounting["accounting_complete"] is not True:
                errors.append("resource_accounting_incomplete")
            if information_accounting["accounting_complete"] is not True:
                errors.append("information_accounting_incomplete")
            n_layers = receipt.get("n_layers")
            if (
                type(n_layers) is int
                and n_layers > 0
                and resource_accounting["model_profile"]["num_hidden_layers"] != n_layers
            ):
                errors.append("resource_model_profile_mismatch")
        except (ImportError, TypeError, ValueError):
            errors.append("resource_accounting_unproven")
            errors.append("information_accounting_unproven")

        if not str(receipt.get("episode_id") or ""):
            errors.append("missing_episode_id")
        # Compatibility booleans and cached verdicts carry no authority here.
        # The worker-bound measured receipt is reconstructed independently.
        params_verdict = _integrity_verdict(
            receipt,
            "params_unchanged",
            expected_worker_identity=expected_worker_identity,
        )
        if params_verdict == "refuted":
            errors.append("checkpoint_invariant_refuted")
        elif params_verdict != "proven":
            errors.append("checkpoint_invariant_unproven")
        if (
            not sha256(receipt.get("checkpoint_fingerprint"))
            or receipt.get("checkpoint_fingerprint_method") != "sha256"
            or not positive_int(receipt, "checkpoint_file_count")
        ):
            errors.append("exact_checkpoint_identity_unproven")
        from core.brain.llm.latent_cortex.runtime_identity import worker_identity_errors

        worker_identity = receipt.get("worker_identity")
        errors.extend(
            worker_identity_errors(
                worker_identity,
                expected=(expected_worker_identity or None),
            )
        )
        guidance = receipt.get("verifier_guidance")
        if (
            isinstance(guidance, dict)
            and guidance.get("schema")
            in {"aura.latent_task_verifier.v3", "aura.latent_task_verifier.v4"}
            and type(guidance.get("evaluations")) is int
            and guidance["evaluations"] > 0
        ):
            try:
                from core.brain.llm.latent_cortex.atomic_decomposition import (
                    validate_atomic_decomposition_envelope,
                )

                atomic = validate_atomic_decomposition_envelope(
                    guidance.get("atomic_decomposition")
                )
                if (
                    atomic["grade_admissible"] is not True
                    or guidance.get("grade_admissible") is not True
                ):
                    raise ValueError("atomic decomposition denied grading authority")
                if guidance.get("schema") == "aura.latent_task_verifier.v4":
                    from core.brain.llm.latent_cortex.deterministic_verifier_router import (
                        validate_deterministic_router_envelope,
                    )

                    routed = validate_deterministic_router_envelope(
                        guidance.get("deterministic_router"),
                        atomic_receipt=atomic,
                    )
                    if routed["hard_pass"] is not True:
                        raise ValueError("deterministic verifier refuted candidate")
            except (ImportError, TypeError, ValueError):
                errors.append("atomic_decomposition_unproven")
        generative = receipt.get("generative_verifier")
        verified_generation: dict[str, Any] | None = None
        if config.get("generative_verifier_enabled") is True:
            if (
                isinstance(generative, dict)
                and generative.get("schema") == "aura.rlc.generative_verifier.v1"
            ):
                try:
                    from core.brain.llm.latent_cortex.generative_verifier import (
                        validate_generative_verifier_envelope,
                    )

                    verified_generation = validate_generative_verifier_envelope(generative)
                    effect = verified_generation["selection_effect"]
                    if effect == "winner_replaced" and receipt.get("selected_branch") != (
                        verified_generation["replacement_branch"]
                    ):
                        raise ValueError("generative verifier replacement was not selected")
                    if verified_generation["causal_refutation"] and effect == "none":
                        raise ValueError("generative refutation was not applied to selection")
                    if effect == "no_alternative":
                        raise ValueError("generative verifier refuted the only branch")
                except (ImportError, KeyError, TypeError, ValueError):
                    errors.append("generative_verifier_unproven")
                    verified_generation = None
            elif not (
                isinstance(generative, dict)
                and generative.get("requested") is True
                and generative.get("available") is False
                and generative.get("selection_effect") == "none"
                and isinstance(generative.get("reason"), str)
                and generative.get("reason")
            ):
                errors.append("generative_verifier_unreceipted")
        counterfactual = receipt.get("counterfactual_verifier")
        verified_counterfactual: dict[str, Any] | None = None
        if config.get("counterfactual_verifier_enabled") is True:
            if (
                isinstance(counterfactual, dict)
                and counterfactual.get("schema") == "aura.rlc.counterfactual_verifier.v1"
            ):
                try:
                    from core.brain.llm.latent_cortex.counterfactual_verifier import (
                        validate_counterfactual_verifier_envelope,
                    )

                    verified_counterfactual = (
                        validate_counterfactual_verifier_envelope(counterfactual)
                    )
                    blind_review = receipt.get("blind_review")
                    blind_rows = (
                        blind_review.get("rows")
                        if isinstance(blind_review, dict)
                        else None
                    )
                    if not isinstance(blind_rows, list):
                        raise ValueError("counterfactual verifier lacks blind score evidence")
                    blind_scores = {
                        int(row["branch"]): float(row["score"])
                        for row in blind_rows
                        if isinstance(row, dict)
                    }
                    expected_scores = {
                        str(branch): round(score, 6)
                        for branch, score in blind_scores.items()
                    }
                    if (
                        len(expected_scores) != len(blind_rows)
                        or verified_counterfactual["task_scores"] != expected_scores
                    ):
                        raise ValueError("counterfactual task scores differ from blind review")
                    source_selected = verified_counterfactual["source_selected_branch"]
                    if source_selected != max(
                        range(len(blind_scores)),
                        key=lambda branch: blind_scores[branch],
                    ):
                        raise ValueError("counterfactual source winner differs from task scores")
                    generated_effect = (
                        verified_generation.get("selection_effect")
                        if isinstance(verified_generation, dict)
                        else "none"
                    )
                    if generated_effect == "winner_replaced":
                        if (
                            verified_generation.get("vetoed_branch")
                            != verified_counterfactual["selected_branch"]
                        ):
                            raise ValueError(
                                "generative verifier did not follow counterfactual selection"
                            )
                    elif (
                        verified_counterfactual["selection_authority_admitted"] is True
                        and receipt.get("selected_branch")
                        != verified_counterfactual["selected_branch"]
                    ):
                        raise ValueError(
                            "counterfactual verifier selection was not applied"
                        )
                except (ImportError, KeyError, TypeError, ValueError):
                    errors.append("counterfactual_verifier_unproven")
                    verified_counterfactual = None
            elif not (
                isinstance(counterfactual, dict)
                and counterfactual.get("requested") is True
                and counterfactual.get("available") is False
                and counterfactual.get("selection_effect") == "none"
                and isinstance(counterfactual.get("reason"), str)
                and counterfactual.get("reason")
            ):
                errors.append("counterfactual_verifier_unreceipted")
        prefix_stability = receipt.get("prefix_stability")
        if config.get("prefix_stability_enabled") is True:
            if (
                isinstance(prefix_stability, dict)
                and prefix_stability.get("schema")
                == "aura.rlc.prefix_stability_verifier.v1"
            ):
                try:
                    from core.brain.llm.latent_cortex.prefix_stability import (
                        validate_prefix_stability_envelope,
                    )

                    validate_prefix_stability_envelope(
                        prefix_stability,
                        expected_calibrator_config=config.get(
                            "prefix_stability_calibrator"
                        ),
                    )
                except (ImportError, KeyError, OSError, TypeError, ValueError):
                    errors.append("prefix_stability_unproven")
            elif not (
                isinstance(prefix_stability, dict)
                and set(prefix_stability)
                == {
                    "requested",
                    "available",
                    "reason",
                    "selection_effect",
                    "correctness_effect",
                }
                and prefix_stability.get("requested") is True
                and prefix_stability.get("available") is False
                and prefix_stability.get("selection_effect") == "none"
                and prefix_stability.get("correctness_effect") == "none"
                and isinstance(prefix_stability.get("reason"), str)
                and prefix_stability.get("reason")
            ):
                errors.append("prefix_stability_unreceipted")
        if config.get("critic_blind_spot_evidence") is not None:
            try:
                from core.brain.llm.latent_cortex.critic_identity import (
                    validate_critic_identity,
                    validate_shared_blind_spot_evidence,
                )

                identity = validate_critic_identity(
                    receipt.get("critic_identity"),
                    worker_identity=(expected_worker_identity or receipt),
                )
                blind_spots = validate_shared_blind_spot_evidence(
                    receipt.get("shared_blind_spots"),
                    generator_function_sha256=identity["generator_identity"]["function_sha256"],
                    critic_function_sha256=identity["critic_function_sha256"],
                )
                if blind_spots != config.get("critic_blind_spot_evidence"):
                    raise ValueError("worker critic evidence differs from service snapshot")
                if blind_spots["critic_reliability_admitted"] is not True:
                    raise ValueError("critic reliability gate did not admit authority")
                if (
                    not isinstance(guidance, dict)
                    or guidance.get("requested") is not True
                    or guidance.get("available") is not True
                ):
                    raise ValueError("admitted critic was not causally used")
            except (ImportError, OSError, RuntimeError, TypeError, ValueError):
                errors.append("disjoint_critic_authority_unproven")
        controls = dict(runtime_controls or {})
        if controls:
            expected_alpha = controls.get("clean_user_surface_steering_alpha")
            expected_loops = controls.get("clean_user_surface_recurrent_loops")
            steering_requested = (
                not isinstance(expected_alpha, bool)
                and isinstance(expected_alpha, (int, float))
                and math.isfinite(expected_alpha)
                and expected_alpha > 0
            )
            if steering_requested:
                if receipt.get("worker_affective_steering_active") is not True:
                    errors.append("affective_steering_inactive")
                if receipt.get("episode_affective_steering_applied") is not True:
                    errors.append("episode_affective_steering_unapplied")
            if (
                isinstance(expected_alpha, bool)
                or not isinstance(expected_alpha, (int, float))
                or not math.isfinite(expected_alpha)
                or not 0.0 <= expected_alpha <= 1.0
                or isinstance(receipt.get("episode_affective_steering_alpha"), bool)
                or not isinstance(receipt.get("episode_affective_steering_alpha"), (int, float))
                or not math.isfinite(receipt["episode_affective_steering_alpha"])
                or not math.isclose(
                    float(receipt["episode_affective_steering_alpha"]),
                    float(expected_alpha),
                    rel_tol=0.0,
                    abs_tol=1e-6,
                )
            ):
                errors.append("affective_steering_alpha_mismatch")
            # A depth request the episode HALTED below is not the same thing
            # as a depth request that was never applied. Adaptive halting is a
            # designed feature -- convergence, an invariant, or the learned
            # stop policy can all end an episode at one step -- and reporting
            # that as "unproven" made a correct outcome indistinguishable from
            # a dropped request, on every turn where affect asked for two.
            if (
                type(expected_loops) is not int
                or expected_loops <= 0
                or not positive_int(receipt, "steps_taken")
            ):
                errors.append("live_recurrence_depth_unproven")
            elif receipt["steps_taken"] < expected_loops and not _recurrence_halt_reason(
                receipt
            ):
                # Halting below the request with a named reason is the policy
                # working; halting with no reason recorded means the depth was
                # never applied. Only the second is a fault.
                errors.append("live_recurrence_depth_not_applied")
        # CP126 f22c4ed8: this checked only that the field LOOKED like a
        # 64-character hex digest. The facade never hashed the actual
        # question, messages, config or budget, so a receipt could describe a
        # different request entirely and still pass — the digest proved the
        # worker could format a string, not that it answered this call.
        claimed_request_sha256 = receipt.get("request_payload_sha256")
        if not sha256(claimed_request_sha256):
            errors.append("request_payload_identity_unproven")
        elif expected_request_payload_sha256:
            if str(claimed_request_sha256) != expected_request_payload_sha256:
                errors.append("request_payload_identity_mismatch")
        else:
            # Say when the binding could not be performed rather than letting
            # a shape check stand in for it.
            errors.append("request_payload_identity_unbound")
        if not sha256(receipt.get("input_tokens_sha256")) or not positive_int(
            receipt, "input_token_count"
        ):
            errors.append("tokenized_input_identity_unproven")
        elif information_accounting is not None:
            rendered_inputs = [
                source
                for source in information_accounting["sources"]
                if source.get("source_id") == "rendered_model_input"
                and source.get("kind") == "model_input_tokens"
            ]
            if (
                len(rendered_inputs) != 1
                or rendered_inputs[0].get("content_sha256") != receipt.get("input_tokens_sha256")
                or rendered_inputs[0].get("token_count") != receipt.get("input_token_count")
            ):
                errors.append("input_information_binding_unproven")
        input_context_max_chars = config.get("input_context_max_chars", 0)
        if type(input_context_max_chars) is int and input_context_max_chars > 0:
            compaction = receipt.get("input_context_compaction")
            if not isinstance(compaction, dict):
                errors.append("input_context_compaction_missing")
            elif (
                compaction.get("schema") != "aura.latent_context_compaction.v1"
                or compaction.get("policy") != "resident_latent_salience_v1"
                or compaction.get("max_chars") != input_context_max_chars
                or not sha256(compaction.get("original_sha256"))
                or not sha256(compaction.get("compacted_sha256"))
                or not positive_int(compaction, "original_message_count")
                or not positive_int(compaction, "compacted_message_count")
                or not positive_int(compaction, "original_char_count")
                or not positive_int(compaction, "compacted_char_count")
                or compaction["compacted_char_count"] > input_context_max_chars
                or compaction["original_char_count"] < compaction["compacted_char_count"]
                or compaction["compacted_message_count"] > compaction["original_message_count"]
                or type(compaction.get("applied")) is not bool
                or type(compaction.get("omitted_char_count")) is not int
                or compaction["omitted_char_count"] < 0
                or compaction["omitted_char_count"]
                != compaction["original_char_count"] - compaction["compacted_char_count"]
                or (
                    compaction["applied"]
                    and (
                        compaction["original_sha256"] == compaction["compacted_sha256"]
                        or compaction["omitted_char_count"] == 0
                    )
                )
                or (
                    not compaction["applied"]
                    and (
                        compaction["original_sha256"] != compaction["compacted_sha256"]
                        or compaction["omitted_char_count"] != 0
                    )
                )
            ):
                errors.append("input_context_compaction_invalid")
        runtime_identity = receipt.get("runtime_identity")
        if not isinstance(runtime_identity, dict):
            errors.append("runtime_identity_missing")
        else:
            if runtime_identity.get("identity_bound") is not True:
                errors.append("runtime_identity_unbound")
            if not git_oid(runtime_identity.get("source_commit")):
                errors.append("runtime_source_commit_unproven")
            if not sha256(runtime_identity.get("workspace_state_sha256")):
                errors.append("runtime_workspace_identity_unproven")
            if not sha256(runtime_identity.get("shell_assets_sha256")):
                errors.append("runtime_shell_identity_unproven")
            if (
                runtime_identity.get("installed_app_required") is True
                and runtime_identity.get("installed_app_verified") is not True
            ):
                errors.append("installed_app_identity_unproven")
        if not sha256(receipt.get("schedule_hash")):
            errors.append("invalid_schedule_hash")
        if not positive_int(receipt, "steps_taken"):
            errors.append("no_recurrent_steps")
        if type(config.get("n_slots")) is not int or receipt.get("n_slots") != config.get(
            "n_slots"
        ):
            errors.append("workspace_cardinality_mismatch")
        if type(config.get("n_branches")) is not int or receipt.get("n_branches") != config.get(
            "n_branches"
        ):
            errors.append("branch_cardinality_mismatch")
        try:
            from core.brain.llm.latent_cortex.recurrent_grounding import (
                validate_recurrent_grounding_receipt,
            )

            validate_recurrent_grounding_receipt(
                receipt.get("recurrent_grounding"),
                input_tokens_sha256=str(receipt.get("input_tokens_sha256") or ""),
                input_token_count=int(receipt.get("input_token_count") or 0),
                cognitive_slots=list(receipt.get("cognitive_slots") or []),
                n_slots=int(config.get("n_slots") or 0),
                n_branches=int(config.get("n_branches") or 0),
                selected_branch=int(receipt.get("selected_branch") or 0),
            )
        except (ImportError, TypeError, ValueError):
            errors.append("recurrent_grounding_unproven")
        try:
            from core.brain.llm.latent_cortex.loop_core import (
                build_loop_core_contract,
            )
            from core.brain.llm.latent_cortex.loop_stability import (
                validate_loop_stability_receipt,
            )
            from core.brain.llm.latent_cortex.worker_handler import config_from_job

            executed_config = config_from_job(config)
            prelude_end = receipt.get("prelude_end")
            coda_start = receipt.get("coda_start")
            if type(prelude_end) is not int or type(coda_start) is not int:
                raise ValueError("loop boundaries must be integers")
            expected_loop_core = build_loop_core_contract(
                prelude_end=prelude_end,
                coda_start=coda_start,
                max_steps=executed_config.recurrence.max_steps,
                min_steps=executed_config.recurrence.min_steps,
                alpha=executed_config.recurrence.alpha,
                alpha_schedule=executed_config.recurrence.alpha_schedule,
                rms_clip_ratio=executed_config.recurrence.rms_clip_ratio,
                convergence_eps=executed_config.recurrence.convergence_eps,
                divergence_ratio=executed_config.recurrence.divergence_ratio,
                fixed_depth=executed_config.recurrence.fixed_depth,
            )
            validate_loop_stability_receipt(
                receipt.get("loop_stability"),
                recurrent_grounding=receipt.get("recurrent_grounding"),
                expected_loop_core=expected_loop_core,
            )
        except (ImportError, TypeError, ValueError):
            errors.append("loop_stability_unproven")
        try:
            from core.brain.llm.latent_cortex.kv_state_tree import (
                validate_kv_state_tree_receipt,
            )
            from core.brain.llm.latent_cortex.worker_handler import config_from_job

            executed_config = config_from_job(config)
            validate_kv_state_tree_receipt(
                receipt.get("kv_state_tree"),
                episode_id=str(receipt.get("episode_id") or ""),
                input_tokens_sha256=str(receipt.get("input_tokens_sha256") or ""),
                n_layers=int(receipt.get("n_layers") or 0),
                expected_n_branches=executed_config.branches.n_branches,
                require_final=True,
            )
        except (ImportError, OSError, TypeError, ValueError):
            errors.append("kv_state_tree_unproven")
        try:
            from core.brain.llm.latent_cortex.worker_handler import config_from_job

            executed_config = config_from_job(config)
            policy = receipt.get("decode_incumbent_policy")
            prompt_logits_sha256 = receipt.get(
                "decode_incumbent_prompt_logits_sha256"
            )
            if (
                policy != executed_config.decode_incumbent_policy
                or not isinstance(prompt_logits_sha256, str)
                or len(prompt_logits_sha256) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in prompt_logits_sha256
                )
            ):
                raise ValueError("decode incumbent identity is invalid")
            if policy == "vanilla_incumbent":
                tree = receipt.get("kv_state_tree")
                if not isinstance(tree, dict):
                    raise ValueError("decode incumbent KV tree is absent")
                finals = [
                    node
                    for node in tree.get("nodes", [])
                    if isinstance(node, dict) and node.get("final") is True
                ]
                if (
                    len(finals) != 1
                    or finals[0].get("latent_sha256") != ""
                    or receipt.get("first_logits_digest")
                    != prompt_logits_sha256
                ):
                    raise ValueError("decode incumbent final node is invalid")
                authority = str(finals[0].get("authority") or "")
                admitted_purposes = {
                    "vanilla_incumbent_output": {
                        "bind_captured_vanilla_incumbent",
                        "final_vanilla_incumbent_decode",
                    },
                    "canonical_ordinary_decode_artifact": {
                        "bind_canonical_incumbent_artifact",
                    },
                }
                purposes = admitted_purposes.get(authority)
                if purposes is None:
                    raise ValueError("decode incumbent authority is invalid")
                commits = [
                    event
                    for event in tree.get("events", [])
                    if isinstance(event, dict)
                    and event.get("purpose") in purposes
                    and event.get("disposition") == "committed"
                ]
                if (
                    len(commits) != 1
                    or commits[0].get("parent_node_sha256")
                    != tree.get("root_node_sha256")
                    or commits[0].get("result_node_sha256")
                    != finals[0].get("node_sha256")
                ):
                    raise ValueError("decode incumbent lineage is invalid")
        except (ImportError, KeyError, TypeError, ValueError):
            errors.append("decode_incumbent_unproven")
        try:
            from core.brain.llm.latent_cortex.update_gate import (
                UpdateGateRuntime,
                validate_update_gate_receipt,
            )
            from core.brain.llm.latent_cortex.worker_handler import config_from_job

            executed_config = config_from_job(config)
            expected_update_gate = UpdateGateRuntime.from_config(executed_config.update_gate)
            validate_update_gate_receipt(
                receipt.get("update_acceptance"),
                expected_gate=expected_update_gate,
                recurrent_grounding=receipt.get("recurrent_grounding"),
                loop_stability=receipt.get("loop_stability"),
            )
        except (ImportError, OSError, TypeError, ValueError):
            errors.append("update_acceptance_unproven")
        try:
            from core.brain.llm.latent_cortex.bidirectional_reflector import (
                validate_bidirectional_reflector_receipt,
            )
            from core.brain.llm.latent_cortex.worker_handler import config_from_job

            executed_config = config_from_job(config)

            validate_bidirectional_reflector_receipt(
                receipt.get("bidirectional_reflector"),
                update_acceptance=receipt.get("update_acceptance"),
                expected_n_branches=executed_config.branches.n_branches,
            )
        except (ImportError, TypeError, ValueError):
            errors.append("bidirectional_reflector_unproven")
        try:
            from core.brain.llm.latent_cortex.contradiction_tensor import (
                ContradictionTensorRuntime,
                validate_contradiction_tensor_receipt,
            )
            from core.brain.llm.latent_cortex.worker_handler import config_from_job

            executed_config = config_from_job(config)
            expected_contradiction = ContradictionTensorRuntime.from_config(
                executed_config.contradiction_head
            )
            validate_contradiction_tensor_receipt(
                receipt.get("contradiction_tensor"),
                expected_runtime=expected_contradiction,
                reflector=receipt.get("bidirectional_reflector"),
                expected_n_branches=executed_config.branches.n_branches,
            )
        except (ImportError, OSError, TypeError, ValueError):
            errors.append("contradiction_tensor_unproven")
        try:
            from core.brain.llm.latent_cortex.contradiction_perturber import (
                ContradictionPerturberConfig,
                validate_contradiction_perturbation_receipt,
            )
            from core.brain.llm.latent_cortex.worker_handler import config_from_job

            executed_config = config_from_job(config)
            information = (
                receipt.get("budget", {}).get("information_accounting", {})
                if isinstance(receipt.get("budget"), dict)
                else {}
            )
            policies = information.get("policies", {}) if isinstance(information, dict) else {}
            decoy = receipt.get("decoy_verification")
            cognitive_slots = receipt.get("cognitive_slots")
            protected_positions = (
                sorted(
                    {
                        int(row["slot"])
                        for row in cognitive_slots
                        if (isinstance(row, dict) and type(row.get("slot")) is int)
                    }
                )
                if isinstance(cognitive_slots, list)
                else []
            )
            validate_contradiction_perturbation_receipt(
                receipt.get("contradiction_perturbation"),
                expected_config=ContradictionPerturberConfig.from_value(
                    executed_config.contradiction_perturber
                ),
                contradiction_tensor=receipt.get("contradiction_tensor"),
                expected_selected_branch=int(receipt.get("selected_branch", -1)),
                expected_protected_positions=protected_positions,
                verifier_policy_sha256=str(policies.get("verifier", "")),
                decoy_review_sha256=(
                    str(decoy.get("receipt_sha256", ""))
                    if (isinstance(decoy, dict) and decoy.get("selection_admitted") is True)
                    else ""
                ),
            )
        except (ImportError, TypeError, ValueError):
            errors.append("contradiction_perturbation_unproven")
        try:
            from core.brain.llm.latent_cortex.neural_uncertainty import (
                NeuralUncertaintyRuntime,
                validate_neural_uncertainty_receipt,
            )
            from core.brain.llm.latent_cortex.worker_handler import config_from_job

            executed_config = config_from_job(config)
            expected_uncertainty = NeuralUncertaintyRuntime.from_config(
                executed_config.uncertainty_head
            )
            validate_neural_uncertainty_receipt(
                receipt.get("neural_uncertainty"),
                expected_runtime=expected_uncertainty,
                update_acceptance=receipt.get("update_acceptance"),
                expected_n_branches=executed_config.branches.n_branches,
            )
        except (ImportError, OSError, TypeError, ValueError):
            errors.append("neural_uncertainty_unproven")
        try:
            from core.brain.llm.latent_cortex.local_exploration import (
                LocalExplorationConfig,
                validate_local_exploration_receipt,
            )
            from core.brain.llm.latent_cortex.worker_handler import config_from_job

            executed_config = config_from_job(config)
            information = (
                receipt.get("budget", {}).get("information_accounting", {})
                if isinstance(receipt.get("budget"), dict)
                else {}
            )
            policies = information.get("policies", {}) if isinstance(information, dict) else {}
            decoy = receipt.get("decoy_verification")
            cognitive_slots = receipt.get("cognitive_slots")
            protected_positions = (
                sorted(
                    {
                        int(row["slot"])
                        for row in cognitive_slots
                        if (isinstance(row, dict) and type(row.get("slot")) is int)
                    }
                )
                if isinstance(cognitive_slots, list)
                else []
            )
            validate_local_exploration_receipt(
                receipt.get("local_exploration"),
                expected_config=LocalExplorationConfig.from_value(
                    executed_config.local_exploration
                ),
                contradiction_tensor=receipt.get("contradiction_tensor"),
                contradiction_perturbation=receipt.get("contradiction_perturbation"),
                neural_uncertainty=receipt.get("neural_uncertainty"),
                expected_selected_branch=int(receipt.get("selected_branch", -1)),
                expected_protected_positions=protected_positions,
                verifier_policy_sha256=str(policies.get("verifier", "")),
                decoy_review_sha256=(
                    str(decoy.get("receipt_sha256", ""))
                    if (isinstance(decoy, dict) and decoy.get("selection_admitted") is True)
                    else ""
                ),
            )
        except (ImportError, TypeError, ValueError):
            errors.append("local_exploration_unproven")
        try:
            from core.brain.llm.latent_cortex.heterogeneous_integrator import (
                HeterogeneousIntegrationConfig,
                validate_heterogeneous_integration_receipt,
            )
            from core.brain.llm.latent_cortex.worker_handler import config_from_job

            executed_config = config_from_job(config)
            information = (
                receipt.get("budget", {}).get("information_accounting", {})
                if isinstance(receipt.get("budget"), dict)
                else {}
            )
            policies = information.get("policies", {}) if isinstance(information, dict) else {}
            decoy = receipt.get("decoy_verification")
            validate_heterogeneous_integration_receipt(
                receipt.get("heterogeneous_integration"),
                expected_config=HeterogeneousIntegrationConfig.from_value(
                    executed_config.heterogeneous_integration
                ),
                contradiction_perturbation=receipt.get("contradiction_perturbation"),
                local_exploration=receipt.get("local_exploration"),
                verifier_policy_sha256=str(policies.get("verifier", "")),
                decoy_review_sha256=(
                    str(decoy.get("receipt_sha256", ""))
                    if (isinstance(decoy, dict) and decoy.get("selection_admitted") is True)
                    else ""
                ),
            )
        except (ImportError, TypeError, ValueError):
            errors.append("heterogeneous_integration_unproven")
        try:
            from core.brain.llm.latent_cortex.heterogeneous_integrator import (
                validate_heterogeneous_decode_receipt,
            )

            replacement_applied = (
                isinstance(receipt.get("answer_replacement"), dict)
                and receipt["answer_replacement"].get("decision") == "replace"
            )
            if replacement_applied:
                if (
                    not isinstance(answer_replacement_private, dict)
                    or not isinstance(
                        answer_replacement_private.get("baseline_tokens"),
                        list,
                    )
                ):
                    raise ValueError("replacement baseline tokens are unavailable")
                validate_heterogeneous_decode_receipt(
                    receipt.get("heterogeneous_decode"),
                    integration=receipt.get("heterogeneous_integration"),
                    expected_output_tokens=answer_replacement_private[
                        "baseline_tokens"
                    ],
                )
            else:
                validate_heterogeneous_decode_receipt(
                    receipt.get("heterogeneous_decode"),
                    integration=receipt.get("heterogeneous_integration"),
                    expected_output_tokens=output_tokens,
                )
        except (ImportError, TypeError, ValueError):
            errors.append("heterogeneous_decode_unproven")
        try:
            from core.brain.llm.latent_cortex.mistake_locator import (
                MistakeLocatorRuntime,
                validate_mistake_locator_receipt,
            )
            from core.brain.llm.latent_cortex.worker_handler import config_from_job

            executed_config = config_from_job(config)
            expected_locator = MistakeLocatorRuntime.from_config(executed_config.mistake_locator)
            validate_mistake_locator_receipt(
                receipt.get("mistake_locator"),
                expected_runtime=expected_locator,
                update_acceptance=receipt.get("update_acceptance"),
                expected_n_branches=executed_config.branches.n_branches,
                expected_domain=expected_domain,
            )
        except (ImportError, OSError, TypeError, ValueError):
            errors.append("mistake_locator_unproven")
        try:
            from core.brain.llm.latent_cortex.verifier_fusion import (
                validate_verifier_fusion_receipt,
            )

            validate_verifier_fusion_receipt(
                receipt.get("verifier_fusion"),
                blind_review=receipt.get("blind_review"),
                decoy_verification=receipt.get("decoy_verification"),
                generative_verifier=receipt.get("generative_verifier"),
                counterfactual_verifier=receipt.get("counterfactual_verifier"),
                prefix_stability=receipt.get("prefix_stability"),
                neural_uncertainty=receipt.get("neural_uncertainty"),
                mistake_locator=receipt.get("mistake_locator"),
                selected_branch=receipt.get("selected_branch"),
                evidence=config.get("verifier_fusion_evidence"),
            )
        except (ImportError, TypeError, ValueError):
            errors.append("verifier_fusion_unproven")
        try:
            from core.brain.llm.latent_cortex.stop_gate import (
                StopGateRuntime,
                validate_stop_gate_receipt,
            )
            from core.brain.llm.latent_cortex.worker_handler import config_from_job

            executed_config = config_from_job(config)
            expected_stop_gate = StopGateRuntime.from_config(executed_config.halting)
            validate_stop_gate_receipt(
                receipt.get("halting"),
                expected_gate=expected_stop_gate,
                expected_n_branches=executed_config.branches.n_branches,
                update_acceptance=receipt.get("update_acceptance"),
                loop_stability=receipt.get("loop_stability"),
                cognitive_action_trace=receipt.get("cognitive_action_trace"),
            )
        except (ImportError, OSError, TypeError, ValueError):
            errors.append("halting_unproven")
        if receipt.get("last_stage") != "action_state_captured":
            try:
                from core.brain.llm.latent_cortex.terminal_disposition import (
                    validate_terminal_disposition_receipt,
                )

                if not isinstance(output_text, str) or not isinstance(output_tokens, list):
                    raise ValueError("terminal output is unavailable")
                validate_terminal_disposition_receipt(
                    receipt.get("terminal_disposition"),
                    halting_reason=receipt.get("halting_reason"),
                    halting=receipt.get("halting"),
                    loop_stability=receipt.get("loop_stability"),
                    cognitive_action_trace=receipt.get("cognitive_action_trace"),
                    budget=receipt.get("budget"),
                    output_tokens=output_tokens,
                    output_text=output_text,
                    full_bridge_tokens_sha256=(
                        receipt.get("decode_bridge_tokens_sha256")
                        if config.get(
                            "decode_incumbent_policy", "vanilla_incumbent"
                        )
                        == "latent"
                        else hashlib.sha256(b"[]").hexdigest()
                    ),
                )
                language = receipt["terminal_disposition"]["language"]
                latent_output_authority = (
                    config.get("decode_incumbent_policy", "vanilla_incumbent")
                    == "latent"
                )
                expected_instruction_policy = (
                    "applied" if latent_output_authority else "suppressed"
                )
                if (
                    language.get("source")
                    not in {"resident_model_decode", "resident_model_repair"}
                    or language.get("instruction_policy")
                    != expected_instruction_policy
                    or language.get("instruction_applied")
                    is not latent_output_authority
                ):
                    raise ValueError("terminal language was not resident-model generated")
            except (ImportError, KeyError, TypeError, ValueError):
                errors.append("terminal_disposition_unproven")
            try:
                from core.brain.llm.latent_cortex.causal_receipt import (
                    validate_causal_receipt,
                )

                validate_causal_receipt(
                    receipt.get("causal_receipt"),
                    worker_receipt=receipt,
                    require_complete=True,
                )
            except (ImportError, TypeError, ValueError):
                errors.append("causal_receipt_unproven")
        try:
            from core.brain.llm.latent_cortex.verified_best import (
                validate_verified_best_receipt,
            )

            validate_verified_best_receipt(
                receipt.get("verified_best_state"),
                cognitive_action_trace=receipt.get("cognitive_action_trace"),
                loop_stability=receipt.get("loop_stability"),
                expected_n_branches=int(config.get("n_branches") or 0),
            )
        except (ImportError, TypeError, ValueError):
            errors.append("verified_best_state_unproven")
        try:
            from core.brain.llm.latent_cortex.transient_constraints import (
                TransientConstraintConfig,
                validate_transient_constraint_receipt,
            )
            from core.brain.llm.latent_cortex.worker_handler import config_from_job

            cognitive_slots = receipt.get("cognitive_slots")
            protected = (
                tuple(
                    sorted(
                        {
                            int(row["slot"])
                            for row in cognitive_slots
                            if (isinstance(row, dict) and type(row.get("slot")) is int)
                        }
                    )
                )
                if isinstance(cognitive_slots, list)
                else ()
            )
            executed_config = config_from_job(config)
            expected_branches = executed_config.branches.n_branches
            validate_transient_constraint_receipt(
                receipt.get("transient_negative_constraints"),
                episode_id=str(receipt.get("episode_id") or ""),
                objective_sha256=str(receipt.get("input_tokens_sha256") or ""),
                n_branches=expected_branches,
                protected_positions={index: protected for index in range(expected_branches)},
                expected_config=TransientConstraintConfig.from_value(
                    executed_config.transient_negative_constraints
                ),
                cognitive_action_trace=receipt.get("cognitive_action_trace"),
                verifier_preflight=receipt.get("verifier_preflight"),
                information_accounting=information_accounting,
                resource_accounting=resource_accounting,
                kv_state_tree=receipt.get("kv_state_tree"),
                verified_best_state=receipt.get("verified_best_state"),
                loop_stability=receipt.get("loop_stability"),
                require_verified_best_binding=True,
                require_external_bindings=True,
            )
        except (ImportError, TypeError, ValueError):
            errors.append("transient_negative_constraints_unproven")
        try:
            from core.brain.llm.latent_cortex.virtual_quanta import (
                VirtualQuantaConfig,
                validate_virtual_quanta_receipt,
            )
            from core.brain.llm.latent_cortex.worker_handler import config_from_job

            executed_config = config_from_job(config)
            virtual_receipt = receipt.get("virtual_quanta")
            validate_virtual_quanta_receipt(
                virtual_receipt,
                episode_id=str(receipt.get("episode_id") or ""),
                objective_sha256=str(receipt.get("input_tokens_sha256") or ""),
                n_branches=executed_config.branches.n_branches,
                expected_config=VirtualQuantaConfig.from_value(executed_config.virtual_quanta),
                cognitive_slots=receipt.get("cognitive_slots"),
                verifier_preflight=receipt.get("verifier_preflight"),
                information_accounting=information_accounting,
                resource_accounting=resource_accounting,
                kv_state_tree=receipt.get("kv_state_tree"),
                require_external_bindings=bool(
                    isinstance(virtual_receipt, dict) and virtual_receipt.get("arms")
                ),
            )
        except (ImportError, TypeError, ValueError):
            errors.append("virtual_quanta_unproven")
        try:
            from core.brain.llm.latent_cortex.latent_tree_search import (
                LatentTreeSearchConfig,
                validate_latent_tree_receipt,
            )
            from core.brain.llm.latent_cortex.worker_handler import config_from_job

            executed_config = config_from_job(config)
            tree_receipt = receipt.get("latent_tree_search")
            validate_latent_tree_receipt(
                tree_receipt,
                episode_id=str(receipt.get("episode_id") or ""),
                objective_sha256=str(receipt.get("input_tokens_sha256") or ""),
                expected_config=LatentTreeSearchConfig.from_value(
                    executed_config.latent_tree_search
                ),
                kv_state_tree=receipt.get("kv_state_tree"),
                cognitive_action_trace=receipt.get("cognitive_action_trace"),
                resource_accounting=resource_accounting,
                loop_stability=receipt.get("loop_stability"),
                require_external_bindings=bool(
                    isinstance(tree_receipt, dict) and tree_receipt.get("transactions")
                ),
            )
        except (ImportError, TypeError, ValueError):
            errors.append("latent_tree_search_unproven")
        one_shot_slots = [
            row
            for row in (receipt.get("cognitive_slots") or [])
            if isinstance(row, dict)
            and row.get("knowledge_class") == "one_shot_nonparametric_memory"
        ]
        one_shot_receipt = receipt.get("nonparametric_memory")
        try:
            from core.brain.llm.latent_cortex.nonparametric_context import (
                validate_receipt as validate_nonparametric_receipt,
            )

            if one_shot_receipt:
                validated_one_shot = validate_nonparametric_receipt(one_shot_receipt)
                if validated_one_shot["applied"]:
                    if (
                        len(one_shot_slots) != 1
                        or one_shot_slots[0].get("instruction_authority") is not False
                        or one_shot_slots[0].get("text_sha256")
                        != validated_one_shot["observation_sha256"]
                    ):
                        raise ValueError("admitted one-shot evidence is not bound to its slot")
                elif one_shot_slots:
                    raise ValueError("one-shot evidence slot exists without admitted retrieval")
                expected_resource = validated_one_shot["resource_accounting"]
                operation = (
                    resource_accounting.get("operations", {}).get("nonparametric_memory_retrieval")
                    if resource_accounting is not None
                    else None
                )
                if not isinstance(operation, dict) or any(
                    operation.get(resource_name) != expected_resource[receipt_name]
                    for resource_name, receipt_name in (
                        ("tensor_element_reads", "tensor_element_reads"),
                        ("tensor_element_writes", "tensor_element_writes"),
                        ("tensor_scalar_ops", "tensor_scalar_ops"),
                        ("host_scalar_ops", "host_scalar_ops"),
                    )
                ):
                    raise ValueError("one-shot retrieval work differs from resource ledger")
                if information_accounting is None:
                    raise ValueError("one-shot information accounting is absent")
                from core.brain.llm.latent_cortex.resource_accounting import (
                    policy_sha256,
                )

                source_identity = validated_one_shot["source_identity"]
                store_sources = [
                    row
                    for row in information_accounting["sources"]
                    if row.get("source_id") == "one_shot_nonparametric_memory"
                ]
                if source_identity:
                    if (
                        len(store_sources) != 1
                        or store_sources[0].get("kind") != "local_nonparametric_memory_store"
                        or store_sources[0].get("content_sha256")
                        != source_identity["content_sha256"]
                        or store_sources[0].get("byte_count") != source_identity["source_bytes"]
                    ):
                        raise ValueError("one-shot store differs from information ledger")
                elif store_sources:
                    raise ValueError("information ledger claims an unavailable one-shot store")
                expected_policy = policy_sha256(
                    {
                        "policy": "context_only_prompt_tail_recall_v1",
                        "active_source_receipt_sha256": source_identity.get(
                            "receipt_sha256", "none"
                        ),
                    }
                )
                if (
                    information_accounting["policies"].get("nonparametric_memory")
                    != expected_policy
                ):
                    raise ValueError("one-shot retrieval policy is not bound")
                context_sources = [
                    row
                    for row in information_accounting["sources"]
                    if str(row.get("source_id") or "").endswith(":one_shot_memory")
                ]
                if validated_one_shot["applied"]:
                    slot = one_shot_slots[0]
                    expected_source_id = (
                        f"cognitive_context:{slot.get('context_index')}:one_shot_memory"
                    )
                    if (
                        len(context_sources) != 1
                        or context_sources[0].get("source_id") != expected_source_id
                        or context_sources[0].get("kind") != "typed_cognitive_context"
                        or context_sources[0].get("content_sha256")
                        != validated_one_shot["observation_sha256"]
                    ):
                        raise ValueError("one-shot observation differs from information ledger")
                elif context_sources:
                    raise ValueError("information ledger claims an unadmitted one-shot observation")
            elif one_shot_slots:
                raise ValueError("one-shot evidence slot has no retrieval receipt")
        except (ImportError, TypeError, ValueError):
            errors.append("nonparametric_memory_binding_unproven")
        isolation_steps = config.get("isolation_steps")
        if type(isolation_steps) is int:
            isolation = receipt.get("branch_isolation")
            isolation_valid = isinstance(isolation, dict)
            if isolation_valid:
                candidates = isolation.get("candidates")
                cache_discipline = isolation.get("cache_discipline")
                branch_count = config.get("n_branches")
                candidate_rows_valid = (
                    isinstance(candidates, list)
                    and type(branch_count) is int
                    and len(candidates) == branch_count
                    and all(
                        isinstance(row, dict)
                        and row.get("index") == index
                        and isinstance(row.get("role"), str)
                        and bool(row["role"])
                        and sha256(row.get("context_sha256"))
                        and sha256(row.get("rng_stream_sha256"))
                        and sha256(row.get("seed_sha256"))
                        and sha256(row.get("candidate_sha256"))
                        and type(row.get("candidate_step")) is int
                        and row["candidate_step"] >= isolation_steps
                        for index, row in enumerate(candidates)
                    )
                )
                unique_commitments = bool(candidate_rows_valid) and all(
                    len({row[key] for row in candidates}) == len(candidates)
                    for key in (
                        "rng_stream_sha256",
                        "seed_sha256",
                        "candidate_sha256",
                    )
                )
                one_context = (
                    bool(candidate_rows_valid)
                    and len({row["context_sha256"] for row in candidates}) == 1
                )
                cache_valid = (
                    isinstance(cache_discipline, dict)
                    and set(cache_discipline)
                    == {
                        "schema",
                        "nonpersistent_calls",
                        "restored_calls",
                        "restore_failures",
                        "all_restored",
                    }
                    and cache_discipline.get("schema") == "aura.rlc.cache_discipline.v1"
                    and positive_int(cache_discipline, "nonpersistent_calls")
                    and cache_discipline.get("restored_calls")
                    == cache_discipline.get("nonpersistent_calls")
                    and cache_discipline.get("restore_failures") == 0
                    and cache_discipline.get("all_restored") is True
                )
                exchanges = receipt.get("exchanges")
                first_exchange_step = isolation.get("first_exchange_step")
                exposure_valid = (
                    nonnegative_int(isolation, "blocked_cross_exposures")
                    and (
                        (exchanges == 0 and first_exchange_step is None)
                        or (
                            type(exchanges) is int
                            and exchanges > 0
                            and type(first_exchange_step) is int
                            and first_exchange_step >= isolation_steps
                        )
                    )
                    and isolation.get("cross_exposure_started")
                    is (type(exchanges) is int and exchanges > 0)
                )
                isolation_valid = (
                    set(isolation)
                    == {
                        "schema",
                        "n_branches",
                        "required_steps",
                        "sealed",
                        "certified",
                        "reason",
                        "configured_role_lesion",
                        "seed_alias_free",
                        "seed_states_unique",
                        "rng_streams_unique",
                        "cross_exposure_started",
                        "first_exchange_step",
                        "blocked_cross_exposures",
                        "candidates",
                        "cache_discipline",
                    }
                    and isolation.get("schema") == "aura.rlc.branch_isolation.v1"
                    and isolation.get("n_branches") == branch_count
                    and isolation.get("required_steps") == isolation_steps
                    and isolation.get("sealed") is True
                    and isolation.get("certified") is True
                    and isolation.get("reason") == "certified"
                    and isolation.get("configured_role_lesion") is False
                    and isolation.get("seed_alias_free") is True
                    and isolation.get("seed_states_unique") is True
                    and isolation.get("rng_streams_unique") is True
                    and candidate_rows_valid
                    and unique_commitments
                    and one_context
                    and cache_valid
                    and exposure_valid
                )
            if not isolation_valid:
                errors.append("branch_isolation_unproven")
        exchanges = receipt.get("exchanges")
        _check_the_exchange_count_contract(
            config=config,
            errors=errors,
            exchanges=exchanges,
            receipt=receipt,
            resource_accounting=resource_accounting,
        )
        if (
            not (type(exchanges) is int and exchanges > 0)
            and resource_accounting is not None
            and "branch_exchange" in resource_accounting.get("operations", {})
        ):
            errors.append("branch_exchange_resource_binding_unproven")
        raw_action_trace = receipt.get("cognitive_action_trace")
        LatentCortexService._receipt_action_trace_errors(answer_replacement_private, config, errors, expected_objective, output_text, output_tokens, raw_action_trace, receipt, resource_accounting)
        preflight: dict[str, Any] | None = None
        if receipt.get("verifier_preflight"):
            try:
                from core.brain.llm.latent_cortex.blind_review import (
                    validate_decoy_preflight_receipt,
                )

                preflight = validate_decoy_preflight_receipt(
                    receipt.get("verifier_preflight"),
                    episode_id=receipt.get("episode_id"),
                    objective_sha256=receipt.get("input_tokens_sha256"),
                )
                if preflight[
                    "verifier_admitted"
                ] is False and "verifier_preflight_decoy_calibration_failed" not in (
                    receipt.get("honest_flags") or []
                ):
                    raise ValueError("decoy preflight rejection was not disclosed")
            except (ImportError, TypeError, ValueError):
                errors.append("decoy_verifier_preflight_unproven")
        LatentCortexService._receipt_counterfactual_errors(config, errors, receipt, verified_counterfactual, verified_generation)
        # The first exchange an episode may take is not the interval. The
        # ensemble refuses every exchange until isolation seals, so an episode
        # with isolation above the interval reaches its first eligible step
        # later than the interval says, and demanding an exchange before then
        # accuses a worker of skipping something it was never offered.
        exchange_interval = config.get("exchange_interval")
        if (
            type(exchange_interval) is int
            and exchange_interval > 0
            and type(config.get("n_branches")) is int
            and config["n_branches"] > 1
            and positive_int(receipt, "steps_taken")
            and not positive_int(receipt, "exchanges")
            and eligible_exchange_steps(
                n_branches=int(config["n_branches"]),
                max_steps=int(receipt["steps_taken"]),
                isolation_steps=int(config.get("isolation_steps") or 1),
                exchange_interval=exchange_interval,
            )
        ):
            errors.append("branch_exchange_unproven")
        budget = receipt.get("budget")
        if not isinstance(budget, dict) or not positive_int(budget, "spent_layer_apps"):
            errors.append("missing_compute_receipt")
        elif (
            not positive_int(budget, "max_layer_apps")
            or budget["spent_layer_apps"] > budget["max_layer_apps"]
            or budget.get("exhausted") is not False
        ):
            errors.append("incomplete_or_exhausted_compute_receipt")
        elif isinstance(allocated_budget, dict):
            # CP126 0673f84a: the checks above compare the worker's spend
            # against the worker's OWN reported maximum, which any receipt can
            # satisfy by reporting a large enough ceiling. Nothing compared
            # either number with what this facade actually allocated, so an
            # episode could spend far more than its budget and still produce a
            # self-consistent, passing receipt.
            allocated_max = allocated_budget.get("max_layer_apps")
            if type(allocated_max) is int and allocated_max > 0:
                # Independent, not elif: a receipt can declare a ceiling above
                # its allocation without spending it, and a spend above the
                # allocation is the fact that matters even when the declared
                # ceiling explains it. Chaining them made the spend check
                # unreachable.
                if budget["max_layer_apps"] > allocated_max:
                    errors.append("compute_ceiling_exceeds_allocation")
                if budget["spent_layer_apps"] > allocated_max:
                    errors.append("compute_spend_exceeds_allocation")
            allocated_wall = allocated_budget.get("wall_clock_s")
            spent_wall = budget.get("wall_clock_s")
            if (
                isinstance(allocated_wall, (int, float))
                and math.isfinite(float(allocated_wall))
                and allocated_wall > 0
                and isinstance(spent_wall, (int, float))
                and not isinstance(spent_wall, bool)
                and math.isfinite(float(spent_wall))
                # A small overrun is scheduling noise; a large one means the
                # deadline was not honoured.
                and float(spent_wall) > float(allocated_wall) * 1.25
            ):
                errors.append("wall_clock_exceeds_allocation")
        if not positive_int(receipt, "decode_requested_tokens") or receipt.get(
            "decode_requested_tokens"
        ) != config.get("decode_max_tokens"):
            errors.append("decode_request_mismatch")
        if not positive_int(receipt, "decode_generated_tokens"):
            errors.append("decode_output_empty")
        decode_contract = config.get("decode_contract", "none")
        contract_required = decode_contract == "final_answer_v1"
        configured_contract_grace = config.get(
            "decode_contract_grace_tokens",
            0,
        )
        if contract_required:
            if receipt.get("decode_contract_required") is not True:
                errors.append("decode_contract_requirement_unreceipted")
            if receipt.get("decode_contract_satisfied") is not True:
                errors.append("decode_contract_unsatisfied")
            if receipt.get("decode_termination") not in {
                "contract_complete",
                "confidence_bound_replacement",
            }:
                errors.append("decode_contract_termination_mismatch")
            if (
                type(configured_contract_grace) is not int
                or configured_contract_grace < 0
                or receipt.get("decode_contract_grace_tokens") != configured_contract_grace
            ):
                errors.append("decode_contract_grace_mismatch")
            grace_used = receipt.get("decode_contract_grace_used_tokens")
            expected_grace_used = max(
                0,
                int(receipt.get("decode_generated_tokens") or 0)
                - int(receipt.get("decode_requested_tokens") or 0),
            )
            if (
                type(grace_used) is not int
                or not 0 <= grace_used <= configured_contract_grace
                or grace_used != expected_grace_used
            ):
                errors.append("decode_contract_grace_accounting_invalid")
        elif receipt.get("decode_contract_required") is True:
            errors.append("unexpected_decode_contract")
        configured_probe_tokens = config.get("verifier_probe_max_tokens", 48)
        if (
            type(configured_probe_tokens) is not int
            or receipt.get("verifier_probe_max_tokens") != configured_probe_tokens
        ):
            errors.append("verifier_probe_profile_mismatch")
        configured_probe_contract = config.get("verifier_probe_contract", "none")
        if (
            "verifier_probe_contract" in config
            or "verifier_probe_contract" in receipt
        ) and (
            configured_probe_contract not in {"none", "final_answer_v1"}
            or receipt.get("verifier_probe_contract") != configured_probe_contract
        ):
            errors.append("verifier_probe_contract_mismatch")
        if receipt.get("decode_termination") not in {
            "eos",
            # The public answer contract completed (one FINAL_ANSWER JSON
            # object closed and parsed) — a complete answer by construction.
            "contract_complete",
            "token_limit",
            # Sentence grace: the limit landed mid-sentence and sampling
            # continued a few model-chosen tokens to the natural boundary.
            "token_limit_sentence_grace",
            # Wall-clock analogues: time pressure ended decoding, ideally at
            # a sentence boundary (wind-down). The output-quality gate is
            # the completeness judge either way.
            "wall_reserve_sentence_grace",
            # Raw "wall_reserve" is deliberately absent: the engine now emits
            # it only when the reserve was crossed with no sentence boundary
            # reached, which is a known fragment rather than a time-bounded
            # answer.
            "confidence_bound_replacement",
        }:
            errors.append("decode_incomplete")
        decode_bridge_policy = config.get("decode_bridge_policy", "none")
        latent_output_authority = (
            config.get("decode_incumbent_policy", "vanilla_incumbent") == "latent"
        )
        decode_bridge_token_count = receipt.get("decode_bridge_token_count")
        if latent_output_authority and decode_bridge_policy in {
            "assistant_answer_v1",
            "assistant_answer_v2",
            "assistant_answer_v3",
        }:
            if receipt.get("decode_bridge_applied") is not True:
                errors.append("decode_bridge_unapplied")
            if receipt.get("decode_bridge_policy") != decode_bridge_policy:
                errors.append("decode_bridge_policy_mismatch")
            if not positive_int(receipt, "decode_bridge_token_count"):
                errors.append("decode_bridge_tokens_missing")
            if not sha256(receipt.get("decode_bridge_tokens_sha256")):
                errors.append("decode_bridge_token_identity_unproven")
            if not sha256(receipt.get("decode_bridge_logits_digest")):
                errors.append("decode_bridge_logits_unproven")
        elif not latent_output_authority and (
            receipt.get("decode_bridge_applied") is True
            or type(decode_bridge_token_count) is not int
            or decode_bridge_token_count != 0
            or receipt.get("decode_bridge_tokens_sha256") not in {None, ""}
            or receipt.get("decode_bridge_logits_digest") not in {None, ""}
        ):
            errors.append("decode_bridge_applied_to_vanilla_incumbent")
        if not nonnegative_int(receipt, "decode_newline_suppressions"):
            errors.append("decode_newline_discipline_unreceipted")
        configured_repetition = config.get("decode_repetition_penalty", 1.0)
        applied_repetition = receipt.get("decode_repetition_penalty_applied")
        if (
            isinstance(applied_repetition, bool)
            or not isinstance(applied_repetition, (int, float))
            or not isinstance(configured_repetition, (int, float))
            or isinstance(configured_repetition, bool)
            or abs(float(applied_repetition) - float(configured_repetition)) > 1e-9
        ):
            errors.append("decode_repetition_guard_unproven")
        configured_temperature = config.get("decode_temperature", 0.0)
        configured_top_p = config.get("decode_top_p", 1.0)
        if (
            isinstance(configured_temperature, bool)
            or not isinstance(configured_temperature, (int, float))
            or isinstance(receipt.get("decode_temperature"), bool)
            or not isinstance(receipt.get("decode_temperature"), (int, float))
            or not math.isclose(
                float(receipt["decode_temperature"]),
                float(configured_temperature),
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            errors.append("decode_temperature_mismatch")
        if (
            isinstance(configured_top_p, bool)
            or not isinstance(configured_top_p, (int, float))
            or isinstance(receipt.get("decode_top_p"), bool)
            or not isinstance(receipt.get("decode_top_p"), (int, float))
            or not math.isclose(
                float(receipt["decode_top_p"]),
                float(configured_top_p),
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            errors.append("decode_top_p_mismatch")
        raw_flags = receipt.get("honest_flags")
        if not isinstance(raw_flags, list) or any(not isinstance(flag, str) for flag in raw_flags):
            errors.append("invalid_honest_flags")
            flags: list[str] = []
        else:
            flags = raw_flags
        if any(flag.startswith("fallback_vanilla") for flag in flags):
            errors.append("vanilla_fallback")
        _check_the_latent_optimiser_contract(
            config=config,
            errors=errors,
            nonnegative_int=nonnegative_int,
            positive_int=positive_int,
            receipt=receipt,
            verifier_arbitration_valid=verifier_arbitration_valid,
        )
        LatentCortexService._receipt_fast_weight_errors(config, errors, expected_worker_identity, finite_number_list, nonnegative_int, output_text, output_tokens, positive_int, receipt, resource_accounting)
        return errors
