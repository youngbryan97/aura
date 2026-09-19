"""The steps deep reasoning is assembled from.

Lifted whole out of `latent_cortex_service`. Every name taken from it is imported at
CALL time: that module imports this one to build the class, and a test that
patches a name on it has to reach the code that reads it.
"""
from __future__ import annotations

import time


class _ReasonsDeeply:
    """Lifted whole out of LatentCortexService; see latent_cortex_service.py."""

    def _deep_reason_part_1(self, config, controller_decision, domain, messages, question, stakes, uncertainty):
        from .latent_cortex_service import (
            logger,
        )

        try:
            from core.brain.llm.latent_cortex.branches import BRANCH_ROLES
            from core.brain.llm.latent_cortex.correlated_support import (
                get_branch_correlation_ledger,
            )
            from core.brain.llm.latent_cortex.execution_controller import context_bucket

            branch_count = int(config.get("n_branches") or 1)
            correlation_roles = list(BRANCH_ROLES[:branch_count])
            correlation_bucket = (
                str(controller_decision.get("bucket") or "")
                if controller_decision is not None
                else context_bucket(
                    self._visible_objective(question, messages),
                    domain,
                    stakes,
                    uncertainty,
                )
            )
            correlation_ledger = get_branch_correlation_ledger()
            config["branch_correlation_evidence"] = correlation_ledger.evidence(
                bucket=correlation_bucket,
                roles=correlation_roles,
            )
            self._last_allocation["correlated_support"] = {
                **correlation_ledger.status(),
                "bucket": correlation_bucket,
                "roles": correlation_roles,
                "evidence_state": config["branch_correlation_evidence"]["evidence_state"],
            }
        except (
            ImportError,
            AttributeError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            logger.debug("Branch correlation evidence unavailable: %s", exc)
            config["branch_correlation_evidence"] = None
        try:
            from core.brain.llm.latent_cortex.execution_controller import context_bucket
            from core.brain.llm.latent_cortex.verifier_fusion import (
                get_verifier_fusion_ledger,
            )

            verifier_bucket = (
                str(controller_decision.get("bucket") or "")
                if controller_decision is not None
                else context_bucket(
                    self._visible_objective(question, messages),
                    domain,
                    stakes,
                    uncertainty,
                )
            )
            verifier_ledger = get_verifier_fusion_ledger()
            config["verifier_fusion_evidence"] = verifier_ledger.evidence(
                bucket=verifier_bucket
            )
            self._last_allocation["verifier_fusion"] = {
                **verifier_ledger.status(),
                "bucket": verifier_bucket,
                "evidence_state": config["verifier_fusion_evidence"][
                    "evidence_state"
                ],
            }
        except (
            ImportError,
            AttributeError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            logger.debug("Verifier fusion evidence unavailable: %s", exc)
            config["verifier_fusion_evidence"] = None

    def _deep_reason_build_critic_source_identity(self, config, controller_decision, domain, messages, question, stakes, uncertainty, worker_identity):
        from core.brain.llm.latent_cortex.critic_identity import (
            build_critic_source_identity,
            build_generator_function_identity,
            get_critic_blind_spot_ledger,
        )
        from core.brain.llm.latent_cortex.execution_controller import (
            context_bucket,
        )

        generator_identity = build_generator_function_identity(worker_identity)
        critic_source = build_critic_source_identity()
        critic_ledger = get_critic_blind_spot_ledger()
        critic_bucket = (
            str(controller_decision.get("bucket") or "")
            if controller_decision is not None
            else context_bucket(
                self._visible_objective(question, messages),
                domain,
                stakes,
                uncertainty,
            )
        )
        config["critic_blind_spot_evidence"] = critic_ledger.evidence(
            bucket=critic_bucket,
            generator_function_sha256=generator_identity["function_sha256"],
            critic_function_sha256=critic_source["source_closure_sha256"],
        )
        self._last_allocation["critic_blind_spots"] = {
            **critic_ledger.status(),
            "bucket": critic_bucket,
            "evidence_state": config["critic_blind_spot_evidence"]["evidence_state"],
            "critic_reliability_admitted": config["critic_blind_spot_evidence"][
                "critic_reliability_admitted"
            ],
        }

    def _deep_reason_recommended_completion_tokens(self, capacity_decode_tokens, messages, question, target_decode_tokens):
        from core.brain.llm.measured_admission import (
            recommended_completion_tokens,
            recommended_foreground_deadline,
        )
        from core.brain.llm.model_registry import runtime_model_measurement_key

        from .latent_cortex_service import (
            _input_prompt_tokens,
        )

        prompt_tokens = _input_prompt_tokens(messages, question)
        target_decode_tokens, length_confidence, length_samples = (
            recommended_completion_tokens(
                model=runtime_model_measurement_key(),
                prompt_tokens=prompt_tokens,
                maximum_tokens=capacity_decode_tokens,
                prior_tokens=target_decode_tokens,
            )
        )
        required_wall_clock_s = 65.0 + (0.26 * target_decode_tokens)
        measured_s, _confidence, samples = recommended_foreground_deadline(
            model=runtime_model_measurement_key(),
            prompt_tokens=prompt_tokens,
            decode_tokens=max(1, target_decode_tokens),
            minimum_seconds=0.0,
            maximum_seconds=float("inf"),
        )
        if samples > 0 and measured_s > 0.0:
            required_wall_clock_s = float(measured_s)
        else:
            from core.brain.llm.generation_allowance import resident_generation_seconds

            live_seconds = resident_generation_seconds(
                messages or [{"role": "user", "content": question}],
                target_decode_tokens,
                private_tokens_included=True,
            )
            if live_seconds > 0.0:
                required_wall_clock_s = live_seconds
        self._last_allocation[
            "answer_surface_wall_clock_samples"
        ] = int(samples)
        self._last_allocation.update(
            {
                "answer_surface_prompt_tokens": prompt_tokens,
                "answer_surface_capacity_tokens": capacity_decode_tokens,
                "answer_surface_planning_tokens": target_decode_tokens,
                "answer_surface_length_confidence": length_confidence.value,
                "answer_surface_length_samples": int(length_samples),
            }
        )
        return required_wall_clock_s, target_decode_tokens

    @staticmethod
    def _deep_reason_policy_receipt(action_policy_evidence, result_receipt):
        from .latent_cortex_service import (
            _WRITTEN_WHEN_ACTIONS_ARE_SELECTED,
            _ActionSelectionNeverRanError,
        )

        policy_receipt = result_receipt.get("value_of_computation")
        raw_trace = result_receipt.get("cognitive_action_trace")
        policy_fields = {
            "schema",
            "bucket",
            "snapshot_sha256",
            "active",
            "executors",
            "actions_selected",
            "checked_transitions",
            "selected_actions",
        }
        # Nine conditions, and the one that fires is the one worth
        # saying. Together they were "the receipt is incomplete",
        # which is where the trail went cold on a browser action that
        # never started.
        if not isinstance(policy_receipt, dict):
            raise ValueError(
                "worker action policy receipt is incomplete: no receipt at all "
                f"(got {type(policy_receipt).__name__})"
            )
        missing = policy_fields - set(policy_receipt)
        extra = set(policy_receipt) - policy_fields
        # Never reaching action selection is not disagreeing about it.
        #
        # These four are written at the very end of a full episode. An
        # episode that stopped earlier — on its budget, or on any of
        # the ways one ends — has a receipt without them, and that was
        # read as the worker's policy CONTRADICTING the host's. It is
        # not a contradiction: nothing was claimed, so nothing can
        # disagree. Reported as a mismatch it is ineligible for bypass,
        # and a browser action that had cleared every authority gate
        # was refused before it started.
        #
        # LIVE 2026-08-31, and this is what took the demo down.
        if missing == _WRITTEN_WHEN_ACTIONS_ARE_SELECTED and not extra:
            raise _ActionSelectionNeverRanError(
                "the episode ended before it selected any actions"
            )
        if missing or extra:
            raise ValueError(
                "worker action policy receipt is incomplete: fields differ"
                + (f", missing {sorted(missing)}" if missing else "")
                + (f", unexpected {sorted(extra)}" if extra else "")
            )
        for part in ("schema", "snapshot_sha256", "bucket"):
            if policy_receipt.get(part) != action_policy_evidence[part]:
                raise ValueError(
                    f"worker action policy receipt is incomplete: {part} differs "
                    f"({policy_receipt.get(part)!r} against "
                    f"{action_policy_evidence[part]!r})"
                )
        if policy_receipt.get("active") is not True:
            raise ValueError(
                "worker action policy receipt is incomplete: it is not active"
            )
        if not isinstance(raw_trace, list) or not raw_trace:
            raise ValueError(
                "worker action policy receipt is incomplete: no action trace "
                f"(got {type(raw_trace).__name__} of "
                f"{len(raw_trace) if isinstance(raw_trace, list) else 0})"
            )
        if policy_receipt.get("actions_selected") != len(raw_trace):
            raise ValueError(
                "worker action policy receipt is incomplete: it counts "
                f"{policy_receipt.get('actions_selected')!r} action(s) and the "
                f"trace has {len(raw_trace)}"
            )
        return policy_receipt, raw_trace

    @staticmethod
    def _deep_reason_part_5(action_policy_evidence, action_transitions, external_execution_offer, policy_receipt, raw_trace, result_receipt, validated_trace):
        for validated_row in validated_trace["rows"]:
            decision = validated_row["decision"]
            transition = validated_row["transition"]
            if (
                transition["snapshot_sha256"] != action_policy_evidence["snapshot_sha256"]
                or transition["bucket"] != action_policy_evidence["bucket"]
                or decision["snapshot_sha256"] != action_policy_evidence["snapshot_sha256"]
                or decision["bucket"] != action_policy_evidence["bucket"]
            ):
                raise ValueError("worker action transition authority differs")
            action_transitions.append(transition)
        selected_actions = [row["action"] for row in action_transitions]
        checked_transitions = sum(int(row["checked"]) for row in action_transitions)
        if (
            validated_trace["selected_actions"] != selected_actions
            or policy_receipt.get("selected_actions") != selected_actions
            or policy_receipt.get("checked_transitions") != checked_transitions
        ):
            raise ValueError("worker action policy summary differs from trace")
        raw_handoff = result_receipt.get("external_execution_handoff")
        if external_execution_offer is not None:
            from core.brain.llm.latent_cortex.external_execution import (
                validate_external_execution_handoff,
            )

            validate_external_execution_handoff(
                raw_handoff,
                offer=external_execution_offer,
                cognitive_action_trace=raw_trace,
            )
        elif raw_handoff not in ({}, None):
            raise ValueError("worker emitted unoffered external execution handoff")
        action_policy_matches = True
        return action_policy_matches

    @staticmethod
    def _deep_reason_part_6(adaptive_plan, budget, config, contract_errors, quality_receipt, result, result_receipt, visible_objective):
        from .latent_cortex_service import (
            evaluate_latent_output,
        )

        if not contract_errors and adaptive_plan is not None:
            try:
                from core.brain.llm.latent_cortex.adaptive_compute import (
                    build_adaptive_execution_receipt,
                    validate_adaptive_execution_receipt,
                )

                adaptive_execution = build_adaptive_execution_receipt(
                        plan=adaptive_plan,
                        config=config,
                        budget=budget,
                        worker_receipt=result_receipt,
                )
                validate_adaptive_execution_receipt(adaptive_execution)
                result_receipt["adaptive_compute"] = adaptive_execution
            except (ImportError, TypeError, ValueError, OverflowError):
                contract_errors.append("adaptive_compute_execution_unproven")
        if not contract_errors:
            quality_receipt = evaluate_latent_output(
                result.get("text"),
                generated_tokens=result_receipt.get("decode_generated_tokens"),
                termination=result_receipt.get("decode_termination"),
                objective=visible_objective,
            )
            result_receipt["output_quality"] = quality_receipt
            result["receipt"] = result_receipt
        return quality_receipt

    def _deep_reason_part_7(self, action_policy_evidence, action_policy_matches, epistemic_state, result, result_receipt, selective_memory_result):
        if action_policy_matches and action_policy_evidence is not None:
            result_receipt["host_action_policy_evidence"] = dict(
                action_policy_evidence
            )
            result["receipt"] = result_receipt
        if epistemic_state is not None:
            epistemic_state_receipt = {
                "schema": epistemic_state.schema,
                "episode_id": epistemic_state.episode_id,
                "version": epistemic_state.version,
                "state_sha256": epistemic_state.state_sha256,
            }
            if selective_memory_result is not None:
                epistemic_state_receipt.update(
                    {
                        "memory_result_sha256": selective_memory_result.result_sha256,
                        "memory_evidence_ids": [
                            candidate.evidence_id
                            for candidate in selective_memory_result.candidates
                        ],
                    }
                )
            result_receipt["epistemic_state"] = epistemic_state_receipt
        raw_progress = result.get("progress")
        self._last_progress = dict(raw_progress) if isinstance(raw_progress, dict) else {}

    def _deep_reason_reason(self, contract_errors, host_incumbent, result, result_receipt):
        from .latent_cortex_service import (
            logger,
            record_degradation,
        )

        reason = "receipt_contract_failed:" + ",".join(contract_errors)
        # Report this condition when it APPEARS or CHANGES, not once
        # per turn.
        #
        # LIVE, 2026-08-10: the recurrent cortex declined every single
        # foreground turn with an identical contract failure — the
        # decode bridge is not wired on this path, so the refusal is
        # correct and it is also unchanging. Recording it per turn
        # opened a fresh incident every turn (INC-…-0003, -0005, -0006,
        # -0007 in one hour, each auto-resolving after 300s only to be
        # replaced), and each one pushed resilience toward the
        # depletion state that suppresses execution.
        #
        # This is exactly the case CONTRIBUTING/CLAUDE describe: log a
        # persistent, total condition at info and record a degradation
        # when it is new or has changed. _record_failure below still
        # increments the streak and retains the receipt, so nothing
        # about the refusal becomes invisible — only the duplicate
        # incident does.
        if reason == self._last_refusal:
            logger.info(
                "Recurrent latent cortex still declining on the same "
                "unchanged contract failure (streak=%d): %s",
                self._failure_streak + 1,
                reason,
            )
        else:
            record_degradation(
                "latent_cortex",
                RuntimeError(reason),
                action="refused to count incomplete latent episode as successful",
                severity="degraded",
            )
        failed = dict(result)
        failed.update(self._record_failure(reason))
        if host_incumbent is not None:
            failed["text"], failed["tokens"] = host_incumbent
        failed["receipt"] = result_receipt
        self._last_failure_receipt = result_receipt
        return failed

    def _deep_reason_part_9(self, action_transitions, budget, controller_decision, result, result_receipt, started):
        from .latent_cortex_service import (
            _controller_outcome,
            logger,
            record_degradation,
        )

        result["receipt"] = result_receipt
        # CP126 e1c09324 / 9b110bc5 / fc19e25e / 265b0fae. Everything this
        # facade verifies about the episode comes from the worker's own
        # receipt. The format checks are real and the internal-consistency
        # checks are real, but a dishonest or corrupted worker can submit a
        # self-consistent receipt and nothing here would know: the facade
        # and the worker are not separate trust domains.
        #
        # Rather than let "ok" imply more than it earns, every episode says
        # exactly what it proved and what it did not. A consumer promoting
        # an adapter, citing a reasoning gain, or trusting an answer's
        # provenance must read this block, not the boolean.
        result["attestation"] = self._attestation_disclosure(result_receipt)
        self._ok_episodes += 1
        self._failure_streak = 0
        self._last_refusal = ""
        self._last_success_at = time.time()
        # The grading path for cognition.effort. Until this existed the
        # control point was registered, recording, and permanently
        # unpromotable: nothing ever called note_grade(), so every effort
        # decision resolved UNOBSERVED. It reports the SAME independently
        # graded outcome the bandit is allowed to learn from — a verifier's
        # judgement of the answer — and nothing else. If no verifier graded
        # this episode (outcome_checked is False), nothing is reported and
        # the decision stays honestly UNOBSERVED rather than being taught
        # from latency, convergence, or the answer's own confidence.
        #
        # Deliberately outside the controller branch below: the effort
        # choice is made on every episode, so it is graded on every episode
        # a verifier actually graded, not only on the ones that also took
        # the execution-controller path.
        effort_episode_id = str(budget.get("ontogeny_episode") or "")
        if effort_episode_id:
            try:
                effort_score, effort_checked, _passed, _reason = _controller_outcome(
                    result_receipt.get("verifier_guidance")
                )
                if effort_checked:
                    from core.ontogeny.control_points import get_effort_resolver

                    get_effort_resolver().note_grade(
                        effort_episode_id, verified_score=effort_score
                    )
            except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
                record_degradation(
                    "latent_cortex_service", exc, severity="debug",
                    action="effort decision left ungraded for this episode",
                )
        # Controller learning accepts only an independently graded task
        # outcome. Candidate-local arithmetic, syntax, facet, and
        # grounding scores still steer this episode, but cannot become a
        # Wilson trial or teach the bandit that the whole answer was right.
        if controller_decision is not None:
            try:
                from core.brain.llm.latent_cortex.execution_controller import (
                    get_execution_controller,
                )

                verifier_evidence = result_receipt.get("verifier_guidance")
                (
                    best_score,
                    outcome_checked,
                    outcome_passed,
                    outcome_reason,
                ) = _controller_outcome(verifier_evidence)
                # CP126 3b3d44e8: the outcome must be bound to the DECISION
                # that produced it — a caller-asserted bucket/arm could
                # credit any arm, including recording a base execution as
                # a treatment.
                self._controller_outcome_recorded_for = str(
                    controller_decision.get("decision_id") or ""
                )
                outcome_recorded = get_execution_controller().record_outcome(
                    bucket=str(controller_decision.get("bucket") or ""),
                    arm=str(controller_decision.get("arm") or "base"),
                    verified_score=best_score,
                    success=outcome_passed,
                    checked=outcome_checked,
                    wall_clock_s=time.monotonic() - started,
                    decision_id=str(controller_decision.get("decision_id") or ""),
                )
                checked_action_transitions = [
                    row for row in action_transitions if row["checked"] is True
                ]
                action_outcomes_recorded = (
                    get_execution_controller().record_action_transitions(
                        checked_action_transitions
                    )
                    if checked_action_transitions
                    else False
                )
                result_receipt["execution_controller"] = {
                    **controller_decision,
                    "outcome_recorded": outcome_recorded,
                    "outcome_checked": outcome_checked,
                    "outcome_passed": (outcome_passed if outcome_checked else None),
                    "outcome_reason": outcome_reason,
                    "action_transitions_checked": len(checked_action_transitions),
                    "action_outcomes_recorded": action_outcomes_recorded,
                }
                result["receipt"] = result_receipt
            except (
                ImportError,
                AttributeError,
                RuntimeError,
                TypeError,
                ValueError,
                OSError,
            ) as exc:
                logger.debug("Controller outcome not recorded: %s", exc)
        # Identity consistency: the canonical self verifies the
        # conclusion (persona displacement, forbidden intentions, core
        # values). The verdict PRICES the broadcast — an inconsistent
        # thought must outcompete honestly, never silently erased.
        try:
            from core.self.identity_consistency import (
                check_identity_consistency,
            )

            result_receipt["identity_consistency"] = check_identity_consistency(
                str(result.get("text") or "")
            )
            result["receipt"] = result_receipt
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("Identity consistency check skipped: %s", exc)

    def _deep_reason_part_10(self, elapsed, result, result_receipt):
        from .latent_cortex_service import (
            logger,
        )

        self._last_failure_receipt = result_receipt
        # CP126 5879d2b5: the refusal receipt was BUILT here and then
        # thrown away — the raw client dict was returned, so a client
        # failure reached the caller with no stage, no timing and nothing
        # tying it to this call. Attach it.
        refusal = self._record_failure(
            str(result.get("reason") or "unknown"),
            stage=str(
                result_receipt.get("last_stage")
                or self._last_progress.get("stage")
                or "client"
            ),
            evidence={
                "input_token_count": result_receipt.get("input_token_count"),
                "elapsed_s": round(elapsed, 3),
            },
        )
        result.setdefault("refusal_receipt", refusal["refusal_receipt"])
        logger.info(
            "🧠 Latent episode refused/failed: %s (%.1fs) stage=%s "
            "input_tokens=%s timings=%s progress=%s",
            self._last_refusal,
            elapsed,
            result_receipt.get("last_stage") or self._last_progress.get("stage") or "unknown",
            result_receipt.get("input_token_count")
            or self._last_progress.get("input_tokens")
            or "unknown",
            result_receipt.get("stage_timings_s") or {},
            self._last_progress,
        )

