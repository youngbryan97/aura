"""One episode of latent reasoning, start to finish.

Lifted whole out of `engine`. Every name taken from it is imported at
CALL time: that module imports this one to build the class, and a test that
patches a name on it has to reach the code that reads it.
"""
from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .engine import (
        ComputeBudget,
        LatentReasoningResult,
    )


class _ReasonsThroughAnEpisode:
    """Lifted whole out of LatentCortexEngine; see engine.py."""

    def _reason_episode_part_1(self, capture_decode_logprobs, decode_sentence_grace_tokens, incumbent_artifact, sample_seed):
        if type(capture_decode_logprobs) is not bool:
            raise TypeError("capture_decode_logprobs must be boolean")
        # An incumbent artifact under a latent-owned policy is a contradiction
        # in the CALL, not a failure of the episode, and it needs nothing from
        # the runtime to detect.
        #
        # It used to be checked after the receipt existed, so `reason()` read
        # it as an admission failure and returned ok=False. Worse, a
        # serving-identity failure raised first and swallowed it, which meant
        # the guard against a latent-owned answer smuggling an ordinary-decode
        # artifact did not run at all on the path that would smuggle one.
        if incumbent_artifact is not None and (
            self.config.decode_incumbent_policy != "vanilla_incumbent"
        ):
            raise ValueError(
                "an incumbent artifact requires decode_incumbent_policy=vanilla_incumbent"
            )
        if decode_sentence_grace_tokens is not None and (
            type(decode_sentence_grace_tokens) is not int
            or not 0 <= decode_sentence_grace_tokens <= 4096
        ):
            raise ValueError("decode_sentence_grace_tokens must be null or inside [0, 4096]")
        if sample_seed is not None and (
            type(sample_seed) is not int or not 0 <= sample_seed <= 0xFFFFFFFF
        ):
            raise ValueError("sample_seed must be null or an integer inside [0, 2^32-1]")

    def _reason_episode_part_2(self, action_continuation_capture, action_continuation_capture_only, action_continuation_restore, action_continuation_restore_verified, action_continuation_runner_state, episode_id, memory_principal, nonparametric_memory_enabled):
        from .engine import (
            EpisodeReceipt,
            uuid,
        )

        if episode_id is not None and (
            not isinstance(episode_id, str)
            or not episode_id
            or len(episode_id) > 192
            or not episode_id[0].isalnum()
            or any(not (character.isalnum() or character in "._:/;=+-") for character in episode_id)
        ):
            raise ValueError("episode_id is not a valid campaign identifier")
        if type(action_continuation_capture_only) is not bool:
            raise TypeError("action_continuation_capture_only must be boolean")
        if type(nonparametric_memory_enabled) is not bool:
            raise TypeError("nonparametric_memory_enabled must be boolean")
        if not isinstance(memory_principal, str) or len(memory_principal) > 192:
            raise TypeError("memory_principal must be a string naming who this episode is for")
        continuation_requested = (
            action_continuation_capture is not None
            or action_continuation_restore is not None
            or action_continuation_runner_state is not None
            or action_continuation_capture_only
        )
        if continuation_requested:
            from core.brain.llm.latent_cortex.action_continuation import (
                ActionOpportunityContinuation,
            )

            if action_continuation_capture is not None and not callable(
                action_continuation_capture
            ):
                raise TypeError("action_continuation_capture must be callable")
            if action_continuation_restore_verified is not None and not callable(
                action_continuation_restore_verified
            ):
                raise TypeError("action_continuation_restore_verified must be callable")
            if (
                action_continuation_restore_verified is not None
                and action_continuation_restore is None
            ):
                raise ValueError("action_continuation_restore_verified requires a restore")
            if action_continuation_restore is not None and not isinstance(
                action_continuation_restore,
                ActionOpportunityContinuation,
            ):
                raise TypeError("action_continuation_restore has the wrong type")
            if not isinstance(action_continuation_runner_state, Mapping) or set(
                action_continuation_runner_state
            ) != {"durable_state", "rng_state"}:
                raise ValueError("action continuation requires exact runner state")
            if action_continuation_capture_only and action_continuation_capture is None:
                raise ValueError("capture-only continuation requires a capture callback")
        receipt = EpisodeReceipt(
            episode_id=(episode_id if episode_id is not None else uuid.uuid4().hex[:12])
        )
        from core.brain.llm.latent_cortex.recurrence_adapter import (
            RecurrenceAdapterActivation,
        )

        self._coda_adapter_activation = RecurrenceAdapterActivation()
        self._callback_faults = {}
        return receipt

    def _reason_episode_read_actual_serialized(self, budget, context_items, messages, policy_evidence, prompt, receipt, tokens, verifier):
        # Read the actual serialized prefix; model names and requested modes
        # do not establish which channel the tokenizer left open.
        from .engine import (
            hashlib,
        )

        decode_prefix = getattr(self.tokenizer, "decode", None)
        self._episode_native_thinking = bool(
            callable(decode_prefix)
            and str(decode_prefix(tokens)).rstrip().endswith("<think>")
        )
        if self._episode_native_thinking:
            receipt.flag("native_thinking_prefix_open")
        verification_objective = str(prompt or "")
        if not verification_objective and messages:
            for message in reversed(messages):
                if isinstance(message, dict) and message.get("role") == "user":
                    content = message.get("content")
                    if isinstance(content, str) and content.strip():
                        verification_objective = content
                        break
        self._admit_input_length(len(tokens), budget)
        encoded_tokens = json.dumps(tokens, separators=(",", ":"), allow_nan=False).encode("ascii")
        receipt.input_tokens_sha256 = hashlib.sha256(encoded_tokens).hexdigest()
        receipt.input_token_count = len(tokens)
        budget.bind_information(
            self._information_receipt(
                encoded_tokens=encoded_tokens,
                token_count=len(tokens),
                context_items=context_items,
                policy_evidence=policy_evidence,
                verifier=verifier,
            )
        )
        return encoded_tokens, verification_objective

    def _reason_episode_part_4(self, budget, decode_max_tokens, incumbent_artifact, receipt, tokens):
        receipt.decode_temperature = float(self.config.decode_temperature)
        receipt.decode_top_p = float(self.config.decode_top_p)
        receipt.decode_bridge_policy = self.config.decode_bridge_policy
        receipt.decode_incumbent_policy = self.config.decode_incumbent_policy
        receipt.verifier_probe_max_tokens = self.config.verifier_probe_max_tokens
        receipt.verifier_probe_contract = self.config.verifier_probe_contract
        receipt.decode_contract_required = self.config.decode_contract == "final_answer_v1"
        receipt.decode_contract_grace_tokens = (
            self.config.decode_contract_grace_tokens if receipt.decode_contract_required else 0
        )

        self.invariant.pre_episode()
        self._episode_invariant_armed = True
        receipt.checkpoint_fingerprint = self.invariant.file_receipt.get("fingerprint", "")
        receipt.checkpoint_fingerprint_method = self.invariant.file_receipt.get("method", "")
        receipt.checkpoint_file_count = int(self.invariant.file_receipt.get("files", 0) or 0)
        validated_incumbent = None
        if incumbent_artifact is not None:
            # The policy contradiction is refused at the call boundary above,
            # before anything runs; this is what still needs the runtime.
            if self.tokenizer is None:
                raise ValueError("an incumbent artifact requires the serving tokenizer")
            from core.brain.llm.latent_cortex.incumbent_artifact import (
                validate_incumbent_artifact,
            )

            validated_incumbent = validate_incumbent_artifact(
                incumbent_artifact,
                input_tokens=tokens,
                checkpoint_fingerprint=receipt.checkpoint_fingerprint,
                checkpoint_fingerprint_method=receipt.checkpoint_fingerprint_method,
                max_tokens=(
                    decode_max_tokens
                    if decode_max_tokens is not None
                    else self.config.decode_max_tokens
                ),
                n_layers=self.n_layers,
                decode=lambda values: self._decode_public_text(list(values), receipt=receipt),
            )
            receipt.incumbent_artifact = dict(validated_incumbent.receipt)
            budget.charge_layer_apps(
                int(validated_incumbent.receipt["compute"]["transformer_layer_apps"]),
                operation="bound_incumbent_generation",
            )

        failure_reason = ""
        return failure_reason, validated_incumbent

    def _reason_episode_ok_says_machinery(self, budget, episode_started, failure_reason, progress, receipt, verifier):
        # `ok` says the machinery ran. These two say what it established.
        receipt.verifier_identity = (
            f"{type(verifier).__module__}.{type(verifier).__qualname__}"
            if verifier is not None
            else ""
        )
        receipt.quality_verified = bool(
            verifier is not None
            and receipt.branch_selection_admitted
            and not receipt.has_flag("branch_verifier_skipped_budget")
        )
        receipt.gain_established = bool(
            receipt.quality_verified
            and receipt.fast_weight_verifier.get("decision")
            == "accepted_causal_improvement"
        )
        for channel, kind in sorted(self._callback_faults.items()):
            # Monitoring health is reported separately from model success: a
            # consumer that lost stage updates must not read the gap as an
            # authoritative absence of stages.
            receipt.flag(f"{channel}_callback_failed:{kind}")
        receipt.last_stage = "complete" if not failure_reason else receipt.last_stage
        receipt.stage_timings_s["total"] = round(
            max(0.0, time.monotonic() - episode_started),
            6,
        )
        self._emit_progress(
            progress,
            {
                "stage": "complete" if not failure_reason else "failed",
                "last_stage": receipt.last_stage,
                "elapsed_s": receipt.stage_timings_s["total"],
                "reason": failure_reason,
                "spent_layer_apps": int(budget.spent_layer_apps),
            },
        )
        receipt.budget = budget.to_receipt()
        self._flush_consolidation_export(receipt, failure_reason=failure_reason or "")
        from core.brain.llm.latent_cortex.causal_receipt import (
            build_causal_receipt,
        )

        receipt.causal_receipt = build_causal_receipt(receipt.to_dict())

    @staticmethod
    def _reason_episode_part_6(failure_reason, out_tokens, receipt):
        if not failure_reason and receipt.decode_termination not in {
            "eos",
            # The public answer contract completed: one FINAL_ANSWER JSON
            # object closed and parsed — the strongest completion signal a
            # contract task has (CP180).
            "contract_complete",
            # A bounded negative output remains valid scientific evidence.
            # The live service rejects it as product-incomplete.
            "token_limit_contract_incomplete",
            "token_limit",
            # The limit landed mid-sentence and sampling continued a few
            # model-chosen tokens to the natural boundary — a complete
            # answer, receipted under its own termination kind.
            "token_limit_sentence_grace",
            # Ran out of layer-app budget mid-decode, with tokens already
            # sampled. The comment just below spells out why a wall-clock
            # stop is accepted, and every word of it applies here: the
            # product-quality gate judges whether the text stands as an
            # answer, not which budget dimension ended sampling. Three
            # dimensions bound a decode — tokens, wall clock, layer
            # applications — and only two were listed, so the third killed
            # live turns with "decode_incomplete:budget_exhausted" and the
            # person got "I couldn't get to an answer I'd stand behind".
            # Exhausting before the first token is a different termination
            # and is deliberately NOT accepted.
            "budget_exhausted",
            # Time pressure ended decoding at a sentence boundary (the
            # wall-clock analogue of the token-limit grace). A time-bounded
            # stop has the same epistemic status as a token-bounded one:
            # the product-quality gate — terminal completeness, facet and
            # subject coverage — judges whether the text stands as an
            # answer, not the budget dimension that ended sampling.
            "wall_reserve_sentence_grace",
            # Raw "wall_reserve" is deliberately absent. It now means the
            # reserve was crossed with no sentence boundary reached — a known
            # fragment. The wind-down above emits the accepted kind when the
            # text actually ends somewhere.
            # A separately generated, exactly round-tripped repair cleared
            # the confidence-bound authority gate and replaced the ordinary
            # neural decode.
            "confidence_bound_replacement",
        }:
            failure_reason = f"decode_incomplete:{receipt.decode_termination}"
        if not failure_reason and not out_tokens:
            # An immediate EOS appends nothing and terminates as "eos", which
            # the acceptance set above reads as a complete decode. The episode
            # then returned ok with no answer in it. "The tokenizer said stop"
            # is a different claim from "here is the answer".
            receipt.flag("decode_produced_no_tokens")
            failure_reason = "decode_incomplete:no_tokens_generated"
        if receipt.params_unchanged is False:
            receipt.flag("checkpoint_invariant_violated")
            failure_reason = failure_reason or "checkpoint_invariant_violated"
        return failure_reason

    def _reason_episode(
        self,
        prompt: str | None = None,
        *,
        messages: list | None = None,
        token_ids: list[int] | None = None,
        budget: ComputeBudget | None = None,
        verifier: Callable[[str], float] | None = None,
        domain: str = "general",
        decode_max_tokens: int | None = None,
        ablate_slot: int | None = None,
        ablate_mode: str = "zero",
        cognitive_context: list | None = None,
        action_policy_evidence: dict[str, Any] | None = None,
        action_intervention: dict[str, Any] | None = None,
        action_intervention_consumption: dict[str, Any] | None = None,
        external_execution_offer: dict[str, Any] | None = None,
        cancel_check: Callable[[], bool] | None = None,
        progress: Callable[[dict], None] | None = None,
        capture_decode_logprobs: bool = False,
        decode_sentence_grace_tokens: int | None = None,
        sample_seed: int | None = None,
        incumbent_artifact: Any | None = None,
        episode_id: str | None = None,
        action_continuation_capture: Callable[[Any], None] | None = None,
        action_continuation_restore: Any | None = None,
        action_continuation_runner_state: Mapping[str, Any] | None = None,
        action_continuation_capture_only: bool = False,
        action_continuation_restore_verified: Callable[[str], None] | None = None,
        nonparametric_memory_enabled: bool = True,
        memory_principal: str = "",
    ) -> LatentReasoningResult:
        from .engine import (
            _FALL_THROUGH,
            _LATENT_PHASE_ERRORS,
            ComputeBudget,
            ComputeBudgetUnaffordable,
            UnknownActionStateApplicationError,
            _ActionContinuationCapturedError,
            _FastWeightCleanupError,
            _LatentEpisodeCancelledError,
            _public_reason,
            _reason_episode_part_1_1,
            build_evidence_snapshot,
            hashlib,
            record_degradation,
            validate_evidence_snapshot,
        )

        self._reason_episode_part_1(capture_decode_logprobs, decode_sentence_grace_tokens, incumbent_artifact, sample_seed)
        receipt = self._reason_episode_part_2(action_continuation_capture, action_continuation_capture_only, action_continuation_restore, action_continuation_restore_verified, action_continuation_runner_state, episode_id, memory_principal, nonparametric_memory_enabled)
        # Every probe decode inherits the episode's cancellation channel and
        # its cleanup reserve. Threading them through fifteen call sites is
        # how they went missing; the single-flight guard makes one place the
        # honest place to put them.
        self._episode_cancel_check = cancel_check
        self._episode_wall_reserve_forwards = 0
        self._staged_consolidation_export = None
        episode_started = time.monotonic()
        receipt.n_layers = self.n_layers
        receipt.recurrence_support = dict(self.recurrence_support)
        receipt.prelude_end = self.prelude_end
        receipt.coda_start = self.coda_start
        budget = budget or ComputeBudget()
        budget.bind_model(self.model)
        if decode_max_tokens is not None:
            if type(decode_max_tokens) is not int:
                raise TypeError("decode_max_tokens override must be an integer")
            if not 1 <= decode_max_tokens <= 8192:
                raise ValueError("decode_max_tokens override outside [1, 8192]")
        context_items = self._validate_cognitive_context(cognitive_context)
        policy_evidence = validate_evidence_snapshot(
            action_policy_evidence
            if action_policy_evidence is not None
            else build_evidence_snapshot(
                bucket=f"{str(domain or 'general')[:24]}|none|short|s:mid|u:mid",
                cells={},
            )
        )
        receipt.value_of_computation = {
            "schema": policy_evidence["schema"],
            "bucket": policy_evidence["bucket"],
            "snapshot_sha256": policy_evidence["snapshot_sha256"],
            "active": True,
        }
        normalized_execution_offer = None
        if external_execution_offer is not None:
            from core.brain.llm.latent_cortex.external_execution import (
                validate_external_execution_offer,
            )

            normalized_execution_offer = validate_external_execution_offer(external_execution_offer)
        normalized_action_intervention = None
        action_intervention_execution_claim = None
        if action_intervention is not None:
            from core.brain.llm.latent_cortex.action_intervention import (
                action_intervention_engine_request_sha256,
                claim_action_intervention_execution,
                validate_action_intervention,
                validate_action_intervention_objective,
            )

            normalized_action_intervention = validate_action_intervention(
                action_intervention,
                require_current_policy=True,
            )
            validate_action_intervention_objective(
                normalized_action_intervention,
                prompt=prompt,
                messages=messages,
                token_ids=token_ids,
            )
            if (
                decode_max_tokens is not None
                or capture_decode_logprobs
                or decode_sentence_grace_tokens is not None
            ):
                raise ValueError("action intervention does not permit direct decode overrides")
            engine_request_sha256 = action_intervention_engine_request_sha256(
                prompt=prompt,
                domain=str(domain),
                config=self.config,
                budget=budget,
                cognitive_context=context_items,
                action_policy_evidence=policy_evidence,
                external_execution_offer=normalized_execution_offer,
                verifier_present=verifier is not None,
                ablate_slot=ablate_slot,
                ablate_mode=ablate_mode,
            )
            if (
                engine_request_sha256
                != normalized_action_intervention["authority_payload"]["engine_request_sha256"]
            ):
                raise ValueError("action intervention engine request differs")
            if not isinstance(action_intervention_consumption, dict):
                raise ValueError("action intervention lacks a worker consumption event")
            action_intervention_execution_claim = claim_action_intervention_execution(
                normalized_action_intervention,
                action_intervention_consumption,
            )
        elif action_intervention_consumption is not None:
            raise ValueError("action intervention consumption lacks an intervention")
        # Payload validation is done, and every refusal above is a caller
        # contract violation that must stay loud — a tampered memory
        # authority especially. Everything past here depends on live runtime
        # state: the tokenizer, the compute budget, the checkpoint invariant.
        # Publishing the receipt is what tells the outer boundary to return a
        # receipted failure for those rather than let a bare exception out of
        # the one-call contract.
        self._episode_receipt = receipt
        tokens = self._encode(prompt, messages, token_ids)
        encoded_tokens, verification_objective = self._reason_episode_read_actual_serialized(budget, context_items, messages, policy_evidence, prompt, receipt, tokens, verifier)
        if sample_seed is None and self.config.decode_temperature > 0.0:
            # Derived from this episode's own commitment, so it is stable for
            # the same inputs and different across episodes. A fresh random
            # number would be neither, and global MLX randomness left the
            # answer unreproducible and untied to the state it came from.
            sample_seed = int.from_bytes(
                hashlib.sha256(
                    f"{receipt.episode_id}:{receipt.input_tokens_sha256}".encode()
                ).digest()[:4],
                "big",
            )
        metered_verifier = self._meter_verifier(verifier, budget)
        failure_reason, validated_incumbent = self._reason_episode_part_4(budget, decode_max_tokens, incumbent_artifact, receipt, tokens)
        continuation_captured_only = False
        out_tokens: list[int] = []
        decode_token_logprobs: list[float] = []
        answer_replacement_private: dict[str, Any] = {}
        transient_cleanup_registry: list[Any] = []
        try:
            try:
                (
                    out_tokens,
                    receipt,
                    answer_replacement_private,
                ) = self._latent_episode(
                    tokens,
                    budget,
                    metered_verifier,
                    domain,
                    receipt,
                    decode_max_tokens,
                    ablate_slot=ablate_slot,
                    ablate_mode=ablate_mode,
                    cognitive_context_items=context_items,
                    action_policy_evidence=policy_evidence,
                    action_intervention=normalized_action_intervention,
                    action_intervention_consumption=action_intervention_consumption,
                    action_intervention_execution_claim=(action_intervention_execution_claim),
                    external_execution_offer=normalized_execution_offer,
                    information_encoded_tokens=encoded_tokens,
                    information_verifier=verifier,
                    verification_objective=verification_objective,
                    cancel_check=cancel_check,
                    progress=progress,
                    episode_started=episode_started,
                    token_logprobs_out=(decode_token_logprobs if capture_decode_logprobs else None),
                    decode_sentence_grace_tokens=decode_sentence_grace_tokens,
                    sample_seed=sample_seed,
                    incumbent_artifact=validated_incumbent,
                    transient_cleanup_registry=transient_cleanup_registry,
                    action_continuation_capture=action_continuation_capture,
                    action_continuation_restore=action_continuation_restore,
                    action_continuation_runner_state=(
                        dict(action_continuation_runner_state)
                        if action_continuation_runner_state is not None
                        else None
                    ),
                    action_continuation_capture_only=action_continuation_capture_only,
                    action_continuation_restore_verified=(action_continuation_restore_verified),
                    nonparametric_memory_enabled=nonparametric_memory_enabled,
                    memory_principal=memory_principal,
                )
                if receipt.answer_replacement.get("decision") == "abstain":
                    # Abstention may end the episode only when the latent lane
                    # owns the output. Under vanilla_incumbent the answer on
                    # the table IS ordinary decode, and failing the episode
                    # discards it -- ok=False, zero tokens, empty text.
                    #
                    # This is the SECOND abstain path; guarding only the one
                    # that sets decode_termination left this one live. Measured
                    # on the 32B: four of fourteen cells returned empty text
                    # against receipts reporting 278-560 generated tokens and
                    # clean eos/token_limit terminations, and three of those
                    # four were tasks ordinary decode got RIGHT. The reason was
                    # known_refutation_has_no_dominant_repair -- the verifier
                    # judged the baseline refuted when it was correct, and a
                    # false refutation then threw the correct answer away.
                    #
                    # Serving the incumbent is weakly better in every case:
                    # equal when the refutation is right (vanilla was wrong
                    # anyway) and strictly better when it is wrong.
                    if self.config.decode_incumbent_policy == "latent":
                        failure_reason = "answer_replacement_abstained"
                    else:
                        receipt.flag("answer_replacement_abstention_declined_under_incumbent")
            except _FastWeightCleanupError as exc:
                record_degradation(
                    "latent_cortex",
                    exc,
                    action="refused fallback decode and requested resident-worker recycle",
                    severity="critical",
                )
                failure_reason = "fast_weight_cleanup_unproven"
            except _LatentEpisodeCancelledError:
                receipt.flag("soft_cancelled")
                receipt.halting_reason = receipt.halting_reason or "soft_cancelled"
                failure_reason = "soft_cancelled"
            except _ActionContinuationCapturedError:
                continuation_captured_only = True
                receipt.last_stage = "action_state_captured"
                receipt.halting_reason = "action_state_captured_before_first_action"
            except UnknownActionStateApplicationError:
                raise
            except _LATENT_PHASE_ERRORS as exc:
                fallback_permitted = (
                    self.config.allow_vanilla_fallback and normalized_action_intervention is None
                )
                record_degradation(
                    "latent_cortex",
                    exc,
                    action=(
                        "served vanilla decode with honest fallback receipt"
                        if fallback_permitted
                        else "failed the full-stack episode without replacing it with vanilla decode"
                    ),
                )
                receipt.halting_reason = receipt.halting_reason or "latent_phase_error"
                if not fallback_permitted:
                    receipt.flag(
                        "campaign_vanilla_fallback_forbidden"
                        if normalized_action_intervention is not None
                        else "vanilla_fallback_disabled"
                    )
                    # A refusal to spend is not a broken decode. Kept as its
                    # own class so latent_owner_exhausted() can release the
                    # resident model and the ordinary path can answer the turn,
                    # instead of the person getting "I couldn't get to an
                    # answer I'd stand behind" because an optional enhancement
                    # was priced out.
                    failure_reason = _public_reason(
                        "latent_budget_declined"
                        if isinstance(exc, ComputeBudgetUnaffordable)
                        else "latent_phase_failed",
                        exc,
                    )
                elif isinstance(exc, MemoryError):
                    # The fallback allocates a fresh cache and re-runs the
                    # whole prefill and decode. Under host or device
                    # exhaustion that is the one thing that cannot help: it
                    # asks for more of what just ran out, and it can take the
                    # resident worker down with it. Release what this episode
                    # holds, say so, and let the caller recycle.
                    receipt.flag("fallback_refused_memory_exhaustion")
                    self._release_episode_memory()
                    failure_reason = "latent_memory_exhausted"
                elif (
                    receipt.fast_weights_applied
                    or receipt.fast_weights_attach_attempted
                ) and receipt.fast_weights_erased is not True:
                    receipt.flag("fallback_refused_unproven_model_state")
                    failure_reason = "fast_weight_cleanup_unproven"
                else:
                    receipt.flag(f"fallback_vanilla:{type(exc).__name__}")
                    try:
                        decode_token_logprobs.clear()
                        if (
                            self.config.decode_incumbent_policy == "vanilla_incumbent"
                            and self._episode_incumbent_tokens is not None
                        ):
                            out_tokens = list(self._episode_incumbent_tokens)
                            decode_termination = self._episode_incumbent_termination
                            if capture_decode_logprobs:
                                decode_token_logprobs.extend(
                                    self._episode_incumbent_logprobs
                                )
                            receipt.flag("fallback_reused_materialized_incumbent")
                        else:
                            cache = self._fresh_cache()
                            _, tail_logits = self._prefill(tokens, cache, budget)
                            out_tokens, decode_termination = self._decode(
                                cache,
                                budget,
                                tail_logits,
                                max_tokens=decode_max_tokens,
                                cancel_check=cancel_check,
                                progress=progress,
                                token_logprobs_out=(
                                    decode_token_logprobs
                                    if capture_decode_logprobs
                                    else None
                                ),
                                sentence_grace_tokens=decode_sentence_grace_tokens,
                                sample_seed=sample_seed,
                            )
                        self._record_decode_discipline(
                            receipt,
                            requested_tokens=(
                                decode_max_tokens
                                if decode_max_tokens is not None
                                else self.config.decode_max_tokens
                            ),
                            generated_tokens=len(out_tokens),
                            termination=decode_termination,
                        )
                        if decode_termination.startswith("budget_"):
                            receipt.flag(f"decode_{decode_termination}")
                    except _LatentEpisodeCancelledError:
                        receipt.flag("soft_cancelled")
                        receipt.halting_reason = receipt.halting_reason or "soft_cancelled"
                        failure_reason = "soft_cancelled"
                    except _LATENT_PHASE_ERRORS as inner_exc:
                        record_degradation(
                            "latent_cortex",
                            inner_exc,
                            action=("reported failed episode after vanilla fallback also failed"),
                            severity="degraded",
                        )
                        failure_reason = f"latent_and_fallback_failed:{inner_exc}"
        finally:
            for transient_ledger in transient_cleanup_registry:
                try:
                    transient_ledger.abort_all()
                except _LATENT_PHASE_ERRORS as exc:
                    receipt.flag("transient_constraint_cleanup_unproven")
                    failure_reason = failure_reason or "transient_constraint_cleanup_unproven"
                    record_degradation(
                        "latent_cortex",
                        exc,
                        action="refused the episode because transient authority cleanup failed",
                        severity="critical",
                    )
            try:
                receipt.params_unchanged = self.invariant.post_episode(receipt)
                self._episode_invariant_armed = False
            except _LATENT_PHASE_ERRORS as exc:
                receipt.params_unchanged = False
                receipt.flag(f"checkpoint_post_probe_failed:{type(exc).__name__}")
                record_degradation(
                    "latent_cortex",
                    exc,
                    action="refused output because the post-episode invariant probe failed",
                    severity="critical",
                )
        self._reason_episode_ok_says_machinery(budget, episode_started, failure_reason, progress, receipt, verifier)
        _left = _reason_episode_part_1_1(self, answer_replacement_private, continuation_captured_only, decode_token_logprobs, failure_reason, out_tokens, receipt)
        if _left is not _FALL_THROUGH:
            return _left

