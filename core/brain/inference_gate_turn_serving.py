"""What a turn decides once the prompt exists and the lane is chosen.

Lifted whole out of `inference_gate`. Every name taken from it is imported at
CALL time: that module imports this one to build the class, and a test that
patches a name on it has to reach the code that reads it.
"""
from __future__ import annotations

import logging
from typing import Any


class _ServesTheTurn:
    """Lifted whole out of InferenceGate; see inference_gate.py."""

    def _generate_with_metadata_sink_use_compact_foreground_context(
        self,
        context: dict[str, Any] | None,
        deep_handoff: bool,
        initial_visible_user_prompt: Any,
        is_background: bool,
        origin: str,
        requested_tier: Any,
    ) -> tuple[Any, Any]:
        from .inference_gate import (
            _record_inference_degradation,
        )

        use_compact_foreground_context = self._should_use_compact_foreground_context(
            origin,
            requested_tier,
            deep_handoff=deep_handoff,
            is_background=is_background,
            prompt=initial_visible_user_prompt,
            context=context,
        )
        provided_messages = context.get("messages")
        if provided_messages is not None and not isinstance(provided_messages, list):
            # Dropping it silently is what let a malformed payload become a
            # system-only generation: the merge iterated nothing, inserted the
            # system message, and sent a prompt with no user turn in it. The
            # caller's `prompt` still carries the request, so the turn is
            # served — but the caller is told its payload was not used.
            _record_inference_degradation(
                TypeError(
                    f"context['messages'] is {type(provided_messages).__name__}, not a list"
                ),
                action="ignored a malformed prebuilt message payload and used the prompt instead",
                extra={"origin": str(origin or "")},
            )
            context["prebuilt_messages_rejected"] = "not_a_list"
            provided_messages = None
        if not isinstance(provided_messages, list):
            provided_messages = None
        return provided_messages, use_compact_foreground_context

    @staticmethod
    async def _generate_with_metadata_sink_task_grounding_blocks(
        context: dict[str, Any] | None,
        contract_grounding_blocks: list[str],
        isolated_generation_contract: Any,
        living_mind_context: Any,
        prompt_contract_block: Any,
        somatic_temperature: Any,
        visible_user_prompt: Any,
    ) -> tuple[list[str], list[str]]:
        from .inference_gate import (
            _INFERENCE_RECOVERABLE_ERRORS,
            _attach_the_present_moment,
            logger,
            record_degradation,
        )

        task_grounding_blocks: list[str] = []
        ambient_grounding_blocks: list[str] = []
        # The response contract describes THIS turn — its reason label and the
        # current local date both change per turn — so it belongs beside the
        # turn, not in the persistent system prompt. Measured live: it landed at
        # token 125 and divergence began at "## RESPONSE CONTRACT\n- Reason:
        # compound_prompt\n- Current local date: ...", stranding 3,884 tokens of
        # conversation behind it (3% reused).
        if prompt_contract_block and not isolated_generation_contract:
            contract_grounding_blocks.append(prompt_contract_block)
        # Current mind state changes independently of identity and policy. It
        # belongs with this turn's evidence, never inside the stable system
        # prefix. The old path inserted it above and then inserted it a second
        # time into prebuilt messages below. Besides presenting one source twice,
        # that made every affect tick invalidate the conversation's KV prefix.
        if living_mind_context and not isolated_generation_contract:
            ambient_grounding_blocks.append(living_mind_context)
        await _attach_the_present_moment(
            ambient_grounding_blocks=ambient_grounding_blocks,
            isolated_generation_contract=isolated_generation_contract,
            recent_actions_already_grounded=bool(
                context.get("recent_actions_already_grounded", False)
            ),
            task_grounding_blocks=task_grounding_blocks,
            visible_user_prompt=visible_user_prompt,
        )
        # Keep prompt growth aligned with the actual local model context window
        # instead of assuming 128k+ headroom on the primary Qwen lane.

        # ── Somatic narrative: brief felt-state line in the system prompt ────────
        if somatic_temperature is not None and not isolated_generation_contract:
            try:
                from core.affect.affective_circumplex import get_circumplex

                _soma_narrative = get_circumplex().describe()
                if _soma_narrative:
                    # Felt state changes on every tick; it travels with the rest
                    # of the volatile grounding, after the conversation.
                    ambient_grounding_blocks.append(
                        f"## SOMATIC STATE\n{_soma_narrative}"
                    )
            except _INFERENCE_RECOVERABLE_ERRORS as _exc:
                record_degradation(
                    "inference_gate",
                    _exc,
                    severity="warning",
                    action="continued without somatic-state prompt section",
                )
                logger.debug("Suppressed Exception: %s", _exc)
        return ambient_grounding_blocks, task_grounding_blocks

    @staticmethod
    def _generate_with_metadata_sink_architecture_self_awareness(
        context: dict[str, Any] | None,
        contract_grounding_blocks: list[str],
        isolated_generation_contract: Any,
        prompt_user_facing: bool,
        task_grounding_blocks: Any,
        visible_user_prompt: Any,
    ) -> Any:
        # ── Architecture Self-Awareness: inject relevant subsystem context ──────
        # Only for user-facing requests that mention architecture/code keywords.
        from .inference_gate import (
            _INFERENCE_RECOVERABLE_ERRORS,
            conversation_reliability_system_block,
            logger,
            record_degradation,
        )

        if prompt_user_facing and not isolated_generation_contract:
            try:
                import re as _re

                _arch_triggers = _re.compile(
                    r"\b(how|explain|what|which|where|why|trace|show|describe)\b.{0,60}"
                    r"\b(module|subsystem|file|class|method|function|work|does|handles|manages|routes|sends|wires)\b",
                    _re.IGNORECASE,
                )
                if _arch_triggers.search(visible_user_prompt):
                    from core.self.architecture_index import get_architecture_index

                    arch_excerpt = get_architecture_index().query(
                        visible_user_prompt,
                        max_results=3,
                    )
                    if arch_excerpt:
                        # The excerpt depends on this question, so it travels
                        # with turn-local grounding. Putting it in the stable
                        # system prefix invalidates cached conversation tokens
                        # when the next question is about another subsystem.
                        task_grounding_blocks.append(str(arch_excerpt))
            except _INFERENCE_RECOVERABLE_ERRORS as _ae:
                record_degradation(
                    "inference_gate",
                    _ae,
                    severity="warning",
                    action="continued without architecture self-awareness excerpt",
                )
                logger.debug("ArchIndex injection skipped: %s", _ae)
            contract_grounding_blocks.append(
                conversation_reliability_system_block(visible_user_prompt)
            )
        history = context.get("history", [])
        return history

    def _generate_with_metadata_sink_messages(
        self,
        context: dict[str, Any] | None,
        deep_probe_context: bool,
        foreground_profile: Any,
        is_background: bool,
        messages: Any,
        visible_user_prompt: Any,
    ) -> tuple[Any, str]:
        messages = self._compact_prebuilt_messages(
            messages,
            history_limit=(
                4
                if is_background
                else self._foreground_prebuilt_history_limit(
                    visible_user_prompt,
                    context,
                    deep_probe=deep_probe_context,
                )
            ),
            deep_probe=deep_probe_context,
            budget_profile=foreground_profile,
            current_user_content=visible_user_prompt,
        )
        # The compacted message set is now AUTHORITATIVE. Turn-local mind
        # context and reliability guidance are attached below, after this
        # compaction, as one bounded grounding message.
        #
        # `system_prompt` is a separate identity/policy string that grew
        # independently and is never compacted. It is still handed to the
        # client alongside these
        # messages, and the client merges a separately-passed system_prompt
        # into messages[0] — so it silently undid every compaction above.
        # Measured live: a 2,399-char compacted system message reached the
        # worker at 106,861 chars, turning a 278-char question into a
        # 27,129-token prefill (a 384:1 scaffold-to-request ratio) that
        # could not produce a first token inside the turn budget. None of
        # it was visible, because the prompt plan logs the compacted
        # messages and the re-inflation happens after that.
        #
        # The compacted structured messages now carry all system policy.
        # Passing a scalar copy would let the client merge the unbounded
        # pre-compaction prompt back into the first system message.
        system_prompt = ""
        return messages, system_prompt

    def _generate_with_metadata_sink_volatile_grounding_rides(
        self,
        ambient_grounding_blocks: Any,
        context: dict[str, Any] | None,
        contract_grounding_blocks: list[str],
        messages: Any,
        morpho_kwargs: Any,
        system_prompt: Any,
        task_grounding_blocks: Any,
    ) -> tuple[Any, Any]:
        # Volatile grounding rides LAST, behind the conversation, so the KV
        # prefix covering the history survives from one turn to the next.
        # Appended after compaction on purpose: compaction rewrites the history
        # it is given, and this block must not be trimmed away — it is the
        # read-not-inferred ground truth (clock, receipts, felt state) that
        # stops her narrating a present she was never given.
        from .inference_gate import (
            _refresh_volatile_grounding,
            logger,
        )

        has_volatile_grounding = bool(
            contract_grounding_blocks
            or task_grounding_blocks
            or ambient_grounding_blocks
        )
        messages, system_prompt = _refresh_volatile_grounding(
            ambient_grounding_blocks=ambient_grounding_blocks,
            context=context,
            contract_grounding_blocks=contract_grounding_blocks,
            has_volatile_grounding=has_volatile_grounding,
            messages=messages,
            self=self,
            system_prompt=system_prompt,
            task_grounding_blocks=task_grounding_blocks,
        )
        # Cache policy is not a caller preference.
        #
        # morpho_kwargs is populated from `context` early, then several
        # contracts (strict proof, operator evidence, health probe) set
        # context["disable_prompt_cache"] = True LATER — after the copy. A
        # caller that passed disable_prompt_cache=False therefore kept its
        # False in the kwargs that actually reach the worker, and an exact-cold
        # prompt contract silently ran on reused KV. Re-sync here, once, after
        # every contract has had its say: policy wins.
        for _cache_key in ("disable_prompt_cache", "clear_prompt_cache"):
            if bool(context.get(_cache_key, False)):
                if not bool(morpho_kwargs.get(_cache_key, False)):
                    logger.debug(
                        "Cache policy overrides caller %s=%r for this contract.",
                        _cache_key,
                        morpho_kwargs.get(_cache_key),
                    )
                morpho_kwargs[_cache_key] = True
        return messages, system_prompt

    @staticmethod
    def _generate_with_metadata_sink_authority_block_moves(
        context: dict[str, Any] | None,
        max_tokens: Any,
        messages: Any,
        system_prompt: Any,
    ) -> tuple[list[Any], Any]:
        # WHICH authority block moves. The scaffold total said 1809 on one turn
        # and 1817 on the next, which is enough to make the merged front system
        # message a different token sequence and cost the whole conversation
        # its prompt-cache prefix — 17.7s of a 22s turn. A total cannot say
        # which block did it; a per-block digest can.
        from .inference_gate import (
            _observable_dispatch_markers,
            hashlib,
            logger,
        )

        if logger.isEnabledFor(logging.INFO):
            _blocks = [
                (len(str(msg.get("content", "") or "")),
                 hashlib.sha256(
                     str(msg.get("content", "") or "").encode("utf-8", "replace")
                 ).hexdigest()[:8],
                 str(msg.get("content", "") or "")[:48].replace("\n", "⏎"))
                for msg in messages
                if str(msg.get("role", "")).strip().lower() == "system"
            ]
            if _blocks:
                logger.info(
                    "🧩 [PROMPT BLOCKS] %s",
                    " | ".join(f"{n}c {d} {h!r}" for n, d, h in _blocks),
                )
        # The separately-passed system_prompt is merged into messages[0] at the
        # client boundary, so it is part of the prefill even though it is not in
        # `messages` here. Leaving it out of this line is how a 106,861-char
        # re-inflation stayed invisible behind a plan that reported 4,479.
        # Which grounding blocks actually survived to the worker. Attachment was
        # already logged at the builder and the block still never arrived, so
        # the only useful signal is presence in the final text.
        _grounded = [
            name
            for name, marker in (
                ("present", "## PRESENT MOMENT"),
                ("instruments", "## YOUR OWN INSTRUMENTS"),
                ("receipts", "## WHAT YOU ACTUALLY JUST DID"),
                # Grounding that cannot be seen cannot be verified — the file
                # block spent a day being built into a prompt nobody sent.
                #
                # DERIVED from the registry, never hand-listed. Written out by
                # hand, this list immediately drifted: screen and beliefs were
                # registered as observables and left out here, so a screen
                # reading that WAS taken reported as not surviving, and an hour
                # went into looking for a delivery bug that did not exist.
                *_observable_dispatch_markers(),
            )
            if marker in str(system_prompt or "")
            or any(marker in str(msg.get("content", "") or "") for msg in messages)
        ]
        # FINAL word on the budget for an execution turn.
        #
        # The earlier raise fired ("raising the reply budget 288 -> 1024") and
        # was then overwritten by the compact-foreground path, so the caller
        # still asked for 288 — "Foreground starvation floor raised budget
        # 284->288 (caller asked 288)" — and a multi-step JSON plan cannot be
        # written in 288 tokens. Applied here, immediately before dispatch,
        # after every other budget computation has had its say.
        if bool(context.get("desktop_execution_contract", False)):
            _plan_floor_final = 1024
            if int(max_tokens or 0) < _plan_floor_final:
                logger.info(
                    "🖥️ [PLAN BUDGET] Execution turn: %s → %d tokens at dispatch.",
                    max_tokens,
                    _plan_floor_final,
                )
                max_tokens = _plan_floor_final
                context["max_tokens"] = max_tokens
        return _grounded, max_tokens

    @staticmethod
    def _generate_with_metadata_sink_part_24(
        _answer_floor_final: Any,
        context: dict[str, Any] | None,
        initial_visible_user_prompt: Any,
        max_tokens: Any,
    ) -> tuple[int, Any]:
        from .inference_gate import (
            logger,
        )

        if 0 < _answer_floor_final and int(max_tokens or 0) < _answer_floor_final:
            logger.info(
                "🧠 [ANSWER BUDGET] Answer turn: %s → %d tokens at dispatch.",
                max_tokens,
                _answer_floor_final,
            )
            max_tokens = _answer_floor_final
            context["max_tokens"] = max_tokens

        # A deadline that cannot deliver the budget the same request just
        # computed is two derived numbers contradicting each other, and
        # neither side could see the other.
        #
        # LIVE, 2026-08-27: the floor asked for 896 tokens and the
        # deliberate lane allowed about 150 seconds. The observed decode
        # rate made 896 tokens roughly 150 seconds of decoding on its own,
        # so the generation was cut mid-thought every time and the turn
        # served nothing. Raising the budget alone made it worse: 1,792
        # tokens were granted and the clock ended it at 43 seconds.
        #
        # The extension is bounded by the two measured quantities that
        # caused it — the floor and the observed rate — so there is no
        # invented number here and no open-ended wait. An unmeasured rate
        # extends nothing.
        # A turn that has to go and fetch something spends a whole
        # generation on the call before the answer is even started.
        #
        # LIVE, 2026-08-28: a diagnosis turn was offered the right tool,
        # spent forty-five seconds emitting one call, and the request
        # deadline expired fifty seconds later with nothing said about what
        # came back. The clock covered one generation and the turn needed
        # two.
        _generations = 1
        try:
            from core.intent.capability_selection import (
                points_at_something_real,
            )

            if points_at_something_real(initial_visible_user_prompt):
                _generations = 2
        except (ImportError, AttributeError, OSError, TypeError, ValueError):
            _generations = 1
        return _generations, max_tokens

    def _generate_with_metadata_sink__decode_s(
        self,
        _tokens_to_pay_for: Any,
        messages: Any,
        system_prompt: Any,
        model: str='',
    ) -> tuple[Any, Any]:
        # The rate belongs to the model. `_seconds_to_decode` has taken one
        # since the 9B's readings aborted three generations on the 27B, and
        # both call sites in this file passed nothing — so the parameter that
        # names the fix was never given the thing it needed.
        from .inference_gate import (
            _seconds_to_decode,
            _seconds_to_read,
        )

        _decode_s = _seconds_to_decode(_tokens_to_pay_for, model)
        # Reading the prompt is the other half of a generation, and
        # on this hardware it is the larger half. A turn was given time
        # to SAY its answer and none to read the question.
        _prompt_chars_for_clock = len(str(system_prompt or "")) + sum(
            len(str((msg or {}).get("content") or ""))
            for msg in (messages or [])
            if isinstance(msg, dict)
        )
        _read_s = _seconds_to_read(_prompt_chars_for_clock)
        # And what the worker that will serve this says, which is the
        # number it will cancel itself by.
        #
        # A percentile over past readings cannot follow a rate that
        # halves under memory pressure, and the worker measures its
        # own. LIVE 2026-09-04, one line apart: "the prompt takes
        # about 2s to read", granting 25 seconds, and "a 2867-char
        # prompt takes about 8.8s to read at 82 tok/s", needing 26.3.
        # Cancelled at 25, every user-facing turn, with the runtime
        # healthy throughout.
        _worker_says = 0.0
        _asking = getattr(self, "_mlx_client", None)
        _knows = getattr(_asking, "least_time_to_read", None)
        if callable(_knows):
            try:
                _worker_says = float(_knows(_prompt_chars_for_clock) or 0.0)
            except (TypeError, ValueError):
                # not a failure: a rate that will not parse is not one.
                _worker_says = 0.0
        _read_s = max(_read_s, _worker_says)
        return _decode_s, _read_s

    @staticmethod
    def _generate_with_metadata_sink_part_26(
        _decode_s: Any,
        _generations: Any,
        _needed: Any,
        _read_s: Any,
        _reserve_the_worker_adds: Any,
        _tokens_to_pay_for: Any,
        max_tokens: Any,
        timeout_val: Any,
    ) -> tuple[Any, Any]:
        from .inference_gate import (
            _DELIVERY_MARGIN_S,
            logger,
        )

        logger.info(
            "🧠 [ANSWER CLOCK] %d tokens (%d asked + %d reserve the "
            "worker adds) decode in about %.0fs and the prompt takes "
            "about %.0fs to read, at the measured rates, and this turn "
            "needs %d of them; deadline %.0fs → %.0fs.",
            _tokens_to_pay_for,
            max_tokens,
            _reserve_the_worker_adds,
            _decode_s,
            _read_s,
            _generations,
            float(timeout_val),
            _needed,
        )
        # Never past the ceiling the wait outside this one
        # uses. A deadline of 557 seconds inside a wait that
        # gives up at 480 is two numbers disagreeing again,
        # with the outer one winning silently.
        from core.runtime.response_policy import (
            USER_FACING_COMPLETION_DEADLINE_MAX_S,
        )

        _cap = float(USER_FACING_COMPLETION_DEADLINE_MAX_S)
        # Forecasts inform progress reporting. They do not
        # authorize substituting a less capable cortex.
        timeout_val = min(_cap, _needed)
        primary_timeout = max(8.0, timeout_val - _DELIVERY_MARGIN_S)
        return primary_timeout, timeout_val

    def _generate_with_metadata_sink_serving_lane(
        self,
        _grounded: Any,
        context: dict[str, Any] | None,
        initial_visible_user_prompt: Any,
        max_tokens: Any,
        messages: Any,
        morpho_kwargs: Any,
        system_prompt: Any,
    ) -> Any:
        from .inference_gate import (
            estimate_context_tokens,
            get_active_cortex_serving_limits,
            logger,
        )

        serving_lane = self._cortex_serving_lane(
            initial_visible_user_prompt,
            context,
            input_tokens=(
                estimate_context_tokens(str(system_prompt or ""))
                + sum(
                    estimate_context_tokens(str(message.get("content") or "")) + 12
                    for message in messages
                    if isinstance(message, dict)
                )
            ),
        )
        serving_limits = get_active_cortex_serving_limits()
        if serving_limits is not None and serving_limits.qualified:
            lane_limits = serving_limits.lane(serving_lane)
            if lane_limits is not None:
                admitted_tokens = min(max_tokens, lane_limits.max_output_tokens)
                if admitted_tokens < max_tokens:
                    logger.info(
                        "🧠 [SERVING PROFILE] %s output ceiling reduced %d→%d "
                        "(profile=%s).",
                        serving_lane,
                        max_tokens,
                        admitted_tokens,
                        serving_limits.profile_sha256[:12],
                    )
                max_tokens = max(1, admitted_tokens)
                context["max_tokens"] = max_tokens
                context["cortex_serving_lane"] = serving_lane
                context["cortex_serving_profile_sha256"] = (
                    serving_limits.profile_sha256
                )
                context["cortex_serving_profile_source"] = serving_limits.source
        morpho_kwargs["serving_lane"] = serving_lane

        logger.info(
            "🧭 [GROUNDING] survived to dispatch: %s (sys_prompt=%d)",
            ",".join(_grounded) or "NONE",
            len(str(system_prompt or "")),
        )
        return max_tokens

    @staticmethod
    def _generate_with_metadata_sink_part_28(
        max_tokens: Any,
        messages: Any,
        origin: str,
        prompt_chars: Any,
        prompt_mode: Any,
        request_chars: Any,
        scaffold_chars: Any,
        system_prompt: Any,
    ) -> None:
        from .inference_gate import (
            logger,
        )

        logger.info(
            "🧠 [ZENITH] Prompt plan: mode=%s messages=%d chars=%d "
            "(scaffold=%d request=%d ratio=%.1fx sys_prompt=%d) "
            "origin=%s max_tokens=%d",
            prompt_mode,
            len(messages),
            prompt_chars,
            scaffold_chars,
            request_chars,
            (scaffold_chars / request_chars) if request_chars else float("inf"),
            len(str(system_prompt or "")),
            origin or "unknown",
            max_tokens,
        )
        # What the scaffold IS, when it dwarfs the question.
        #
        # A ratio is a number nobody can act on. Eight thousand characters of
        # scaffold against two hundred and fifty of question is the shape of a
        # real defect, and the log said only that it was thirty-two to one —
        # so which part of it was eight thousand characters could not be found
        # without adding this line first.
        if request_chars and scaffold_chars > (8 * request_chars):
            logger.info(
                "🧠 [ZENITH] Scaffold breakdown: %s",
                "; ".join(
                    f"{str(msg.get('role', '?'))}={len(str(msg.get('content', '') or ''))}"
                    f":{str(msg.get('content', '') or '')[:70]!r}"
                    for msg in messages
                    if isinstance(msg, dict)
                ),
            )

    def _generate_with_metadata_sink__foreground_cap(
        self,
        context: dict[str, Any] | None,
        initial_visible_user_prompt: Any,
        morpho_kwargs: Any,
        visible_user_prompt: Any,
    ) -> None:
        from .inference_gate import (
            _GENERATE_EXPLICIT_KWARGS,
        )

        _foreground_floor, _foreground_cap, foreground_loops = (
            self._foreground_compute_profile(initial_visible_user_prompt)
        )
        foreground_profile = self._foreground_prompt_profile(
            visible_user_prompt,
            context,
        )
        # Every _generate_with_client call site passes these EXPLICITLY and also
        # splats **morpho_kwargs, so any overlap is a guaranteed TypeError —
        # "got multiple values for keyword argument" — which fails the
        # inference_gate closed and reaches the person as user_cycle_no_response:
        # the engine returning nothing at all, in two seconds, while ordinary
        # conversation through the same engine keeps working. One added key did
        # exactly that to every desktop turn.
        #
        # Scrubbed here rather than trusted to every future writer: the explicit
        # argument is the authority, and a duplicate in the splat can only ever
        # be the same value or a bug.
        for _reserved in _GENERATE_EXPLICIT_KWARGS:
            morpho_kwargs.pop(_reserved, None)
        morpho_kwargs.setdefault("clean_user_surface_contract", True)
        morpho_kwargs.setdefault(
            "user_surface_validation_prompt",
            initial_visible_user_prompt or visible_user_prompt,
        )
        morpho_kwargs.setdefault(
            "clean_user_surface_recurrent_loops",
            foreground_loops,
        )
        morpho_kwargs.setdefault(
            "clean_user_surface_steering_alpha",
            0.35 if foreground_profile == "extended" else 0.25,
        )

    @staticmethod
    def _generate_with_metadata_sink_say_quality_check(
        local_label: Any,
        primary_surface_receipt: Any,
    ) -> tuple[Any, ...]:
        # Say WHICH quality check rejected the text.
        #
        # This refusal is the last step before the person gets
        # "I couldn't get to an answer I'd stand behind", and it
        # logged only that retries were exhausted. The reasons
        # were computed by _surface_quality_failure_reasons,
        # carried on the receipt as surface_quality_gate_reasons,
        # and written down nowhere: that key appears ZERO times
        # in a 20,000-record log full of these refusals.
        #
        # So the one canned reply that must never be reachable
        # was also the least diagnosable thing in the runtime —
        # every occurrence said a gate had said no, and nothing
        # said what it objected to. The gate keeps only
        # INTEGRITY failures (leaks, corruption, prompt
        # artefacts, text that is not language), so the reason
        # is exactly what distinguishes a model producing
        # garbage from a gate that is too strict, and those want
        # opposite fixes.
        # Every key the receipt keeps a reason under, not one.
        #
        # The fix above read surface_quality_gate_reasons, and
        # the worker writes its actual objections under
        # semantic_completion_quality_reasons — the first key is
        # only written on the telemetry-sanitizer path. Two
        # names for one fact, so the diagnosis that was added to
        # end "rejected_for=no_reasons_reported" reported
        # no_reasons_reported.
        from .inference_gate import (
            logger,
        )

        _quality_reasons = tuple(
            dict.fromkeys(
                str(reason).strip()[:120]
                for key in (
                    "surface_quality_gate_reasons",
                    "semantic_completion_quality_reasons",
                    "telemetry_sanitizer_reasons",
                    # The fourth. The gate that keeps the best
                    # rejected draft records its objections
                    # here, and this is the one that carries
                    # them on the path a simple "read this file
                    # and tell me what it says" takes.
                    "surface_quality_rejected_reasons",
                )
                for reason in (
                    primary_surface_receipt.get(key) or ()
                )
                if str(reason).strip()
            )
        )
        # And when there are none, the draft itself.
        #
        # Four keys hold reasons and a path was found tonight
        # that populates none of them. A refusal that can name
        # neither its objection nor what it objected to is the
        # least diagnosable thing in the runtime, and it sits
        # one step before the one canned reply that must never
        # be reachable. The draft is already kept for the
        # repair path; nothing was reading it here.
        _rejected_draft = ""
        if not _quality_reasons:
            _rejected_draft = str(
                primary_surface_receipt.get(
                    "surface_quality_rejected_text"
                )
                or ""
            ).strip()[:220]
        if not _quality_reasons and not _rejected_draft:
            # Four keys and a draft, and this receipt has none
            # of them. Then the question is no longer what the
            # gate objected to but whether this is the receipt
            # the gate wrote, and the only way to tell is to
            # see what it does carry.
            logger.warning(
                "🧠 the refusing receipt carries no reasons and no "
                "draft; it holds: %s",
                ",".join(
                    f"{name}={primary_surface_receipt.get(name)!r}"[:90]
                    for name in sorted(map(str, primary_surface_receipt))
                    # Substring, deliberately: `name` is a
                    # receipt FIELD NAME — `surface_quality`,
                    # `rejected_by` — and this line exists to
                    # show what the receipt carries when it
                    # carries no reason. Narrowing it hides
                    # the fields worth seeing.
                    if any(
                        word in name
                        for word in (
                            "quality",
                            "reason",
                            "rejected",
                            "surface",
                        )
                    )
                )
                or "nothing about quality at all",
            )
        logger.warning(
            "🧠 %s exhausted its worker-owned semantic quality retries; "
            "preserving the lane and refusing a duplicate inference-gate "
            "retry. rejected_for=%s%s",
            local_label,
            ",".join(str(reason) for reason in _quality_reasons)
            or "no_reasons_reported",
            f" draft={_rejected_draft!r}" if _rejected_draft else "",
        )
        return _quality_reasons

    @staticmethod
    def _generate_with_metadata_sink_retry_morpho_kwargs(
        morpho_kwargs: Any,
        retry_attempt: Any,
        somatic_temperature: Any,
    ) -> tuple[dict[str, Any], Any]:
        retry_morpho_kwargs = dict(morpho_kwargs)
        retry_morpho_kwargs.update(
            {
                "disable_prompt_cache": True,
                "clear_prompt_cache": retry_attempt == 1,
                "top_p": min(float(retry_morpho_kwargs.get("top_p", 0.9) or 0.9), 0.85),
                "min_p": max(float(retry_morpho_kwargs.get("min_p", 0.02) or 0.02), 0.02),
                "repetition_penalty": max(
                    float(retry_morpho_kwargs.get("repetition_penalty", 1.1) or 1.1),
                    1.12,
                ),
                "repetition_context_size": max(
                    int(retry_morpho_kwargs.get("repetition_context_size", 64) or 64),
                    96,
                ),
                # The runtime TELEMETRY payload is what the
                # first attempt drowned in, so it is
                # skipped. The turn's evidence is not: it
                # travels in the repair messages built
                # above, which now carry grounding.
                "skip_runtime_payload": True,
                "repair_retains_grounding": True,
            }
        )
        retry_temperature = min(
            float(somatic_temperature if somatic_temperature is not None else 0.35),
            0.35,
        )
        return retry_morpho_kwargs, retry_temperature

