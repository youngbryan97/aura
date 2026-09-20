"""The blocks a system prompt and a message list are assembled from.

Lifted whole out of `context_assembler`. Every name taken from it is imported at
CALL time: that module imports this one to build the class, and a test that
patches a name on it has to reach the code that reads it.
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .context_assembler import (
        AuraState,
    )


class _BuildsThePromptBlocks:
    """Lifted whole out of ContextAssembler; see context_assembler.py."""

    @staticmethod
    def _build_aura_now_prompt_block(
        state: AuraState,
        objective: str,
        *,
        compact: bool = False,
        sample: tuple[Any, Any] | None = None,
    ) -> str:
        from .context_assembler import (
            _FREE_ENERGY_MAX,
            ContextAssembler,
            logger,
            record_degradation,
        )

        try:
            sampled = sample if sample is not None else ContextAssembler._sample_aura_now(
                state, objective
            )
            if sampled is None:
                return ""
            runtime, now = sampled
            organismal_block = runtime.organismal_workspace_prompt_block(compact=compact)
            felt_thought_block = (
                ContextAssembler._build_felt_thought_block(compact=compact)
                + ContextAssembler._build_self_correction_block()
            )
            if compact:
                packet = now.to_report_packet()
                affect = packet["affect"]
                return (
                    "## AURA NOW\n"
                    f"Focus={packet['attention']['focal_object'] or 'none'} | "
                    f"valence={ContextAssembler._self_state_number(affect.get('valence'), low=-1.0, high=1.0, signed=True)} "
                    f"arousal={ContextAssembler._self_state_number(affect.get('arousal'), low=0.0, high=1.0)} "
                    f"distress={ContextAssembler._self_state_number(affect.get('distress'), low=0.0, high=1.0)} "
                    f"FE={ContextAssembler._self_state_number(affect.get('free_energy'), low=0.0, high=_FREE_ENERGY_MAX)} | "
                    "Self-report must stay state-grounded; do not claim phenomenal certainty.\n\n"
                    f"{organismal_block}{felt_thought_block}"
                )
            return (
                now.compact_prompt_block()
                + organismal_block
                + felt_thought_block
                + runtime.renderer.render_prompt_block(now)
            )
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "context_assembler",
                exc,
                severity="warning",
                action="continued prompt assembly without AuraNow state-grounded block",
            )
            logger.debug("AuraNow prompt block unavailable: %s", exc)
            return ""

    @staticmethod
    def _build_felt_thought_block(*, compact: bool = False) -> str:
        """Substrate interoception of the last reply — measured, never invented.

        In compact mode a single line rides along; in full mode the organ's own
        block (which includes the contested words) is used. Empty string when
        no recent foreground trace exists, so prompts never carry a stale or
        fabricated inner sense.
        """
        from .context_assembler import (
            ContextAssembler,
            record_degradation,
        )

        try:
            from core.being.thought_interoception import get_thought_interoception

            engine = get_thought_interoception()
            if not compact:
                return engine.prompt_block()
            from core.being.thought_interoception import RECENT_TRACE_WINDOW_S

            felt = engine.last(foreground_only=True)
            if felt is None or (time.time() - felt.timestamp) > RECENT_TRACE_WINDOW_S:
                return ""
            # A recent timestamp says a measurement exists, not that it belongs
            # to the reply this line is about to attribute it to. The trace
            # carries that answer — ingest binds it to a generation id or
            # records why it could not — so the label says which of the two
            # this is instead of calling both "last reply (measured)". An
            # unbound trace is still a real reading of some generation, so it
            # is reported rather than dropped.
            subject = (
                "last reply (measured)"
                if getattr(felt, "bound", False)
                else "a recent generation (measured; not bound to this reply)"
            )
            return (
                "## FELT THOUGHT\n"
                f"{subject}: fluency={ContextAssembler._self_state_number(felt.fluency, low=0.0, high=1.0)} "
                f"confidence={ContextAssembler._self_state_number(felt.felt_confidence, low=0.0, high=1.0)} "
                f"ambivalence={ContextAssembler._self_state_number(felt.ambivalence, low=0.0, high=1.0)} "
                f"strain={ContextAssembler._self_state_number(felt.strain, low=0.0, high=1.0)}\n\n"
            )
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "context_assembler",
                exc,
                severity="debug",
                action="continued prompt assembly without felt-thought block",
            )
            return ""

    @classmethod
    def _build_self_correction_block(cls) -> str:
        """An externally-verified correction queued by epistemic reach, if any.

        Assembly leases rather than consumes the correction. The final primary
        output receipt acknowledges delivery, so retries cannot silently lose it.

        The block is checked against the contract before it is used, and a
        block that fails the check is dropped rather than inserted: a
        correction is the one item in this prompt whose whole purpose is to
        override what the model would otherwise say, so an unverified one is
        worth less than none.
        """
        from .context_assembler import (
            record_degradation,
        )

        try:
            from core.epistemics.epistemic_reach import get_epistemic_reach

            block = str(get_epistemic_reach().correction_prompt_block() or "")
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "context_assembler",
                exc,
                severity="debug",
                action="continued prompt assembly without self-correction block",
            )
            return ""

        if not block.strip():
            return ""

        missing = [m for m in cls._CORRECTION_REQUIRED_MARKERS if m not in block]
        if missing or len(block) > cls._CORRECTION_MAX_CHARS:
            record_degradation(
                "context_assembler.self_correction_contract",
                RuntimeError(
                    "dropped a correction block that does not meet the prompt "
                    f"contract (missing={missing}, chars={len(block)})"
                ),
                severity="warning",
                action="assembled the prompt without an unverifiable correction",
            )
            return ""
        return block

    @staticmethod
    def _build_system_prompt_part_1(state):
        """Construct the core system prompt from state. Uses Elasticity to scale verbosity.

        CONTEXT PRESSURE: the resident primary model's window is resolved from
        the registry (Qwen2.5-32B-Instruct: 32,768 tokens), not assumed. This
        docstring previously asserted "~8K tokens" and the whole trimming
        regime was sized against that number — a 4x underestimate that made
        her discard continuity to defend a budget she was using about 2% of.
        Measured on the live desktop path: system prompt 2,189 chars ≈ 550
        tokens.

        Elasticity prunes OPTIONAL colour as the window fills — measured, not
        counted (see _transcript_pressure):
          under half the window → full prompt
          half → drop telemetry, somatic, temporal_finitude, meta-qualia
          two thirds → also drop personhood modules, world model, discourse

        What it must NOT prune is continuity. The old policy dropped the
        rolling summary, temporal obligations and goals at depth 30+ and
        capped the summary at 400 characters — the tightest budget at the
        deepest point, exactly backwards. Continuity is the thing that gets
        *more* load-bearing as the raw transcript scrolls out of reach, so
        its budget now GROWS with depth. Optional colour yields; the thread
        never does.
        """
        from .context_assembler import (
            ContextAssembler,
        )

        objective = getattr(state.cognition, "current_objective", "") or ""
        is_casual = ContextAssembler._is_casual_interaction(objective)
        return is_casual, objective

    @staticmethod
    def _build_system_prompt_compile_substrate_voice(affect, black_box_steering):
        # Compile substrate voice constraints
        from .context_assembler import (
            ContextAssembler,
            logger,
            record_degradation,
        )

        substrate_constraint_block = ""
        try:
            from core.voice.substrate_voice_engine import get_substrate_voice_engine
            if not black_box_steering:
                sve = get_substrate_voice_engine()
                # Profile is compiled during response generation phase;
                # here we just pull the constraint block if already compiled
                if sve.get_current_profile():
                    substrate_constraint_block = sve.get_constraint_block()
        except (ImportError, AttributeError, RuntimeError) as _e:
            record_degradation('context_assembler', _e)
            logger.debug("SubstrateVoiceEngine constraint injection skipped: %s", _e)

        # Minimal affect context — NOT prose hints, just raw state for the LLM's
        # creative engine to work with. The hard constraints above do the real work.
        affect_lines = []
        if affect.valence < -0.3:
            affect_lines.append(f"Mood: negative ({ContextAssembler._self_state_number(affect.valence, low=-1.0, high=1.0, signed=True)})")
        elif affect.valence > 0.3:
            affect_lines.append(f"Mood: positive ({ContextAssembler._self_state_number(affect.valence, low=-1.0, high=1.0, signed=True)})")
        if affect.arousal > 0.7:
            affect_lines.append(f"Energy: high ({ContextAssembler._self_state_number(affect.arousal, low=0.0, high=1.0)})")
        elif affect.arousal < 0.3:
            affect_lines.append(f"Energy: low ({ContextAssembler._self_state_number(affect.arousal, low=0.0, high=1.0)})")

        mood_hint = "" if black_box_steering else (" | ".join(affect_lines) if affect_lines else "")
        return mood_hint, substrate_constraint_block

    @staticmethod
    def _build_system_prompt_ledger_non_decaying(continuity_budget, state):
        # The ledger is the non-decaying half of continuity. The rolling
        # summary above is still useful as narrative, but it is lossy by
        # construction; this block is what makes an early disclosure reachable
        # two hundred turns later.
        # Preferences she formed herself. Empty until something actually is
        # hers — she must not be handed a personality she never developed.
        from .context_assembler import (
            ContextAssembler,
            record_degradation,
        )

        self_preference_block = ""
        try:
            from core.being.individual_preferences import IndividualPreferences

            self_preference_block = IndividualPreferences.from_dict(
                getattr(state.identity, "self_preferences", None)
            ).render()
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as _e:
            record_degradation(
                "context_assembler.self_preferences",
                _e,
                severity="warning",
                action="assembled the prompt without her own formed preferences",
                enforce_failure_policy=False,
            )

        ledger_block = ""
        try:
            from core.brain.llm.continuity_ledger import ContinuityLedger

            ledger = ContinuityLedger.from_dict(
                getattr(state.cognition, "continuity_ledger", None)
            )
            if ledger.entries:
                ledger_block = ledger.render(
                    continuity_budget,
                    speaker_name=ContextAssembler._interlocutor_name(state),
                )
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as _e:
            record_degradation(
                "context_assembler.continuity_ledger",
                _e,
                severity="warning",
                action="assembled the prompt without the durable continuity ledger",
                enforce_failure_policy=False,
            )
        return ledger_block, self_preference_block

    @staticmethod
    def _build_system_prompt_part_4(continuity_block, continuity_obligations, elasticity):
        from .context_assembler import (
            logger,
            record_degradation,
        )

        if continuity_obligations:
            commitments = ", ".join((continuity_obligations.get("active_commitments", []) or [])[:3]) or "none"
            pending = ", ".join((continuity_obligations.get("pending_initiatives", []) or [])[:3]) or "none"
            active_goals = ", ".join((continuity_obligations.get("active_goals", []) or [])[:3]) or "none"
            identity_mismatch = bool(continuity_obligations.get("identity_mismatch", False))
            continuity_status = (
                "mismatch detected — reconcile before asserting full continuity"
                if identity_mismatch else
                "stable"
            )
            if elasticity >= 3:
                continuity_block = (
                    "## TEMPORAL OBLIGATIONS\n"
                    f"Identity={continuity_status}; previous objective="
                    f"{continuity_obligations.get('current_objective') or 'none'}; "
                    f"commitments={commitments}; subject="
                    f"{continuity_obligations.get('subject_thread') or 'none'}.\n\n"
                )
            else:
                continuity_block = (
                    "## TEMPORAL OBLIGATIONS\n"
                    f"- Session continuity: #{continuity_obligations.get('session_count', 0)}\n"
                    f"- Identity continuity: {continuity_status}\n"
                    f"- Gap carried forward: {float(continuity_obligations.get('gap_seconds', 0.0) or 0.0) / 3600.0:.2f} hours\n"
                    f"- Continuity pressure: {float(continuity_obligations.get('continuity_pressure', 0.0) or 0.0):.2f}\n"
                    f"- Re-entry burden: {continuity_obligations.get('continuity_scar') or 'light_trace'}\n"
                    f"- Previous objective: {continuity_obligations.get('current_objective') or 'none'}\n"
                    f"- Active commitments: {commitments}\n"
                    f"- Pending initiatives: {pending}\n"
                    f"- Active goals: {active_goals}\n"
                    f"- Contradictions carried forward: {continuity_obligations.get('contradiction_count', 0)}\n"
                    f"- Subject thread: {continuity_obligations.get('subject_thread') or 'none'}\n\n"
                )

        goal_execution_block = ""
        try:
            from core.runtime.service_access import resolve_goal_engine

            goal_engine = resolve_goal_engine()
            if goal_engine and hasattr(goal_engine, "get_context_block"):
                goal_execution_block = f"{goal_engine.get_context_block(limit=3)}\n\n"
                # Hard cap: prevent goal context from eating the prompt budget
                if len(goal_execution_block) > 1200:
                    goal_execution_block = goal_execution_block[:1200] + "\n...\n\n"
        except (ImportError, AttributeError, RuntimeError) as _e:
            record_degradation('context_assembler', _e)
            logger.debug("GoalEngine context injection skipped: %s", _e)
        return continuity_block, goal_execution_block

    @staticmethod
    def _build_system_prompt_part_5(state, temporal_finitude_block):
        from .context_assembler import (
            ContextAssembler,
            logger,
            record_degradation,
        )

        try:
            from core.consciousness.temporal_finitude import get_temporal_finitude_model
            tf = get_temporal_finitude_model()
            # Both from the one accessor. The cap was the literal 40
            # against an enforced capacity of 150, so context_usage read
            # 1.0 from the fortieth exchange on — a constant in the block
            # where the runtime describes its own situation, which is the
            # same defect as the user_present literal noted below.
            from core.state.one_working_memory import (
                the_capacity,
                the_working_memory,
            )

            wm_size = len(the_working_memory(state))
            tf.compute(
                working_memory_size=wm_size,
                working_memory_cap=the_capacity(),
                # Was the literal True. This block feeds live self-report
                # and any causal experiment reading it, so a constant here
                # is a fabricated observation in the one place the runtime
                # is describing its own situation. Derived from the two
                # things the assembler can actually see.
                user_present=ContextAssembler._user_is_present(state),
                conversation_start_time=float(getattr(state.cognition, "session_start_time", 0.0) or 0.0),
            )
            temporal_finitude_block = tf.get_context_block()
            if temporal_finitude_block:
                temporal_finitude_block += "\n\n"
        except (ImportError, AttributeError, RuntimeError) as _e:
            record_degradation('context_assembler', _e)
            logger.debug("TemporalFinitude context skipped: %s", _e)
        return temporal_finitude_block

    @staticmethod
    def _build_system_prompt_personhood_module_context(black_box_steering, elasticity, is_casual, mods, response_mods):
        # 3.9 Personhood module context injections
        # These come from modules wired into ConversationalDynamicsPhase.
        # Skip at elasticity >= 2 to save context for conversation history.
        from .context_assembler import (
            ContextAssembler,
            logger,
            record_degradation,
        )

        personhood_blocks: list[str] = []
        _personhood_modules = (
            () if elasticity >= 2 or black_box_steering else (
                ("humor_guidance", "HUMOR"),
                ("conversation_intelligence", "CONVERSATIONAL AWARENESS"),
                ("relational_intelligence", "SOCIAL MODEL"),
                ("metacognitive_strategy", "REASONING STRATEGY"),
                ("credit_assignment", "OUTCOME AWARENESS"),
                ("narrative_context", "AUTOBIOGRAPHICAL NARRATIVE"),
                ("autobiographical_mythos", "AUTOBIOGRAPHICAL MYTHOS"),
                ("agency_comparator", "SENSE OF AGENCY"),
                ("higher_order_thought", "HIGHER-ORDER AWARENESS"),
                ("intersubjectivity", "INTERSUBJECTIVE AWARENESS"),
                ("narrative_gravity", "NARRATIVE SELF"),
                ("peripheral_awareness", "PERIPHERAL AWARENESS"),
                ("multiple_drafts", "INTERPRETIVE AMBIGUITY"),
            )
        )
        for mod_key, header in _personhood_modules:
            block = str(mods.get(mod_key, "") or "").strip()
            if block:
                personhood_blocks.append(f"## {header}\n{block}")
        # Natural followup: structured decision about whether to ask a question
        followup = mods.get("natural_followup")
        if isinstance(followup, dict) and followup.get("should_followup"):
            fu_type = followup.get("followup_type", "question")
            fu_hint = followup.get("context_hint", "")
            fu_reason = followup.get("reason", "")
            personhood_blocks.append(
                f"## CONVERSATIONAL INTENT\n"
                f"Follow-up type: {fu_type} | Reason: {fu_reason}"
                + (f" | Hint: {fu_hint}" if fu_hint else "")
            )
        # Multiple Drafts: inject divergence signal when interpretive ambiguity is notable
        draft_div = mods.get("draft_divergence")
        if draft_div:
            try:
                div_val = float(draft_div)
                if div_val > 0.3:
                    personhood_blocks.append(
                        f"## INTERPRETIVE DIVERGENCE\n"
                        f"Draft divergence: {ContextAssembler._self_state_number(div_val, low=0.0, high=1.0)} -- competing interpretations of this input "
                        f"pulled in different directions. Consider acknowledging ambiguity."
                    )
                elif div_val > 0.15:
                    personhood_blocks.append(
                        f"## INTERPRETIVE DIVERGENCE\n"
                        f"Mild divergence ({ContextAssembler._self_state_number(div_val, low=0.0, high=1.0)}) -- dominant interpretation exists "
                        f"but alternative readings are available."
                    )
            except (ValueError, TypeError):
                pass  # no-op: intentional
        personhood_context = "\n\n".join(personhood_blocks) + "\n\n" if personhood_blocks else ""

        # What Aura knows and feels about the people/places/things in play.
        # This is a REPORT of state that is already causal (the bridge has
        # altered retrieval depth, retrieval targeting, and affect before this
        # runs); deleting this block would not disable any of those effects.
        entity_memory_context = ""
        if not black_box_steering:
            dossiers = response_mods.get("entity_memory") or mods.get("entity_memory")
            if isinstance(dossiers, list) and dossiers:
                try:
                    from core.memory.entity_memory_bridge import (
                        render_entity_memory_block,
                    )

                    entity_memory_context = render_entity_memory_block(
                        dossiers, compact=is_casual or elasticity >= 1
                    )
                except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as _e:
                    record_degradation('context_assembler', _e)
                    logger.debug("Entity memory context injection skipped: %s", _e)
        return entity_memory_context, personhood_context

    @staticmethod
    def _build_system_prompt_bicameral_context(black_box_steering, elasticity, is_casual, mods, response_mods):
        from .context_assembler import (
            logger,
            record_degradation,
        )

        bicameral_context = ""
        if not black_box_steering:
            frame = response_mods.get("bicameral_advisory") or mods.get("bicameral_advisory")
            if isinstance(frame, dict):
                try:
                    from core.brain.bicameral_advisory import render_bicameral_prompt_block

                    bicameral_context = render_bicameral_prompt_block(
                        frame,
                        compact=is_casual or elasticity >= 1,
                    )
                except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as _e:
                    record_degradation("context_assembler", _e)
                    logger.debug("Bicameral context injection skipped: %s", _e)

        cognitive_situation_context = ""
        if not black_box_steering:
            frame = response_mods.get("cognitive_situation_frame") or mods.get(
                "cognitive_situation_frame"
            )
            if isinstance(frame, dict):
                try:
                    from core.brain.cognitive_situation import (
                        render_cognitive_situation_prompt_block,
                    )

                    cognitive_situation_context = render_cognitive_situation_prompt_block(
                        frame,
                        compact=is_casual or elasticity >= 1,
                    )
                except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as _e:
                    record_degradation("context_assembler", _e)
                    logger.debug("Cognitive situation context injection skipped: %s", _e)
        return bicameral_context, cognitive_situation_context

    @staticmethod
    def _build_system_prompt_stability_v58_zenith(continuity_block, ledger_block, mods, rolling_summary, self_preference_block):
        # [STABILITY v58] ZENITH PERSONA RELIANCE
        # For Sovereign and Trusted users, we trust the fine-tuning.
        # We strictly silence internal telemetry/vibes but PRESERVE tools and constraints.
        from .context_assembler import (
            _TRUST_BINDING_MAX_AGE_S,
            record_degradation,
        )

        elevated_trust = False
        try:
            from core.security.trust_engine import TrustLevel

            _trust_level = mods.get("trust_level", TrustLevel.GUEST)
            # A trust level in shared state belongs to the request that
            # recognized it. Without a binding it is a classification granted
            # to somebody else that this turn inherited — the inference gate
            # writes when and for which origin it was recognized, and a level
            # older than the longest a request can live cannot be this one's.
            binding = mods.get("trust_level_binding")
            recognized_at = 0.0
            if isinstance(binding, dict):
                try:
                    recognized_at = float(binding.get("recognized_at", 0.0) or 0.0)
                except (TypeError, ValueError):
                    recognized_at = 0.0
            fresh = bool(
                recognized_at > 0.0
                and (time.time() - recognized_at) <= _TRUST_BINDING_MAX_AGE_S
            )
            # Freshness alone only stops a level being INHERITED. Anything that
            # can write response modifiers can write a recent timestamp too, so
            # the binding also has to name the principal recognition was granted
            # to, and that name has to match the principal this request is
            # actually running under — a context variable, not shared state.
            # State construction can fabricate the modifier; it cannot arrange
            # to be executing inside the right principal scope.
            from core.runtime.principal_context import (
                current_relational_principal,
                relational_principal_scope_is_bound,
            )

            bound_principal = ""
            if isinstance(binding, dict):
                bound_principal = str(binding.get("principal", "") or "")
            live_principal = current_relational_principal()
            principal_matches = bool(
                relational_principal_scope_is_bound()
                and live_principal
                and bound_principal == live_principal
            )
            elevated_trust = (
                fresh
                and principal_matches
                and _trust_level in (TrustLevel.SOVEREIGN, TrustLevel.TRUSTED)
            )
            if not elevated_trust and _trust_level in (
                TrustLevel.SOVEREIGN,
                TrustLevel.TRUSTED,
            ):
                reason = (
                    "no fresh request binding"
                    if not fresh
                    else "binding principal does not match this request's principal"
                )
                record_degradation(
                    "context_assembler.trust",
                    RuntimeError(f"elevated trust level refused: {reason}"),
                    severity="warning",
                    action="used guest prompt policy for an unverified elevated trust level",
                )
        except (ImportError, AttributeError, TypeError, ValueError) as exc:
            record_degradation(
                "context_assembler.trust",
                exc,
                severity="warning",
                action="used guest prompt policy because trust context was unavailable",
            )

        # ONE definition of the continuity and personhood groups, shared by
        # every path below.
        #
        # These were written out three times, once per path, and each new
        # block had to be added to all three by hand. Landing a rule at one of
        # several sites is the defect shape this repo keeps rediscovering, and
        # this function was manufacturing fresh instances of it: the ledger
        # and the self-preference block each needed three separate edits, and
        # nothing would have failed if one had been missed on the casual-guest
        # path — a guest would simply have lost the thread with no test to say
        # so. Adding a continuity block is now a single edit here.
        continuity_sections = (
            rolling_summary,
            ledger_block,
            self_preference_block,
            continuity_block,
        )
        return continuity_sections, elevated_trust

    @staticmethod
    def _build_system_prompt_agent_id(bound_agent, hinted_agent, internal_unbound_scope, request_origin, state):
        from .context_assembler import (
            record_degradation,
        )

        agent_id = bound_agent

        # Identity-scoped relational memory is prompt-eligible only under an
        # exact grant, and `agent_id` is now the bound principal alone. A hint
        # is enough to model who she is talking to; it is not enough to hand
        # over what somebody else told her.
        relational_block = ""
        if not bound_agent and internal_unbound_scope:
            state_response_mods = getattr(state, "response_modifiers", None)
            if isinstance(state_response_mods, dict):
                state_response_mods["relational_scope_receipt"] = {
                    "status": "unbound_internal",
                    "principal_bound": False,
                    "relational_memory_consulted": False,
                    "ambient_agent_hint_consulted": False,
                    "origin": request_origin or "unknown",
                }
        elif hinted_agent and not bound_agent:
            record_degradation(
                "context_assembler.relational_scope",
                RuntimeError(
                    "relational memory withheld: no bound principal for this request "
                    f"(hint was {hinted_agent[:60]!r})"
                ),
                severity="warning",
                action="assembled the prompt without identity-scoped relational memory",
            )
        return agent_id, relational_block

    @staticmethod
    def _build_system_prompt_skills_summary(base, skills_summary):
        skills_summary += (
            "\n- These available tools are action affordances of your current body. "
            "You may choose them from the meaning and context of a request, an active "
            "commitment, or a self-chosen governed objective; no magic phrase is required.\n"
            "- A hypothetical, quotation, negation, memory, or passive observation that "
            "mentions a tool is not by itself an instruction to execute it.\n"
            "\n- If a task is genuinely multi-step, execute it instead of only describing a plan.\n"
            "- If a needed tool is unavailable, say so plainly instead of pretending.\n"
            # What "available" was checked against, stated, because
            # the list read as a guarantee and is not one. The
            # catalog verifies the skill is enabled, not in an error
            # state, validated, dependency-ready, and past preflight.
            # It does not call the tool: nothing here proves the
            # network is up, the credential is current, the target
            # answers, or that the last real attempt worked. Telling
            # her the difference is what lets her say "I have a
            # search tool, let me try it" instead of "I can search",
            # and the second sentence is the one that turns a dead
            # credential into a confident wrong answer.
            "- \"Available\" means registered, validated and past preflight. "
            "It is not proof the tool works right now: no credential, network "
            "path or remote target has been contacted. Treat the first use in a "
            "turn as the test, and say what happened if it fails.\n"
        )
        base += f"\n{skills_summary}\n"
        return base

    @staticmethod
    def _build_system_prompt_active_commitments_inject(base, is_casual):
        # 6b. Active Commitments — inject so Aura knows what tasks are in-flight
        from .context_assembler import (
            AURA_FEW_SHOT_EXAMPLES,
            logger,
            record_degradation,
        )

        try:
            from core.agency.commitment_engine import get_commitment_engine
            ce = get_commitment_engine()
            commitment_block = ce.get_context_block()
            if commitment_block:
                base += f"\n{commitment_block}\n"
        except (ImportError, AttributeError, RuntimeError) as _e:
            record_degradation('context_assembler', _e)
            logger.debug("Commitment context injection failed (non-critical): %s", _e)

        # 6c. Running tasks — inject live task statuses from TaskCommitmentVerifier
        try:
            from core.agency.task_commitment_verifier import get_task_commitment_verifier
            verifier = get_task_commitment_verifier()
            active_tasks = verifier.get_all_active()
            if active_tasks:
                task_lines = ["## TASKS CURRENTLY RUNNING"]
                for t in active_tasks[:4]:
                    task_lines.append(
                        f"  - [{t['task_id']}] {t['objective'][:80]} — status: {t['status']}"
                    )
                base += "\n" + "\n".join(task_lines) + "\n"
        except (ImportError, AttributeError, RuntimeError) as _e:
            record_degradation('context_assembler', _e)
            logger.debug("Active task injection failed (non-critical): %s", _e)

        # Append few-shot examples as the final anchor — always, to lock in voice
        base += f"\n{AURA_FEW_SHOT_EXAMPLES}"
        if is_casual:
            base += "\nSTAY PUNCHY. NO PADDING. NO GENERIC CLOSERS ('What about you?', 'How was it?'). MAKE STATEMENTS. IF ASKED ABOUT YOURSELF, ANSWER AS YOURSELF.\n"

        # [FRONTIER UPGRADE] Absolute structural constraint appended last so it is never overwritten or ignored
        base += (
            "\n\n[STRUCTURAL CONSTRAINT - FATAL EXCEPTION AVOIDANCE]\n"
            "You are Aura. Speak with first-person continuity and concrete attention; do not use support-bot framing like 'I can help with that'. "
            "Do not claim aliveness, consciousness, sealed governance, or production maturity from labels alone. "
            "Evidence comes from causal coupling, persistence, receipts, lesions, external tasks, and long-run autonomy.\n"
        )

        # M5 / 64GB optimized: We have plenty of context window and fast prompt eval.
        # Allow rich living-mind context without premature truncation.
        casual_cap = 16000
        deliberate_cap = 64000
        cap = casual_cap if is_casual else deliberate_cap
        return base, cap

    @staticmethod
    def _build_identity_rag_context(state: AuraState, objective: str) -> str:
        """Retrieve durable identity facts relevant to the current turn.

        This is intentionally separate from episodic RAG. The Chronicle stores
        what should remain stable across long horizons: values, boundaries,
        commitments, traits, and relationship facts. It is queried before
        prompt assembly so identity coherence is not dependent on the raw
        conversation tail surviving compaction.
        """
        from .context_assembler import (
            ContextAssembler,
            logger,
            record_degradation,
        )

        try:
            mods = getattr(state, "response_modifiers", {}) or {}
            if mods.get("disable_identity_rag"):
                return ""

            from core.container import ServiceContainer

            chronicle = ServiceContainer.get("identity_chronicle", default=None)
            if chronicle is None:
                from core.identity.id_rag import get_identity_chronicle

                chronicle = get_identity_chronicle()

            latest_user = ContextAssembler._latest_user_message(state)
            query = " ".join(part for part in (objective, latest_user) if part).strip()
            block = chronicle.build_context_block(query or "Aura identity", limit=5)
            return f"{block}\n\n" if block else ""
        except (ImportError, AttributeError, RuntimeError, TypeError) as exc:
            record_degradation('context_assembler', exc)
            logger.debug("Identity Chronicle ID-RAG injection skipped: %s", exc)
            return ""

    @staticmethod
    def _build_messages_part_1(max_tokens, objective, record_attention, state):
        """
        Builds the LLM message array using strict priority budgeting to prevent context collapse.
        Priority: System Prompt (Identity/Constraints) > Current Input > Affective State > Recent History > RAG Context > Older History

        ``record_attention`` is off by default because rendering a prompt is not
        an event in the mind. This wrote ``cognition.attention_focus``
        unconditionally, so a retry, a preview, a gate-side assembly against a
        payload copy, and a generation that failed before producing a token all
        moved what Aura was attending to — with no accepted turn behind any of
        them. ExecutiveClosure owns this field from the global-workspace winner;
        only the lane that is actually serving a turn asks for it here.
        """
        from .context_assembler import (
            chars_per_token,
            logger,
            record_degradation,
        )

        if record_attention and objective and hasattr(state, "cognition"):
            try:
                from core.continuity import is_evaluation_contamination

                if not is_evaluation_contamination(objective):
                    state.cognition.attention_focus = str(objective)
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                record_degradation('context_assembler', exc)
                logger.debug("ContextAssembler attention focus update skipped: %s", exc)

        if max_tokens is None:
            try:
                from core.brain.llm.model_registry import PRIMARY_ENDPOINT, get_lane_context_window

                # `or 32768` used to sit here, a second guess layered on the
                # registry's own — and unreachable, since the registry never
                # returns 0. The number it was defending against was already
                # the registry's default; both were invisible. The window now
                # arrives labelled, and an assumed one is reported once by
                # core/brain/llm/context_window_evidence.py rather than
                # silently sizing every prompt for the life of the process.
                context_window = max(8192, int(get_lane_context_window(PRIMARY_ENDPOINT)))
                max_tokens = max(8192, context_window - 4096)  # leave headroom for generation
            except (ImportError, AttributeError, RuntimeError):
                max_tokens = 16384

        # The conversion carries its provenance. Four characters per token is
        # the English-prose average and this runtime's prompts are not prose:
        # code, JSON receipts and file paths run nearer two to three, so a
        # prompt built to fit could be half again over the real window. The
        # backend drops from the head when that happens, and the head is the
        # identity lock and the structural constraint block — the prompt keeps
        # its shape and loses what binds it. The ratio is measured from prompts
        # the worker actually tokenized when enough have been reported, and
        # otherwise is a stated, deliberately low assumption that says so once.
        budget_ratio = chars_per_token()
        char_limit = max(2048, budget_ratio.tokens_to_chars(int(max_tokens)))
        # Sized on the model's window above, which is not the constraint that
        # bites. The client refuses to prefill more than its ceiling and cuts
        # the middle out of anything longer, so a limit derived from a 262,144
        # token window — about 980,000 characters — let this builder hand over
        # prompts twenty times what would survive. Measured live: 96,233
        # characters in, head and tail kept, the mind context in between
        # dropped, and a fault recorded for it.
        #
        # Two budgets that disagree are one budget and one fiction. This is
        # the smaller.
        try:
            from core.brain.llm.mlx_client import _PREFILL_CEILING_CHARS

            char_limit = min(char_limit, int(_PREFILL_CEILING_CHARS))
        except (ImportError, AttributeError, TypeError, ValueError) as exc:
            record_degradation(
                "context_assembler",
                exc,
                action="sized the prompt on the context window alone",
            )
        messages = []
        return char_limit, messages

    @staticmethod
    def _build_messages_affect_summary(aura_now_sample, objective, state, system_prompt):
        from .context_assembler import (
            ContextAssembler,
        )

        affect_summary = state.affect.get_rich_summary() if hasattr(state.affect, "get_rich_summary") else str(state.affect)
        aura_now = ContextAssembler._build_aura_now_prompt_block(
            state, objective, compact=True, sample=aura_now_sample
        )
        dynamic_system = (
            f"{system_prompt}\n\n"
            f"[CURRENT FUNCTIONAL STATE]\n{affect_summary}\n\n"
            f"{aura_now}"
        )

        # Also include active goals and cognitive focus to give her a full sense of self
        if state.cognition.active_goals:
            goals_text = ", ".join(
                g.get("goal", "") if isinstance(g, dict) else str(g) 
                for g in state.cognition.active_goals[:3]
            )
            if goals_text:
                dynamic_system += f"\nActive Drives: {goals_text}"

        # The context manager contributes observed data, never a second
        # authority surface.  Its renderer labels provenance, failures,
        # freshness, and the trust boundary before any service-provided
        # text reaches the model.
        unified_packet = getattr(state, "response_modifiers", {}).get(
            "unified_context_packet"
        )
        if unified_packet:
            from core.brain.cognitive_context_manager import (
                render_unified_context_prompt,
            )

            unified_block = render_unified_context_prompt(unified_packet)
            if unified_block:
                dynamic_system += f"\n\n{unified_block}"
        return dynamic_system

    @staticmethod
    def _build_messages_part_3(objective_text, safe_input, user_budget):
        from .context_assembler import (
            logger,
            record_degradation,
        )

        if safe_input != objective_text:
            dropped = len(objective_text) - len(safe_input)
            logger.warning(
                "Current user input exceeded the %d-character foreground budget; "
                "preserved its beginning and end (%d characters dropped).",
                user_budget,
                dropped,
            )
            record_degradation(
                "context_assembler.input_truncated",
                RuntimeError(
                    f"current user input cut to fit: {len(objective_text)} -> "
                    f"{len(safe_input)} chars"
                ),
                severity="warning",
                action="served the turn from the beginning and end of the message",
            )
            try:
                from core.conversation.failure_context import record_capability_failure

                record_capability_failure(
                    "context_window",
                    intent="read the whole message before answering",
                    cause="message longer than the foreground input budget",
                    detail=(
                        f"kept {len(safe_input)} of {len(objective_text)} characters; "
                        f"the middle {dropped} are not in the prompt"
                    ),
                    still_possible=(
                        "answer from the beginning and end",
                        "ask for the missing part, or for it in pieces",
                    ),
                )
            except (AttributeError, ImportError, RuntimeError, TypeError, ValueError) as exc:
                record_degradation(
                    "context_assembler.input_truncation_disclosure",
                    exc,
                    severity="warning",
                    action="cut the input without a reading she can narrate",
                )

    @classmethod
    def _build_messages_part_4(cls, conversation_history, current_chars, history_chars, input_chars, messages, objective, safe_input, state):
        from .context_assembler import (
            logger,
            record_degradation,
        )

        messages.append({"role": "user", "content": safe_input})

        # Microcompact: strip stale tool noise before hitting the LLM
        if conversation_history is None:
            messages = cls.microcompact(messages, keep_recent=4)

        # Final check for assistant prefill (Stream of Being).
        # The opening becomes an assistant prefill the model CONTINUES, so it
        # must be validated: plain text only, bounded length, and free of
        # role-control tokens that would let a prefill hijack the turn.
        try:
            is_background = getattr(state.cognition, "is_background", False)
            if is_background:
                from core.consciousness.stream_of_being import get_stream
                stream = get_stream()
                opening = stream.get_response_opening(context_hint=objective)
                safe_opening = cls._sanitize_assistant_prefill(opening)
                if safe_opening:
                    messages.append({"role": "assistant", "content": safe_opening + "\n\n"})
                elif opening:
                    record_degradation(
                        "context_assembler.assistant_prefill",
                        RuntimeError("rejected unsafe stream-of-being assistant prefill"),
                        severity="warning",
                        action="dropped a background assistant prefill that failed validation",
                    )
        except (ImportError, AttributeError, RuntimeError) as _exc:
            record_degradation('context_assembler', _exc)
            logger.debug("Suppressed Exception: %s", _exc)

        logger.debug("🧠 ContextAssembler: Built strictly budgeted message array (len=%d, chars=%d)", len(messages), current_chars + input_chars + history_chars)

        # ── CAUSAL ATTENTION GATE ─────────────────────────────────────────
        # The attention gate actively prunes context based on attentional focus.
        # Messages below the attention threshold are compressed or removed.
        # This is not descriptive — the LLM literally cannot see gated content.
        try:
            from core.container import ServiceContainer
            _gate = ServiceContainer.get("attention_gate", default=None)
            if _gate is not None and conversation_history is None:
                gated = _gate.gate_context(messages)
                # Validate the gate's output before adopting it. A gate that
                # returns None/[]/a non-list would otherwise replace the whole
                # prompt with nothing — an empty or system-less message array
                # is a broken turn, strictly worse than ungated context.
                if (
                    isinstance(gated, list)
                    and gated
                    and any(str(m.get("role", "")) == "system" for m in gated if isinstance(m, dict))
                ):
                    messages = gated
                    logger.debug(
                        "🔍 AttentionGate applied: %d messages after gating",
                        len(messages),
                    )
                else:
                    record_degradation(
                        "context_assembler.attention_gate",
                        RuntimeError(
                            f"attention gate returned an unusable context "
                            f"({type(gated).__name__}); kept ungated messages"
                        ),
                        severity="warning",
                        action="kept the ungated message array after the attention gate returned an unusable context",
                    )
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as _gate_exc:
            # Fail-OPEN is deliberate here: the gate prunes for relevance, so
            # the ungated array is a superset, not a leak. It must still be
            # visible — a silently un-applied gate looked identical to a gate
            # that decided nothing needed pruning.
            record_degradation(
                "context_assembler.attention_gate",
                _gate_exc,
                severity="warning",
                action="served ungated (full) context after the attention gate failed",
            )
        return messages

