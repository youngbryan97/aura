"""The two amplifiers a draft passes through before it is served.

The reasoning amplifier asks for candidates a checked verifier can promote, and
the conversational one ranks candidates by her taste.

Lifted whole out of `response_generation_unitary`. Every name taken from it is imported at
CALL time: that module imports this one to build the class, and a test that
patches a name on it has to reach the code that reads it.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .response_generation_unitary import (
        AuraState,
    )


class _AmplifiesTheDraft:
    """Lifted whole out of UnitaryResponsePhase; see response_generation_unitary.py."""

    async def _maybe_amplify_response(
        self,
        *,
        objective: str,
        draft: str,
        llm: Any,
        state: AuraState,
        request_timeout: float,
        is_user_facing: bool,
        is_background: bool,
        proof_or_benchmark: bool,
        seed_candidates: list[str] | None = None,
        evidence: list[str] | None = None,
    ) -> str:
        """Re-derive a verifiable hard-turn answer through the reasoning amplifier.

        Runs only on the foreground lane for verifiable hard turns, is bounded by
        the turn's own timeout, and fails open to ``draft``. The first draft remains
        the incumbent. It is replaced only by an objectively verified result or by
        independent executable consensus, whose probabilistic authority remains
        explicit in the receipt.
        """
        from .response_generation_unitary import (
            _FLAG_AMPLIFIER_TIER_ESCALATION,
            _RESPONSE_RECOVERABLE_ERRORS,
            _record_response_degradation,
            logger,
            reasoning_amplifier_v2_enabled,
        )

        if not is_user_facing or is_background or proof_or_benchmark or not draft:
            return draft
        if not reasoning_amplifier_v2_enabled():
            return draft
        try:
            from core.brain.reasoning_amplifier_v2 import amplify_turn, is_amplifiable
        except ImportError:
            return draft
        task_type = is_amplifiable(objective)
        if task_type is None:
            return draft
        from core.brain.executable_reasoning import should_use_executable_reasoning

        executable_reasoning = should_use_executable_reasoning(
            objective,
            task_type=task_type,
        )

        def _make_gen(tier: str) -> Any:
            async def _gen(prompt: str, temperature: float) -> str:
                try:
                    out = await llm.think(
                        prompt,
                        temperature=temperature,
                        prefer_tier=tier,
                        allow_cloud_fallback=False,
                    )
                except _RESPONSE_RECOVERABLE_ERRORS as exc:
                    _record_response_degradation(exc, "UnitaryResponse: amplifier generate failed: %s")
                    return ""
                if isinstance(out, dict):
                    out = out.get("content") or out.get("response") or ""
                return str(out or "").strip()

            return _gen

        _gen = _make_gen("primary")

        # Tier escalation (verifier-of-last-resort): when a hard turn finishes
        # verifier-dirty with budget left, retry once on the local deep tier.
        # Off by default so the running foreground lane keeps its latency contract;
        # opt in with AURA_AMPLIFIER_TIER_ESCALATION=1.
        escalate_gen = None
        if str(_FLAG_AMPLIFIER_TIER_ESCALATION.value()).strip().lower() in {"1", "true", "on", "yes"}:
            escalate_gen = _make_gen("deep")

        # A resident-32B program-of-thought generation takes roughly 45-55s on
        # this host. The old universal 30s ceiling made admitted executable
        # tasks impossible by construction. Spend a larger but still bounded
        # share of the foreground contract only when structured computation is
        # actually applicable; evidence-only amplification keeps its 30s cap.
        requires_full_program_budget = bool(
            executable_reasoning and task_type != "math"
        )
        budget_floor = 60.0 if requires_full_program_budget else 8.0
        budget_ceiling = 150.0 if executable_reasoning else 30.0
        available_budget = max(1.0, float(request_timeout or 20.0) * 0.8)
        budget = float(min(budget_ceiling, available_budget))
        if requires_full_program_budget and budget < budget_floor:
            return draft
        budget = max(min(budget_floor, available_budget), budget)
        result = await amplify_turn(
            objective,
            _gen,
            task_type=task_type,
            evidence=list(evidence or []),
            time_budget_s=budget,
            sample_budget=3 if executable_reasoning else None,
            extra_context={
                "seed_candidates": list(seed_candidates or [draft]),
                "enable_executable_reasoning": executable_reasoning,
                "allow_textual_fallback_after_executable": True,
            },
            escalate_generate=escalate_gen,
        )
        receipt = result.receipt.to_dict()
        self._last_reasoning_receipt = receipt
        try:
            if hasattr(state, "metadata") and isinstance(state.metadata, dict):
                state.metadata["reasoning_receipt"] = receipt
        except (AttributeError, TypeError):
            pass
        logger.info(
            "🧠 [AmplifyV2-live/phase] task=%s mode=%s verified=%s conf=%.2f → %s",
            task_type, receipt.get("mode"), result.verified, result.confidence,
            (
                "adopted"
                if (
                    result.answer
                    and receipt.get("promotion_authority")
                    in {"checked_verifier", "independent_executable_consensus"}
                )
                else "kept draft"
            ),
        )
        authority = str(receipt.get("promotion_authority") or "none")
        if authority == "checked_verifier" and result.answer and len(result.answer.strip()) >= 3:
            return result.answer.strip()
        if authority == "independent_executable_consensus":
            consensus_answer = str(result.source_answer or result.answer or "").strip()
            if len(consensus_answer) >= 1:
                return consensus_answer
        return draft

    async def _maybe_amplify_conversation(
        self,
        *,
        objective: str,
        draft: str,
        llm: Any,
        state: AuraState,
        request_timeout: float,
        is_user_facing: bool,
        is_background: bool,
        proof_or_benchmark: bool,
    ) -> str:
        """Best-of-N taste-selection + self-revise for substantive conversational turns.

        The unverifiable analogue of the reasoning amplifier: there's no truth-engine for
        wit/voice, so candidates are ranked by the personalized TasteModel and the winner
        is optionally self-revised. Foreground conversational turns only; excludes actions
        and verifiable-reasoning turns (those are owned elsewhere). Bounded, fail-open.
        """
        from .response_generation_unitary import (
            _FLAG_CONVERSATIONAL_AMPLIFIER_LIVE,
            _RESPONSE_RECOVERABLE_ERRORS,
            _breath_in_words,
            _record_response_degradation,
            _taste_conversation_id,
            is_witnessing,
        )

        if not is_user_facing or is_background or proof_or_benchmark or not draft:
            return draft
        live_flag = str(_FLAG_CONVERSATIONAL_AMPLIFIER_LIVE.value()).strip().lower()
        if live_flag not in {"1", "true", "on", "yes"}:
            return draft
        try:
            from core.utils.memory_monitor import get_memory_pressure_snapshot

            pressure = get_memory_pressure_snapshot()
            if bool(getattr(pressure, "refuse_heavy_local_generation", False)):
                return draft
            if float(getattr(pressure, "pressure_pct", 0.0) or 0.0) >= 85.0:
                return draft
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            pass
        try:
            from core.brain.conversational_amplifier import (
                amplify_conversation,
                is_conversationally_amplifiable,
            )
        except ImportError:
            return draft
        origin = self._normalize_origin(getattr(getattr(state, "cognition", None), "current_origin", "") or "user")
        if not is_conversationally_amplifiable(objective, origin):
            return draft

        async def _gen(prompt: str, temperature: float) -> str:
            try:
                out = await llm.think(
                    prompt, temperature=temperature, prefer_tier="primary", allow_cloud_fallback=False
                )
            except _RESPONSE_RECOVERABLE_ERRORS as exc:
                _record_response_degradation(exc, "UnitaryResponse: conversational amplifier generate failed: %s")
                return ""
            if isinstance(out, dict):
                out = out.get("content") or out.get("response") or ""
            return str(out or "").strip()

        # Grounding tokens (rolling summary + recent working memory) feed the callback
        # feature so wit-via-memory is rewarded.
        grounding_tokens: set[str] = set()
        try:
            cog = getattr(state, "cognition", None)
            summary = str(getattr(cog, "rolling_summary", "") or "")
            grounding_tokens = {w.lower() for w in re.findall(r"[A-Za-z0-9']+", summary)}
        except (AttributeError, TypeError, ValueError):
            grounding_tokens = set()
        word_budget = 0
        try:
            word_budget = int(state.response_modifiers.get("voice_word_budget", 0) or 0)
        except (AttributeError, TypeError, ValueError):
            word_budget = 0
        if word_budget <= 0:
            # Nothing set a spoken-length budget, so the breath does: what she
            # has left this cycle, priced in the effort ledger's own unit.
            # Working twice as hard halves it. See core/expression/delivery.py.
            word_budget = _breath_in_words(state, draft)

        budget = float(min(10.0, max(3.0, (request_timeout or 20.0) * 0.25)))
        n_candidates = 2 if budget < 8.0 else 3
        # A draft given up for some seconds is a trade, and the series of them
        # has a price. While that price is falling she stops taking it: the
        # third draft is kept, and the turns that follow are the pairs the
        # next reading is taken from. See core/self/what_it_cost_her.py.
        try:
            from core.self.what_it_cost_her import get_price_ledger

            prices = get_price_ledger()
            if n_candidates < 3:
                if prices.holding():
                    n_candidates = 3
                else:
                    prices.note_concession(gave_up=1.0, got=max(0.0, 8.0 - float(budget)))
        except (ImportError, AttributeError, TypeError, ValueError) as exc:
            _record_response_degradation(exc, "UnitaryResponse: price of a dropped draft unread: %s")
        # How much of what they said her own memory has anything like, from
        # this turn's recall. Asking beats assuming where she has not lived it.
        # See `lived_analogue` in core/brain/response_quality.py.
        analogue = None
        try:
            from core.brain.response_quality import lived_analogue

            analogue = lived_analogue(getattr(state, "cognition", None), objective)
        except (ImportError, AttributeError, TypeError, ValueError) as exc:
            _record_response_degradation(exc, "UnitaryResponse: lived analogue unread: %s")
            analogue = None
        try:
            result = await amplify_conversation(
                draft,
                generate=_gen,
                objective=objective,
                user_message=objective,
                grounding_tokens=grounding_tokens,
                word_budget=word_budget,
                n=n_candidates,
                time_budget_s=budget,
                # Witnessing suspends judgement, so a draft for somebody who is
                # testifying is not revised against a taste model. See
                # core/social/witness.py.
                revise=budget >= 6.0 and not is_witnessing(state.cognition),
                conversation_id=_taste_conversation_id(state),
                lived_analogue=analogue,
            )
        except _RESPONSE_RECOVERABLE_ERRORS as exc:
            _record_response_degradation(exc, "UnitaryResponse: conversational amplifier failed: %s")
            return draft
        if result.answer and len(result.answer.strip()) >= 2:
            try:
                if hasattr(state, "metadata") and isinstance(state.metadata, dict):
                    state.metadata["conversation_amplification"] = result.to_dict()
            except (AttributeError, TypeError):
                pass
            # The one that won is what the next candidate is measured against.
            # See core/cognition/convenience.py.
            try:
                from core.cognition.convenience import get_convenience_ledger

                get_convenience_ledger().note_reply(result.answer)
            except (ImportError, AttributeError, TypeError, ValueError) as exc:
                _record_response_degradation(exc, "UnitaryResponse: reply not kept for distinctness: %s")
            # And the regard it carried, as a form this person will take one
            # way or another and as something given.
            # How risky it was to say, before its forms are kept: the reading
            # asks how this person has taken replies shaped like this one,
            # which the reply itself must not yet be part of. See
            # core/soma/on_the_edge.py.
            try:
                from core.soma.on_the_edge import get_edge_ledger, risk_of

                get_edge_ledger().note_sent(
                    risk_of(
                        result.answer,
                        objective,
                        partner=str(getattr(state.cognition, "current_partner", "") or ""),
                    )
                )
            except (ImportError, AttributeError, TypeError, ValueError) as exc:
                _record_response_degradation(exc, "UnitaryResponse: the risk of the reply was not felt: %s")
            try:
                from core.social.the_form_they_welcome import note_sent

                note_sent(result.answer, objective, str(getattr(state.cognition, "current_partner", "") or ""))
            except (ImportError, AttributeError, TypeError, ValueError) as exc:
                _record_response_degradation(exc, "UnitaryResponse: reply forms not kept: %s")
            return result.answer.strip()
        return draft

