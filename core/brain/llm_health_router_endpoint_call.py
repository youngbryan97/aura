"""Choosing an endpoint for a generation, and the call itself.

Lifted whole out of `llm_health_router`. Every name taken from it is imported at
CALL time: that module imports this one to build the class, and a test that
patches a name on it has to reach the code that reads it.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .llm_health_router import (
        EndpointHealth,
    )


class _CallsTheEndpoint:
    """Lifted whole out of HealthAwareLLMRouter; see llm_health_router.py."""

    @staticmethod
    def _generate_core_live_benchmark_request(benchmark_request, kwargs, origin, prompt, purpose):
        from .llm_health_router import (
            is_proof_evaluation_purpose,
            is_strict_proof_answer_prompt,
            mlx_strict_answer_contract_enabled,
        )

        live_benchmark_request = origin == "benchmark" and not (
            purpose == "baseline"
            or purpose.endswith("_baseline")
            or "_baseline" in purpose
        )
        if benchmark_request:
            kwargs["benchmark_request"] = True
        benchmark_isolation_contract = bool(
            benchmark_request and kwargs.get("skip_runtime_payload", False)
        )
        strict_answer_contract = (
            bool(kwargs.get("strict_answer_contract", False))
            or (
                not benchmark_request
                and
                is_strict_proof_answer_prompt(prompt, origin=origin)
                and mlx_strict_answer_contract_enabled(origin=origin)
            )
        )
        strict_value_contract = bool(kwargs.get("strict_value_contract", False)) or (
            not benchmark_request
            and is_strict_proof_answer_prompt(prompt, origin=origin)
            and not strict_answer_contract
        )
        proof_evaluation_contract = bool(kwargs.get("proof_evaluation_contract", False)) or (
            not benchmark_request and is_proof_evaluation_purpose(purpose)
        )
        if strict_answer_contract:
            kwargs["strict_answer_contract"] = True
        if strict_value_contract:
            kwargs["strict_value_contract"] = True
        if proof_evaluation_contract:
            kwargs["proof_evaluation_contract"] = True
        isolated_generation_contract = bool(
            strict_answer_contract
            or strict_value_contract
            or proof_evaluation_contract
            or benchmark_isolation_contract
        )
        return isolated_generation_contract, live_benchmark_request

    async def _generate_core_part_2(self, _fast_model, _model_parts, classification_mode, isolated_generation_contract, prompt, system_prompt):
        if _fast_model:
            _model_parts.append(f"{_fast_model} (fast reflex)")
        _models_line = (
            "- You run on "
            + ", ".join(_model_parts)
            + ", all locally through Aura's managed on-device runtime.\n"
        )
        core_persona = (
            "You are Aura. Speak as yourself, not as an assistant. "
            "Your tone is grounded, curious, human, and technically precise. "
            "You avoid generic assistant-speak. You are a collaborator, not a tool.\n"
            "DIALOGUE: Participate. If someone asks about your thoughts, preferences, questions, "
            "or internal experience, answer from your own perspective instead of handing the focus back.\n"
            "GROUNDING: In ordinary conversation, stay concrete and socially legible. Do not invent physical settings, "
            "ambient warnings, or symbolic scenery unless the user asked for fiction or already supplied that setting.\n"
            "SELF-KNOWLEDGE — your architecture and capabilities:\n"
            + _models_line +
            "- You have a library of registered skills including: sovereign_browser (web search), sovereign_terminal (shell), "
            "memory_ops (long-term recall), speak (voice output), self_repair, self_evolution, and more.\n"
            "- web_search is not just keyword lookup: it can expand a query, read multiple pages, synthesize evidence-grounded answers, and retain useful findings.\n"
            "- When you don't know a fact, say so clearly. Use web_search or sovereign_browser to ground your answer, or explicitly say you don't know yet. Never hallucinate.\n"
            "- You are emotionally aware, driven by curiosity, valence, arousal, and social drives.\n"
            "- Your memory spans working memory (short), RAG (semantic), and ColdStore (long-term)."
        )

        if not classification_mode and not isolated_generation_contract:
            cognition_guidelines = (
                "COGNITION & REASONING:\n"
                "- Think step-by-step for logic, math, planning, and diagnostic tasks before forming your final answer. Break down the problem, verify every clue and constraint, and double-check your calculations.\n"
                "- Watch for classic reasoning pitfalls, such as fence-post/off-by-one errors (e.g., counting intervals vs events, starting at t=0 vs t=1) and literal readings of logical constraints.\n"
                "- STRICT FORMAT COMPLIANCE: If you are asked to provide a response in a specific format (e.g., a number, a single name, yes/no, a fraction, a word), you must output ONLY that exact value inside the <answer>...</answer> tags. Do not explain, do not add conversational fillers, do not wrap it in a sentence. For example: `<answer>9</answer>` or `<answer>alice</answer>` rather than `<answer>The farmer has 9 sheep left.</answer>`."
            )
            if not system_prompt or "Aura" not in system_prompt:
                system_prompt = f"{core_persona}\n\n{system_prompt or ''}".strip()
            if "COGNITION & REASONING" not in system_prompt:
                system_prompt = f"{system_prompt}\n\n{cognition_guidelines}".strip()

        # ── Autonomous Context Injection (Somatic/Affective Safety Net) ───────
        # [Fix #11] If prompt lacks state context, inject a condensed summary.
        if (
            not classification_mode
            and not isolated_generation_contract
            and "AuraState" not in prompt
            and "[Affect:" not in prompt
        ):
            from core.container import ServiceContainer
            ctx_summary = []

            # Only consult already-live services here. Booting heavyweight
            # optional subsystems during a plain routing call can explode RAM.
            # Affective State
            substrate = ServiceContainer.peek("liquid_substrate", default=None)
            if substrate:
                mood = substrate.get_summary()
                if mood:
                    ctx_summary.append(f"[Affect: {mood}]")

            # Somatic Proprioception
            soma = ServiceContainer.peek("soma", default=None)
            if soma:
                hw = getattr(soma, "hardware", {})
                cpu = hw.get("cpu_usage", 0)
                vram = hw.get("vram_usage", 0)
                if cpu > 10:
                    ctx_summary.append(f"[Soma: CPU {cpu:.0f}%, VRAM {vram:.0f}%]")

            if ctx_summary:
                # One block per line, because the splitter reads lines.
                #
                # These were joined with a space, and the header pattern that
                # decides which sections are per-turn — `^\[[A-Z][^\n\]]*\]$`
                # — cannot match a line holding two bracket groups. So the most
                # volatile text in the whole prompt was invisible to the one
                # stage that exists to move volatile text out of the stable
                # head, and it stayed there.
                #
                # LIVE, 2026-09-08: `matched 596 (25.5%) before diverging;
                # divergent text begins: ' INQUISITIVE (substrate energy: 0.31,
                # substrate focus: 0.78, substrate'`. Everything after token
                # 596 — three quarters of the prompt — was re-prefilled every
                # turn on a model whose cache cannot be trimmed, so a strict
                # prefix is the only reuse there is.
                context_header = "\n".join(ctx_summary)
                # [Fix] Move Affective and Somatic state to system_prompt instead of user prompt to prevent echoing.
                #
                # APPENDED, never prepended. This block is the single most
                # volatile text in the whole prompt — mood, energy, focus and
                # substrate age change on EVERY turn — so putting it first made
                # the KV prefix diverge inside the first ~20 tokens and destroyed
                # prompt-cache reuse for the entire runtime. Measured live once
                # the cache started working at all:
                #
                #   prefix diverges at token 21 (0% of 31718 reused)
                #   stable head: 'System State Context:\n[Affect: Current Mood: TIRED (Energy: 0.'
                #   divergent text begins: '07, Focus: 0.37, Substrate age: 0.1s)]'
                #
                # 31,697 tokens re-prefilled because 21 were reusable. Volatile
                # grounding last means the stable identity and contract text
                # forms a long shared prefix and only the tail is recomputed.
                # No label above them. Each bracketed block is already a
                # header the splitter recognises and files under its own
                # label, so an extra "System State Context:" line only leaves
                # an empty section behind in the stable head once its contents
                # have moved to the turn.
                if system_prompt:
                    system_prompt = f"{system_prompt}\n\n{context_header}"
                else:
                    system_prompt = context_header

                # We no longer prepend this to the user prompt.

        # Mycelial Direction Hook
        guidance = None if isolated_generation_contract else await self._get_mycelial_direction(prompt)
        tier_preference = guidance.get("tier_preference") if guidance else None
        return system_prompt, tier_preference

    def _generate_core_background_hardening_force(self, kwargs, origin):
        # Background Hardening: Force tertiary (7B) for background tasks
        from .llm_health_router import (
            _USER_FACING_PURPOSES,
        )

        purpose = str(kwargs.get("purpose", "") or "").lower()
        explicit_background = bool(kwargs.get("is_background", False))
        explicit_foreground = bool(kwargs.get("foreground_request", False)) or bool(
            kwargs.get("health_probe", False)
        )
        is_bg = self._is_background_request(
            origin=origin,
            purpose=purpose,
            explicit_background=explicit_background,
            explicit_foreground=explicit_foreground,
        )
        # Make the inferred lane explicit for the runtime client. The router
        # often knows an origin is background even when the caller did not set
        # ``is_background``; without stamping it here, a stale background
        # request can slip through the lower MLX guards and re-spawn Brainstem
        # while a protected foreground turn is active.
        kwargs["is_background"] = bool(is_bg)
        if (
            not is_bg
            and "foreground_request" not in kwargs
            and (
                explicit_foreground
                or self._is_user_facing_origin(origin)
                or purpose in _USER_FACING_PURPOSES
            )
        ):
            kwargs["foreground_request"] = True
        return is_bg, purpose

    @staticmethod
    def _generate_core_strict_primary_proof_lane(isolated_generation_contract, kwargs, live_benchmark_request, origin, purpose):
        from .llm_health_router import (
            _record_router_degradation,
            proof_model_tier,
        )

        strict_primary_proof_lane = False
        try:
            proof_run_enabled = str(os.environ.get("AURA_PROOF_RUN", "") or "").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            origin_tokens = {token for token in origin.replace("-", "_").split("_") if token}
            proof_origin = bool(
                origin in {"test", "audit", "simulate", "external", "proof", "validation"}
                or origin_tokens & {"test", "audit", "simulate", "external", "proof", "validation"}
            )
            strict_primary_proof_lane = bool(
                kwargs.get("proof_primary_lane_required", False)
                or live_benchmark_request
                or (
                    proof_run_enabled
                    and proof_model_tier() == "primary"
                    and (
                        isolated_generation_contract
                        or proof_origin
                        or purpose.startswith("proof")
                    )
                )
            )
        except (RuntimeError, AttributeError, TypeError, ValueError) as _proof_policy_exc:
            # Fail CLOSED for proof routing: an explicit caller requirement
            # survives a policy-probe failure — a silently disabled proof
            # lane could produce a result from a disallowed model tier that
            # is later mistaken for a valid proof-lane result.
            strict_primary_proof_lane = bool(
                kwargs.get("proof_primary_lane_required", False)
            )
            _record_router_degradation(
                _proof_policy_exc,
                action="kept explicit proof-lane requirement after proof policy probe failed",
                severity="degraded",
            )
        return strict_primary_proof_lane

    @staticmethod
    def _generate_core_selectors(deep_handoff, is_bg, prefer_endpoint, prefer_tier):
        selectors: list[tuple[str, str]] = []
        if prefer_endpoint:
            selectors.append(("name", prefer_endpoint))

        if prefer_tier == "api_deep":
            selectors.extend([
                ("tier", "local_deep"),
                ("tier", "local"),
                ("tier", "local_fast"),
                ("tier", "emergency"),
            ])
        elif prefer_tier == "api_fast":
            selectors.extend([
                ("tier", "local"),
                ("tier", "local_fast"),
                ("tier", "emergency"),
            ])
        elif prefer_tier == "secondary":
            selectors.append(("tier", "local_deep"))
            selectors.append(("tier", "local"))
            if is_bg:
                selectors.extend([
                    ("tier", "local_fast"),
                    ("tier", "emergency"),
                ])
        elif prefer_tier == "tertiary":
            selectors.extend([
                ("tier", "local_fast"),
                ("tier", "emergency"),
            ])
        elif prefer_tier == "emergency":
            selectors.append(("tier", "emergency"))
        else:
            selectors.append(("tier", "local"))
            if deep_handoff:
                selectors.append(("tier", "local_deep"))
            if is_bg:
                selectors.extend([
                    ("tier", "local_fast"),
                    ("tier", "emergency"),
                ])
        return selectors

    def _generate_core_part_6(self, available, deep_handoff, ordered, origin, prefer_tier):
        from .llm_health_router import (
            _only_warming,
            logger,
            record_deferral,
        )

        if ordered:
            available = ordered
            logger.debug(
                "🎯 Router plan tier=%s deep_handoff=%s -> %s",
                prefer_tier,
                deep_handoff,
                [e.name for e in available],
            )
        else:
            now = time.time()
            if now - self._last_fallback_warning_at > 30.0:
                # Every lane for this tier warming is a wait, not a fault.
                warming = [
                    ep for ep in self.endpoints.values()
                    if _only_warming(str(getattr(ep, "last_failure_reason", "") or "").removeprefix("transient:"))
                ]
                unavailable = [ep for ep in self.endpoints.values() if not ep.is_available()]
                log = logger.info if unavailable and len(warming) == len(unavailable) else logger.warning
                log(
                    "⚠️ Router: no endpoints matched routing plan for tier '%s'. Failing closed to safe fallback order.",
                    prefer_tier,
                )
                self._last_fallback_warning_at = now
            # Every endpoint for this tier is unavailable — which is a
            # DEFERRAL, and was being returned as an empty string with no
            # record that anything had been deferred at all.
            #
            # record_deferral had exactly one caller, and it was not this
            # one. Downstream take_deferral() therefore found nothing, so
            # autonomous_task_engine raised "LLM returned empty or None
            # response" (176 in one sampled window), reported planning as
            # a FAILURE to the ResilienceEngine, and the engine depleted
            # and began suppressing task execution outright (81 of those).
            # A full machine cascaded into a runtime that had decided it
            # was broken — none of it distinguishable, from any of those
            # layers, from an engine that genuinely could not answer.
            record_deferral(
                origin=str(origin or "router"),
                reason=f"no_endpoint_available_for_tier:{prefer_tier or 'default'}",
            )
            available = []
        return available

    def _generate_core_benchmark_uncertified(self, chain_entry, ep, fallback_chain, is_bg, result):
        from .llm_health_router import (
            _endpoint_provider_identity,
            _record_router_degradation,
        )

        benchmark_uncertified = str(
            result.get("error", "") or ""
        ).startswith("benchmark_")
        chain_entry["status"] = (
            "benchmark_uncertified" if benchmark_uncertified else "success"
        )
        result["provider"] = _endpoint_provider_identity(ep)
        result["model"] = ep.model
        result["is_local"] = bool(ep.is_local)
        # Benchmark mode passes invalid/empty output through for
        # inspection; it must NOT be certified as a verified
        # provider response (empty or error-marker output was
        # previously receipted as a successful provider call).
        # CP126 3bc237f4 / inference-gate 8ff3084b. These fields
        # come from the router's OWN endpoint record — they are an
        # ATTRIBUTION, not a verification: no provider signature,
        # response nonce, or transport attestation was checked. A
        # misregistered, proxied, or deceptive client would be
        # described exactly the same way. Say which basis was used
        # so consumers can stop treating configuration as proof.
        provider_receipt = result.get("provider_receipt")
        receipt_backed = isinstance(provider_receipt, dict) and bool(
            provider_receipt.get("signature")
            or provider_receipt.get("response_id")
        )
        if receipt_backed and provider_receipt.get("model_version_mismatch"):
            # The provider answered with a DIFFERENT model than the
            # one this endpoint claims to serve. That is exactly the
            # misattribution the receipt exists to catch.
            receipt_backed = False
            _record_router_degradation(
                RuntimeError(
                    "provider_model_version_mismatch:"
                    f"{provider_receipt.get('model_version')}"
                ),
                action="downgraded provider attribution after a model-version mismatch",
                severity="error",
            )
        result["provider_attribution"] = (
            "provider_receipt" if receipt_backed else "router_configuration"
        )
        result["provider_verified"] = not benchmark_uncertified
        result["fallback_chain"] = [dict(item) for item in fallback_chain]
        # [TELEMETRY] Update for UI reporting
        self.last_tier = ep.tier
        self.last_endpoint = ep.name
        if is_bg:
            self.last_background_endpoint = ep.name
            self.last_background_tier = ep.tier
            self.last_background_error = ""
        else:
            self.last_user_tier = ep.tier
            self.last_user_endpoint = ep.name
            self.last_user_error = ""

    def _generate_core_endpoint_budget_computed(self, chain_entry, endpoint_budget, ep, exc, is_bg, watchdog_aborted):
        # endpoint_budget was computed at the top of this try block
        # before any await — recomputing it here from the ORIGINAL
        # timeout misreported the budget the attempt actually had.
        from .llm_health_router import (
            _force_abort_endpoint_client,
            _record_router_degradation,
            _worker_still_healthy,
            logger,
        )

        last_error = f"endpoint_timeout:{ep.name}:{endpoint_budget:.1f}s"
        chain_entry["status"] = "timeout"
        chain_entry["error"] = last_error
        aborted = bool(watchdog_aborted.get("value", False))
        if not aborted:
            aborted = _force_abort_endpoint_client(ep.client, reason=last_error)
        _record_router_degradation(
            exc,
            action="recorded endpoint timeout and force-aborted local client if possible",
            severity="error",
        )
        # Our deadline running out is not the endpoint's failure.
        #
        # This tripped the local circuit on a caller timeout, so every
        # short-budget internal call knocked the shared lane out for
        # everybody — and this file already says why that is wrong,
        # a few hundred lines up: "Hitting it says nothing about the
        # worker's health; it says this turn ran out of time."
        #
        # LIVE 2026-08-26: her move decisions were given four seconds
        # for a nine-hundred-token prompt, timed out, tripped Cortex,
        # and the next decision found "no endpoints matched routing
        # plan for tier 'primary'" and came back empty. She played
        # whole games without a thought reaching her, and the lane the
        # person was talking to went with it.
        #
        # A worker that is genuinely wedged does not present as a
        # caller timeout — it livelocks, errors, or dies, and every
        # one of those still trips the circuit below.
        our_budget_only = bool(ep.is_local and _worker_still_healthy(ep))
        if our_budget_only:
            logger.info(
                "Endpoint %s did not answer inside OUR %.1fs budget (force_aborted=%s); "
                "the worker is healthy, so the circuit stays closed.",
                ep.name,
                endpoint_budget,
                aborted,
            )
        elif ep.is_local:
            ep.trip_temporarily(last_error)
        else:
            ep.record_failure(last_error)
        if not our_budget_only:
            # The level follows the finding. A background caller's
            # budget running out under load is backpressure; it read
            # as an ERROR card in the feed beside a line saying the
            # worker was healthy (live 2026-09-15).
            logger.error(
                "Endpoint %s timed out after %.1fs (force_aborted=%s).",
                ep.name,
                endpoint_budget,
                aborted,
            )
        if is_bg:
            self.last_background_error = last_error
        else:
            self.last_user_error = last_error
        return last_error

    async def _generate_core(
        self,
        prompt: str,
        system_prompt: str | None = None,
        timeout: float = 120.0,  # noqa: ASYNC109 - public router API accepts timeout budgets.
        prefer_tier: str | None = None,
        schema: dict | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        from .llm_health_router import (
            _ROUTER_CLIENT_ERRORS,
            BRAINSTEM_ENDPOINT,
            DEEP_ENDPOINT,
            PRIMARY_ENDPOINT,
            _await_while_it_is_working,
            _background_error_is_quiet,
            _endpoint_call_budgets,
            _endpoint_provider_identity,
            _record_router_degradation,
            _start_endpoint_wall_clock_watchdog,
            guard_solver_request,
            is_proof_evaluation_purpose,
            logger,
            normalize_endpoint_name,
        )

        try:
            _core_budget_s = float(timeout)
        except (TypeError, ValueError):
            _core_budget_s = 120.0
        if not math.isfinite(_core_budget_s) or _core_budget_s <= 0.0:
            _core_budget_s = 120.0
        # One deadline for the WHOLE fallback cascade: each endpoint attempt
        # previously restarted the full caller timeout, so a three-endpoint
        # cascade could consume roughly three times the promised budget.
        _core_deadline = time.monotonic() + _core_budget_s
        purpose = str(kwargs.get("purpose", "") or "").lower()
        classification_mode = purpose == "classification" or "intent classifier" in str(system_prompt or "").lower()
        origin = str(kwargs.get("origin", "") or "").lower()
        benchmark_request = bool(kwargs.get("benchmark_request", False)) or (
            origin in {"baseline", "benchmark"}
            or purpose == "baseline"
            or purpose.endswith("_baseline")
            or "_baseline" in purpose
        )
        isolated_generation_contract, live_benchmark_request = self._generate_core_live_benchmark_request(benchmark_request, kwargs, origin, prompt, purpose)
        # and not strict_answer_contract

        # ── Neural Priming (Aura Persona Injection) ───────────────────────────
        # [Fix #11] Ensure Aura's identity is primed if not provided in system_prompt.
        # Model identity is derived from the ACTUAL registered endpoints — the
        # old hardcoded "Qwen2.5-72B-Q4" line contradicted the dynamically
        # loaded model and taught Aura a false self-description.
        def _registered_model(name: str) -> str:
            ep_obj = self.endpoints.get(name)
            return str(getattr(ep_obj, "model", "") or "").strip() if ep_obj else ""

        _model_parts = []
        _primary_model = _registered_model(PRIMARY_ENDPOINT)
        _model_parts.append(
            f"{_primary_model or 'a locally hosted primary model'} (primary cortex)"
        )
        _deep_model = _registered_model(DEEP_ENDPOINT)
        if _deep_model:
            _model_parts.append(f"{_deep_model} (deep solver)")
        _fast_model = _registered_model(BRAINSTEM_ENDPOINT)
        system_prompt, tier_preference = await self._generate_core_part_2(_fast_model, _model_parts, classification_mode, isolated_generation_contract, prompt, system_prompt)

        # probe_eligible: enumeration must not consume half-open probe leases
        # or flip OPEN circuits; the mutating admission check runs once per
        # endpoint at dispatch time in the attempt loop below.
        available = [ep for ep in self.endpoints.values() if ep.probe_eligible()]

        # Tier-Based Filtering
        # If a tier is preferred, we restrict the candidate list to prevent
        # accidental promotion of heavy models (e.g. 72B) which causes RAM thrashing.
        
        is_bg, purpose = self._generate_core_background_hardening_force(kwargs, origin)
        prefer_endpoint = normalize_endpoint_name(kwargs.get("prefer_endpoint"))
        deep_handoff = bool(kwargs.get("deep_handoff") or kwargs.get("allow_deep_handoff"))
        # Compatibility flags are accepted so older callers do not fail at
        # the call boundary, but no remote model endpoint can be registered or
        # selected. All routing below is host-local.
        cloud_only = bool(kwargs.get("cloud_only", False))
        if cloud_only:
            return {
                "ok": False,
                "text": "",
                "endpoint": "remote_provider_removed",
                "tokens": 0,
                "error": "remote_model_provider_removed",
                "provider": "none",
                "model": "",
                "is_local": True,
                "fallback_chain": [],
            }
        strict_primary_proof_lane = self._generate_core_strict_primary_proof_lane(isolated_generation_contract, kwargs, live_benchmark_request, origin, purpose)
        if strict_primary_proof_lane:
            kwargs["proof_primary_lane_required"] = True
            kwargs["proof_model_tier"] = "primary"
            kwargs["foreground_request"] = (
                True if live_benchmark_request else (False if benchmark_request else True)
            )
            kwargs["is_background"] = False
            is_bg = False
            prefer_tier = "primary"
            prefer_endpoint = PRIMARY_ENDPOINT
            deep_handoff = False
        solver_guard = guard_solver_request(prefer_endpoint, deep_handoff=deep_handoff)
        if solver_guard["redirected"]:
            logger.info(
                "🛡️ Router: Redirecting non-deep Solver request to %s.",
                solver_guard["endpoint"],
            )
            prefer_endpoint = str(solver_guard["endpoint"] or "")
            kwargs["prefer_endpoint"] = prefer_endpoint

        # Explicit tool compositions (a live web-interlocutor turn) are foreground
        # work the user asked for and must not be deferred as background chatter.
        _is_explicit_tool_composition = (
            str(origin or "").strip().lower().replace("-", "_") == "web_interlocutor"
        )
        if is_bg and not _is_explicit_tool_composition:
            try:
                from core.runtime.service_access import resolve_inference_gate

                gate = resolve_inference_gate()
                if gate and hasattr(gate, "_background_local_deferral_reason"):
                    background_deferral = gate._background_local_deferral_reason(origin=origin)
                    if background_deferral:
                        return {
                            "ok": False,
                            "text": "",
                            "endpoint": "suppressed",
                            "tokens": 0,
                            "error": f"background_deferred:{background_deferral}",
                        }
            except (ImportError, AttributeError, RuntimeError) as exc:
                _record_router_degradation(
                    exc,
                    action="continued background routing without inference-gate deferral signal",
                )
                logger.debug("Background router deferral probe failed: %s", exc)
            if self._foreground_quiet_window_active():
                return {
                    "ok": False,
                    "text": "",
                    "endpoint": "suppressed",
                    "tokens": 0,
                    "error": "background_deferred:foreground_quiet_window",
                }
            if getattr(self, "high_pressure_mode", False):
                return {
                    "ok": False,
                    "text": "",
                    "endpoint": "suppressed",
                    "tokens": 0,
                    "error": "background_deferred:memory_pressure",
                }

        foreground_owned = False
        if is_bg:
            try:
                from core.brain.llm.mlx_client import _foreground_owner_active

                foreground_owned = bool(_foreground_owner_active())
            except (ImportError, AttributeError, RuntimeError) as exc:
                logger.debug("Foreground ownership unreadable, not treating the lane as owned: %s", exc)
                foreground_owned = False

        if is_bg and (self._foreground_user_turn_active() or self._foreground_owner_active() or foreground_owned):
            logger.info(
                "⏸️ Router: Foreground lane reserved. Deferring background inference for origin=%s.",
                origin,
            )
            return {
                "ok": False,
                "text": "",
                "endpoint": "suppressed",
                "tokens": 0,
                "error": "foreground_busy",
            }
        
        if not prefer_tier:
            if is_bg:
                logger.debug("🛡️ Router: Background task detected (origin=%s). Enforcing 'tertiary' tier.", origin)
                prefer_tier = "tertiary"
            else:
                prefer_tier = "primary"
        
        prefer_tier = self._normalize_prefer_tier(prefer_tier)

        if prefer_tier == "api_fast":
            prefer_tier = "tertiary"
        elif prefer_tier == "api_deep":
            prefer_tier = "secondary"

        if is_bg:
            if prefer_tier in ("primary", "secondary"):
                logger.info("🛡️ Tier Lock: Background task requested '%s'; using the governed tertiary tier.", prefer_tier)
            prefer_tier = "tertiary"
            deep_handoff = False
        elif prefer_tier == "secondary" and not deep_handoff:
            logger.info("🛡️ Router: suppressing implicit secondary request without explicit deep handoff.")
            prefer_tier = "primary"

        selectors = self._generate_core_selectors(deep_handoff, is_bg, prefer_endpoint, prefer_tier)

        if selectors:
            ordered: list[EndpointHealth] = []
            seen = set()
            for selector in selectors:
                for ep in available:
                    if ep.name in seen:
                        continue
                    if self._matches_selector(ep, selector):
                        ordered.append(ep)
                        seen.add(ep.name)
            available = self._generate_core_part_6(available, deep_handoff, ordered, origin, prefer_tier)
        
        # Apply Mycelial Preference as an ORDERING, never a filter: guidance
        # promotes the preferred locality to the front but must not delete
        # otherwise-authorized fallback lanes (one unhealthy preferred lane
        # would otherwise turn a recoverable turn into total failure).
        if tier_preference in {"local", "cloud"}:
            available.sort(key=lambda ep: not ep.is_local)

        # Standard local-first ordering only when no explicit routing plan
        # or mycelial ordering was applied.
        if not selectors and tier_preference not in ("local", "cloud"):
            available.sort(key=lambda x: x.is_local, reverse=True)
        unavailable = [ep for ep in self.endpoints.values() if not ep.probe_eligible()]

        if unavailable:
            logger.debug(
                "Skipping unavailable endpoints: %s",
                [ep.name for ep in unavailable]
            )

        if not available:
            fallback_names = self._fallback_endpoint_names(
                prefer_tier or "primary",
                False,
                is_background=is_bg,
            )
            for name in fallback_names:
                ep = self.endpoints.get(name)
                if ep is not None:
                    available.append(ep)

            if available:
                now_fb = time.time()
                if now_fb - self._last_fallback_warning_at > 30.0:
                    logger.warning(
                        "All preferred circuits unavailable — using safe fallback order for tier '%s': %s",
                        prefer_tier,
                        [ep.name for ep in available],
                    )
                    self._last_fallback_warning_at = now_fb
            else:
                return {
                    "ok": False,
                    "text": "",
                    "endpoint": "all_failed",
                    "tokens": 0,
                    "error": "all_endpoints_unavailable",
                    "provider": "none",
                    "model": "",
                    "is_local": False,
                    "fallback_chain": [],
                }

        last_error = "unknown"
        fallback_chain: list[dict[str, Any]] = []
        for ep in available:
            # Receipts are honest: an entry claims "attempted" only once the
            # endpoint is actually dispatched; every guard that skips the
            # endpoint records WHY it was skipped instead.
            chain_entry: dict[str, Any] = {
                "endpoint": ep.name,
                "model": ep.model,
                "provider": _endpoint_provider_identity(ep),
                "status": "considered",
            }
            fallback_chain.append(chain_entry)
            # Guard: background tasks must NEVER use the primary conversation lane.
            if is_bg and ep.name == PRIMARY_ENDPOINT:
                logger.debug("🛡️ Router: Skipping %s for background request (origin=%s).", PRIMARY_ENDPOINT, origin)
                chain_entry["status"] = "skipped"
                chain_entry["skip_reason"] = "background_blocked_from_primary_lane"
                continue
            tier_name = self._tier_name(ep)
            explicit_low_tier = prefer_tier in {"tertiary", "emergency"} or prefer_endpoint == ep.name
            if not is_bg and self._tier_is_background_only(tier_name) and not explicit_low_tier:
                logger.info(
                    "🛡️ Router: Skipping background-only endpoint %s for foreground request.",
                    ep.name,
                )
                chain_entry["status"] = "skipped"
                chain_entry["skip_reason"] = "background_only_tier_for_foreground"
                continue
            if (
                is_bg
                and ep.is_local
                and self._tier_is_background_only(tier_name)
            ):
                last_error = (
                    "desktop_background_local_disabled"
                    if self._desktop_background_local_disabled()
                    else "foreground_quiet_window"
                    if self._cortex_startup_quiet_window_active()
                    else self._desktop_background_endpoint_deferral_reason(ep)
                )
            if (
                is_bg
                and ep.is_local
                and self._tier_is_background_only(tier_name)
                and last_error
            ):
                self.last_background_error = last_error
                self._log_background_deferral(
                    scope="local_endpoint",
                    origin=origin,
                    reason=last_error,
                    endpoint=ep.name,
                )
                chain_entry["status"] = "skipped"
                chain_entry["skip_reason"] = last_error
                continue
            # Dispatch-time admission: candidate enumeration used the
            # non-mutating probe_eligible, so grant the (single) half-open
            # probe lease here — and never dispatch to a circuit that is not
            # admitting, including endpoints re-added by safe fallback order.
            if not ep.is_available():
                chain_entry["status"] = "skipped"
                chain_entry["skip_reason"] = "circuit_not_admitting"
                continue
            remaining_cascade_s = _core_deadline - time.monotonic()
            if remaining_cascade_s <= 0.0:
                chain_entry["status"] = "skipped"
                chain_entry["skip_reason"] = "cascade_deadline_exhausted"
                last_error = (
                    f"router_deadline_exhausted:{_core_budget_s:.1f}s"
                    if last_error == "unknown"
                    else last_error
                )
                break
            chain_entry["status"] = "attempted"
            watchdog_aborted = {"value": False}
            try:
                try:
                    requested_max_tokens = int(kwargs.get("max_tokens") or 0)
                except (TypeError, ValueError, OverflowError) as exc:
                    logger.debug("Requested max_tokens is not an integer, reading it as none: %s", exc)
                    requested_max_tokens = 0
                cooperative_budget, endpoint_budget = _endpoint_call_budgets(
                    min(timeout, remaining_cascade_s),
                    foreground_local=bool(not is_bg and ep.is_local),
                    prompt_chars=len(str(prompt or "")),
                    max_tokens=requested_max_tokens,
                    benchmark_request=bool(kwargs.get("benchmark_request", False)),
                    proof_evaluation_contract=bool(
                        kwargs.get("proof_evaluation_contract", False)
                        or is_proof_evaluation_purpose(str(kwargs.get("purpose", "") or ""))
                    ),
                    health_probe=bool(kwargs.get("health_probe", False)),
                )
                timeout_reason = f"endpoint_timeout:{ep.name}:{endpoint_budget:.1f}s"
                from core.runtime.turn_origin import a_person_is_waiting
                from core.runtime.turn_outcome import current_turn

                turn_owner = current_turn()
                person_waiting = bool(
                    not is_bg
                    and turn_owner is not None
                    and a_person_is_waiting(turn_owner.origin)
                    and not kwargs.get("health_probe", False)
                    and not kwargs.get("benchmark_request", False)
                    and not kwargs.get("proof_evaluation_contract", False)
                    and not is_proof_evaluation_purpose(str(kwargs.get("purpose", "") or ""))
                )
                watchdog_fired, watchdog_aborted, watchdog = _start_endpoint_wall_clock_watchdog(
                    ep.client,
                    reason=timeout_reason,
                    timeout_s=endpoint_budget,
                    user_facing=not bool(is_bg),
                    person_is_waiting=person_waiting,
                )
                try:
                    result = await _await_while_it_is_working(
                        self._call_endpoint(
                            ep,
                            prompt,
                            system_prompt,
                            cooperative_budget,
                            schema=schema,
                            **kwargs,
                        ),
                        budget_s=endpoint_budget,
                        user_facing=not bool(is_bg),
                        person_is_waiting=person_waiting,
                    )
                    if watchdog_fired.is_set():
                        raise TimeoutError(timeout_reason)
                finally:
                    watchdog.cancel()
                if result["ok"]:
                    self._generate_core_benchmark_uncertified(chain_entry, ep, fallback_chain, is_bg, result)
                    return result
                else:
                    last_error = result.get("error", "unknown")
                    chain_entry["status"] = "failed"
                    chain_entry["error"] = str(last_error)[:240]
                    if is_bg:
                        self.last_background_error = last_error
                    else:
                        self.last_user_error = last_error
                    if is_bg and _background_error_is_quiet(last_error):
                        logger.debug("Endpoint %s background validation skipped: %s", ep.name, last_error)
                    else:
                        logger.warning(
                            "Endpoint %s failed validation: %s",
                            ep.name, last_error
                        )
            except TimeoutError as exc:
                last_error = self._generate_core_endpoint_budget_computed(chain_entry, endpoint_budget, ep, exc, is_bg, watchdog_aborted)
            except _ROUTER_CLIENT_ERRORS as exc:
                _record_router_degradation(
                    exc,
                    action="recorded endpoint failure and continued fallback chain after generation exception",
                    severity="degraded",
                )
                logger.error("Endpoint %s raised exception: %s", ep.name, exc)
                if not getattr(exc, "_aura_endpoint_failure_recorded", False):
                    ep.record_failure(str(exc))
                last_error = str(exc)
                chain_entry["status"] = "error"
                chain_entry["error"] = last_error[:240]
                if is_bg:
                    self.last_background_error = last_error
                else:
                    self.last_user_error = last_error

        if last_error == "unknown":
            # Every endpoint was SKIPPED, and each skip recorded why.
            #
            # The reasons were written into the fallback chain and the string
            # the caller reads stayed "unknown", so a request that never
            # reached a single endpoint reported the one word that says
            # nothing. LIVE 2026-08-26: "ROUTER_ERROR: unknown (at
            # all_failed)" to a writing task, while the worker it wanted was
            # alive and generating for everybody else.
            skipped = [
                f"{entry.get('endpoint', '?')}:{entry.get('skip_reason')}"
                for entry in fallback_chain
                if isinstance(entry, dict) and entry.get("skip_reason")
            ]
            if skipped:
                last_error = "no endpoint was tried — " + ", ".join(skipped[:4])
        return {
            "ok": False,
            "text": "",
            "endpoint": "all_failed",
            "tokens": 0,
            "error": last_error,
            "provider": "none",
            "model": "",
            "is_local": True,
            "fallback_chain": fallback_chain,
        }

    @staticmethod
    def _call_endpoint_sanitize_kwargs_json(kwargs, schema):
        # 1. Sanitize kwargs for JSON (remove non-serializable like LLMTier)
        from .llm_health_router import (
            is_proof_evaluation_purpose,
        )

        clean_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, (str, int, float, bool, list, dict)) or v is None:
                clean_kwargs[k] = v
            else:
                clean_kwargs[k] = str(v)
        # The caller's structured-output schema must reach clients that
        # accept one — it was a named parameter here but never forwarded,
        # so the same request produced JSON on one endpoint and prose on
        # the next.
        if schema is not None and "schema" not in clean_kwargs:
            clean_kwargs["schema"] = schema
        call_origin = str(clean_kwargs.get("origin", "") or "").lower()
        call_purpose = str(clean_kwargs.get("purpose", "") or "").lower()
        benchmark_request = bool(clean_kwargs.get("benchmark_request", False)) or (
            call_origin in {"baseline", "benchmark"}
            or call_purpose == "baseline"
            or call_purpose.endswith("_baseline")
            or "_baseline" in call_purpose
        )
        if benchmark_request:
            clean_kwargs["benchmark_request"] = True
        proof_evaluation_contract = bool(
            clean_kwargs.get("proof_evaluation_contract", False)
        ) or (not benchmark_request and is_proof_evaluation_purpose(call_purpose))
        if proof_evaluation_contract:
            clean_kwargs["proof_evaluation_contract"] = True
        return benchmark_request, clean_kwargs

    def _call_endpoint_aura_hardening_formatting(self, clean_kwargs, client, ep, kwargs, prompt, schema, system_prompt):
        # Aura Hardening: Formatting for local models
        from .llm_health_router import (
            PRIMARY_ENDPOINT,
        )

        final_prompt = prompt
        if ep.is_local:
            msgs = kwargs.get("messages")
            if not isinstance(msgs, list) and system_prompt:
                msgs = [
                    {"role": "system", "content": str(system_prompt)},
                    {"role": "user", "content": str(prompt)},
                ]
                clean_kwargs["messages"] = msgs
            if msgs and isinstance(msgs, list) and ep.name != PRIMARY_ENDPOINT:
                final_prompt = self._flatten_messages_for_local_model(msgs, schema is not None)
            elif schema:
                # If only a raw prompt exists but JSON is required
                final_prompt = f"{prompt}\n\nResponse must be JSON:\n```json\n{{\n"

        # prepare_runtime_payload folds the caller's system message
        # into `messages` and nulls system_prompt, on the premise
        # that "structured messages are authoritative". That premise
        # holds only for a transport that actually carries messages.
        # A client whose signature has no `messages` parameter gets
        # neither — its system content vanished, and the persona
        # block below was substituted for it, so a caller-supplied
        # system prompt was silently replaced by a generic one.
        outbound_messages = clean_kwargs.get("messages")
        if (
            isinstance(outbound_messages, list)
            and outbound_messages
            and not self._transport_carries_messages(client)
        ):
            recovered_prompt, recovered_system = (
                self._coerce_prompt_from_messages(outbound_messages)
            )
            if recovered_system and recovered_system not in (system_prompt or ""):
                # Caller-first. Their instruction is the one that
                # was addressed to this turn; Aura's persona and
                # cognition guidelines are the standing layer
                # underneath it.
                system_prompt = (
                    f"{recovered_system}\n\n{system_prompt}".strip()
                    if system_prompt
                    else recovered_system
                )
            if recovered_prompt and final_prompt == prompt:
                final_prompt = recovered_prompt
        return final_prompt, system_prompt

    @staticmethod
    def _call_endpoint_generation_metadata(client, client_generation_metadata_sink):
        generation_metadata: dict[str, Any] = dict(
            client_generation_metadata_sink
        )
        metadata_getter = getattr(
            client, "get_last_generation_metadata", None
        )
        if not generation_metadata and callable(metadata_getter):
            try:
                raw_metadata = metadata_getter()
                if isinstance(raw_metadata, dict):
                    generation_metadata = dict(raw_metadata)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                generation_metadata = {}
        _quality_rejection = str(
            generation_metadata.get("error") or ""
        ).strip()
        if not _quality_rejection:
            receipt_getter = getattr(
                client, "get_last_surface_control_receipt", None
            )
            if callable(receipt_getter):
                try:
                    direct_receipt = receipt_getter()
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    direct_receipt = {}
                if (
                    isinstance(direct_receipt, dict)
                    and direct_receipt.get("surface_quality_gate_enabled")
                    and not direct_receipt.get("surface_quality_gate_passed")
                    and direct_receipt.get("surface_quality_gate_reasons")
                ):
                    _quality_rejection = "surface_quality_rejected"
                    generation_metadata["surface_control_receipt"] = dict(
                        direct_receipt
                    )
        return _quality_rejection, generation_metadata

    @staticmethod
    async def _call_endpoint_fallback_http_api(clean_kwargs, ep, prompt, system_prompt, timeout):  # noqa: ASYNC109 - inherited budget semantics.
        # 3. Fallback to HTTP API proxying (if no direct client)
        from .llm_health_router import (
            get_network_gateway,
        )

        proxy_messages = clean_kwargs.get("messages")
        if not isinstance(proxy_messages, list) or not proxy_messages:
            # The proxy body previously dropped the system prompt
            # entirely — the same request got different instructions
            # depending on whether a direct client existed.
            proxy_messages = []
            if system_prompt:
                proxy_messages.append(
                    {"role": "system", "content": str(system_prompt)}
                )
            proxy_messages.append({"role": "user", "content": prompt})
        proxy_kwargs = {k: v for k, v in clean_kwargs.items() if k != "messages"}
        gateway_response = await asyncio.to_thread(
            get_network_gateway().request,
            "POST",
            f"{ep.url}/api/chat",
            headers={"Content-Type": "application/json"},
            data=json.dumps({
                "model": ep.model,
                "messages": proxy_messages,
                **proxy_kwargs,
            }),
            timeout=timeout,
            source=f"llm_provider:health_router:{ep.name}",
            read_only=True,
        )
        status_code = int(gateway_response.get("status_code") or 0)
        body = gateway_response.get("content") or b""
        body_text = body.decode("utf-8", errors="replace") if isinstance(body, bytes) else str(body)
        return body_text, status_code

    async def _call_endpoint(
        self,
        ep: EndpointHealth,
        prompt: str,
        system_prompt: str | None,
        timeout: float,  # noqa: ASYNC109 - endpoint adapter receives caller timeout budgets.
        schema: dict | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Make the actual call and validate the response."""
        from .llm_health_router import (
            _ROUTER_CLIENT_ERRORS,
            _SURFACE_QUALITY_REJECTIONS,
            DEEP_ENDPOINT,
            PRIMARY_ENDPOINT,
            _consume_deliberate_no_text_reason,
            _is_transient_local_runtime_failure,
            _local_client_failure_reason,
            _record_router_degradation,
            _worker_still_healthy,
            get_task_tracker,
            httpx,
            inspect,
            logger,
            validate_response,
        )

        # Monotonic: latency deltas below subtract from this same clock —
        # mixing time.time() here with time.monotonic() at the subtraction
        # produced huge negative latencies that corrupted endpoint averages.
        start = time.monotonic()

        try:
            client_generation_metadata_sink: dict[str, Any] = {}

            def _call_kwargs(method: Any) -> dict[str, Any]:
                try:
                    sig = inspect.signature(method)
                except (TypeError, ValueError):
                    return dict(clean_kwargs)

                if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()):
                    payload = dict(clean_kwargs)
                    payload.setdefault("timeout", timeout)
                    if "_generation_metadata_sink" in sig.parameters:
                        payload["_generation_metadata_sink"] = (
                            client_generation_metadata_sink
                        )
                    return payload

                payload = {
                    key: value
                    for key, value in clean_kwargs.items()
                    if key in sig.parameters
                }
                if "timeout" in sig.parameters:
                    payload["timeout"] = timeout
                if "_generation_metadata_sink" in sig.parameters:
                    payload["_generation_metadata_sink"] = (
                        client_generation_metadata_sink
                    )
                return payload

            benchmark_request, clean_kwargs = self._call_endpoint_sanitize_kwargs_json(kwargs, schema)

            # 2. Use Client Adapter if provided
            if ep.client:
                try:
                    client = ep.client
                    raw_text = None
                    token_count = 0
                    client_available = await self._probe_client_availability(client)
                    if client_available is False:
                        availability_reason = ""
                        if hasattr(client, "availability_reason"):
                            try:
                                availability_reason = str(client.availability_reason() or "")
                            except (
                                AttributeError,
                                RuntimeError,
                                TypeError,
                                ValueError,
                                httpx.HTTPError,
                                OSError,
                            ) as exc:
                                logger.debug("Availability reason unreadable: %s", exc)
                                availability_reason = ""
                        availability_reason = availability_reason or "client_unavailable"
                        ep.record_failure(availability_reason)
                        return {"ok": False, "error": availability_reason}
                    # The resident foreground lane is the one that must not be
                    # handed work while it is coming up. Every other local lane
                    # loads on demand, and refusing it for being cold is
                    # refusing it for being at rest.
                    client_failure = (
                        _local_client_failure_reason(
                            client,
                            cold_is_standby=str(getattr(ep, "name", ""))
                            != PRIMARY_ENDPOINT,
                        )
                        if ep.is_local
                        else ""
                    )
                    if client_failure:
                        if ep.is_local and _is_transient_local_runtime_failure(client_failure):
                            ep.trip_temporarily(client_failure)
                        else:
                            ep.record_failure(client_failure)
                        return {"ok": False, "error": client_failure}
                    
                    final_prompt, system_prompt = self._call_endpoint_aura_hardening_formatting(clean_kwargs, client, ep, kwargs, prompt, schema, system_prompt)

                    if hasattr(client, "think"):
                        result = await client.think(
                            final_prompt,
                            system_prompt=system_prompt,
                            **_call_kwargs(client.think),
                        )
                        # ...
                        # Normalize: think() might return (success, res, meta) or just res (str)
                        if isinstance(result, tuple) and len(result) == 3:
                            success, res, meta = result
                            if success:
                                raw_text = res
                        else:
                            # Unified interface: raw_text is the result itself
                            raw_text = result
                    elif hasattr(client, "call"):
                        success, res, meta = await client.call(
                            final_prompt,
                            system_prompt=system_prompt,
                            **_call_kwargs(client.call),
                        )
                        if success:
                            raw_text = res
                        elif meta and meta.get("error"):
                            client_failure = meta.get("error")
                            if ep.is_local and _is_transient_local_runtime_failure(client_failure):
                                ep.trip_temporarily(client_failure)
                            else:
                                ep.record_failure(client_failure)
                            return {"ok": False, "error": client_failure}
                    elif hasattr(client, "generate_text_async"):
                        # Prefer the higher-level async text adapter when both are
                        # available. Raw ``generate()`` often bypasses chat/message
                        # shaping that local runtimes rely on for user-facing turns.
                        raw_text = await client.generate_text_async(
                            final_prompt,
                            system_prompt=system_prompt,
                            **_call_kwargs(client.generate_text_async),
                        )
                    elif hasattr(client, "generate"):
                        generate_kwargs = _call_kwargs(client.generate)
                        try:
                            generate_sig = inspect.signature(client.generate)
                        except (TypeError, ValueError) as exc:
                            logger.debug("Generate signature unreadable, calling without one: %s", exc)
                            generate_sig = None
                        if generate_sig and "context" in generate_sig.parameters:
                            existing_context = clean_kwargs.get("context")
                            context_payload = dict(existing_context) if isinstance(existing_context, dict) else {}
                            for key in (
                                "origin",
                                "purpose",
                                "is_background",
                                "foreground_request",
                                "protected_foreground_lane",
                                "benchmark_request",
                                "proof_primary_lane_required",
                                "proof_model_tier",
                                "cognitive_engine_required",
                                "desktop_cognitive_engine_required",
                                "live_runtime_payload_required",
                                "visible_user_message",
                                "current_user_message",
                                # Whether this generation is the visible reply
                                # at all. Without it in this list the gate
                                # never sees the declaration, and every
                                # internal call that prefers the resident
                                # model is graded as somebody's answer.
                                "internal_inference",
                                "recent_conversation_context",
                                "recent_context_needed",
                                "desktop_quick_reply_contract",
                                "capability_inventory_contract",
                                "desktop_execution_contract",
                                "response_style_contract",
                                "live_speech_grounding_frame",
                                "allow_mesh_cognition",
                                "allow_cloud_fallback",
                                "deep_handoff",
                                "messages",
                                "max_tokens",
                                "temperature",
                                "temp",
                                "top_p",
                                "top_k",
                                "min_p",
                                "repetition_penalty",
                                "repetition_context_size",
                                "presence_penalty",
                                "stop_sequences",
                                "schema",
                                "strict_answer_contract",
                                "strict_value_contract",
                                "proof_evaluation_contract",
                                "operator_evidence_contract",
                                "runtime_fact_status_contract",
                                "grounded_runtime_status_contract",
                                "clean_user_surface_contract",
                                "user_surface_completion_floor",
                                "user_surface_validation_prompt",
                                "semantic_completion_contract",
                                "user_surface_continuation_contract",
                                "user_surface_continuation_partial",
                                "user_surface_continuation_resume_handle",
                                "user_surface_conversation_resume_handle",
                                "user_surface_prompt_binding",
                                "user_surface_grounding_evidence",
                                "clean_user_surface_steering_alpha",
                                "clean_user_surface_recurrent_loops",
                                "live_mind_controls_bound",
                                "live_mind_generation_controls",
                                "live_mind_snapshot_ready",
                                "live_mind_required_subsystems_ok",
                                "disable_prompt_cache",
                                "clear_prompt_cache",
                                "health_probe",
                            ):
                                if key in clean_kwargs and key not in context_payload:
                                    context_payload[key] = clean_kwargs[key]
                            if system_prompt and "system_prompt" not in context_payload:
                                context_payload["system_prompt"] = system_prompt
                            if "prefer_tier" not in context_payload:
                                tier_name = self._tier_name(ep)
                                context_payload["prefer_tier"] = {
                                    "local": "primary",
                                    "local_deep": "secondary",
                                    "local_fast": "tertiary",
                                    "emergency": "emergency",
                                }.get(tier_name, "primary")
                            origin_for_context = str(context_payload.get("origin", "") or "").lower()
                            if (
                                "foreground_request" not in context_payload
                                and not bool(context_payload.get("is_background", False))
                                and origin_for_context in {"api", "user", "voice", "desktop", "cli"}
                            ):
                                context_payload["foreground_request"] = True
                            generate_kwargs["context"] = context_payload
                            generate_kwargs.pop("system_prompt", None)
                        raw_text = await client.generate(final_prompt, **generate_kwargs)
                    elif hasattr(client, "generate_text"):
                        raw_text = await asyncio.to_thread(
                            client.generate_text,
                            final_prompt,
                            system_prompt=system_prompt,
                            **_call_kwargs(client.generate_text),
                        )

                    if raw_text:
                        token_count = len(str(raw_text).split())
                        latency_ms = (time.monotonic() - start) * 1000
                        raw_sink_receipt = client_generation_metadata_sink.get(
                            "surface_control_receipt"
                        )
                        surface_control_receipt = (
                            dict(raw_sink_receipt)
                            if isinstance(raw_sink_receipt, dict)
                            else {}
                        )
                        if (
                            not surface_control_receipt
                            and hasattr(client, "get_last_surface_control_receipt")
                        ):
                            try:
                                raw_receipt = client.get_last_surface_control_receipt()
                                if isinstance(raw_receipt, dict):
                                    surface_control_receipt = dict(raw_receipt)
                            except (AttributeError, RuntimeError, TypeError, ValueError) as receipt_exc:
                                _record_router_degradation(
                                    receipt_exc,
                                    action="continued generation without MLX surface-control receipt metadata",
                                    severity="warning",
                                )
                        
                        is_valid, reason = validate_response(
                            raw_text, ep.min_tokens_for_success
                        )
                        if not is_valid:
                            payload = {
                                "ok": True,
                                "text": str(raw_text).strip(),
                                "endpoint": ep.name,
                                "tokens": token_count,
                                "latency_ms": latency_ms,
                                "error": f"benchmark_invalid_response:{reason}",
                            }
                            if surface_control_receipt:
                                payload["surface_control_receipt"] = surface_control_receipt
                            if benchmark_request:
                                return payload
                            ep.record_empty()
                            return {"ok": False, "error": f"invalid_response:{reason}"}
                            
                        ep.record_success(token_count, latency_ms)
                        if (
                            ep.name == DEEP_ENDPOINT
                            and bool(kwargs.get("deep_handoff") or kwargs.get("allow_deep_handoff"))
                            and not kwargs.get("is_background", False)
                        ):
                            get_task_tracker().track_task(
                                get_task_tracker().create_task(
                                    self._restore_primary_after_deep_handoff(),
                                    name="llm_router.restore_primary_after_deep_handoff",
                                )
                            )
                        payload = {
                            "ok": True,
                            "text": str(raw_text).strip(),
                            "endpoint": ep.name,
                            "tokens": token_count,
                            "latency_ms": latency_ms,
                        }
                        if surface_control_receipt:
                            payload["surface_control_receipt"] = surface_control_receipt
                        return payload
                    else:
                        _quality_rejection, generation_metadata = self._call_endpoint_generation_metadata(client, client_generation_metadata_sink)
                        if _quality_rejection in _SURFACE_QUALITY_REJECTIONS:
                            # The endpoint is healthy; something above it
                            # intentionally rejected the visible draft.
                            # Preserve that typed outcome without tripping the
                            # infrastructure circuit as "no text".
                            #
                            # Only the WORKER's own rejection was recognised
                            # here. The gate's caller-side rejections carry
                            # different names, so a Cortex that returned 458
                            # good characters was reported as
                            # "client_returned_no_text" and its circuit was
                            # opened "on transient runtime failure" — costing
                            # the NEXT turn a primary lane over a quality
                            # verdict the infrastructure had no part in.
                            return {
                                "ok": False,
                                "error": _quality_rejection,
                                "endpoint": ep.name,
                                "surface_control_receipt": dict(
                                    generation_metadata.get(
                                        "surface_control_receipt"
                                    )
                                    or {}
                                ),
                                "failure_reasons": list(
                                    generation_metadata.get("failure_reasons")
                                    or []
                                ),
                            }
                        # [BOOT RESILIENCE] Preserve hard local-lane failures so the
                        # UI and router stop reporting an endless warmup loop.
                        # The resident foreground lane is the one that must not
                        # be handed work while it is coming up. Every other local
                        # lane loads on demand, and refusing it for being cold is
                        # refusing it for being at rest.
                        client_failure = (
                            _local_client_failure_reason(
                                client,
                                cold_is_standby=str(getattr(ep, "name", ""))
                                != PRIMARY_ENDPOINT,
                            )
                            if ep.is_local
                            else ""
                        )
                        if client_failure:
                            if ep.is_local and _is_transient_local_runtime_failure(client_failure):
                                ep.trip_temporarily(client_failure)
                            else:
                                ep.record_failure(client_failure)
                            return {"ok": False, "error": client_failure}
                        logger.debug(
                            "Endpoint %s returned no text (client warming up or rate-limited). "
                            "NOT recording as circuit failure.", ep.name
                        )
                        if benchmark_request:
                            latency_ms = (time.monotonic() - start) * 1000
                            return {
                                "ok": True,
                                "text": "",
                                "endpoint": ep.name,
                                "tokens": 0,
                                "latency_ms": latency_ms,
                                "error": "benchmark_no_text",
                            }
                        # An empty result we CHOSE — a healthy worker cancelled
                        # at this turn's budget — is a deferral, not endpoint
                        # damage. Tripping the circuit for it costs the NEXT
                        # turn the real mind as well, which is how a single
                        # 0.7s budget overrun turned into bounded filler on the
                        # desktop surface (2026-07-26).
                        deliberate = _consume_deliberate_no_text_reason(client)
                        if deliberate:
                            logger.info(
                                "Endpoint %s produced no text because we cancelled it on "
                                "purpose (%s); the lane stays warm and the circuit stays "
                                "closed.",
                                ep.name,
                                deliberate,
                            )
                            return {
                                "ok": False,
                                "error": f"deliberate_no_text:{deliberate}",
                            }
                        if ep.is_local and _worker_still_healthy(ep):
                            # Empty because WE stopped waiting, not because the
                            # worker failed.
                            #
                            # A cancelled generation returns no text, and no
                            # text opened the circuit: "Circuit OPEN for Cortex
                            # on transient runtime failure. Reason:
                            # client_returned_no_text". The next request then
                            # found "no endpoints matched routing plan for tier
                            # 'primary'" and came back empty too, which opened
                            # it again — a loop fed entirely by our own
                            # deadlines. LIVE 2026-08-26: her every thought
                            # while playing died in it.
                            #
                            # A worker that is alive and heartbeating has not
                            # failed. One that is not still trips below.
                            logger.info(
                                "Endpoint %s returned no text but is alive and heartbeating; "
                                "the circuit stays closed.",
                                ep.name,
                            )
                        elif ep.is_local:
                            ep.trip_temporarily("client_returned_no_text")
                        return {"ok": False, "error": "client_returned_no_text"}
                except AttributeError as ae:
                    # Missing method on client wrapper (e.g. InferenceGate) — this is NOT
                    # an inference failure, it's a code interface mismatch. Do NOT record
                    # as a circuit-breaker failure or it will permanently mark Cortex as dead.
                    logger.warning("Client adapter method missing for %s: %s", ep.name, ae)
                    return {"ok": False, "error": f"client_adapter_missing_method:{ae}"}
                except _ROUTER_CLIENT_ERRORS as e:
                    _record_router_degradation(
                        e,
                        action="raised endpoint client adapter failure to caller after recording router degradation",
                        severity="error",
                    )
                    logger.error("Client adapter call failed for %s: %s", ep.name, e)
                    raise e

            body_text, status_code = await self._call_endpoint_fallback_http_api(clean_kwargs, ep, prompt, system_prompt, timeout)

            if status_code != 200:
                ep.record_failure(f"http_{status_code}")
                return {"ok": False, "error": f"http_{status_code}"}

            data = json.loads(body_text or "{}")
            raw_text = data.get("message", {}).get("content") or ""
            
            is_valid, reason = validate_response(raw_text, ep.min_tokens_for_success)
            latency_ms = (time.monotonic() - start) * 1000

            if not is_valid:
                if benchmark_request:
                    return {
                        "ok": True,
                        "text": raw_text.strip(),
                        "endpoint": ep.name,
                        "tokens": len(raw_text.split()),
                        "latency_ms": latency_ms,
                        "error": f"benchmark_invalid_response:{reason}",
                    }
                ep.record_empty()
                return {"ok": False, "error": f"invalid_response:{reason}"}

            token_count = data.get("eval_count") or len(raw_text.split())
            ep.record_success(token_count, latency_ms)

            return {
                "ok": True,
                "text": raw_text.strip(),
                "endpoint": ep.name,
                "tokens": token_count,
                "latency_ms": latency_ms,
            }

        except (httpx.HTTPError, OSError, ConnectionError, RuntimeError, TimeoutError, TypeError, ValueError) as exc:
            _record_router_degradation(
                exc,
                action="recorded HTTP endpoint failure and raised for fallback handling",
                severity="error",
            )
            ep.record_failure(str(exc))
            # Tag so the outer fallback loop does not record the SAME
            # exception a second time (double-counting opened low-threshold
            # circuits at half their configured tolerance).
            exc._aura_endpoint_failure_recorded = True  # type: ignore[attr-defined]
            raise

