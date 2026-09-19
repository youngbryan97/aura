"""The living context a turn is answered inside, and the tool evidence it rests on.

Lifted whole out of InferenceGate, which is the second largest module in
this tree. Assembling what she knows right now and deciding whether a
grounded tool result is present are one subject: both answer what the model
is allowed to treat as true for this turn.

Every name taken from `inference_gate` is imported at CALL time. The module
these came from imports this one to build the class, so a module-level
import would be a cycle, and a test patches the name on `inference_gate`
rather than on wherever it came from.
"""
from __future__ import annotations

import asyncio
import os
from typing import Any

from core.brain.living_mind_context import PRIORITY_COLOUR, TRUST_LEARNED
from core.utils.completed_capability import remaining_capabilities


class _BuildsTheLivingContext:
    """Lifted whole out of InferenceGate; see inference_gate.py."""

    @staticmethod
    async def _tool_grounded_answer_part_1(ceiling, client, decode_budget, evidence, origin, required, text, timeout_s, tools):
        from .inference_gate import (
            _REQUESTED_ARTIFACT_EFFECT_CEILING,
            _SELF_SERVICE_EFFECT_CEILING,
            _answer_reserve_seconds,
            _tool_loop_budget,
            logger,
        )

        logger.info(
            "🔧 Tool handoff: wanted=%s offered=%s",
            ",".join(required),
            ",".join(sorted(tools)),
        )
        # The receipts this loop writes belong to the turn that started it.
        # That used to need a hand-threaded lease, because custody was
        # keyed on the exact (thread, task) that opened the turn and this
        # loop does not always run there. Belonging is inherited from the
        # turn's context now, so the loop needs nothing to be part of it.
        # The tool loop is the part of a turn that takes the time, and it
        # was the one clock left holding an absolute number. Live on
        # 2026-08-28 a ledgerkit turn read three files and died here at
        # 138.9 seconds while every clock around it was holding itself
        # open, because this one still counted.
        from core.brain.llm_health_router import _await_while_it_is_working

        result = await _await_while_it_is_working(
            client.think_and_act(
                objective=text,
                # The budget this turn decided on, not the client's default.
                #
                # LIVE, 2026-08-28: "read the docs, then use it" read three
                # files, said "Running it now:", and emitted a code_repl
                # call whose argument was cut off mid-import. The turn's
                # clock had allocated 1536 tokens and every generation in
                # the loop got 399, so the narration and the opening of the
                # program together reached the ceiling and the call was
                # never a call — it arrived as prose, was judged prose
                # containing prompt scaffolding, and was correctly refused.
                #
                # She decided to run the code. The room to say so was the
                # thing missing, and the room had already been worked out
                # one function up.
                **(
                    {"max_tokens": int(decode_budget)}
                    if int(decode_budget or 0) > 0
                    else {}
                ),
                # An execution turn is not a conversation turn.
                #
                # The foreground system prompt is the full conversational
                # scaffold — persona, instruments, present moment, running
                # to five thousand tokens. Wrapped around a tool call it
                # produced an immediate end-of-turn: one token, no text,
                # every time. A call needs the objective and the tools; the
                # voice belongs to the reply, which is generated
                # separately.
                #
                # It is also most of the latency of a tool turn.
                system_prompt="",
                tools=tools,
                # A step, a look at what it returned, and a chance to do
                # something else because of it — three is one attempt with
                # no room to be wrong. Scaled to the working set so a
                # single-capability turn stays cheap.
                #
                # And one more than the calls, because the last turn has to
                # be free to WRITE. Two tools gave five turns; a turn that
                # was refused twice and then read three files used all five
                # on calls, and the person got a list of what ran instead of
                # an answer.
                #
                # LIVE, 2026-08-28: "read the docs, then actually use it"
                # spent turns on a denied path, a refused execution, and
                # three successful reads. Nothing was left to say what it
                # had found.
                max_turns=max(4, 2 * len(tools) + 2),
                context={
                    "required_skills": list(required),
                    "foreground_request": True,
                    # Who asked, and what they said.
                    #
                    # The conscience holds a skill whose worst case looks
                    # harmful unless a person asked for it directly, in the
                    # foreground, on their own machine — and it decides
                    # that from the origin and the message on this context.
                    # Neither was here, so every dispatch arrived as
                    # origin=unknown and the override could not fire.
                    #
                    # LIVE, 2026-08-29: asked to use a library at a named
                    # path, the model called code_repl with that path, was
                    # held at "worst-case harm 0.80", tried sys, importlib
                    # and exec in turn — each correctly refused — and came
                    # back to the right call, which was held again. The
                    # person had asked for it in those words.
                    "origin": origin or "user",
                    "message": text,
                    # The fact, rather than a name to be parsed again. This
                    # gate already decided whether somebody is waiting on
                    # this turn; the conscience downstream needs the same
                    # answer, and deriving it twice from origin strings is
                    # how the two came to disagree.
                    "a_person_is_waiting": True,
                    # What this turn may do. The dispatch refuses any
                    # action ranked above it, so a skill can be offered
                    # for its safe actions without offering its
                    # dangerous ones.
                    "authorised_effect_scope": ceiling,
                    # Consent the request itself carries.
                    #
                    # The permission model already asks whether the person
                    # pre-approved this class of action, and nothing ever
                    # answered. So "build me a small web app, one
                    # self-contained file" was refused with "Requires user
                    # confirmation" — a confirmation prompt for the thing
                    # that had just been asked for in those words.
                    #
                    # Deliberately narrow, and it was narrower than the
                    # thing it was arguing for.
                    #
                    # Set only for the artifact ceiling, it left the
                    # SELF-SERVICE ceiling asking for a confirmation
                    # nobody can give — and that ceiling is defined, where
                    # it is declared, as "the most a turn may do without
                    # the person having asked for that effect... it can
                    # calculate anything and change nothing outside its own
                    # sandbox". Something that by definition needs no
                    # permission was being refused for want of one.
                    #
                    # LIVE, 2026-08-28: "read the docs, then actually use
                    # it" reached code_repl and came back "Permission
                    # denied: Requires user confirmation: Typed execution
                    # contract: scope=sandboxed_compute". She read the
                    # library three times over and never ran it.
                    #
                    # Still narrow: these are the two ceilings a request
                    # can establish for itself. Nothing here authorises
                    # external_io, privileged mutation, deleting, sending
                    # or spending — those need their own consent, because
                    # nobody asked for them.
                    "user_explicitly_authorized": (
                        ceiling
                        in {
                            _SELF_SERVICE_EFFECT_CEILING,
                            _REQUESTED_ARTIFACT_EFFECT_CEILING,
                        }
                    ),
                },
                # What the turn has already read. Without it the loop
                # fetched the same document a second time, from a URL it
                # rebuilt from memory, and got a 400.
                evidence=evidence,
            ),
            # The tool loop's job is to GET the evidence; the reply's job
            # is to SAY it, and evidence nobody can say is worth nothing.
            #
            # LIVE, 2026-08-27: a repository diagnosis ran in 389ms and
            # came back complete — the contradiction, the line, the
            # project's own broken invariant. The tool-calling pass had
            # taken 65s of a 148s turn and the presenting pass was refused
            # before dispatch, "because the request budget was already
            # spent". The answer was in hand and there was no time left to
            # say it.
            budget_s=_tool_loop_budget(
                timeout_s,
                _answer_reserve_seconds(
                    client,
                    # What the answer will be read from: everything handed
                    # to the loop, because that is what comes back with it.
                    len(str(text or ""))
                    + sum(
                        len(str((item or {}).get("content") or ""))
                        for item in (evidence or [])
                        if isinstance(item, dict)
                    ),
                ),
            ),
            user_facing=True,
            # The same fact the loop puts on its own context. A person
            # asked for this in the foreground and is sitting in front of
            # it, so the bound is the turn's ceiling rather than a
            # multiple of one step's budget — which cut a loop that was
            # producing 0.2s earlier, LIVE 2026-08-29.
            person_is_waiting=True,
        )
        return result

    def _tool_grounded_answer_model_path(self, called, client, text):
        from .inference_gate import (
            logger,
        )

        model_path = str(getattr(client, "model_path", "") or "").strip()
        if text and model_path:
            # The receipt for the work, carried with the record of it.
            #
            # Ownership of a foreground answer is proven by a surface-control
            # receipt with a token count in it: the resident model generated
            # these words, on this turn, once. This path recorded that a tool
            # loop had run and nothing about the generation inside it, so an
            # answer the 27B plainly wrote could not be shown to have been
            # written by anything.
            #
            # LIVE 2026-08-29: six tool calls, the last one returning the
            # trial balance, an answer composed from them and trimmed, and
            # then "missing: foreground_model_generation_ownership_unproven"
            # with generations=0 consumed=False. The turn failed closed on
            # bookkeeping for work it had done.
            tool_loop_metadata = {
                "provider": "mlx_local",
                "model": model_path,
                "endpoint": os.path.basename(model_path),
                "is_local": True,
                "provider_verified": True,
                "tool_loop": True,
                "tool_calls": len(called),
            }
            receipt = None
            reader = getattr(client, "get_last_surface_control_receipt", None)
            if callable(reader):
                try:
                    receipt = reader()
                except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                    logger.debug(
                        "tool loop could not read the surface control receipt: %s", exc
                    )
            if isinstance(receipt, dict) and receipt:
                tool_loop_metadata["live_mind_surface_control_receipt"] = dict(receipt)
            self._record_client_generation_metadata(
                client,
                label=os.path.basename(model_path),
                success=True,
                text=text,
                generation_metadata=tool_loop_metadata,
            )
            # And on the turn, which is the one thing every path shares.
            #
            # The line above publishes on this gate. The layer that checks
            # authorship reads the health router. Both are right about their
            # own object, and the answer falls between them — LIVE 2026-08-29,
            # "ownership_evidence=[live_mind(tokens=-,decode=-); latent_cortex
            # (decode=0)]" on a turn whose fifth tool call had just returned
            # the trial balance.
            _tokens = 0
            for _key in ("generated_tokens", "decode_generated_tokens"):
                _value = (receipt or {}).get(_key)
                if isinstance(_value, int) and _value > 0:
                    _tokens = _value
                    break
            if _tokens <= 0:
                # What the client counted arriving, when the worker attached
                # no total. The tokens are the same tokens; only the reporting
                # differs, and an answer must not fail to prove itself because
                # of which branch of the worker replied.
                _counter = getattr(client, "tokens_generated_for_this_request", None)
                if callable(_counter):
                    try:
                        _tokens = max(0, int(_counter() or 0))
                    except (TypeError, ValueError) as exc:
                        logger.debug("Token counter is not an integer, counting none: %s", exc)
                        _tokens = 0
            try:
                from core.conversation.turn_evidence_custody import (
                    record_turn_model_generation,
                )

                _recorded = (
                    record_turn_model_generation(
                        model_path, tokens=_tokens, path="tool_loop"
                    )
                    if _tokens > 0
                    else False
                )
            except (ImportError, RuntimeError, TypeError, ValueError) as exc:
                _recorded = False
                logger.debug("tool loop could not record its generation: %s", exc)
            # Which of the two, when the answer cannot prove who wrote it.
            #
            # A receipt with no count and a turn that would not take the
            # record are different faults: the first is a generation whose
            # tokens nobody added up, the second is custody this execution
            # does not belong to. Both end as
            # "foreground_model_generation_ownership_unproven" and the name
            # says neither.
            if not _recorded:
                logger.info(
                    "🧾 tool loop generation unrecorded: tokens=%d receipt_keys=%s",
                    _tokens,
                    ",".join(sorted(receipt or {}))[:300] or "none",
                )

    async def _tool_grounded_answer(
        self,
        client: Any,
        *,
        visible: Any,
        system_prompt: Any,
        timeout_s: float,
        evidence: Any = None,
        completed_capability_evidence: Any = None,
        allow_tools: bool = True,
        decode_budget: int = 0,
        origin: str = "",
    ) -> str | None:
        """Answer by running the capability the request needs, or return None.

        LIVE DEFECT, 2026-08-19. Asked to run Python and report the number,
        with code_repl READY, she wrote a snippet and stated an invented
        "Output:". The runtime HAS a tool loop — parse a call, bind it to the
        tool's advertised schema, execute, feed the result back — and reaching
        it goes through `should_force_tool_handoff` in the health router. Chat
        never gets there: this lane calls the MLX client directly
        (`local_client = self._mlx_client`), so the router's contract, its
        handoff, and the loop behind it apply to every OTHER caller and not to
        the one people actually type into.

        Returns text only when a tool really ran. Anything else — no
        capability needed, no tool map, the model declining to call — returns
        None so the ordinary generation proceeds untouched.
        """
        from .inference_gate import (
            _INFERENCE_RECOVERABLE_ERRORS,
            _tools_within_reach,
            _what_a_tool_returned,
            logger,
            record_degradation,
        )

        # A typed completion/continuation turn is an answer segment, not a new
        # execution request. Enforce that before capability inference so words
        # such as "code" or "build" inside a correction cannot open a tool
        # lane the caller explicitly closed.
        if not allow_tools:
            return None

        # The person's own words, not the assembled prompt. The scaffold runs
        # to thousands of characters around a request of a hundred, and asking
        # a question about the whole envelope answers about the envelope.
        text = str(visible or "").strip()
        if not text:
            return None
        try:
            from core.brain.llm.runtime_wiring import build_agentic_tool_map
            from core.phases.response_contract import (
                derive_capability_set,
                requested_effect_ceiling,
            )

            # The same ceiling selection used. Offering a capability the
            # dispatch then refuses is worse than not offering it: the turn
            # spends itself reaching for something it was never allowed to
            # use, which is how "build me a web app" ended in an executive
            # veto on code_repl.
            ceiling, allowed_scopes = requested_effect_ceiling(text)

            required = derive_capability_set(text)
            if not required:
                return None
            pending = remaining_capabilities(required, completed_capability_evidence)
            if required and not pending:
                logger.info(
                    "🔧 Tool handoff skipped: every required capability already "
                    "has runtime-stamped evidence (%s).",
                    ",".join(sorted(required)),
                )
                return None
            required = pending
            tools = build_agentic_tool_map(
                required, objective=text, max_tools=len(required)
            )
            if not tools:
                logger.info(
                    "🔧 Tool handoff: skill=%s offered=NONE (no tool definition)",
                    ",".join(required),
                )
                return None
            # The ceiling above was computed and then discarded, so the rule
            # the comment states was never enforced.
            #
            # LIVE, 2026-08-25: asked to diagnose a project, the turn was
            # offered diagnose_repo, code_repl and file_operation. It reached
            # for file_operation, which is state_mutation and above this
            # turn's sandboxed_compute ceiling, and the executive vetoed it;
            # then for code_repl, which the permission model refused for want
            # of a confirmation nobody could give. Two of the turn's two tool
            # calls were spent on tools that could never have run, and the
            # one that would have answered was never called.
            tools, withheld = _tools_within_reach(tools, allowed_scopes)
            if withheld:
                logger.info(
                    "🔧 Tool handoff: withheld %s — above the %s ceiling for this turn.",
                    ",".join(sorted(withheld)),
                    ceiling,
                )
            if not tools:
                logger.info(
                    "🔧 Tool handoff: skill=%s offered=NONE (all above the %s ceiling)",
                    ",".join(required),
                    ceiling,
                )
                return None
            result = await self._tool_grounded_answer_part_1(ceiling, client, decode_budget, evidence, origin, required, text, timeout_s, tools)
        except asyncio.CancelledError:
            # Stop belongs to the whole turn, including its fallback path.
            raise
        except TimeoutError as exc:
            record_degradation(
                "inference_gate.tool_grounded_answer",
                exc,
                severity="info",
                action="answered on the ordinary lane after the tool loop ran out of time",
                enforce_failure_policy=False,
            )
            return None
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            # Named, like every other handler in this file. A bare `except
            # Exception` here also swallowed KeyboardInterrupt-adjacent and
            # programming errors — a NameError in the tool loop would have been
            # recorded as "the tool loop failed" and answered around, so the
            # defect would never surface as a defect.
            record_degradation(
                "inference_gate.tool_grounded_answer",
                exc,
                severity="warning",
                action="answered on the ordinary lane after the tool loop failed",
                enforce_failure_policy=False,
            )
            return None

        if not isinstance(result, dict):
            return None
        called = result.get("tool_calls") or []
        text = str(result.get("content") or "").strip()
        if not called:
            # The model was handed the tool and answered without it. That
            # answer is ungrounded by construction, and it is exactly how
            # "Output: 7" reached the screen.
            return None
        self._tool_grounded_answer_model_path(called, client, text)
        from core.conversation.surface_disposition import record_tool_receipt

        for call in called:
            if not isinstance(call, dict):
                continue
            # What it RETURNED, not only that it ran.
            #
            # A receipt saying a tool executed and not what came back cannot
            # support any answer. LIVE, 2026-08-27: file_operation read a
            # project's docs in 6ms, the turn had nothing to say afterwards,
            # and the record of the read held the arguments and no result — so
            # the fallback that reports what the tools found had nothing to
            # report.
            observed = _what_a_tool_returned(call.get("result"))
            record_tool_receipt(
                str(call.get("tool") or call.get("name") or "tool"),
                ok=bool(call.get("ok", True)),
                action="execute",
                object_ref=str(call.get("args") or "")[:200],
                effect_observed=True,
                verification="tool loop returned a result for this turn",
                observed_content=observed[:2000],
            )
        return text or None

    @staticmethod
    def _build_living_mind_context_memory_pressure(mem_monitor, segments, state):
        from .inference_gate import (
            InferenceGate,
            psutil,
        )

        memory_pressure = None
        if mem_monitor is not None:
            memory_pressure = getattr(mem_monitor, "pressure", None)
        if memory_pressure is None and psutil is not None:
            memory_pressure = InferenceGate._recent_virtual_memory().percent
        # Only render fields that were actually observed. Missing hardware
        # telemetry must appear as UNAVAILABLE — fabricating 0% CPU and a
        # "stable" thermal label would present dead sensors as calm
        # physiology.
        temperature: float | None = None
        cpu_usage: float | None = None
        if state is not None:
            hw = getattr(getattr(state, "soma", None), "hardware", {}) or {}
            if hw.get("temperature") is not None:
                temperature = float(hw.get("temperature") or 0.0)
            if hw.get("cpu_usage") is not None:
                cpu_usage = float(hw.get("cpu_usage") or 0.0)
        physiology_lines = ["## LIVE PHYSIOLOGY"]
        physiology_lines.append(
            f"- CPU usage: {cpu_usage:.1f}%"
            if cpu_usage is not None
            else "- CPU usage: unavailable (no hardware telemetry)"
        )
        if temperature is not None:
            thermal_label = (
                "critical"
                if temperature >= 85.0
                else "warm"
                if temperature >= 75.0
                else "stable"
            )
            physiology_lines.append(
                f"- Thermal state: {thermal_label} ({temperature:.1f} C)"
            )
        else:
            physiology_lines.append(
                "- Thermal state: unavailable (no hardware telemetry)"
            )
        physiology_lines.append(
            f"- Memory pressure: {float(memory_pressure):.1f}%"
            if memory_pressure is not None
            else "- Memory pressure: unavailable"
        )
        segments.add("physiology", "\n".join(physiology_lines))

    @staticmethod
    def _build_living_mind_context_pneuma_active_inference(prompt, segments):
        from .inference_gate import (
            _INFERENCE_RECOVERABLE_ERRORS,
            _record_inference_degradation,
            logger,
        )

        # ── PNEUMA (Active Inference) ─────────────────────────────────────────
        try:
            from core.pneuma import get_pneuma

            _pneuma = get_pneuma()
            _pneuma_block = _pneuma.get_context_block()
            if _pneuma_block:
                segments.add("pneuma", _pneuma_block)
            # Push the current prompt into the belief flow as UNTRUSTED
            # observation — raw user text is not verified evidence, so it
            # gets a capped weight and an attributable provenance tag.
            _pneuma.on_evidence(
                prompt[:300], weight=0.2, source="user_prompt", trusted=False
            )
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            segments.omit("pneuma", exc)
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("PNEUMA injection unavailable: %s", exc)

        # ── MHAF (Mycelial Hypergraph) ────────────────────────────────────────
        try:
            from core.consciousness.mhaf_field import get_mhaf

            _mhaf = get_mhaf()
            _mhaf_block = _mhaf.get_context_block()
            if _mhaf_block:
                segments.add("mhaf", _mhaf_block)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            segments.omit("mhaf", exc)
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("MHAF injection unavailable: %s", exc)

        # ── Private Lexicon (Neologism Engine) ───────────────────────────────
        try:
            from core.consciousness.neologism_engine import get_neologism_engine

            _neo = get_neologism_engine()
            _neo.collect_state()
            lex_block = _neo.get_lexicon_block()
            if lex_block:
                segments.add("neologisms", lex_block, priority=PRIORITY_COLOUR)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            segments.omit("neologisms", exc)
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("NeologismEngine injection unavailable: %s", exc)

    def _build_living_mind_context_part_3(self, _affect_observed, _circ, _shared_arousal, _shared_valence):
        if _circ and hasattr(_circ, "get_llm_params"):
            # The PUBLIC reader first. _sample_raw_axes is private, and
            # reaching past a public accessor into a subsystem's internals
            # is how a rename becomes an outage.
            _cp = _circ.get_llm_params()
            if isinstance(_cp, dict):
                _v = self._bounded_affect(_cp.get("valence"), low=-1.0, high=1.0)
                _a = self._bounded_affect(_cp.get("arousal"), low=0.0, high=1.0)
                if _v is not None:
                    _shared_valence, _affect_observed["valence"] = _v, True
                if _a is not None:
                    _shared_arousal, _affect_observed["arousal"] = _a, True
        if (
            not _affect_observed["valence"]
            and _circ
            and hasattr(_circ, "_sample_raw_axes")
        ):
            _raw = _circ._sample_raw_axes()
            if isinstance(_raw, (tuple, list)) and len(_raw) == 2:
                _v = self._bounded_affect(_raw[0], low=-1.0, high=1.0)
                _a = self._bounded_affect(_raw[1], low=0.0, high=1.0)
                if _v is not None:
                    _shared_valence, _affect_observed["valence"] = _v, True
                if _a is not None:
                    _shared_arousal, _affect_observed["arousal"] = _a, True
        return _shared_arousal, _shared_valence

    @staticmethod
    def _build_living_mind_context_part_4(_affect_observed, _shared_arousal, _shared_curiosity, _shared_energy, _shared_valence, advance_state, segments):
        from .inference_gate import (
            _INFERENCE_RECOVERABLE_ERRORS,
            _record_inference_degradation,
            logger,
        )

        if not all(_affect_observed.values()):
            # Which axes are real and which are the constructor's defaults.
            # Without this, "valence 0.0" means either "measured neutral" or
            # "nothing answered", and three subsystems consume it as the first.
            segments.omit(
                "affect_axes",
                "defaults used for: "
                + ", ".join(
                    name for name, seen in _affect_observed.items() if not seen
                ),
            )

        try:
            from core.consciousness.crsm import get_crsm

            _crsm = get_crsm()
            if advance_state:
                _crsm.update(
                    valence=_shared_valence,
                    arousal=_shared_arousal,
                    curiosity=_shared_curiosity,
                    energy=_shared_energy,
                    surprise=_crsm.surprise_signal,  # self-referential: own recent error
                )
                segments.advanced("crsm")
            _crsm_block = _crsm.get_context_block()
            if _crsm_block:
                segments.add("crsm", _crsm_block)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            segments.omit("crsm", exc)
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("CRSM injection unavailable: %s", exc)

        # ── Higher-Order Thought Engine (HOT) ────────────────────────────────
        try:
            from core.consciousness.hot_engine import get_hot_engine

            _hot = get_hot_engine()
            _hot.generate_fast(
                {
                    "valence": _shared_valence,
                    "arousal": _shared_arousal,
                    "curiosity": _shared_curiosity,
                    "energy": _shared_energy,
                    "surprise": 0.0,
                }
            )
            _hot_block = _hot.get_context_block()
            if _hot_block:
                segments.add("higher_order_thought", _hot_block)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            segments.omit("higher_order_thought", exc)
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("HOT Engine injection unavailable: %s", exc)

        # ── Hedonic Gradient ──────────────────────────────────────────────────
        try:
            from core.consciousness.hedonic_gradient import get_hedonic_gradient

            _hg = get_hedonic_gradient()
            # Update with current affect state before reading context block
            if advance_state:
                _hg.update(
                    valence=_shared_valence,
                    arousal=_shared_arousal,
                    curiosity=_shared_curiosity,
                    energy=_shared_energy,
                )
                segments.advanced("hedonic_gradient")
            _hg_block = _hg.get_context_block()
            if _hg_block:
                segments.add("hedonic_gradient", _hg_block)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            segments.omit("hedonic_gradient", exc)
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("HedoniGradient injection unavailable: %s", exc)

    @staticmethod
    def _build_living_mind_context_hierarchical_goals(advance_state, prompt, segments):
        from .inference_gate import (
            _INFERENCE_RECOVERABLE_ERRORS,
            _record_inference_degradation,
            logger,
        )

        # ── Hierarchical Goals ────────────────────────────────────────────────
        try:
            from core.agi.hierarchical_planner import get_hierarchical_planner

            _hp = get_hierarchical_planner()
            _hp_block = _hp.get_context_block()
            if _hp_block:
                segments.add("hierarchical_plan", _hp_block)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            segments.omit("hierarchical_plan", exc)
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("HierarchicalPlanner injection unavailable: %s", exc)

        # ── Active Commitments ────────────────────────────────────────────────
        try:
            from core.agency.commitment_engine import get_commitment_engine

            _ce = get_commitment_engine()
            _ce_block = _ce.get_context_block()
            if _ce_block:
                segments.add("commitments", _ce_block)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            segments.omit("commitments", exc)
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("CommitmentEngine injection unavailable: %s", exc)

        # ── Curiosity Explorer (active learning findings) ─────────────────────
        try:
            from core.agi.curiosity_explorer import get_curiosity_explorer

            _cx = get_curiosity_explorer()
            _cx_block = _cx.get_context_block()
            if _cx_block:
                segments.add("curiosity", _cx_block, priority=PRIORITY_COLOUR)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            segments.omit("curiosity", exc)
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("CuriosityExplorer injection unavailable: %s", exc)

        # ── Circadian Rhythm ──────────────────────────────────────────────────
        try:
            from core.senses.circadian import get_circadian

            _circ_eng = get_circadian()
            if advance_state:
                _circ_eng.update()
                segments.advanced("circadian")
            _circ_block = _circ_eng.get_context_block()
            if _circ_block:
                segments.add("circadian", _circ_block, priority=PRIORITY_COLOUR)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            segments.omit("circadian", exc)
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("CircadianEngine injection unavailable: %s", exc)

        # ── Identity Narrative (Experience Consolidator) ──────────────────────
        try:
            from core.consciousness.experience_consolidator import get_experience_consolidator

            _ec = get_experience_consolidator()
            _ec_block = _ec.get_context_block()
            if _ec_block:
                segments.add("experience_consolidation", _ec_block, priority=PRIORITY_COLOUR)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            segments.omit("experience_consolidation", exc)
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("ExperienceConsolidator injection unavailable: %s", exc)

        # ── Substrate Learning (CRSM LoRA Bridge) ─────────────────────────────
        try:
            from core.consciousness.crsm_lora_bridge import get_crsm_lora_bridge

            _lora_bridge = get_crsm_lora_bridge()
            _lora_block = _lora_bridge.get_context_block()
            if _lora_block:
                segments.add("crsm_lora_bridge", _lora_block)
            # Pre-inference capture: record current state before thinking
            from core.consciousness.crsm import get_crsm as _get_crsm2

            _crsm2 = _get_crsm2()
            from core.consciousness.hedonic_gradient import get_hedonic_gradient as _get_hg2

            _hg2 = _get_hg2()
            _lora_bridge.pre_inference_capture(
                context_text=prompt,
                surprise_magnitude=_crsm2.surprise_signal,
                hedonic_score=_hg2.score,
                crsm_hidden_norm=float(
                    sum(x**2 for x in _crsm2.hidden_state) ** 0.5
                    if hasattr(_crsm2, "hidden_state")
                    else 0.0
                ),
            )
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            segments.omit("crsm_lora_bridge", exc)
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("CRSMLoraBridge injection unavailable: %s", exc)

    async def _build_living_mind_context(
        self,
        prompt: str,
        origin: str,
        *,
        token_budget: int | None = None,
        advance_state: bool = True,
    ) -> str:
        """Inject live self-model state so speech is driven by current mind.

        The assembled text is bounded and receipted — see
        :meth:`living_mind_context_receipt` for what reached the prompt and
        what did not.
        """
        from .inference_gate import (
            _INFERENCE_RECOVERABLE_ERRORS,
            PRIORITY_GATING,
            LivingMindContext,
            _grounded_state_signal_text,
            _record_inference_degradation,
            inspect,
            logger,
            standing_disposition,
        )


        async def _resolve(value):
            if inspect.isawaitable(value):
                return await value
            return value

        try:
            from core.container import ServiceContainer
        except _INFERENCE_RECOVERABLE_ERRORS:
            return ""

        segments = LivingMindContext(
            token_budget=self._living_mind_token_budget(prompt, token_budget)
        )

        try:
            repo = ServiceContainer.get("state_repository", default=None)
            state = getattr(repo, "_current", None) if repo is not None else None
            mem_monitor = ServiceContainer.get("memory_monitor", default=None)
            self._build_living_mind_context_memory_pressure(mem_monitor, segments, state)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            segments.omit("physiology", exc)
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("Physiology injection unavailable: %s", exc)

        # Unity is assessed BEFORE the grounded self-report so that an unsafe
        # fragmentation verdict actually suppresses self-report material —
        # printing "Safe to self-report: False" under an already-appended
        # report would gate nothing.
        safe_to_self_report = True
        try:
            unity_state = ServiceContainer.get("unity_state", default=None)
            unity_report = ServiceContainer.get("unity_fragmentation_report", default=None)
            unity_repair = ServiceContainer.get("unity_repair_plan", default=None)
            if unity_state:
                lines = [
                    "## UNITY",
                    f"- Level: {getattr(unity_state, 'level', 'unknown')}",
                    f"- Unity score: {float(getattr(unity_state, 'unity_score', 0.0) or 0.0):.3f}",
                    f"- Fragmentation: {float(getattr(unity_state, 'fragmentation_score', 0.0) or 0.0):.3f}",
                ]
                if unity_report is not None:
                    # The verdict used to be read only when top_causes was
                    # non-empty, so an unsafe report that listed no causes left
                    # the default True standing and gated nothing.
                    safe_to_self_report = bool(
                        getattr(unity_report, "safe_to_self_report", True)
                    )
                    lines.append(f"- Safe to self-report: {safe_to_self_report}")
                if unity_report and getattr(unity_report, "top_causes", None):
                    rendered = ", ".join(
                        f"{str(name).replace('_', ' ')}={float(weight):.2f}"
                        for name, weight, _text in list(unity_report.top_causes)[:3]
                    )
                    lines.append(f"- Top causes: {rendered}")
                if unity_repair and getattr(unity_repair, "steps", None):
                    lines.append(f"- Repair bias: {str(unity_repair.steps[0])[:180]}")
                segments.add("unity", "\n".join(lines), priority=PRIORITY_GATING)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            segments.omit("unity", exc)
            # The unity check was ATTEMPTED and failed. That is not the same as
            # a runtime with no unity service — the assessment that decides
            # whether she may describe her own state broke, and leaving the
            # default True is the absence of a check counted as a pass.
            safe_to_self_report = False
            _record_inference_degradation(
                exc,
                action="suppressed the grounded self-report because unity could not be assessed",
                severity="warning",
            )
            logger.debug("Unity injection unavailable: %s", exc)

        try:
            if not safe_to_self_report:
                logger.info(
                    "🧩 Grounded self-report suppressed: unity assessment marked "
                    "self-report unsafe this turn, or could not be made."
                )
                segments.omit("self_report", "unity_verdict_unsafe_or_unavailable")
            else:
                self_report = ServiceContainer.get("self_report_engine", default=None)
                if self_report and hasattr(self_report, "generate_state_report"):
                    report = await _resolve(self_report.generate_state_report())
                    if report:
                        segments.add("self_report", f"## GROUNDED SELF-REPORT\n{report}", priority=PRIORITY_GATING)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            segments.omit("self_report", exc)
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("Self-report injection unavailable: %s", exc)

        try:
            personality = ServiceContainer.get("personality_engine", default=None)
            if personality:
                if advance_state and hasattr(personality, "update"):
                    await _resolve(personality.update())
                    segments.advanced("personality")
                emo = await _resolve(personality.get_emotional_context_for_response())
                mood = emo.get("mood", "neutral")
                tone = emo.get("tone", "balanced")
                dominant = ", ".join(list(emo.get("dominant_emotions", []))[:4]) or "none"
                segments.add("personality", "## LIVE PERSONALITY DRIVE\n"
                    f"- Mood: {mood}\n"
                    f"- Tone: {tone}\n"
                    f"- Dominant emotions: {dominant}")
                sovereign = await _resolve(
                    getattr(personality, "get_sovereign_context", lambda: "")()
                )
                if sovereign:
                    segments.add("personality", str(sovereign).strip()[:400])
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            segments.omit("personality", exc)
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("Personality injection unavailable: %s", exc)

        try:
            experiencer = ServiceContainer.get("phenomenological_experiencer", default=None)
            if experiencer:
                fragment = ""
                if hasattr(experiencer, "get_phenomenal_context_fragment"):
                    fragment = await _resolve(experiencer.get_phenomenal_context_fragment())
                elif hasattr(experiencer, "phenomenal_context_string"):
                    fragment = getattr(experiencer, "phenomenal_context_string", "")
                if fragment:
                    grounded_fragment = _grounded_state_signal_text(fragment, limit=500)
                    segments.add("phenomenology", f"## FUNCTIONAL STATE SIGNALS\n{grounded_fragment}")
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            segments.omit("phenomenology", exc)
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("Phenomenology injection unavailable: %s", exc)

        # A position she formed before this conversation wins. Absent one, she
        # still gets the standing disposition — she is entitled to a view on a
        # subject she is meeting for the first time, and that entitlement must
        # not depend on the opinion service being up.
        held_position = ""
        try:
            topic_hint = self._topic_hint_from_prompt(prompt)
            opinion_engine = ServiceContainer.get("opinion_engine", default=None)
            if opinion_engine and topic_hint and hasattr(opinion_engine, "get_context_injection"):
                opinion_context = await _resolve(opinion_engine.get_context_injection(topic_hint))
                held_position = str(opinion_context or "").strip()[:400]
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            segments.omit("opinion", exc)
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("Opinion injection unavailable: %s", exc)
        if held_position or self._origin_is_user_facing(origin):
            segments.add(
                "opinion",
                f"## {'HELD POSITIONS' if held_position else 'HOLDING A VIEW'}\n"
                f"{standing_disposition(held_position)}",
            )

        try:
            if self._origin_is_user_facing(origin):
                spine = ServiceContainer.get("spine", default=None)
                if spine and hasattr(spine, "pre_response_check"):
                    check = await spine.pre_response_check(
                        prompt,
                        topic=self._topic_hint_from_prompt(prompt),
                    )
                    if check and getattr(check, "injection", ""):
                        segments.add("spine", f"## SPIRITUAL SPINE\n{check.injection}", priority=PRIORITY_GATING)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            segments.omit("spine", exc)
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("Spine injection unavailable: %s", exc)

        # ── Heartstone Values: evolved drive weights in every prompt ──────────
        try:
            from core.affect.heartstone_values import get_heartstone_values

            _hsv = get_heartstone_values()
            _hsv_block = _hsv.to_context_block()
            if _hsv_block:
                segments.add("heartstone_values", _hsv_block, priority=PRIORITY_GATING)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            segments.omit("heartstone_values", exc)
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("HeartstoneValues injection unavailable: %s", exc)

        # ── Architecture self-awareness ─────────────────────────────────────
        try:
            arch_idx = ServiceContainer.get("architecture_index", default=None)
            if arch_idx is None:
                from core.self.architecture_index import get_architecture_index

                arch_idx = get_architecture_index()
            if arch_idx and arch_idx._index:
                overview = arch_idx.get_overview()
                if overview:
                    segments.add("architecture_overview", overview[:800])
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            segments.omit("architecture_overview", exc)
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("Architecture overview injection unavailable: %s", exc)

        self._build_living_mind_context_pneuma_active_inference(prompt, segments)

        # ── Continuous Recurrent Self-Model (CRSM) ───────────────────────────
        # Shared affect state — pulled once, fed to CRSM, HOT and Hedonic.
        #
        # Every value here comes off another subsystem, and each read used to
        # be a bare float() with an assumed scale: valence and arousal from a
        # PRIVATE `_sample_raw_axes`, curiosity and energy divided by 100 on
        # the assumption they are percentages, with no check of type, range or
        # finiteness. A subsystem returning 0..1 instead of 0..100 silently
        # became 0.005, and a NaN propagated into CRSM, the hedonic gradient
        # and the higher-order thought engine at once. A partial failure also
        # left a mix of observed and default values that nothing could tell
        # apart afterwards.
        _shared_valence, _shared_arousal, _shared_curiosity, _shared_energy = 0.0, 0.5, 0.5, 0.7
        _affect_observed: dict[str, bool] = {
            "valence": False,
            "arousal": False,
            "curiosity": False,
            "energy": False,
        }
        try:
            from core.container import ServiceContainer

            # valence + arousal from AffectiveCircumplex (authoritative source)
            _circ = ServiceContainer.get("affective_circumplex", default=None)
            _shared_arousal, _shared_valence = self._build_living_mind_context_part_3(_affect_observed, _circ, _shared_arousal, _shared_valence)
            # curiosity + energy from liquid_state, reported as percentages
            _ls = ServiceContainer.get("liquid_state", default=None)
            if _ls and hasattr(_ls, "get_status"):
                _lsd = _ls.get_status()
                if isinstance(_lsd, dict):
                    _c = self._bounded_affect(_lsd.get("curiosity"), low=0.0, high=100.0)
                    _e = self._bounded_affect(_lsd.get("energy"), low=0.0, high=100.0)
                    if _c is not None:
                        _shared_curiosity, _affect_observed["curiosity"] = _c / 100.0, True
                    if _e is not None:
                        _shared_energy, _affect_observed["energy"] = _e / 100.0, True
        except _INFERENCE_RECOVERABLE_ERRORS as _exc:
            _record_inference_degradation(
                _exc,
                action="used default affect axes after the affect snapshot failed",
                extra={"observed": dict(_affect_observed)},
            )
            logger.debug("Affect snapshot unavailable: %s", _exc)
        self._build_living_mind_context_part_4(_affect_observed, _shared_arousal, _shared_curiosity, _shared_energy, _shared_valence, advance_state, segments)

        # ── Hierarchical Goals ────────────────────────────────────────────────
        try:
            goal_engine = ServiceContainer.get("goal_engine", default=None)
            if goal_engine and hasattr(goal_engine, "get_context_block"):
                goal_block = goal_engine.get_context_block(limit=5)
                if goal_block:
                    segments.add("goals", goal_block)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            segments.omit("goals", exc)
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("GoalEngine injection unavailable: %s", exc)

        self._build_living_mind_context_hierarchical_goals(advance_state, prompt, segments)

        # ══════════════════════════════════════════════════════════════════
        # DEEPENED CONSCIOUSNESS CONTEXT BLOCKS
        # These modules now provide real computation that influences behavior
        # ══════════════════════════════════════════════════════════════════

        # ── Homeostasis (Adaptive Drive State) ────────────────────────────────
        try:
            homeostasis = ServiceContainer.get("homeostasis", default=None)
            if homeostasis and hasattr(homeostasis, "get_context_block"):
                _block = homeostasis.get_context_block()
                if _block:
                    segments.add("homeostasis", _block)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            segments.omit("homeostasis", exc)
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("Homeostasis injection unavailable: %s", exc)

        # ── Free Energy (Active Inference State) ──────────────────────────────
        try:
            fe_engine = ServiceContainer.get("free_energy_engine", default=None)
            if fe_engine and hasattr(fe_engine, "get_context_block"):
                _block = fe_engine.get_context_block()
                if _block:
                    segments.add("free_energy", _block)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            segments.omit("free_energy", exc)
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("FreeEnergy injection unavailable: %s", exc)

        # ── Attention Schema (Current Focus + Coherence) ──────────────────────
        try:
            attention = ServiceContainer.get("attention_schema", default=None)
            if attention and hasattr(attention, "get_context_block"):
                _block = attention.get_context_block()
                if _block:
                    segments.add("attention_schema", _block)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            segments.omit("attention_schema", exc)
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("AttentionSchema injection unavailable: %s", exc)

        # ── Cognitive Credit (Domain Performance Landscape) ───────────────────
        try:
            credit = ServiceContainer.get("credit_assignment", default=None)
            if credit and hasattr(credit, "get_context_block"):
                _block = credit.get_context_block()
                if _block:
                    segments.add("credit_assignment", _block)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            segments.omit("credit_assignment", exc)
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("CreditAssignment injection unavailable: %s", exc)

        # ── Theory of Mind (User Model) ───────────────────────────────────────
        try:
            tom = ServiceContainer.get("theory_of_mind", default=None)
            if tom and hasattr(tom, "get_context_block"):
                _block = tom.get_context_block()
                if _block:
                    segments.add("theory_of_mind", _block, trust=TRUST_LEARNED)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            segments.omit("theory_of_mind", exc)
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("TheoryOfMind injection unavailable: %s", exc)

        # ── World Model (Active Beliefs) ──────────────────────────────────────
        try:
            world_model = ServiceContainer.get("epistemic_state", default=None)
            if world_model and hasattr(world_model, "get_context_block"):
                topic = self._topic_hint_from_prompt(prompt)
                _block = world_model.get_context_block(topic_hint=topic)
                if _block:
                    segments.add("world_model", _block, trust=TRUST_LEARNED)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            segments.omit("world_model", exc)
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("WorldModel injection unavailable: %s", exc)

        # ── Temporal Binding (Autobiographical Continuity) ────────────────────
        try:
            temporal = ServiceContainer.get("temporal_binding", default=None)
            if temporal:
                narrative = await _resolve(temporal.get_narrative())
                if narrative and len(str(narrative)) > 30:
                    segments.add("temporal_continuity", f"## TEMPORAL CONTINUITY\n{str(narrative)[:200]}", priority=PRIORITY_COLOUR)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            segments.omit("temporal_continuity", exc)
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("TemporalBinding injection unavailable: %s", exc)

        # ── Predictive Engine (Surprise & Precision) ──────────────────────────
        try:
            predictive = ServiceContainer.get("predictive_engine", default=None)
            if predictive and hasattr(predictive, "get_context_block"):
                _block = predictive.get_context_block()
                if _block:
                    segments.add("predictive_engine", _block)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            segments.omit("predictive_engine", exc)
            _record_inference_degradation(
                exc,
                action="omitted unavailable living-mind context signal and continued prompt assembly",
            )
            logger.debug("PredictiveEngine injection unavailable: %s", exc)

        rendered, receipt = segments.render()
        self._living_mind_receipt = receipt
        return rendered

