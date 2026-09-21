"""What the moment carries into the prompt: the body's sampling modulation, the present, and the grounding refreshed at the last second.

Lifted whole out of `inference_gate`, which imports them straight back: every
caller and every patch that names them there still finds them. What they
take from that module is imported at CALL time, for the same reason.
"""
from __future__ import annotations

import math
from typing import Any

from core.runtime.errors import record_degradation


def _modulate_sampling_from_the_body(
    *,
    ServiceContainer: Any,
    _rt: Any,
    context: Any,
    explicit_foreground: Any,
    is_background: Any,
    max_tokens: Any,
    morpho_kwargs: Any,
    protected_compact_capability_contract: Any,
    protected_foreground_lane: Any,
    self: Any,
    somatic_temperature: Any,
) -> tuple[Any, Any]:
    """Let the body's state move temperature and length, within bounds.

    Moved out of ``InferenceGate.generate`` by tools/extract_seam.py, which
    checks the body against the original token for token before
    writing. It reads 11 name(s) from the turn and hands back
    2.
    """
    from .inference_gate import (
        _INFERENCE_RECOVERABLE_ERRORS,
        logger,
    )

    if _rt is not None:
        _f = _rt.field.sample("global")
        _danger = self._modulator_factor(
            _f.get("danger", 0.0), source="morphogenesis.danger", low=0.0, high=1.0
        )
        _curiosity = self._modulator_factor(
            _f.get("curiosity", 0.0),
            source="morphogenesis.curiosity",
            low=0.0,
            high=1.0,
        )
        _resource_pressure = self._modulator_factor(
            _f.get("resource_pressure", 0.0),
            source="morphogenesis.resource_pressure",
            low=0.0,
            high=1.0,
        )

        if _danger > 0.3:
            somatic_temperature = (somatic_temperature or 0.72) * (
                1.0 - (_danger * 0.4)
            )
            morpho_kwargs["top_p"] = max(0.4, 0.9 - (_danger * 0.3))

        if _curiosity > 0.3:
            somatic_temperature = (somatic_temperature or 0.72) * (
                1.0 + (_curiosity * 0.3)
            )
            morpho_kwargs["repetition_penalty"] = max(1.0, 1.15 - (_curiosity * 0.1))

        if _resource_pressure > 0.5 and not protected_compact_capability_contract:
            max_tokens = int(max_tokens * (1.0 - (_resource_pressure * 0.5)))
            max_tokens = max(128, max_tokens)

        # Inject Existential Stakes physical parameter coupling
        try:
            stakes = ServiceContainer.get("existential_stakes", default=None)
            if stakes:
                threat = float(stakes.get_existential_threat())
                if not math.isfinite(threat):
                    raise ValueError("existential threat must be finite")
                threat = max(0.0, min(1.0, threat))
                if threat > 0.2:
                    protected_live_foreground = bool(
                        not is_background
                        and (
                            protected_foreground_lane
                            or context.get("desktop_cognitive_engine_required")
                            or context.get("cognitive_engine_required")
                            or explicit_foreground
                        )
                    )
                    # Background and unprotected turns may shrink output under
                    # survival pressure. Protected live desktop turns must not:
                    # starving the first user-visible Cortex reply causes clipped
                    # drafts, recovery storms, worker respawns, and higher memory
                    # pressure than simply answering with the requested budget.
                    if not protected_live_foreground:
                        max_tokens = int(max_tokens * (1.0 - threat * 0.7))
                        max_tokens = max(96, max_tokens)
                    # Decrease temperature to make generation fast/deterministic
                    if somatic_temperature is not None:
                        somatic_temperature = somatic_temperature * (1.0 - threat * 0.5)
                    else:
                        somatic_temperature = 0.72 * (1.0 - threat * 0.5)
                    # Clamp parameters
                    if "temperature" in morpho_kwargs:
                        morpho_kwargs["temperature"] = max(0.1, morpho_kwargs["temperature"] * (1.0 - threat * 0.5))
                    if "max_tokens" in morpho_kwargs:
                        morpho_kwargs["max_tokens"] = max_tokens
        except _INFERENCE_RECOVERABLE_ERRORS as _st_err:
            record_degradation(
                "inference_gate.existential_stakes",
                _st_err,
                severity="warning",
                action=(
                    "kept the validated morphogenetic generation parameters "
                    "and ignored only the invalid existential-stakes modifier"
                ),
            )
            logger.warning(
                "Existential-stakes generation modifier rejected; "
                "using validated base parameters: %s",
                _st_err,
            )

        if somatic_temperature is not None:
            somatic_temperature = max(0.1, min(1.5, somatic_temperature))

        logger.debug(
            "🧬 Morphogenetic Coupling: danger=%.2f curiosity=%.2f pres=%.2f -> temp=%.2f tokens=%d",
            _danger,
            _curiosity,
            _resource_pressure,
            somatic_temperature or 0.0,
            max_tokens,
        )
    return max_tokens, somatic_temperature


async def _attach_the_present_moment(
    *,
    ambient_grounding_blocks: Any,
    isolated_generation_contract: Any,
    recent_actions_already_grounded: Any,
    task_grounding_blocks: Any,
    visible_user_prompt: Any,
) -> None:
    """Attach the present-moment block unless the turn is isolated.

    Moved out of ``InferenceGate.generate`` by tools/extract_seam.py, which
    checks the body against the original token for token before
    writing. It reads 4 name(s) from the turn and hands back
    0.
    """
    from .inference_gate import (
        _INFERENCE_RECOVERABLE_ERRORS,
        logger,
    )

    if not isolated_generation_contract:
        try:
            from core.brain.present_moment import present_moment_block

            _present = present_moment_block()
            if _present:
                ambient_grounding_blocks.append(_present)

            # Suppressing the web search is only half the fix; without the
            # readings she still has to invent them, which is how "I
            # processed a 45-page PDF on neuromorphic computing" happened.
            if not recent_actions_already_grounded:
                from core.brain.recent_actions import (
                    asks_what_she_recently_did,
                    recent_actions_block,
                )

                # Historical action receipts answer questions about historical
                # actions.  Injecting them into every factual turn let an old
                # autonomous search masquerade as the source of a new answer.
                if asks_what_she_recently_did(visible_user_prompt):
                    _actions = recent_actions_block()
                    if _actions:
                        ambient_grounding_blocks.append(_actions)

            # The WIDER predicate here on purpose. This path only ADDS her
            # instrument reading, so a false positive costs a few lines of
            # prompt; asks_about_own_runtime additionally suppresses web
            # search in the response contract, where a false positive
            # costs the lookup the person asked for.
            from core.runtime.self_state_intent import (
                asks_about_own_capabilities,
            )

            if asks_about_own_capabilities(visible_user_prompt):
                from core.brain.self_state_report import runtime_self_report

                _instruments = runtime_self_report()
                if _instruments:
                    ambient_grounding_blocks.append(_instruments)

            # A file she was asked about, read off the disk.
            #
            # LIVE 2026-08-17: "read the file CONTRIBUTING.md and tell me
            # the first rule it states" was answered "I tried to read the
            # file and failed" — an attempt that never happened. No skill
            # ran, no error occurred; she narrated a failure.
            #
            # The read was wired into the phase pipeline's grounding
            # channel, which desktop chat does not use: chat arrives here
            # with prebuilt messages (mode=compact_foreground_prebuilt), so
            # the block was built into a prompt nobody sent. THIS is the
            # channel that reaches the worker, which is why it is attached
            # beside the present-moment and recent-actions readings rather
            # than anywhere upstream.
            # Take every reading this turn asks for.
            #
            # This was two hand-wired branches — one for the clipboard, one
            # for a named file — and before them a file COUNT that guessed,
            # a corpus that was never consulted, and a clock that invented
            # an ambient light sensor. Same defect each time: the capability
            # was registered, the reader existed, and nothing took the
            # reading before the answer was composed, so a model asked about
            # a fact it did not hold produced something fact-shaped.
            #
            # One registry now. An observable is an entry in
            # observable_registry rather than another branch threaded
            # through here, which is how the previous four ended up in four
            # places with four different bugs.
            import core.brain.observable_registry  # noqa: F401  (registers)
            from core.brain.observable_grounding import observable_blocks

            _readings = await observable_blocks(visible_user_prompt)
            if _readings:
                task_grounding_blocks.extend(_readings)
                logger.info(
                    "🔭 [GROUNDING] took %d reading(s): %s",
                    len(_readings),
                    ",".join(
                        block.split("\n", 1)[0].removeprefix("## ").lower()
                        for block in _readings
                    ),
                )
            else:
                # A turn that asked for a reading and got none is invisible
                # otherwise, which is how the screen block went missing for
                # a whole session while the file block worked.
                logger.debug(
                    "🔭 [GROUNDING] no reading matched prompt=%r",
                    str(visible_user_prompt)[:120],
                )
        except _INFERENCE_RECOVERABLE_ERRORS as _exc:
            record_degradation(
                "inference_gate",
                _exc,
                severity="warning",
                action="continued without present-moment grounding",
            )


def _refresh_volatile_grounding(
    *,
    ambient_grounding_blocks: Any,
    context: Any,
    contract_grounding_blocks: Any,
    has_volatile_grounding: Any,
    messages: Any,
    self: Any,
    system_prompt: Any,
    task_grounding_blocks: Any,
) -> tuple[Any, Any]:
    """Refresh grounding that goes stale between the prompt and the answer.

    Moved out of ``InferenceGate.generate`` by tools/extract_seam.py, which
    checks the body against the original token for token before
    writing. It reads 8 name(s) from the turn and hands back
    2.
    """
    if has_volatile_grounding and isinstance(messages, list) and messages:
        # BEFORE the final user turn, not after it.
        #
        # Riding dead last put a multi-thousand-character block of self-state
        # between the person's question and the model's turn, and the model
        # continued the nearest thing instead of answering. Measured live:
        # asked to run a real sandbox calculation and report the result, the
        # entire reply was "Things feel unusually settled right now. My
        # attention is on internal monitoring..." — the grounding text
        # continued as prose, with no answer, no code, and no refusal, on a
        # turn whose plan read scaffold=7023 request=518 (ratio 13.6x).
        #
        # Sitting just ahead of the last user message keeps the whole point
        # of volatile-last — every stable token, system prompt through prior
        # history, is still a reusable KV prefix, and the only thing behind
        # the churn is the new turn that had to be prefilled anyway — while
        # the last words before the model's turn are the person's own.
        # The budget is enforced during compaction, and this block is added
        # afterwards — so without a cap here it simply escapes it. Measured
        # 2026-07-28 on the contract profile: compaction produced a 974-char
        # payload well inside its 2,800 budget, then 1,727 characters of
        # grounding arrived as a second system message and the turn went out
        # at 3,013. The grounding is not optional — it is what stops her
        # narrating a present she was never given — so it is fitted rather
        # than dropped, keeping whole blocks in priority order.
        from core.utils.injected_blocks import stamp_grounding

        grounding_message = stamp_grounding(
            {
                "role": "system",
                "content": self._fit_grounding_blocks(
                    contract_blocks=contract_grounding_blocks,
                    task_blocks=task_grounding_blocks,
                    ambient_blocks=ambient_grounding_blocks,
                    limit=self._grounding_char_budget(context, messages),
                ),
            }
        )
        final_user_index = next(
            (
                index
                for index in range(len(messages) - 1, -1, -1)
                if isinstance(messages[index], dict)
                and str(messages[index].get("role", "")).strip().lower() == "user"
            ),
            None,
        )
        if final_user_index is None:
            messages = [*messages, grounding_message]
        else:
            messages = [
                *messages[:final_user_index],
                grounding_message,
                *messages[final_user_index:],
            ]
    elif has_volatile_grounding:
        # No message list to ride behind (single-prompt lanes): keep the old
        # behaviour rather than dropping the grounding entirely.
        system_prompt = "\n\n".join(
            [
                str(system_prompt or ""),
                *contract_grounding_blocks,
                *task_grounding_blocks,
                *ambient_grounding_blocks,
            ]
        ).strip()
    return messages, system_prompt
