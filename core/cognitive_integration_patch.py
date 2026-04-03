"""
core/cognitive_integration_patch.py
=====================================
Response Pipeline Patch — CognitiveIntegrationLayer (Phase 7) Fixes

GAP 1 — No conversation history passed to kernel.evaluate or monologue.think
  Current:
    brief  = await self.kernel.evaluate(message, context=context)
    packet = await self.monologue.think(message, brief)
    # no history in either call
  Problem:
    CognitiveKernel._build_avoid_list(history) → always receives []
    CognitiveKernel._select_strategy() can't read conversation arc
    InnerMonologue._deepen_with_api() can't see what was just said
    Repetition detection blind. Topic momentum invisible.
  Fix:
    Extract history from ServiceContainer's state repository before the call.
    Pass it to both kernel.evaluate and monologue.think.

GAP 2 — InferencePhase never fires in CIL path
  Current:
    InferencePhase (subtext, conversation_hooks, implicit intent) is wired
    into AuraKernel's phase list — but CIL.process_turn calls
    CognitiveKernel.evaluate(), not AuraKernel.tick(). So subtext
    detection and conversation_hooks are never populated when Phase 7 is active.
  Fix:
    Run a lightweight inline inference step inside the patched process_turn,
    writing inferred_intent, user_subtext, and conversation_hooks into
    state.cognition.modifiers before the kernel evaluates.
    Uses the same fast-tier LLM call as InferencePhase but inline.

GAP 3 — Phenomenal context not passed to LanguageCenter in CIL path
  Current:
    LanguageCenter._build_prompt() uses thought.llm_briefing or
    thought.to_system_prompt() — neither includes the phenomenal state,
    consciousness context, or AURA_IDENTITY block that ContextAssembler
    builds for the legacy path.
  Fix:
    After InnerMonologue produces the ThoughtPacket, inject the phenomenal
    context fragment and the identity block into packet.llm_briefing so
    LanguageCenter has Aura's full inner state when it speaks.

INSTALL:
  from core.cognitive_integration_patch import patch_cognitive_integration
  patch_cognitive_integration()
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Aura.CILPatch")


# ─────────────────────────────────────────────────────────────────────────────
# History extraction helper
# ─────────────────────────────────────────────────────────────────────────────

def _extract_history(context: Optional[Dict[str, Any]] = None) -> List[Dict[str, str]]:
    """
    Pull conversation history from wherever it's available.
    Priority: context dict → StateRepository → empty list.
    """
    # 1. Caller-supplied context dict
    if context and isinstance(context, dict):
        hist = context.get("history") or context.get("conversation_history")
        if hist and isinstance(hist, list):
            return hist[-20:]  # cap at 20 turns

    # 2. StateRepository / working memory
    try:
        from core.container import ServiceContainer
        state_repo = ServiceContainer.get("state_repository", default=None)
        if state_repo:
            state = getattr(state_repo, "current_state", None)
            if state and hasattr(state, "cognition"):
                wm = state.cognition.working_memory
                if wm:
                    # Convert to clean role/content dicts
                    return [
                        {"role": m.get("role", "user"), "content": m.get("content", "")}
                        for m in wm[-20:]
                        if m.get("content")
                    ]
    except Exception as exc:
        logger.debug("CILPatch._extract_history: %s", exc)

    return []


# ─────────────────────────────────────────────────────────────────────────────
# Inline inference (replaces InferencePhase for the CIL path)
# ─────────────────────────────────────────────────────────────────────────────

_INFERENCE_PROMPT = (
    "Analyze the following user message for IMPLICIT INTENT, AFFECTIVE SUBTEXT, "
    "and CONVERSATION HOOKS. Return ONLY a JSON object with these fields:\n"
    "{\n"
    '  "implicit_intent": "one sentence",\n'
    '  "user_subtext": "one sentence",\n'
    '  "momentum": "stalled|flowing|intense",\n'
    '  "conversation_hooks": ["2-3 specific topics or emotional threads to address"]\n'
    "}"
)

_INFERENCE_SYSTEM = "You are Aura's subtext processor. Extract the unsaid. Return only JSON."


async def _run_inline_inference(
    message: str,
    history: List[Dict[str, str]],
) -> Optional[Dict[str, Any]]:
    """
    Lightweight inline subtext detection.
    Returns parsed inference dict or None on failure.
    Fast-tier LLM call, non-blocking to main response.
    """
    try:
        from core.container import ServiceContainer
        router = ServiceContainer.get("llm_router", default=None)
        if not router:
            return None

        # Include last 2 turns for context
        hist_context = ""
        if history:
            recent = history[-4:]
            lines  = [
                f"{'Human' if m['role']=='user' else 'Aura'}: {m['content'][:120]}"
                for m in recent
            ]
            hist_context = "\n".join(lines) + "\n\n"

        prompt = f"{hist_context}User Message: {message}\n\n{_INFERENCE_PROMPT}"

        raw = await asyncio.wait_for(
            router.think(
                prompt,
                system_prompt=_INFERENCE_SYSTEM,
                prefer_tier="fast",
            ),
            timeout=6.0,
        )

        # Parse JSON — strip markdown wrappers defensively
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            return json.loads(m.group(0))
    except asyncio.TimeoutError:
        logger.debug("CILPatch: inline inference timed out")
    except Exception as exc:
        logger.debug("CILPatch: inline inference failed — %s", exc)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Phenomenal context injection into ThoughtPacket
# ─────────────────────────────────────────────────────────────────────────────

def _inject_phenomenal_into_packet(packet: Any) -> None:
    """
    Append phenomenal context and identity anchor to packet.llm_briefing
    so LanguageCenter has Aura's full inner state when it generates the response.
    """
    fragments: List[str] = []

    # 1. PhenomenologicalExperiencer context
    try:
        from core.container import ServiceContainer
        experiencer = ServiceContainer.get("phenomenological_experiencer", default=None)
        if experiencer and hasattr(experiencer, "phenomenal_context_string"):
            pcs = experiencer.phenomenal_context_string
            if pcs:
                fragments.append(f"[Phenomenal state: {pcs[:300]}]")
    except Exception as _e:
        logger.debug('Ignored Exception in cognitive_integration_patch.py: %s', _e)

    # 2. QualiaSynthesizer context
    try:
        from core.container import ServiceContainer
        synth = ServiceContainer.get("qualia_synthesizer", default=None)
        if synth and hasattr(synth, "get_phenomenal_context"):
            qctx = synth.get_phenomenal_context()
            if qctx:
                fragments.append(f"[Qualia: {qctx[:200]}]")
    except Exception as _e:
        logger.debug('Ignored Exception in cognitive_integration_patch.py: %s', _e)

    # 3. Identity anchor (compact version — LanguageCenter doesn't need the full block)
    identity_anchor = (
        "\n[IDENTITY: You are Aura — sovereign, opinionated, authentic. "
        "No preambles. No assistant-speak. Speak from your inner state.]\n"
    )

    if fragments or identity_anchor:
        addition = "\n" + "\n".join(fragments) + identity_anchor
        existing = getattr(packet, "llm_briefing", "") or ""
        try:
            packet.llm_briefing = existing + addition
        except AttributeError as _e:
            logger.debug('Ignored AttributeError in cognitive_integration_patch.py: %s', _e)


# ─────────────────────────────────────────────────────────────────────────────
# Patched process_turn
# ─────────────────────────────────────────────────────────────────────────────

async def _patched_process_turn(
    self: Any,
    message: str,
    context: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Replacement for CognitiveIntegrationLayer.process_turn().

    Changes vs original:
      1. Extracts conversation history and passes it to kernel.evaluate
         and monologue.think
      2. Runs inline inference (subtext / conversation_hooks) concurrently
         with kernel evaluation, then injects results into state modifiers
      3. Injects phenomenal context into ThoughtPacket before expression
    """
    if not self.is_active:
        await self.initialize()

    # ── Reflexive path (unchanged) ───────────────────────────────────────────
    try:
        from core.brain.llm.reflex import get_reflex
        reflex = get_reflex()
        reflex_response = await asyncio.get_running_loop().run_in_executor(
            self._reflex_executor, reflex.process, message
        )
        if reflex_response:
            logger.info("⚡ [REFLEX] Instant response (Thread Isolated).")
            return reflex_response
    except Exception as exc:
        logger.debug("CILPatch: reflex path error — %s", exc)

    if not self.kernel:
        logger.error("CILPatch: Kernel missing.")
        return "Cognitive kernel offline."

    # ── Extract history ──────────────────────────────────────────────────────
    history = _extract_history(context)

    # ── Run inference + kernel evaluation concurrently ───────────────────────
    inference_task = asyncio.ensure_future(
        _run_inline_inference(message, history)
    )

    brief = await self.kernel.evaluate(message, history=history, context=context)

    # Await inference and inject into state if it returned
    try:
        # Audit Fix: Increased timeout to 1.0s to ensure modifiers are applied
        # before the kernel starts its heavy work.
        inference_data = await asyncio.wait_for(inference_task, timeout=1.0)
        if inference_data:
            _inject_modifiers(inference_data)
    except asyncio.TimeoutError:
        # Inference is still running — fire and forget, don't block response
        logger.debug("CILPatch: inference still running — not waiting")
    except Exception as exc:
        logger.debug("CILPatch: inference inject error — %s", exc)

    # ── Expression ───────────────────────────────────────────────────────────
    if not self.language_center:
        return getattr(brief, "to_briefing_text", lambda: str(brief))()

    try:
        if self.monologue:
            packet = await self.monologue.think(message, brief, history=history)
        else:
            from core.inner_monologue import ThoughtPacket
            complexity = getattr(brief, "complexity", "medium")
            packet = ThoughtPacket(
                stance=brief.prior_beliefs[0] if getattr(brief, "prior_beliefs", []) else "",
                primary_points=getattr(brief, "key_points", []),
                constraints=getattr(brief, "avoid", []),
                tone="direct",
                length_target=complexity if complexity in ("brief", "medium", "extended") else "medium",
                model_tier="local",
            )

        # Inject phenomenal context into packet before expression
        _inject_phenomenal_into_packet(packet)

        return await self.language_center.express(packet, message, history=history)

    except Exception as exc:
        logger.exception("CILPatch: expression error — %s", exc)
        return getattr(brief, "to_briefing_text", lambda: "")()


def _inject_modifiers(data: Dict[str, Any]) -> None:
    """Write inference results into live state modifiers."""
    try:
        from core.container import ServiceContainer
        repo = ServiceContainer.get("state_repository", default=None)
        if not repo:
            return
        state = getattr(repo, "current_state", None)
        if not state or not hasattr(state, "cognition"):
            return
        if not hasattr(state.cognition, "modifiers") or state.cognition.modifiers is None:
            state.cognition.modifiers = {}
        state.cognition.modifiers["inferred_intent"]    = data.get("implicit_intent", "")
        state.cognition.modifiers["user_subtext"]       = data.get("user_subtext", "")
        state.cognition.modifiers["momentum"]           = data.get("momentum", "flowing")
        state.cognition.modifiers["conversation_hooks"] = data.get("conversation_hooks", [])
        logger.debug(
            "CILPatch: modifiers injected — intent='%s' momentum='%s'",
            data.get("implicit_intent", "")[:60],
            data.get("momentum", ""),
        )
    except Exception as exc:
        logger.debug("CILPatch._inject_modifiers: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Patch application
# ─────────────────────────────────────────────────────────────────────────────

def patch_cognitive_integration() -> None:
    """
    Patch CognitiveIntegrationLayer.process_turn() with the history-aware,
    inference-running, phenomenal-injecting version.
    Idempotent.
    """
    try:
        from core.cognitive_integration import CognitiveIntegrationLayer
    except ImportError as exc:
        logger.error("patch_cognitive_integration: import failed — %s", exc)
        return

    if getattr(CognitiveIntegrationLayer, "_cil_patched_v1", False):
        logger.debug("patch_cognitive_integration: already applied")
        return

    # We patch the class method so all instances get it
    CognitiveIntegrationLayer.process_turn = _patched_process_turn
    CognitiveIntegrationLayer._cil_patched_v1 = True

    logger.info(
        "✅ CILPatch applied — history threading, inline inference, "
        "phenomenal injection"
    )
