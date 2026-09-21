"""How a reply is shaped for the person reading it, and the deep-honesty pass over it.

Lifted whole out of `response_generation_unitary`. Every name taken from it is imported at
CALL time: that module imports this one to build the class, and a test that
patches a name on it has to reach the code that reads it.
"""
from __future__ import annotations

import re


class _ShapesTheReply:
    """Lifted whole out of UnitaryResponsePhase; see response_generation_unitary.py."""

    @staticmethod
    def _shape_user_facing_response(text: str, user_message: str = "") -> str:
        from .response_generation_unitary import (
            _RESPONSE_RECOVERABLE_ERRORS,
            ServiceContainer,
            _record_response_degradation,
            logger,
        )

        authored = str(text or "").strip()
        shaped = authored
        if not shaped:
            return shaped
        shaped = re.sub(r"^\s*[.。]\s+(?=[A-Z0-9\"'“‘])", "", shaped).strip()
        try:
            from core.synthesis import cure_personality_leak, stabilize_user_facing_response

            shaped = cure_personality_leak(shaped)
            shaped = stabilize_user_facing_response(shaped, user_message)
        except _RESPONSE_RECOVERABLE_ERRORS as exc:
            _record_response_degradation(
                exc, "UnitaryResponse: initial user-facing stabilization skipped: %s"
            )

        try:
            personality = ServiceContainer.get("personality_engine", default=None)
            if personality:
                if hasattr(personality, "filter_response"):
                    filtered = personality.filter_response(shaped)
                    if isinstance(filtered, str) and filtered.strip():
                        shaped = filtered.strip()
                if hasattr(personality, "apply_lexical_style"):
                    styled = personality.apply_lexical_style(shaped)
                    if isinstance(styled, str) and styled.strip():
                        shaped = styled.strip()
        except _RESPONSE_RECOVERABLE_ERRORS as exc:
            _record_response_degradation(
                exc,
                "UnitaryResponse: response shaping skipped: %s",
                action="continued user-facing response shaping without personality lexical filter",
            )
            logger.debug("UnitaryResponse: response shaping skipped: %s", exc)
        try:
            from core.synthesis import stabilize_user_facing_response

            shaped = stabilize_user_facing_response(shaped, user_message)
        except _RESPONSE_RECOVERABLE_ERRORS as exc:
            _record_response_degradation(
                exc, "UnitaryResponse: final user-facing stabilization skipped: %s"
            )
        shaped = re.sub(r"^\s*[.。]\s+(?=[A-Z0-9\"'“‘])", "", shaped).strip()
        if shaped != authored:
            try:
                from core.conversation.surface_disposition import repair_is_an_improvement

                if not repair_is_an_improvement(authored, shaped, user_message):
                    logger.warning(
                        "UnitaryResponse rejected a post-generation transform that lost request semantics "
                        "(before_len=%d after_len=%d).",
                        len(authored),
                        len(shaped),
                    )
                    return authored
            except (ImportError, RuntimeError, TypeError, ValueError) as exc:
                _record_response_degradation(
                    exc,
                    "UnitaryResponse: semantic transform admission failed: %s",
                    action="preserved the model-authored user-facing response",
                    severity="error",
                )
                return authored
        return shaped

    async def _apply_deep_honesty(self, text: str) -> str:
        """Opt-in (AURA_DEEP_HONESTY=1) inline fact-check of the final user-facing
        response via the Data honesty governor. Off by default — so it never taxes a
        response unless explicitly enabled — bounded ~8s, and fail-open to the text
        as-is. The model can only annotate an unverified claim, never alter intent."""
        from .response_generation_unitary import (
            ServiceContainer,
            _record_response_degradation,
        )

        try:
            from core.morality.honesty_governor import deep_honesty_enabled

            if not text or not deep_honesty_enabled():
                return text
            from core.container import ServiceContainer

            gov = ServiceContainer.get("data", default=None)
            if gov is None or not hasattr(gov, "vet_output_deep"):
                return text
            vetted = await gov.vet_output_deep(text, force=True, timeout=8.0)
            return vetted if isinstance(vetted, str) and vetted.strip() else text
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            _record_response_degradation(
                exc, "UnitaryResponse: deep honesty pass skipped: %s"
            )
            return text

