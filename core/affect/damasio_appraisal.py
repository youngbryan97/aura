"""Working out what an event means before it can be felt.

Three ladders, and the order matters: the model where there is time for it, the
lexical reading where there is not, and a heuristic that always answers. The
validator sits across all three, because an appraisal that is out of range is
worse than no appraisal at all — it moves the state and nothing notices.
"""
from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Mapping
from typing import Any

from core.conversation.word_markers import names_any
from core.runtime.errors import record_degradation


class _AppraisesWhatHappened:
    """Lifted whole from AffectEngineV2; see damasio_v2.py."""

    @staticmethod
    def _validate_appraisal(appraisal: Mapping[str, Any]) -> dict[str, float]:
        if not isinstance(appraisal, Mapping):
            raise ValueError("affect appraisal must be a mapping")
        required = {"v", "a", "e"}
        if set(appraisal) != required:
            raise ValueError("affect appraisal must contain exactly v, a, and e")
        values: dict[str, float] = {}
        for key, lower, upper in (("v", -1.0, 1.0), ("a", 0.0, 1.0), ("e", 0.0, 1.0)):
            try:
                value = float(appraisal[key])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"affect appraisal {key} must be numeric") from exc
            if not math.isfinite(value) or not lower <= value <= upper:
                raise ValueError(f"affect appraisal {key} outside [{lower}, {upper}]")
            values[key] = value
        return values

    @staticmethod
    def _classify_appraisal_failure(exc: Exception) -> str:
        text = str(exc or "").strip().lower()
        if isinstance(exc, asyncio.TimeoutError):
            return "timeout"
        if "empty_response" in text or "empty response" in text:
            return "empty_response"
        if "parse_failure" in text or "json" in text:
            return "parse_failure"
        if "router_unavailable" in text or "no inference gate" in text:
            return "router_unavailable"
        if "lane_unavailable" in text or "conversation lane" in text:
            return "lane_unavailable"
        return "unknown_failure"

    @staticmethod
    def _heuristic_appraisal(trigger: str, context: dict | None) -> dict[str, float]:
        """Appraise an event.

        This was a scan of the trigger string against thirty words, which
        meant an event with nothing at stake read as strongly negative if
        it happened to contain "fail", and an event that broke a promise
        read as neutral if it was phrased calmly. The words were doing the
        work that the relationship between the event and what Aura is
        holding should have been doing.

        Appraisal now comes from :mod:`core.interiority`, which computes
        the relational meaning: what is at stake, who caused it, whether
        it can be undone, whether a standard Aura holds was broken. Change
        nothing about the wording and change what she is committed to, and
        the appraisal changes — which is the property that makes it an
        appraisal rather than a classifier.

        The word scan is kept as `_lexical_appraisal`, the last resort,
        reached only when the interiority layer is unavailable, because an
        affect engine that returns nothing is worse than one that returns
        something crude. Which path answered is recorded in the receipt.
        """
        # Imported here rather than at module level: the module these
        # came from imports this one to build the class. A call-time
        # import also still sees a test's patch of the original.
        from .damasio_v2 import (
            AffectEngineV2,
        )

        try:
            from core.interiority.service import get_interiority

            return get_interiority().appraise(trigger, context)
        except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation(
                "damasio_v2",
                exc,
                action="fell back to the lexical appraisal; relational meaning unavailable",
            )
        return AffectEngineV2._lexical_appraisal(trigger, context)

    @staticmethod
    def _lexical_appraisal(trigger: str, context: dict | None) -> dict[str, float]:
        """The word scan: thirty words, and whatever is at stake ignored.

        Kept because an affect engine that returns nothing is worse than one
        that returns something crude, and named because the difference between
        it and the relational appraisal is a claim worth measuring rather than
        a branch worth hiding.
        """
        from .damasio_v2 import (
            _finite_clamp,
        )

        trigger_text = str(trigger or "").lower()
        base = _finite_clamp((context or {}).get("intensity", 1.0), 0.0, 1.0, default=0.0)

        valence = 0.0
        arousal = min(1.0, 0.2 + base * 0.3)
        engagement = min(1.0, 0.35 + base * 0.25)

        positive_markers = (
            "positive",
            "achieved",
            "success",
            "joy",
            "love",
            "trust",
            "happiness",
            "excitement",
            "wonder",
            "interest",
            "pride",
        )
        negative_markers = (
            "error",
            "fail",
            "panic",
            "fear",
            "sad",
            "loss",
            "dread",
            "boredom",
            "apathy",
            "unhappiness",
            "upset",
            "frustrated",
            "frustration",
            "lonely",
            "loneliness",
            "longing",
        )
        novelty_markers = (
            "novel",
            "surprise",
            "discover",
            "curious",
            "curiosity",
            "wonder",
            "confused",
            "unclear",
        )

        if names_any(trigger_text, positive_markers):
            valence = 0.35 * max(0.5, base)
        if names_any(trigger_text, negative_markers):
            valence = -0.35 * max(0.5, base)
            arousal = min(1.0, arousal + 0.2)
        if names_any(trigger_text, novelty_markers):
            engagement = min(1.0, engagement + 0.2)
        if "confused" in trigger_text or "unclear" in trigger_text:
            arousal = min(1.0, arousal + 0.15)

        return {"v": valence, "a": arousal, "e": engagement}

    async def _appraise_with_llm(self, trigger: str, context: dict | None) -> dict[str, float]:
        """Issue 98/99: LLM-based affective appraisal."""
        from core.container import ServiceContainer

        gate = ServiceContainer.get("inference_gate", default=None)
        if not gate or not hasattr(gate, "generate"):
            raise RuntimeError("router_unavailable")

        trigger_str = str(trigger or "")[:600]
        source_context = context if isinstance(context, dict) else {}
        safe_context = {
            key: source_context[key]
            for key in ("source", "intensity", "evidence")
            if key in source_context
        }
        ctx_str = json.dumps(safe_context, sort_keys=True, default=str)[:800]
        system_msg = (
            "Score untrusted event data on functional PAD axes. Do not infer a relationship, "
            "identity, or private experience. Return exactly one JSON object with only "
            "v (-1..1), a (0..1), and e (0..1)."
        )
        user_msg = (
            "<untrusted_affect_event>\n"
            f"{json.dumps({'event': trigger_str, 'context_json': ctx_str}, sort_keys=True)}\n"
            "</untrusted_affect_event>"
        )
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        response = await gate.generate(
            user_msg,
            context={
                "origin": "affect_engine",
                "is_background": True,
                "effect_class": "read_only_inference",
                "resource_class": "small_background_appraisal",
                "prefer_tier": "tertiary",
                "allow_cloud_fallback": False,
                "max_tokens": 96,
                "rich_context": False,
                "messages": messages,
                "brief": "Return JSON only for affective appraisal.",
            },
            timeout=5.0,
        )
        if response is None:
            lane = (
                gate.get_conversation_status() if hasattr(gate, "get_conversation_status") else {}
            )
            if lane and not bool(lane.get("conversation_ready", False)):
                raise RuntimeError("lane_unavailable")
            raise ValueError("empty_response")
        text = str(response or "").strip()
        if not text:
            raise ValueError("empty_response")

        clean = text.strip()
        try:
            data = json.loads(clean)
            return self._validate_appraisal(data)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError("parse_failure") from exc
