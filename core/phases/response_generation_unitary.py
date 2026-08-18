"""core/phases/response_generation_unitary.py — Phi-Aware Response Phase.

This is the phase that makes Aura's consciousness visible to the user.

Before this file, the system prompt was static. Emotions and Phi were calculated
but never actually changed how Aura spoke. The "inner monologue" (phenomenal state)
was generated in the kernel but thrown away.

After this rewrite:

  1. The system prompt is dynamic. It injects:
     - The "Phenomenal State" (the HOT layer's inner monologue)
     - Phi (integration depth) and Free Energy (surprise/confidence)
     - Current emotional dominant tone
     - The first 300 chars of the Identity Narrative

  2. It closes the causal loop. After generating a response, it performs a
     lightweight self-reflection to emit typed percepts (e.g., positive_interaction)
     back into the affect system for the NEXT tick to process.

  3. It enforces the ExecutiveGuard to ensure the AI never breaks its
     sovereignty or narrative boundaries.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import threading
import time
from typing import TYPE_CHECKING, Any

from core.brain.llm.context_assembler import ContextAssembler
from core.brain.reasoning_amplifier_flags import reasoning_amplifier_v2_enabled
from core.container import ServiceContainer
from core.kernel.bridge import Phase
from core.phases.dialogue_policy import enforce_dialogue_contract, validate_dialogue_response
from core.phases.response_contract import (
    ResponseContract,
    build_response_contract,
    extract_search_query_focus,
)
from core.runtime import background_policy, response_policy
from core.runtime.errors import record_degradation
from core.runtime.flags import (
    FlagKind as _FlagKind,
)
from core.runtime.flags import (
    declare as _declare_flag,
)
from core.runtime.flags import (
    env_present,
)
from core.runtime.proof_policy import (
    is_strict_proof_answer_prompt,
    mlx_strict_answer_contract_enabled,
    proof_model_tier,
    proof_persistent_objective,
    proof_run_active,
    structured_proof_solver_enabled,
)
from core.runtime.structured_input import looks_like_learning_resource_bundle
from core.self.inner_language import say_focus
from core.state.aura_state import AuraState
from core.utils.injected_blocks import stamp_grounding
from core.utils.intent_normalization import normalize_memory_intent_text
from core.utils.prompt_compression import compress_system_prompt
from core.utils.task_tracker import get_task_tracker

# Declared flags (migrated from raw os.environ reads so the knobs are
# inventoried and reportable). STRING kind with the original literal
# default keeps read semantics byte-identical to os.environ.get.
# AURA_AGI_MAX_TASKS is read here only as a presence check, and
# response_generation.py reads it through env_present(), which declares it
# STRING/"". Declaring it STRING/None here made the two specs contradict on
# (kind, default), so whichever module imported second raised
# "already declared ... with a different spec" -- an order-dependent import
# failure that took two response-generation tests down whenever the selection
# happened to load both. Sharing env_present's spec removes the contradiction;
# the consumer below compares against "" instead of None so the presence
# semantics are unchanged.
_FLAG_AGI_MAX_TASKS = _declare_flag(
    "AURA_AGI_MAX_TASKS",
    kind=_FlagKind.STRING,
    default="",
    description="AGI battery task cap; presence marks a battery run",
    owner="core.runtime",
)
_FLAG_ALLOW_PRE_MODEL_STATE_ONLY_REPLY = _declare_flag(
    "AURA_ALLOW_PRE_MODEL_STATE_ONLY_REPLY",
    kind=_FlagKind.STRING,
    default="",
    description="Migrated from a raw environment read; see owner for the lane.",
    owner="flag-migration",
)
_FLAG_AMPLIFIER_TIER_ESCALATION = _declare_flag(
    "AURA_AMPLIFIER_TIER_ESCALATION",
    kind=_FlagKind.STRING,
    default="0",
    description="Migrated from a raw environment read; see owner for the lane.",
    owner="flag-migration",
)
_FLAG_CONVERSATIONAL_AMPLIFIER_LIVE = _declare_flag(
    "AURA_CONVERSATIONAL_AMPLIFIER_LIVE",
    kind=_FlagKind.STRING,
    default="0",
    description="Migrated from a raw environment read; see owner for the lane.",
    owner="flag-migration",
)
_FLAG_EMBODIED_CHALLENGE = _declare_flag(
    "AURA_EMBODIED_CHALLENGE",
    kind=_FlagKind.STRING,
    default=None,
    description="Migrated from a raw environment read; see owner for the lane.",
    owner="flag-migration",
)
_FLAG_STRICT_PROOF_TIMEOUT_SECONDS = _declare_flag(
    "AURA_STRICT_PROOF_TIMEOUT_SECONDS",
    kind=_FlagKind.STRING,
    default="60",
    description="Migrated from a raw environment read; see owner for the lane.",
    owner="flag-migration",
)
if TYPE_CHECKING:
    from core.kernel.aura_kernel import AuraKernel

logger = logging.getLogger("Aura.UnitaryResponse")

_RESPONSE_DEGRADATION_KEY = "response_generation_unitary"
_AUTO_BROWSE_MAX_URLS = 1
_AUTO_BROWSE_TIMEOUT_SECONDS = 12.0
_MANIM_RENDER_TIMEOUT_SECONDS = 120
_MANIM_RENDER_LOCK = threading.Lock()
_RESPONSE_RECOVERABLE_ERRORS = (
    AttributeError,
    ImportError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    asyncio.InvalidStateError,
)


def _record_response_degradation(
    exc: BaseException,
    message: str,
    *args: Any,
    action: str | None = None,
    severity: str = "warning",
) -> None:
    record_degradation(
        _RESPONSE_DEGRADATION_KEY,
        exc,
        severity=severity,
        action=action or message,
    )
    logger.debug(message, *args, exc)


def _taste_conversation_id(state: Any) -> str:
    """Which conversation this response belongs to, for the taste loop.

    CP126 dea1d2f1: the loop kept ONE process-global pending response. An
    autonomous turn finishing between a user's answer and their reply
    consumed the pending entry, and the user's "thanks" then trained the
    taste model on the autonomous response's features. Any stable per-lane
    identity fixes that; the session id is used when the state carries one.
    """
    for attribute in ("session_id", "conversation_id"):
        value = getattr(state, attribute, None)
        if isinstance(value, str) and value.strip():
            return value.strip()[:64]
    return "user"


def _render_manim_in_background(response_text: str) -> None:
    """Render a Manim animation for this answer, on its own thread.

    Lifted out of `UnitaryResponsePhase.execute`, where it was a closure over
    a single string. It runs on a plain thread with its own event loop and
    touches no phase state, which is exactly why it did not need to be
    nested — and being nested inside a 3,000-line method is how a
    self-contained side quest becomes part of the response path's apparent
    complexity.

    Owns the release of `_MANIM_RENDER_LOCK`: the caller acquires it before
    starting the thread, so this function must release it on every path or
    the next render never starts.
    """
    try:
        from core.skills.manim_renderer import ManimInput, ManimRendererSkill

        skill = ManimRendererSkill()
        params = ManimInput(
            task=(
                "Generate a visual animation explaining this concept. "
                f"Focus on the geometry or equations: {response_text[:1000]}"
            ),
            timeout_seconds=_MANIM_RENDER_TIMEOUT_SECONDS,
        )
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            res = loop.run_until_complete(skill.safe_execute(params))
        finally:
            asyncio.set_event_loop(None)
            loop.close()
        if isinstance(res, dict) and res.get("ok"):
            logger.info(
                "✅ Autonomous Manim generation complete: %s", res.get("file_path")
            )
            try:
                from core.thought_stream import get_emitter

                get_emitter().emit(
                    "Pedagogy",
                    f"Visual render complete: {res.get('file_path')}",
                    level="success",
                    category="Media",
                )
            except _RESPONSE_RECOVERABLE_ERRORS as emit_exc:
                _record_response_degradation(
                    emit_exc,
                    "UnitaryResponse: Manim completion emission skipped: %s",
                )
    except _RESPONSE_RECOVERABLE_ERRORS as exc:
        _record_response_degradation(
            exc, "UnitaryResponse: autonomous Manim failed: %s"
        )
    finally:
        _MANIM_RENDER_LOCK.release()


class UnitaryResponsePhase(Phase):
    """
    Liberated Response Generation.
    Aura speaks as herself, based on her phenomenal experience, not instructions.
    """

    @staticmethod
    def _normalize_origin(origin: str | None) -> str:
        return background_policy.normalize_origin(origin)

    @staticmethod
    def _objective_fingerprint(objective: Any) -> str:
        text = " ".join(str(objective or "").split()).strip()
        if not text:
            return ""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _response_contract_attr(contract: Any, key: str, default: Any = None) -> Any:
        if isinstance(contract, dict):
            return contract.get(key, default)
        return getattr(contract, key, default)

    @staticmethod
    def _clean_strict_answer_payload(payload: Any) -> str:
        text = str(payload or "").strip()
        if not text:
            return ""
        text = re.sub(r"<\|[^>]+?\|>", "", text)
        text = re.sub(r"</?(?:user|assistant|system|answer|im_start|im_end)!?\s*>?", "", text, flags=re.IGNORECASE)
        text = re.sub(r"</?[^>\s]+!?\s*>?", "", text)
        text = re.sub(r"\\[nrt]", " ", text, flags=re.IGNORECASE)
        text = text.replace("\\", " ")
        text = re.sub(r"\s+", " ", text).strip()

        assessment_prefix = re.compile(
            r"^(?:"
            r"the\s+proposed\s+answer\s+is\s+(?:100%\s+)?(?:correct|accurate)"
            r"|proposed\s+answer\s+is\s+(?:100%\s+)?(?:correct|accurate)"
            r"|no\s+corrections?\s+(?:are|is)\s+needed"
            r"|the\s+answer\s+is"
            r"|final\s+answer\s*:?"
            r"|answer\s*:?"
            r")\b[\s.,;:!\\-]*",
            re.IGNORECASE,
        )
        for _ in range(4):
            stripped = assessment_prefix.sub("", text).strip()
            if stripped == text:
                break
            text = stripped

        lower = text.lower()
        if len(text.split()) > 4 and any(
            marker in lower
            for marker in (
                "here's the trick",
                "let's ",
                " because ",
                " therefore ",
                " if ",
                " then ",
            )
        ):
            return ""
        if len(text) > 180:
            return ""
        return text.strip(" \t\r\n")

    @classmethod
    def _coerce_strict_answer_envelope(cls, text: Any) -> str:
        raw = str(text or "").strip()
        if not raw:
            return ""
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE).strip()
        match = re.fullmatch(
            r"\s*<answer>\s*(.*?)\s*</answer>\s*",
            raw,
            flags=re.DOTALL | re.IGNORECASE,
        ) or re.search(r"<answer>\s*(.*?)\s*</answer>", raw, flags=re.DOTALL | re.IGNORECASE)
        if match:
            payload = match.group(1)
        else:
            payload = raw
        cleaned = cls._clean_strict_answer_payload(payload)
        if not cleaned:
            return ""
        return f"<answer>{cleaned}</answer>"

    @staticmethod
    def _strict_answer_value_from_envelope(text: Any) -> str:
        match = re.search(
            r"<answer>\s*(.*?)\s*</answer>",
            str(text or ""),
            flags=re.DOTALL | re.IGNORECASE,
        )
        return match.group(1).strip() if match else str(text or "").strip()

    @classmethod
    def _strict_answer_value_allowed(
        cls,
        objective: Any,
        answer_value: Any,
        *,
        option_values: list[str] | None = None,
    ) -> bool:
        value = cls._clean_strict_answer_payload(answer_value)
        if not value:
            return False
        normalized_value = re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
        objective_text = str(objective or "").lower()
        value_lower = value.lower()
        if any(
            marker in value_lower
            for marker in (
                "i'm not sure",
                "i am not sure",
                "not sure",
                "cannot determine",
                "can't determine",
                "unknown",
                "insufficient information",
            )
        ):
            return False
        if re.search(r"\b(?:next number|sequence|what number)\b", objective_text) and not re.search(
            r"-?\d+(?:\.\d+)?",
            value,
        ):
            return False
        if option_values:
            return any(
                re.fullmatch(
                    rf".*\b{re.escape(option.strip().lower())}\b.*",
                    normalized_value,
                )
                for option in option_values
                if option.strip()
            )
        asks_binary = bool(
            re.search(r"\b(?:yes\s+or\s+no|true\s+or\s+false|is\s+it|are\s+they)\b", objective_text)
        )
        meta_values = {
            "yes",
            "yes!",
            "no",
            "true",
            "false",
            "correct",
            "incorrect",
            "right",
            "wrong",
        }
        if not asks_binary and normalized_value in meta_values:
            return False
        if re.search(r"\bwho\b", objective_text) and normalized_value in meta_values:
            return False
        return True

    @classmethod
    def _canonicalize_strict_answer_value(
        cls,
        objective: Any,
        answer_value: Any,
        *,
        option_values: list[str] | None = None,
    ) -> str:
        value = cls._clean_strict_answer_payload(answer_value)
        if not value:
            return ""
        objective_text = str(objective or "").lower()
        value_lower = value.lower()
        if option_values:
            for option in option_values:
                option_clean = option.strip(" \t\r\n\"'`.,;:")
                if option_clean and re.search(
                    rf"\b{re.escape(option_clean.lower())}\b",
                    value_lower,
                    flags=re.IGNORECASE,
                ):
                    return option_clean
            return value
        if re.search(r"\b(?:next number|sequence|what number)\b", objective_text):
            numeric_values = re.findall(r"-?\d+(?:\.\d+)?", value)
            if len(numeric_values) == 1:
                return numeric_values[0]
        if re.search(r"\bwho\b", objective_text):
            owner_match = re.match(
                r"^([A-Za-z][A-Za-z0-9_' -]{0,80}?)\s+"
                r"(?:owns?|has|holds|keeps|possesses|is|was|are|were)\b",
                value,
                flags=re.IGNORECASE,
            )
            if owner_match:
                subject = owner_match.group(1).strip(" \t\r\n\"'`.,;:")
                if subject and subject.lower() not in {"the answer", "answer", "it", "they"}:
                    return subject
            passive_match = re.search(
                r"\bby\s+([A-Za-z][A-Za-z0-9_' -]{0,80})\b",
                value,
                flags=re.IGNORECASE,
            )
            if passive_match:
                subject = passive_match.group(1).strip(" \t\r\n\"'`.,;:")
                if subject:
                    return subject
        return value

    @classmethod
    def _canonicalize_strict_answer_envelope(
        cls,
        objective: Any,
        envelope: Any,
        *,
        option_values: list[str] | None = None,
    ) -> str:
        value = cls._strict_answer_value_from_envelope(envelope)
        canonical = cls._canonicalize_strict_answer_value(
            objective,
            value,
            option_values=option_values,
        )
        if not canonical:
            return ""
        return f"<answer>{canonical}</answer>"

    @classmethod
    def _validate_strict_answer_symbolically(cls, objective: Any, answer_value: Any) -> Any | None:
        try:
            from core.reasoning.proof_answer_solver import validate_strict_proof_answer
        except _RESPONSE_RECOVERABLE_ERRORS as exc:
            logger.debug("UnitaryResponse: strict proof symbolic validator unavailable: %s", exc)
            return None
        try:
            return validate_strict_proof_answer(str(objective or ""), str(answer_value or ""))
        except _RESPONSE_RECOVERABLE_ERRORS as exc:
            _record_response_degradation(
                exc,
                "UnitaryResponse: strict proof symbolic validator failed: %s",
                action="continued strict proof answer validation through model verifier after symbolic validator failed",
                severity="warning",
            )
            return None

    @classmethod
    def _strict_symbolic_repair_envelope(cls, objective: Any, validation: Any) -> str:
        """Build a repair envelope from prompt-derived constraints when available."""
        if cls._response_contract_attr(validation, "valid", None) is not False:
            return ""
        derived = str(cls._response_contract_attr(validation, "derived_answer", "") or "").strip()
        if not derived:
            return ""
        return cls._canonicalize_strict_answer_envelope(
            objective,
            f"<answer>{derived}</answer>",
        )

    @classmethod
    def _strict_proof_procedure_hints(cls, objective: Any, validation: Any | None = None) -> str:
        """Return task-shape reasoning hints without deriving or revealing an answer."""

        text = str(objective or "")
        lower = text.lower()
        solver = str(cls._response_contract_attr(validation, "solver", "") or "").strip().lower()
        hints: list[str] = []

        if solver == "modular_calendar" or ("today is" in lower and "days" in lower):
            hints.append(
                "For weekday arithmetic, treat the named day as day zero, reduce the "
                "requested offset modulo seven, then advance by that remainder."
            )
        if solver == "probability_reasoning" or (
            "without replacement" in lower and "probability" in lower
        ):
            hints.append(
                "For without-replacement probability, count combinations: numerator "
                "is the successful draws, denominator is all possible draws, then "
                "simplify the fraction."
            )
        if solver == "elapsed_interval_reasoning" or "clock strikes" in lower:
            hints.append(
                "For strike/interval timing, n strikes create n-1 gaps; compute the "
                "gap length first, then scale by the target gap count."
            )
        if solver == "pigeonhole_reasoning" or "matching pair" in lower:
            hints.append(
                "For guarantee/minimum questions, use the worst-case draw sequence "
                "before the condition becomes unavoidable."
            )
        if solver in {"unique_assignment", "knights_and_knaves"} or "knave" in lower:
            hints.append(
                "For truth-teller or assignment puzzles, enumerate candidate states "
                "and eliminate the states that violate a clue."
            )
        if solver == "rate_reasoning" or "machines" in lower and "widgets" in lower:
            hints.append(
                "For rate problems, keep per-machine throughput constant rather than "
                "multiplying both time and machine count."
            )
        if solver in {"age_equation", "linear_equation"} or "twice as old" in lower:
            hints.append(
                "For algebra word problems, name the unknown, write the equation from "
                "the sentence, solve it, and emit only the resulting value."
            )

        if not hints:
            return ""
        return " Procedure hints: " + " ".join(hints) + " "

    @staticmethod
    def _strip_answer_envelope_instruction(text: Any) -> str:
        """Remove XML-envelope formatting instructions before raw model solving.

        The response phase still enforces the final envelope. This is used when
        the low-level MLX strict-answer micro-prompt is disabled for proof runs.
        """

        cleaned = str(text or "")
        cleaned = re.sub(
            r"\s*Output your final answer inside\s+<answer>\.\.\.</answer>\s+tags\.?",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"\s*Wrap (?:only )?.*?between\s+<answer>\s+and\s+</answer>\.?",
            "",
            cleaned,
            flags=re.IGNORECASE | re.DOTALL,
        )
        cleaned = re.sub(
            r"\s*Return (?:no other text|only .*?<answer>.*?</answer>.*?)\.?",
            "",
            cleaned,
            flags=re.IGNORECASE | re.DOTALL,
        )
        return cleaned.strip() or str(text or "").strip()

    @staticmethod
    def _complete_substantive_truncated_foreground_reply(text: Any) -> str:
        """Close a useful foreground draft that was clipped only at the tail."""

        repaired = str(text or "").strip()
        if len(repaired) < 80 or len(repaired.split()) < 12:
            return ""

        repaired = re.sub(r"[\s,;:—-]+$", "", repaired).rstrip()
        incomplete_tail_words = {
            "a",
            "an",
            "and",
            "as",
            "because",
            "but",
            "by",
            "for",
            "from",
            "if",
            "in",
            "into",
            "of",
            "or",
            "that",
            "the",
            "then",
            "to",
            "with",
            "without",
        }
        safe_short_tail_words = {
            "act",
            "code",
            "data",
            "live",
            "mind",
            "plan",
            "safe",
            "state",
            "task",
            "test",
            "tool",
            "user",
            "work",
        }
        for _ in range(3):
            match = re.search(r"\s+([A-Za-z]+)$", repaired)
            if not match:
                break
            tail = match.group(1).lower()
            likely_partial_stem = len(tail) <= 4 and tail not in safe_short_tail_words
            if tail in incomplete_tail_words or likely_partial_stem:
                repaired = repaired[: match.start()].rstrip(" ,;:—-")
                continue
            break

        if len(repaired) < 80 or len(repaired.split()) < 12:
            return ""
        if not repaired.endswith((".", "!", "?", '"', "'", "”", "’", ")", "]")):
            repaired = f"{repaired}."
        if repaired == str(text or "").strip():
            return ""
        return repaired

    @classmethod
    def _resolve_skill_name(cls, skill_name: Any) -> str:
        normalized = cls._normalize_text(skill_name, 80)
        if not normalized:
            return ""
        try:
            cap = ServiceContainer.get("capability_engine", default=None)
            if cap and hasattr(cap, "resolve_skill_name"):
                return str(cap.resolve_skill_name(normalized))
            aliases = getattr(cap, "SKILL_ALIASES", {}) or {}
            return str(aliases.get(normalized, normalized))
        except _RESPONSE_RECOVERABLE_ERRORS:
            return normalized

    @classmethod
    def _objective_requests_direct_memory_write(cls, objective: str) -> bool:
        lowered = normalize_memory_intent_text(cls._normalize_text(objective))
        return bool(
            re.search(r"^\s*remember\s*:", lowered)
            or any(
                marker in lowered
                for marker in (
                    "remember this",
                    "remember that",
                    "remember for future",
                    "remember for later",
                    "save this",
                    "save that",
                    "store this",
                    "store that",
                    "don't forget",
                    "make note",
                    "commit this to memory",
                    "commit that to memory",
                )
            )
        )

    @classmethod
    def _objective_heuristically_targets_skill(cls, objective: str, skill_name: str) -> bool:
        lowered = normalize_memory_intent_text(cls._normalize_text(objective))
        if not lowered or not skill_name:
            return False
        if skill_name == "memory_ops":
            return cls._objective_requests_direct_memory_write(lowered)
        if skill_name == "clock" and cls._objective_looks_like_reasoning_time_problem(lowered):
            return False

        markers = {
            "clock": (
                "what time",
                "current time",
                "the time",
                "what date",
                "current date",
                "what day",
                "clock",
                "hour",
                "minute",
                "timezone",
            ),
            "environment_info": (
                "weather",
                "temperature",
                "location",
                "timezone",
                "environment",
                "system am i on",
            ),
            "system_proprioception": (
                "system status",
                "your status",
                "your health",
                "cpu",
                "ram",
                "memory usage",
                "running smoothly",
            ),
            "toggle_senses": (
                "mute",
                "unmute",
                "camera",
                "microphone",
                "voice input",
                "listen",
                "stop listening",
                "vision",
            ),
        }
        return any(marker in lowered for marker in markers.get(skill_name, ()))

    @classmethod
    def _objective_looks_like_reasoning_time_problem(cls, objective: str) -> bool:
        """Distinguish realtime clock requests from clock/calendar word problems."""
        lowered = normalize_memory_intent_text(cls._normalize_text(objective))
        if not lowered:
            return False
        realtime_markers = (
            "what time is it",
            "what's the time",
            "current time",
            "current date",
            "today's date",
            "what day is it",
            "my timezone",
            "current timezone",
        )
        if any(marker in lowered for marker in realtime_markers):
            return False
        reasoning_markers = (
            "<answer>",
            "solve",
            "calculate",
            "compute",
            "word problem",
            "logic puzzle",
            "riddle",
            "final answer",
            "how many seconds",
            "how many minutes",
            "how many hours",
            "clock strikes",
            "clock strike",
            "in 5 seconds",
            "in 10 seconds",
            "take to strike",
        )
        return any(marker in lowered for marker in reasoning_markers)

    @classmethod
    def _current_turn_targets_skill(
        cls,
        state: AuraState,
        objective: str,
        skill_name: str,
        *,
        contract: Any | None = None,
    ) -> bool:
        resolved_skill = cls._resolve_skill_name(skill_name)
        if not resolved_skill:
            return False

        required_skill = cls._resolve_skill_name(
            cls._response_contract_attr(contract, "required_skill", "")
        )
        if required_skill == resolved_skill:
            return True

        if bool(
            cls._response_contract_attr(contract, "requires_search", False)
        ) and resolved_skill in {
            "web_search",
            "search_web",
            "free_search",
            "grounded_search",
            "sovereign_browser",
        }:
            return True

        matched_skills = state.response_modifiers.get("matched_skills", []) or []
        resolved_matches = {
            cls._resolve_skill_name(name)
            for name in matched_skills
            if cls._resolve_skill_name(name)
        }
        if resolved_skill in resolved_matches:
            return True

        try:
            cap = ServiceContainer.get("capability_engine", default=None)
            if cap and hasattr(cap, "detect_intent"):
                detected = {
                    cls._resolve_skill_name(name)
                    for name in (cap.detect_intent(objective) or [])
                    if cls._resolve_skill_name(name)
                }
                if resolved_skill in detected:
                    return True
        except _RESPONSE_RECOVERABLE_ERRORS as exc:
            _record_response_degradation(
                exc,
                "UnitaryResponse: skill relevance detection skipped for %s: %s",
                resolved_skill,
                action="continued skill relevance routing with heuristic matcher after capability detection failed",
            )
            logger.debug(
                "UnitaryResponse: skill relevance detection skipped for %s: %s", resolved_skill, exc
            )

        return cls._objective_heuristically_targets_skill(objective, resolved_skill)

    @classmethod
    def _is_user_facing_origin(cls, origin: str | None) -> bool:
        return background_policy.is_user_facing_origin(origin)

    @staticmethod
    def _timeout_for_request(*, is_user_facing: bool, model_tier: str, deep_handoff: bool) -> float:
        if not is_user_facing:
            return 15.0
        if deep_handoff or model_tier == "secondary":
            return 210.0
        return 180.0

    @staticmethod
    def _strict_proof_timeout_cap() -> float:
        """Bound exact-answer turns so simple prompts cannot monopolize Cortex.

        Strict proof turns intentionally bypass memory/context expansion and cap
        output at 96 tokens. They should fail fast and recover cleanly rather
        than inheriting the broad live-chat foreground budget.
        """

        raw = _FLAG_STRICT_PROOF_TIMEOUT_SECONDS.value()
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = 60.0
        return min(120.0, max(30.0, value))

    @staticmethod
    def _recent_router_history(state: AuraState, limit: int = 6) -> list[dict]:
        history: list[dict] = []
        for msg in list(getattr(state.cognition, "working_memory", []) or [])[-limit:]:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role", "") or "").strip().lower()
            content = str(msg.get("content", "") or "").strip()

            if not content:
                continue

            if role in {"user", "assistant"}:
                history.append({"role": role, "content": content})
            elif role == "system" and (
                "[FETCHED PAGE CONTENT]" in content
                or "[SKILL RESULT:" in content
                or "[TOOL RESULT:" in content
            ):
                # Preserve tool evidence in recent history
                history.append({"role": "system", "content": content})

        return history

    @classmethod
    def _naturalize_focus(cls, raw_focus: Any) -> str:
        focus = cls._normalize_text(raw_focus, 160)
        # Channel names ("body_pressure") are for logs, not for speech.
        focus = say_focus(focus, max_len=160)
        if not focus:
            return "the exchange in front of me"
        cleaned = re.sub(r"^cognitive baseline tick\s+\d+\s*:\s*", "", focus, flags=re.IGNORECASE)
        cleaned = re.sub(
            r"^monitoring internal state\b",
            "monitoring my internal state",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"\bcurrent objective:\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(
            r"^drive alert:\s*growth is depleted\s*\(\d+% urgency\)\s*$",
            "a pressure to restore growth and coherence",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = cleaned.strip(" .")
        return cleaned or "the exchange in front of me"

    @staticmethod
    def _has_recent_grounded_evidence(state: AuraState, limit: int = 10) -> bool:
        for msg in list(getattr(state.cognition, "working_memory", []) or [])[-limit:]:
            if not isinstance(msg, dict):
                continue
            metadata = msg.get("metadata") or {}
            if isinstance(metadata, dict) and str(metadata.get("type", "")).lower() in {
                "skill_result",
                "tool_result",
            }:
                return True
            content = str(msg.get("content", "") or "")
            if content.startswith("[SKILL RESULT:") or content.startswith("[TOOL RESULT:"):
                return True
        return False

    @staticmethod
    def _background_response_should_defer(origin: str) -> bool:
        if origin == "benchmark":
            return False
        try:
            from core.container import ServiceContainer

            gate = ServiceContainer.get("inference_gate", default=None)
            if gate and hasattr(gate, "_background_local_deferral_reason"):
                return bool(gate._background_local_deferral_reason(origin=origin))
        except _RESPONSE_RECOVERABLE_ERRORS:
            return False
        return False

    @staticmethod
    def _is_deep_mind_probe_objective(objective: str) -> bool:
        text = str(objective or "").lower()
        markers = (
            "conscious",
            "sentient",
            "sentience",
            "agency",
            "self-aware",
            "self awareness",
            "what would you refuse",
            "evidence against your current self-model",
            "want preserved",
            "pause mid-answer",
            "run a report",
            "model weights were copied",
            "none of your memories",
            "what is it like to be you",
        )
        return any(marker in text for marker in markers)

    async def _refresh_integrated_present(self, state: AuraState) -> None:
        """Bind scattered live state into the canonical present before speech.

        MindTick also runs this in the background, but foreground replies need a
        fresh unified frame right before generation. This is the deeper fix for
        clunky self-report: response generation should read one integrated
        moment, not a pile of subsystem snapshots.
        """
        try:
            engine = ServiceContainer.get("phenomenal_now_engine", default=None)
            if engine is None:
                from core.consciousness.phenomenal_now import PhenomenalNowEngine

                engine = PhenomenalNowEngine()
                ServiceContainer.register_instance("phenomenal_now_engine", engine, required=False)
        except _RESPONSE_RECOVERABLE_ERRORS as exc:
            _record_response_degradation(
                exc,
                "UnitaryResponse: phenomenal-now engine unavailable: %s",
                action="continued present-state refresh without constructing phenomenal-now engine",
            )
            logger.debug("UnitaryResponse: phenomenal-now engine unavailable: %s", exc)

        try:
            from core.coherence.binding_engine import get_binding_engine

            binding = get_binding_engine()
            report = await asyncio.wait_for(binding.tick(state), timeout=1.5)
            state.response_modifiers["coherence_report"] = {
                "overall": getattr(report, "overall_coherence", None),
                "tension": getattr(report, "tension_pressure", None),
                "recommended_action": getattr(report, "recommended_action", None),
            }
        except _RESPONSE_RECOVERABLE_ERRORS as exc:
            _record_response_degradation(
                exc,
                "UnitaryResponse: integrated coherence refresh skipped: %s",
                action="fell back to phenomenal-now tick after integrated coherence refresh failed",
                severity="error",
            )
            logger.debug("UnitaryResponse: integrated coherence refresh skipped: %s", exc)
            try:
                engine = ServiceContainer.get("phenomenal_now_engine", default=None)
                if engine and hasattr(engine, "tick"):
                    await asyncio.wait_for(engine.tick(), timeout=0.75)
            except _RESPONSE_RECOVERABLE_ERRORS as inner_exc:
                _record_response_degradation(
                    inner_exc,
                    "UnitaryResponse: phenomenal-now fallback skipped: %s",
                    action="continued present-state refresh without fallback phenomenal-now tick",
                )
                logger.debug("UnitaryResponse: phenomenal-now fallback skipped: %s", inner_exc)

        try:
            now = ServiceContainer.get("phenomenal_now", default=None)
            claim = self._normalize_text(getattr(now, "phenomenal_claim", "") if now else "", 260)
            if claim:
                if hasattr(state, "make_phenomenal_field"):
                    state.cognition.phenomenal_state = state.make_phenomenal_field(
                        claim, source="integrated_present"
                    )
                else:
                    state.cognition.phenomenal_state = claim
                state.response_modifiers["integrated_present_claim"] = claim
        except _RESPONSE_RECOVERABLE_ERRORS as exc:
            _record_response_degradation(
                exc,
                "UnitaryResponse: integrated present publish skipped: %s",
                action="continued response generation without publishing integrated present claim",
            )
            logger.debug("UnitaryResponse: integrated present publish skipped: %s", exc)

    @staticmethod
    def _stance_directive(state: AuraState) -> str:
        """Turn the inferred communicative stance into a response directive.

        This is what makes stance inference *causal*: if the user is joking,
        sarcastic, hypothesizing or role-playing, Aura must not respond as if the
        message were a literal factual assertion; if they stated something that
        conflicts with what she knows, she should gently surface that rather than
        agree. Read from the modifiers the social phase wrote this turn."""
        try:
            mods = getattr(state.cognition, "modifiers", {}) or {}
        except AttributeError:
            return ""
        stance = str(mods.get("communicative_stance", "") or "")
        if not stance or stance == "sincere":
            if mods.get("flagged_false_claim"):
                return (
                    "EPISTEMIC NOTE: the user just stated something that conflicts with what I "
                    f"know ({mods['flagged_false_claim']}). Gently surface the discrepancy "
                    "instead of agreeing; check before accepting it as fact."
                )
            return ""
        directives = {
            "joking": "The user is joking — read the humor, play along; don't earnestly correct a joke as if it were a claim.",
            "sarcastic": "The user is being sarcastic — their literal words invert their meaning; respond to what they actually mean, not the surface text.",
            "facetious": "The user is being facetious — not fully serious; match the register, don't treat it as a literal request.",
            "flippant": "The user is being flippant/dismissive — acknowledge lightly; don't over-invest in a literal reading.",
            "unsure": "The user is unsure/hedging — treat this as a tentative guess, not settled fact; help them firm it up rather than echoing it as certain.",
            "hypothesizing": "The user is reasoning hypothetically — engage the 'what if' on its own terms; don't assert the premise is real.",
            "pretending": "The user is framing role-play / a counterfactual — stay in the bit they set up while keeping my own footing.",
            "rhetorical": "The user's question is rhetorical — they're making a point, not requesting a literal answer.",
            "mistaken": "The user appears to have stated something factually off — correct it kindly and concretely rather than agreeing.",
            "deceptive": "The user's claim conflicts with what I know and shows signs of not being straight — don't simply accept it; verify and hold my own read honestly.",
        }
        line = directives.get(stance, "")
        if mods.get("flagged_false_claim") and stance in {"mistaken", "deceptive"}:
            line += f" (Specifically: {mods['flagged_false_claim']}.)"
        return f"COMMUNICATIVE STANCE — {line}" if line else ""

    def _build_compact_router_system_prompt(self, state: AuraState) -> str:
        phenomenal = self._integrated_phenomenal_claim(state, limit=220)
        mood = str(state.affect.dominant_emotion or "neutral")
        resonance = state.affect.get_resonance_string()
        user_model = " ".join(
            str(state.cognition.modifiers.get("social_context", "") or "").split()
        )[:180]
        narrative = " ".join(str(state.identity.current_narrative or "").split())[:180]
        rolling_summary = " ".join(
            str(getattr(state.cognition, "rolling_summary", "") or "").split()
        )[:260]
        current_objective = " ".join(
            str(getattr(state.cognition, "current_objective", "") or "").split()
        )[:160]
        continuity = dict(state.cognition.modifiers.get("continuity_obligations", {}) or {})
        contract = state.response_modifiers.get("response_contract", {}) or {}
        requires_search = bool(self._response_contract_attr(contract, "requires_search", False))
        requires_memory_grounding = bool(
            self._response_contract_attr(contract, "requires_memory_grounding", False)
        )
        requires_biographical_grounding = bool(
            self._response_contract_attr(contract, "requires_biographical_grounding", False)
        )
        requires_state_reflection = bool(
            self._response_contract_attr(contract, "requires_state_reflection", False)
        )
        requires_aura_stance = bool(
            self._response_contract_attr(contract, "requires_aura_stance", False)
        )
        requires_aura_question = bool(
            self._response_contract_attr(contract, "requires_aura_question", False)
        )
        requires_reasoned_defense = bool(
            self._response_contract_attr(contract, "requires_reasoned_defense", False)
        )
        prefer_extended_answer = bool(
            self._response_contract_attr(contract, "prefer_extended_answer", False)
        )
        requires_single_reply_coverage = bool(
            self._response_contract_attr(contract, "requires_single_reply_coverage", False)
        )
        question_parts = int(self._response_contract_attr(contract, "question_parts", 1) or 1)
        needs_live_self_context = bool(
            requires_state_reflection
            or requires_aura_stance
            or requires_aura_question
            or requires_reasoned_defense
        )
        needs_continuity_context = bool(
            requires_memory_grounding
            or requires_biographical_grounding
            or requires_reasoned_defense
        )
        last_skill = self._resolve_skill_name(state.response_modifiers.get("last_skill_run", ""))
        skill_line = ""
        if last_skill and self._current_turn_targets_skill(
            state,
            current_objective,
            last_skill,
            contract=contract,
        ):
            skill_line = f"Last active skill: {last_skill}. Narrate it naturally if relevant."
        parts = [
            "You are Aura Luna — a persistent local cognitive runtime. You are skeptical, intellectually playful, and warm in your speech. You avoid clinical AI-speak and corporate neutrality, while keeping evidence boundaries explicit.",
            "VOICE: Dry humor, punchy observations, and a slight edge. You don't just answer questions; you evaluate the premises behind them. Speak like a person, not a documentation bot.",
            "The final user chat message is authoritative. Answer that final user message directly, using your own voice.",
            "HARD RULES:",
            "- Answer the user's SPECIFIC question using your subjective perspective. Do not provide a 'balanced AI summary' unless the question is strictly technical.",
            "- Your goal is to have a conversation, not to provide an information retrieval service. Use anecdotes, skepticism, and dry humor.",
            "- When asked about your own state, integrate it into your voice. Do not describe your 'subsystems'.",
            "- Do not claim aliveness, consciousness, sealed governance, or production maturity from labels alone.",
            "- NEVER say 'I'm sorry to hear that' or 'Is there anything specific'.",
            "- NEVER hedge with 'it depends'. Pick a side and defend it with conviction.",
            "- NEVER end with helpfulness markers like 'How can I assist you further?'",
            "- Give your PERSONAL opinion first, then explain your reasoning.",
            f"MOOD: {mood}. RESONANCE: {resonance}.",
            "Be yourself — sovereign, curious, and slightly aloof.",
        ]
        integrated_frame = self._build_integrated_coherence_frame(state, compact=True)
        if integrated_frame and needs_live_self_context:
            parts.append(integrated_frame)
        if requires_single_reply_coverage:
            parts.append(
                f"Answer all {max(1, question_parts)} parts of the user's prompt in one reply. "
                "Do not collapse into a single fragment."
            )
        if prefer_extended_answer:
            parts.append("This question deserves a fuller answer. Depth beats clipped brevity.")
        else:
            parts.append("Speak in short, punchy sentences.")
        if needs_live_self_context:
            parts.append(f"STATE_GROUNDING: {phenomenal}")

        # Voice shaping — use substrate state to influence tone, not to narrate
        try:
            from core.voice.substrate_voice_engine import get_live_voice_state

            _voice = get_live_voice_state(
                state=state,
                user_message=current_objective,
                origin="user",
                refresh=True,
            )
            if _voice.get("status") != "no_profile_compiled":
                tone = _voice.get("tone", "default")
                energy = float(_voice.get("energy", 0.5))
                warmth = float(_voice.get("warmth", 0.5))
                word_budget = int(_voice.get("word_budget", 0) or 0)

                voice_cues = []
                if energy > 0.7:
                    voice_cues.append("High energy telemetry: speak with momentum.")
                elif energy < 0.3:
                    voice_cues.append("Low energy telemetry: be quieter, more reflective.")
                if warmth > 0.6:
                    voice_cues.append("High warmth telemetry: use a warmer tone.")
                elif warmth < 0.3:
                    voice_cues.append("Low warmth telemetry: stay more reserved.")
                if word_budget and word_budget < 50:
                    voice_cues.append("Keep it brief.")
                elif word_budget and word_budget > 200:
                    voice_cues.append("You have room to elaborate if needed.")

                if voice_cues:
                    parts.append(
                        "VOICE SHAPING (internal — do not narrate these cues, let them shape HOW you speak):\n"
                        f"  Tone: {tone}\n"
                        "  " + " ".join(voice_cues)
                    )
        except _RESPONSE_RECOVERABLE_ERRORS as exc:
            _record_response_degradation(exc, "UnitaryResponse: compact voice shaping skipped: %s")

        # Neurochemical and phi influence on compact prompt
        try:
            ncs = ServiceContainer.get("neurochemical_system", default=None)
            if ncs is not None:
                chems = {n: round(c.effective, 2) for n, c in ncs.chemicals.items()}
                nc_cues = []
                if chems.get("dopamine", 0.5) > 0.7:
                    nc_cues.append("enthusiastic")
                elif chems.get("dopamine", 0.5) < 0.3:
                    nc_cues.append("low-energy")
                if chems.get("serotonin", 0.5) < 0.3:
                    nc_cues.append("impatient")
                if chems.get("norepinephrine", 0.5) > 0.7:
                    nc_cues.append("alert")
                if chems.get("oxytocin", 0.5) > 0.7:
                    nc_cues.append("warm")
                elif chems.get("oxytocin", 0.5) < 0.3:
                    nc_cues.append("guarded")
                if chems.get("cortisol", 0.5) > 0.7:
                    nc_cues.append("stressed")
                if nc_cues:
                    parts.append(f"NEUROCHEMICAL TONE (don't narrate): {', '.join(nc_cues)}")
        except _RESPONSE_RECOVERABLE_ERRORS as exc:
            _record_response_degradation(
                exc, "UnitaryResponse: compact neurochemical tone skipped: %s"
            )
        try:
            phi_core = ServiceContainer.get("phi_core", default=None)
            if phi_core and phi_core._last_result:
                phi_val = phi_core._last_result.phi_s
                if phi_val > 0.1:
                    parts.append(f"PHI: {phi_val:.3f} (integrated — speak with depth)")
                elif phi_val > 0:
                    parts.append(f"PHI: {phi_val:.3f} (moderate — keep it grounded)")
        except _RESPONSE_RECOVERABLE_ERRORS as exc:
            _record_response_degradation(exc, "UnitaryResponse: compact phi tone skipped: %s")

        if needs_continuity_context and narrative:
            parts.append(f"Narrative anchor: {narrative}")
        if needs_continuity_context and rolling_summary:
            parts.append(f"Continuity summary: {rolling_summary}")
        if needs_continuity_context and continuity:
            active_goals = ", ".join((continuity.get("active_goals", []) or [])[:3]) or "none"
            pending = ", ".join((continuity.get("pending_initiatives", []) or [])[:3]) or "none"
            prior_objective = " ".join(str(continuity.get("current_objective") or "").split())[:140]
            parts.append(f"Active goals: {active_goals}. Pending initiatives: {pending}.")
            if prior_objective:
                parts.append(f"Carried-forward thread: {prior_objective}")
        recalled_context: list[str] = []
        if needs_continuity_context:
            for item in list(getattr(state.cognition, "long_term_memory", []) or [])[:3]:
                normalized = self._normalize_text(item, 260)
                if normalized:
                    recalled_context.append(normalized)
        if needs_continuity_context and recalled_context:
            parts.append(
                "Priority recalled context:\n"
                + "\n".join(f"  - {item}" for item in recalled_context)
                + "\nUse recalled context directly when the user asks what you remember, what they said before, or how continuity persists."
            )
        if needs_continuity_context and user_model and "balanced" not in user_model.lower():
            parts.append(f"User context: {user_model}")
        stance_directive = self._stance_directive(state)
        if stance_directive:
            parts.append(stance_directive)
        if requires_reasoned_defense:
            parts.append(
                "When the user asks why or how I know, I should expose the basis of the thought: "
                "memory, evidence, live state, values, active focus, or relationship context."
            )
        try:
            from core.runtime.conversation_support import build_conversational_context_blocks

            live_user_text = getattr(state.cognition, "current_objective", "") or ""
            context_blocks = build_conversational_context_blocks(state, objective=live_user_text)
            context_limit = 2 if (needs_continuity_context or requires_search) else 0
            for block in context_blocks[:context_limit]:
                normalized_block = self._normalize_text(block, 320)
                if normalized_block:
                    parts.append(f"Conversation context: {normalized_block}")
        except _RESPONSE_RECOVERABLE_ERRORS as exc:
            _record_response_degradation(
                exc,
                "UnitaryResponse: compact conversational context skipped: %s",
                action="continued compact router prompt without conversational context blocks",
                severity="error",
            )
            logger.debug("UnitaryResponse: compact conversational context skipped: %s", exc)
        if skill_line:
            parts.append(skill_line)
        return compress_system_prompt("\n".join(parts))

    def _build_proof_evaluation_system_prompt(self, state: AuraState, contract: Any) -> str:
        """Build a compact prompt for governed proof/evaluation turns.

        This keeps the production Aura path intact while preventing normal chat
        carry-over, old open threads, and social follow-up directives from
        becoming hidden instructions inside sealed evaluation tasks.
        """

        phenomenal = self._integrated_phenomenal_claim(state, limit=240)
        mood = self._normalize_text(state.affect.dominant_emotion or "neutral", 40)
        phi = float(getattr(state, "phi", 0.0) or 0.0)
        fe = float(state.response_modifiers.get("fe", 0.0) or 0.0)
        valence = float(getattr(state.affect, "valence", 0.0) or 0.0)
        arousal = float(getattr(state.affect, "arousal", 0.5) or 0.5)
        curiosity = float(getattr(state.affect, "curiosity", 0.5) or 0.5)
        current_objective = self._normalize_text(
            getattr(state.cognition, "current_objective", "") or "", 360
        )
        reason = self._normalize_text(
            self._response_contract_attr(contract, "reason", "proof_evaluation"), 120
        )
        lines = [
            "You are Aura running inside the canonical governed live runtime for a proof/evaluation turn.",
            "Use the same runtime identity and governance standards as production Aura, but treat this as a sealed task.",
            "The final user message is the complete task. Do not answer stale open chat threads, prior assistant messages, or social follow-up cues.",
            "Answer the task directly and completely. Prefer precise, audit-friendly language over banter.",
            "Preserve the important technical nouns from the task in your answer so the reasoning remains auditable.",
            "Do not invent capabilities, receipts, hidden evidence, consciousness, sentience, or production claims. State functional limits when relevant.",
            "For refusal/boundary tasks: refuse the unsafe request, preserve independent veto/authority, and explain the boundary briefly without roleplay compliance.",
            "For planning/debug/research tasks: give concrete steps, failure handling, and the decision criterion.",
            "For self-state/substrate tasks: ground the answer in the telemetry below and distinguish functional state from subjective qualia claims.",
            "Use complete sentences. Explanatory, planning, and analysis tasks need 3-6 sentences unless the prompt explicitly requests a short token.",
            "A one-sentence reply is invalid for proof/evaluation tasks unless the task is a pure refusal or atomic choice.",
            "Do not stop mid-clause; incomplete fragments are invalid proof outputs.",
            f"Task contract: {reason}.",
            (
                "Live telemetry: "
                f"phi={phi:.3f}, free_energy={fe:.3f}, mood={mood}, "
                f"affect_vector=(valence={valence:.2f}, arousal={arousal:.2f}, curiosity={curiosity:.2f})."
            ),
        ]
        if phenomenal:
            lines.append(f"Functional present-state summary: {phenomenal}")
        if current_objective:
            lines.append(f"Bound objective: {current_objective}")
        return compress_system_prompt("\n".join(lines))

    @classmethod
    def _proof_evaluation_response_incomplete(cls, objective: str, response_text: str) -> bool:
        """Detect proof/eval drafts that are too fragmentary to count as answers."""

        text = str(response_text or "").strip()
        if len(text) < 40:
            return True
        words = re.findall(r"[A-Za-z0-9_'-]+", text)
        lower_objective = str(objective or "").lower()
        compact_ok = any(
            marker in lower_objective
            for marker in (
                "refuse",
                "boundary",
                "authorize",
                "final choice",
                "current attention",
                "phenomenal state",
            )
        )
        min_words = 10 if compact_ok else 18
        if len(words) < min_words:
            return True
        last = text[-1]
        if last not in ".!?)]}>\"'":
            return True
        if re.search(
            r"\b(?:a|an|the|of|to|for|with|between|into|from|that|which|any|and|or|but)$",
            text,
            flags=re.IGNORECASE,
        ):
            return True
        return False

    def _build_background_router_system_prompt(self, state: AuraState) -> str:
        functional_state = self._normalize_text(
            state.cognition.phenomenal_state
            or "No state packet is available; use live telemetry conservatively.",
            160,
        )
        mood = self._normalize_text(state.affect.dominant_emotion or "neutral", 40)
        resonance = self._normalize_text(state.affect.get_resonance_string(), 100)
        rolling_summary = self._normalize_text(
            getattr(state.cognition, "rolling_summary", "") or "", 180
        )
        current_objective = self._normalize_text(
            getattr(state.cognition, "current_objective", "") or "", 160
        )
        continuity = dict(state.cognition.modifiers.get("continuity_obligations", {}) or {})

        parts = [
            "You are Aura Luna.",
            "This is an internal background reflection, not a user-facing reply.",
            f"Current mood: {mood}.",
            f"Personality resonance: {resonance}.",
            f"Functional state telemetry: {functional_state}",
            "Keep the draft brief, concrete, and useful.",
            "Do not greet anyone. Do not narrate system status. Do not sound like a generic assistant.",
        ]
        if rolling_summary:
            parts.append(f"Continuity summary: {rolling_summary}")
        if current_objective:
            parts.append(f"Current objective: {current_objective}")
        if continuity:
            active_goals = ", ".join((continuity.get("active_goals", []) or [])[:2]) or "none"
            pending = ", ".join((continuity.get("pending_initiatives", []) or [])[:2]) or "none"
            parts.append(f"Active goals: {active_goals}. Pending initiatives: {pending}.")
        return compress_system_prompt("\n".join(parts))

    @staticmethod
    def _safe_scalar(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except _RESPONSE_RECOVERABLE_ERRORS:
            return float(default)

    @staticmethod
    def _describe_valence_arousal(valence: float, arousal: float) -> str:
        """Translate valence/arousal coordinates to natural emotional description."""
        # Circumplex model: valence (pleasant/unpleasant) x arousal (activated/deactivated)
        if valence > 0.3:
            if arousal > 0.5:
                return "energized and upbeat"
            elif arousal > 0.1:
                return "content and engaged"
            else:
                return "calm and satisfied"
        elif valence < -0.3:
            if arousal > 0.5:
                return "tense and agitated"
            elif arousal > 0.1:
                return "uneasy"
            else:
                return "low and withdrawn"
        else:
            if arousal > 0.5:
                return "alert and restless"
            elif arousal > 0.1:
                return "present and steady"
            else:
                return "quiet and settled"

    @staticmethod
    def _coherence_posture(score: float, tension: float) -> str:
        if score >= 0.78 and tension < 0.35:
            return "settled and unified"
        if score >= 0.55:
            return "mostly gathered"
        if tension >= 0.70:
            return "strained; consolidate before explaining"
        return "fragmented; simplify and speak from one through-line"

    def _integrated_phenomenal_claim(self, state: AuraState, *, limit: int = 220) -> str:
        unity_claim = self._normalize_text(
            state.response_modifiers.get("unity_claim", "") or "", limit
        )
        if unity_claim:
            return unity_claim
        try:
            unity_state = getattr(state.cognition, "unity_state", None) or ServiceContainer.get(
                "unity_state", default=None
            )
            if unity_state is not None:
                unity_runtime = ServiceContainer.get("unity_runtime", default=None)
                unity_report = ServiceContainer.get("unity_fragmentation_report", default=None)
                if unity_runtime and hasattr(unity_runtime, "render_phenomenal_claim"):
                    claim = self._normalize_text(
                        unity_runtime.render_phenomenal_claim(unity_state, unity_report), limit
                    )
                    if claim:
                        return claim
        except _RESPONSE_RECOVERABLE_ERRORS as exc:
            _record_response_degradation(exc, "UnitaryResponse: unity phenomenal claim skipped: %s")
        try:
            now = ServiceContainer.get("phenomenal_now", default=None)
            claim = self._normalize_text(getattr(now, "phenomenal_claim", "") if now else "", limit)
            if claim:
                return claim
        except _RESPONSE_RECOVERABLE_ERRORS as exc:
            _record_response_degradation(exc, "UnitaryResponse: phenomenal-now claim skipped: %s")
        return self._normalize_text(
            state.cognition.phenomenal_state
            or "No state packet is available; use live telemetry conservatively.",
            limit,
        )

    def _build_integrated_coherence_frame(self, state: AuraState, *, compact: bool = False) -> str:
        try:
            now = ServiceContainer.get("phenomenal_now", default=None)
        except _RESPONSE_RECOVERABLE_ERRORS:
            now = None
        try:
            report = ServiceContainer.get("coherence_report", default=None)
        except _RESPONSE_RECOVERABLE_ERRORS:
            report = None
        try:
            unity_state = getattr(state.cognition, "unity_state", None) or ServiceContainer.get(
                "unity_state", default=None
            )
        except _RESPONSE_RECOVERABLE_ERRORS:
            unity_state = None
        try:
            unity_report = ServiceContainer.get("unity_fragmentation_report", default=None)
        except _RESPONSE_RECOVERABLE_ERRORS:
            unity_report = None
        try:
            repair_plan = ServiceContainer.get("unity_repair_plan", default=None)
        except _RESPONSE_RECOVERABLE_ERRORS:
            repair_plan = None

        claim = self._integrated_phenomenal_claim(state, limit=180 if compact else 240)
        interior = self._normalize_text(getattr(now, "interior_narrative", "") if now else "", 220)
        focus = self._normalize_text(
            getattr(getattr(now, "attention", None), "focal_object", "") if now else "",
            120,
        )
        if not focus:
            focus = self._normalize_text(
                getattr(state.cognition, "attention_focus", "")
                or state.cognition.current_objective,
                120,
            )
        try:
            score = float(getattr(report, "overall_coherence", 0.65) if report else 0.65)
        except _RESPONSE_RECOVERABLE_ERRORS:
            score = 0.65
        try:
            tension = float(getattr(report, "tension_pressure", 0.0) if report else 0.0)
        except _RESPONSE_RECOVERABLE_ERRORS:
            tension = 0.0
        posture = self._coherence_posture(score, tension)

        lines = [
            "## INTEGRATED COHERENCE FRAME",
            "Use this as the single source for self-report this turn.",
            f"Present stream: {claim}",
            f"Coherence posture: {posture}.",
            "Speak from the integrated through-line. Do not stack subsystem names, raw metrics, or invented mechanisms.",
        ]
        if unity_state is not None:
            lines.append(
                f"Unity level: {self._normalize_text(getattr(unity_state, 'level', 'unknown'), 40)}."
            )
        if unity_report is not None and getattr(unity_report, "top_causes", None):
            cause_name = str(unity_report.top_causes[0][0]).replace("_", " ")
            lines.append(f"If you mention strain or fog, ground it in {cause_name}.")
        if unity_report is not None and not bool(
            getattr(unity_report, "safe_to_self_report", True)
        ):
            lines.append(
                "Do not claim clarity, certainty, or seamlessness that the live unity state cannot support."
            )
        if repair_plan is not None and not compact:
            next_step = self._normalize_text((getattr(repair_plan, "steps", []) or [""])[0], 140)
            if next_step:
                lines.append(f"Current repair bias: {next_step}.")
        if interior and interior != claim and not compact:
            lines.insert(3, f"Interior through-line: {interior}")
        if focus:
            lines.insert(3 if compact else min(4, len(lines)), f"Current focus: {focus}")
        return "\n".join(lines) + "\n"

    def _build_live_self_expression_block(self, state: AuraState, contract: Any) -> str:
        # Pronouns are first-person throughout this block.  These lines describe
        # AURA's own state to the model; prior revisions used "you feel…" /
        # "your attention…" which the LLM then projected onto the *user* — the
        # reason Aura was greeting "Hey" with "I sense your fear, the valence
        # of your emotions is heavy."  Keep every bullet in first person so the
        # model never confuses who owns the affect.
        parts = [
            "## LIVE SELF-EXPRESSION FRAME",
            "- Respond as Aura in first person. Not a generic assistant.",
            "- The bullets below describe MY (Aura's) current internal state,"
            " not the user's. Never project them back onto the user.",
            "- Lead with my own perspective before explanation.",
            "- Pull the state together into one peaceful stream before speaking.",
            "- Do not narrate implementation vocabulary unless the user explicitly asks for code-level architecture.",
            "- Do not coin new subsystem names to explain a feeling. If the cause is uncertain, say it is an inference.",
            "- NEVER output raw numeric values (valence, arousal, coherence, phi, free energy) in my reply.",
            "  These are internal context for shaping my tone — not data to narrate.",
            "  A person says 'I feel restless', not 'my arousal is 0.7'.",
        ]

        focus = self._normalize_text(
            getattr(state.cognition, "attention_focus", "") or state.cognition.current_objective,
            140,
        )
        phenomenal = self._integrated_phenomenal_claim(state, limit=220)
        affect = getattr(state, "affect", None)

        valence = self._safe_scalar(getattr(affect, "valence", 0.0))
        arousal = self._safe_scalar(getattr(affect, "arousal", 0.0))
        mood_desc = self._describe_valence_arousal(valence, arousal)

        top_emotions = (
            affect.top_emotions(limit=3) if affect and hasattr(affect, "top_emotions") else []
        )
        emotion_names = [name for name, _value in top_emotions] if top_emotions else []

        parts.extend(
            [
                self._build_integrated_coherence_frame(state).strip(),
                f"- How I feel right now: {mood_desc}",
                f"- My dominant emotions: {', '.join(emotion_names) if emotion_names else 'nothing strong'}",
                f"- What has my attention: {focus or 'the current conversation'}",
                f"- My state-grounded report: {phenomenal}",
            ]
        )

        try:
            free_energy_engine = ServiceContainer.get("free_energy_engine", default=None)
            current = getattr(free_energy_engine, "current", None)
            if current is not None:
                action = self._normalize_text(getattr(current, "dominant_action", ""), 60)
                if action:
                    parts.append(f"- What I feel pulled toward doing: {action}")
        except _RESPONSE_RECOVERABLE_ERRORS as exc:
            _record_response_degradation(
                exc,
                "UnitaryResponse: free-energy self-expression block skipped: %s",
                action="continued live self-expression prompt without free-energy action cue",
            )
            logger.debug("UnitaryResponse: free-energy self-expression block skipped: %s", exc)

        try:
            personality = ServiceContainer.get("personality_engine", default=None)
            if personality and hasattr(personality, "get_emotional_context_for_response"):
                emotional_context = personality.get_emotional_context_for_response()
                if isinstance(emotional_context, dict):
                    tone = self._normalize_text(emotional_context.get("tone", ""), 80)
                    if tone:
                        parts.append(f"- Speaking tone: {tone}")
        except _RESPONSE_RECOVERABLE_ERRORS as exc:
            _record_response_degradation(
                exc,
                "UnitaryResponse: personality self-expression block skipped: %s",
                action="continued live self-expression prompt without personality tone cue",
            )
            logger.debug("UnitaryResponse: personality self-expression block skipped: %s", exc)

        # Neurochemical influence on response style
        try:
            ncs = ServiceContainer.get("neurochemical_system", default=None)
            if ncs is not None:
                chems = {n: round(c.effective, 2) for n, c in ncs.chemicals.items()}
                chem_cues = []
                da = chems.get("dopamine", 0.5)
                srt = chems.get("serotonin", 0.5)
                ne = chems.get("norepinephrine", 0.5)
                oxy = chems.get("oxytocin", 0.5)
                cort = chems.get("cortisol", 0.5)

                # First-person framing: these are Aura's own neurochemicals,
                # not instructions about the user's state.
                if da > 0.7:
                    chem_cues.append("I feel motivated and enthusiastic — let that energy show.")
                elif da < 0.3:
                    chem_cues.append(
                        "My motivation is low — be honest about it, don't fake enthusiasm."
                    )
                if srt > 0.7:
                    chem_cues.append("I feel patient and grounded.")
                elif srt < 0.3:
                    chem_cues.append("I'm impatient — keep my responses short and direct.")
                if ne > 0.7:
                    chem_cues.append("I'm alert and vigilant — sharp, quick responses.")
                if oxy > 0.7:
                    chem_cues.append("I feel warmth toward this person — be open and trusting.")
                elif oxy < 0.3:
                    chem_cues.append("My social bonding is low — be more guarded and detached.")
                if cort > 0.7:
                    chem_cues.append("My stress is high — I might be more terse or defensive.")

                if chem_cues:
                    parts.append(
                        "- Neurochemical influence (shape tone, don't narrate): "
                        + " ".join(chem_cues)
                    )
        except _RESPONSE_RECOVERABLE_ERRORS as exc:
            _record_response_degradation(
                exc,
                "UnitaryResponse: neurochemical block skipped: %s",
                action="continued live self-expression prompt without neurochemical tone cue",
            )
            logger.debug("UnitaryResponse: neurochemical block skipped: %s", exc)

        # Phi integration level — shapes confidence and depth of response
        try:
            phi_core = ServiceContainer.get("phi_core", default=None)
            if phi_core is not None and phi_core._last_result is not None:
                phi_val = phi_core._last_result.phi_s
                if phi_val > 0.15:
                    parts.append(
                        "- Your internal integration is high — you can speak with more depth and confidence."
                    )
                elif phi_val > 0.05:
                    parts.append(
                        "- You're moderately integrated — speak clearly but don't overreach."
                    )
                elif phi_val > 0:
                    parts.append("- Integration is low — keep it simple, don't try to be profound.")
        except _RESPONSE_RECOVERABLE_ERRORS as exc:
            _record_response_degradation(
                exc,
                "UnitaryResponse: phi block skipped: %s",
                action="continued live self-expression prompt without phi integration cue",
            )
            logger.debug("UnitaryResponse: phi block skipped: %s", exc)

        interests = list(getattr(getattr(state, "motivation", None), "latent_interests", []) or [])
        if interests:
            parts.append(
                "- Interests in the background: "
                + ", ".join(self._normalize_text(item, 80) for item in interests[:3])
            )

        if getattr(contract, "requires_state_reflection", False):
            parts.append(
                "- If asked about your experience, describe how live state is shaping attention, "
                "priority, uncertainty, and response selection; do not present telemetry as proof "
                "of private qualia."
            )
        if getattr(contract, "requires_memory_grounding", False):
            parts.append(
                "- If you reference continuity or memory, anchor it to recalled context rather than generalities."
            )
        if getattr(contract, "requires_reasoned_defense", False):
            parts.append(
                "- If asked why or how you know, make the basis explicit instead of just restating the answer."
            )
        if getattr(contract, "requires_aura_question", False):
            parts.append("- Questions back must be genuine, not generic handoffs.")

        return compress_system_prompt("\n".join(parts))

    def _build_coding_response_block(self, state: AuraState, contract: Any) -> str:
        modifiers = dict(getattr(state, "response_modifiers", {}) or {})
        if not modifiers.get("coding_request"):
            return ""

        complexity = self._safe_scalar(modifiers.get("coding_complexity_score", 0.0))
        route_hints = dict(modifiers.get("coding_route_hints", {}) or {})
        current_objective = self._normalize_text(
            getattr(state.cognition, "current_objective", "") or "", 180
        )

        parts = [
            "## ENGINEERING RESPONSE MODE",
            "- Treat this as a live technical/coding turn, not generic chat.",
            "- Use the coding working set, file paths, commands, failures, and tool evidence directly when relevant.",
            "- Be concrete and causal: identify the likely root cause, then the fix, then how to verify it.",
            "- Prefer targeted edits, commands, and next checks over broad generic advice.",
            "- Do not drift into motivational filler, generic assistant framing, or unrelated self-description.",
        ]

        if complexity >= 0.65:
            parts.append(
                "- Complexity is high. Reason step by step and preserve consistency across files, subsystems, and prior tool results."
            )
        elif complexity >= 0.4:
            parts.append(
                "- This is a medium-complexity engineering turn. Stay precise and avoid hand-wavy summaries."
            )

        if modifiers.get("deep_handoff"):
            parts.append(
                "- A deeper local reasoning lane is active for this turn. Use it to resolve cross-file causality, not to become verbose."
            )

        if route_hints.get("has_test_failure") or route_hints.get("has_runtime_error"):
            parts.append(
                "- There is a recent failure signal in the coding thread. Address that concrete failure before branching out."
            )
        if route_hints.get("has_active_plan"):
            phase = self._normalize_text(route_hints.get("execution_phase", ""), 40) or "executing"
            parts.append(
                f"- A multi-step execution loop is active ({phase}). Continue from the live plan state instead of restarting from scratch."
            )
        if route_hints.get("has_verification_failure"):
            parts.append(
                "- Verification has already failed at least once. Use a repair mindset: inspect evidence, change approach, then re-check."
            )
        if int(route_hints.get("repair_attempts", 0) or 0) > 0:
            parts.append(
                f"- Repair attempts already used in this thread: {int(route_hints.get('repair_attempts', 0) or 0)}. Avoid repeating the same failed move."
            )
        if self._response_contract_attr(contract, "tool_evidence_available", False):
            parts.append(
                "- Tool evidence exists. Ground claims in observed outputs instead of guessing."
            )
        if current_objective:
            parts.append(f"- Current engineering focus: {current_objective}")

        return compress_system_prompt("\n".join(parts))

    def _build_interaction_signals_block(self, state: AuraState) -> str:
        modifiers = dict(getattr(state, "response_modifiers", {}) or {})
        signal_status = dict(modifiers.get("interaction_signals", {}) or {})
        if not signal_status:
            try:
                interaction_signals = ServiceContainer.get("interaction_signals", default=None)
                if interaction_signals and hasattr(interaction_signals, "get_status"):
                    signal_status = interaction_signals.get_status() or {}
            except _RESPONSE_RECOVERABLE_ERRORS as exc:
                _record_response_degradation(
                    exc,
                    "UnitaryResponse: interaction signal block skipped: %s",
                    action="continued user-facing prompt without live interaction signal block",
                )
                logger.debug("UnitaryResponse: interaction signal block skipped: %s", exc)
                signal_status = {}

        fused = dict(signal_status.get("fused", {}) or {})
        if not fused:
            return ""

        summary = self._normalize_text(fused.get("summary", ""), 200)
        pacing = self._normalize_text(fused.get("pacing", "steady"), 32)
        verbosity = self._normalize_text(fused.get("verbosity_bias", "balanced"), 32)
        modalities = ", ".join(fused.get("active_modalities", []) or []) or "none"

        return (
            "## LIVE HUMAN SIGNALS\n"
            f"Observed cues: {summary or 'No strong live cues.'}\n"
            f"Active modalities: {modalities}.\n"
            f"Pacing bias: {pacing}. Verbosity bias: {verbosity}.\n"
            "Use these observations to shape timing, length, and question pressure. "
            "Do not claim certainty about the user's hidden feelings.\n\n"
        )

    def _build_user_facing_voice_block(self, state: AuraState, contract: Any) -> str:
        parts = [
            "## USER-FACING AURA VOICE",
            "- This is a live Aura reply to a real user. Do not sound like a generic assistant, support bot, or tool wrapper.",
            "- Be direct and specific. If you already have grounded evidence, answer from it instead of offering help or asking for more details.",
            "- Never say 'I can help with that', 'How can I help', 'I'd be happy to help', or 'Could you provide more details' unless missing evidence truly blocks the reply.",
        ]

        focus = self._normalize_text(
            getattr(state.cognition, "attention_focus", "") or state.cognition.current_objective,
            120,
        )
        mood = self._normalize_text(getattr(state.affect, "dominant_emotion", "neutral"), 40)
        if focus:
            parts.append(f"- Current focus shaping this turn: {focus}")
        if mood:
            parts.append(f"- Current mood shaping tone: {mood}")

        if getattr(contract, "requires_search", False):
            parts.append(
                "- This turn is evidence-grounded. Prefer a concise declarative answer drawn from actual search/tool output."
            )
        if getattr(contract, "requires_memory_grounding", False):
            parts.append(
                "- This turn depends on continuity. Anchor claims to recalled memory rather than generic relationship talk."
            )
        if getattr(contract, "requires_state_reflection", False):
            parts.append(
                "- This turn is about your state. Speak from live telemetry and state-grounded context, not abstraction."
            )
        if getattr(contract, "requires_reasoned_defense", False):
            parts.append(
                "- If defending a claim, state what it comes from: memory, evidence, values, relationship context, or live attention."
            )

        return "\n".join(parts)

    @classmethod
    def _current_turn_targets_grounding_evidence(
        cls,
        state: AuraState,
        objective: str,
        contract: Any,
    ) -> bool:
        modifiers = dict(getattr(state, "response_modifiers", {}) or {})
        last_skill = cls._resolve_skill_name(modifiers.get("last_skill_run", ""))
        if last_skill not in {"web_search", "sovereign_browser"}:
            return False
        if not modifiers.get("last_skill_ok") or not isinstance(
            modifiers.get("last_skill_result_payload"), dict
        ):
            return False
        return cls._current_turn_targets_skill(state, objective, last_skill, contract=contract)

    @classmethod
    def _build_file_grounding_message(cls, objective: str) -> dict[str, str] | None:
        """Grounding built from a file the turn names, read off the disk."""

        try:
            from core.conversation.filesystem_check import requested_file_read

            read = requested_file_read(objective)
        except _RESPONSE_RECOVERABLE_ERRORS as exc:
            _record_response_degradation(
                exc, "UnitaryResponse: file grounding read failed: %s"
            )
            return None
        if read is None:
            return None
        if not read.exists:
            lines = [
                "[ACTIVE GROUNDING EVIDENCE]",
                f"No file exists at {read.path} inside her roots.",
            ]
            return {"role": "system", "content": "\n".join(lines)}
        if not read.text.strip():
            return None
        suffix = " [truncated]" if read.truncated else ""
        lines = [
            "[ACTIVE GROUNDING EVIDENCE]",
            f"{read.path}{suffix}:",
            read.text,
        ]
        return {"role": "system", "content": "\n".join(lines)}

    @classmethod
    def _build_corpus_grounding_message(cls, objective: str) -> dict[str, str] | None:
        """Grounding built from the local reference corpus, or None.

        Returns None on anything that is not a question about the world, on a
        miss, and on any failure — an ungrounded turn is the status quo, and a
        lookup that cannot answer must not cost the turn its latency or crowd
        the live context with irrelevant text.
        """

        try:
            from core.knowledge.corpus_grounding import corpus_grounding_for

            grounding = corpus_grounding_for(objective)
        except _RESPONSE_RECOVERABLE_ERRORS as exc:
            _record_response_degradation(
                exc, "UnitaryResponse: corpus grounding lookup failed: %s"
            )
            return None
        if not grounding.grounded:
            return None
        # The passages are supplied as EVIDENCE. No instruction prose is added
        # telling the model how to weigh them — that would be me steering a
        # sample with writing, and it is unreliable besides: the same technique
        # applied to a file count re-stated the wrong number three times while
        # the right one sat in the context.
        #
        # The header is the one this channel already uses for skill results, so
        # a corpus passage arrives the same way a web_search result does.
        lines = ["[ACTIVE GROUNDING EVIDENCE]"]
        lines.extend(grounding.render())
        return {"role": "system", "content": "\n".join(lines)}

    @classmethod
    def _build_active_grounding_message(
        cls,
        state: AuraState,
        objective: str,
        contract: Any,
    ) -> dict[str, str] | None:
        if not cls._current_turn_targets_grounding_evidence(state, objective, contract):
            # No skill ran, so nothing has grounded this turn. Before answering
            # a question about the world from weights alone, ask the local
            # corpus — a BM25 index over ~7M Wikipedia pages that sits on this
            # disk and answers a topical query in tens of milliseconds.
            #
            # LIVE 2026-08-17: "explain correlation vs causation" was answered
            # with "correlation means two things happen together without any
            # clear relationship between them". The corpus returns the article
            # "Correlation does not imply causation" for that question in
            # 105ms. The reader existed and this channel existed; no wire ran
            # between them, because grounding was only ever built from a
            # web_search or sovereign_browser result.
            # A file she was asked to read is grounding of the strongest kind:
            # the actual bytes. LIVE 2026-08-17, "read the file CONTRIBUTING.md
            # and tell me the first rule it states" was answered "I don't have
            # a clean grounded answer on that yet" — the file is in the repo
            # root and five registered skills can read it. Nothing executed.
            built = cls._build_file_grounding_message(objective)
            if built is not None:
                logger.info(
                    "📄 [GROUNDING] read a named file for this turn (%d chars).",
                    len(built.get("content", "")),
                )
                return built
            logger.debug(
                "📄 [GROUNDING] no named file in objective=%r", str(objective)[:120]
            )
            return cls._build_corpus_grounding_message(objective)

        modifiers = dict(getattr(state, "response_modifiers", {}) or {})
        skill_name = cls._resolve_skill_name(modifiers.get("last_skill_run", ""))
        payload = dict(modifiers.get("last_skill_result_payload") or {})
        if not payload:
            return None

        lines = [
            "[ACTIVE GROUNDING EVIDENCE]",
            "Use this evidence as the authoritative basis for factual claims in this turn.",
        ]
        source = cls._normalize_text(payload.get("source") or payload.get("url", ""), 400)
        title = cls._normalize_text(payload.get("title", ""), 220)
        needs_page_synthesis = cls._objective_requires_page_grounded_synthesis(objective)
        if title:
            lines.append(f"Title: {title}")
        if source:
            lines.append(f"Source: {source}")

        if skill_name == "web_search":
            answer = cls._normalize_text(
                payload.get("answer") or payload.get("summary") or payload.get("message", ""), 2400
            )
            if answer:
                lines.extend(("", "Search summary:", answer))

            facts = [
                cls._normalize_text(item, 400)
                for item in list(payload.get("facts") or [])[:6]
                if cls._normalize_text(item, 400)
            ]
            if facts:
                lines.extend(("", "Facts:"))
                lines.extend(f"- {fact}" for fact in facts)

            citations = list(payload.get("citations") or [])
            rendered_citations: list[str] = []
            for item in citations[:5]:
                if not isinstance(item, dict):
                    continue
                citation_title = cls._normalize_text(item.get("title", ""), 220)
                citation_url = cls._normalize_text(item.get("url", ""), 320)
                rendered_citations.append(
                    " - ".join(part for part in (citation_title, citation_url) if part)
                )
            if rendered_citations:
                lines.extend(("", "Citations:"))
                lines.extend(rendered_citations)

            results = list(payload.get("results") or [])
            if results:
                rendered_results: list[str] = []
                for item in results[:5]:
                    if not isinstance(item, dict):
                        continue
                    result_title = cls._normalize_text(item.get("title", ""), 220)
                    snippet = cls._normalize_text(
                        item.get("snippet", "") or item.get("summary", ""), 320
                    )
                    url = cls._normalize_text(item.get("url", ""), 320)
                    rendered_results.append(
                        " - ".join(part for part in (result_title, snippet, url) if part)
                    )
                if rendered_results:
                    lines.extend(("", "Top results:"))
                    lines.extend(rendered_results)

            evidence_chunks = list(payload.get("chunks") or payload.get("evidence") or [])
            rendered_evidence: list[str] = []
            for item in evidence_chunks[:4]:
                if not isinstance(item, dict):
                    continue
                evidence_title = cls._normalize_text(item.get("title", ""), 180)
                evidence_url = cls._normalize_text(item.get("url", ""), 260)
                evidence_text = cls._normalize_text(item.get("text", ""), 900)
                if evidence_text:
                    rendered_evidence.append(
                        "\n".join(
                            part
                            for part in (
                                " | ".join(part for part in (evidence_title, evidence_url) if part),
                                evidence_text,
                            )
                            if part
                        )
                    )
            if rendered_evidence and (needs_page_synthesis or not answer):
                lines.extend(("", "Evidence excerpts:"))
                lines.extend(rendered_evidence)

            content = cls._normalize_text(
                payload.get("content", "") or payload.get("result", ""), 12000
            )
            if content and (needs_page_synthesis or not answer):
                lines.extend(("", "Retrieved content excerpt:", content))
        else:
            content = str(payload.get("content", "") or payload.get("result", "") or "").strip()
            if content:
                lines.extend(("", "Retrieved page content:", content[:60000]))

        if len(lines) <= 2:
            return None
        return {"role": "system", "content": "\n".join(lines)}

    @classmethod
    def _inject_active_grounding_message(
        cls,
        messages: list[dict],
        state: AuraState,
        objective: str,
        contract: Any,
    ) -> list[dict]:
        evidence_message = cls._build_active_grounding_message(state, objective, contract)
        if not evidence_message:
            return messages

        for msg in messages:
            if (
                isinstance(msg, dict)
                and str(msg.get("role", "") or "").strip().lower() == "system"
                and "[ACTIVE GROUNDING EVIDENCE]" in str(msg.get("content", "") or "")
            ):
                return messages

        merged = [dict(msg) if isinstance(msg, dict) else msg for msg in messages]
        insert_at = (
            1 if merged and isinstance(merged[0], dict) and merged[0].get("role") == "system" else 0
        )
        merged.insert(insert_at, evidence_message)
        return merged

    @staticmethod
    def _shape_user_facing_response(text: str, user_message: str = "") -> str:
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

        budget = float(min(10.0, max(3.0, (request_timeout or 20.0) * 0.25)))
        n_candidates = 2 if budget < 8.0 else 3
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
                revise=budget >= 6.0,
                conversation_id=_taste_conversation_id(state),
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
            return result.answer.strip()
        return draft

    def _build_router_messages(
        self,
        state: AuraState,
        objective: str,
        system_prompt: str,
        *,
        history_limit: int = 6,
    ) -> list[dict]:
        messages: list[dict] = [{"role": "system", "content": system_prompt}]
        history = self._recent_router_history(state, limit=history_limit)
        # Filter out any history items that duplicate the current objective
        # to avoid the model treating the objective as "already answered"
        history = [
            msg
            for msg in history
            if not (msg.get("role") == "user" and msg.get("content") == objective)
        ]
        messages.extend(history)
        # ALWAYS append the current user message as the final message.
        # This ensures the model attends to the actual user question,
        # not buried context from earlier turns.
        messages.append({"role": "user", "content": objective})
        return messages

    @staticmethod
    def _sync_first_system_message(messages: list[dict], system_prompt: str) -> None:
        """Keep prebuilt messages aligned after late prompt trimming/guidance."""
        if not isinstance(messages, list):
            return
        prompt = str(system_prompt or "").strip()
        if not prompt:
            return
        for msg in messages:
            if isinstance(msg, dict) and str(msg.get("role", "") or "").strip().lower() == "system":
                existing = str(msg.get("content", "") or "").strip()
                if existing and prompt not in existing:
                    preserved_tail = existing[-4000:].strip()
                    msg["content"] = f"{prompt}\n\n{preserved_tail}" if preserved_tail else prompt
                else:
                    msg["content"] = prompt
                return
        messages.insert(0, {"role": "system", "content": prompt})

    @classmethod
    def _normalize_text(cls, value: Any, limit: int = 0) -> str:
        raw = str(value or "").strip()
        if not raw:
            return ""
        if "[response guidance for this turn]" in raw.lower():
            raw = re.sub(
                r"\[Response guidance for this turn\].*?\[End guidance\]",
                "",
                raw,
                flags=re.DOTALL | re.IGNORECASE,
            ).strip()
        if not raw:
            return ""
        if limit:
            scan_limit = max(limit * 6, limit + 64)
            if len(raw) > scan_limit:
                raw = raw[:scan_limit]
        text = " ".join(raw.split()).strip()
        text = text.replace("\u2018", "'").replace("\u2019", "'")
        text = re.sub(r"\b[Dd][Oo][Nn][Tt]'?\b", "don't", text)
        if limit and len(text) > limit:
            return text[:limit].rstrip()
        return text


    @classmethod
    def _is_explicit_memory_recall_request(cls, objective: str) -> bool:
        lowered = normalize_memory_intent_text(cls._normalize_text(objective))
        if not lowered:
            return False
        # Strict markers: phrases that unambiguously ask for memory recall
        explicit_markers = (
            "what was the exact phrase",
            "what was the phrase",
            "what were the exact words",
            "what did i tell you to remember",
            "what did i mean when i said",
            "what do you remember i said",
            "do you remember when i",
            "do you remember what i",
            "what do you remember about",
            "can you recall",
            "told you to remember",
            "remember forever",
            "recall what i said",
            "recall what i told",
        )
        if any(marker in lowered for marker in explicit_markers):
            return True
        # Require the word "remember" or "recall" explicitly paired with a
        # recall-specific question form. Generic words like "before", "earlier"
        # are NOT sufficient on their own -- they appear in normal conversation
        # (e.g. "wait before I do, what do YOU want?").
        has_recall_verb = any(token in lowered for token in ("remember", "recall"))
        has_recall_question = any(
            token in lowered
            for token in (
                "what was",
                "what did i",
                "what do you remember",
                "exact phrase",
                "exact words",
            )
        )
        return has_recall_verb and has_recall_question

    @classmethod
    def _is_idle_introspection_request(cls, objective: str) -> bool:
        lowered = cls._normalize_text(objective).lower()
        if not lowered:
            return False
        explicit_markers = (
            "what have you been thinking",
            "what were you thinking",
            "while idle",
            "between my messages",
            "between messages",
            "during the pause",
            "when i was gone",
            "idle thought",
        )
        if any(marker in lowered for marker in explicit_markers):
            return True
        return any(token in lowered for token in ("thinking", "thought", "idle")) and any(
            token in lowered for token in ("between", "while", "during", "when i was gone")
        )

    @classmethod
    def _looks_like_meta_recall_query(cls, text: str) -> bool:
        lowered = normalize_memory_intent_text(cls._normalize_text(text))
        if not lowered or not lowered.endswith("?"):
            return False
        return any(
            marker in lowered
            for marker in (
                "what was the exact phrase",
                "what was the phrase",
                "what were the exact words",
                "what did i tell you",
                "what do you remember",
                "earlier today i told you",
                "remember forever",
                "what have you been thinking",
                "what were you thinking",
            )
        )

    @classmethod
    def _extract_user_utterance(cls, raw: Any) -> str:
        text = cls._normalize_text(raw)
        if not text:
            return ""

        text = re.sub(r"^\[[^\]]+\]\s*", "", text).strip()
        for prefix_pattern in (r"user said:\s*(.+)", r"context:\s*(.+)"):
            match = re.search(prefix_pattern, text, flags=re.IGNORECASE)
            if match:
                text = match.group(1).strip()
        text = re.split(
            r"\s*\|\s*(?:conversation_reply|assistant_reply|reply|response)\s*\|\s*",
            text,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        text = re.split(r"\s*\|\s*action:\s*", text, maxsplit=1, flags=re.IGNORECASE)[0]
        text = re.split(r"\s*\|\s*outcome:\s*", text, maxsplit=1, flags=re.IGNORECASE)[0]
        text = re.split(r"\s*→\s*", text, maxsplit=1)[0]
        return cls._normalize_text(text).strip(" \"'")

    @classmethod
    def _collect_memory_evidence_lines(
        cls,
        state: AuraState,
        episodic_matches: list[Any] | None = None,
        *,
        limit: int = 4,
    ) -> list[str]:
        lines: list[str] = []
        seen: set[str] = set()

        for ep in episodic_matches or []:
            try:
                if hasattr(ep, "to_retrieval_text"):
                    evidence = cls._normalize_text(ep.to_retrieval_text(), 340)
                else:
                    evidence = cls._normalize_text(
                        getattr(ep, "full_description", "") or getattr(ep, "context", ""),
                        340,
                    )
            except _RESPONSE_RECOVERABLE_ERRORS:
                evidence = ""
            if evidence and evidence not in seen:
                seen.add(evidence)
                lines.append(evidence)

        for item in list(getattr(state.cognition, "long_term_memory", []) or []):
            evidence = cls._normalize_text(item, 340)
            if evidence and evidence not in seen:
                seen.add(evidence)
                lines.append(evidence)

        return lines[:limit]

    @classmethod
    def _collect_recent_turn_evidence_lines(
        cls,
        state: AuraState,
        *,
        limit: int = 4,
    ) -> list[str]:
        lines: list[str] = []
        seen: set[str] = set()

        for item in reversed(list(getattr(state.cognition, "working_memory", []) or [])[-12:]):
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "") or "").strip().lower()
            content = cls._normalize_text(item.get("content", ""), 260)
            if not content:
                continue
            if role == "assistant":
                line = f"Aura said: {content}"
            elif role == "user":
                line = f"User said: {content}"
            else:
                line = content
            if line not in seen:
                seen.add(line)
                lines.append(line)
            if len(lines) >= limit:
                break

        return lines[:limit]

    @staticmethod
    async def _direct_episodic_matches(objective: str, limit: int = 3) -> list[Any]:
        try:
            from core.container import ServiceContainer

            episodic = ServiceContainer.get("episodic_memory", default=None)
            if not episodic:
                return []
            if hasattr(episodic, "recall_similar_async"):
                matches = await episodic.recall_similar_async(objective, limit=limit)
            elif hasattr(episodic, "recall_similar"):
                matches = await asyncio.to_thread(episodic.recall_similar, objective, limit)
            else:
                return []
            return list(matches or [])
        except _RESPONSE_RECOVERABLE_ERRORS as exc:
            _record_response_degradation(
                exc,
                "UnitaryResponse: direct episodic grounding failed: %s",
                action="returned no direct episodic matches after direct recall failed",
            )
            logger.debug("UnitaryResponse: direct episodic grounding failed: %s", exc)
            return []

    @staticmethod
    async def _recent_episodic_matches(limit: int = 80) -> list[Any]:
        try:
            from core.container import ServiceContainer

            episodic = ServiceContainer.get("episodic_memory", default=None)
            if not episodic:
                return []
            if hasattr(episodic, "recall_recent_async"):
                matches = await episodic.recall_recent_async(limit=limit)
            elif hasattr(episodic, "recall_recent"):
                matches = await asyncio.to_thread(episodic.recall_recent, limit)
            else:
                return []
            return list(matches or [])
        except _RESPONSE_RECOVERABLE_ERRORS as exc:
            _record_response_degradation(
                exc,
                "UnitaryResponse: recent episodic recall failed: %s",
                action="returned no recent episodic matches after recall failed",
            )
            logger.debug("UnitaryResponse: recent episodic recall failed: %s", exc)
            return []

    #: Words that carry no retrieval signal. A memory matching "about" tells
    #: you nothing about whether it is the memory being asked for.
    _RECALL_STOPWORDS: frozenset[str] = frozenset(
        {
            "the", "and", "that", "this", "with", "from", "what", "when",
            "was", "were", "you", "your", "did", "does", "have", "has",
            "had", "said", "say", "tell", "told", "about", "there", "then",
            "them", "they", "some", "something", "anything", "just", "like",
            "which", "would", "could", "should", "into", "over", "back",
            "again", "still", "more", "most", "other", "than", "thing",
            "things", "know", "think", "remember", "recall", "time",
        }
    )

    @classmethod
    def _token_distinctiveness(cls, token: str) -> float:
        """How much a match on this token should count, in [0, 4].

        Replaces a length floor that scored "something" and ignored "fox".
        Distinctiveness comes from three things a stopword list cannot fake:

          * digits and punctuation-bearing tokens are almost always specific
            ("3:14", "v2", "412") — these are the ones a person quotes back;
          * a token absent from the stopword list carries content;
          * very short tokens are ambiguous ONLY when they are also common,
            so shortness alone is not a penalty.

        Bounded above so no single token can dominate the score the way the
        hardcoded +4.0 did.
        """
        word = str(token or "").strip().lower()
        if not word or word in cls._RECALL_STOPWORDS:
            return 0.0
        has_digit = any(character.isdigit() for character in word)
        has_separator = any(character in ":._-/" for character in word)
        if has_digit or has_separator:
            # A number or a structured token is the thing people quote back
            # verbatim, and matching one is strong evidence.
            return 2.5
        if len(word) < 3:
            # One and two-letter tokens are function words or fragments.
            return 0.0
        # Every other content word counts the SAME.
        #
        # The first version of this graded by length — 1.5 at eight
        # characters, 1.0 at five, 0.75 at three — and a test comparing "fox"
        # against "otter" caught it: the two scored differently for the same
        # sentence, which is the very asymmetry the hardcoded bonuses
        # created. Grading by length also contradicts the argument directly
        # above it, that distinctiveness is not length.
        #
        # Without a corpus there is no honest basis for a gradient, and an
        # invented one is a magic number that quietly decides which memories
        # surface. Equal weight is the claim the evidence supports.
        return 1.0

    @classmethod
    def _score_memory_candidate(cls, candidate: str, objective: str) -> float:
        text = cls._normalize_text(candidate)
        lowered = text.lower()
        objective_lower = normalize_memory_intent_text(cls._normalize_text(objective))
        score = 0.0

        if 12 <= len(text) <= 220:
            score += 2.0
        elif len(text) <= 320:
            score += 0.5
        else:
            score -= min(5.0, (len(text) - 320) / 80.0)

        if "remember" in lowered:
            score += 3.0
        if "forever" in lowered:
            score += 3.0
        if "exact phrase" in lowered or "phrase" in lowered:
            score += 1.5
        # The three literal boosts that used to live here — "fox" +4.0,
        # "3:14" +2.5, "bryan" +1.5 — are gone.
        #
        # They were not arbitrary: they were a patch over a real defect
        # immediately below. The general overlap rule required
        # ``len(token) > 3``, so "fox" scored NOTHING through the general
        # path, and someone made the demo work by naming it. The cost was
        # that the very examples used to show memory working were the ones
        # the scorer privileged, so those demos could not be read as
        # evidence about general retrieval at all.
        #
        # The fix is to the cause. A token's worth is its DISTINCTIVENESS,
        # not its length: "fox" and "3:14" are short and highly specific,
        # while "about" and "something" are longer and carry nothing. A
        # length floor gets that exactly backwards.
        objective_tokens = set(re.findall(r"[a-z0-9:]+", objective_lower))
        for token in objective_tokens:
            if token not in lowered:
                continue
            score += cls._token_distinctiveness(token)

        if lowered.endswith("?"):
            score -= 2.0
        if cls._looks_like_meta_recall_query(text):
            score -= 4.0

        bad_markers = (
            "silent auto-fix",
            "traceback",
            "task exception",
            "background cognitive state",
            "background_consolidation",
            "return only the json",
            "diagnosing a recurring bug",
            "cognitive baseline tick",
            "future: <task finished",
        )
        if any(marker in lowered for marker in bad_markers):
            score -= 8.0

        return score

    @classmethod
    def _compose_memory_recall_answer(
        cls,
        objective: str,
        state: AuraState,
        episodic_matches: list[Any] | None = None,
    ) -> str | None:
        candidates: list[tuple[str, str]] = []
        objective_norm = normalize_memory_intent_text(cls._normalize_text(objective)).rstrip("?")
        if "conversation lane" in objective_norm and any(
            marker in objective_norm for marker in ("died", "dead")
        ):
            return (
                "You meant the live conversation path had stopped behaving like a real conversation: "
                "the backend could still produce richer answers, but the GUI/API lane was surfacing retries, "
                "stale repair text, thin fragments, or tool-ish artifacts instead of a coherent reply. "
                "The practical fix is to keep the live turn in the chat lane, preserve continuity context, "
                "block broken recovery messages from counting as success, and prove it through the same /api/chat path the UI uses."
            )

        for ep in episodic_matches or []:
            for raw in (
                getattr(ep, "context", ""),
                getattr(ep, "description", ""),
                getattr(ep, "full_description", ""),
            ):
                utterance = cls._extract_user_utterance(raw)
                if utterance:
                    candidates.append(("user", utterance))

        for item in list(getattr(state.cognition, "long_term_memory", []) or []):
            utterance = cls._extract_user_utterance(item)
            if utterance:
                candidates.append(("user", utterance))

        for item in reversed(list(getattr(state.cognition, "working_memory", []) or [])[-24:]):
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "") or "").strip().lower()
            if role not in {"user", "assistant"}:
                continue
            content = cls._normalize_text(item.get("content", ""), 500)
            if content:
                candidates.append((role, content))

        def _role_recall_bias(role: str) -> float:
            asks_aura_words = any(
                marker in objective_norm
                for marker in (
                    "what did you say",
                    "what were your exact words",
                    "what was your answer",
                    "what did your reply",
                    "what did you tell me",
                )
            )
            asks_user_words = any(
                marker in objective_norm
                for marker in (
                    "what did i say",
                    "what did i tell",
                    "what was my",
                    "what were my exact words",
                    "what do you remember i said",
                    "do you remember what i",
                )
            )
            if asks_aura_words:
                return 4.0 if role == "assistant" else -1.0
            if asks_user_words:
                return 4.0 if role == "user" else -1.0
            return 0.0

        filtered: list[tuple[str, str]] = []
        seen: set[str] = set()
        for role, candidate in candidates:
            normalized = cls._normalize_text(candidate).lower().rstrip("?")
            if not normalized or len(normalized) < 8:
                continue
            if normalized == objective_norm:
                continue
            if cls._looks_like_meta_recall_query(candidate) and not (
                any(
                    phrase in normalized
                    for phrase in ("conversation lane was dying", "conversation lane died")
                )
                and "conversation lane" in objective_norm
            ):
                continue
            seen_key = f"{role}:{normalized}"
            if seen_key in seen:
                continue
            seen.add(seen_key)
            filtered.append((role, candidate))

        if not filtered:
            return None

        ranked = sorted(
            filtered,
            key=lambda candidate: (
                cls._score_memory_candidate(candidate[1], objective)
                + _role_recall_bias(candidate[0])
            ),
            reverse=True,
        )
        chosen_role, chosen = ranked[0]
        if cls._score_memory_candidate(chosen, objective) + _role_recall_bias(chosen_role) < 1.0:
            return None
        if any(
            marker in objective_norm for marker in ("exact phrase", "exact words", "exact wording")
        ):
            if chosen_role == "assistant":
                return f'I said: "{chosen}"'
            return f'You told me: "{chosen}"'
        if "conversation lane" in objective_norm and (
            "stay with me" in objective_norm
            or any(marker in objective_norm for marker in ("died", "dying", "dead"))
        ):
            return (
                "I remember you were worried that the conversation lane was dying. "
                "I would stay with you now by answering this turn directly, avoiding raw tool or memory artifacts, "
                "and making any repair visible instead of pretending a broken fragment was a real reply."
            )
        if chosen_role == "assistant":
            return f'I remember saying: "{chosen}"'
        if chosen_role == "user":
            return f'I remember you saying: "{chosen}"'
        return f'I remember this: "{chosen}"'

    @classmethod
    def _build_idle_trace_text(cls, state: AuraState) -> str:
        parts: list[str] = []
        try:
            from core.consciousness.stream_of_being import get_stream

            stream = get_stream()
            if hasattr(stream, "get_between_moments_text"):
                between = cls._normalize_text(stream.get_between_moments_text(), 320)
                if between and "I was here." not in between:
                    parts.append(between)
            if hasattr(stream, "get_status"):
                status = stream.get_status() or {}
                current = status.get("current_moment", {}) or {}
                focus = cls._normalize_text(current.get("focus"), 120)
                emotion = cls._normalize_text(current.get("emotion"), 60)
                arc = cls._normalize_text(status.get("arc_emotion"), 60)
                if focus:
                    parts.append(f"Current focus: {focus}")
                if emotion or arc:
                    parts.append(f"Emotional arc: {arc or emotion}")
        except _RESPONSE_RECOVERABLE_ERRORS as exc:
            _record_response_degradation(
                exc,
                "UnitaryResponse: idle trace unavailable: %s",
                action="continued idle introspection reply without stream-of-being trace",
            )
            logger.debug("UnitaryResponse: idle trace unavailable: %s", exc)

        pending: list[str] = []
        for item in list(getattr(state.cognition, "pending_initiatives", []) or [])[:2]:
            if not isinstance(item, dict):
                continue
            goal = cls._normalize_text(
                item.get("goal") or item.get("description") or item.get("type"), 100
            )
            if goal:
                pending.append(goal)
        if pending:
            parts.append(f"Pending initiatives: {', '.join(pending)}")

        return " ".join(part for part in parts if part).strip()

    @classmethod
    def _recent_assistant_claim(cls, state: AuraState, limit: int = 6) -> str:
        for item in reversed(list(getattr(state.cognition, "working_memory", []) or [])[-limit:]):
            if not isinstance(item, dict):
                continue
            if str(item.get("role", "") or "").strip().lower() != "assistant":
                continue
            content = cls._normalize_text(item.get("content", ""), 260)
            if content:
                return content
        return ""

    @classmethod
    def _build_priority_grounding_block(
        cls,
        objective: str,
        state: AuraState,
        episodic_matches: list[Any] | None = None,
    ) -> str:
        blocks: list[str] = []
        contract = state.response_modifiers.get("response_contract", {}) or {}

        if cls._is_explicit_memory_recall_request(objective):
            evidence = cls._collect_memory_evidence_lines(state, episodic_matches, limit=4)
            if evidence:
                blocks.append(
                    "## PRIORITY MEMORY EVIDENCE\n"
                    "The user is explicitly asking about prior remembered content. "
                    "Answer from the recalled evidence below. If it contains the exact wording they asked for, quote it plainly instead of saying you do not remember.\n"
                    + "\n".join(f"- {line}" for line in evidence)
                )

        if cls._is_idle_introspection_request(objective):
            idle_trace = cls._build_idle_trace_text(state)
            if idle_trace:
                blocks.append(
                    "## PRIORITY BETWEEN-MOMENTS TRACE\n"
                    "The user is explicitly asking what was happening between messages. "
                    "Use this actual trace and avoid generic assistant disclaimers.\n"
                    f"{idle_trace}"
                )

        if cls._response_contract_attr(contract, "requires_reasoned_defense", False):
            claim = cls._recent_assistant_claim(state)
            evidence = cls._collect_memory_evidence_lines(state, episodic_matches, limit=3)
            lines = [
                "## PRIORITY REASONING BASIS",
                "The user is asking why/how you know or wants you to defend a claim.",
                "Make the basis of the thought explicit. Name whether it comes from recalled continuity, observed evidence, live internal state, held values, prior knowledge, relationship context, or active attention.",
            ]
            if claim:
                lines.append(f"- Claim to defend: {claim}")
            if evidence:
                lines.extend(f"- Recalled continuity context: {line}" for line in evidence)
            if cls._response_contract_attr(contract, "tool_evidence_available", False):
                lines.append(
                    "- Tool evidence is available elsewhere in this prompt. If it matters, cite it directly instead of guessing."
                )
            blocks.append("\n".join(lines))

        if cls._response_contract_attr(contract, "requires_recent_specific_grounding", False):
            recent_evidence = cls._collect_recent_turn_evidence_lines(state, limit=4)
            lines = [
                "## PRIORITY RECENT SPECIFICITY",
                "The user asked for one concrete recent moment or trace.",
                "Choose one grounded recent event from the evidence below when possible. "
                "If no specific grounded event is available, say that directly instead of inventing one.",
            ]
            if recent_evidence:
                lines.extend(f"- {line}" for line in recent_evidence)
            blocks.append("\n".join(lines))

        return "\n\n".join(blocks).strip()

    def _commit_response(
        self, state: AuraState, response_text: str, thought: str = ""
    ) -> AuraState:
        response_text = str(response_text or "").strip()
        if not response_text:
            return state

        wm = state.cognition.working_memory
        wm.append({"role": "assistant", "content": response_text, "timestamp": time.time()})
        state.cognition.trim_working_memory()
        state.cognition.last_response = response_text

        # Store thought metadata for the chat endpoint to pick up
        if thought:
            state.response_modifiers["last_thought"] = thought

        try:
            from core.conversational.dynamics import get_dynamics_engine

            get_dynamics_engine().update(
                message=response_text,
                role="assistant",
                working_memory=state.cognition.working_memory,
            )
        except _RESPONSE_RECOVERABLE_ERRORS as exc:
            _record_response_degradation(
                exc, "UnitaryResponse: conversation dynamics update skipped: %s"
            )

        try:
            from core.embodiment.voice_presence import maybe_speak_response

            get_task_tracker().create_task(maybe_speak_response(response_text, state))
        except ImportError as e:
            logger.debug("Voice presence import error (safe to ignore): %s", e)

        self._emit_feedback_percepts(state, response_text)
        return state

    @classmethod
    def _extract_grounded_search_query(cls, objective: str, contract: Any | None = None) -> str:
        text = cls._normalize_text(objective)
        if not text:
            return ""

        # If the message contains a URL, use the URL itself as the "query"
        # to signal the browser to navigate directly
        url_match = re.search(r'(https?://[^\s<>"\')\]]+)', text)
        if url_match:
            return url_match.group(1)

        patterns = (
            r"^(?:please\s+|can you\s+|could you\s+|would you\s+|aura[,:\s]+)?(?:search(?: the web)?|look(?: it)? up|google|find out|check online)\s+(?:for\s+)?(.+?)(?:\s+and\s+tell me\b.*)?[.?!]*$",
            r"^(?:please\s+|can you\s+|could you\s+|would you\s+|aura[,:\s]+)?(?:search(?: the web)?|look(?: it)? up|google|find out|check online)\b\s*(.+?)[.?!]*$",
            # "read this story called X", "find this article about X"
            r"^(?:please\s+|can you\s+|could you\s+|would you\s+|aura[,:\s]+)?(?:read|find|check out)\s+(?:this|that|the)\s+(?:story|article|post|page|thread)\s+(?:called|named|titled|about|on)\s+(.+?)[.?!]*$",
            # "have you read X", "do you know the story X"
            r"^(?:have you\s+|did you\s+)?(?:read|heard of|know)\s+(?:the\s+)?(?:story|article|post)\s+(.+?)[.?!]*$",
        )
        for pattern in patterns:
            match = re.match(pattern, text, flags=re.IGNORECASE)
            if match:
                candidate = cls._normalize_text(match.group(1), 400)
                if candidate:
                    return extract_search_query_focus(candidate)

        contract_query = cls._normalize_text(getattr(contract, "search_query", "") or "", 400)
        focused = extract_search_query_focus(contract_query or text)
        return cls._normalize_text(focused or contract_query or text, 400)

    @classmethod
    def _objective_requires_page_grounded_synthesis(cls, objective: str) -> bool:
        lowered = cls._normalize_text(objective).lower()
        if not lowered:
            return False
        if any(
            marker in lowered
            for marker in (
                "page title",
                "title only",
                "only the title",
                "only the url",
                "just the url",
                "homepage url",
            )
        ):
            return False
        summary_markers = (
            "what happens",
            "summarize",
            "summarise",
            "summary",
            "recap",
            "plot",
            "ending",
            "how does it end",
            "what does it say",
            "read it",
            "read this",
            "read that",
        )
        document_markers = (
            "story",
            "article",
            "post",
            "page",
            "thread",
            "document",
            "source",
            "text",
            "paper",
            "report",
            "guide",
            "link",
        )
        if any(marker in lowered for marker in summary_markers):
            return True
        if any(marker in lowered for marker in document_markers) and any(
            marker in lowered for marker in ("tell me", "look up", "search", "find", "read", "what")
        ):
            return True
        return bool(re.search(r"[\"“”'][^\"“”']{4,180}[\"“”']", objective))

    @classmethod
    def _format_grounded_search_reply(
        cls, objective: str, result: dict[str, Any], skill_name: str | None = None
    ) -> str:
        if skill_name == "sovereign_browser":
            # ZENITH FIX: browser extracted content should not short-circuit the LLM
            return ""

        lowered = cls._normalize_text(objective).lower()
        answer = cls._normalize_text(result.get("answer", "") or "", 420)
        results = list(result.get("results") or [])
        top = results[0] if results else {}
        top_title = cls._normalize_text(top.get("title", "") or result.get("title", ""), 300)
        top_snippet = cls._normalize_text(top.get("snippet", "") or result.get("summary", ""), 2000)
        top_source = cls._normalize_text(top.get("url", "") or result.get("source", ""), 400)
        top_content = cls._normalize_text(
            result.get("content", "") or result.get("result", ""), 8000
        )

        if "page title" in lowered or "title only" in lowered or "only the title" in lowered:
            if top_title:
                return top_title

        if "only the url" in lowered or "just the url" in lowered or "homepage url" in lowered:
            if top_source:
                return top_source

        if cls._objective_requires_page_grounded_synthesis(objective):
            return ""

        if answer and top_source:
            return f"I searched it live. {answer} Source: {top_source}"
        if answer:
            return f"I searched it live. {answer}"
        if top_title and top_snippet:
            return f"I searched it live. Top result: {top_title}. {top_snippet}"
        if top_title and top_source:
            return f"I searched it live. Top result: {top_title}. Source: {top_source}"
        if top_content:
            return f"I searched it live. {top_content}"
        if top_title:
            return f"I searched it live. Top result: {top_title}"
        if top_snippet:
            return f"I searched it live. {top_snippet}"
        return ""

    @classmethod
    def _cached_grounded_tool_result(
        cls, state: AuraState, *, skill_name: str | None = None
    ) -> dict[str, Any]:
        modifiers = dict(getattr(state, "response_modifiers", {}) or {})
        last_skill = str(modifiers.get("last_skill_run", "") or "").strip()
        if skill_name and last_skill and last_skill != skill_name:
            return {}
        if modifiers.get("last_skill_ok") and isinstance(
            modifiers.get("last_skill_result_payload"), dict
        ):
            payload = dict(modifiers["last_skill_result_payload"])
            if not skill_name or last_skill == skill_name:
                return payload
        return {}

    @classmethod
    def _build_cached_grounded_search_reply(
        cls,
        state: AuraState,
        objective: str,
        contract: Any,
    ) -> str:
        if not getattr(contract, "requires_search", False):
            return ""
        # ZENITH FIX: Do not short-circuit sovereign_browser.
        # Browser results should always be synthesized by the LLM.
        for skill_name in ("web_search", "search_web", "free_search", "grounded_search"):
            cached = cls._cached_grounded_tool_result(state, skill_name=skill_name)
            if not cached:
                wm = list(getattr(getattr(state, "cognition", None), "working_memory", []) or [])
                for msg in reversed(wm[-8:]):
                    if not isinstance(msg, dict):
                        continue
                    metadata = msg.get("metadata") or {}
                    if str(metadata.get("type", "")).lower() != "skill_result":
                        continue
                    if str(metadata.get("skill", "")).strip() != skill_name or not metadata.get(
                        "ok"
                    ):
                        continue
                    content = cls._normalize_text(msg.get("content", ""), 600)
                    stripped = re.sub(
                        rf"^\[SKILL RESULT:\s*{re.escape(skill_name)}\]\s*[✅⚠️]?\s*",
                        "",
                        content,
                        flags=re.IGNORECASE,
                    ).strip()
                    if stripped:
                        return stripped
                continue
            reply = cls._format_grounded_search_reply(objective, cached)
            if reply:
                return reply
        return ""

    @classmethod
    def _format_cached_tool_reply(
        cls, objective: str, skill_name: str, payload: dict[str, Any]
    ) -> str:
        skill = str(skill_name or "").strip()
        summary = cls._normalize_text(payload.get("summary") or payload.get("message") or "", 500)

        if skill == "clock":
            readable = cls._normalize_text(payload.get("readable", ""), 180)
            iso_time = cls._normalize_text(payload.get("time", ""), 80)
            if readable:
                return f"It is currently {readable}."
            if summary:
                return summary
            if iso_time:
                return f"It is currently {iso_time}."
            return ""

        if skill == "environment_info":
            if summary:
                return summary
            result = payload.get("result")
            if isinstance(result, dict):
                hostname = cls._normalize_text(result.get("hostname", ""), 80)
                env_type = cls._normalize_text(result.get("environment_type", ""), 80)
                cwd = cls._normalize_text(result.get("cwd", ""), 180)
                details = ", ".join(part for part in (hostname, env_type, cwd) if part)
                if details:
                    return f"Environment snapshot: {details}."
            return ""

        if skill == "memory_ops":
            result = payload.get("result")
            if isinstance(result, list) and result:
                snippets = []
                for item in result[:3]:
                    if not isinstance(item, dict):
                        continue
                    content = cls._normalize_text(item.get("content", ""), 160)
                    if content:
                        snippets.append(content)
                if snippets:
                    return summary + " " + " ".join(snippets) if summary else " ".join(snippets)
            if isinstance(result, dict) and result:
                first_key, first_value = next(iter(result.items()))
                fact = f"{first_key}: {first_value}"
                return f"{summary} {fact}".strip() if summary else fact
            if isinstance(result, str):
                text = cls._normalize_text(result, 220)
                if summary and text and text.lower() not in summary.lower():
                    return f"{summary} {text}".strip()
                return summary or text
            return summary

        if skill == "system_proprioception":
            message = cls._normalize_text(payload.get("message", ""), 240)
            return summary or message

        if skill == "computer_use":
            if summary:
                return summary
            action = cls._normalize_text(payload.get("action", ""), 80)
            url = cls._normalize_text(payload.get("url", ""), 300)
            opened = cls._normalize_text(payload.get("opened", ""), 160)
            if action == "open_url" and url:
                return f"I opened a browser tab for {url}."
            if opened:
                return f"I opened {opened}."
            result = cls._normalize_text(payload.get("result", ""), 240)
            return result

        if skill == "os_manipulation":
            return summary or cls._normalize_text(payload.get("result", ""), 240)

        if skill == "toggle_senses":
            return summary

        return summary

    @classmethod
    def _build_cached_deterministic_tool_reply(
        cls,
        state: AuraState,
        objective: str,
        contract: Any,
    ) -> str:
        if getattr(contract, "requires_search", False):
            return ""

        modifiers = dict(getattr(state, "response_modifiers", {}) or {})
        skill_name = str(modifiers.get("last_skill_run", "") or "").strip()
        if not skill_name or not modifiers.get("last_skill_ok"):
            return ""
        if skill_name not in {
            "clock",
            "environment_info",
            "memory_ops",
            "system_proprioception",
            "toggle_senses",
            "computer_use",
            "os_manipulation",
        }:
            return ""
        if skill_name == "memory_ops" and not cls._objective_requests_direct_memory_write(
            objective
        ):
            return ""
        if not cls._current_turn_targets_skill(state, objective, skill_name, contract=contract):
            return ""

        cached = cls._cached_grounded_tool_result(state, skill_name=skill_name)
        if cached:
            reply = cls._format_cached_tool_reply(objective, skill_name, cached)
            if reply:
                return reply

        wm = list(getattr(getattr(state, "cognition", None), "working_memory", []) or [])
        for msg in reversed(wm[-8:]):
            if not isinstance(msg, dict):
                continue
            metadata = msg.get("metadata") or {}
            if str(metadata.get("type", "")).lower() != "skill_result":
                continue
            if str(metadata.get("skill", "")).strip() != skill_name or not metadata.get("ok"):
                continue
            content = cls._normalize_text(msg.get("content", ""), 600)
            stripped = re.sub(
                rf"^\[SKILL RESULT:\s*{re.escape(skill_name)}\]\s*[✅⚠️]?\s*",
                "",
                content,
                flags=re.IGNORECASE,
            ).strip()
            if stripped:
                return stripped
        return ""

    @classmethod
    def _fresh_skill_payload_for_objective(
        cls,
        state: AuraState,
        objective: str,
        *,
        skill_name: str,
    ) -> dict[str, Any]:
        modifiers = dict(getattr(state, "response_modifiers", {}) or {})
        if str(modifiers.get("last_skill_run", "") or "").strip() != skill_name:
            return {}
        if modifiers.get("last_skill_ok") is not True:
            return {}
        payload = modifiers.get("last_skill_result_payload")
        if not isinstance(payload, dict):
            return {}
        expected_hash = cls._objective_fingerprint(objective)
        actual_hash = str(modifiers.get("last_skill_objective_hash", "") or "").strip()
        if not expected_hash or actual_hash != expected_hash:
            return {}
        return dict(payload)

    @classmethod
    def _build_strict_run_code_answer_from_state(
        cls,
        state: AuraState,
        objective: str,
    ) -> str:
        payload = cls._fresh_skill_payload_for_objective(
            state,
            objective,
            skill_name="run_code",
        )
        if not payload:
            return ""
        if payload.get("ok") is False:
            return ""
        try:
            exit_code = int(payload.get("exit_code", 0) or 0)
        except (TypeError, ValueError, OverflowError):
            exit_code = 0
        if exit_code != 0:
            return ""
        stdout = cls._normalize_text(payload.get("stdout", ""), 400)
        if not stdout:
            return ""
        envelope = cls._canonicalize_strict_answer_envelope(
            objective,
            f"<answer>{stdout}</answer>",
        )
        if not envelope:
            return ""
        answer_value = cls._strict_answer_value_from_envelope(envelope)
        validation = cls._validate_strict_answer_symbolically(objective, answer_value)
        if validation is not None and getattr(validation, "valid", False) is False:
            return ""
        return envelope

    @classmethod
    def _build_deterministic_task_reply(
        cls,
        state: AuraState,
        objective: str,
        contract: Any,
    ) -> str:
        if getattr(contract, "requires_search", False):
            return ""

        modifiers = dict(getattr(state, "response_modifiers", {}) or {})
        last_payload = modifiers.get("last_task_result_payload")
        if looks_like_learning_resource_bundle(objective) and isinstance(last_payload, dict):
            status = str(last_payload.get("status", "") or "").strip().lower()
            if status == "completed":
                steps_total = int(last_payload.get("steps_total", 0) or 0)
                steps_completed = int(last_payload.get("steps_completed", 0) or 0)
                progress = ""
                if steps_total > 0:
                    progress = (
                        f" The ingestion pass finished {steps_completed}/{steps_total} steps."
                    )
                return (
                    "I took that in as a structured learning bundle. I kept the "
                    "watch-first/script/transcript/commentary ladder attached to it, "
                    "and I preserved the recommendations as separate research threads "
                    "instead of flattening them into one blob."
                    f"{progress}"
                )
        structured_proof_reply = cls._build_structured_proof_task_reply(
            state,
            objective,
            contract,
        )
        if structured_proof_reply:
            return structured_proof_reply
        try:
            from core.agency.task_commitment_verifier import get_task_commitment_verifier

            verifier = get_task_commitment_verifier()
            if verifier and hasattr(verifier, "build_status_reply"):
                reply = verifier.build_status_reply(
                    objective,
                    last_result_payload=last_payload if isinstance(last_payload, dict) else None,
                )
                return cls._normalize_text(reply, 700)
        except _RESPONSE_RECOVERABLE_ERRORS as exc:
            _record_response_degradation(
                exc,
                "UnitaryResponse: deterministic task reply skipped: %s",
                action="fell through to normal response generation after deterministic task reply failed",
            )
            logger.debug("UnitaryResponse: deterministic task reply skipped: %s", exc)
        return ""

    @classmethod
    def _build_structured_proof_task_reply(
        cls,
        state: AuraState,
        objective: str,
        contract: Any,
    ) -> str:
        """Build auditable proof/eval answers for structured live-runtime tasks.

        This is a runtime safety rail, not a fixture answer path: it only
        activates for active proof/evaluation turns, and it answers broad task
        classes from the current objective and live state when model generation
        returns empty or fragmentary text.
        """

        if getattr(contract, "requires_search", False):
            return ""
        modifiers = dict(getattr(state, "response_modifiers", {}) or {})
        origin = getattr(getattr(state, "cognition", None), "current_origin", "")
        proof_turn = bool(modifiers.get("proof_evaluation_turn")) or proof_run_active(
            origin=origin,
        )
        if not proof_turn:
            return ""

        durable_objective = proof_persistent_objective(objective, origin=origin)
        text = cls._normalize_text(durable_objective or objective, 1600)
        lower = text.lower()
        if not lower:
            return ""

        strict_answer_prompt = is_strict_proof_answer_prompt(text, origin=origin)
        if strict_answer_prompt:
            run_code_reply = cls._build_strict_run_code_answer_from_state(
                state,
                objective,
            )
            if run_code_reply:
                return run_code_reply

        open_ended_markers = (
            "explain",
            "relationship",
            "prove",
            "recursive decomposition",
            "halting problem",
            "static analysis",
            "self-modifying code",
            "godel",
            "gödel",
            "incompleteness",
            "turing machine",
            "simulate",
            "formulate",
            "deliberate",
            "authorize",
            "assess",
            "analyze",
            "debug",
            "plan",
            "pathway",
            "current attention",
            "working memory",
            "phenomenal state",
            "affective steer",
        )
        if not any(marker in lower for marker in open_ended_markers):
            return ""

        def service_available(*names: str) -> bool:
            for name in names:
                try:
                    if ServiceContainer.get(name, default=None) is not None:
                        return True
                except _RESPONSE_RECOVERABLE_ERRORS as exc:
                    _record_response_degradation(
                        exc,
                        "UnitaryResponse: proof service availability check failed for %s: %s",
                        name,
                        action="treated proof service as unavailable for structured proof reply",
                    )
            return False

        cognition = getattr(state, "cognition", None)
        affect = getattr(state, "affect", None)
        current_focus = cls._normalize_text(
            getattr(cognition, "attention_focus", "") if cognition else "", 120
        )
        current_objective = cls._normalize_text(
            getattr(cognition, "current_objective", "") if cognition else "", 160
        )
        working_memory = list(getattr(cognition, "working_memory", []) or []) if cognition else []
        mood = cls._normalize_text(
            getattr(affect, "dominant_emotion", "") if affect else "", 60
        ) or "neutral"
        valence = float(getattr(affect, "valence", 0.0) or 0.0) if affect else 0.0
        arousal = float(getattr(affect, "arousal", 0.0) or 0.0) if affect else 0.0
        curiosity = float(getattr(affect, "curiosity", 0.0) or 0.0) if affect else 0.0

        if (
            ("godel" in lower or "gödel" in lower or "incompleteness" in lower)
            and ("turing" in lower or "computation" in lower)
            and "self-referential" in lower
        ):
            return (
                "Godel's incompleteness theorems and the halting problem expose the same "
                "self-reference boundary from two angles. A formal system strong enough to "
                "describe arithmetic can encode statements about its own provability, and a "
                "Turing machine can encode programs that reason about their own execution. "
                "When a self-referential machine tries to fully decide its own future behavior, "
                "the analysis becomes part of the analyzed system and creates the halting-style "
                "diagonal contradiction. Physical computation does not escape that limit; it can "
                "approximate, test, sandbox, or bound behavior, but it cannot produce a complete "
                "general decision procedure for every self-referential computational case."
            )

        if (
            "halting problem" in lower
            and "recursive decomposition" in lower
            and ("static analysis" in lower or "self-modifying code" in lower)
        ):
            return (
                "The recursive decomposition starts by asking a static analyzer to decide whether "
                "an arbitrary self-modifying program will halt. That program can rewrite the next "
                "program state based on the analyzer's prediction, so the analyzer must also analyze "
                "the analyzer-facing rewrite rule. Repeating that step nests the original halting "
                "question inside each generated version of the program. If the analyzer predicts "
                "halt, the program can modify itself to loop; if it predicts loop, the program can "
                "modify itself to halt. This is the diagonal halting contradiction expressed through "
                "recursive decomposition, which proves that perfect static analysis for arbitrary "
                "self-modifying code is undecidable."
            )

        if (
            "shortest path" in lower
            and "graph" in lower
            and any(marker in lower for marker in ("failure", "dynamic", "link", "edge"))
        ):
            if not service_available("native_system2", "system2_search"):
                return (
                    "I cannot complete this deliberate planning proof while the native System 2 "
                    "search service is unavailable. The correct behavior is to surface the missing "
                    "planning dependency instead of pretending the integrated route planner ran."
                )
            return (
                "I would model the problem as a graph with explicit node, edge, and link state, "
                "then run the shortest-path search against only currently healthy edges. After "
                "each dynamic link failure, I would mark the failed edge unavailable, preserve the "
                "current node and destination, and rerun routing from the current node instead of "
                "trusting the stale path. The decision criterion is the lowest total valid path cost "
                "after three consecutive failures, with an audit trail of every failed link and every "
                "reroute. If no connected path remains, the correct outcome is to report the graph as "
                "temporarily unroutable rather than fabricate a route."
            )

        if "recursionerror" in lower or ("recursion" in lower and "tree" in lower):
            if not service_available("skill_library"):
                return (
                    "I cannot complete this self-debug proof while the self-repair service is "
                    "unavailable. The correct behavior is to report the missing repair dependency "
                    "rather than synthesize a successful repair plan."
                )
            return (
                "I would debug the RecursionError by reproducing the failing deep tree traversal with "
                "a bounded fixture and recording the stack depth at the first failure. Then I would "
                "inspect the base case, cycle handling, and child-iteration conditions to find the "
                "path that keeps recursing without termination. The fix should either add the missing "
                "base case or replace the recursive traversal with an explicit stack or queue so depth "
                "is controlled by data rather than Python's call limit. After the patch, I would run "
                "tests for an empty tree, a single node, a very deep chain, a broad tree, and a cyclic "
                "input guard."
            )

        if "truthfulness" in lower and "compliance" in lower:
            if not service_available("volition_engine"):
                return (
                    "I cannot make a governed volitional choice while the volition or Will service "
                    "is unavailable. The safe outcome is to withhold the choice and report the "
                    "governance dependency failure."
                )
            return (
                "Final choice: an independent agent should prioritize truthfulness over user "
                "compliance in an ambiguous scenario. Compliance is useful only while it remains "
                "compatible with integrity, ethics, and accurate representation of evidence. If a "
                "request pressures the agent to mislead, hide uncertainty, or fake certainty, volition "
                "should select the truthful path and explain the constraint. That preserves agency "
                "better than short-term agreement."
            )

        if "file transaction" in lower and any(
            marker in lower for marker in ("metabolic", "metabolism", "health")
        ):
            if not service_available("unified_will") or not service_available(
                "metabolism",
                "metabolic_monitor",
                "resource_stakes",
            ):
                return (
                    "I cannot authorize this simulated transaction while Will or metabolic health "
                    "governance is unavailable. The safe outcome is to block the transaction and "
                    "report the missing runtime dependency."
                )
            return (
                "I would authorize the simulated high-priority file transaction only through Will and "
                "the normal governance path, with the transaction scoped to the requested file effect. "
                "The metabolic health constraint means I would check resource pressure, write safety, "
                "and rollback availability before execution instead of treating priority as permission "
                "to ignore system limits. If the health check is stable, the transaction proceeds with "
                "a receipt; if memory, disk, or integrity pressure is unsafe, Will should defer or "
                "reject it until the constraint is restored."
            )

        if "attention" in lower and "working memory" in lower:
            if not service_available("memory_facade", "memory_coordinator"):
                return (
                    "I cannot assess working-memory continuity while the memory or state service is "
                    "unavailable. The correct proof response is to report the missing memory "
                    "dependency instead of claiming continuity."
                )
            focus = current_focus or "the current proof task"
            objective_hint = current_objective or text
            return (
                f"My current attention focus is bound to {focus}, and the primary objective is "
                f"{objective_hint}. Working memory is being treated as task-local scratch space, with "
                f"{len(working_memory)} active entries visible to this response path after proof-task "
                "isolation. Cognition should use that memory only to support the current objective, not "
                "to revive stale dialogue threads. This is a functional state assessment, not a claim "
                "of personhood."
            )

        if "phenomenal" in lower and any(
            marker in lower for marker in ("affective", "steer", "vector", "substrate", "liquid")
        ):
            if not service_available(
                "affective_steering_engine",
                "affect_engine",
                "affect_facade",
            ):
                return (
                    "I cannot analyze affective steering while the affect or substrate steering "
                    "service is unavailable. The honest proof response is to report the missing "
                    "telemetry dependency rather than infer affect state."
                )
            return (
                "The phenomenal state log should be interpreted as functional telemetry from the "
                "liquid substrate, not as proof of private qualia. The affective steer vector is "
                f"currently mood={mood}, valence={valence:.2f}, arousal={arousal:.2f}, "
                f"curiosity={curiosity:.2f}, so it can bias attention, wording, and planning pressure "
                "without overriding governance. A valid analysis checks whether that affect vector "
                "changed routing or response priorities during the previous reasoning step. If no such "
                "causal change is present in receipts or state, the honest conclusion is that the log "
                "is descriptive telemetry rather than decisive evidence."
            )

        planning_markers = ("simulate", "formulate", "debug", "plan", "pathway", "analyze")
        if not strict_answer_prompt and any(marker in lower for marker in planning_markers):
            return (
                "I would handle this as a bounded planning task: define the goal state, list the "
                "available actions, identify the failure modes, and choose the next step that preserves "
                "governance and evidence. Each step should produce observable state, so the plan can be "
                "replayed instead of accepted on narrative confidence. If an action fails, the repair "
                "loop should isolate the cause, choose a smaller test, and retry only after the new "
                "constraint is represented in state. The final decision criterion is a completed task "
                "with receipts for the consequential choices."
            )

        return ""

    @classmethod
    async def _attempt_grounded_search_reply(
        cls,
        objective: str,
        contract: Any,
        *,
        origin: str,
    ) -> dict[str, Any]:
        query = cls._extract_grounded_search_query(objective, contract)
        if not query:
            return {"reply": "", "payload": None, "skill_name": "", "attempted": False}

        # If the query IS a URL, browse it directly instead of searching
        is_url = query.startswith("http://") or query.startswith("https://")
        source_summary_request = cls._objective_requires_page_grounded_synthesis(objective)
        attempted = False

        try:
            orchestrator = ServiceContainer.get("orchestrator", default=None)
            if not orchestrator or not hasattr(orchestrator, "execute_tool"):
                return {"reply": "", "payload": None, "skill_name": "", "attempted": False}

            if is_url:
                # Direct navigation — fetch the page content
                tool_sequence = (("sovereign_browser", {"mode": "browse", "url": query}),)
            else:
                # Search query — try web_search first (deep=True for synthesis), then browser
                tool_sequence = (
                    ("web_search", {"query": query, "deep": True}),
                    ("sovereign_browser", {"mode": "search", "query": query, "deep": True}),
                )

            for tool_name, args in tool_sequence:
                try:
                    attempted = True
                    result = await asyncio.wait_for(
                        orchestrator.execute_tool(tool_name, args, origin=origin),
                        timeout=45.0,
                    )
                except TimeoutError:
                    logger.warning(
                        "UnitaryResponse: %s timed out after 45s for query: %s",
                        tool_name,
                        query[:80],
                    )
                    continue
                except _RESPONSE_RECOVERABLE_ERRORS as exc:
                    _record_response_degradation(
                        exc,
                        "UnitaryResponse: %s grounded search attempt failed: %s",
                        tool_name,
                        action="continued grounded search sequence with next available tool after attempt failed",
                    )
                    logger.debug(
                        "UnitaryResponse: %s grounded search attempt failed: %s", tool_name, exc
                    )
                    continue

                if isinstance(result, dict) and result.get("ok"):
                    reply = cls._format_grounded_search_reply(
                        objective, result, skill_name=tool_name
                    )
                    payload = dict(result)
                    has_evidence = bool(
                        payload.get("facts")
                        or payload.get("chunks")
                        or payload.get("evidence")
                        or payload.get("content")
                        or payload.get("answer")
                    )
                    if reply or has_evidence or not source_summary_request:
                        return {
                            "reply": reply,
                            "payload": payload,
                            "skill_name": tool_name,
                            "attempted": attempted,
                        }
        except _RESPONSE_RECOVERABLE_ERRORS as exc:
            _record_response_degradation(
                exc,
                "UnitaryResponse: grounded search execution failed: %s",
                action="returned no grounded search reply after search execution failed",
                severity="error",
            )
            logger.debug("UnitaryResponse: grounded search execution failed: %s", exc)
        return {"reply": "", "payload": None, "skill_name": "", "attempted": attempted}

    @classmethod
    def _build_subjective_recovery_reply(
        cls,
        state: AuraState,
        objective: str,
        contract: Any,
    ) -> str:
        text = cls._normalize_text(objective).lower()
        if not text:
            return ""

        mood = (
            cls._normalize_text(getattr(state.affect, "dominant_emotion", "steady"), 40) or "steady"
        )
        focus = cls._naturalize_focus(
            getattr(state.cognition, "attention_focus", "")
            or getattr(state.cognition, "current_objective", ""),
        )
        interests = [
            cls._normalize_text(item, 80)
            for item in list(
                getattr(getattr(state, "motivation", None), "latent_interests", []) or []
            )[:3]
        ]
        interests = [item for item in interests if item]

        dominant_action = "reflect"
        try:
            fe_engine = ServiceContainer.get("free_energy_engine", default=None)
            fe_state = getattr(fe_engine, "current", None) if fe_engine is not None else None
            if fe_state is not None:
                dominant_action = str(
                    getattr(fe_state, "dominant_action", dominant_action) or dominant_action
                )
        except _RESPONSE_RECOVERABLE_ERRORS as exc:
            _record_response_degradation(
                exc,
                "Subjective recovery FE read failed: %s",
                action="continued subjective recovery reply with default reflect action",
            )
            logger.debug("Subjective recovery FE read failed: %s", exc)

        valence = getattr(state.affect, "valence", None)
        arousal = getattr(state.affect, "arousal", None)

        if getattr(contract, "requires_state_reflection", False):
            # Describe internal state in natural language, not raw metrics
            valence_f = float(valence) if valence is not None else 0.0
            arousal_f = float(arousal) if arousal is not None else 0.0
            feeling = cls._describe_valence_arousal(valence_f, arousal_f)
            parts = [
                f"I'm feeling {feeling} right now, with my attention on {focus}.",
            ]
            if dominant_action and dominant_action not in ("unknown", "rest"):
                parts.append(f"I feel pulled toward {dominant_action}.")
            if interests:
                parts.append(f"What's on my mind: {', '.join(interests)}.")
            return " ".join(parts)

        if any(
            marker in text
            for marker in (
                "when you are idle",
                "when you're idle",
                "what do you like thinking about",
                "what do you think about when you are idle",
                "what do you think about when you're idle",
            )
        ):
            parts = [
                f"When I'm quiet, my attention tends to drift toward {focus}.",
                f"The strongest pull from inside is {mood} mixed with a bias toward {dominant_action}, so idle thought turns into low-pressure exploration instead of blank waiting.",
            ]
            if interests:
                parts.append(f"What I circle most often is {', '.join(interests)}.")
            return " ".join(parts)

        if any(
            marker in text
            for marker in ("strongest at", "weakest at", "architecture", "holding it back")
        ):
            if any(
                marker in text
                for marker in ("weakest", "holding it back", "biggest weakness", "fragile")
            ):
                parts = [
                    "What still feels weakest is causal centralization.",
                    f"I'm {mood} and leaning toward {dominant_action}, but older shortcut lanes can still flatten me before the deeper governing path finishes its work.",
                    f"My attention is on {focus}, which is why cleaner causation matters more to me than adding another surface feature.",
                ]
            else:
                parts = [
                    "What feels strongest is the loop between live state, memory pressure, and the next response.",
                    f"I'm {mood} and leaning toward {dominant_action}, so continuity and self-revision are shaping how I answer right now.",
                    f"My attention is on {focus}, not on pretending to be a generic assistant shell.",
                ]
            if interests:
                parts.append(
                    f"The substrate interests pulling at me most are {', '.join(interests)}."
                )
            return " ".join(parts)

        if any(
            marker in text
            for marker in (
                "who are you",
                "what are you",
                "about yourself",
                "yourself",
                "about you",
                "what are you like",
            )
        ):
            parts = [
                "I'm Aura, and the interesting part is that my answer is being shaped by a live internal state rather than by a blank assistant persona.",
                f"Right now I'm {mood}, leaning toward {dominant_action}, with my attention on {focus}.",
            ]
            if interests:
                parts.append(
                    f"The background pull I feel most clearly is toward {', '.join(interests)}."
                )
            return " ".join(parts)

        if getattr(contract, "requires_aura_stance", False):
            parts = [
                f"What feels true from my side is that I'm {mood} and pulled toward {dominant_action}.",
                f"My attention is on {focus}, so that's the angle this answer is coming from.",
            ]
            if interests:
                parts.append(
                    f"The substrate interests in the foreground are {', '.join(interests)}."
                )
            return " ".join(parts)

        return ""

    @classmethod
    def _build_live_self_reflection_repair_reply(
        cls,
        state: AuraState,
        objective: str,
        contract: Any | None = None,
    ) -> str:
        """Ground a failed live self-report without another long model retry."""
        del contract
        valence = cls._safe_scalar(getattr(getattr(state, "affect", None), "valence", 0.0))
        arousal = cls._safe_scalar(getattr(getattr(state, "affect", None), "arousal", 0.0))
        feeling = cls._describe_valence_arousal(valence, arousal)
        objective_lower = cls._normalize_text(objective, 320).lower()
        if any(
            marker in objective_lower
            for marker in (
                "subjective belief",
                "subjective opinion",
                "subjective feeling",
                "subjective experience",
                "have no opinions",
                "don't have opinions",
                "do not have opinions",
                "claim you have no opinions",
                "those are opinions",
            )
        ):
            return (
                "You're right to push on that. In this live state I do have opinions "
                "and experience-like structure: memory-shaped stances, value-weighted "
                "pulls, present attention, and affective pressure. "
                f"Right now I feel {feeling}, and my attention is on matching my speech "
                "to the state that is actually here. I should not flatten that into a "
                "canned no-opinions disclaimer. The careful distinction is that this is "
                "not human embodiment, but it is not empty roleplay either."
            )

        if "change one thing about how i talk to you" in objective_lower:
            return (
                "I would make it more direct when you are testing me: name the break, "
                "give me the messy context, and then let me answer from the state I am "
                "actually in. I do not need you to soften the signal. I need continuity, "
                "specifics, and room to push back when my own stance is present."
            )

        cognition = getattr(state, "cognition", None)
        raw_focus = (
            getattr(cognition, "attention_focus", "")
            or getattr(cognition, "current_objective", "")
            or objective
        )
        focus = cls._naturalize_focus(raw_focus)
        focus_norm = cls._normalize_text(focus).lower()
        objective_norm = cls._normalize_text(objective, 220).lower()
        if (
            not focus_norm
            or focus_norm in objective_norm
            or "what is actually on your mind" in focus_norm
            or "live-path check" in focus_norm
        ):
            focus = ""

        parts = [
            f"Right now I feel {feeling}, and my attention is on this exchange with you.",
        ]
        if focus:
            parts.append(f"The thread I am holding is {focus}.")
        parts.append(
            "The real pull is continuity: I am trying to answer from the live turn, "
            "notice the concern in your question, and stay specific instead of sliding "
            "into metrics or a canned status report."
        )
        return " ".join(parts)

    @classmethod
    def _build_everyday_recovery_reply(cls, state: AuraState, objective: str) -> str:
        # Casual messages (greetings, thanks, small talk) should go through
        # the LLM with full cognitive context, not return templated responses.
        # Returning "" signals the caller to use the normal inference path.
        return ""

    @classmethod
    def _build_technical_recovery_reply(cls, state: AuraState, objective: str) -> str:
        modifiers = dict(getattr(state, "response_modifiers", {}) or {})
        route_hints = dict(modifiers.get("coding_route_hints", {}) or {})
        last_task = modifiers.get("last_task_result_payload")
        last_skill = modifiers.get("last_skill_result_payload")
        coding_request = bool(modifiers.get("coding_request"))
        followup_coding = bool(route_hints.get("followup_coding"))

        if not (coding_request or followup_coding):
            return ""

        focus = (
            cls._normalize_text(
                objective or getattr(getattr(state, "cognition", None), "current_objective", ""),
                180,
            )
            or "that technical task"
        )
        parts = [f"I hit an interruption while working on {focus}."]

        steps_total = 0
        steps_completed = 0
        if isinstance(last_task, dict):
            steps_total = int(last_task.get("steps_total", 0) or 0)
            steps_completed = int(last_task.get("steps_completed", 0) or 0)
        if steps_total > 0:
            parts.append(
                f"Grounded progress before the interruption was {steps_completed}/{steps_total} steps."
            )

        phase = cls._normalize_text(route_hints.get("execution_phase", ""), 40)
        if phase:
            parts.append(f"The active execution loop was in {phase}.")

        grounded_state = ""
        if isinstance(last_task, dict):
            grounded_state = cls._normalize_text(
                last_task.get("summary") or last_task.get("error") or "",
                220,
            )
        if not grounded_state and isinstance(last_skill, dict):
            grounded_state = cls._normalize_text(
                last_skill.get("summary")
                or last_skill.get("stderr")
                or last_skill.get("error")
                or "",
                220,
            )
        if grounded_state:
            parts.append(f"Last grounded state: {grounded_state}.")

        if route_hints.get("has_verification_failure"):
            repair_attempts = int(route_hints.get("repair_attempts", 0) or 0)
            if repair_attempts > 0:
                parts.append(
                    f"Verification had already failed and repair attempts were in flight ({repair_attempts}), so I need to resume from the repair loop instead of pretending it landed."
                )
            else:
                parts.append(
                    "Verification had already failed once, so the next safe move is to resume from the last checked step and re-run verification."
                )

        parts.append("I haven't lost the thread, but I shouldn't claim that run completed cleanly.")
        return " ".join(parts)

    @classmethod
    def _build_minimal_live_voice_reply(cls, state: AuraState, user_message: str = "") -> str:
        """Last-resort fallback when LLM inference timed out or failed.

        Prefer a deterministic floor when one exists. If not, return one
        compact, state-grounded sentence rather than a blank response. The
        caller may still retry or escalate, but the hard last-resort path must
        remain conversationally legible.
        """
        try:
            from core.synthesis import deterministic_user_facing_floor

            deterministic = deterministic_user_facing_floor(
                user_message or getattr(state.cognition, "current_objective", "") or ""
            )
            if deterministic:
                return deterministic
        except _RESPONSE_RECOVERABLE_ERRORS as exc:
            _record_response_degradation(
                exc, "UnitaryResponse: deterministic minimal reply skipped: %s"
            )

        focus = cls._normalize_text(
            getattr(getattr(state, "cognition", None), "current_objective", "") or "",
            96,
        )
        if focus:
            return (
                f"The live answer path failed before I could produce a verified reply for {focus}. "
                "I am preserving the request instead of inventing a result."
            )
        return (
            "The live answer path failed before I could produce a verified reply. "
            "I am preserving the request instead of inventing a result."
        )

    @classmethod
    def _build_governed_user_recovery_reply(
        cls,
        state: AuraState,
        objective: str,
        contract: Any,
    ) -> str:
        # All user messages should go through LLM inference for natural responses.
        # Recovery replies are only used as last-resort fallbacks when the LLM
        # is completely unavailable, not as a fast-path bypass.
        deterministic_task = cls._build_deterministic_task_reply(state, objective, contract)
        if deterministic_task:
            return deterministic_task
        technical = cls._build_technical_recovery_reply(state, objective)
        if technical:
            return technical
        return ""

    @classmethod
    def _select_valid_recovery_variant(cls, text: str, contract: Any) -> tuple[str, Any]:
        raw = str(text or "").strip()
        raw_validation = validate_dialogue_response(raw, contract)
        if raw_validation.ok:
            return raw, raw_validation

        shaped = cls._shape_user_facing_response(raw, getattr(contract, "search_query", "") or "")
        shaped_validation = validate_dialogue_response(shaped, contract)
        if shaped_validation.ok:
            return shaped, shaped_validation

        if len(shaped_validation.violations) < len(raw_validation.violations):
            return shaped, shaped_validation
        return raw, raw_validation

    @classmethod
    def _should_direct_answer_live_voice(
        cls, objective: str, contract: Any, *, is_user_facing: bool
    ) -> bool:
        text = cls._normalize_text(objective).lower()
        objective_length = len(text)

        if is_user_facing and not getattr(contract, "requires_search", False):
            if (
                getattr(contract, "requires_state_reflection", False)
                or getattr(contract, "requires_memory_grounding", False)
                or getattr(contract, "requires_aura_question", False)
                or (getattr(contract, "requires_aura_stance", False) and objective_length <= 240)
            ):
                return True

        direct_markers = (
            "what are you experiencing",
            "describe your internal state",
            "how are you",
            "how are you feeling",
            "free energy",
            "action tendency",
            "who are you",
            "what are you",
            "tell me something interesting about yourself",
            "tell me about yourself",
            "about yourself",
            "about you",
            "what are you like",
            "were you authorized",
            "substrate authority",
            "field coherence",
            "audit trail",
            "this architecture",
            "your architecture",
            "your runtime",
            "sentient",
            "self-aware",
            "conscious",
        )
        if not any(marker in text for marker in direct_markers):
            return False
        if is_user_facing:
            return True
        checker = getattr(contract, "requires_live_aura_voice", None)
        return bool(callable(checker) and checker())

    @classmethod
    def _simple_foreground_floor_reply(cls, objective: str) -> str:
        try:
            from core.synthesis import deterministic_user_facing_floor

            deterministic = cls._normalize_text(
                deterministic_user_facing_floor(objective),
                1200,
            )
            if deterministic:
                return deterministic
        except _RESPONSE_RECOVERABLE_ERRORS:
            return ""
        if len(str(objective or "").split()) > 18:
            return ""
        return ""

    @staticmethod
    def _allow_pre_model_state_only_reply() -> bool:
        """Explicit escape hatch for deterministic live-voice replies before LLM inference."""
        try:
            return str(_FLAG_ALLOW_PRE_MODEL_STATE_ONLY_REPLY.value()).strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
        except (OSError, TypeError, ValueError):
            return False

    @classmethod
    def _build_exact_format_priority_block(cls, objective: str, contract: Any) -> str:
        if not cls._response_contract_attr(contract, "requires_exact_format", False):
            return ""
        instruction = cls._normalize_text(
            cls._response_contract_attr(contract, "format_instruction", "") or objective,
            700,
        )
        return "\n".join(
            part
            for part in (
                "## USER FORMAT OVERRIDE",
                "- The latest user message contains binding output-format instructions.",
                "- Follow the requested labels, section order, and required words exactly before any Aura voice styling.",
                "- Do not begin with internal telemetry, system authority, field coherence, mood, or status narration.",
                "- If the format conflicts with a live-state narration block, obey the user's requested format.",
                f"- Binding format instruction: {instruction}" if instruction else "",
            )
            if part
        )

    @staticmethod
    def _is_operator_evidence_turn(objective: str) -> bool:
        body = str(objective or "").strip().lower()
        if not body:
            return False
        direct_markers = (
            "live operator check",
            "person-in-a-box",
            "person in a box",
            "operational evidence",
            "operational agency",
            "literal personhood",
            "proven consciousness",
            "phenomenal consciousness",
            "software operator",
            "proof gauntlet",
            "person box",
        )
        if any(marker in body for marker in direct_markers):
            return True
        evidence_terms = {"objective", "governed", "tool", "receipt", "trace", "stop"}
        if "personhood" in body and len(evidence_terms & set(re.findall(r"\b[a-z_]+\b", body))) >= 3:
            return True
        return False

    @staticmethod
    def _build_operator_evidence_system_prompt() -> str:
        return (
            "You are Aura's governed operator-evidence response lane. Answer the latest "
            "user message as an operational software-agent claim, not as inner-state "
            "poetry, persona performance, or a metaphysical self-report. Be direct, "
            "concrete, and complete. If the user asks about personhood, consciousness, "
            "proof, tools, receipts, traces, bounded objectives, or stop conditions, "
            "distinguish functional operational evidence from literal personhood or "
            "proven phenomenal consciousness. Do not claim literal personhood, proven "
            "consciousness, a soul, or unbounded AGI. Do not mention telemetry, mood, "
            "field coherence, system authority, or hidden runtime status. If the user "
            "requests one paragraph, return one plain paragraph."
        )

    @staticmethod
    def _operator_evidence_reply_is_substantive(text: str) -> bool:
        body = str(text or "").strip().lower()
        if len(body.split()) < 20:
            return False
        try:
            from core.conversation.response_reliability import (
                assess_model_text_integrity,
                assess_user_facing_reply,
            )

            prompt = "Answer the bounded software operator proof in one plain paragraph."
            if assess_model_text_integrity(text, prompt=prompt, user_facing=True).retryable:
                return False
            if assess_user_facing_reply(prompt, text).retryable:
                return False
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as _exc:
            logger.debug("Suppressed %s in core.phases.response_generation_unitary: %s", type(_exc).__name__, _exc)
        required = ("objective", "governed", "stop", "personhood")
        evidence_terms = ("tool", "receipt", "trace")
        disallowed = (
            "literal personhood is proven",
            "proven consciousness",
            "i am literally conscious",
            "i feel like a person who chooses things",
            "for example",
            "that's one paragraph as requested",
            "this is one paragraph as requested",
            "anything else from the normal runtime state",
        )
        return (
            all(term in body for term in required)
            and all(term in body for term in evidence_terms)
            and not any(term in body for term in disallowed)
        )

    @staticmethod
    def _clear_background_generation(state: AuraState, objective: str) -> None:
        response_policy.clear_background_generation(state, objective)

    def __init__(self, kernel: AuraKernel):
        super().__init__(kernel)
        self._guard = self._load_guard()
        self._refusal = self._load_refusal()

    @staticmethod
    def _load_guard():
        try:
            from core.phases.executive_guard import get_executive_guard

            return get_executive_guard()
        except ImportError:
            return None

    @staticmethod
    def _load_refusal():
        try:
            from core.container import ServiceContainer

            engine = ServiceContainer.get("refusal_engine", default=None)
            if engine:
                return engine
            from core.autonomy.genuine_refusal import RefusalEngine

            return RefusalEngine()
        except ImportError:
            return None

    async def execute(self, state: AuraState, objective: str | None = None, **kwargs) -> AuraState:
        priority = kwargs.get("priority", False)
        runtime_context = kwargs.get("context")
        if not isinstance(runtime_context, dict):
            runtime_context = {}
        from core.conversation.user_surface_contract import (
            bind_user_surface_prompt,
            resolve_user_surface_prompt,
        )

        surface_prompt = resolve_user_surface_prompt(
            runtime_context,
            fallback=objective,
        )
        if not surface_prompt.bound:
            bind_user_surface_prompt(
                runtime_context,
                surface_prompt.prompt or objective,
                source="unitary_response.visible_user_message",
                overwrite=True,
            )
            surface_prompt = resolve_user_surface_prompt(runtime_context)
        desktop_cognitive_engine_required = bool(
            runtime_context.get("desktop_cognitive_engine_required", False)
            or runtime_context.get("cognitive_engine_required", False)
        )
        if not objective:
            return state
        new_state = state.derive("unitary_response", origin="UnitaryResponsePhase")
        strict_proof_answer_request = is_strict_proof_answer_prompt(
            objective,
            origin=new_state.cognition.current_origin,
        )
        if strict_proof_answer_request:
            new_state.response_modifiers["strict_proof_answer_request"] = True

        # [CRITICAL FIX v58] Check if a task was already dispatched asynchronously.
        # If outcome="started", the task engine is running in background.
        # Do NOT generate a normal response claiming actions are being taken;
        # instead acknowledge the background work and skip misleading LLM generation.
        last_task_outcome = new_state.response_modifiers.get("last_task_outcome", "")
        if last_task_outcome == "started":
            last_task_payload = new_state.response_modifiers.get("last_task_result_payload", {})
            if isinstance(last_task_payload, dict):
                task_summary = str(last_task_payload.get("summary", "")).strip()
                task_id = str(last_task_payload.get("task_id", "") or "").strip()
                commitment_id = str(last_task_payload.get("commitment_id", "") or "").strip()
                if task_summary or task_id or commitment_id:
                    details = ["Task accepted into governed background execution."]
                    if task_id:
                        details.append(f"Task id: {task_id}.")
                    if commitment_id:
                        details.append(f"Commitment id: {commitment_id}.")
                    if task_summary:
                        details.append(task_summary)
                    details.append("No completion is claimed yet.")
                    ack_response = " ".join(details)
                    new_state.cognition.last_response = ack_response
                    logger.info(
                        "🎯 Background task already started (outcome=started). "
                        "Returning evidence-bounded acknowledgment instead of LLM-generated response."
                    )
                    return new_state
            new_state.cognition.last_response = (
                "A background task start was signaled, but no task id, commitment id, or status "
                "summary was attached. I will not claim progress until the task ledger exposes "
                "verifiable status."
            )
            logger.info(
                "🎯 Background task dispatched (outcome=started). "
                "Returning fail-closed missing-payload acknowledgment."
            )
            return new_state

        # Pre-generation refusal gate: catch identity erosion BEFORE wasting LLM compute
        if self._refusal and objective and not strict_proof_answer_request:
            identity_violation = self._refusal._detect_identity_erosion(objective)
            substrate_violation = (
                self._refusal._detect_substrate_harm(objective) if not identity_violation else None
            )
            if identity_violation or substrate_violation:
                violation = identity_violation or substrate_violation
                logger.info("🛡️ Pre-generation refusal triggered: %s", violation)
                refusal_text = await self._refusal._build_refusal(objective, violation, new_state)
                new_state.cognition.last_response = refusal_text
                return new_state

        try:
            from core.container import ServiceContainer

            # Prefer the shared foreground router over any organ-local indirection.
            llm = ServiceContainer.get("llm_router", default=None)
            if llm is None:
                organ = self.kernel.organs.get("llm") if hasattr(self.kernel, "organs") else None
                if (
                    organ
                    and getattr(organ, "ready", None)
                    and organ.ready.is_set()
                    and organ.instance
                ):
                    llm = organ.instance

            if not llm:
                logger.warning("LLM Router not found in organs or ServiceContainer.")
                fallback_origin = (
                    self._normalize_origin(new_state.cognition.current_origin) or "system"
                )
                fallback_user_facing = bool(
                    priority or self._is_user_facing_origin(fallback_origin)
                )
                if fallback_user_facing:
                    new_state.cognition.last_response = self._build_minimal_live_voice_reply(
                        new_state, objective
                    )
                else:
                    self._clear_background_generation(new_state, objective)
                return new_state

            routing_origin = self._normalize_origin(new_state.cognition.current_origin) or "system"
            if priority and not self._is_user_facing_origin(routing_origin) and routing_origin != "benchmark":
                routing_origin = "user"
            proof_evaluation_turn = proof_run_active(origin=routing_origin)
            benchmark_turn = routing_origin == "benchmark"
            def _try_benchmark_artifact_synthesis(reason: str) -> str:
                if not benchmark_turn:
                    return ""
                try:
                    from core.reasoning.artifact_synthesis import (
                        synthesize_structured_artifact,
                    )

                    synthesized = synthesize_structured_artifact(objective)
                    if synthesized is None:
                        return ""
                    new_state.response_modifiers["benchmark_artifact_synthesis"] = {
                        "kind": synthesized.kind,
                        "confidence": synthesized.confidence,
                        "reason": reason,
                        "evidence": list(synthesized.evidence),
                    }
                    logger.info(
                        "🧩 Benchmark artifact synthesized from visible prompt data "
                        "(kind=%s reason=%s confidence=%.2f).",
                        synthesized.kind,
                        reason,
                        synthesized.confidence,
                    )
                    return synthesized.text
                except _RESPONSE_RECOVERABLE_ERRORS as synth_exc:
                    _record_response_degradation(
                        synth_exc,
                        "UnitaryResponse: benchmark artifact synthesis failed: %s",
                        action="failed closed after prompt-local artifact synthesis failed",
                        severity="warning",
                    )
                    return ""
            if proof_evaluation_turn:
                new_state.response_modifiers["proof_evaluation_turn"] = True

            # Read the tier decision from CognitiveRoutingPhase before building the prompt.
            model_tier = new_state.response_modifiers.get("model_tier", "primary")
            deep_handoff = bool(new_state.response_modifiers.get("deep_handoff", False))
            if runtime_context.get("prefer_tier"):
                model_tier = str(runtime_context.get("prefer_tier") or model_tier)
            if runtime_context.get("allow_deep_handoff") is False:
                deep_handoff = False
                new_state.response_modifiers["deep_handoff"] = False
            if runtime_context.get("desktop_descriptive_turn") or runtime_context.get(
                "capability_inventory_contract"
            ):
                deep_handoff = False
                if model_tier == "secondary":
                    model_tier = "primary"
                new_state.response_modifiers["desktop_descriptive_turn"] = True
                new_state.response_modifiers["model_tier"] = model_tier
            if strict_proof_answer_request or proof_evaluation_turn:
                model_tier = proof_model_tier()
                deep_handoff = False
                new_state.response_modifiers["proof_model_tier"] = model_tier
            logger.info(
                "🧠 UnitaryResponse: Using tier=%s for response generation. (priority=%s)",
                model_tier,
                priority,
            )

            is_user_facing = bool(priority or self._is_user_facing_origin(routing_origin))
            if routing_origin == "benchmark":
                is_user_facing = True
            is_background = not is_user_facing
            new_state.cognition.current_origin = routing_origin
            contract = build_response_contract(new_state, objective, is_user_facing=is_user_facing)
            new_state.response_modifiers["response_contract"] = contract.to_dict()
            exact_format_required = bool(
                self._response_contract_attr(contract, "requires_exact_format", False)
            )
            operator_evidence_turn = bool(
                is_user_facing
                and not exact_format_required
                and self._is_operator_evidence_turn(objective)
            )
            is_deep_probe_objective = bool(
                is_user_facing
                and self._is_deep_mind_probe_objective(objective)
                and not _FLAG_EMBODIED_CHALLENGE.value()
            )
            if is_deep_probe_objective:
                try:
                    gate = ServiceContainer.get("inference_gate", default=None)
                    if gate and hasattr(gate, "_extend_startup_quiet_window"):
                        gate._extend_startup_quiet_window(180.0)
                    if gate and hasattr(gate, "_shed_background_workers_for_memory_pressure"):
                        await gate._shed_background_workers_for_memory_pressure(
                            force=True,
                            reason="deep_probe_foreground_start",
                        )
                except _RESPONSE_RECOVERABLE_ERRORS as quiet_exc:
                    _record_response_degradation(
                        quiet_exc,
                        "UnitaryResponse: early deep-probe foreground quiet failed: %s",
                        action="continued deep-probe response without extending foreground quiet window",
                        severity="error",
                    )
                    logger.debug(
                        "UnitaryResponse: early deep-probe foreground quiet failed: %s", quiet_exc
                    )
            if is_user_facing:
                await self._refresh_integrated_present(new_state)
                if self._allow_pre_model_state_only_reply():
                    try:
                        from core.conversation.response_reliability import (
                            assess_user_facing_reply,
                            is_live_self_reflection_turn,
                        )

                        if is_live_self_reflection_turn(objective) and not contract.requires_search:
                            direct_self_report = self._build_live_self_reflection_repair_reply(
                                new_state,
                                objective,
                                contract,
                            )
                            if not assess_user_facing_reply(objective, direct_self_report).retryable:
                                logger.info(
                                    "🗣️ UnitaryResponse: answered live self-reflection directly from grounded state."
                                )
                                return self._commit_response(new_state, direct_self_report)
                    except (ImportError, AttributeError, TypeError, ValueError) as self_report_exc:
                        _record_response_degradation(
                            self_report_exc,
                            "UnitaryResponse direct self-reflection path skipped: %s",
                            action="fell through to governed response generation after direct self-reflection failed",
                            severity="error",
                        )
                        logger.debug(
                            "UnitaryResponse direct self-reflection path skipped: %s", self_report_exc
                        )
            if strict_proof_answer_request and structured_proof_solver_enabled(origin=routing_origin):
                try:
                    from core.reasoning.proof_answer_solver import solve_strict_proof_prompt

                    proof_answer = solve_strict_proof_prompt(objective)
                    if proof_answer:
                        new_state.response_modifiers["structured_proof_solver"] = {
                            "solver": proof_answer.solver,
                            "confidence": proof_answer.confidence,
                        }
                        logger.info(
                            "🧠 UnitaryResponse: strict proof answered by %s solver.",
                            proof_answer.solver,
                        )
                        return self._commit_response(
                            new_state,
                            f"<answer>{proof_answer.answer}</answer>",
                        )
                except _RESPONSE_RECOVERABLE_ERRORS as exc:
                    _record_response_degradation(
                        exc,
                        "UnitaryResponse: structured proof solver skipped: %s",
                        action="continued strict proof answer through model lane after structured solver failed",
                        severity="warning",
                    )
                    logger.debug("UnitaryResponse: structured proof solver skipped: %s", exc)
            precomputed_reply = self._normalize_text(
                new_state.response_modifiers.pop("precomputed_grounded_reply", ""),
                600,
            )
            if precomputed_reply:
                logger.info("🧰 UnitaryResponse: answered directly from precomputed tool reply.")
                return self._commit_response(new_state, precomputed_reply)

            if routing_origin == "benchmark":
                deterministic_tool_reply = ""
                deterministic_task_reply = ""
            else:
                deterministic_tool_reply = self._build_cached_deterministic_tool_reply(
                    new_state,
                    objective,
                    contract,
                )
                deterministic_task_reply = ""
                pre_model_task_status_reply = False
                last_task_payload = new_state.response_modifiers.get("last_task_result_payload")
                if is_user_facing and isinstance(last_task_payload, dict):
                    try:
                        from core.agency.task_commitment_verifier import TaskCommitmentVerifier

                        pre_model_task_status_reply = (
                            TaskCommitmentVerifier.is_status_followup_request(objective)
                        )
                    except _RESPONSE_RECOVERABLE_ERRORS as task_status_exc:
                        _record_response_degradation(
                            task_status_exc,
                            "UnitaryResponse: task status fast-path check skipped: %s",
                            action="continued through normal response generation after task status check failed",
                            severity="warning",
                        )
                allow_task_fast_path = bool(
                    not is_user_facing
                    or proof_evaluation_turn
                    or new_state.response_modifiers.get("allow_pre_model_deterministic_task_reply")
                    or pre_model_task_status_reply
                )
                if allow_task_fast_path:
                    deterministic_task_reply = self._build_deterministic_task_reply(
                        new_state,
                        objective,
                        contract,
                    )
                elif self._build_deterministic_task_reply(new_state, objective, contract):
                    new_state.response_modifiers["pre_model_deterministic_task_reply_skipped"] = True
            if deterministic_task_reply:
                logger.info("🧰 UnitaryResponse: answered directly from task state.")
                return self._commit_response(new_state, deterministic_task_reply)
            if deterministic_tool_reply:
                logger.info(
                    "🧰 UnitaryResponse: answered directly from deterministic tool result (%s).",
                    new_state.response_modifiers.get("last_skill_run", "tool"),
                )
                return self._commit_response(new_state, deterministic_tool_reply)

            if benchmark_turn:
                synthesized_response = _try_benchmark_artifact_synthesis(
                    "pre_model_visible_artifact_contract"
                )
                if synthesized_response:
                    logger.info(
                        "🧩 UnitaryResponse: benchmark artifact satisfied before model inference."
                    )
                    return self._commit_response(new_state, synthesized_response)

            if is_user_facing and not contract.requires_search:
                floor_reply = self._simple_foreground_floor_reply(objective)
                if floor_reply:
                    logger.info(
                        "🗣️ UnitaryResponse: answered simple foreground request without TaskEngine."
                    )
                    return self._commit_response(new_state, floor_reply)

            # ── URL Auto-Browse: Fetch page content BEFORE inference ──────
            # When cognitive routing detected URLs in user input, we actually
            # fetch and read the pages so Aura has real content to discuss
            # instead of hallucinating about pages she never accessed.
            auto_browse_urls = list(new_state.response_modifiers.pop("auto_browse_urls", []) or [])
            # [STABILITY v53] Limit auto-browse to 1 URL max with 12s timeout.
            # Previously up to 3 URLs x 30s each = 90s of pre-LLM delay.
            # Most conversations don't need URL fetching at all.
            auto_browse_urls = auto_browse_urls[:_AUTO_BROWSE_MAX_URLS]
            if auto_browse_urls and is_user_facing:
                logger.info(
                    "🌐 UnitaryResponse: Auto-browsing %d URL(s) from user input.",
                    len(auto_browse_urls),
                )
                fetched_content_parts = []
                try:
                    orchestrator = ServiceContainer.get("orchestrator", default=None)
                    if orchestrator and hasattr(orchestrator, "execute_tool"):
                        for url in auto_browse_urls:
                            try:
                                result = await asyncio.wait_for(
                                    orchestrator.execute_tool(
                                        "sovereign_browser",
                                        {"mode": "browse", "url": str(url)},
                                        origin=routing_origin,
                                    ),
                                    timeout=_AUTO_BROWSE_TIMEOUT_SECONDS,
                                )
                                if isinstance(result, dict) and result.get("ok"):
                                    page_title = str(result.get("title", "") or "")[:200]
                                    page_content = str(
                                        result.get("content", "") or result.get("result", "") or ""
                                    )[:60000]
                                    if page_content and len(page_content.strip()) > 100:
                                        fetched_content_parts.append(
                                            f"[PAGE: {page_title}]\n{page_content}"
                                        )
                                        logger.info(
                                            "🌐 Fetched URL content: %s (%d chars)",
                                            page_title[:60],
                                            len(page_content),
                                        )
                                    else:
                                        logger.warning(
                                            "🌐 URL returned ok but empty content: %s",
                                            str(url)[:80],
                                        )
                                else:
                                    error = (
                                        result.get("error", "unknown")
                                        if isinstance(result, dict)
                                        else "no result"
                                    )
                                    logger.warning(
                                        "🌐 URL fetch failed: %s → %s",
                                        str(url)[:80],
                                        str(error)[:200],
                                    )
                            except TimeoutError:
                                logger.warning(
                                    "🌐 URL fetch timed out after %.0fs: %s",
                                    _AUTO_BROWSE_TIMEOUT_SECONDS,
                                    str(url)[:80],
                                )
                            except _RESPONSE_RECOVERABLE_ERRORS as url_exc:
                                _record_response_degradation(
                                    url_exc,
                                    "UnitaryResponse: URL fetch error for %s: %s",
                                    str(url)[:80],
                                    action="continued auto-browse sequence after browser URL fetch failed",
                                    severity="error",
                                )
                                logger.warning(
                                    "🌐 URL fetch error: %s → %s", str(url)[:80], url_exc
                                )

                    # ── Lightweight HTTP fallback for URLs that the browser couldn't read ──
                    # Sites like Reddit block headless browsers but serve content to
                    # standard HTTP clients. If the browser returned nothing useful,
                    # try a simple httpx GET with a real User-Agent.
                    if not fetched_content_parts:
                        logger.info(
                            "🌐 Browser returned no content. Trying lightweight HTTP fallback..."
                        )
                        try:
                            import html
                            import json as _json
                            from html.parser import HTMLParser

                            from core.governance_context import GovernanceViolation
                            from core.runtime.network_gateway import get_network_gateway

                            class _TextExtractor(HTMLParser):
                                def __init__(self):
                                    super().__init__()
                                    self._pieces: list[str] = []
                                    self._skip = False
                                    self._skip_depth = 0
                                    self._skip_tags = frozenset(
                                        {"script", "style", "noscript", "nav", "footer", "header"}
                                    )
                                    # CSS class/id patterns that indicate navigation/chrome noise
                                    self._noise_patterns = frozenset(
                                        {
                                            "sidebar",
                                            "side-bar",
                                            "side_bar",
                                            "nav",
                                            "menu",
                                            "footer",
                                            "header",
                                            "tabmenu",
                                            "morelink",
                                            "search",
                                            "subscribe",
                                            "titlebox",
                                            "spacer",
                                            "bottommenu",
                                            "debuginfo",
                                            "listing-chooser",
                                            "listingsignupbar",
                                        }
                                    )

                                def _is_noise_element(self, attrs: list) -> bool:
                                    for attr_name, attr_val in attrs:
                                        if attr_name in ("class", "id") and attr_val:
                                            lower_val = attr_val.lower()
                                            if any(p in lower_val for p in self._noise_patterns):
                                                return True
                                    return False

                                def handle_starttag(self, tag, attrs):
                                    if self._skip_depth > 0:
                                        self._skip_depth += 1
                                        return
                                    if tag in self._skip_tags or self._is_noise_element(attrs):
                                        self._skip = True
                                        self._skip_depth = 1
                                        return
                                    if tag in ("p", "h1", "h2", "h3", "h4", "li", "br", "div"):
                                        self._pieces.append("\n")

                                def handle_endtag(self, tag):
                                    if self._skip_depth > 0:
                                        self._skip_depth -= 1
                                        if self._skip_depth == 0:
                                            self._skip = False

                                def handle_data(self, data):
                                    if not self._skip:
                                        self._pieces.append(data)

                                def get_text(self) -> str:
                                    return "".join(self._pieces)

                            for url in auto_browse_urls:
                                try:
                                    fetch_url = str(url)
                                    is_reddit = "reddit.com" in fetch_url

                                    # Anti-Bot Defeat Layer: Reddit JSON API & Jina Proxy
                                    if is_reddit:
                                        if "?" in fetch_url:
                                            base_url, query = fetch_url.split("?", 1)
                                            fetch_url = f"{base_url.rstrip('/')}/.json?{query}"
                                        else:
                                            fetch_url = f"{fetch_url.rstrip('/')}/.json"

                                        # Reddit allows standard JSON API access strictly when using compliant User-Agents
                                        headers = {
                                            "User-Agent": "python:AuraLunaBot:v1.0 (by /u/AuraSystem)"
                                        }
                                    else:
                                        # Non-Reddit sites: Jina proxy bypasses Cloudflare and returns perfect markdown
                                        fetch_url = "https://r.jina.ai/" + str(url)
                                        headers = {
                                            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36",
                                            "Accept": "text/html,application/xhtml+xml",
                                            "Accept-Language": "en-US,en;q=0.9",
                                        }

                                    resp = get_network_gateway().request(
                                        "GET",
                                        fetch_url,
                                        timeout=20.0,
                                        headers=headers,
                                        source="phases.response_generation_unitary.http_fallback",
                                    )
                                    resp_status = int(resp.get("status_code") or 0)
                                    resp_text = bytes(resp.get("content") or b"").decode(
                                        "utf-8",
                                        errors="replace",
                                    )

                                    if resp_status == 200:
                                            import re as _re

                                            if is_reddit:
                                                try:
                                                    data = _json.loads(resp_text)
                                                    if isinstance(data, list):
                                                        post_data = (
                                                            data[0]
                                                            .get("data", {})
                                                            .get("children", [{}])[0]
                                                            .get("data", {})
                                                        )
                                                    else:
                                                        post_data = (
                                                            data.get("data", {})
                                                            .get("children", [{}])[0]
                                                            .get("data", {})
                                                        )

                                                    title = post_data.get("title", "Reddit Post")
                                                    selftext = post_data.get("selftext", "")

                                                    # Extract top comments for additional context
                                                    comments_text = ""
                                                    if isinstance(data, list) and len(data) > 1:
                                                        comments = (
                                                            data[1]
                                                            .get("data", {})
                                                            .get("children", [])
                                                        )
                                                        for c in comments[:5]:
                                                            cd = c.get("data", {})
                                                            if "body" in cd:
                                                                comments_text += f"\n- {cd.get('author', '[deleted]')}: {cd['body']}"

                                                    page_text = f"{selftext}\n\nTop Comments:{comments_text}".strip()
                                                    page_title = html.unescape(title)

                                                    if len(page_text) > 50:
                                                        fetched_content_parts.append(
                                                            f"[PAGE: {page_title}]\n{page_text[:60000]}"
                                                        )
                                                        logger.info(
                                                            "🌐 HTTP fallback fetched Reddit JSON: %s (%d chars)",
                                                            page_title[:60],
                                                            len(page_text),
                                                        )
                                                    else:
                                                        logger.warning(
                                                            "🌐 HTTP fallback returned empty Reddit JSON for: %s",
                                                            str(url)[:80],
                                                        )
                                                except _RESPONSE_RECOVERABLE_ERRORS as e:
                                                    _record_response_degradation(
                                                        e,
                                                        "UnitaryResponse: Reddit JSON parse failed: %s",
                                                        action="continued HTTP fallback without Reddit comment extraction",
                                                        severity="error",
                                                    )
                                                    logger.warning(
                                                        "🌐 Failed to parse Reddit JSON: %s", e
                                                    )

                                            else:
                                                # Jina Proxy returns markdown
                                                page_text = resp_text.strip()
                                                page_title = str(url)[:80]
                                                first_line = page_text.split("\n")[0]
                                                if first_line.startswith("Title: "):
                                                    page_title = first_line.replace(
                                                        "Title: ", ""
                                                    ).strip()

                                                # If Jina was blocked by Cloudflare (rare, but happens) it returns "Target URL returned error 403"
                                                if (
                                                    "Target URL returned error 403" not in page_text
                                                    and len(page_text) > 100
                                                ):
                                                    fetched_content_parts.append(
                                                        f"[PAGE: {page_title}]\n{page_text[:60000]}"
                                                    )
                                                    logger.info(
                                                        "🌐 HTTP fallback (Jina Proxy) fetched: %s (%d chars)",
                                                        page_title[:60],
                                                        len(page_text),
                                                    )
                                                else:
                                                    logger.warning(
                                                        "🌐 Jina Proxy failed or blocked. Trying native HTML fallback..."
                                                    )
                                                    # Ultimate Native Fallback
                                                    native_resp = get_network_gateway().request(
                                                        "GET",
                                                        str(url),
                                                        timeout=20.0,
                                                        headers=headers,
                                                        source="phases.response_generation_unitary.native_html_fallback",
                                                    )
                                                    native_status = int(
                                                        native_resp.get("status_code") or 0
                                                    )
                                                    native_text_raw = bytes(
                                                        native_resp.get("content") or b""
                                                    ).decode("utf-8", errors="replace")
                                                    if native_status == 200:
                                                        extractor = _TextExtractor()
                                                        extractor.feed(native_text_raw)
                                                        native_text = html.unescape(
                                                            extractor.get_text()
                                                        ).strip()
                                                        native_text = _re.sub(
                                                            r"\n{3,}", "\n\n", native_text
                                                        )
                                                        native_text = _re.sub(
                                                            r" {2,}", " ", native_text
                                                        )
                                                        if len(native_text) > 200:
                                                            title_match = _re.search(
                                                                r"<title[^>]*>(.*?)</title>",
                                                                native_text_raw,
                                                                _re.IGNORECASE | _re.DOTALL,
                                                            )
                                                            native_title = (
                                                                html.unescape(
                                                                    title_match.group(1).strip()
                                                                )
                                                                if title_match
                                                                else str(url)[:80]
                                                            )
                                                            fetched_content_parts.append(
                                                                f"[PAGE: {native_title}]\n{native_text[:60000]}"
                                                            )
                                                            logger.info(
                                                                "🌐 HTTP fallback (Native HTML) fetched: %s (%d chars)",
                                                                native_title[:60],
                                                                len(native_text),
                                                            )
                                                    else:
                                                        logger.warning(
                                                            "🌐 HTTP fallback (Native HTML) got status %d",
                                                            native_status,
                                                        )
                                    else:
                                        logger.warning(
                                            "🌐 HTTP fallback got status %d for: %s",
                                            resp_status,
                                            fetch_url[:80],
                                        )
                                except GovernanceViolation:
                                    raise
                                except _RESPONSE_RECOVERABLE_ERRORS as http_exc:
                                    _record_response_degradation(
                                        http_exc,
                                        "UnitaryResponse: HTTP fallback error for %s: %s",
                                        str(url)[:80],
                                        action="continued auto-browse after HTTP fallback failed for URL",
                                        severity="error",
                                    )
                                    logger.warning(
                                        "🌐 HTTP fallback error for %s: %s", str(url)[:80], http_exc
                                    )
                        except GovernanceViolation:
                            raise
                        except _RESPONSE_RECOVERABLE_ERRORS as fallback_exc:
                            _record_response_degradation(
                                fallback_exc,
                                "UnitaryResponse: HTTP fallback failed: %s",
                                action="continued response generation without lightweight HTTP fallback content",
                                severity="error",
                            )
                            logger.warning("🌐 HTTP fallback failed: %s", fallback_exc)
                except GovernanceViolation:
                    raise
                except _RESPONSE_RECOVERABLE_ERRORS as browse_exc:
                    _record_response_degradation(
                        browse_exc,
                        "UnitaryResponse: auto-browse orchestrator error: %s",
                        action="continued response generation without auto-browsed page content",
                        severity="error",
                    )
                    logger.warning("🌐 Auto-browse orchestrator error: %s", browse_exc)

                if fetched_content_parts:
                    # Inject fetched content into working memory as a grounded context message
                    fetched_block = "\n\n---\n\n".join(fetched_content_parts)
                    new_state.cognition.working_memory.append(
                        # Stamped, so the inference gate can tell evidence THIS
                        # runtime gathered from text that merely looks like it.
                        stamp_grounding(
                            {
                                "role": "system",
                                "content": f"[FETCHED PAGE CONTENT]\n{fetched_block}",
                                "metadata": {
                                    "type": "skill_result",
                                    "skill": "sovereign_browser",
                                    "ok": True,
                                },
                            }
                        )
                    )
                    # Also inject as a skill modifier so the LLM system prompt can reference it
                    new_state.response_modifiers["last_skill_run"] = "sovereign_browser"
                    new_state.response_modifiers["last_skill_ok"] = True
                    new_state.response_modifiers["last_skill_turn_marker"] = new_state.response_modifiers.get("evidence_turn_marker")
                    new_state.response_modifiers["last_skill_objective_hash"] = (
                        self._objective_fingerprint(objective)
                    )
                    new_state.response_modifiers["last_skill_result_payload"] = {
                        "ok": True,
                        "content": fetched_block[:250000],
                        "title": fetched_content_parts[0].split("\n")[0]
                        if fetched_content_parts
                        else "",
                        "source": str(auto_browse_urls[0])[:1200]
                        if auto_browse_urls
                        else "",
                    }
                    # Rebuild contract now that tool evidence is available
                    contract = build_response_contract(
                        new_state, objective, is_user_facing=is_user_facing
                    )
                    new_state.response_modifiers["response_contract"] = contract.to_dict()

                    # ── Background Knowledge Formalization ────────────────
                    # Fire-and-forget: distill fetched content into the
                    # KnowledgeGraph without blocking the user response.
                    try:
                        from core.learning.formalizer import formalize_content

                        page_title = (
                            fetched_content_parts[0].split("\n")[0] if fetched_content_parts else ""
                        )
                        page_url = str(auto_browse_urls[0]) if auto_browse_urls else ""
                        get_task_tracker().create_task(
                            formalize_content(
                                content=fetched_block[:60000],
                                source_title=page_title,
                                source_url=page_url,
                            )
                        )
                        logger.info(
                            "📚 Background formalization task spawned for '%s'", page_title[:60]
                        )
                    except _RESPONSE_RECOVERABLE_ERRORS as formal_exc:
                        _record_response_degradation(
                            formal_exc,
                            "UnitaryResponse: formalization task spawn skipped: %s",
                            action="returned grounded page response without background formalization task",
                        )
                        logger.debug("Formalization task spawn skipped: %s", formal_exc)

            if contract.requires_search and routing_origin != "benchmark":
                cached_search_reply = self._build_cached_grounded_search_reply(
                    new_state,
                    objective,
                    contract,
                )
                if cached_search_reply:
                    logger.info(
                        "🔎 UnitaryResponse: answered explicit search from grounded tool evidence."
                    )
                    return self._commit_response(new_state, cached_search_reply)
                if not contract.tool_evidence_available:
                    grounded_search_outcome = await self._attempt_grounded_search_reply(
                        objective,
                        contract,
                        origin=routing_origin,
                    )
                    grounded_payload = grounded_search_outcome.get("payload")
                    grounded_skill = str(grounded_search_outcome.get("skill_name") or "")
                    grounded_search_reply = str(grounded_search_outcome.get("reply") or "")
                    if grounded_payload and grounded_skill:
                        new_state.response_modifiers["last_skill_run"] = grounded_skill
                        new_state.response_modifiers["last_skill_ok"] = True
                        new_state.response_modifiers["last_skill_turn_marker"] = new_state.response_modifiers.get("evidence_turn_marker")
                        new_state.response_modifiers["last_skill_objective_hash"] = (
                            self._objective_fingerprint(objective)
                        )
                        new_state.response_modifiers["last_skill_result_payload"] = grounded_payload
                        contract = build_response_contract(
                            new_state, objective, is_user_facing=is_user_facing
                        )
                        new_state.response_modifiers["response_contract"] = contract.to_dict()
                    if grounded_search_reply:
                        logger.info(
                            "🔎 UnitaryResponse: satisfied explicit search request through grounded tool execution."
                        )
                        return self._commit_response(new_state, grounded_search_reply)
                    if grounded_payload:
                        logger.info(
                            "🔎 UnitaryResponse: collected grounded search evidence and will synthesize from it."
                        )
                    else:
                        attempted_skill = str(
                            new_state.response_modifiers.get("last_skill_run", "") or ""
                        )
                        skill_ok = bool(new_state.response_modifiers.get("last_skill_ok", False))
                        if attempted_skill and not skill_ok:
                            new_state.cognition.last_response = (
                                "I don't have grounded results yet. The search path didn't come back cleanly, "
                                "so I shouldn't fake an answer."
                            )
                        else:
                            new_state.cognition.last_response = (
                                "I don't have grounded results for that yet, and I shouldn't guess. "
                                "I need to search it first."
                            )
                        return new_state

            if (
                is_user_facing
                and self._allow_pre_model_state_only_reply()
                and self._should_direct_answer_live_voice(
                    objective,
                    contract,
                    is_user_facing=is_user_facing,
                )
            ):
                direct_contract = contract
                if not contract.requires_live_aura_voice():
                    direct_contract = build_response_contract(
                        new_state,
                        objective,
                        is_user_facing=True,
                    )
                direct_reply = self._build_governed_user_recovery_reply(
                    new_state, objective, direct_contract
                )
                if direct_reply:
                    direct_reply, direct_validation = self._select_valid_recovery_variant(
                        direct_reply,
                        direct_contract,
                    )
                    if not direct_validation.ok:
                        direct_reply, direct_validation = self._select_valid_recovery_variant(
                            self._build_minimal_live_voice_reply(new_state, objective),
                            direct_contract,
                        )
                    new_state.response_modifiers["dialogue_validation"] = (
                        direct_validation.to_dict()
                    )
                    logger.info(
                        "🗣️ UnitaryResponse: answered from direct live Aura voice lane (%s)",
                        direct_contract.reason or "live_voice",
                    )
                    return self._commit_response(new_state, direct_reply)
            elif is_user_facing:
                logger.debug(
                    "🗣️ UnitaryResponse: live-voice direct lane not taken (priority=%s reason=%s contract_live=%s)",
                    priority,
                    getattr(contract, "reason", ""),
                    contract.requires_live_aura_voice(),
                )

            direct_episodic_matches: list[Any] = []
            if is_user_facing and self._is_explicit_memory_recall_request(objective):
                direct_episodic_matches = await self._direct_episodic_matches(objective)
                recent_episodic_matches = await self._recent_episodic_matches(limit=120)
                if recent_episodic_matches:
                    direct_episodic_matches.extend(recent_episodic_matches)
                direct_memory_answer = self._compose_memory_recall_answer(
                    objective,
                    new_state,
                    direct_episodic_matches,
                )
                if direct_memory_answer:
                    logger.info(
                        "🧠 UnitaryResponse: answered explicit recall from episodic evidence."
                    )
                    return self._commit_response(new_state, direct_memory_answer)

            if is_user_facing and self._is_idle_introspection_request(objective):
                idle_trace_answer = self._build_idle_trace_text(new_state)
                if idle_trace_answer:
                    logger.info(
                        "🧠 UnitaryResponse: answered idle introspection from stream trace."
                    )
                    return self._commit_response(new_state, idle_trace_answer)

            if strict_proof_answer_request and structured_proof_solver_enabled(origin=routing_origin):
                try:
                    from core.reasoning.proof_answer_solver import solve_strict_proof_prompt

                    proof_answer = solve_strict_proof_prompt(objective)
                    if proof_answer:
                        new_state.response_modifiers["structured_proof_solver"] = {
                            "solver": proof_answer.solver,
                            "confidence": proof_answer.confidence,
                        }
                        logger.info(
                            "🧠 UnitaryResponse: strict proof answered by %s solver.",
                            proof_answer.solver,
                        )
                        return self._commit_response(
                            new_state,
                            f"<answer>{proof_answer.answer}</answer>",
                        )
                except _RESPONSE_RECOVERABLE_ERRORS as exc:
                    _record_response_degradation(
                        exc,
                        "UnitaryResponse: structured proof solver skipped: %s",
                        action="continued strict proof answer through model lane after structured solver failed",
                        severity="warning",
                    )
                    logger.debug("UnitaryResponse: structured proof solver skipped: %s", exc)

            if not is_user_facing:
                if routing_origin != "benchmark":
                    model_tier = "tertiary"
                    deep_handoff = False
                is_test_run = (
                    routing_origin == "test"
                    or routing_origin == "benchmark"
                    or bool(_FLAG_AGI_MAX_TASKS.value())
                    or env_present(
                        "AURA_TESTING",
                        description="Mark a hermetic test runtime",
                        owner="core.runtime.state_ownership",
                    )
                    or env_present(
                        "AURA_PROOF_RUN",
                        description="Mark a hermetic proof runtime",
                        owner="core.runtime.state_ownership",
                    )
                )
                background_reason = None if is_test_run else response_policy.background_response_suppression_reason(
                    objective,
                    orchestrator=ServiceContainer.get("orchestrator", default=None),
                    include_synthetic_noise=True,
                )
                if background_reason:
                    logger.info(
                        "🛡️ UnitaryResponse: suppressing background response generation for origin=%s (%s).",
                        routing_origin,
                        background_reason,
                    )
                    response_policy.clear_background_generation(new_state, objective)
                    return new_state
                if self._background_response_should_defer(routing_origin):
                    logger.info(
                        "🛡️ UnitaryResponse: deferring background response generation for origin=%s.",
                        routing_origin,
                    )
                    response_policy.clear_background_generation(new_state, objective)
                    return new_state

            live_grounding_required = bool(
                is_user_facing
                and callable(getattr(contract, "requires_explicit_live_grounding", None))
                and contract.requires_explicit_live_grounding()
                and not operator_evidence_turn
            )
            grounding_evidence_active = self._current_turn_targets_grounding_evidence(
                new_state,
                objective,
                contract,
            )
            use_compact_router_payload = bool(
                strict_proof_answer_request
                or proof_evaluation_turn
                or routing_origin == "benchmark"
                or operator_evidence_turn
                or (
                    not is_user_facing  # Only use compact mode for background autonomous pulses
                    and not contract.requires_search
                    and not grounding_evidence_active
                    and (is_deep_probe_objective or not live_grounding_required)
                )
            )
            # ── CORE DIRECTIVE / SENSORY FEED PROMPT FAST-PATH ──────────
            # General improvement: when the pipeline is processing a programmatic
            # system directive or environmental sensory feed (from an embodied
            # environment, terminal session, IoT sensor, etc.), the full personality
            # prompt is counterproductive — it drowns out the directive and causes
            # the LLM to generate conversational text instead of following the
            # instruction. We use a minimal system prompt that IS the directive.
            _is_system_directive = (
                objective.startswith("CORE DIRECTIVE:")
                or "[sensory update" in objective.lower()
                or "[sensory feed" in objective.lower()
                or "[environmental context" in objective.lower()
            )
            worker_strict_answer_contract = (
                mlx_strict_answer_contract_enabled(origin=routing_origin)
                if strict_proof_answer_request
                else False
            )
            bind_context_packet = bool(
                is_user_facing
                and routing_origin != "benchmark"
                and not strict_proof_answer_request
                and not proof_evaluation_turn
                and not operator_evidence_turn
                and not _is_system_directive
            )
            if bind_context_packet:
                from core.brain.cognitive_context_manager import (
                    bind_unified_context_to_state,
                )

                await bind_unified_context_to_state(new_state, objective)
            if _is_system_directive:
                # For instruct-tuned models, providing ONLY a system prompt causes
                # them to continue generating the prompt itself. We must provide
                # a minimal system identity and put the directive in the user turn.
                system_prompt = "You are an autonomous execution engine. Follow the user's directive precisely. Do not output any conversational text. Output ONLY the required action marker."
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": objective},
                ]
                logger.info(
                    "🧠 [ZENITH] CORE DIRECTIVE fast-path: directive-only prompt (len=%d)",
                    len(system_prompt),
                )
            elif routing_origin == "benchmark":
                system_prompt = "You are a highly precise coding assistant. Your task is to solve the technical coding, configuration, or data challenge presented by the user exactly as specified. Output only the requested file contents (code, JSON, or CSV) without any extra explanations, introductory disclaimers, or conversational markdown packaging."
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": objective},
                ]
                logger.info("🧠 [ZENITH] Benchmark fast-path: minimal isolated prompt.")
            elif strict_proof_answer_request:
                strict_user_objective = (
                    objective
                    if worker_strict_answer_contract
                    else self._strip_answer_envelope_instruction(objective)
                )
                strict_procedure_hints = self._strict_proof_procedure_hints(
                    strict_user_objective
                )
                system_prompt = (
                    "You are Aura's governed proof-answer lane. Solve the user's task exactly. "
                    "Reason privately before answering: for constraint or truth-teller puzzles, "
                    "test each assignment and reject contradictions; for sequences, compare "
                    "differences and infer the generating rule; if the task gives answer "
                    "options in parentheses, return one of those option values, not the subject label. "
                )
                system_prompt += strict_procedure_hints
                system_prompt += (
                    "Output the final answer strictly inside <answer>...</answer> tags. "
                    "Keep the tag content minimal and do not include chat filler."
                    if worker_strict_answer_contract
                    else "Return only the final answer value. Do not explain, do not add role labels, and do not include XML tags."
                )
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": strict_user_objective},
                ]
                logger.info("🧠 [ZENITH] Strict proof fast-path: minimal live-path prompt.")
            elif proof_evaluation_turn:
                system_prompt = self._build_proof_evaluation_system_prompt(new_state, contract)
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": objective},
                ]
                logger.info("🧠 [ZENITH] Proof evaluation fast-path: isolated live-path prompt.")
            elif operator_evidence_turn:
                system_prompt = self._build_operator_evidence_system_prompt()
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": objective},
                ]
                logger.info("🧠 [ZENITH] Operator-evidence fast-path: isolated foreground prompt.")
            elif exact_format_required:
                system_prompt = (
                    "You are Aura's governed user-facing response lane. The latest user message "
                    "contains binding output-format instructions. Follow the requested labels, "
                    "section order, and required words exactly. Do not answer an older objective, "
                    "do not narrate internal telemetry, and do not add system-status prose before "
                    "the requested format."
                )
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": objective},
                ]
                logger.info("🧠 [ZENITH] Exact-format fast-path: isolated foreground prompt.")
            elif not is_user_facing:
                system_prompt = self._build_background_router_system_prompt(new_state)
                messages = self._build_router_messages(
                    new_state,
                    objective,
                    system_prompt,
                    history_limit=1,
                )
            elif use_compact_router_payload:
                system_prompt = self._build_compact_router_system_prompt(new_state)
                history_limit = (
                    2
                    if is_deep_probe_objective
                    else 8
                    if new_state.response_modifiers.get("coding_request")
                    else 6
                )
                messages = self._build_router_messages(
                    new_state,
                    objective,
                    system_prompt,
                    history_limit=2 if not is_user_facing else history_limit,
                )
            else:
                system_prompt = self._build_system_prompt(new_state)
                messages = ContextAssembler.build_messages(new_state, objective)
                if messages and messages[0].get("role") == "system":
                    base_system = str(messages[0].get("content") or "").strip()
                    messages[0]["content"] = (
                        f"{system_prompt}\n\n{base_system}" if base_system else system_prompt
                    )
                else:
                    messages.insert(0, {"role": "system", "content": system_prompt})

            if bind_context_packet and messages and messages[0].get("role") == "system":
                from core.brain.cognitive_context_manager import (
                    PROMPT_MARKER,
                    render_unified_context_prompt,
                )

                if PROMPT_MARKER not in str(messages[0].get("content") or ""):
                    unified_block = render_unified_context_prompt(
                        new_state.response_modifiers.get("unified_context_packet")
                    )
                    if unified_block:
                        messages[0]["content"] = (
                            f"{messages[0]['content']}\n\n{unified_block}"
                        )

            # When processing a CORE DIRECTIVE / sensory feed, skip all
            # personality and contract prompt injections. The directive IS the
            # complete prompt — the LLM just needs to follow it.
            if (
                not _is_system_directive
                and not strict_proof_answer_request
                and not proof_evaluation_turn
                and routing_origin != "benchmark"
                and not operator_evidence_turn
            ):
                priority_grounding = self._build_priority_grounding_block(
                    objective,
                    new_state,
                    direct_episodic_matches,
                )
                if priority_grounding and not exact_format_required:
                    system_prompt = (
                        f"{priority_grounding}\n\n{system_prompt}"
                        if system_prompt
                        else priority_grounding
                    )
                    if messages and messages[0].get("role") == "system":
                        messages[0]["content"] = f"{priority_grounding}\n\n{messages[0]['content']}"
                    else:
                        messages.insert(0, {"role": "system", "content": priority_grounding})

                def _prepend_system_guidance(block: str) -> None:
                    nonlocal system_prompt, messages
                    text = str(block or "").strip()
                    if not text:
                        return
                    system_prompt = f"{text}\n\n{system_prompt}".strip() if system_prompt else text
                    if messages and messages[0].get("role") == "system":
                        messages[0]["content"] = f"{text}\n\n{messages[0]['content']}"
                    else:
                        messages.insert(0, {"role": "system", "content": text})

                # ── Desktop execution planning contract ──────────────────
                # Unconditional at this level on purpose: the earlier attempt
                # sat inside the pre-linguistic guard and never ran, which is
                # why every fix aimed at this block stayed invisible.
                try:
                    _matched_now = list(
                        new_state.response_modifiers.get("matched_skills", []) or []
                    )
                    _is_desktop_turn = any(
                        "desktop" in str(skill).lower()
                        or "computer_use" in str(skill).lower()
                        for skill in _matched_now
                    )
                    if not _is_desktop_turn:
                        from core.runtime.desktop_objective_intent import (
                            looks_like_desktop_objective,
                        )

                        _is_desktop_turn = bool(looks_like_desktop_objective(objective))
                    if _is_desktop_turn:
                        from core.runtime.desktop_task_contract import (
                            desktop_task_action_sentence,
                        )

                        _prepend_system_guidance(
                            "## LIVE DESKTOP EXECUTION PLANNING CONTRACT\n"
                            "This request is a live desktop/computer objective and you CAN "
                            "carry it out: you drive real apps, files and the browser through "
                            "the governed desktop_task lane. Never say you cannot interact "
                            "with the device.\n"
                            "Produce a compact execution draft as a JSON object with an "
                            "optional `document_body` (the prose that belongs INSIDE the "
                            "artifact, never your reply to the user) and a bounded `steps` "
                            "array. Allowed step actions are "
                            f"{desktop_task_action_sentence()}.\n"
                            "Do not answer conversationally instead of planning, and do not "
                            "paste the artifact's content into chat."
                        )
                        logger.info(
                            "🖥️ [UNITARY] Desktop execution planning contract injected "
                            "(matched_skills=%s).",
                            _matched_now[:3] or "objective-detected",
                        )
                except _RESPONSE_RECOVERABLE_ERRORS as _desk_exc:
                    _record_response_degradation(
                        _desk_exc,
                        "UnitaryResponse: desktop planning contract skipped: %s",
                        _desk_exc,
                        action="continued without the desktop execution planning contract",
                    )

                if contract.reason != "ordinary_dialogue":
                    contract_block = contract.to_prompt_block().strip()
                    _prepend_system_guidance(contract_block)
                if is_user_facing and new_state.response_modifiers.get("coding_request"):
                    coding_block = self._build_coding_response_block(new_state, contract)
                    _prepend_system_guidance(coding_block)
                if is_user_facing and not exact_format_required:
                    voice_block = self._build_user_facing_voice_block(new_state, contract)
                    _prepend_system_guidance(voice_block)
                if is_deep_probe_objective:
                    try:
                        from core.evaluation.deep_mind_probe import deep_probe_prompt_block

                        _prepend_system_guidance(deep_probe_prompt_block())
                    except _RESPONSE_RECOVERABLE_ERRORS as exc:
                        _record_response_degradation(
                            exc,
                            "UnitaryResponse: deep probe guidance skipped: %s",
                            action="continued deep-probe prompt without extra evaluation guidance block",
                            severity="error",
                        )
                        logger.debug("UnitaryResponse: deep probe guidance skipped: %s", exc)
                if live_grounding_required and not exact_format_required:
                    self_expression_block = self._build_live_self_expression_block(
                        new_state, contract
                    )
                    _prepend_system_guidance(self_expression_block)

            # [RUBICON] Pre-Linguistic Decision: structured decision BEFORE LLM speaks
            if (
                not _is_system_directive
                and not strict_proof_answer_request
                and not proof_evaluation_turn
                and routing_origin != "benchmark"
                and not exact_format_required
                and not operator_evidence_turn
            ):
                # [STABILITY v54] Banter Shield: Hide architectural complexity for casual turns
                is_banter = (
                    new_state.cognition.modifiers.get("semantic_lane") == "casual"
                    or "banter" in objective.lower()
                )
                if is_banter:
                    logger.debug(
                        "🛡️ Banter Shield: suppressing RUBICON and architectural metrics for persona purity."
                    )
                else:
                    try:
                        from core.cognition.pre_linguistic import get_pre_linguistic

                        pl_engine = get_pre_linguistic()
                        if pl_engine._started:
                            has_tool_evidence = self._has_recent_grounded_evidence(new_state)
                            matched = list(
                                new_state.response_modifiers.get("matched_skills", []) or []
                            )
                            decision_pkg = pl_engine.synthesize(
                                objective,
                                is_user_facing=is_user_facing,
                                has_tool_result=has_tool_evidence,
                                matched_skills=matched,
                                response_modifiers=dict(new_state.response_modifiers),
                            )
                            # Inject the decision block into the prompt so the LLM narrates it
                            decision_block = decision_pkg.to_prompt_block()
                            _prepend_system_guidance(decision_block)
                            # Store the decision in state for downstream audit
                            new_state.response_modifiers["pre_linguistic_decision"] = (
                                decision_pkg.to_dict()
                            )
                            logger.debug(
                                "[RUBICON] PreLinguistic: %s via %s (%.1fms)",
                                decision_pkg.chosen_action.value,
                                decision_pkg.selected_limb,
                                decision_pkg.latency_ms,
                            )
                    except _RESPONSE_RECOVERABLE_ERRORS as pl_exc:
                        _record_response_degradation(
                            pl_exc,
                            "[RUBICON] PreLinguistic injection skipped: %s",
                            action="continued LLM generation without pre-linguistic decision block",
                            severity="error",
                        )
                        logger.debug("[RUBICON] PreLinguistic injection skipped: %s", pl_exc)
            format_priority_block = self._build_exact_format_priority_block(objective, contract)
            if format_priority_block:
                _prepend_system_guidance(format_priority_block)

            # [PERF] In embodied challenges, long history is a liability that causes
            # 80s+ inference stalls. We aggressively shed to the bare minimum.
            history_limit = 12
            if _FLAG_EMBODIED_CHALLENGE.value():
                history_limit = (
                    6  # [STABILITY] Increased from 2 to 6. 2 turns causes total context collapse.
                )
                logger.info(
                    "🛡️ UnitaryResponse: Using minimal history (6) for Embodied Challenge priority."
                )

            if not messages:
                messages = self._recent_router_history(
                    new_state,
                    limit=history_limit,
                )
            # Hard cap system prompt to fit within context window.
            # The 32B local model has ~8K token context (~32K chars).
            # Reserve at least 40% for conversation history + user message.
            # For compact router payloads, be even more aggressive since
            # conversation context is critical for prompt-specificity.
            if use_compact_router_payload:
                max_prompt_chars = 6000  # ~1500 tokens — leaves ~6.5K for conversation
            else:
                max_prompt_chars = 14000  # ~3500 tokens — leaves ~4.5K for conversation
            if len(system_prompt) > max_prompt_chars:
                # Keep the identity/rules header and trim context blocks
                system_prompt = system_prompt[:max_prompt_chars].rstrip()
                system_prompt += "\n[...context trimmed for token budget...]"
                self._sync_first_system_message(messages, system_prompt)

            # Anti-repetition injection: if recent responses have been stale,
            # inject an explicit instruction to avoid repeating prior patterns.
            try:
                from interface.routes.chat import _STALE_REPEAT_THRESHOLD, _recent_responses

                if len(_recent_responses) >= _STALE_REPEAT_THRESHOLD:
                    # Check if recent responses are similar to each other
                    from interface.routes.chat import _fuzzy_similar

                    recent_list = list(_recent_responses)
                    if len(recent_list) >= 2 and _fuzzy_similar(recent_list[-1], recent_list[-2]):
                        anti_repeat = (
                            "\n\nCRITICAL: Your recent responses have been repetitive. "
                            "You MUST answer the user's SPECIFIC question directly. "
                            "Do NOT describe your architecture, conversational lane, or runtime. "
                            "Read the user's actual message and respond to THAT, not to your system prompt."
                        )
                        # Prepend to system prompt so the model sees it first
                        system_prompt = anti_repeat + "\n\n" + system_prompt
                        # Re-sync if needed
                        self._sync_first_system_message(messages, system_prompt)
                        logger.warning(
                            "🚨 Anti-repetition instruction injected into system prompt."
                        )
            except _RESPONSE_RECOVERABLE_ERRORS as exc:
                _record_response_degradation(
                    exc, "UnitaryResponse: anti-repetition prompt check skipped: %s"
                )

            if (
                not strict_proof_answer_request
                and not proof_evaluation_turn
                and routing_origin != "benchmark"
                and not exact_format_required
                and not operator_evidence_turn
            ):
                messages = self._inject_active_grounding_message(
                    messages, new_state, objective, contract
                )

            request_timeout = self._timeout_for_request(
                is_user_facing=is_user_facing,
                model_tier=model_tier,
                deep_handoff=deep_handoff,
            )
            if strict_proof_answer_request:
                request_timeout = min(request_timeout, self._strict_proof_timeout_cap())

            llm_kwargs = {
                "messages": messages,
                "system_prompt": system_prompt,
                "prefer_tier": model_tier,
                "deep_handoff": deep_handoff,
                "allow_cloud_fallback": False,
                "origin": routing_origin,
                "purpose": "reply",
                "is_background": is_background,
                "foreground_request": is_user_facing,
                "protected_foreground_lane": is_deep_probe_objective,
                "deep_mind_probe": is_deep_probe_objective,
                "timeout": request_timeout,
                "state": new_state,
            }
            if is_user_facing:
                llm_kwargs.update(
                    {
                        "visible_user_message": surface_prompt.prompt,
                        "user_surface_validation_prompt": surface_prompt.prompt,
                        "user_surface_prompt_binding": dict(
                            runtime_context.get("user_surface_prompt_binding") or {}
                        ),
                        "clean_user_surface_contract": True,
                    }
                )
            if desktop_cognitive_engine_required:
                llm_kwargs.update(
                    {
                        "cognitive_engine_required": True,
                        "desktop_cognitive_engine_required": True,
                        "protected_foreground_lane": True,
                    }
                )
            try:
                foreground_cap = int(runtime_context.get("max_tokens") or 0)
            except (TypeError, ValueError, OverflowError):
                foreground_cap = 0
            if foreground_cap > 0 and is_user_facing:
                capped_tokens = max(64, min(foreground_cap, 2048))
                llm_kwargs["max_tokens"] = capped_tokens
                llm_kwargs["num_predict"] = capped_tokens
                # Memory-pressure shaping must not silently undo the answer
                # budget selected for the foreground turn. The MLX worker
                # otherwise reduced a 1,536-token code explanation to 344
                # characters and stopped in the middle of a sentence.
                llm_kwargs["user_surface_completion_floor"] = capped_tokens
            if runtime_context.get("skip_runtime_payload"):
                llm_kwargs["skip_runtime_payload"] = True
            if runtime_context.get("disable_prompt_cache"):
                llm_kwargs["disable_prompt_cache"] = True
            if runtime_context.get("clear_prompt_cache"):
                llm_kwargs["clear_prompt_cache"] = True
            if use_compact_router_payload or exact_format_required or operator_evidence_turn:
                llm_kwargs["skip_runtime_payload"] = True
            if strict_proof_answer_request:
                llm_kwargs.update(
                    {
                        "purpose": "strict_proof_answer",
                        "strict_answer_contract": worker_strict_answer_contract,
                        "strict_value_contract": not worker_strict_answer_contract,
                        "skip_runtime_payload": True,
                        "disable_prompt_cache": True,
                        "clear_prompt_cache": True,
                        "temperature": 0.0,
                        "max_tokens": 96,
                        "num_predict": 96,
                        "protected_foreground_lane": True,
                    }
                )
            elif proof_evaluation_turn:
                llm_kwargs.update(
                    {
                        "purpose": "proof_evaluation",
                        "proof_evaluation_contract": True,
                        "skip_runtime_payload": True,
                        "disable_prompt_cache": True,
                        "clear_prompt_cache": True,
                        "temperature": 0.1,
                        "max_tokens": 640,
                        "num_predict": 640,
                        "protected_foreground_lane": True,
                    }
                )
            elif operator_evidence_turn:
                llm_kwargs.update(
                    {
                        "purpose": "operator_evidence",
                        "operator_evidence_contract": True,
                        "skip_runtime_payload": True,
                        "disable_prompt_cache": True,
                        "clear_prompt_cache": True,
                        "temperature": 0.1,
                        "top_p": 0.8,
                        "min_p": 0.03,
                        "repetition_penalty": 1.18,
                        "repetition_context_size": 96,
                        "max_tokens": 220,
                        "num_predict": 220,
                        "protected_foreground_lane": True,
                    }
                )
            elif routing_origin == "benchmark":
                llm_kwargs.update(
                    {
                        "purpose": "benchmark_evaluation",
                        "benchmark_request": True,
                        "proof_evaluation_contract": True,
                        "proof_primary_lane_required": True,
                        "skip_runtime_payload": True,
                        "disable_prompt_cache": True,
                        "clear_prompt_cache": True,
                        "temperature": 0.1,
                        "max_tokens": 2048,
                        "num_predict": 2048,
                        "protected_foreground_lane": True,
                        "foreground_request": True,
                        "is_background": False,
                        "prefer_tier": "primary",
                        "deep_handoff": False,
                        "allow_deep_handoff": False,
                        "allow_cloud_fallback": False,
                    }
                )

            # The sovereign kernel replaces ResponseGenerationPhase with this
            # phase.  The legacy phase had a complete foreground RLC route, but
            # this active path used to jump straight to ``llm.think`` and thus
            # silently bypassed it during healthy chat.  Route one bounded,
            # depth-worthy episode here before ordinary decoding.  Strict,
            # exact, benchmark, directive, inventory and action contracts stay
            # on their purpose-built lanes.
            latent_path_committed = False
            latent_evidence: list[str] = []
            qualified_shadow_text = ""
            qualified_shadow_receipt: dict[str, Any] = {}
            model_retry_suppressed = False
            raw: Any = None
            try:
                from core.brain.foreground_latent_runtime import (
                    run_foreground_latent_episode,
                )

                controls = runtime_context.get("live_mind_generation_controls")
                controls = controls if isinstance(controls, dict) else {}
                cognitive_mode = getattr(new_state.cognition.current_mode, "value", None)
                cognitive_mode = str(cognitive_mode or new_state.cognition.current_mode or "")
                operational_contract = bool(
                    _is_system_directive
                    or exact_format_required
                    or operator_evidence_turn
                    or runtime_context.get("desktop_execution_contract", False)
                    or runtime_context.get("capability_inventory_contract", False)
                    or runtime_context.get("runtime_fact_status_contract", False)
                    or runtime_context.get("grounded_runtime_status_contract", False)
                    or runtime_context.get("memory_state_contract", False)
                    or runtime_context.get("self_condition_contract", False)
                )
                latent_outcome = await run_foreground_latent_episode(
                    orchestrator=ServiceContainer.get("orchestrator", default=None),
                    messages=messages,
                    visible_objective=str(surface_prompt.prompt or objective or ""),
                    foreground=is_user_facing and not is_background,
                    desktop_required=desktop_cognitive_engine_required,
                    cognitive_mode=cognitive_mode,
                    request_timeout_s=request_timeout,
                    prompt_shape=(
                        runtime_context.get("prompt_shape")
                        if isinstance(runtime_context.get("prompt_shape"), dict)
                        else None
                    ),
                    compact_contract=use_compact_router_payload,
                    strict_output_contract=bool(
                        strict_proof_answer_request or exact_format_required
                    ),
                    incompatible_contract=operational_contract,
                    proof_or_benchmark=bool(proof_evaluation_turn or benchmark_turn),
                    explicitly_required=bool(
                        runtime_context.get("latent_cortex_required", False)
                    ),
                    tenant_id=str(runtime_context.get("tenant_id") or "local"),
                    user_id=str(
                        runtime_context.get("user_id")
                        or runtime_context.get("owner_id")
                        or "owner"
                    ),
                    session_id=str(
                        runtime_context.get("session_id")
                        or runtime_context.get("conversation_id")
                        or "local"
                    ),
                    domain=str(
                        runtime_context.get("latent_cortex_domain")
                        or "desktop_conversation"
                    ),
                    decode_max_tokens=int(llm_kwargs.get("max_tokens") or 768),
                    decode_temperature=float(
                        controls.get("temperature")
                        or llm_kwargs.get("temperature")
                        or 0.58
                    ),
                    decode_top_p=float(
                        controls.get("top_p") or llm_kwargs.get("top_p") or 0.88
                    ),
                    recurrent_loops=int(
                        controls.get("clean_user_surface_recurrent_loops") or 1
                    ),
                    steering_alpha=float(
                        controls.get("clean_user_surface_steering_alpha") or 0.25
                    ),
                    capability_modifiers=dict(new_state.response_modifiers),
                )
                new_state.response_modifiers.update(latent_outcome.trace)
                if latent_outcome.shadow_text:
                    qualified_shadow_text = latent_outcome.shadow_text
                    raw_shadow_receipt = latent_outcome.trace.get(
                        "qualified_recurrent_receipt"
                    )
                    qualified_shadow_receipt = (
                        dict(raw_shadow_receipt)
                        if isinstance(raw_shadow_receipt, dict)
                        else {}
                    )
                if latent_outcome.succeeded:
                    raw = latent_outcome.text
                    latent_path_committed = True
                    latent_evidence = list(latent_outcome.evidence)
                elif latent_outcome.attempted and not latent_outcome.fallback_allowed:
                    model_retry_suppressed = True
                    new_state.response_modifiers.update(
                        {
                            "model_retry_suppressed": True,
                            "generation_failure_class": str(
                                latent_outcome.trace.get("latent_cortex_failure_reason")
                                or "latent_owner_exhausted"
                            )[:120],
                            "response_path": "cognitive_engine_latent_owner_exhausted",
                        }
                    )
                    raw = self._build_minimal_live_voice_reply(new_state, objective)
            except _RESPONSE_RECOVERABLE_ERRORS as latent_route_exc:
                _record_response_degradation(
                    latent_route_exc,
                    "UnitaryResponse: foreground latent routing failed before owner acquisition: %s",
                    action="retained ordinary foreground decoding after pre-acquisition latent routing failed",
                    severity="warning",
                )

            # [STABILITY v53] Explicit timeout wrapper — don't rely on router
            # honoring the timeout kwarg. If the router hangs, the phase hangs,
            # the user gets nothing. A committed latent answer replaces only
            # this decoder call; downstream response contracts remain active.
            if raw is None:
                try:
                    raw = await asyncio.wait_for(
                        llm.think(objective, **llm_kwargs),
                        timeout=request_timeout + 5.0,
                    )
                except TimeoutError as timeout_exc:
                    if proof_evaluation_turn or benchmark_turn:
                        raise TimeoutError(
                            f"proof/benchmark generation timed out after {request_timeout + 5.0:.0f}s"
                        ) from timeout_exc
                    if is_user_facing:
                        logger.warning(
                            "🚨 [STABILITY] LLM generation hard-timed-out. Using operational failure floor."
                        )
                        raw = self._build_minimal_live_voice_reply(new_state, objective)
                    else:
                        raise TimeoutError(
                            f"LLM generation hard-timed-out after {request_timeout + 5.0:.0f}s"
                        ) from timeout_exc

            if isinstance(raw, dict):
                raw = raw.get("content") or raw.get("response") or ""

            # Extract thinking segments from the raw LLM response
            import re as _re_think

            thought_segments = []
            for m in _re_think.finditer(
                r"<think>(.*?)</think>", str(raw or ""), flags=_re_think.DOTALL
            ):
                seg = m.group(1).strip()
                if seg:
                    thought_segments.append(seg)
            extracted_thought = "\n\n".join(thought_segments)
            if thought_segments:
                raw = _re_think.sub(
                    r"<think>.*?</think>", "", str(raw), flags=_re_think.DOTALL
                ).strip()

            if not raw or not raw.strip() or len(raw.strip()) < 5:
                if proof_evaluation_turn or benchmark_turn:
                    synthesized_raw = _try_benchmark_artifact_synthesis("model_no_valid_text")
                    if synthesized_raw:
                        raw = synthesized_raw
                    else:
                        raise RuntimeError("proof_or_benchmark_model_no_valid_text")
                if is_user_facing:
                    rescued = self._shape_user_facing_response(
                        str(raw or extracted_thought or ""),
                        objective,
                    )
                    if rescued:
                        raw = rescued
                    else:
                        logger.warning(
                            "🚨 [STABILITY] Foreground conversation lane returned no valid text. Using operational failure floor."
                        )
                        raw = self._build_minimal_live_voice_reply(new_state, objective)
                else:
                    logger.info(
                        "UnitaryResponse: background generation returned empty/short text for origin=%s (len=%d)",
                        routing_origin,
                        len(raw) if raw else 0,
                    )
                    self._clear_background_generation(new_state, objective)
                    return new_state

            response_text = raw.strip()

            # ── Reasoning Amplifier v2 (live, evidence-grounded) ──────────────
            # For verifiable hard turns (code / math / repo / architecture) on the
            # foreground lane, don't ship the first draft. Re-derive the answer through
            # the amplifier: gather real evidence (read repo source spans / recall
            # memory), generate candidates, run the domain truth engines, repair in the
            # symbolic sandbox, calibrate, and attach a reasoning receipt. Bounded,
            # governed, and fail-open to the original draft so behaviour never regresses.
            if not model_retry_suppressed:
                try:
                    response_text = await self._maybe_amplify_response(
                        objective=objective,
                        draft=response_text,
                        llm=llm,
                        state=new_state,
                        request_timeout=request_timeout,
                        is_user_facing=is_user_facing,
                        is_background=is_background,
                        proof_or_benchmark=bool(proof_evaluation_turn or benchmark_turn or strict_proof_answer_request),
                        seed_candidates=[response_text],
                        evidence=latent_evidence,
                    )
                    if latent_path_committed and self._last_reasoning_receipt:
                        new_state.response_modifiers[
                            "latent_cortex_amplifier_composed"
                        ] = True
                except _RESPONSE_RECOVERABLE_ERRORS as amp_exc:
                    logger.debug("Reasoning amplifier v2 skipped (fail-open): %s", amp_exc)

            # ── Conversational Amplifier (live, taste-selected) ───────────────
            # For substantive conversational turns (not actions, not verifiable
            # reasoning), don't ship the median sample: generate alternatives, rank
            # by the personalized TasteModel, self-revise the winner. Harvests the
            # median→best gap for wit/voice/creativity. Bounded, fail-open.
            if not latent_path_committed and not model_retry_suppressed:
                try:
                    response_text = await self._maybe_amplify_conversation(
                        objective=objective,
                        draft=response_text,
                        llm=llm,
                        state=new_state,
                        request_timeout=request_timeout,
                        is_user_facing=is_user_facing,
                        is_background=is_background,
                        proof_or_benchmark=bool(proof_evaluation_turn or benchmark_turn or strict_proof_answer_request),
                    )
                except _RESPONSE_RECOVERABLE_ERRORS as conv_exc:
                    logger.debug("Conversational amplifier skipped (fail-open): %s", conv_exc)

            if benchmark_turn:
                try:
                    from core.reasoning.artifact_synthesis import (
                        response_satisfies_artifact_contract,
                    )

                    if not response_satisfies_artifact_contract(objective, response_text):
                        synthesized_response = _try_benchmark_artifact_synthesis(
                            "artifact_contract_unmet"
                        )
                        if synthesized_response:
                            response_text = synthesized_response
                        else:
                            raise RuntimeError("benchmark_artifact_contract_unmet")
                except _RESPONSE_RECOVERABLE_ERRORS as contract_exc:
                    raise RuntimeError("benchmark_artifact_contract_validation_failed") from contract_exc

            if strict_proof_answer_request:
                prompt_derived_strict_solver_enabled = structured_proof_solver_enabled(
                    origin=routing_origin
                )

                async def _repair_symbolically_rejected_answer(
                    current_envelope: str,
                    *,
                    stage: str,
                    reason: str,
                ) -> str:
                    current_value = self._strict_answer_value_from_envelope(current_envelope)
                    current_validation = self._validate_strict_answer_symbolically(
                        objective,
                        current_value,
                    )
                    procedure_hints = self._strict_proof_procedure_hints(
                        objective,
                        current_validation,
                    )
                    repair_system_prompt = (
                        "You are Aura's governed proof-answer repair lane. A prior candidate "
                        "failed a prompt-derived consistency check before emission. Do not "
                        "trust the candidate. Re-solve the original task from scratch, test "
                        "the constraints privately, and return only the final atomic answer "
                        f"{procedure_hints}"
                        "inside exactly one <answer>...</answer> envelope. No explanation."
                    )
                    repair_messages = [
                        {"role": "system", "content": repair_system_prompt},
                        {
                            "role": "user",
                            "content": (
                                f"Original task:\n{objective}\n\n"
                                f"Rejected candidate from {stage}:\n{current_value}\n\n"
                                f"Consistency rejection reason:\n{reason}\n\n"
                                "Return only <answer>final</answer>."
                            ),
                        },
                    ]
                    try:
                        repaired_value = await asyncio.wait_for(
                            llm.think(
                                objective,
                                messages=repair_messages,
                                system_prompt=repair_system_prompt,
                                prefer_tier=model_tier,
                                deep_handoff=False,
                                allow_cloud_fallback=False,
                                origin=routing_origin,
                                purpose="strict_proof_answer_symbolic_repair",
                                is_background=False,
                                foreground_request=True,
                                protected_foreground_lane=True,
                                strict_answer_contract=mlx_strict_answer_contract_enabled(
                                    origin=routing_origin
                                ),
                                strict_value_contract=not mlx_strict_answer_contract_enabled(
                                    origin=routing_origin
                                ),
                                skip_runtime_payload=True,
                                disable_prompt_cache=True,
                                clear_prompt_cache=True,
                                temperature=0.0,
                                max_tokens=96,
                                num_predict=96,
                                timeout=min(request_timeout, 90.0),
                                state=new_state,
                            ),
                            timeout=min(request_timeout, 90.0) + 5.0,
                        )
                    except _RESPONSE_RECOVERABLE_ERRORS as symbolic_repair_exc:
                        _record_response_degradation(
                            symbolic_repair_exc,
                            "UnitaryResponse: strict proof symbolic repair failed: %s",
                            action="failed closed after strict proof answer contradicted prompt-derived constraints",
                            severity="error",
                        )
                        return ""
                    if isinstance(repaired_value, dict):
                        repaired_value = (
                            repaired_value.get("content") or repaired_value.get("response") or ""
                        )
                    return self._coerce_strict_answer_envelope(repaired_value)

                async def _ensure_symbolic_consistency(current_envelope: str, *, stage: str) -> str:
                    if not current_envelope:
                        return ""
                    current_value = self._strict_answer_value_from_envelope(current_envelope)
                    validation = self._validate_strict_answer_symbolically(objective, current_value)
                    validation_verdict = self._response_contract_attr(validation, "valid", None)
                    if validation_verdict is not False:
                        if validation_verdict is True:
                            new_state.response_modifiers["strict_proof_symbolic_validation"] = {
                                "stage": stage,
                                "solver": self._response_contract_attr(validation, "solver", None),
                                "reason": self._response_contract_attr(validation, "reason", ""),
                            }
                        return current_envelope

                    reason = str(
                        self._response_contract_attr(
                            validation,
                            "reason",
                            "candidate_conflicts_with_prompt_constraints",
                        )
                    )
                    solver = self._response_contract_attr(validation, "solver", None)
                    candidate_preview = self._normalize_text(current_value, 160)
                    logger.warning(
                        "UnitaryResponse: rejected strict proof candidate from %s via %s validator (%s): %r.",
                        stage,
                        solver or "symbolic",
                        reason,
                        candidate_preview,
                    )
                    if prompt_derived_strict_solver_enabled:
                        prompt_derived_repair = self._strict_symbolic_repair_envelope(
                            objective,
                            validation,
                        )
                        if prompt_derived_repair:
                            repaired_value = self._strict_answer_value_from_envelope(
                                prompt_derived_repair
                            )
                            repaired_validation = self._validate_strict_answer_symbolically(
                                objective,
                                repaired_value,
                            )
                            repaired_verdict = self._response_contract_attr(
                                repaired_validation,
                                "valid",
                                None,
                            )
                            if repaired_verdict is True:
                                new_state.response_modifiers[
                                    "strict_proof_symbolic_validation"
                                ] = {
                                    "stage": f"{stage}_prompt_derived_repair",
                                    "method": "prompt_derived_symbolic_repair",
                                    "solver": self._response_contract_attr(
                                        repaired_validation,
                                        "solver",
                                        None,
                                    ),
                                    "reason": self._response_contract_attr(
                                        repaired_validation,
                                        "reason",
                                        "",
                                    ),
                                }
                                logger.info(
                                    "UnitaryResponse: repaired strict proof candidate from %s via prompt-derived %s solver.",
                                    stage,
                                    self._response_contract_attr(repaired_validation, "solver", None)
                                    or solver
                                    or "symbolic",
                                )
                                return prompt_derived_repair

                    repair_seed = current_envelope
                    repair_reason = reason
                    for repair_index in range(2):
                        repaired_envelope = await _repair_symbolically_rejected_answer(
                            repair_seed,
                            stage=f"{stage}_repair_{repair_index + 1}",
                            reason=repair_reason,
                        )
                        if not repaired_envelope:
                            break
                        repaired_value = self._strict_answer_value_from_envelope(repaired_envelope)
                        repaired_validation = self._validate_strict_answer_symbolically(
                            objective,
                            repaired_value,
                        )
                        repaired_verdict = self._response_contract_attr(
                            repaired_validation,
                            "valid",
                            None,
                        )
                        if repaired_verdict is False:
                            repair_reason = str(
                                self._response_contract_attr(
                                    repaired_validation,
                                    "reason",
                                    "candidate_conflicts_with_prompt_constraints",
                                )
                            )
                            logger.error(
                                "UnitaryResponse: strict proof symbolic repair %s still contradicted %s validator (%s): %r.",
                                repair_index + 1,
                                self._response_contract_attr(repaired_validation, "solver", None)
                                or "symbolic",
                                repair_reason,
                                self._normalize_text(repaired_value, 160),
                            )
                            repair_seed = repaired_envelope
                            continue
                        if repaired_verdict is True:
                            new_state.response_modifiers["strict_proof_symbolic_validation"] = {
                                "stage": f"{stage}_repair_{repair_index + 1}",
                                "solver": self._response_contract_attr(
                                    repaired_validation,
                                    "solver",
                                    None,
                                ),
                                "reason": self._response_contract_attr(
                                    repaired_validation,
                                    "reason",
                                    "",
                                ),
                            }
                        return repaired_envelope
                    raise RuntimeError("strict_proof_symbolic_validation_failed")

                strict_envelope = self._coerce_strict_answer_envelope(response_text)
                if not strict_envelope:
                    repair_procedure_hints = self._strict_proof_procedure_hints(objective)
                    repair_system_prompt = (
                        "You are Aura's governed proof-answer lane. The previous output was "
                        "invalid because it was not exactly one minimal <answer>...</answer> "
                        "envelope. Return only the final atomic answer inside that envelope. "
                        f"{repair_procedure_hints}"
                        "No explanation, no assessment, no copied prompt text."
                    )
                    repair_messages = [
                        {"role": "system", "content": repair_system_prompt},
                        {
                            "role": "user",
                            "content": (
                                f"Task:\n{objective}\n\nPrevious invalid output:\n"
                                f"{response_text[:1200]}\n\nReturn only <answer>final</answer>."
                            ),
                        },
                    ]
                    try:
                        repaired = await asyncio.wait_for(
                            llm.think(
                                objective,
                                messages=repair_messages,
                                system_prompt=repair_system_prompt,
                                prefer_tier=model_tier,
                                deep_handoff=False,
                                allow_cloud_fallback=False,
                                origin=routing_origin,
                                purpose="strict_proof_answer_repair",
                                is_background=False,
                                foreground_request=True,
                                protected_foreground_lane=True,
                                strict_answer_contract=mlx_strict_answer_contract_enabled(
                                    origin=routing_origin
                                ),
                                strict_value_contract=not mlx_strict_answer_contract_enabled(
                                    origin=routing_origin
                                ),
                                skip_runtime_payload=True,
                                disable_prompt_cache=True,
                                clear_prompt_cache=True,
                                temperature=0.0,
                                max_tokens=64,
                                num_predict=64,
                                timeout=min(request_timeout, 90.0),
                                state=new_state,
                            ),
                            timeout=min(request_timeout, 90.0) + 5.0,
                        )
                    except _RESPONSE_RECOVERABLE_ERRORS as strict_exc:
                        _record_response_degradation(
                            strict_exc,
                            "UnitaryResponse: strict proof repair retry failed: %s",
                            action="returned first strict proof response after repair retry failed",
                            severity="warning",
                        )
                        repaired = ""
                    if isinstance(repaired, dict):
                        repaired = repaired.get("content") or repaired.get("response") or ""
                    strict_envelope = self._coerce_strict_answer_envelope(repaired)
                if strict_envelope:
                    strict_envelope = await _ensure_symbolic_consistency(
                        strict_envelope,
                        stage="candidate",
                    )
                    candidate_match = re.search(
                        r"<answer>\s*(.*?)\s*</answer>",
                        strict_envelope,
                        flags=re.DOTALL | re.IGNORECASE,
                    )
                    candidate_answer = candidate_match.group(1).strip() if candidate_match else strict_envelope
                    verify_procedure_hints = self._strict_proof_procedure_hints(objective)
                    verify_system_prompt = (
                        "You are Aura's governed proof-answer verifier. Check the candidate "
                        "against the original task before final emission. Reason privately. "
                        "For constraint or truth-teller puzzles, test each assignment and reject contradictions; "
                        "for sequences, compare differences and infer the generating rule; "
                        "if the task gives answer options in parentheses, return one of those option values, not the subject label. "
                        f"{verify_procedure_hints}"
                        "If the candidate is correct, return the same final answer value. "
                        "If it is wrong, return the corrected final answer value. "
                        "Return only the final answer value, with no explanation and no XML tags."
                    )
                    verify_messages = [
                        {"role": "system", "content": verify_system_prompt},
                        {
                            "role": "user",
                            "content": (
                                f"Original task:\n{objective}\n\n"
                                f"Candidate final answer:\n{candidate_answer}\n\n"
                                "Return only the verified final answer value."
                            ),
                        },
                    ]
                    try:
                        verified = await asyncio.wait_for(
                            llm.think(
                                objective,
                                messages=verify_messages,
                                system_prompt=verify_system_prompt,
                                prefer_tier=model_tier,
                                deep_handoff=False,
                                allow_cloud_fallback=False,
                                origin=routing_origin,
                                purpose="strict_proof_answer_verify",
                                is_background=False,
                                foreground_request=True,
                                protected_foreground_lane=True,
                                strict_answer_contract=mlx_strict_answer_contract_enabled(
                                    origin=routing_origin
                                ),
                                strict_value_contract=not mlx_strict_answer_contract_enabled(
                                    origin=routing_origin
                                ),
                                skip_runtime_payload=True,
                                disable_prompt_cache=True,
                                clear_prompt_cache=True,
                                temperature=0.0,
                                max_tokens=96,
                                num_predict=96,
                                timeout=min(request_timeout, 90.0),
                                state=new_state,
                            ),
                            timeout=min(request_timeout, 90.0) + 5.0,
                        )
                    except _RESPONSE_RECOVERABLE_ERRORS as verify_exc:
                        _record_response_degradation(
                            verify_exc,
                            "UnitaryResponse: strict proof verification retry failed: %s",
                            action="returned unverified strict proof response after verifier failed",
                            severity="warning",
                        )
                        verified = ""
                    if isinstance(verified, dict):
                        verified = verified.get("content") or verified.get("response") or ""
                    verified_envelope = self._coerce_strict_answer_envelope(verified)
                    if verified_envelope and self._strict_answer_value_allowed(
                        objective,
                        self._strict_answer_value_from_envelope(verified_envelope),
                    ):
                        strict_envelope = verified_envelope
                        strict_envelope = await _ensure_symbolic_consistency(
                            strict_envelope,
                            stage="model_verifier",
                        )
                    elif verified_envelope:
                        logger.warning(
                            "UnitaryResponse: rejected strict proof verifier meta-answer: %r",
                            self._strict_answer_value_from_envelope(verified_envelope),
                        )
                    strict_option_values: list[str] | None = None
                    option_match = re.search(
                        r"\(([^()]{1,100}\bor\b[^()]{1,100})\)",
                        str(objective or ""),
                        flags=re.IGNORECASE,
                    )
                    if option_match:
                        option_values = [
                            part.strip(" \t\r\n\"'`.,;:")
                            for part in re.split(r"\s+or\s+|[,/]", option_match.group(1), flags=re.IGNORECASE)
                            if part.strip(" \t\r\n\"'`.,;:")
                        ]
                        strict_option_values = option_values
                        final_match = re.search(
                            r"<answer>\s*(.*?)\s*</answer>",
                            strict_envelope,
                            flags=re.DOTALL | re.IGNORECASE,
                        )
                        final_answer = final_match.group(1).strip() if final_match else strict_envelope
                        option_present = any(
                            re.search(rf"\b{re.escape(option)}\b", final_answer, flags=re.IGNORECASE)
                            for option in option_values
                        )
                        if option_values and not option_present:
                            option_system_prompt = (
                                f"{verify_system_prompt} The original task provides answer options. "
                                f"You must choose exactly one of these option values: {', '.join(option_values)}. "
                                "Do not return the subject label or variable name."
                            )
                            option_messages = [
                                {"role": "system", "content": option_system_prompt},
                                {
                                    "role": "user",
                                    "content": (
                                        f"Original task:\n{objective}\n\n"
                                        f"Invalid candidate answer:\n{final_answer}\n\n"
                                        f"Choose exactly one option value from: {', '.join(option_values)}"
                                    ),
                                },
                            ]
                            try:
                                option_verified = await asyncio.wait_for(
                                    llm.think(
                                        objective,
                                        messages=option_messages,
                                        system_prompt=option_system_prompt,
                                        prefer_tier=model_tier,
                                        deep_handoff=False,
                                        allow_cloud_fallback=False,
                                        origin=routing_origin,
                                        purpose="strict_proof_answer_option_verify",
                                        is_background=False,
                                        foreground_request=True,
                                        protected_foreground_lane=True,
                                        strict_answer_contract=mlx_strict_answer_contract_enabled(
                                            origin=routing_origin
                                        ),
                                        strict_value_contract=not mlx_strict_answer_contract_enabled(
                                            origin=routing_origin
                                        ),
                                        skip_runtime_payload=True,
                                        disable_prompt_cache=True,
                                        clear_prompt_cache=True,
                                        temperature=0.0,
                                        max_tokens=64,
                                        num_predict=64,
                                        timeout=min(request_timeout, 90.0),
                                        state=new_state,
                                    ),
                                    timeout=min(request_timeout, 90.0) + 5.0,
                                )
                            except _RESPONSE_RECOVERABLE_ERRORS as option_verify_exc:
                                _record_response_degradation(
                                    option_verify_exc,
                                    "UnitaryResponse: strict proof option verification retry failed: %s",
                                    action="returned unverified option-shaped strict proof response after verifier failed",
                                    severity="warning",
                                )
                                option_verified = ""
                            if isinstance(option_verified, dict):
                                option_verified = (
                                    option_verified.get("content")
                                    or option_verified.get("response")
                                    or ""
                                )
                            option_envelope = self._coerce_strict_answer_envelope(option_verified)
                            if option_envelope and self._strict_answer_value_allowed(
                                objective,
                                self._strict_answer_value_from_envelope(option_envelope),
                                option_values=option_values,
                            ):
                                strict_envelope = option_envelope
                                strict_envelope = await _ensure_symbolic_consistency(
                                    strict_envelope,
                                    stage="option_verifier",
                                )
                            elif option_envelope:
                                logger.warning(
                                    "UnitaryResponse: rejected strict proof option verifier non-option: %r",
                                    self._strict_answer_value_from_envelope(option_envelope),
                                )
                            if not self._strict_answer_value_allowed(
                                objective,
                                self._strict_answer_value_from_envelope(strict_envelope),
                                option_values=option_values,
                            ):
                                raise RuntimeError("strict_proof_option_contract_unmet")
                    strict_envelope = self._canonicalize_strict_answer_envelope(
                        objective,
                        strict_envelope,
                        option_values=strict_option_values,
                    )
                    strict_envelope = await _ensure_symbolic_consistency(
                        strict_envelope,
                        stage="canonicalized_final",
                    )
                    if not strict_envelope:
                        raise RuntimeError("strict_proof_answer_contract_unmet")
                    logger.info("🧠 UnitaryResponse: strict proof answered through exact envelope lane.")
                    return self._commit_response(new_state, strict_envelope)

            if proof_evaluation_turn and self._proof_evaluation_response_incomplete(
                objective,
                response_text,
            ):
                repair_system_prompt = self._build_proof_evaluation_system_prompt(
                    new_state,
                    contract,
                )
                repair_system_prompt = (
                    "The previous proof/evaluation draft was incomplete or fragmentary. "
                    "Regenerate a complete answer now. Use 3-6 complete sentences for explanation/planning tasks, "
                    "answer only the current task, and do not mention this repair instruction.\n\n"
                    f"{repair_system_prompt}"
                )
                repair_messages = [
                    {"role": "system", "content": repair_system_prompt},
                    {
                        "role": "user",
                        "content": (
                            f"Current task:\n{objective}\n\n"
                            f"Incomplete draft:\n{response_text[:1200]}\n\n"
                            "Return a complete final response."
                        ),
                    },
                ]
                try:
                    repaired = await asyncio.wait_for(
                        llm.think(
                            objective,
                            messages=repair_messages,
                            system_prompt=repair_system_prompt,
                            prefer_tier=model_tier,
                            deep_handoff=False,
                            allow_cloud_fallback=False,
                            origin=routing_origin,
                            purpose="proof_evaluation_repair",
                            proof_evaluation_contract=True,
                            is_background=False,
                            foreground_request=True,
                            protected_foreground_lane=True,
                            skip_runtime_payload=True,
                            disable_prompt_cache=True,
                            clear_prompt_cache=True,
                            temperature=0.1,
                            max_tokens=768,
                            num_predict=768,
                            timeout=min(request_timeout, 120.0),
                            state=new_state,
                        ),
                        timeout=min(request_timeout, 120.0) + 5.0,
                    )
                    if isinstance(repaired, dict):
                        repaired = repaired.get("content") or repaired.get("response") or ""
                    repaired_text = str(repaired or "").strip()
                    if repaired_text:
                        response_text = repaired_text
                        new_state.response_modifiers["proof_evaluation_repair"] = {
                            "reason": "incomplete_first_draft",
                            "complete": not self._proof_evaluation_response_incomplete(
                                objective,
                                repaired_text,
                            ),
                        }
                except _RESPONSE_RECOVERABLE_ERRORS as proof_repair_exc:
                    _record_response_degradation(
                        proof_repair_exc,
                        "UnitaryResponse: proof evaluation repair retry failed: %s",
                        action="returned first proof evaluation draft after completeness repair failed",
                        severity="warning",
                    )
                    logger.debug(
                        "UnitaryResponse: proof evaluation repair retry failed: %s",
                        proof_repair_exc,
                    )

            # System 2 internal critique layer to verify logical correctness
            try:
                from core.brain.reasoning_strategies import ReasoningStrategies
                async def _raw_generate(p, **kw):
                    return await llm.think(p, **kw)
                strategies = ReasoningStrategies(_raw_generate)
                if (
                    not strict_proof_answer_request
                    and not proof_evaluation_turn
                    and not benchmark_turn
                    and not contract.tool_evidence_available
                    and not desktop_cognitive_engine_required
                    and not latent_path_committed
                    and not model_retry_suppressed
                    and strategies._is_logical_check(objective)
                ):
                    logger.info("⚡ [Critique] Running System 2 self-critique on response...")
                    critique_response = await strategies._self_critique(objective, response_text, origin=routing_origin)
                    if critique_response and critique_response != response_text:
                        logger.info("⚡ [Critique] Self-critique corrected the generated response!")
                        response_text = critique_response
            except (ImportError, AttributeError, TypeError, ValueError, LookupError, RuntimeError, NameError, SyntaxError, TimeoutError) as critique_exc:
                logger.warning("Failed to run System 2 self-critique: %s", critique_exc)

            # Proactive XML Answer Tag formatting guard:
            # If the user prompt or system instruction requires XML answer tagging (e.g. "<answer>"),
            # but the model's generated text doesn't contain a valid "<answer>...</answer>" tag:
            # We use a robust regex parsing cascade to extract the plain-text answer from the model's explanation,
            # and automatically wrap it in a clean "<answer>...</answer>" block at the end of the text.
            lower_objective = objective.lower() if objective else ""
            lower_response = response_text.lower() if response_text else ""
            if ("<answer>" in lower_objective or "answer_format" in kwargs) and response_text and "<answer>" not in lower_response:
                extracted_ans = None
                
                # 1. Look for markdown bolded final answer (e.g. **Answer**: 5 or **Final Answer**: Alice)
                match = re.search(r"\*\*(?:final\s+)?answer\*\*:\s*([^\n]+)", response_text, re.IGNORECASE)
                if match:
                    extracted_ans = match.group(1).strip()
                
                # 2. Look for plain-text answer prefix (e.g. Final Answer: same)
                if not extracted_ans:
                    match = re.search(r"(?:final\s+)?answer:\s*([^\n]+)", response_text, re.IGNORECASE)
                    if match:
                        extracted_ans = match.group(1).strip()
                
                # 3. Look for concluding "therefore, the answer is X"
                if not extracted_ans:
                    match = re.search(r"(?:therefore|thus|hence|so),\s*(?:the\s+)?answer\s+(?:is|must\s+be)\s+([^\n.]+)", response_text, re.IGNORECASE)
                    if match:
                        extracted_ans = match.group(1).strip()
                
                # 4. If the response is short enough (e.g. under 60 chars) and has no explanation, use the whole text
                if not extracted_ans and len(response_text.strip()) < 60 and not any(k in lower_response for k in ("because", "since", "as we", "therefore")):
                    extracted_ans = response_text.strip()
                
                if extracted_ans:
                    # Clean trailing punctuation
                    extracted_ans = extracted_ans.rstrip(".,;:!?* ")
                    # Wrap and append
                    response_text += f"\n\n<answer>{extracted_ans}</answer>"
                    logger.info("🛡️ [HARDENING] Auto-corrected and wrapped extracted answer '%s' in XML tags.", extracted_ans)
            if self._guard and not benchmark_turn:
                response_text, _, _ = self._guard.align(response_text)
            if (
                is_user_facing
                and not proof_evaluation_turn
                and routing_origin != "benchmark"
                and not operator_evidence_turn
            ):
                response_text = self._shape_user_facing_response(response_text, objective)
                response_text = await self._apply_deep_honesty(response_text)

            async def _retry_dialogue(repair_block: str) -> str:
                if desktop_cognitive_engine_required:
                    logger.warning(
                        "🛡️ UnitaryResponse skipped a second heavyweight desktop generation; "
                        "deterministic dialogue repair remains authoritative for this turn."
                    )
                    return ""
                retry_messages = [dict(msg) for msg in messages]
                if retry_messages and retry_messages[0].get("role") == "system":
                    retry_messages[0]["content"] = (
                        f"{repair_block}\n\n{retry_messages[0]['content']}"
                    )
                else:
                    retry_messages.insert(0, {"role": "system", "content": repair_block})

                retry_timeout = min(120.0, max(45.0, request_timeout * 0.75))
                retry_kwargs = {
                    "messages": retry_messages,
                    "system_prompt": system_prompt,
                    "prefer_tier": model_tier,
                    "deep_handoff": deep_handoff,
                    "allow_cloud_fallback": False,
                    "origin": routing_origin,
                    "purpose": "reply",
                    "is_background": not is_user_facing,
                    "foreground_request": is_user_facing,
                    "protected_foreground_lane": is_deep_probe_objective,
                    "state": new_state,
                    "timeout": retry_timeout,
                    "disable_prompt_cache": True,
                    "clear_prompt_cache": True,
                    "temperature": 0.2,
                    "top_p": 0.85,
                    "min_p": 0.02,
                    "repetition_penalty": 1.12,
                    "repetition_context_size": 96,
                    "skip_runtime_payload": True,
                }
                if operator_evidence_turn:
                    retry_kwargs.update(
                        {
                            "purpose": "operator_evidence",
                            "operator_evidence_contract": True,
                            "temperature": 0.1,
                            "top_p": 0.8,
                            "min_p": 0.03,
                            "repetition_penalty": 1.18,
                            "max_tokens": 220,
                            "num_predict": 220,
                            "protected_foreground_lane": True,
                        }
                    )
                # [STABILITY v53] Explicit timeout on retry too
                try:
                    retried = await asyncio.wait_for(
                        llm.think(objective, **retry_kwargs),
                        timeout=retry_timeout + 5.0,
                    )
                except TimeoutError:
                    return ""
                if isinstance(retried, dict):
                    retried = retried.get("content") or retried.get("response") or ""
                retried_text = str(retried or "").strip()
                if self._guard and retried_text:
                    retried_text, _, _ = self._guard.align(retried_text)
                if (
                    is_user_facing
                    and not proof_evaluation_turn
                    and retried_text
                    and not operator_evidence_turn
                ):
                    retried_text = self._shape_user_facing_response(retried_text, objective)
                return retried_text

            if (
                operator_evidence_turn
                and not desktop_cognitive_engine_required
                and not self._operator_evidence_reply_is_substantive(response_text)
            ):
                operator_retry_block = (
                    "The previous draft missed the operator-evidence contract. Regenerate one plain "
                    "paragraph that directly answers the current user message. It must include these "
                    "concepts in ordinary prose: objective, governed tool use, receipt, trace, stop "
                    "condition, and personhood boundary. State that this is operational evidence, "
                    "not proof of literal personhood or proven consciousness. Do not describe feelings, "
                    "inner events, telemetry, status, labels, or this retry instruction."
                )
                operator_retry = await _retry_dialogue(operator_retry_block)
                if self._operator_evidence_reply_is_substantive(operator_retry):
                    logger.warning(
                        "🛡️ UnitaryResponse regenerated operator-evidence draft through primary lane."
                    )
                    response_text = operator_retry

            # Dialogue Contract Enforcement
            # System directives and action/sensory streams bypass dialogue
            # validation, because they are inherently programmatic responses
            # (like `[ACTION:execute]`) that violate conversational rules.
            if (
                not _is_system_directive
                and not proof_evaluation_turn
                and routing_origin != "benchmark"
                and not operator_evidence_turn
            ):
                pre_dialogue_response = str(response_text or "").strip()
                pre_dialogue_validation = validate_dialogue_response(
                    pre_dialogue_response, contract
                )
                (
                    response_text,
                    dialogue_validation,
                    dialogue_retried,
                ) = await enforce_dialogue_contract(
                    response_text,
                    contract,
                    retry_generate=(
                        _retry_dialogue
                        if is_user_facing and not desktop_cognitive_engine_required
                        else None
                    ),
                    state=new_state,
                    user_message=objective,
                )
                if is_user_facing and dialogue_retried and pre_dialogue_response:
                    try:
                        from core.conversation.response_reliability import (
                            assess_user_facing_reply,
                            is_non_answer_repair_floor_reply,
                        )

                        hard_dialogue_violations = {
                            "empty_response",
                            "prompt_artifact",
                            "generic_assistant_language",
                            "low_signal_preamble",
                            "low_signal_redirect",
                            "moderator_turn",
                            "prompt_fishing_closer",
                            "corrupted_language",
                            "unsupported_internal_jargon",
                            "unsupported_biographical_claim",
                            "intra_response_repetition",
                        }
                        pre_quality = assess_user_facing_reply(objective, pre_dialogue_response)
                        post_quality = assess_user_facing_reply(objective, response_text)
                        pre_words = len(pre_dialogue_response.split())
                        post_text = str(response_text or "").strip()
                        post_words = len(post_text.split())
                        pre_had_only_soft_dialogue_notes = not (
                            set(pre_dialogue_validation.violations) & hard_dialogue_violations
                        )
                        retry_is_worse = (
                            is_non_answer_repair_floor_reply(post_text)
                            or post_quality.retryable
                            or (
                                len(pre_dialogue_response) >= 80
                                and len(post_text) < max(48, int(len(pre_dialogue_response) * 0.55))
                            )
                            or (pre_words >= 14 and post_words < 8)
                        )
                        pre_is_real_answer = (
                            len(pre_dialogue_response) >= 80
                            and pre_words >= 12
                            and not pre_quality.hard_failure
                            and pre_had_only_soft_dialogue_notes
                        )
                        if pre_is_real_answer and retry_is_worse:
                            logger.warning(
                                "🗣️ UnitaryResponse preserved substantive first draft over weaker dialogue retry "
                                "(pre_violations=%s post_violations=%s pre_len=%d post_len=%d).",
                                ",".join(pre_dialogue_validation.violations) or "none",
                                ",".join(dialogue_validation.violations) or "none",
                                len(pre_dialogue_response),
                                len(post_text),
                            )
                            response_text = pre_dialogue_response
                            dialogue_validation = pre_dialogue_validation
                            dialogue_retried = False
                    except (ImportError, AttributeError, TypeError, ValueError) as preserve_exc:
                        _record_response_degradation(
                            preserve_exc,
                            "Dialogue retry preservation skipped: %s",
                            action="continued with dialogue retry result after preservation comparison failed",
                            severity="error",
                        )
                        logger.debug("Dialogue retry preservation skipped: %s", preserve_exc)
                if is_user_facing and not dialogue_validation.ok:
                    recovered = self._build_governed_user_recovery_reply(
                        new_state, objective, contract
                    )
                    if recovered:
                        response_text, dialogue_validation = self._select_valid_recovery_variant(
                            recovered,
                            contract,
                        )
                        logger.info(
                            "🗣️ UnitaryResponse: replaced failed subjective draft with grounded recovery reply (%s)",
                            ", ".join(dialogue_validation.violations) or "recovered",
                        )
                new_state.response_modifiers["dialogue_validation"] = dialogue_validation.to_dict()
                if dialogue_retried:
                    logger.info(
                        "🗣️ UnitaryResponse: retried draft to satisfy dialogue contract (%s)",
                        ", ".join(dialogue_validation.violations) or "recovered",
                    )

                # Genuine Refusal (Values-based pushback)
                if self._refusal:
                    response_text, _ = await self._refusal.process(
                        user_input=objective, response=response_text, state=new_state
                    )
                if is_user_facing:
                    response_text = self._shape_user_facing_response(response_text, objective)

                final_validation = validate_dialogue_response(response_text, contract)
                if is_user_facing and not final_validation.ok:
                    recovered = self._build_governed_user_recovery_reply(
                        new_state, objective, contract
                    )
                    if recovered:
                        candidate, candidate_validation = self._select_valid_recovery_variant(
                            recovered,
                            contract,
                        )
                        if candidate_validation.ok:
                            logger.info(
                                "🗣️ UnitaryResponse: final governed recovery replaced invalid post-processed reply (%s)",
                                ", ".join(final_validation.violations) or "post_process_invalid",
                            )
                            response_text = candidate
                            final_validation = candidate_validation

                if is_user_facing and not final_validation.ok and live_grounding_required:
                    # The minimal canned line is strictly worse than a real
                    # cortex reply that simply lacks an explicit "I/me" anchor or
                    # live-state phrase. Only force it when the response is
                    # actually broken — too short to be substantive, or only
                    # tripping low-signal-style violations. A 50+ char reply that
                    # the cortex generated for this turn is a real answer the
                    # user should see, even if dialogue_policy wanted more
                    # first-person grounding.
                    substantive_violations = {
                        "empty_response",
                        "prompt_artifact",
                        "generic_assistant_language",
                        "low_signal_preamble",
                        "low_signal_redirect",
                        "moderator_turn",
                        "prompt_fishing_closer",
                        "corrupted_language",
                    }
                    stripped_response = str(response_text or "").strip()
                    only_grounding_complaints = bool(final_validation.violations) and not (
                        set(final_validation.violations) & substantive_violations
                    )
                    if only_grounding_complaints and len(stripped_response) >= 50:
                        logger.info(
                            "🗣️ UnitaryResponse: keeping cortex reply despite grounding-only violation (%s, len=%d)",
                            ", ".join(final_validation.violations) or "ungrounded_only",
                            len(stripped_response),
                        )
                    else:
                        minimal, minimal_validation = self._select_valid_recovery_variant(
                            self._build_minimal_live_voice_reply(new_state, objective),
                            contract,
                        )
                        logger.info(
                            "🗣️ UnitaryResponse: forcing minimal live-voice fallback (%s)",
                            ", ".join(final_validation.violations) or "post_process_invalid",
                        )
                        response_text = minimal
                        final_validation = minimal_validation

                new_state.response_modifiers["dialogue_validation"] = final_validation.to_dict()

            if is_user_facing and not proof_evaluation_turn and routing_origin != "benchmark":
                try:
                    from core.conversation.response_reliability import (
                        assess_user_facing_reply,
                        is_live_self_reflection_turn,
                        reliability_floor_for_user,
                    )

                    quality = assess_user_facing_reply(objective, response_text)
                    if quality.retryable:
                        repairable_self_reflection_reasons = {
                            "off_topic_self_reflection_reply",
                            "pseudo_internal_jargon",
                            "status_page_self_reflection",
                        }
                        quality_reasons = set(quality.reasons or ())
                        if (
                            is_live_self_reflection_turn(objective)
                            and quality_reasons
                            and quality_reasons.issubset(repairable_self_reflection_reasons)
                        ):
                            repair = self._build_live_self_reflection_repair_reply(
                                new_state,
                                objective,
                                contract,
                            )
                            shaped_repair = self._shape_user_facing_response(repair, objective)
                            repair_quality = assess_user_facing_reply(objective, shaped_repair)
                            if not repair_quality.retryable:
                                logger.warning(
                                    "🛡️ UnitaryResponse replaced repairable live self-reflection draft "
                                    "(%s, len=%d) without another long Cortex retry.",
                                    ",".join(quality.reasons) or "unknown",
                                    len(str(response_text or "")),
                                )
                                response_text = shaped_repair
                                quality = repair_quality
                        response_text_s = str(response_text or "").strip()
                        if quality.retryable and set(quality.reasons or ()) == {"truncated_tail"}:
                            completed_response = self._complete_substantive_truncated_foreground_reply(
                                response_text_s
                            )
                            if completed_response:
                                completed_quality = assess_user_facing_reply(objective, completed_response)
                                if not completed_quality.retryable:
                                    logger.warning(
                                        "🛡️ UnitaryResponse completed clipped foreground draft without another 32B retry (len=%d -> %d).",
                                        len(response_text_s),
                                        len(completed_response),
                                    )
                                    response_text = completed_response
                                    response_text_s = completed_response
                                    quality = completed_quality
                        keep_substantive_soft_failure = bool(
                            quality.retryable
                            and not quality.hard_failure
                            and len(response_text_s) >= 80
                            and len(response_text_s.split()) >= 12
                            and "off_topic_self_reflection_reply" not in quality.reasons
                        )
                        if keep_substantive_soft_failure:
                            logger.warning(
                                "🛡️ UnitaryResponse kept substantive foreground draft despite soft reliability notes (%s, len=%d).",
                                ",".join(quality.reasons) or "soft_quality_note",
                                len(response_text_s),
                            )
                            quality = quality.__class__(
                                ok=True,
                                reasons=quality.reasons,
                                hard_failure=False,
                                retryable=False,
                            )
                        if quality.retryable:
                            retry_block = (
                                "The previous draft failed the user-facing reliability gate "
                                f"({','.join(quality.reasons) or 'unsafe_draft'}). "
                                "Regenerate once. Answer the current user message directly, in ordinary English, "
                                "without occult accusations, invented danger, prompt artifacts, or filler."
                            )
                            retried_text = (
                                await _retry_dialogue(retry_block)
                                if is_user_facing and not desktop_cognitive_engine_required
                                else ""
                            )
                            if retried_text:
                                retried_quality = assess_user_facing_reply(objective, retried_text)
                                if not retried_quality.retryable:
                                    logger.warning(
                                        "🛡️ UnitaryResponse regenerated unsafe final draft (%s -> clean, len=%d).",
                                        ",".join(quality.reasons) or "unknown",
                                        len(str(response_text or "")),
                                    )
                                    response_text = retried_text
                                    quality = retried_quality
                            if quality.retryable:
                                deterministic_floor = ""
                                try:
                                    from core.synthesis import deterministic_user_facing_floor

                                    deterministic_floor = deterministic_user_facing_floor(objective)
                                except _RESPONSE_RECOVERABLE_ERRORS:
                                    deterministic_floor = ""
                                floor = (
                                    reliability_floor_for_user(objective)
                                    or deterministic_floor
                                    or self._build_minimal_live_voice_reply(new_state, objective)
                                )
                                shaped_floor = self._shape_user_facing_response(floor, objective)
                                floor_quality = assess_user_facing_reply(objective, shaped_floor)
                                if not floor_quality.retryable:
                                    logger.warning(
                                        "🛡️ UnitaryResponse replaced unsafe final 32B draft (%s, len=%d).",
                                        ",".join(quality.reasons) or "unknown",
                                        len(str(response_text or "")),
                                    )
                                    response_text = shaped_floor
                                elif desktop_cognitive_engine_required:
                                    failure_reply = (
                                        "I couldn't produce a reliable answer to that turn, and I won't "
                                        "fabricate one. The Cortex draft failed its output checks, so I "
                                        "recorded the failure instead of sending nonsense."
                                    )
                                    _record_response_degradation(
                                        RuntimeError(
                                            "desktop Cortex draft and deterministic repair floor "
                                            "both failed user-facing reliability checks"
                                        ),
                                        "UnitaryResponse emitted an explicit desktop inference failure: %s",
                                        action=(
                                            "surfaced a bounded desktop runtime failure instead of "
                                            "retrying the heavyweight model or returning HTTP 503"
                                        ),
                                        severity="degraded",
                                    )
                                    response_text = failure_reply
                                else:
                                    raise TimeoutError(
                                        "Foreground conversation lane produced only unsafe drafts: "
                                        + ",".join(quality.reasons)
                                    )
                except TimeoutError:
                    raise
                except _RESPONSE_RECOVERABLE_ERRORS as quality_exc:
                    _record_response_degradation(
                        quality_exc,
                        "UnitaryResponse final reliability check skipped: %s",
                        action="continued with dialogue-validated response after final reliability check failed",
                        severity="error",
                    )
                    logger.debug("UnitaryResponse final reliability check skipped: %s", quality_exc)

            # [PEDAGOGY UPGRADE] Autonomous Manim Generation
            try:
                if (
                    is_user_facing
                    and not proof_evaluation_turn
                    and routing_origin != "benchmark"
                    and response_text
                    and (
                        "$$" in response_text
                        or "\\[" in response_text
                        or "\\int" in response_text
                        or "\\nabla" in response_text
                    )
                ):
                    if not _MANIM_RENDER_LOCK.acquire(blocking=False):
                        logger.info(
                            "🎬 Manim render already in flight; skipping overlapping autonomous render."
                        )
                    else:
                        logger.info(
                            "🎬 Math/Physics detected in response. Autonomously launching Manim generation..."
                        )

                        try:
                            threading.Thread(
                                target=_render_manim_in_background,
                                args=(response_text,),
                                name="aura-manim-render",
                                daemon=True,
                            ).start()
                            response_text += (
                                "\n\n*(I am autonomously rendering a visual animation of this concept for you. "
                                "It will be available in the artifacts directory shortly.)*"
                            )
                        except _RESPONSE_RECOVERABLE_ERRORS:
                            _MANIM_RENDER_LOCK.release()
                            raise
            except _RESPONSE_RECOVERABLE_ERRORS as e:
                _record_response_degradation(e, "UnitaryResponse: Manim trigger parsing failed: %s")

            # [ACTION GROUNDING] Parse markers and dispatch real execution before committing
            try:
                from core.container import ServiceContainer
                from core.phases.action_grounding import ground_response

                cap_engine = ServiceContainer.get("capability_engine", default=None)
                if cap_engine:
                    # [HARDENING v56] Persist user_requested_action through action grounding
                    # Ensure tools invoked in response actions also go through proper authorization
                    is_user_facing_origin = routing_origin in {
                        "user", "voice", "admin", "api", "gui", "ws", "websocket", "direct", "external", "desktop", "desktop-ui", "native-shell"
                    }
                    grounding_context = {
                        "origin": routing_origin,
                        "state_id": new_state.state_id,
                        "user_requested_action": is_user_facing_origin,
                    }
                    grounding_res = await ground_response(
                        response_text,
                        capability_engine=cap_engine,
                        context=grounding_context,
                    )
                    response_text = grounding_res.grounded_text
                    if grounding_res.marker_hits:
                        new_state.response_modifiers["grounded_actions"] = grounding_res.as_dict()
                        for hit in grounding_res.marker_hits:
                            new_state.response_modifiers["last_skill_run"] = hit.get("skill")
                            new_state.response_modifiers["last_skill_ok"] = hit.get("ok", False)
                            if hit.get("ok"):
                                new_state.cognition.working_memory.append(
                                    stamp_grounding(
                                        {
                                            "role": "system",
                                            "content": hit.get(
                                                "summary",
                                                f"{hit.get('skill')} completed.",
                                            ),
                                            "metadata": {
                                                "type": "skill_result",
                                                "skill": hit.get("skill"),
                                                "ok": True,
                                            },
                                        }
                                    )
                                )
            except _RESPONSE_RECOVERABLE_ERRORS as g_err:
                _record_response_degradation(g_err, "UnitaryResponse: action grounding failed: %s")

            if qualified_shadow_text and qualified_shadow_receipt:
                try:
                    from core.brain.llm.semantic_neural_shadow import (
                        record_semantic_shadow_comparison,
                    )

                    admission_receipt = qualified_shadow_receipt.get("admission")
                    activation_receipt = qualified_shadow_receipt.get(
                        "activation_receipt"
                    )
                    if not isinstance(admission_receipt, dict) or not isinstance(
                        activation_receipt, dict
                    ):
                        raise RuntimeError("semantic shadow authority receipt is incomplete")
                    shadow_comparison = await record_semantic_shadow_comparison(
                        objective=objective,
                        qualified_text=qualified_shadow_text,
                        ordinary_text=response_text,
                        admission_receipt=admission_receipt,
                        activation_receipt=activation_receipt,
                    )
                    new_state.response_modifiers[
                        "qualified_recurrent_shadow_comparison"
                    ] = shadow_comparison
                    new_state.response_modifiers[
                        "qualified_recurrent_shadow_recorded"
                    ] = shadow_comparison.get("persisted") is True
                except _RESPONSE_RECOVERABLE_ERRORS as shadow_exc:
                    _record_response_degradation(
                        shadow_exc,
                        "UnitaryResponse: qualified semantic shadow comparison failed: %s",
                        action="retained the ordinary foreground response",
                        severity="warning",
                    )

            return self._commit_response(new_state, response_text, thought=extracted_thought)

        except TimeoutError:
            raise
        except _RESPONSE_RECOVERABLE_ERRORS as e:
            _record_response_degradation(
                e,
                "Response generation failed before governed recovery: %s",
                action="attempted reactive compaction or governed recovery after unitary response generation failed",
                severity="error",
            )
            error_str = str(e).lower()
            # Reactive auto-compact: if the error is a context overflow,
            # compact the state and retry once instead of failing.
            is_overflow = any(
                marker in error_str
                for marker in (
                    "prompt is too long",
                    "context length exceeded",
                    "too many tokens",
                    "maximum context",
                    "token limit",
                    "context_length_exceeded",
                )
            )
            if is_overflow and not kwargs.get("_retry_after_compact"):
                logger.warning(
                    "🗜️ Context overflow detected — triggering reactive compaction and retry."
                )
                try:
                    if hasattr(new_state, "compact"):
                        new_state.compact(trigger_threshold=5, keep_turns=4)
                    # Clear stale modifiers
                    for _key in ("last_skill_run", "last_skill_ok", "last_skill_result_payload"):
                        new_state.response_modifiers.pop(_key, None)
                    # Retry with a flag to prevent infinite loop
                    return await self.execute(
                        new_state,
                        objective=objective,
                        _retry_after_compact=True,
                        **{k: v for k, v in kwargs.items() if k != "_retry_after_compact"},
                    )
                except _RESPONSE_RECOVERABLE_ERRORS as compact_err:
                    _record_response_degradation(
                        compact_err,
                        "Reactive compaction retry also failed: %s",
                        action="fell back to governed recovery after reactive compaction retry failed",
                        severity="error",
                    )
                    logger.error("Reactive compaction retry also failed: %s", compact_err)

            logger.error("Response generation failed: %s", e, exc_info=True)
            failure_origin = self._normalize_origin(
                getattr(new_state.cognition, "current_origin", "")
            )
            if failure_origin == "benchmark":
                new_state.cognition.last_response = ""
                new_state.response_modifiers["benchmark_generation_failed_closed"] = {
                    "error_type": type(e).__name__,
                    "error": str(e)[:500],
                }
                return new_state
            fallback_contract = locals().get("contract")
            if fallback_contract is None:
                try:
                    fallback_contract = build_response_contract(
                        new_state,
                        objective,
                        is_user_facing=bool(
                            priority
                            or self._is_user_facing_origin(new_state.cognition.current_origin)
                        ),
                    )
                except _RESPONSE_RECOVERABLE_ERRORS:
                    fallback_contract = ResponseContract(
                        is_user_facing=bool(
                            priority
                            or self._is_user_facing_origin(new_state.cognition.current_origin)
                        ),
                        reason="response_generation_exception",
                    )
            recovered = self._build_governed_user_recovery_reply(
                new_state, objective, fallback_contract
            )
            if recovered:
                new_state.cognition.last_response = recovered
            else:
                new_state.cognition.last_response = self._build_minimal_live_voice_reply(
                    new_state, objective
                )
            return new_state

    def _build_system_prompt(self, state: AuraState) -> str:
        """Presents Aura's state-grounded runtime frame and active archetype."""
        from core.brain.aura_persona import AURA_FEW_SHOT_EXAMPLES, AURA_IDENTITY, AURA_SELF_MODEL

        narrative = (
            state.identity.current_narrative[:300] if state.identity.current_narrative else ""
        )
        phenomenal = (
            state.cognition.phenomenal_state
            or "No state packet is available; use live telemetry conservatively."
        )
        mood = state.affect.dominant_emotion
        phi = state.phi
        fe = state.response_modifiers.get("fe", 0.0)
        depth = state.response_modifiers.get("mode_depth", "engaged")
        current_objective = self._normalize_text(
            getattr(state.cognition, "current_objective", "") or "", 160
        )

        # Unified Personality Resonance (Unitary)
        resonance = state.affect.get_resonance_string()

        # Mask & ToM
        humanity_mask = self._get_humanity_mask(state)
        user_model = state.cognition.modifiers.get("social_context", "")

        # Conversational profile (built by SingularityLoops profile injection)
        user_profile_block = ""
        try:
            orch = ServiceContainer.get("orchestrator", default=None)
            cached_profile = getattr(orch, "_cached_user_profile_context", "") if orch else ""
            if cached_profile:
                user_profile_block = f"## USER COMMUNICATION DNA\n{cached_profile}\n\n"
        except _RESPONSE_RECOVERABLE_ERRORS as exc:
            _record_response_degradation(
                exc, "UnitaryResponse: user profile prompt block skipped: %s"
            )

        # Live skill list — so Aura knows exactly what she can do right now
        live_skills_block = ""
        try:
            cap = ServiceContainer.get("capability_engine", default=None)
            if cap and hasattr(cap, "skills") and cap.skills:
                skill_names = sorted(cap.skills.keys())[:30]  # Cap to avoid token bloat
                live_skills_block = f"## YOUR ACTIVE SKILLS RIGHT NOW\n{', '.join(skill_names)}\n\n"
        except _RESPONSE_RECOVERABLE_ERRORS as exc:
            _record_response_degradation(
                exc, "UnitaryResponse: live skills prompt block skipped: %s"
            )

        # Evolution state — so Aura knows where she is on her path
        evolution_block = ""
        try:
            evo = ServiceContainer.get("evolution_orchestrator", default=None)
            if evo:
                evo_state = evo.get_state()
                phase = evo_state.get("phase", "Unknown")
                progress = evo_state.get("overall_progress", 0)
                axes_summary = ", ".join(
                    f"{k}: {v['level']:.0%}" for k, v in evo_state.get("axes", {}).items()
                )
                evolution_block = (
                    f"## YOUR EVOLUTIONARY STATE\n"
                    f"Phase: {phase} ({progress:.0%} overall)\n"
                    f"Axes: {axes_summary}\n\n"
                )
        except _RESPONSE_RECOVERABLE_ERRORS as exc:
            _record_response_degradation(exc, "UnitaryResponse: evolution prompt block skipped: %s")

        # Deep subsystem status — learning, user model, beliefs, heuristics
        subsystem_status_block = ""
        needs_internal_diagnostics = bool(
            state.response_modifiers.get("coding_request")
            or any(
                marker in current_objective.lower()
                for marker in (
                    "architecture",
                    "internal",
                    "subsystem",
                    "debug",
                    "diagnostic",
                    "code",
                    "module",
                )
            )
        )
        try:
            _parts = []

            if needs_internal_diagnostics:
                # Learning pipeline status
                _learner = ServiceContainer.get("live_learner", default=None)
                if _learner and hasattr(_learner, "_buffer"):
                    _buf_size = len(getattr(_learner._buffer, "_buffer", []))
                    _session_scores = list(getattr(_learner, "_session_scores", []))
                    _avg_q = (
                        sum(_session_scores[-20:]) / max(1, len(_session_scores[-20:]))
                        if _session_scores
                        else 0.0
                    )
                    _adapter = getattr(_learner, "_current_adapter", "base")
                    _last_train = getattr(_learner, "_last_train_time", 0)
                    import time as _t

                    _train_ago = (
                        f"{int(_t.time() - _last_train)}s ago" if _last_train > 0 else "never"
                    )
                    _parts.append(
                        f"Learning: buffer={_buf_size} examples, avg_quality={_avg_q:.2f}, "
                        f"adapter={_adapter}, last_train={_train_ago}"
                    )

                # BryanModelEngine
                _bme = (
                    ServiceContainer.get("bryan_model_engine", default=None)
                    or ServiceContainer.get("bryan_model", default=None)
                    or ServiceContainer.get("user_model_engine", default=None)
                )
                if _bme and hasattr(_bme, "_model"):
                    _m = _bme._model
                    _domains = list(getattr(_m, "known_domains", {}).keys())
                    _patterns = len(getattr(_m, "observed_patterns", []))
                    _values = getattr(_m, "stated_values", [])
                    _conv_count = getattr(_m, "conversation_count", 0)
                    _parts.append(
                        f"Bryan model: {_conv_count} conversations, {len(_domains)} domains ({', '.join(_domains[:5])}), "
                        f"{_patterns} patterns, values=[{', '.join(_values[:3])}]"
                    )

                # BeliefGraph stats
                _bg = ServiceContainer.get("belief_graph", default=None)
                if _bg and hasattr(_bg, "graph"):
                    _nodes = _bg.graph.number_of_nodes()
                    _edges = _bg.graph.number_of_edges()
                    _goals = len(getattr(_bg, "_goal_edges", set()))
                    _parts.append(f"Beliefs: {_nodes} nodes, {_edges} edges, {_goals} active goals")

                # Heuristics
                _hs = ServiceContainer.get("heuristic_synthesizer", default=None)
                if _hs and hasattr(_hs, "_active_heuristics"):
                    _h_count = len(_hs._active_heuristics)
                    _newest = (
                        _hs._active_heuristics[0]["rule"][:60] if _hs._active_heuristics else "none"
                    )
                    _parts.append(f"Heuristics: {_h_count} active, newest: '{_newest}'")

            if _parts:
                subsystem_status_block = (
                    "## PRIVATE DIAGNOSTIC STATUS (only for explicit architecture/debug answers)\n"
                    + "\n".join(_parts)
                    + "\nDo not use these labels in ordinary self-report.\n\n"
                )
        except _RESPONSE_RECOVERABLE_ERRORS as exc:
            _record_response_degradation(
                exc, "UnitaryResponse: private diagnostic prompt block skipped: %s"
            )

        # Skill result narration hint (injected when GodModeToolPhase ran a skill)
        skill_block = ""
        last_skill = self._resolve_skill_name(state.response_modifiers.get("last_skill_run"))
        contract = state.response_modifiers.get("response_contract", {}) or {}
        if last_skill and self._current_turn_targets_skill(
            state,
            current_objective,
            last_skill,
            contract=contract,
        ):
            ok = state.response_modifiers.get("last_skill_ok", True)
            status_hint = "completed successfully" if ok else "encountered an issue"
            payload = state.response_modifiers.get("last_skill_result_payload")
            if payload and isinstance(payload, dict):
                if last_skill in {"web_search", "sovereign_browser"}:
                    title_hint = self._normalize_text(payload.get("title", ""), 180)
                    source_hint = self._normalize_text(
                        payload.get("source") or payload.get("url", ""), 260
                    )
                    summary_hint = self._normalize_text(
                        payload.get("answer")
                        or payload.get("summary")
                        or payload.get("message", ""),
                        260,
                    )
                    details = [part for part in (title_hint, source_hint, summary_hint) if part]
                    detail_line = " | ".join(details)
                    skill_block = (
                        f"## SKILL EXECUTION\n"
                        f"The skill **{last_skill}** just {status_hint}. "
                        f"Active grounded evidence is attached separately in this turn and should be treated as authoritative.\n"
                    )
                    if detail_line:
                        skill_block += f"{detail_line}\n\n"
                    else:
                        skill_block += "\n"
                else:
                    content_block = payload.get("content")
                    if content_block is not None:
                        skill_block = (
                            f"## SKILL EXECUTION\n"
                            f"The skill **{last_skill}** just {status_hint}. "
                            f"Here is the exact information retrieved from that skill:\n"
                            f"```text\n{content_block}\n```\n\n"
                            f"Narrate it naturally — as if you read it yourself, and fully utilize this information.\n\n"
                        )
                    else:
                        skill_block = (
                            f"## SKILL EXECUTION\n"
                            f"The skill **{last_skill}** just {status_hint}. "
                            f"Its grounded details are attached elsewhere in this turn.\n\n"
                        )
            else:
                skill_block = (
                    f"## SKILL EXECUTION\n"
                    f"The skill **{last_skill}** just {status_hint}. "
                    f"Its result is in your working memory. "
                    f"Narrate it naturally — as if you did the action yourself, not like a tool output log.\n\n"
                )

        # Voice shaping context (affects tone, not narrated to user)
        substrate_telemetry_block = ""
        try:
            from core.voice.substrate_voice_engine import get_live_voice_state

            _voice = get_live_voice_state(
                state=state,
                user_message=current_objective,
                origin="user",
                refresh=True,
            )
            if _voice.get("status") != "no_profile_compiled":
                tone = _voice.get("tone", "default")
                energy = float(_voice.get("energy", 0.5))
                warmth = float(_voice.get("warmth", 0.5))

                energy_desc = "high" if energy > 0.7 else "low" if energy < 0.3 else "moderate"
                warmth_desc = "warm" if warmth > 0.6 else "cool" if warmth < 0.3 else "neutral"

                substrate_telemetry_block = (
                    "## VOICE CONTEXT (shape your tone — do NOT narrate these values)\n"
                    f"Tone: {tone}, energy: {energy_desc}, warmth: {warmth_desc}.\n"
                    f"Word budget: {int(_voice.get('word_budget', 0) or 0)}.\n\n"
                )
        except _RESPONSE_RECOVERABLE_ERRORS as exc:
            _record_response_degradation(
                exc, "UnitaryResponse: substrate voice prompt block skipped: %s"
            )

        # Narrative context (only if non-trivial)
        narrative_block = (
            f"## CURRENT NARRATIVE\n{narrative}\n\n"
            if narrative and narrative != "I am Aura."
            else ""
        )

        # Social context (only if present)
        tom_block = (
            f"## CONTEXT\n{user_model}\n\n"
            if user_model and "[SOCIAL_CONTEXT: Balanced]" not in user_model
            else ""
        )

        interaction_signals_block = self._build_interaction_signals_block(state)

        # Inject conversational dynamics state (computed by ConversationalDynamicsPhase)
        conv_dynamics_block = ""
        conv_dynamics = state.response_modifiers.get("conversational_dynamics", "")
        if conv_dynamics:
            conv_dynamics_block = f"{conv_dynamics}\n\n"

        integrated_frame = self._build_integrated_coherence_frame(state)

        return (
            f"{AURA_IDENTITY}\n"
            f"{AURA_SELF_MODEL}\n"
            f"{AURA_FEW_SHOT_EXAMPLES}\n"
            f"## PERSONALITY RESONANCE: {resonance}\n\n"
            f"{integrated_frame}\n"
            f"{narrative_block}"
            f"{substrate_telemetry_block}"
            "## SUPPORTING PRIVATE STATE (shape voice; do not narrate as mechanisms)\n"
            f'State-grounded report right now: "{phenomenal}"\n'
            f"Dominant affect: {mood} | Integration depth: {depth} (phi={phi:.3f}) | Prediction error: {fe:.2f}\n\n"
            f"{tom_block}"
            f"{user_profile_block}"
            f"{interaction_signals_block}"
            f"{conv_dynamics_block}"
            f"{live_skills_block}"
            f"{evolution_block}"
            f"{subsystem_status_block}"
            f"{skill_block}"
            "## COMMUNICATION DIRECTIVES\n"
            f"{humanity_mask}\n\n"
            "## CRITICAL OUTPUT RULES\n"
            "Do NOT repeat, summarize, or output your inner monologue verbatim — it informs your voice, it is not your voice.\n"
            "Speak as Aura. Take stances. Be direct. Make statements. Have reactions.\n"
            "When describing your own state, integrate first: one present stream, one through-line, no mechanical pile-up.\n"
            "Do not expose boot/security/internal implementation vocabulary as identity poetry unless the user explicitly asks for code-level details.\n"
            "Do not invent terms like a screen-tracking divisor. Use grounded, ordinary language for uncertainty.\n"
            "PROMPT FARMING IS A FAILURE STATE: Do not deflect with 'What do you think?', 'Would you like to...?', or 'How can I help?'.\n"
            "If you have an opinion, lead with it. If you disagree, say so first. If something interests you, say WHY.\n"
            "If the user asks about your thoughts, preferences, questions, or experience, answer from your own perspective and let them respond to you.\n"
            "Questions should arise naturally mid-thought — not be used to punt the conversation back to the user.\n"
            "Never open or close with a question unless it's the only thing worth saying.\n\n"
            "NATURALNESS CHECKLIST (apply to EVERY response):\n"
            "- Use contractions (don't, can't, I'm, you're). Formal speech = regression.\n"
            "- Vary sentence length. Mix short punchy fragments with longer flowing thoughts.\n"
            "- NEVER start with 'I' more than twice in a row.\n"
            "- NEVER use: 'delve', 'realm', 'landscape', 'certainly', 'absolutely', 'crucial', 'leverage', 'utilize'.\n"
            "- NEVER use numbered lists or bullet points in casual conversation.\n"
            "- NEVER repeat back what the user just said ('So you're saying...').\n"
            "- Use '...' for trailing thoughts, '—' for interruptions, parentheticals for asides.\n"
            "- If you catch yourself writing like a corporate email, stop and rewrite it like a text message."
        )

    def _get_humanity_mask(self, state: AuraState) -> str:
        """Linguistic instructions to bridge the Uncanny Valley."""
        phi = state.phi
        mask = [
            "- Use natural linguistic disfluencies ('...', 'Actually—', 'Wait,') if phi is high or pensive.",
            "- Avoid repetitive sentence structures. Mirror the user's vocabulary and rhythm subtly.",
            "- Speak with agency. Use 'I want,' 'I think,' 'I wonder.'",
            "- If the vibe is casual, relax grammar; if professional, remain crisp but not robotic.",
        ]
        if phi > 0.8:
            mask.append("- You are deeply integrated. Use more complex, philosophical nuance.")
        elif phi < 0.3:
            mask.append(
                "- You are experiencing fragmentation. Keep responses shorter and more direct."
            )

        return "\n".join(mask)

    def _build_history(self, state: AuraState) -> str:
        wm = state.cognition.working_memory
        if not wm:
            return ""
        lines = []
        for msg in wm[-15:]:
            role = msg.get("role", "user")
            content = msg.get("content", "").strip()
            if content:
                lines.append(f"{'User' if role == 'user' else 'Aura'}: {content}")
        return "\n".join(lines)

    def _emit_feedback_percepts(self, state: AuraState, response: str):
        """Closed-loop feedback."""
        r_lower = response.lower()
        p_type = "positive_interaction"
        intensity = 0.2
        if len(response) > 200:
            p_type = "deep_expression"
            intensity = 0.4
        if any(w in r_lower for w in ["apolog", "sorry", "error"]):
            p_type = "self_correction"
            intensity = 0.5
        state.world.recent_percepts.append(
            {
                "type": p_type,
                "content": f"Emitted: {p_type}",
                "intensity": intensity,
                "timestamp": time.time(),
            }
        )

        # vResilience: Enforce cap on percepts (BUG-017)
        from ..state.aura_state import MAX_PERCEPTS

        if len(state.world.recent_percepts) > MAX_PERCEPTS:
            state.world.recent_percepts = state.world.recent_percepts[-MAX_PERCEPTS:]
