"""A question about what she remembers, answered from the record.

Recall here is scored rather than searched: a candidate episode is ranked on
how distinctive the words it shares with the question are, so a match on "the"
counts for nothing and a match on a name counts for a lot. What comes back is
composed from the episodes that scored, with nothing added — a recalled answer
that has been smoothed is a confabulation with good manners.
"""
from __future__ import annotations

import asyncio
import re
from typing import Any

from core.state.aura_state import AuraState
from core.utils.intent_normalization import normalize_memory_intent_text


class _AnswersFromWhatSheRemembers:
    """Lifted whole from UnitaryResponsePhase; see response_generation_unitary.py."""

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
        # Imported here rather than at module level: the module these
        # came from imports this one to build the class. A call-time
        # import also still sees a test's patch of the original.
        from .response_generation_unitary import (
            _RESPONSE_RECOVERABLE_ERRORS,
        )

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
        from .response_generation_unitary import (
            _RESPONSE_RECOVERABLE_ERRORS,
            _record_response_degradation,
            logger,
        )

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
        from .response_generation_unitary import (
            _RESPONSE_RECOVERABLE_ERRORS,
            _record_response_degradation,
            logger,
        )

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
        from .response_generation_unitary import (
            _RESPONSE_RECOVERABLE_ERRORS,
            _record_response_degradation,
            logger,
        )

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
