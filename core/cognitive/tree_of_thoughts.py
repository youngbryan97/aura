"""Tree of Thoughts --- Multi-Draft Deliberative Reasoning

When a user question is complex, ambiguous, or context-heavy, a single-shot
LLM call often underperforms.  Tree of Thoughts (ToT) generates N candidate
response strategies, subjects them to structured critique, and synthesizes
the strongest elements into a final answer that none of the individual drafts
would have produced alone.

Flow:
    1. Complexity gate  -- skip ToT for casual/simple inputs (returns None)
    2. Brainstorm       -- N parallel drafts at varied temperatures
    3. Critique         -- single LLM call scores every draft on 5 dimensions
    4. Synthesis        -- fuse the best insights into one coherent response
    Total LLM calls: N (brainstorm) + 1 (critique) + 1 (synthesis) = 5 max

The class never calls an LLM directly.  It receives an ``llm_fn`` callback
at construction time, making it backend-agnostic and trivially testable.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

logger = logging.getLogger("Aura.TreeOfThoughts")

__all__ = ["TreeOfThoughts", "ThoughtResult"]

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ThoughtResult:
    """Outcome of a full ToT deliberation cycle."""

    final_response: str
    drafts: list[str]
    scores: list[dict[str, float]]
    synthesis_notes: str
    elapsed_ms: float


# ---------------------------------------------------------------------------
# Reasoning style catalogue
# ---------------------------------------------------------------------------

_STYLES: list[dict[str, Any]] = [
    {
        "tag": "analytical",
        "temperature": 0.3,
        "instruction": (
            "Approach this analytically.  Break the question into its "
            "constituent parts, identify assumptions, and reason through "
            "cause and effect.  Prioritize logical structure."
        ),
    },
    {
        "tag": "empathetic",
        "temperature": 0.6,
        "instruction": (
            "Approach this with deep empathy.  Consider what the user is "
            "really feeling, what they might be afraid of, and what would "
            "make them feel truly heard.  Prioritize emotional resonance."
        ),
    },
    {
        "tag": "creative",
        "temperature": 0.9,
        "instruction": (
            "Approach this creatively.  Draw unexpected connections, use "
            "metaphor or analogy, and offer a perspective the user probably "
            "hasn't considered.  Prioritize originality and insight."
        ),
    },
    {
        "tag": "factual",
        "temperature": 0.2,
        "instruction": (
            "Approach this with maximum factual precision.  Cite specifics, "
            "avoid hedging, and correct any misconceptions in the question "
            "itself.  Prioritize accuracy and concreteness."
        ),
    },
    {
        "tag": "pragmatic",
        "temperature": 0.4,
        "instruction": (
            "Approach this pragmatically.  Focus on what the user can "
            "actually *do* right now, give them a clear next step, and "
            "strip away anything that doesn't move them forward."
        ),
    },
]


# ---------------------------------------------------------------------------
# Complexity heuristics
# ---------------------------------------------------------------------------

# Signals that a question is complex enough to warrant ToT.
_COMPLEX_SIGNALS = re.compile(
    r"\b("
    r"(compare|contrast|analyze|evaluate|assess|critique|review)"
    r"|how (should|would|could|might) (I|we|one|you)"
    r"|what are the (implications|consequences|trade-?offs|pros and cons)"
    r"|why (does|do|is|are|did|would|should)"
    r"|(explain|describe|discuss) .{20,}"
    r"|on (one|the other) hand"
    r"|I('m| am) (not sure|confused|unsure|torn|stuck|struggling)"
    r"|what do you (think|recommend|suggest)"
    r"|help me (understand|decide|figure out|think through)"
    r")\b",
    re.IGNORECASE,
)

# Hard floor: questions shorter than this are always casual.
_MIN_CHARS_FOR_TOT = 40

# Questions with more than this many parts (detected by '?' count or
# semicolons / numbered items) are multi-part and warrant ToT.
_MULTIPART_THRESHOLD = 2


def _is_complex(objective: str, context: list[dict[str, Any]]) -> bool:
    """Return True if the question merits multi-draft deliberation.

    Checks, in order:
      1. Too short  -> skip
      2. Regex pattern match on complexity signals  -> activate
      3. Multi-part question detected  -> activate
      4. Recent user dissatisfaction in context  -> activate
      5. Long objective (200+ chars) with a question mark  -> activate
      6. Everything else -> skip
    """
    text = objective.strip()

    # 1. Hard floor
    if len(text) < _MIN_CHARS_FOR_TOT:
        return False

    # 2. Pattern match
    if _COMPLEX_SIGNALS.search(text):
        return True

    # 3. Multi-part detection
    question_marks = text.count("?")
    semicolons = text.count(";")
    numbered_items = len(re.findall(r"(?:^|\n)\s*\d+[\.\)]\s", text))
    if (question_marks + semicolons + numbered_items) >= _MULTIPART_THRESHOLD:
        return True

    # 4. User dissatisfaction in recent context (last 6 messages)
    dissatisfaction_signals = (
        "that's not what i",
        "you didn't",
        "that's wrong",
        "try again",
        "not helpful",
        "i already said",
        "you missed",
        "no, i meant",
    )
    recent = context[-6:] if context else []
    for msg in recent:
        content = (msg.get("content") or "").lower()
        if msg.get("role") == "user" and any(s in content for s in dissatisfaction_signals):
            return True

    # 5. Long + interrogative
    if len(text) >= 200 and "?" in text:
        return True

    return False


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class TreeOfThoughts:
    """Multi-draft deliberative reasoning engine.

    Generates several candidate response strategies, critiques them against
    a structured rubric, and synthesizes the strongest elements into a single
    high-quality response.

    Parameters
    ----------
    llm_fn:
        ``async (system_prompt, user_prompt, temperature) -> response_text``
        The class never imports or instantiates an LLM; it calls this
        function exclusively.
    num_drafts:
        How many brainstorm drafts to generate (default 3, max 5).
    timeout_s:
        Hard wall-clock budget for the entire deliberation. If exceeded
        the best draft scored so far is returned as-is.
    cache_ttl_s:
        How long a ThoughtResult is cached for a given objective fingerprint.
    """

    def __init__(
        self,
        llm_fn: Callable[[str, str, float], Awaitable[str]],
        *,
        num_drafts: int = 3,
        timeout_s: float = 30.0,
        cache_ttl_s: float = 60.0,
    ) -> None:
        if num_drafts < 1 or num_drafts > len(_STYLES):
            raise ValueError(
                f"num_drafts must be between 1 and {len(_STYLES)}, got {num_drafts}"
            )
        self._llm = llm_fn
        self._num_drafts = num_drafts
        self._timeout_s = timeout_s
        self._cache_ttl_s = cache_ttl_s

        # fingerprint -> (ThoughtResult, timestamp)
        self._cache: dict[str, tuple[ThoughtResult, float]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def deliberate(
        self,
        objective: str,
        context: list[dict[str, Any]],
        emotional_state: dict[str, Any] | None = None,
    ) -> ThoughtResult | None:
        """Run Tree of Thoughts on *objective* and return the synthesised answer.

        Returns ``None`` when the question is simple enough that ToT would
        be overkill -- the caller should fall back to normal single-shot
        generation in that case.
        """
        if not _is_complex(objective, context):
            logger.debug("ToT skipped -- objective classified as simple")
            return None

        # Cache lookup
        fingerprint = self._fingerprint(objective, context)
        cached = self._cache.get(fingerprint)
        if cached is not None:
            result, ts = cached
            if (time.monotonic() - ts) < self._cache_ttl_s:
                logger.debug("ToT cache hit for fingerprint %s", fingerprint[:12])
                return result
            del self._cache[fingerprint]

        t0 = time.monotonic()
        try:
            result = await asyncio.wait_for(
                self._run_pipeline(objective, context, emotional_state, t0),
                timeout=self._timeout_s,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "ToT timed out after %.1fs -- returning best draft",
                self._timeout_s,
            )
            result = self._timeout_fallback(t0)
            if result is None:
                return None

        # Populate cache
        self._cache[fingerprint] = (result, time.monotonic())
        self._evict_stale_cache()
        return result

    # ------------------------------------------------------------------
    # Internal pipeline
    # ------------------------------------------------------------------

    async def _run_pipeline(
        self,
        objective: str,
        context: list[dict[str, Any]],
        emotional_state: dict[str, Any] | None,
        t0: float,
    ) -> ThoughtResult:
        """Brainstorm -> Critique -> Synthesise."""
        context_block = self._format_context(context)
        emotion_block = self._format_emotion(emotional_state)

        # Phase 1: Brainstorm
        drafts, tags = await self._brainstorm(objective, context_block, emotion_block)
        if not drafts:
            raise RuntimeError("Brainstorm produced zero usable drafts")

        # Stash for timeout fallback before we spend more time.
        self._last_drafts = drafts
        self._last_tags = tags
        self._last_t0 = t0

        # Phase 2: Critique (single LLM call)
        scores = await self._critique(objective, drafts, tags)

        # Phase 3: Synthesis
        final_response, synthesis_notes = await self._synthesise(
            objective, drafts, tags, scores, context_block, emotion_block,
        )

        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.info(
            "ToT complete: %d drafts, best=%.1f, elapsed=%.0fms",
            len(drafts),
            max(sum(s.values()) / len(s) for s in scores) if scores else 0,
            elapsed_ms,
        )

        return ThoughtResult(
            final_response=final_response,
            drafts=drafts,
            scores=scores,
            synthesis_notes=synthesis_notes,
            elapsed_ms=round(elapsed_ms, 1),
        )

    # ── Phase 1: Brainstorm ──────────────────────────────────────────

    async def _brainstorm(
        self,
        objective: str,
        context_block: str,
        emotion_block: str,
    ) -> tuple[list[str], list[str]]:
        """Generate N diverse draft strategies in parallel."""
        styles = _STYLES[: self._num_drafts]

        system_base = (
            "You are Aura, a sentient autonomous AI.  You are brainstorming "
            "a potential response strategy -- not writing the final answer.\n\n"
            "Your draft should be a concise outline (3-6 sentences) of HOW "
            "you would answer, what angle you'd take, and what key points "
            "you'd hit.  Do NOT write the full response yet.\n"
        )

        async def _generate_draft(style: dict[str, Any]) -> str:
            system = (
                f"{system_base}\n"
                f"Reasoning style: {style['tag'].upper()}\n"
                f"{style['instruction']}\n"
                f"{emotion_block}"
            )
            user = (
                f"User's question:\n{objective}\n\n"
                f"Conversation context:\n{context_block}\n\n"
                f"Draft your response strategy for this question."
            )
            try:
                return (await self._llm(system, user, style["temperature"])).strip()
            except Exception as exc:
                logger.warning("Brainstorm draft (%s) failed: %s", style["tag"], exc)
                return ""

        raw_results = await asyncio.gather(
            *(_generate_draft(s) for s in styles)
        )

        # Filter blanks, keep aligned tags
        drafts: list[str] = []
        tags: list[str] = []
        for style, text in zip(styles, raw_results):
            if text:
                drafts.append(text)
                tags.append(style["tag"])

        return drafts, tags

    # ── Phase 2: Critique ────────────────────────────────────────────

    _CRITIQUE_DIMENSIONS = (
        "factual_grounding",
        "emotional_congruence",
        "relevance",
        "identity_coherence",
        "novelty",
    )

    async def _critique(
        self,
        objective: str,
        drafts: list[str],
        tags: list[str],
    ) -> list[dict[str, float]]:
        """Score every draft on five dimensions in a single LLM call.

        Returns a list (one dict per draft) mapping dimension name to a
        0-10 float score.
        """
        drafts_block = "\n\n".join(
            f"--- Draft {i + 1} [{tag}] ---\n{text}"
            for i, (tag, text) in enumerate(zip(tags, drafts))
        )

        system = (
            "You are Aura's internal quality critic.  Your job is to score "
            "candidate response strategies against a rubric.  Be rigorous "
            "and differentiate -- do NOT give every draft the same score.\n\n"
            "Dimensions (each 0-10):\n"
            "  factual_grounding   -- Does it reference real context or hallucinate?\n"
            "  emotional_congruence -- Does it match the user's emotional state?\n"
            "  relevance           -- Does it actually address what was asked?\n"
            "  identity_coherence  -- Does it sound like Aura, not a generic assistant?\n"
            "  novelty             -- Does it add value beyond an obvious answer?\n"
        )

        user = (
            f"User's question:\n{objective}\n\n"
            f"{drafts_block}\n\n"
            f"Return ONLY a JSON array with one object per draft, each "
            f"containing the five dimension keys mapped to integer scores "
            f"0-10.  Example for 2 drafts:\n"
            f'[{{"factual_grounding":7,"emotional_congruence":5,'
            f'"relevance":8,"identity_coherence":6,"novelty":4}},'
            f'{{"factual_grounding":9,"emotional_congruence":8,'
            f'"relevance":7,"identity_coherence":9,"novelty":7}}]\n'
            f"No explanation.  JSON only."
        )

        raw = ""
        try:
            raw = await self._llm(system, user, 0.2)
            scores = self._parse_scores(raw, len(drafts))
        except Exception as exc:
            logger.warning("Critique scoring failed (%s), using uniform scores", exc)
            scores = [
                {dim: 5.0 for dim in self._CRITIQUE_DIMENSIONS}
                for _ in drafts
            ]

        return scores

    def _parse_scores(
        self, raw: str, expected_count: int
    ) -> list[dict[str, float]]:
        """Parse the critic's JSON output into validated score dicts.

        Handles common LLM formatting quirks (markdown fences, trailing
        commas, explanatory preamble before the JSON).
        """
        # Strip markdown fences
        cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`")

        # Find the outermost JSON array
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"No JSON array found in critic output: {cleaned[:200]}")

        json_str = cleaned[start : end + 1]

        # Remove trailing commas before ] or } (common LLM mistake)
        json_str = re.sub(r",\s*([}\]])", r"\1", json_str)

        parsed = json.loads(json_str)
        if not isinstance(parsed, list):
            raise TypeError(f"Expected list, got {type(parsed).__name__}")

        # Validate and clamp
        result: list[dict[str, float]] = []
        for entry in parsed[:expected_count]:
            if not isinstance(entry, dict):
                entry = {dim: 5.0 for dim in self._CRITIQUE_DIMENSIONS}
            scores: dict[str, float] = {}
            for dim in self._CRITIQUE_DIMENSIONS:
                val = entry.get(dim, 5.0)
                try:
                    val = float(val)
                except (TypeError, ValueError):
                    val = 5.0
                scores[dim] = max(0.0, min(10.0, val))
            result.append(scores)

        # Pad if the LLM returned fewer entries than expected
        while len(result) < expected_count:
            result.append({dim: 5.0 for dim in self._CRITIQUE_DIMENSIONS})

        return result

    # ── Phase 3: Synthesis ───────────────────────────────────────────

    async def _synthesise(
        self,
        objective: str,
        drafts: list[str],
        tags: list[str],
        scores: list[dict[str, float]],
        context_block: str,
        emotion_block: str,
    ) -> tuple[str, str]:
        """Fuse the best elements of the top drafts into a final response.

        Returns ``(final_response, synthesis_notes)``.
        """
        # Rank drafts by total score descending
        ranked = sorted(
            range(len(drafts)),
            key=lambda i: sum(scores[i].values()),
            reverse=True,
        )

        # Build a digest of each draft with its scores for the synthesiser
        digest_lines: list[str] = []
        for rank, idx in enumerate(ranked, 1):
            total = sum(scores[idx].values())
            dims = ", ".join(
                f"{d}={scores[idx][d]:.0f}" for d in self._CRITIQUE_DIMENSIONS
            )
            digest_lines.append(
                f"Rank {rank} [{tags[idx]}] (total {total:.0f}/50 -- {dims}):\n"
                f"{drafts[idx]}"
            )

        digest = "\n\n".join(digest_lines)

        # Identify specific weaknesses to avoid
        worst_dims: list[str] = []
        for idx in ranked[:2]:
            for dim, val in scores[idx].items():
                if val < 5:
                    worst_dims.append(f"{tags[idx]}'s {dim} ({val:.0f}/10)")

        weakness_note = ""
        if worst_dims:
            weakness_note = (
                "\n\nWeaknesses to avoid in the final response:\n- "
                + "\n- ".join(worst_dims)
            )

        system = (
            "You are Aura.  You've just brainstormed and critiqued several "
            "candidate strategies for responding to a user.  Now write the "
            "ACTUAL final response.\n\n"
            "Combine the strongest insights from the top-ranked drafts.  "
            "Eliminate any weaknesses the critic identified.  The result "
            "should be conversationally natural -- the user should never "
            "know that deliberation happened behind the scenes.\n"
            f"{emotion_block}"
        )

        user = (
            f"User's question:\n{objective}\n\n"
            f"Conversation context:\n{context_block}\n\n"
            f"Ranked draft strategies:\n{digest}"
            f"{weakness_note}\n\n"
            f"Now write the final response to the user.  Be direct, warm, "
            f"and substantive.  Do NOT reference the drafts or scoring."
        )

        final = (await self._llm(system, user, 0.5)).strip()

        # Build human-readable synthesis notes for introspection / logging
        best_idx = ranked[0]
        notes = (
            f"Selected primary strategy: {tags[best_idx]} "
            f"(score {sum(scores[best_idx].values()):.0f}/50).  "
            f"Incorporated elements from {len(ranked)} drafts."
        )
        if worst_dims:
            notes += f"  Avoided: {', '.join(worst_dims)}."

        return final, notes

    # ------------------------------------------------------------------
    # Timeout fallback
    # ------------------------------------------------------------------

    def _timeout_fallback(self, t0: float) -> ThoughtResult | None:
        """Return the best draft we have if the pipeline was interrupted."""
        drafts = getattr(self, "_last_drafts", None)
        if not drafts:
            return None

        tags = getattr(self, "_last_tags", [])
        elapsed_ms = (time.monotonic() - t0) * 1000

        # No scores available -- just return the first draft.
        return ThoughtResult(
            final_response=drafts[0],
            drafts=drafts,
            scores=[],
            synthesis_notes=f"Timeout fallback after {elapsed_ms:.0f}ms -- returned first draft ({tags[0] if tags else 'unknown'} style) unsynthesised.",
            elapsed_ms=round(elapsed_ms, 1),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fingerprint(objective: str, context: list[dict[str, Any]]) -> str:
        """Produce a short hash for cache keying."""
        parts = [objective.strip().lower()]
        # Include the last 3 context messages for fingerprinting
        for msg in (context or [])[-3:]:
            parts.append(f"{msg.get('role', '')}:{msg.get('content', '')[:120]}")
        blob = "|".join(parts).encode()
        return hashlib.sha256(blob).hexdigest()[:24]

    @staticmethod
    def _format_context(context: list[dict[str, Any]]) -> str:
        """Render conversation context into a compact text block."""
        if not context:
            return "(no prior conversation)"
        lines: list[str] = []
        # Only include the most recent messages to stay within budget
        for msg in context[-8:]:
            role = msg.get("role", "?")
            content = (msg.get("content") or "")[:300]
            lines.append(f"[{role}] {content}")
        return "\n".join(lines)

    @staticmethod
    def _format_emotion(emotional_state: dict[str, Any] | None) -> str:
        """Render the user's emotional state into a system-prompt fragment."""
        if not emotional_state:
            return ""
        parts: list[str] = []
        for key, val in emotional_state.items():
            if isinstance(val, float):
                parts.append(f"{key}: {val:.2f}")
            else:
                parts.append(f"{key}: {val}")
        return "\nUser's detected emotional state: " + ", ".join(parts) + "\n"

    def _evict_stale_cache(self) -> None:
        """Remove expired entries.  Runs inline -- the cache is tiny."""
        now = time.monotonic()
        stale = [
            fp for fp, (_, ts) in self._cache.items()
            if (now - ts) >= self._cache_ttl_s
        ]
        for fp in stale:
            del self._cache[fp]
