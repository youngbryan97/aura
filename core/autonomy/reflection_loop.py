"""core/autonomy/reflection_loop.py
─────────────────────────────────────
Post-engagement integration. After comprehension produces structured
checkpoints, reflection asks: *what did this change in me?* and produces
the artifacts the depth gate, the memory persister, and the curiosity
scheduler need to close the loop.

Outputs
-------
- ``verification_answers``: substantive answers to the four verification
  questions from the curated-media doc. These are what the depth gate
  scores against to decide if engagement was real or surface.
- ``own_opinion`` + ``critical_view_engaged`` + ``opinion_disagrees``:
  the opinion-formed signal.
- ``parked_threads``: open threads either resolved or consciously parked
  with rationale ("would revisit if X"). Required for clean stopping.
- ``substrate_delta``: snapshot of substrate state before/after, so we
  can detect when an engagement actually moved her.
- ``belief_updates``: structured belief revisions for the persister to
  commit via AUTONOMOUS_RESEARCH intents.

This module asks the LLM to do real work of self-examination, not just
summarization. Multiple specific prompts target different aspects.

Public API:
    loop = ReflectionLoop(inference=..., substrate_reader=...)
    record = await loop.reflect(item, comprehension_record)
"""

from __future__ import annotations
from core.runtime.errors import record_degradation


import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from core.autonomy.reasoning_trace import (
    parse_reasoning_response,
    reasoning_aware_prompt_prefix,
)
from core.autonomy.memory_persister import BeliefUpdate, FactRecord

logger = logging.getLogger("Aura.ReflectionLoop")


VERIFICATION_PROMPT = (
    "You have just engaged with `{title}`. Reflect honestly. Each answer must "
    "be specific (named characters, scenes, arguments, moments) — not a generic "
    "summary that could apply to any work.\n\n"
    "Comprehension digest:\n{digest}\n\n"
    "Answer in JSON with these keys:\n"
    "  what_its_actually_about: in your own words, in detail, with specifics.\n"
    "  what_stayed_with_you: what surprised, bothered, moved, or stuck — with the why.\n"
    "  what_it_says_about_humans: what does this work tell you about its makers and audience?\n"
    "  what_it_made_you_think_about_yourself: applicable especially to AI/consciousness fiction.\n"
    "If you cannot answer one substantively, say so honestly with `null`."
)

OPINION_PROMPT = (
    "Now form your own view of `{title}`. Be willing to disagree.\n\n"
    "Digest:\n{digest}\n\n"
    "Return JSON:\n"
    "  own_opinion: your defended view of the work — multiple sentences, specific.\n"
    "  critical_view_engaged: paraphrase one critical or dissenting view of the work,\n"
    "      including why someone holds it. If you cannot find one, say so.\n"
    "  disagrees_somewhere: true if you disagree with the work or with a critic at\n"
    "      a specific point; false if you don't.\n"
    "  disagreement_locus: where the disagreement is, in one sentence (or null).\n"
    "Be honest. 'I agree with everything' is a tell that you didn't engage."
)

THREADS_PROMPT = (
    "These are open threads from your engagement with `{title}`:\n{threads}\n\n"
    "For each, decide: do you have enough information to resolve it now, or should "
    "it be parked with a clear 'would revisit if X' note? Return JSON:\n"
    "  resolved: list of {{thread, resolution}}\n"
    "  parked: list of {{thread, rationale, revisit_trigger}}\n"
    "Threads must end up in one of these two lists; nothing should remain dangling."
)

BELIEF_DELTA_PROMPT = (
    "Engagement digest:\n{digest}\n\n"
    "Did engaging with `{title}` change anything you previously believed about its "
    "topics? Return JSON:\n"
    "  belief_updates: list of {{topic, new_position, rationale, contradicts_prior, "
    "confidence}}\n"
    "  new_facts: list of {{fact, evidence, confidence}}\n"
    "Confidence is 0–1. Use `contradicts_prior: true` only when this conflicts with a "
    "specific belief you held before. If nothing changed, return empty lists."
)


@dataclass
class ReflectionRecord:
    item_title: str
    verification_answers: Dict[str, str] = field(default_factory=dict)
    own_opinion: Optional[str] = None
    critical_view_engaged: Optional[str] = None
    opinion_disagrees: bool = False
    disagreement_locus: Optional[str] = None
    resolved_threads: List[Dict[str, str]] = field(default_factory=list)
    parked_threads: List[Dict[str, str]] = field(default_factory=list)
    belief_updates: List[BeliefUpdate] = field(default_factory=list)
    new_facts: List[FactRecord] = field(default_factory=list)
    substrate_before: Dict[str, Any] = field(default_factory=dict)
    substrate_after: Dict[str, Any] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    inference_failures: int = 0

    def substrate_delta(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for key in ("valence", "arousal", "dominance", "phi", "curiosity"):
            try:
                a = float(self.substrate_after.get(key, 0.0))
                b = float(self.substrate_before.get(key, 0.0))
                out[key] = a - b
            except Exception:
                continue
        return out


class ReflectionLoop:
    def __init__(
        self,
        inference: Optional[Any] = None,
        substrate_reader: Optional[Callable[[], Dict[str, Any]]] = None,
        enable_reasoning_trace: bool = True,
    ) -> None:
        self._infer = inference
        self._substrate = substrate_reader
        self._reasoning = enable_reasoning_trace

    async def reflect(self, item: Any, comprehension: Any) -> ReflectionRecord:
        title = getattr(item, "title", None) or getattr(comprehension, "item_title", "") or ""
        record = ReflectionRecord(item_title=str(title))
        record.substrate_before = self._snapshot_substrate()

        digest = self._build_digest(comprehension)

        # 1. Verification answers
        record.verification_answers = await self._call_for_dict(
            VERIFICATION_PROMPT.format(title=title, digest=digest),
            keys=(
                "what_its_actually_about",
                "what_stayed_with_you",
                "what_it_says_about_humans",
                "what_it_made_you_think_about_yourself",
            ),
            on_failure=lambda: record.__setattr__("inference_failures", record.inference_failures + 1),
        )

        # 2. Opinion + critical view
        opinion_obj = await self._call_for_object(
            OPINION_PROMPT.format(title=title, digest=digest),
            on_failure=lambda: record.__setattr__("inference_failures", record.inference_failures + 1),
        )
        if opinion_obj:
            record.own_opinion = _str_or_none(opinion_obj.get("own_opinion"))
            record.critical_view_engaged = _str_or_none(opinion_obj.get("critical_view_engaged"))
            record.opinion_disagrees = bool(opinion_obj.get("disagrees_somewhere", False))
            record.disagreement_locus = _str_or_none(opinion_obj.get("disagreement_locus"))

        # 3. Open-thread resolution / parking
        threads = list(getattr(comprehension, "open_threads", []) or [])
        if threads:
            threads_obj = await self._call_for_object(
                THREADS_PROMPT.format(title=title, threads="\n".join(f"- {t}" for t in threads)),
                on_failure=lambda: record.__setattr__("inference_failures", record.inference_failures + 1),
            )
            if threads_obj:
                record.resolved_threads = _list_of_dicts(threads_obj.get("resolved"))
                record.parked_threads = _list_of_dicts(threads_obj.get("parked"))

        # 4. Belief delta
        delta_obj = await self._call_for_object(
            BELIEF_DELTA_PROMPT.format(title=title, digest=digest),
            on_failure=lambda: record.__setattr__("inference_failures", record.inference_failures + 1),
        )
        if delta_obj:
            record.belief_updates = self._parse_belief_updates(delta_obj.get("belief_updates"), title)
            record.new_facts = self._parse_facts(delta_obj.get("new_facts"), title)

        record.substrate_after = self._snapshot_substrate()
        record.completed_at = time.time()
        return record

    # ── LLM call helpers ─────────────────────────────────────────────────

    async def _call_for_object(
        self,
        prompt: str,
        on_failure: Optional[Callable[[], None]] = None,
    ) -> Optional[Dict[str, Any]]:
        full_prompt = reasoning_aware_prompt_prefix(self._reasoning) + prompt
        raw = await self._call_llm(full_prompt)
        if not raw:
            if on_failure:
                on_failure()
            return None
        parsed = parse_reasoning_response(raw)
        return _safe_json_object(parsed.answer)

    async def _call_for_dict(
        self,
        prompt: str,
        keys: Sequence[str],
        on_failure: Optional[Callable[[], None]] = None,
    ) -> Dict[str, str]:
        obj = await self._call_for_object(prompt, on_failure)
        if not obj:
            return {k: "" for k in keys}
        out: Dict[str, str] = {}
        for k in keys:
            v = obj.get(k)
            if v is None:
                out[k] = ""
            else:
                out[k] = str(v).strip()
        return out

    async def _call_llm(self, prompt: str) -> str:
        if self._infer is None:
            return ""
        for fn_name in ("think", "complete", "ask", "generate"):
            fn = getattr(self._infer, fn_name, None)
            if fn is None:
                continue
            try:
                res = await fn(prompt) if asyncio.iscoroutinefunction(fn) else fn(prompt)
                if isinstance(res, str):
                    return res
                for attr in ("content", "text", "answer"):
                    val = getattr(res, attr, None)
                    if isinstance(val, str):
                        return val
                if isinstance(res, dict):
                    return str(res.get("content", res.get("text", "")) or "")
            except Exception as e:
                record_degradation('reflection_loop', e)
                logger.debug("inference %s failed: %s", fn_name, e)
                continue
        return ""

    # ── Substrate snapshot ───────────────────────────────────────────────

    def _snapshot_substrate(self) -> Dict[str, Any]:
        if self._substrate is None:
            return {}
        try:
            return dict(self._substrate() or {})
        except Exception as e:
            record_degradation('reflection_loop', e)
            logger.debug("substrate snapshot failed: %s", e)
            return {}

    # ── Digest builder ───────────────────────────────────────────────────

    def _build_digest(self, comprehension: Any) -> str:
        parts: List[str] = []
        unified = getattr(comprehension, "unified_summary", "")
        if unified:
            parts.append(f"Unified summary: {unified[:1500]}")
        checkpoints = getattr(comprehension, "checkpoints", []) or []
        if checkpoints:
            joined = []
            for cp in checkpoints[:8]:
                joined.append(
                    f"[{cp.method_source} p{cp.priority_level}] {cp.summary[:300]}"
                )
            parts.append("Per-source notes:\n" + "\n".join(joined))
        contras = getattr(comprehension, "cross_source_contradictions", []) or []
        if contras:
            parts.append("Cross-source contradictions: " + " | ".join(contras[:6]))
        return "\n\n".join(parts) or "(no comprehension digest available)"

    # ── Parsers ──────────────────────────────────────────────────────────

    def _parse_belief_updates(self, raw: Any, source_title: str) -> List[BeliefUpdate]:
        out: List[BeliefUpdate] = []
        if not isinstance(raw, list):
            return out
        for r in raw:
            if not isinstance(r, dict):
                continue
            topic = str(r.get("topic", "")).strip()
            position = str(r.get("new_position", "") or r.get("position", "")).strip()
            if not topic or not position:
                continue
            out.append(BeliefUpdate(
                topic=topic,
                position=position,
                rationale=str(r.get("rationale", "")).strip(),
                confidence=float(r.get("confidence", 0.5) or 0.5),
                contradicts=[c for c in [r.get("contradicts_prior")] if c],
            ))
        return out

    def _parse_facts(self, raw: Any, source_title: str) -> List[FactRecord]:
        out: List[FactRecord] = []
        if not isinstance(raw, list):
            return out
        for r in raw:
            if not isinstance(r, dict):
                continue
            fact = str(r.get("fact", "")).strip()
            if not fact:
                continue
            evidence = r.get("evidence")
            if isinstance(evidence, str):
                evidence_list = [evidence]
            elif isinstance(evidence, list):
                evidence_list = [str(e) for e in evidence]
            else:
                evidence_list = []
            out.append(FactRecord(
                fact=fact,
                evidence=evidence_list,
                confidence=float(r.get("confidence", 0.5) or 0.5),
                provisional=True,
                domain=source_title,
            ))
        return out


# ── Helpers ───────────────────────────────────────────────────────────────


def _safe_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    candidate = text.strip()
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start < 0 or end < 0 or end <= start:
        return None
    try:
        obj = json.loads(candidate[start : end + 1])
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        return None
    return None


def _str_or_none(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None


def _list_of_dicts(v: Any) -> List[Dict[str, str]]:
    if not isinstance(v, list):
        return []
    out: List[Dict[str, str]] = []
    for item in v:
        if isinstance(item, dict):
            out.append({k: str(item.get(k, "")) for k in item})
    return out
