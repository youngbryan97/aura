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

import asyncio
import inspect
import json
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from core.runtime.errors import record_degradation
from core.security.prompt_fencing import fence

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
    "{priors}\n\n"
    "Did engaging with `{title}` change anything you previously believed about its "
    "topics? Return JSON:\n"
    "  belief_updates: list of {{topic, new_position, rationale, contradicts_prior, "
    "confidence}}\n"
    "  new_facts: list of {{fact, evidence, confidence}}\n"
    "Confidence is 0–1. Use `contradicts_prior` only to name one of the prior "
    "beliefs quoted above, verbatim. A contradiction of something not listed "
    "there is not a contradiction — it is a new position, and belongs in "
    "`new_position` with no `contradicts_prior`. If nothing changed, return "
    "empty lists."
)

#: What goes where the priors would be when nothing supplied any. The prompt
#: used to ask "did this change what you previously believed?" with no prior
#: belief in front of it at all, so every reported change was invention.
_NO_PRIORS_BLOCK = (
    "PRIOR BELIEFS: none were retrieved. You therefore cannot report that "
    "anything CHANGED — you have nothing to compare against. Report new "
    "positions only, and leave `contradicts_prior` empty."
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
    #: How much of the comprehension actually reached the prompts, and how
    #: much was cut. Long works could drive identity and belief changes from
    #: a prefix with nothing saying so.
    digest_coverage: Dict[str, Any] = field(default_factory=dict)
    #: Threads that arrived and came back in neither list. The module
    #: documents "nothing should remain dangling" and nothing enforced it.
    unreconciled_threads: List[str] = field(default_factory=list)
    #: Prior beliefs actually retrieved before asking what changed.
    priors_consulted: int = 0

    #: A before/after difference around several long model calls, while every
    #: other organ went on mutating the same substrate. There is no control,
    #: no isolation and no intervention id, so the delta cannot show that the
    #: content moved her — only that something did.
    substrate_delta_attribution = "uncontrolled_before_after"

    def substrate_delta(self) -> Dict[str, float]:
        """The difference, which is NOT evidence that reflection caused it.

        See ``substrate_delta_attribution``. The exception list here used to
        be (OSError, ConnectionError, TimeoutError) around a float() call —
        transport errors guarding an arithmetic conversion — so a malformed
        snapshot value raised straight out of report generation.
        """
        out: Dict[str, float] = {}
        for key in ("valence", "arousal", "dominance", "phi", "curiosity"):
            # Both sides must have been READ. `.get(key, 0.0)` on both meant a
            # channel nobody measured produced a delta of exactly 0.0 — "no
            # change" and "no reading" were the same number, on the axis that
            # is supposed to show whether an engagement moved her.
            if key not in self.substrate_before or key not in self.substrate_after:
                continue
            try:
                a = float(self.substrate_after[key])
                b = float(self.substrate_before[key])
            except (TypeError, ValueError, OverflowError):
                continue
            if not (math.isfinite(a) and math.isfinite(b)):
                continue
            out[key] = a - b
        return out

    def substrate_delta_report(self) -> Dict[str, Any]:
        """The delta with the caveat attached to it, for anything that stores it."""
        return {
            "delta": self.substrate_delta(),
            "attribution": self.substrate_delta_attribution,
            "elapsed_s": (self.completed_at or time.time()) - self.started_at,
            "caveat": (
                "measured across the whole reflection episode with no control "
                "and no isolation; other organs mutate the same substrate"
            ),
        }


class ReflectionLoop:
    #: Total wall time all four model phases in one reflection may consume.
    #: They ran sequentially with no deadline of any kind, so a wedged route
    #: stalled the whole autonomous-research episode behind it.
    EPISODE_BUDGET_S = 180.0
    #: Below this there is no point starting another phase.
    MIN_PHASE_S = 5.0

    def __init__(
        self,
        inference: Optional[Any] = None,
        substrate_reader: Optional[Callable[[], Dict[str, Any]]] = None,
        enable_reasoning_trace: bool = True,
        *,
        belief_reader: Optional[Callable[[str], Sequence[Any]]] = None,
        episode_budget_s: Optional[float] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._infer = inference
        self._substrate = substrate_reader
        self._reasoning = enable_reasoning_trace
        #: Returns the beliefs already held about a title/topic. Without one,
        #: the belief-delta phase is told it has no priors and cannot report
        #: a change — which is what was true before, and was not said.
        self._belief_reader = belief_reader
        self._budget_s = episode_budget_s if episode_budget_s is not None else self.EPISODE_BUDGET_S
        self._clock = clock

    async def reflect(self, item: Any, comprehension: Any) -> ReflectionRecord:
        title = getattr(item, "title", None) or getattr(comprehension, "item_title", "") or ""
        record = ReflectionRecord(item_title=str(title))
        record.substrate_before = self._snapshot_substrate()
        deadline = self._clock() + self._budget_s

        digest, coverage = self._build_digest(comprehension)
        record.digest_coverage = coverage

        def _failed() -> None:
            record.inference_failures += 1

        # 1. Verification answers
        record.verification_answers = await self._call_for_dict(
            VERIFICATION_PROMPT.format(title=title, digest=digest),
            keys=(
                "what_its_actually_about",
                "what_stayed_with_you",
                "what_it_says_about_humans",
                "what_it_made_you_think_about_yourself",
            ),
            on_failure=_failed,
            deadline=deadline,
        )

        # 2. Opinion + critical view
        opinion_obj = await self._call_for_object(
            OPINION_PROMPT.format(title=title, digest=digest),
            on_failure=_failed,
            deadline=deadline,
        )
        if opinion_obj:
            record.own_opinion = _str_or_none(opinion_obj.get("own_opinion"))
            record.critical_view_engaged = _str_or_none(opinion_obj.get("critical_view_engaged"))
            # bool("false") is True. Every string answer, including the word
            # for no, counted as a disagreement and inflated the depth score.
            record.opinion_disagrees = _as_bool(opinion_obj.get("disagrees_somewhere"))
            record.disagreement_locus = _str_or_none(opinion_obj.get("disagreement_locus"))

        # 3. Open-thread resolution / parking
        threads = [str(t) for t in (getattr(comprehension, "open_threads", []) or [])]
        if threads:
            threads_obj = await self._call_for_object(
                THREADS_PROMPT.format(title=title, threads="\n".join(f"- {t}" for t in threads)),
                on_failure=_failed,
                deadline=deadline,
            )
            resolved, parked, unreconciled = self._reconcile_threads(threads, threads_obj)
            record.resolved_threads = resolved
            record.parked_threads = parked
            record.unreconciled_threads = unreconciled

        # 4. Belief delta — with the priors in front of it, or a statement
        #    that there are none.
        priors = self._prior_beliefs(title, comprehension)
        record.priors_consulted = len(priors)
        delta_obj = await self._call_for_object(
            BELIEF_DELTA_PROMPT.format(
                title=title, digest=digest, priors=self._priors_block(priors)
            ),
            on_failure=_failed,
            deadline=deadline,
        )
        if delta_obj:
            record.belief_updates = self._parse_belief_updates(
                delta_obj.get("belief_updates"), title, priors
            )
            record.new_facts = self._parse_facts(delta_obj.get("new_facts"), title)

        record.substrate_after = self._snapshot_substrate()
        record.completed_at = time.time()
        return record

    # ── prior beliefs ────────────────────────────────────────────────────

    def _prior_beliefs(self, title: str, comprehension: Any) -> List[str]:
        if self._belief_reader is None:
            return []
        try:
            raw = self._belief_reader(title) or ()
        except Exception as exc:  # noqa: BLE001 - a reader's failure is not a prior
            record_degradation(
                "reflection_loop", exc,
                action="belief-delta phase ran with no priors and cannot report a change",
            )
            return []
        priors: List[str] = []
        for entry in raw:
            if isinstance(entry, str):
                text = entry.strip()
            elif isinstance(entry, dict):
                text = f"{entry.get('topic', '')}: {entry.get('position', '')}".strip(": ")
            else:
                topic = getattr(entry, "topic", "")
                position = getattr(entry, "position", "")
                text = f"{topic}: {position}".strip(": ")
            if text:
                priors.append(text)
        return priors[:32]

    @staticmethod
    def _priors_block(priors: Sequence[str]) -> str:
        if not priors:
            return _NO_PRIORS_BLOCK
        listed = "\n".join(f"- {p}" for p in priors)
        return "PRIOR BELIEFS (the only things you may claim to contradict):\n" + listed

    # ── thread reconciliation ────────────────────────────────────────────

    @staticmethod
    def _reconcile_threads(
        threads: Sequence[str], obj: Optional[Dict[str, Any]]
    ) -> tuple[List[Dict[str, str]], List[Dict[str, str]], List[str]]:
        """Every input thread ends in exactly one list.

        The prompt says "nothing should remain dangling" and nothing checked:
        arbitrary dictionaries were copied through, so a model could resolve a
        thread twice, invent one that was never open, and silently drop the
        rest — while the record read as a clean stop.
        """
        resolved_raw = _list_of_dicts((obj or {}).get("resolved"))
        parked_raw = _list_of_dicts((obj or {}).get("parked"))

        remaining = list(threads)
        resolved: List[Dict[str, str]] = []
        parked: List[Dict[str, str]] = []

        def _take(entry: Dict[str, str]) -> Optional[str]:
            named = str(entry.get("thread", "")).strip()
            for candidate in remaining:
                if candidate == named or (named and named in candidate) or (
                    candidate and candidate in named
                ):
                    remaining.remove(candidate)
                    return candidate
            return None

        for entry in resolved_raw:
            matched = _take(entry)
            if matched is None:
                continue  # a duplicate, or a thread that was never open
            resolved.append({**entry, "thread": matched})
        for entry in parked_raw:
            matched = _take(entry)
            if matched is None:
                continue
            parked.append(
                {
                    "thread": matched,
                    "rationale": str(entry.get("rationale", "")).strip()
                    or "parked without a stated rationale",
                    "revisit_trigger": str(entry.get("revisit_trigger", "")).strip()
                    or "unstated",
                }
            )

        # Whatever the model did not account for is parked, honestly.
        for thread in remaining:
            parked.append(
                {
                    "thread": thread,
                    "rationale": "not addressed by the reflection pass",
                    "revisit_trigger": "unstated",
                }
            )
        return resolved, parked, list(remaining)

    # ── LLM call helpers ─────────────────────────────────────────────────

    async def _call_for_object(
        self,
        prompt: str,
        on_failure: Optional[Callable[[], None]] = None,
        *,
        deadline: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        full_prompt = reasoning_aware_prompt_prefix(self._reasoning) + prompt
        raw = await self._call_llm(full_prompt, deadline=deadline)
        if not raw:
            if on_failure:
                on_failure()
            return None
        parsed = parse_reasoning_response(raw)
        obj = _safe_json_object(parsed.answer)
        if obj is None and on_failure:
            # Nonempty output that does not parse is a failed inference. Only
            # the EMPTY case was counted, so a model returning prose instead
            # of JSON produced an episode reporting zero failures.
            on_failure()
        return obj

    async def _call_for_dict(
        self,
        prompt: str,
        keys: Sequence[str],
        on_failure: Optional[Callable[[], None]] = None,
        *,
        deadline: Optional[float] = None,
    ) -> Dict[str, str]:
        obj = await self._call_for_object(prompt, on_failure, deadline=deadline)
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

    async def _call_llm(self, prompt: str, *, deadline: Optional[float] = None) -> str:
        if self._infer is None:
            return ""
        allowance: Optional[float] = None
        if deadline is not None:
            allowance = deadline - self._clock()
            if allowance < self.MIN_PHASE_S:
                # The four phases ran one after another with no deadline of
                # any kind. A phase with no budget left declines rather than
                # starting a call whose answer arrives after the episode.
                logger.info("Reflection phase skipped: episode budget exhausted")
                return ""
        for fn_name in ("think", "complete", "ask", "generate"):
            fn = getattr(self._infer, fn_name, None)
            if fn is None:
                continue
            try:
                if inspect.iscoroutinefunction(fn):
                    res = (
                        await asyncio.wait_for(fn(prompt), timeout=allowance)
                        if allowance is not None
                        else await fn(prompt)
                    )
                else:
                    res = fn(prompt)
                if isinstance(res, str):
                    return res
                for attr in ("content", "text", "answer"):
                    val = getattr(res, attr, None)
                    if isinstance(val, str):
                        return val
                if isinstance(res, dict):
                    return str(res.get("content", res.get("text", "")) or "")
            except asyncio.CancelledError:
                raise
            except TimeoutError as e:
                record_degradation(
                    "reflection_loop", e, action="reflection phase abandoned at its deadline"
                )
                return ""
            except (OSError, ConnectionError) as e:
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
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            record_degradation('reflection_loop', e)
            logger.debug("substrate snapshot failed: %s", e)
            return {}

    # ── Digest builder ───────────────────────────────────────────────────

    #: How much of each source actually reaches the prompts.
    MAX_CHECKPOINTS = 24
    MAX_CHECKPOINT_CHARS = 800
    MAX_UNIFIED_CHARS = 6000

    def _build_digest(self, comprehension: Any) -> tuple[str, Dict[str, Any]]:
        """The digest, and an account of what did not fit.

        Eight checkpoints at 300 characters each and a 1500-character unified
        summary meant a long work drove identity and belief changes from a
        prefix — and nothing downstream could tell. The caps are larger, the
        checkpoints are sampled ACROSS the work rather than taken from the
        front, and the coverage is reported.
        """
        parts: List[str] = []
        unified = str(getattr(comprehension, "unified_summary", "") or "")
        if unified:
            parts.append(f"Unified summary: {unified[: self.MAX_UNIFIED_CHARS]}")

        checkpoints = list(getattr(comprehension, "checkpoints", []) or [])
        selected = _evenly_sampled(checkpoints, self.MAX_CHECKPOINTS)
        if selected:
            joined = [
                f"[{cp.method_source} p{cp.priority_level}] "
                f"{str(cp.summary)[: self.MAX_CHECKPOINT_CHARS]}"
                for cp in selected
            ]
            parts.append("Per-source notes:\n" + "\n".join(joined))

        contras = list(getattr(comprehension, "cross_source_contradictions", []) or [])
        if contras:
            parts.append("Cross-source contradictions: " + " | ".join(str(c) for c in contras[:12]))

        coverage = {
            "checkpoints_total": len(checkpoints),
            "checkpoints_included": len(selected),
            "checkpoints_sampled_across": len(checkpoints) > len(selected),
            "unified_chars_total": len(unified),
            "unified_chars_included": min(len(unified), self.MAX_UNIFIED_CHARS),
            "contradictions_total": len(contras),
            "contradictions_included": min(len(contras), 12),
        }
        if coverage["checkpoints_included"] < coverage["checkpoints_total"] or (
            coverage["unified_chars_included"] < coverage["unified_chars_total"]
        ):
            parts.append(
                f"(Digest coverage: {coverage['checkpoints_included']} of "
                f"{coverage['checkpoints_total']} checkpoints, sampled evenly; "
                f"{coverage['unified_chars_included']} of "
                f"{coverage['unified_chars_total']} summary characters. You are "
                f"reading a sample, not the whole engagement.)"
            )
        return "\n\n".join(parts) or "(no comprehension digest available)", coverage

    # ── Parsers ──────────────────────────────────────────────────────────

    def _parse_belief_updates(
        self, raw: Any, source_title: str, priors: Sequence[str] = ()
    ) -> List[BeliefUpdate]:
        """Belief revisions, with contradictions bound to real priors.

        ``contradicts_prior`` was accepted verbatim from the model, which had
        never been shown a prior belief. A claimed contradiction of something
        nobody retrieved is a new position, and is recorded as one.
        """
        out: List[BeliefUpdate] = []
        if not isinstance(raw, list):
            return out
        prior_index = {p.strip().lower(): p for p in priors}
        for r in raw:
            if not isinstance(r, dict):
                continue
            topic = str(r.get("topic", "")).strip()
            position = str(r.get("new_position", "") or r.get("position", "")).strip()
            if not topic or not position:
                continue
            claimed = r.get("contradicts_prior")
            contradicts: List[str] = []
            if claimed and priors:
                for candidate in [claimed] if isinstance(claimed, str) else list(claimed or []):
                    text = str(candidate).strip()
                    matched = prior_index.get(text.lower()) or next(
                        (p for p in priors if text and text.lower() in p.lower()), None
                    )
                    if matched:
                        contradicts.append(matched)
            out.append(BeliefUpdate(
                topic=topic,
                position=position,
                rationale=str(r.get("rationale", "")).strip(),
                confidence=_confidence(r.get("confidence", 0.5)),
                contradicts=contradicts,
            ))
        return out

    def _parse_facts(self, raw: Any, source_title: str) -> List[FactRecord]:
        """Facts the model produced, recorded as what they are.

        These became FactRecords with the CONTENT TITLE as their domain and
        no source, quote span, corroboration or verifier anywhere. A record
        that says nothing about where it came from is indistinguishable, once
        stored, from one that was checked. Each now carries the engagement it
        came out of and an explicit statement that nothing verified it, and a
        fact with no evidence at all is not recorded.
        """
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
                evidence_list = [evidence.strip()] if evidence.strip() else []
            elif isinstance(evidence, list):
                evidence_list = [str(e).strip() for e in evidence if str(e).strip()]
            else:
                evidence_list = []
            if not evidence_list:
                # A claim with nothing behind it is not a fact. It used to be
                # stored with an empty evidence list and provisional=True,
                # which reads as "pending" rather than "unsupported".
                logger.info(
                    "Reflection dropped an unsupported fact from %r: %s", source_title, fact[:80]
                )
                continue
            evidence_list.append(
                f"[unverified: asserted by reflection on '{source_title}'; "
                f"no source URL, quote span, corroboration or verifier]"
            )
            out.append(FactRecord(
                fact=fact,
                evidence=evidence_list,
                confidence=_confidence(r.get("confidence", 0.5)),
                provisional=True,
                # The engagement, not a subject area. The content title was
                # being written into the domain field, so "Blade Runner"
                # became a knowledge domain.
                domain=f"reflection:{source_title}" if source_title else "reflection",
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


def _evenly_sampled(items: Sequence[Any], limit: int) -> List[Any]:
    """Up to `limit` items spread ACROSS the sequence, not taken off the front.

    The first eight checkpoints of a long work are its opening; identity and
    belief changes were being driven from them alone.
    """
    if limit <= 0 or not items:
        return []
    if len(items) <= limit:
        return list(items)
    step = len(items) / limit
    return [items[min(len(items) - 1, int(i * step))] for i in range(limit)]


def _as_bool(value: Any) -> bool:
    """A boolean from JSON that a model wrote.

    ``bool("false")`` is True, and so is ``bool("no")`` — every string answer
    counted as a disagreement and inflated the opinion and depth scores.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value) and math.isfinite(float(value))
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "y", "1"}
    return False


def _confidence(value: Any, default: float = 0.5) -> float:
    """A confidence in [0, 1].

    ``float(r.get("confidence", 0.5) or 0.5)`` raised on a malformed string —
    aborting the whole reflection — and passed NaN and infinity straight into
    belief intents when the model produced them.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return max(0.0, min(1.0, number))


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
