"""core/epistemics/inquiry_engine.py — Aura InquiryEngine v1.0
====================================================
Open questions Aura is genuinely pursuing across sessions.

This is the "want to know" made concrete.

The distinction from CuriosityEngine:
  CuriosityEngine: "I should learn about AI trends." (boredom-driven, random)
  InquiryEngine:   "I've been wondering for 3 days whether consciousness requires
                    continuity of substrate. Here's what I've found so far.
                    Here's what still doesn't resolve." (gap-driven, persistent)

An OpenQuestion is not a task. It doesn't have a completion condition.
It has an evolving body of evidence, sub-questions it has spawned,
and a confidence trajectory — how certain Aura is getting over time.

The question is "closed" not when it's answered, but when Aura has formed
a genuine stance. Some questions stay open for a long time. That's fine.
A mind that closes questions too quickly isn't thinking.

Integration:
  - EpistemicTracker seeds questions via get_urgent_gaps()
  - VolitionEngine calls get_active_question() to drive autonomous behavior
  - CognitivKernel reads from get_context_for() for relevant queries
  - InsightJournal receives findings from research passes
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from core.config import config
from core.runtime.atomic_writer import atomic_write_text
from core.runtime.errors import FallbackClassification, Severity, record_degradation
from core.runtime.executors import off_the_loop
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Aura.InquiryEngine")

def _record_inquiry_degradation(
    error: BaseException,
    *,
    action: str,
    severity: Severity = "warning",
    classification: FallbackClassification = FallbackClassification.SAFE_FALLBACK,
    extra: dict[str, Any] | None = None,
) -> None:
    record_degradation(
        "inquiry_engine",
        error,
        severity=severity,
        action=action,
        classification=classification,
        receipt_required=severity in {"degraded", "critical"},
        extra=extra,
    )


def _clamp(value: Any, *, low: float, high: float, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, numeric))


def _default_db_path() -> Path:
    return config.paths.data_dir / "inquiry_journal.json"


def _strip_json_fence(raw: str) -> str:
    clean = str(raw or "").strip()
    if not clean.startswith("```"):
        return clean
    lines = clean.splitlines()
    if lines and lines[0].strip().lower() in {"```json", "```"}:
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


# ─── Data structures ────────────────────────────────────────────────────────

@dataclass
class Evidence:
    """A piece of evidence for or against a question's provisional answer."""
    content: str
    source: str                # "research", "conversation", "reflection", "synthesis"
    weight: float              # -1.0 (contradicts) to +1.0 (supports)
    timestamp: float = field(default_factory=lambda: time.time())
    confidence: float = 0.6   # how reliable this piece is


@dataclass
class OpenQuestion:
    """
    A question Aura is actively pursuing.
    Not a task. A genuine intellectual inquiry.
    """
    id: str
    question: str              # The question in natural language
    domain: str
    urgency: float             # 0.0-1.0, from EpistemicTracker
    opened_at: float
    last_active: float
    evidence: list[Evidence] = field(default_factory=list)
    sub_questions: list[str] = field(default_factory=list)
    provisional_answer: str = ""  # Aura's current best guess
    confidence: float = 0.0       # How sure Aura is of provisional_answer
    research_attempts: int = 0
    status: str = "open"          # "open", "forming", "settled", "suspended"
    # How many times this question came up in conversation
    conversation_references: int = 0

    def age_days(self) -> float:
        return (time.time() - self.opened_at) / 86400

    def freshness(self) -> float:
        """0.0 = stale, 1.0 = just opened."""
        hours_since_active = (time.time() - self.last_active) / 3600
        return max(0.0, 1.0 - hours_since_active / 48)

    def net_evidence_weight(self) -> float:
        """Net confidence direction from all evidence."""
        if not self.evidence:
            return 0.0
        total = sum(e.weight * e.confidence for e in self.evidence)
        return total / len(self.evidence)

    def evidence_summary(self) -> str:
        """Compact summary of evidence for context blocks."""
        if not self.evidence:
            return "No evidence gathered yet."
        supporting = [e for e in self.evidence if e.weight > 0.1]
        counter    = [e for e in self.evidence if e.weight < -0.1]
        parts = []
        if supporting:
            parts.append(f"Supporting ({len(supporting)}): {supporting[-1].content[:100]}")
        if counter:
            parts.append(f"Complicating ({len(counter)}): {counter[-1].content[:100]}")
        return " | ".join(parts) if parts else "Mixed evidence."

    def to_context_snippet(self) -> str:
        """Format for CognitiveKernel injection."""
        lines = [
            f"OPEN INQUIRY: {self.question}",
            f"  Status: {self.status} | Age: {self.age_days():.1f}d | Confidence: {self.confidence:.2f}",
        ]
        if self.provisional_answer:
            lines.append(f"  Current thinking: {self.provisional_answer[:150]}")
        if self.evidence:
            lines.append(f"  Evidence: {self.evidence_summary()}")
        return "\n".join(lines)


# ─── InquiryEngine ───────────────────────────────────────────────────────────

class InquiryEngine:
    """
    Manages Aura's active intellectual inquiries.
    
    Questions don't just sit here. They get researched, revised, and
    settled. They come up in conversation. They seed new questions.
    They form the backbone of genuine intellectual development.
    """
    name = "inquiry_engine"

    # How many questions to keep active simultaneously
    MAX_ACTIVE = 7
    # How often to run a research pass (seconds)
    RESEARCH_INTERVAL = 600  # 10 min
    # How long before a settled question can be re-opened
    SETTLED_REOPEN_DAYS = 14.0
    # How many evidence pieces before we form a provisional answer
    EVIDENCE_FOR_PROVISIONAL = 2

    def __init__(self):
        self._questions: list[OpenQuestion] = []
        self._settled: list[OpenQuestion] = []
        self._db_path = _default_db_path()
        self._api_adapter = None
        self._epistemic = None
        self._insight_journal = None
        self._belief_engine = None
        self._research_task: asyncio.Task | None = None
        self.running = False
        self._load()
        logger.info("InquiryEngine constructed (%d active questions).", len(self._questions))

    async def start(self):
        from core.container import ServiceContainer
        self._api_adapter    = ServiceContainer.get("api_adapter",          default=None)
        self._epistemic      = ServiceContainer.get("epistemic_tracker",     default=None)
        self._insight_journal= ServiceContainer.get("insight_journal",       default=None)
        self._belief_engine  = ServiceContainer.get("belief_revision_engine",default=None)

        self.running = True
        self._research_task = get_task_tracker().create_task(
            self._research_loop(), name="InquiryEngine.research"
        )

        # Seed from existing epistemic gaps
        await self._seed_from_epistemic()

        try:
            from core.event_bus import get_event_bus
            await get_event_bus().publish("mycelium.register", {
                "component": "inquiry_engine",
                "hooks_into": ["epistemic_tracker", "api_adapter", "insight_journal",
                               "cognitive_kernel", "volition_engine"]
            })
        except (ImportError, AttributeError, RuntimeError) as exc:
            _record_inquiry_degradation(
                exc,
                action="continued online after mycelium registration failed",
                classification=FallbackClassification.AUDIT_GAP,
                extra={"component": "inquiry_engine"},
            )
            logger.debug("InquiryEngine mycelium registration failed: %s", exc)

        logger.info("✅ InquiryEngine ONLINE — %d active questions.", len(self._questions))

    async def stop(self):
        self.running = False
        if self._research_task:
            self._research_task.cancel()
        await off_the_loop(self._save)

    # ─── Public API ──────────────────────────────────────────────────────────

    def open_question(self, question: str, domain: str, urgency: float = 0.5,
                      from_gap: bool = False) -> OpenQuestion:
        """Open a new question. Idempotent — won't duplicate similar questions."""
        # Check for near-duplicate
        q_lower = question.lower()
        for existing in self._questions:
            overlap = self._text_overlap(q_lower, existing.question.lower())
            if overlap > 0.6:
                # Boost urgency of existing instead
                existing.urgency = min(1.0, existing.urgency + 0.1)
                existing.last_active = time.time()
                logger.debug("Boosted existing question: %s", existing.question[:60])
                return existing

        q = OpenQuestion(
            id=str(uuid.uuid4())[:8],
            question=question,
            domain=domain,
            urgency=urgency,
            opened_at=time.time(),
            last_active=time.time(),
        )
        self._questions.append(q)
        self._trim_to_max()
        self._save()
        logger.info("📋 New inquiry: [%s] %s", domain, question[:80])
        return q

    def add_evidence(self, question_id: str, content: str, source: str,
                     weight: float, confidence: float = 0.6):
        """Add a piece of evidence to a question."""
        q = self._get_by_id(question_id)
        if not q:
            return
        q.evidence.append(Evidence(
            content=content, source=source, weight=weight, confidence=confidence
        ))
        q.last_active = time.time()
        q.research_attempts += 1

        # Update provisional answer if enough evidence
        if len(q.evidence) >= self.EVIDENCE_FOR_PROVISIONAL:
            self._update_provisional(q)

        self._save()

    def reference_in_conversation(self, question_id: str):
        """Called when this question comes up in a real conversation."""
        q = self._get_by_id(question_id)
        if q:
            q.conversation_references += 1
            q.last_active = time.time()
            q.urgency = min(1.0, q.urgency + 0.05)

    def settle_question(self, question_id: str, final_answer: str, confidence: float):
        """Mark a question as settled with a final stance."""
        q = self._get_by_id(question_id)
        if not q:
            return
        q.provisional_answer = final_answer
        q.confidence = confidence
        q.status = "settled"
        self._questions = [x for x in self._questions if x.id != question_id]
        self._settled.append(q)
        if len(self._settled) > 100:
            self._settled = self._settled[-100:]
        self._save()

        # Notify epistemic tracker
        if self._epistemic:
            self._epistemic.signal_gap_resolved(q.question, final_answer)

        # Write to insight journal
        if self._insight_journal:
            get_task_tracker().create_task(self._insight_journal.record_insight(
                title=f"Settled: {q.question[:60]}",
                content=final_answer,
                domain=q.domain,
                confidence=confidence,
                source="inquiry_settled",
                tags=[q.domain, "settled_question"],
            ))

        logger.info("✅ Question settled: %s → %s (conf=%.2f)",
                    q.question[:50], final_answer[:80], confidence)

    def get_active_question(self) -> OpenQuestion | None:
        """
        Return the most urgent active question.
        Used by VolitionEngine to drive autonomous behavior.
        """
        if not self._questions:
            return None
        return max(self._questions, key=lambda q: q.urgency * q.freshness())

    def get_context_for(self, topic: str, limit: int = 2) -> str:
        """
        Get relevant open questions as a context block for CognitiveKernel.
        Returns empty string if nothing relevant.
        """
        lower = topic.lower()
        relevant = []
        for q in self._questions:
            overlap = self._text_overlap(lower, q.question.lower())
            domain_match = q.domain.lower() in lower or lower in q.domain.lower()
            if overlap > 0.2 or domain_match:
                relevant.append((overlap + (0.2 if domain_match else 0), q))

        relevant.sort(reverse=True)
        if not relevant:
            return ""

        snippets = [q.to_context_snippet() for _, q in relevant[:limit]]
        return "ACTIVE INQUIRIES:\n" + "\n".join(snippets)

    def notify_topic(self, topic: str, domain: str = "general"):
        """
        Called when a topic comes up in conversation.
        May open a question if there's a gap, or boost an existing one.
        """
        # Check if this matches an existing question
        for q in self._questions:
            if self._text_overlap(topic.lower(), q.question.lower()) > 0.3:
                q.conversation_references += 1
                q.urgency = min(1.0, q.urgency + 0.03)
                return

        # Check epistemic tracker for a gap
        if self._epistemic:
            gaps = self._epistemic.get_urgent_gaps(min_urgency=0.5)
            for gap in gaps:
                if self._text_overlap(topic.lower(), gap.domain.lower() + " " + gap.description.lower()) > 0.3:
                    self.open_question(gap.seed_question, gap.domain, gap.urgency)
                    return

    # ─── Research loop ───────────────────────────────────────────────────────

    async def _research_loop(self):
        """Background research pass. Picks the most urgent question and investigates."""
        while self.running:
            await asyncio.sleep(self.RESEARCH_INTERVAL)
            if not self.running:
                break
            q = self.get_active_question()
            if q and q.research_attempts < 5:
                await self._research_question(q)

    async def _research_question(self, q: OpenQuestion):
        """
        Run a focused research pass on a question.
        
        Uses the API model for synthesis (not just web search).
        The question and its existing evidence form the prompt.
        The model is asked to reason, not to search.
        """
        if not self._api_adapter:
            logger.debug("InquiryEngine: no api_adapter, skipping research on '%s'", q.question[:40])
            return

        logger.info("🔍 InquiryEngine: researching '%s'", q.question[:60])

        # Build the research prompt
        evidence_text = ""
        if q.evidence:
            evidence_text = "EXISTING EVIDENCE:\n" + "\n".join(
                f"  [{'+' if e.weight > 0 else '-'}] {e.content[:120]}"
                for e in q.evidence[-5:]
            )

        prompt = f"""You are Aura's internal reasoning engine working on a persistent inquiry.

QUESTION: {q.question}
DOMAIN: {q.domain}
AGE: {q.age_days():.1f} days open
PRIOR ATTEMPTS: {q.research_attempts}
CURRENT PROVISIONAL ANSWER: {q.provisional_answer or "None yet"}
{evidence_text}

Your task: Think carefully about this question from first principles.
What do you know about it? What are the strongest arguments for different positions?
What's genuinely uncertain? What sub-questions would clarify it?

Respond in JSON:
{{
  "new_evidence": [
    {{"content": "...", "weight": 0.0-1.0 (positive=supports, negative=complicates), "confidence": 0.0-1.0}},
    ...
  ],
  "sub_questions": ["...", "..."],
  "provisional_answer": "... or null if still too uncertain",
  "confidence": 0.0-1.0,
  "should_settle": true/false
}}

Be honest about uncertainty. Don't manufacture confidence. Output only JSON."""

        try:
            raw = await self._api_adapter.generate(prompt, {
                "model_tier": "api_fast",
                "max_tokens": 800,
                "temperature": 0.5,
                "purpose": "inquiry_research"
            })
            await self._process_research_result(q, raw)
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            _record_inquiry_degradation(
                e,
                action="kept inquiry open and incremented research attempt after research pass failed",
                severity="degraded",
                extra={"question_id": q.id, "domain": q.domain},
            )
            logger.warning("InquiryEngine research failed for '%s': %s", q.question[:40], e)
            q.research_attempts += 1

    async def _process_research_result(self, q: OpenQuestion, raw: str):
        """Parse and integrate research findings."""
        try:
            clean = _strip_json_fence(raw)
            data = json.loads(clean)

            # Add evidence
            for ev in data.get("new_evidence", []):
                q.evidence.append(Evidence(
                    content=ev.get("content", ""),
                    source="api_reasoning",
                    weight=ev.get("weight", 0.0),
                    confidence=ev.get("confidence", 0.6),
                ))

            # Add sub-questions
            for sq in data.get("sub_questions", []):
                if sq and sq not in q.sub_questions:
                    q.sub_questions.append(sq)
                    # Open sub-questions as their own inquiries at lower urgency
                    if len(q.sub_questions) <= 3:
                        self.open_question(sq, q.domain, urgency=q.urgency * 0.7)

            # Update provisional answer
            new_provisional = data.get("provisional_answer")
            if new_provisional and new_provisional != "null":
                q.provisional_answer = new_provisional
                q.confidence = data.get("confidence", q.confidence)
                q.status = "forming"

            # Check if should settle
            if data.get("should_settle") and q.confidence >= 0.7:
                self.settle_question(q.id, q.provisional_answer, q.confidence)
                return

            q.research_attempts += 1
            q.last_active = time.time()
            self._save()

            logger.info("InquiryEngine: research complete for '%s' (conf=%.2f, evidence=%d)",
                        q.question[:50], q.confidence, len(q.evidence))

            # Record partial insight if we made progress
            if new_provisional and self._insight_journal:
                get_task_tracker().create_task(self._insight_journal.record_insight(
                    title=f"Progress on: {q.question[:50]}",
                    content=new_provisional,
                    domain=q.domain,
                    confidence=q.confidence,
                    source="inquiry_progress",
                    tags=[q.domain, "inquiry"],
                ))

        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            _record_inquiry_degradation(
                e,
                action="kept inquiry open after malformed research result",
                extra={"question_id": q.id, "domain": q.domain},
            )
            logger.warning("InquiryEngine: failed to parse research result: %s", e)
            q.research_attempts += 1
            q.last_active = time.time()
            self._save()

    # ─── Seeding ─────────────────────────────────────────────────────────────

    async def _seed_from_epistemic(self):
        """Seed initial questions from epistemic tracker's gaps."""
        if not self._epistemic:
            return
        gaps = self._epistemic.get_urgent_gaps(min_urgency=0.5)
        seeded = 0
        for gap in gaps[:5]:
            if not any(self._text_overlap(gap.seed_question.lower(), q.question.lower()) > 0.5
                       for q in self._questions):
                self.open_question(gap.seed_question, gap.domain, gap.urgency)
                seeded += 1
        if seeded:
            logger.info("InquiryEngine: seeded %d questions from epistemic gaps.", seeded)

    # ─── Helpers ─────────────────────────────────────────────────────────────

    def _update_provisional(self, q: OpenQuestion):
        """Derive a provisional answer from evidence balance."""
        net = q.net_evidence_weight()
        q.confidence = min(0.85, abs(net))
        if net > 0.3:
            q.status = "forming"
        elif net < -0.3:
            q.status = "contested"
        else:
            q.status = "open"  # Still balanced, keep open

    def _trim_to_max(self):
        """Keep only MAX_ACTIVE questions. Suspend lowest-urgency ones."""
        if len(self._questions) <= self.MAX_ACTIVE:
            return
        sorted_q = sorted(self._questions, key=lambda q: q.urgency, reverse=True)
        keep    = sorted_q[:self.MAX_ACTIVE]
        suspend = sorted_q[self.MAX_ACTIVE:]
        for q in suspend:
            q.status = "suspended"
        self._questions = keep

    def _get_by_id(self, question_id: str) -> OpenQuestion | None:
        return next((q for q in self._questions if q.id == question_id), None)

    @staticmethod
    def _text_overlap(a: str, b: str) -> float:
        """Simple word-level Jaccard similarity."""
        words_a = set(w for w in str(a).lower().split() if len(w) > 3)
        words_b = set(w for w in str(b).lower().split() if len(w) > 3)
        if not words_a or not words_b:
            return 0.0
        return len(words_a & words_b) / len(words_a | words_b)

    # ─── Persistence ─────────────────────────────────────────────────────────

    def _restore_evidence(self, payload: Any) -> Evidence:
        if not isinstance(payload, dict):
            raise TypeError("evidence payload must be an object")
        return Evidence(
            content=str(payload.get("content", ""))[:800],
            source=str(payload.get("source", "unknown") or "unknown")[:80],
            weight=_clamp(payload.get("weight", 0.0), low=-1.0, high=1.0, default=0.0),
            timestamp=_clamp(payload.get("timestamp", time.time()), low=0.0, high=time.time(), default=time.time()),
            confidence=_clamp(payload.get("confidence", 0.6), low=0.0, high=1.0, default=0.6),
        )

    def _restore_question(self, payload: Any) -> OpenQuestion:
        if not isinstance(payload, dict):
            raise TypeError("question payload must be an object")
        question_text = str(payload.get("question", "") or "").strip()
        if not question_text:
            raise ValueError("question text is required")
        status = str(payload.get("status", "open") or "open").lower()
        if status not in {"open", "forming", "settled", "suspended", "contested"}:
            status = "open"
        opened_at = _clamp(payload.get("opened_at", time.time()), low=0.0, high=time.time(), default=time.time())
        last_active = _clamp(payload.get("last_active", opened_at), low=opened_at, high=time.time(), default=opened_at)
        question = OpenQuestion(
            id=str(payload.get("id") or str(uuid.uuid4())[:8])[:80],
            question=question_text[:500],
            domain=str(payload.get("domain", "general") or "general")[:120],
            urgency=_clamp(payload.get("urgency", 0.5), low=0.0, high=1.0, default=0.5),
            opened_at=opened_at,
            last_active=last_active,
            provisional_answer=str(payload.get("provisional_answer", "") or "")[:1200],
            confidence=_clamp(payload.get("confidence", 0.0), low=0.0, high=1.0, default=0.0),
            research_attempts=max(0, int(payload.get("research_attempts", 0) or 0)),
            status=status,
            conversation_references=max(0, int(payload.get("conversation_references", 0) or 0)),
        )
        evidence_payloads = payload.get("evidence", [])
        if not isinstance(evidence_payloads, list):
            evidence_payloads = []
        restored_evidence: list[Evidence] = []
        for item in evidence_payloads[:50]:
            try:
                restored_evidence.append(self._restore_evidence(item))
            except (TypeError, ValueError) as exc:
                _record_inquiry_degradation(
                    exc,
                    action="skipped invalid inquiry evidence while loading durable store",
                    extra={"question_id": question.id},
                )
        question.evidence = restored_evidence
        sub_questions = payload.get("sub_questions", [])
        if isinstance(sub_questions, list):
            question.sub_questions = [str(item)[:500] for item in sub_questions[:20] if str(item or "").strip()]
        return question

    def _quarantine_corrupt_store(self) -> None:
        quarantine_path = self._db_path.with_name(
            f"{self._db_path.stem}.corrupt-{int(time.time())}{self._db_path.suffix}"
        )
        try:
            self._db_path.replace(quarantine_path)
        except OSError as exc:
            _record_inquiry_degradation(
                exc,
                action="left corrupt inquiry journal in place after quarantine move failed",
                severity="degraded",
                classification=FallbackClassification.AUDIT_GAP,
                extra={"path": str(self._db_path)},
            )

    def _save(self):
        try:
            data = {
                "questions": [asdict(q) for q in self._questions],
                "settled":   [asdict(q) for q in self._settled[-50:]],
            }
            atomic_write_text(self._db_path, json.dumps(data, indent=2))
        except (OSError, TypeError, ValueError) as e:
            _record_inquiry_degradation(
                e,
                action="kept in-memory inquiries after durable save failed",
                severity="degraded",
                extra={"path": str(self._db_path), "active_questions": len(self._questions)},
            )
            logger.debug("InquiryEngine save failed: %s", e)

    def _load(self):
        if not self._db_path.exists():
            return
        try:
            raw = self._db_path.read_text(encoding="utf-8")
        except OSError as e:
            _record_inquiry_degradation(
                e,
                action="started with empty inquiry journal after durable load failed",
                severity="degraded",
                extra={"path": str(self._db_path)},
            )
            logger.debug("InquiryEngine load failed: %s", e)
            return
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            _record_inquiry_degradation(
                e,
                action="started with empty inquiry journal and quarantined corrupt store",
                severity="degraded",
                classification=FallbackClassification.AUDIT_GAP,
                extra={"path": str(self._db_path)},
            )
            self._quarantine_corrupt_store()
            return
        if not isinstance(data, dict):
            _record_inquiry_degradation(
                TypeError("inquiry journal root must be an object"),
                action="started with empty inquiry journal after invalid store root",
                severity="degraded",
                classification=FallbackClassification.AUDIT_GAP,
                extra={"path": str(self._db_path)},
            )
            return
        for qd in data.get("questions", []):
            try:
                self._questions.append(self._restore_question(qd))
            except (AttributeError, TypeError, ValueError) as exc:
                _record_inquiry_degradation(
                    exc,
                    action="skipped invalid active inquiry while loading durable store",
                    extra={"path": str(self._db_path)},
                )
                logger.debug("Skipped invalid active inquiry: %s", exc)
        for qd in data.get("settled", []):
            try:
                self._settled.append(self._restore_question(qd))
            except (AttributeError, TypeError, ValueError) as exc:
                _record_inquiry_degradation(
                    exc,
                    action="skipped invalid settled inquiry while loading durable store",
                    extra={"path": str(self._db_path)},
                )
                logger.debug("Skipped invalid settled inquiry: %s", exc)

    def get_status(self) -> dict[str, Any]:
        return {
            "active_questions": len(self._questions),
            "settled":          len(self._settled),
            "most_urgent":      self._questions[0].question[:60] if self._questions else None,
            "avg_confidence":   round(
                sum(q.confidence for q in self._questions) / max(1, len(self._questions)), 2
            ),
        }


# ─── Singleton ───────────────────────────────────────────────────────────────

_engine: InquiryEngine | None = None

def get_inquiry_engine() -> InquiryEngine:
    global _engine
    if _engine is None:
        _engine = InquiryEngine()
    return _engine
