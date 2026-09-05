"""Durable conversational continuity.

The rolling summary this replaces was extractive and lossy in a compounding
way: ``AuraState._summarize_messages`` kept the last 8 messages truncated to
160 characters each, and every compaction re-truncated the accumulated result
(``Earlier: {prior[:600]} Recent: {new[:800]}``).  Two compactions in, a
sentence the user said about themselves had become a prefix of a prefix; five
in, it was gone with nothing recording that it had ever existed.  That is the
mechanism behind "she is great for a few turns, loses the plot, is great
again, and it gets worse every time".

A ledger fixes the shape of the loss rather than its size.  Entries are whole
propositions with provenance.  When the budget is exceeded, whole entries are
*evicted* by salience and counted — never shortened.  Forgetting becomes an
explicit, measurable act instead of a silent truncation, so what survives is
always something she can actually use.

Three properties matter for indefinite coherence:

* **Bounded.**  Total rendered size is capped, so a 500-turn conversation
  costs no more prompt than a 50-turn one.
* **Non-decaying.**  A retained entry is byte-identical to when it was first
  written.  Re-summarising a summary is what destroyed information before.
* **Attributed.**  Every entry records the turn it came from, so a claim built
  on it can be checked rather than confabulated.
"""

from __future__ import annotations

import logging
import os
import re
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.Continuity.Ledger")

# ── Levers ────────────────────────────────────────────────────────────────
# Every budget here is a named lever. They are read at call time, not import
# time, so a live runtime picks up a change without a reboot.


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError) as exc:
        record_degradation(
            "continuity_ledger.lever",
            exc,
            severity="warning",
            action=f"used the default for {name} after failing to parse {raw!r}",
            enforce_failure_policy=False,
        )
        return default


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return float(str(raw).strip())
    except (TypeError, ValueError) as exc:
        record_degradation(
            "continuity_ledger.lever",
            exc,
            severity="warning",
            action=f"used the default for {name} after failing to parse {raw!r}",
            enforce_failure_policy=False,
        )
        return default


#: Hard ceiling on the rendered ledger, in characters. This is the whole
#: point of the structure: continuity has a fixed price regardless of depth.
def ledger_budget_chars() -> int:
    return max(0, env_int("AURA_CONTINUITY_LEDGER_CHARS", 3200))


#: How many entries may be held before salience eviction runs. Held entries
#: are cheap (they are not all rendered); this bounds memory, not prompt.
def ledger_capacity() -> int:
    return max(8, env_int("AURA_CONTINUITY_LEDGER_CAPACITY", 240))


#: Kind weights decide what survives an eviction. A thing the user said about
#: themselves outranks a topic marker, because losing it is what makes her
#: sound like she has never met them.
def _kind_weights() -> dict[str, float]:
    return {
        "disclosure": env_float("AURA_CONTINUITY_W_DISCLOSURE", 3.0),
        "commitment": env_float("AURA_CONTINUITY_W_COMMITMENT", 2.6),
        "position": env_float("AURA_CONTINUITY_W_POSITION", 2.0),
        "question": env_float("AURA_CONTINUITY_W_QUESTION", 1.6),
        "subject": env_float("AURA_CONTINUITY_W_SUBJECT", 1.2),
    }


_MAX_ENTRY_CHARS = 320


@dataclass
class LedgerEntry:
    """One whole proposition, with the provenance needed to defend it."""

    kind: str
    text: str
    speaker: str
    first_turn: int
    last_turn: int
    mentions: int = 1
    pinned: bool = False
    created_at: float = field(default_factory=time.time)

    def key(self) -> str:
        return _normalise(self.text)

    def salience(self, now_turn: int) -> float:
        """Rank for eviction. Higher survives.

        Recency is a gentle decay rather than a cliff — an early self
        disclosure ("I've always wanted to learn physics") must still outrank
        a recent topic marker fifty turns later, which is exactly the case
        the truncating summary got wrong.
        """
        if self.pinned:
            return float("inf")
        weight = _kind_weights().get(self.kind, 1.0)
        age = max(0, int(now_turn) - int(self.last_turn))
        half_life = max(1.0, env_float("AURA_CONTINUITY_HALF_LIFE_TURNS", 60.0))
        recency = 0.5 ** (age / half_life)
        repetition = 1.0 + min(3.0, 0.5 * max(0, self.mentions - 1))
        return weight * repetition * (0.35 + 0.65 * recency)


def _normalise(text: str) -> str:
    return " ".join(str(text or "").lower().split())


# ── Extraction ────────────────────────────────────────────────────────────
# Cheap and structural. This runs at compaction time, off the generation hot
# path, and must never force a model call there: an observability step that
# synchronises inside the generation path is what cost this runtime a 20x
# prefill penalty once already.

_DISCLOSURE = re.compile(
    r"\b(i|i'?m|i'?ve|my|mine|me)\b.{0,200}?"
    r"\b(am|was|have|had|want|wanted|like|liked|love|loved|hate|hated|prefer|"
    r"need|needed|think|believe|feel|felt|work|worked|live|lived|studied|"
    r"learning|trying|always|never|really)\b",
    re.IGNORECASE,
)
_POSITION = re.compile(
    r"\b(i think|i believe|in my view|my position|i'd argue|i would argue|"
    r"i see myself|i don'?t think|i disagree|i'm not)\b",
    re.IGNORECASE,
)
_COMMITMENT = re.compile(
    r"\b(i'?ll|i will|let me|i'?m going to|i promise|next step|i'?ll remember)\b",
    re.IGNORECASE,
)
_SENTENCE = re.compile(r"[^.!?\n]+[.!?]?")


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE.findall(str(text or "")) if s.strip()]


def classify(sentence: str, role: str) -> str | None:
    """Name what a sentence is, or None when it carries no continuity value."""
    stripped = sentence.strip()
    if len(stripped) < 12:
        return None
    if role == "user":
        if _DISCLOSURE.search(stripped):
            return "disclosure"
        if stripped.endswith("?"):
            return "question"
        return "subject"
    if role == "assistant":
        if _COMMITMENT.search(stripped):
            return "commitment"
        if _POSITION.search(stripped):
            return "position"
        return None
    return None


@dataclass
class ContinuityLedger:
    """Bounded, non-decaying record of what this conversation established."""

    entries: list[LedgerEntry] = field(default_factory=list)
    turn: int = 0
    subject_trail: list[str] = field(default_factory=list)
    evicted_count: int = 0

    # ── ingest ────────────────────────────────────────────────────────────
    def observe(self, messages: Iterable[dict[str, Any]]) -> int:
        """Fold messages into the ledger. Returns entries added."""
        added = 0
        for message in messages or []:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "") or "").strip().lower()
            if role not in ("user", "assistant"):
                continue
            metadata = message.get("metadata", {}) or {}
            if isinstance(metadata, dict) and metadata.get("synthetic_summary"):
                continue
            content = " ".join(str(message.get("content", "") or "").split())
            if not content:
                continue
            self.turn += 1
            for sentence in _sentences(content):
                kind = classify(sentence, role)
                if kind is None:
                    continue
                if self._record(kind, sentence[:_MAX_ENTRY_CHARS], role):
                    added += 1
            if role == "user":
                self._note_subject(content)
        self._evict()
        return added

    def _record(self, kind: str, text: str, speaker: str) -> bool:
        key = _normalise(text)
        if not key:
            return False
        for entry in self.entries:
            if entry.key() == key:
                entry.mentions += 1
                entry.last_turn = self.turn
                return False
        self.entries.append(
            LedgerEntry(
                kind=kind,
                text=text,
                speaker=speaker,
                first_turn=self.turn,
                last_turn=self.turn,
            )
        )
        return True

    def _note_subject(self, content: str) -> None:
        subject = content[:120]
        if self.subject_trail and _normalise(self.subject_trail[-1]) == _normalise(subject):
            return
        self.subject_trail.append(subject)
        keep = max(2, env_int("AURA_CONTINUITY_SUBJECT_TRAIL", 6))
        if len(self.subject_trail) > keep:
            self.subject_trail = self.subject_trail[-keep:]

    def _evict(self) -> None:
        capacity = ledger_capacity()
        if len(self.entries) <= capacity:
            return
        ranked = sorted(self.entries, key=lambda e: e.salience(self.turn), reverse=True)
        survivors = ranked[:capacity]
        self.evicted_count += len(self.entries) - len(survivors)
        survivor_ids = {id(e) for e in survivors}
        # Preserve chronological order among survivors; ordering by salience
        # would scramble the narrative she reads back.
        self.entries = [e for e in self.entries if id(e) in survivor_ids]

    def pin(self, text: str) -> None:
        """Mark something as never-forgettable (an explicit 'remember this')."""
        key = _normalise(text)
        for entry in self.entries:
            if entry.key() == key:
                entry.pinned = True
                return
        self.entries.append(
            LedgerEntry(
                kind="disclosure",
                text=str(text)[:_MAX_ENTRY_CHARS],
                speaker="user",
                first_turn=self.turn,
                last_turn=self.turn,
                pinned=True,
            )
        )

    # ── render ────────────────────────────────────────────────────────────
    def render(self, budget_chars: int | None = None, *, speaker_name: str = "They") -> str:
        """Render within budget by dropping whole entries, never cutting one.

        A half-sentence is worse than no sentence: it reads as a fact she can
        complete, and she completes it by inventing the rest.

        The budget is a ceiling on the *returned string*, footer included —
        a renderer that overshoots its own cap silently re-creates the
        overflow it exists to prevent.
        """
        budget = ledger_budget_chars() if budget_chars is None else max(0, int(budget_chars))
        if budget <= 0 or not self.entries:
            return ""

        header = "## WHAT THIS CONVERSATION HAS ESTABLISHED\n"
        # Reserve the footer before admitting entries, so the cap holds whether
        # or not anything ends up omitted.
        footer_reserve = 96 + 2
        usable = budget - footer_reserve
        if usable <= len(header):
            return ""

        used = len(header)
        dropped = 0

        ranked = sorted(self.entries, key=lambda e: e.salience(self.turn), reverse=True)
        chosen: list[LedgerEntry] = []
        for entry in ranked:
            rendered = self._render_entry(entry, speaker_name)
            if used + len(rendered) + 1 > usable:
                dropped += 1
                continue
            chosen.append(entry)
            used += len(rendered) + 1

        if not chosen:
            return ""

        order = {id(e): i for i, e in enumerate(self.entries)}
        lines = [
            self._render_entry(entry, speaker_name)
            for entry in sorted(chosen, key=lambda e: order.get(id(e), 0))
        ]

        block = header + "\n".join(lines)
        missing = dropped + self.evicted_count
        if missing:
            # Say what is missing. A silent gap invites her to fill it.
            block += (
                f"\n({missing} earlier point(s) not shown — "
                "ask rather than assume what they were.)"
            )
        return (block + "\n\n")[:budget]

    @staticmethod
    def _render_entry(entry: LedgerEntry, speaker_name: str = "They") -> str:
        who = speaker_name if entry.speaker == "user" else "You"
        label = {
            "disclosure": "said about himself",
            "question": "asked",
            "subject": "raised",
            "commitment": "committed",
            "position": "took the position",
        }.get(entry.kind, "noted")
        return f"- [t{entry.first_turn}] {who} {label}: {entry.text}"

    def render_subject_trail(self) -> str:
        if not self.subject_trail:
            return ""
        recent = self.subject_trail[-3:]
        return (
            "## CURRENT THREAD\n"
            + "\n".join(f"- {s}" for s in recent)
            + "\n\n"
        )

    # ── persistence ───────────────────────────────────────────────────────
    def to_dict(self) -> dict[str, Any]:
        return {
            "entries": [asdict(e) for e in self.entries],
            "turn": int(self.turn),
            "subject_trail": list(self.subject_trail),
            "evicted_count": int(self.evicted_count),
        }

    @classmethod
    def from_dict(cls, payload: Any) -> ContinuityLedger:
        if not isinstance(payload, dict):
            return cls()
        try:
            entries = [
                LedgerEntry(**{k: v for k, v in item.items() if k in LedgerEntry.__annotations__})
                for item in payload.get("entries", []) or []
                if isinstance(item, dict)
            ]
        except (TypeError, ValueError) as exc:
            record_degradation(
                "continuity_ledger.restore",
                exc,
                severity="warning",
                action="started from an empty ledger after a malformed persisted payload",
                enforce_failure_policy=False,
            )
            entries = []
        return cls(
            entries=entries,
            turn=int(payload.get("turn", 0) or 0),
            subject_trail=[str(s) for s in (payload.get("subject_trail", []) or [])],
            evicted_count=int(payload.get("evicted_count", 0) or 0),
        )

    # ── introspection ─────────────────────────────────────────────────────
    def stats(self) -> dict[str, Any]:
        by_kind: dict[str, int] = {}
        for entry in self.entries:
            by_kind[entry.kind] = by_kind.get(entry.kind, 0) + 1
        return {
            "entries": len(self.entries),
            "turn": self.turn,
            "evicted": self.evicted_count,
            "by_kind": by_kind,
            "rendered_chars": len(self.render()),
            "pinned": sum(1 for e in self.entries if e.pinned),
            "budget_chars": ledger_budget_chars(),
        }
