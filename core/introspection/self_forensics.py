"""core/introspection/self_forensics.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Grounded self-forensics: when Aura is asked about her own shutdowns,
crashes, or restarts, hand her the black boxes instead of letting the
model invent a story.

Observed live (July 4): asked "what was the cause?" about her overnight
death, she confabulated electromagnetic interference — three drafts in a
row — while the true answer (generation-gate wedge during a cold model
load; launcher stop) sat in her own incident records, shutdown grace
flag, and memory-sentinel log. The honesty gate rightly rejected every
draft, but nothing supplied the evidence a truthful draft needed.

Durable sources (they survive the very death being asked about):
- ~/.aura/run/grace_exit.flag — the last graceful shutdown's reason+time
- data/error_logs/memory/sentinel.log — guard arm/exit lines with reasons
- data/error_logs/crash|stalls — newest artifact names and ages
- the cross-boot fault evidence store
In-process sources (this session's live state):
- IncidentManager active incidents / summary
- FaultRegistry recent faults

The output is a compact evidence block with an explicit honesty
instruction: cite only this evidence; say unknown where it is silent.
"""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

from core.runtime.errors import record_degradation
from core.runtime.state_ownership import state_root

logger = logging.getLogger("Aura.Introspection.SelfForensics")

_SELF_FORENSICS_RE = re.compile(
    r"(?:\b(?:you|your|u)\b.{0,60}\b(?:crash(?:ed|es)?|shut\s*down|shutdown|die(?:d)?|"
    r"restart(?:ed)?|reboot(?:ed)?|went\s+down|stopped\s+working|froze|wedged|"
    r"k(?:i|1)lled|terminated)\b"
    r"|\b(?:crash(?:ed)?|shutdown|restart(?:ed)?|outage|incident)\b.{0,60}\b(?:last\s+night|"
    r"yesterday|earlier|this\s+morning|overnight|while\s+i\s+was)\b"
    r"|\bwhat\s+(?:was|is)\s+the\s+(?:root\s+)?cause\b"
    r"|\bwhy\s+did\s+you\s+(?:crash|die|stop|shut|restart|go\s+down|disappear)\b"
    r"|\bwhat\s+happened\s+(?:to\s+you|last\s+night|overnight|while)\b)",
    re.IGNORECASE,
)


def is_self_forensics_question(text: str | None) -> bool:
    """True when the user is asking about Aura's own failure history."""
    candidate = str(text or "").strip()
    if not candidate or len(candidate) > 500:
        return False
    return bool(_SELF_FORENSICS_RE.search(candidate))


def _read_grace_flag() -> str:
    try:
        flag = state_root() / "run" / "grace_exit.flag"
        data = json.loads(flag.read_text(encoding="utf-8"))
        age_h = max(0.0, time.time() - float(data.get("created_at_unix", 0.0))) / 3600.0
        reason = str(data.get("reason") or "unspecified")
        return f"last graceful shutdown: reason='{reason}' ({age_h:.1f}h ago, pid {data.get('pid')})"
    except (OSError, json.JSONDecodeError, ValueError, TypeError):
        return ""


def _read_sentinel_tail(lines: int = 6) -> str:
    try:
        from core.utils.paths import forensics_search_dirs

        for memory_dir in forensics_search_dirs("memory"):
            log = memory_dir / "sentinel.log"
            if not log.is_file():
                continue
            tail = (
                log.read_text(encoding="utf-8", errors="replace")
                .strip()
                .splitlines()[-lines:]
            )
            if tail:
                return "; ".join(
                    line.split("] ", 1)[-1][:110] for line in tail if line.strip()
                )
        return ""
    except (OSError, ImportError):
        return ""


def _newest_artifacts(kind: str, count: int = 2) -> str:
    """Newest artifacts of one forensic class across every root that holds them.

    Reading a single hardcoded relative directory is what made this block report
    an empty crash history while the dumps sat in the other tree; asking the
    shared resolver means the reader cannot drift away from the writer again.
    """
    try:
        from core.utils.paths import forensics_search_dirs

        entries: list[Path] = []
        for base in forensics_search_dirs(kind):
            entries.extend(p for p in base.iterdir() if p.is_file())
        entries.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        now = time.time()
        return ", ".join(
            f"{p.name} ({(now - p.stat().st_mtime) / 3600.0:.1f}h old)"
            for p in entries[:count]
        )
    except (OSError, ImportError):
        return ""


def _live_incidents() -> str:
    try:
        from core.resilience.incident_manager import get_incident_manager

        manager = get_incident_manager()
        summary = manager.get_summary()
        active = manager.get_active()
        if not int(summary.get("total", 0) or 0):
            # Zero incidents THIS session is not evidence about a past
            # death — omit rather than pad the block.
            return ""
        parts = [f"incidents this session: {summary.get('total', 0)} total, {len(active)} active"]
        for item in active[:3]:
            parts.append(
                f"[{item.get('severity')}] {item.get('category')}: {str(item.get('message'))[:90]}"
            )
        return "; ".join(parts)
    except (ImportError, AttributeError, RuntimeError, TypeError):
        return ""


def _recent_faults() -> str:
    try:
        from core.resilience.fault_taxonomy import get_fault_registry

        recent = get_fault_registry().recent_faults(window_s=6 * 3600)
        if not recent:
            return ""
        tail = recent[-4:]
        return "recent faults: " + "; ".join(
            f"{r.fault_id}@{r.subsystem}: {str(r.details)[:70]}" for r in tail
        )
    except (ImportError, AttributeError, RuntimeError, TypeError):
        return ""


def _what_the_outcome_record_says() -> str:
    """Her own outcomes, read back, for a question about herself.

    A thing that is registered is a thing somebody wrote down. A thing that is
    available is one whose preconditions hold. A thing that has WORKED is one
    there is a receipt for, and only the third is evidence — which is the
    distinction a question about her own capabilities is asking for.

    `core/self/what_has_ever_worked.py` reads the thirty-two thousand outcomes
    she already keeps and was imported by nothing outside its own test. Asked
    on 2026-08-27 how many of her skills had never once executed successfully,
    she counted the .py files in a directory: an honest method for a different
    question, and this is the answer to the one that was asked.
    """

    try:
        from core.self.what_has_ever_worked import says

        return says()
    except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation("self_forensics", exc, severity="debug")
        return ""


def build_self_forensics_context(max_chars: int = 3200) -> str:
    """Compose the grounded evidence block for a self-forensics question."""
    sections: list[tuple[str, str]] = [
        ("SHUTDOWN", _read_grace_flag()),
        ("MEMORY SENTINEL", _read_sentinel_tail()),
        ("CRASH ARTIFACTS", _newest_artifacts("crash")),
        ("STALL ARTIFACTS", _newest_artifacts("stalls")),
        ("LIVE INCIDENTS", _live_incidents()),
        ("FAULTS", _recent_faults()),
        ("WHAT HAS EVER WORKED", _what_the_outcome_record_says()),
    ]
    body = "\n".join(f"- {name}: {value}" for name, value in sections if value)
    if not body:
        return (
            "GROUNDED SELF-FORENSICS: no failure evidence is currently "
            "readable. Say plainly that the records are unavailable — do "
            "not invent a cause."
        )
    return (
        "GROUNDED SELF-FORENSICS (your own black boxes — answer the user's "
        "question about your shutdown/crash from THIS evidence only; name "
        "what it shows, and say 'unknown' for anything it does not show. "
        "Never invent causes like interference, spikes, or external "
        "systems):\n" + body
    )[:max_chars]
