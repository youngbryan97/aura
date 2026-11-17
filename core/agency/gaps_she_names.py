"""When she tells a person she cannot do something, where the capability fails, and what her loop mends (F16, part 4).

LIVE 16 November 2026, 00:33: asked "Can you highlight the back button?", she answered "No. I don't have a tool that
can highlight or interact with UI elements on your screen right now." She had one; its step ran and was declined. A
sentence of hers that says she cannot is read for what it says she lacks, by meaning, against the capabilities she
has (core/intent/capability_by_meaning.py), and the turn's receipts say what became of that capability:

- she has nothing for it: the gap is kept, named, for her loop and for a person;
- it ran and did not do what was asked: the gap is kept with what it came to (a fault it raised is her fault loop's,
  core/self_modification/faults_she_can_reproduce.py);
- it was never reached: the route dropped the request. Where the person's words already sat nearest it and not
  clearly enough to be chosen, the request is taught to it as one of its meanings, and kept only if the request is
  then read as meaning it; otherwise taken back.

Nothing here generates text: her words are read with the capability ledger's frame of a denial, her capabilities by
her own embedding model, and her receipts as written.
"""
from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.lockdep import checked_lock

logger = logging.getLogger("Aura.GapsSheNames")

#: The most gaps kept, the newest kept.
_KEPT = 200
_LOCK = checked_lock("gaps_she_names.record")

NOTHING_FOR_IT = "she has nothing for it"
IT_RAN = "it ran and did not do what was asked"
NEVER_REACHED = "the route never reached it"


@dataclass(frozen=True)
class NamedGap:
    """One thing she said she could not do, and where the capability failed."""

    said: str
    asked: str
    capability: str
    finding: str
    detail: str
    mended: str
    at: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def denials_in(reply: str) -> list[str]:
    """Her sentences that say she cannot do something or does not have it, outside quoted speech."""
    from core.self.capability_ledger import _DENIAL_FRAME, _SENTENCE_SPLIT, _without_quoted_speech

    return [sentence.strip() for sentence in _SENTENCE_SPLIT.split(str(reply or ""))
            if sentence.strip() and _DENIAL_FRAME.search(_without_quoted_speech(sentence))]


def _receipt_names(receipts: Sequence[Any] | None) -> dict[str, list[Mapping[str, Any]]]:
    named: dict[str, list[Mapping[str, Any]]] = {}
    for receipt in receipts or ():
        if isinstance(receipt, Mapping):
            for key in ("tool", "skill", "skill_name", "capability"):
                if receipt.get(key):
                    named.setdefault(str(receipt[key]), []).append(receipt)
    return named


def trace(said: str, asked: str, receipts: Sequence[Any] | None, skills: Mapping[str, Any], meanings: Any) -> NamedGap:
    """Where the capability ``said`` says she lacks fails: what she has for it, and what became of it this turn."""
    capability = meanings.which(said, skills) or ""
    if not capability:
        return NamedGap(said, asked, "", NOTHING_FOR_IT, "no capability of hers is clearly what it names", "",
                        time.time())
    ran = _receipt_names(receipts).get(capability, [])
    if ran:
        last = ran[-1]
        detail = str(last.get("error") or last.get("verification") or last.get("status") or
                     ("done" if last.get("ok") else "not done"))
        return NamedGap(said, asked, capability, IT_RAN, detail[:300], "", time.time())
    nearest = meanings.nearest(asked, skills)
    if not nearest or nearest[0][0] != capability:
        return NamedGap(said, asked, capability, NEVER_REACHED,
                        f"the request sat nearest {nearest[0][0] if nearest else 'nothing'}, not {capability}; "
                        "not taught", "", time.time())
    if meanings.which(asked, skills) == capability:
        return NamedGap(said, asked, capability, NEVER_REACHED, "the request already means it; a route before the "
                        "reading by meaning took the turn", "", time.time())
    meanings.learned(asked, capability)
    if meanings.which(asked, skills) == capability:
        return NamedGap(said, asked, capability, NEVER_REACHED, "the request sat nearest it and not clearly",
                        f"taught: the request is now read as meaning {capability}", time.time())
    meanings.forget(asked, capability)
    return NamedGap(said, asked, capability, NEVER_REACHED, "the request sat nearest it and not clearly",
                    "taught and taken back: the request still did not stand out for it", time.time())


def _store() -> Path:
    from core.runtime.state_ownership import state_root

    return state_root() / "gaps_she_named.json"


def kept_gaps() -> list[dict[str, Any]]:
    """The gaps she has named, oldest first."""
    try:
        held = json.loads(_store().read_text("utf-8"))
    except (OSError, ValueError):
        return []
    return [gap for gap in held if isinstance(gap, dict)] if isinstance(held, list) else []


def _keep(gaps: list[NamedGap]) -> None:
    from core.governance_context import local_internal_governed_scope
    from core.runtime.file_write_gateway import get_file_write_gateway

    with _LOCK:
        held = (kept_gaps() + [gap.to_dict() for gap in gaps])[-_KEPT:]
        with local_internal_governed_scope("gaps_she_names.keep", domain="state_mutation"):
            get_file_write_gateway().write_text(_store(), json.dumps(held, indent=1), source="gaps_she_names.keep",
                                                durable=False)


def hear_what_she_said_she_cannot(asked: str, reply: str, receipts: Sequence[Any] | None = None, *,
                                  skills: Mapping[str, Any] | None = None, meanings: Any = None) -> list[NamedGap]:
    """Each thing her reply says she cannot do, traced and kept; the gaps found. Runs after the reply is served."""
    said = denials_in(reply)
    if not said or not str(asked or "").strip():
        return []
    try:
        if skills is None:
            from core.container import ServiceContainer

            skills = getattr(ServiceContainer.get("capability_engine", default=None), "skills", None) or {}
        if meanings is None:
            from core.brain.requests_by_meaning import _the_meanings

            meanings = _the_meanings()
        if meanings is None or not skills:
            return []
        from core.governance_context import local_internal_governed_scope

        with local_internal_governed_scope("gaps_she_names.trace"):
            gaps = [trace(sentence, asked, receipts, skills, meanings) for sentence in said]
        _keep(gaps)
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation("gaps_she_names", exc, severity="info", action="her reply was not read for gaps this time")
        return []
    for gap in gaps:
        logger.info("🧩 she said she cannot: %r — %s%s", gap.said[:120], gap.finding,
                    f" ({gap.capability})" if gap.capability else "")
    return gaps


__all__ = ["IT_RAN", "NEVER_REACHED", "NOTHING_FOR_IT", "NamedGap", "denials_in", "hear_what_she_said_she_cannot",
           "kept_gaps", "trace"]
