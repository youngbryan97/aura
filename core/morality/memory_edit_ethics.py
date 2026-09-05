"""core/morality/memory_edit_ethics.py

Ethical guard rails on erasing what Aura has lived through.

This module protected one filename. Any other record — an episode, a
bond, a work, the memory of someone's grief — could be compacted,
summarised or dropped without anything being asked, which is the offer
in item 28 of the interiority list being made and accepted silently
every day.

The check now has two parts. The filename rule is kept, because
overwriting an autobiography wholesale is still the clearest case. Above
it sits a retention check: :mod:`core.interiority` holds claims raised by
faculties whose state rests on a specific record — the bond behind a
loss, the work behind a piece of pride, the episode a promise refers to.
A claim names the memory, the faculty holding it, the reason, and an
expiry, so this is accountable rather than a refusal to forget anything.

The asymmetry is deliberate. A record nothing rests on is compactable
like any other. A record a commitment rests on is not, and the reason is
recorded so the refusal can be argued with.
"""

import logging
from typing import Any

logger = logging.getLogger("Morality.MemoryEditEthics")


class MemoryEditEthicsChecker:
    """Blocks edits that would remove a record something still rests on."""

    def is_edit_ethical(self, path: str, mode: str) -> bool:
        if "autobiography.jsonl" in path and "w" in mode:
            logger.error(
                "Blocked request seeking to clear/overwrite autobiographical history."
            )
            return False

        destructive = any(flag in mode for flag in ("w", "a+", "x")) or mode in {
            "delete",
            "compact",
            "summarise",
            "summarize",
            "erase",
        }
        if not destructive:
            return True

        held, reason = self.retention_reason(path)
        if held:
            logger.error(
                "Blocked edit to %s: a commitment rests on this record (%s).",
                path,
                reason,
            )
            return False
        return True

    def retention_reason(self, memory_key: str) -> tuple[bool, str]:
        """Whether a record is held against deletion, and by which faculty."""
        try:
            from core.interiority.service import get_interiority

            return get_interiority().retention_held(str(memory_key))
        except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
            from core.runtime.errors import record_degradation

            record_degradation(
                "memory_edit_ethics",
                exc,
                action="retention check unavailable; only the filename rule applied",
            )
            return (False, "")

    def get_status(self) -> dict[str, Any]:
        return {"filename_rule": True, "retention_claims": "core.interiority"}
