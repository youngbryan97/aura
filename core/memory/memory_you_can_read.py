"""A readable, versioned projection of semantic memory, tied to what happened.

Letta keeps agent memory as a git-backed filesystem, and the review said what
that buys: memory evolution is unusually inspectable and reversible. Its
closure asked for a canonical subset of Aura's long-term semantic and identity
memory projected into a versioned MemoryFS with commit ids linked to
event-spine receipts, keeping the databases as indexes rather than as the sole
human-readable truth.

The linking is the part that is easy to get wrong. A commit id that is a hash
of the content tells you what changed and not why; one that is a timestamp
tells you neither. Each commit here carries the event-spine sequence number
that was current when it was written, so "what was happening when she came to
believe this" is a lookup rather than a reconstruction.

This is a projection and says so. The stores stay authoritative — a file here
being deleted loses nothing, and that is the property that makes it safe to
write often.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.runtime.lockdep import checked_lock

logger = logging.getLogger("Aura.MemoryYouCanRead")

__all__ = [
    "ACommit",
    "MemoryYouCanRead",
    "what_is_projected",
]

#: What gets projected. A canonical subset, not everything: a projection of
#: the whole of memory is a second copy of memory, which is the thing this is
#: deliberately not.
def what_is_projected() -> tuple[str, ...]:
    return ("identity", "commitments", "people", "what_she_learned", "meanings")


@dataclass(frozen=True)
class ACommit:
    """One written version of one file, and what was happening at the time."""

    name: str
    digest: str
    at: float
    #: The event-spine sequence number current when this was written. Zero
    #: when the spine had nothing to say, which is honest rather than absent.
    after_event: int = 0
    why: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "digest": self.digest,
            "at": self.at,
            "after_event": self.after_event,
            "why": self.why,
        }


class MemoryYouCanRead:
    """Files a person can read, versioned, with a history of what changed."""

    def __init__(self, root: Any = None) -> None:
        self._root = Path(root) if root else None
        self._lock = checked_lock("memory_you_can_read")
        self._history: list[ACommit] = []

    @property
    def root(self) -> Path:
        if self._root is not None:
            return self._root
        from core.runtime.state_ownership import state_root

        return state_root() / "memory-you-can-read"

    def write(self, name: str, body: Any, *, why: str = "") -> ACommit | None:
        """Project one thing. Returns the commit, or None when nothing changed.

        Returning None for an unchanged file is not an optimisation. A history
        in which every save is a commit cannot answer "when did this last
        actually change", which is the question the history is for.
        """
        if name not in what_is_projected():
            raise KeyError(
                f"{name!r} is not projected; the set is "
                f"{', '.join(what_is_projected())}"
            )
        text = _readable(body)
        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()
        with self._lock:
            for one in reversed(self._history):
                if one.name == name:
                    if one.digest == digest:
                        return None
                    break
            commit = ACommit(
                name=str(name),
                digest=digest,
                at=time.time(),
                after_event=_where_the_spine_is(),
                why=str(why),
            )
            self._history.append(commit)
        self._put(name, text, commit)
        return commit

    def read(self, name: str) -> str:
        path = self.root / f"{name}.md"
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            return ""

    def history(self, name: str = "") -> list[dict[str, Any]]:
        """Every commit, newest last. Filtered by name when one is given."""
        with self._lock:
            return [
                one.to_dict()
                for one in self._history
                if not name or one.name == name
            ]

    def what_changed_after(self, sequence: int) -> list[dict[str, Any]]:
        """Every commit written after that point in the event log.

        The lookup the sequence number exists for: what she came to believe
        after a particular thing happened.
        """
        with self._lock:
            return [
                one.to_dict()
                for one in self._history
                if one.after_event > int(sequence)
            ]

    def report(self) -> dict[str, Any]:
        with self._lock:
            return {
                "root": str(self.root),
                "projected": list(what_is_projected()),
                "commits": len(self._history),
                "files": sorted({one.name for one in self._history}),
                "latest": self._history[-1].to_dict() if self._history else None,
            }

    def _put(self, name: str, text: str, commit: ACommit) -> None:
        from core.governance_context import local_internal_governed_scope
        from core.runtime.file_write_gateway import get_file_write_gateway

        try:
            gateway = get_file_write_gateway()
            with local_internal_governed_scope(
                "memory_you_can_read.write", domain="state_mutation"
            ):
                gateway.ensure_directory(self.root, source="memory_you_can_read")
                gateway.write_text(
                    self.root / f"{name}.md",
                    f"<!-- {commit.digest} after event {commit.after_event} -->\n"
                    f"{text}",
                    source="memory_you_can_read",
                )
        except Exception as exc:  # noqa: BLE001 — a projection must not cost a turn
            logger.warning("%s was not projected: %s", name, exc)


def _readable(body: Any) -> str:
    """Markdown a person reads, not JSON they decode.

    The point of the projection is that it can be read. A dict rendered as
    JSON is a second database in a text file.
    """
    if isinstance(body, str):
        return body.rstrip() + "\n"
    if isinstance(body, dict):
        lines = []
        for key in sorted(body):
            value = body[key]
            if isinstance(value, (list, tuple)):
                lines.append(f"## {key}")
                lines.extend(f"- {one}" for one in value)
            elif isinstance(value, dict):
                lines.append(f"## {key}")
                lines.extend(f"- **{inner}**: {value[inner]}" for inner in sorted(value))
            else:
                lines.append(f"## {key}\n{value}")
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"
    if isinstance(body, (list, tuple)):
        return "\n".join(f"- {one}" for one in body) + "\n"
    return json.dumps(body, indent=2, default=str) + "\n"


def _where_the_spine_is() -> int:
    """The event-spine head, or zero when it has nothing to say."""
    try:
        from core.runtime.event_spine import get_spine

        return int(get_spine().log.head)
    except Exception as exc:  # noqa: BLE001 — a missing spine is not a failure
        logger.debug("the spine had nothing to say: %s", exc)
        return 0
