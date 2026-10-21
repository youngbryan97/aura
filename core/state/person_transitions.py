"""What changed in the person between two committed states, and what wrote it.

The lineage judges a step by whether every identity-bearing path it changed was
written by a measured transformation that kept to its contract. That evidence
has to be taken when the state is committed, in the process whose provenance
ring saw the phases run; read back later from the log, the writers are gone.
So each commit carries, on the state itself, the state the same repository
committed before it, a digest of each person path, and for each path the last
write recorded for it. The last write is carried forward from commit to commit,
so a row the log later prunes does not take the evidence for the rows after it
with it.
"""
from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping
from typing import Any

__all__ = ["AURA_PERSON_PATHS", "PersonWatch", "person_paths", "take_person_evidence"]

#: Where each person field lives in Aura's own state. A field with no entry is
#: one her state does not carry, since skills and learned policy live in organs
#: outside it, and it is recorded as absent rather than defaulted. What changes
#: every turn, the conversation and the recalled set among it, is not here: a
#: person is not a different person because she was spoken to.
AURA_PERSON_PATHS: dict[str, tuple[str, ...]] = {
    "autobiographical": ("identity.current_narrative", "identity.narrative_version"),
    "self_model": ("identity.name", "identity.concept_graph"),
    "values": ("identity.core_values",),
    "preferences": ("identity.self_preferences",),
    "dispositions": ("identity.personality_growth",),
    "relationships": ("world.relationship_graph", "identity.bonding_level"),
    "developmental": ("identity.evolution_score",),
}


def person_paths() -> tuple[str, ...]:
    return tuple(path for paths in AURA_PERSON_PATHS.values() for path in paths)


def _at(state: Any, dotted: str) -> Any:
    node = state
    for part in dotted.split("."):
        if node is None:
            return None
        node = node.get(part) if isinstance(node, Mapping) else getattr(node, part, None)
    return node


def _digest(value: Any) -> str:
    # The same bounded rendering the transformation receipts use, so a value
    # too large to digest whole is cut at the same place on both sides.
    try:
        rendered = repr(value)
    except Exception:  # noqa: BLE001 - a __repr__ that raises must not stop a commit
        rendered = f"<unrepresentable {type(value).__name__}>"
    return hashlib.sha1(rendered[:4096].encode("utf-8", errors="ignore")).hexdigest()[:12]


def _writer(path: str, since: float) -> dict[str, Any]:
    """The newest measured write to `path` since `since`, or an empty record."""
    try:
        from core.runtime.cognitive_provenance import recent_graphs
    except ImportError:
        return {}
    for graph in reversed(recent_graphs(64)):
        if graph.ended_at and graph.ended_at < since:
            break
        receipt = graph.last_writer(path)
        if receipt is None:
            continue
        objective = str(getattr(graph, "objective", "") or "")
        return {
            "transform": receipt.transform,
            "honoured": bool(receipt.honoured_contract),
            "contract": receipt.contract_hash,
            "branch": receipt.branch,
            "tick": graph.tick_id,
            "objective": hashlib.sha1(objective.encode("utf-8")).hexdigest()[:12] if objective else "",
            "at": receipt.at,
        }
    return {}


class PersonWatch:
    """Held by a state repository: the person as it stood at the last commit."""

    def __init__(self) -> None:
        self._state_id = ""
        self._version = 0
        self._at = 0.0
        self._digests: dict[str, str] = {}

    def evidence(self, state: Any) -> dict[str, Any]:
        digests = {path: _digest(_at(state, path)) for path in person_paths()}
        before = getattr(state, "transition_evidence", None)
        carried = dict(before.get("last_change", {})) if isinstance(before, Mapping) else {}
        version = int(getattr(state, "version", 0) or 0)
        if self._digests:
            for path, digest in digests.items():
                if digest != self._digests.get(path):
                    carried[path] = {"version": version, **_writer(path, self._at)}
        out = {
            "since_state": self._state_id,
            "since_version": self._version,
            "version": version,
            "paths": digests,
            "last_change": carried,
        }
        self._state_id = str(getattr(state, "state_id", "") or "")
        self._version = version
        self._at = time.time()
        self._digests = digests
        return out


def take_person_evidence(repository: Any, state: Any) -> None:
    """Carry onto the state what changed in the person since the last commit, and what wrote it.

    Taken by the repository's commit, in the process whose provenance ring saw
    the phases run; the owner that writes the row may be another process.
    Best-effort: a failure leaves the state without evidence, which the lineage
    reads as unexplained change, and is recorded.
    """
    try:
        watch = getattr(repository, "_person_watch", None)
        if watch is None:
            watch = PersonWatch()
            repository._person_watch = watch
        state.transition_evidence = watch.evidence(state)
    except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
        from core.runtime.errors import record_degradation

        record_degradation(
            "state_repository",
            exc,
            action="committed state without evidence of what changed the person",
            severity="warning",
        )


def _watch_the_person() -> None:
    try:
        from core.runtime.cognitive_contract import watch_also
    except ImportError:
        return
    watch_also(person_paths())


_watch_the_person()
