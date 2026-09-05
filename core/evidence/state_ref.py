"""core/evidence/state_ref.py — one envelope for a piece of cognitive state.

Aura moves state between organs constantly and every pair invented its own
shape for it. ``AuraNow`` is a self-field snapshot, the workspace passes
candidates, the AtomSpace passes atoms, the world model passes latents, memory
passes records, the planner passes nodes. Each is right for its own organ, and
between any two of them sits an adapter that decides — usually by dropping
them — what happens to provenance, confidence and time.

A :class:`CognitiveStateRef` is the envelope those adapters were missing. It
does not replace any representation: ``payload`` stays whatever the producing
organ made. What it adds is the six things every consumer needed and had to
guess at:

* **identity** — a content hash, so the same state recognised twice is the
  same state, and a receipt can name it later;
* **time** — when it was true, not when it was passed on;
* **owner** — which organ may mutate it, so a consumer that writes to state it
  does not own is a detectable error rather than a race;
* **evidence** — an :class:`~core.evidence.packet.EvidencePacket`, so the
  confidence travelling with the state knows what it rests on;
* **parents** — the state this was computed from, which is what makes a causal
  chain reconstructible without every organ logging its own story;
* **version** — a monotonic counter per identity, so a consumer holding stale
  state can tell.

Why not a base class. Making every organ's state inherit from something would
be a rewrite of a hundred modules and would fail on the ones that pass tuples
and dicts. This wraps instead, which means adoption is per-handoff and
measurable: :func:`handoff_coverage` counts what fraction of registered
handoffs carry the envelope, and the number moving is the evidence that the
contract is real rather than declared.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field, replace
from typing import Any

from core.evidence.packet import EvidencePacket
from core.runtime.lockdep import checked_lock

__all__ = [
    "SCHEMA_VERSION",
    "CognitiveStateRef",
    "StateOwnershipError",
    "register_handoff",
    "record_handoff",
    "handoff_coverage",
    "reset_handoff_ledger_for_test",
]


#: Schema version for the envelope as a cross-organ contract.
SCHEMA_VERSION = "aura.cognition.state_ref.v1"


class StateOwnershipError(RuntimeError):
    """Raised when an organ mutates state another organ owns."""


def _identity(payload: Any, kind: str) -> str:
    try:
        material = repr(payload)
    except (TypeError, ValueError, AttributeError, RecursionError) as exc:
        # An object whose repr raises still needs an identity, and the failure
        # itself is part of what distinguishes it. Named rather than bare, so
        # a genuinely unexpected error still surfaces.
        material = f"<unreprable {type(payload).__name__}: {type(exc).__name__}>"
    return hashlib.blake2s(f"{kind}\x00{material}".encode(), digest_size=12).hexdigest()


@dataclass(frozen=True, slots=True)
class CognitiveStateRef:
    """A piece of cognitive state, with everything a consumer needs to trust it."""

    kind: str
    payload: Any
    owner: str
    at: float = field(default_factory=time.time)
    evidence: EvidencePacket | None = None
    parents: tuple[str, ...] = ()
    version: int = 0
    #: Free-form, never part of identity. For the human reading a trace.
    note: str = ""

    @property
    def identity(self) -> str:
        return _identity(self.payload, self.kind)

    @property
    def confidence(self) -> float:
        """Confidence carried by the evidence, or 0.0 when none was attached.

        Zero rather than a default like 0.5, because state arriving with no
        evidence is not half-believed — nothing was measured about it, and a
        consumer treating that as a coin flip is the failure this prevents.
        """
        return self.evidence.confidence if self.evidence else 0.0

    def derive(
        self,
        payload: Any,
        *,
        kind: str = "",
        owner: str = "",
        evidence: EvidencePacket | None = None,
        note: str = "",
    ) -> CognitiveStateRef:
        """A new ref computed from this one, carrying the causal link."""
        return CognitiveStateRef(
            kind=kind or self.kind,
            payload=payload,
            owner=owner or self.owner,
            at=time.time(),
            evidence=evidence if evidence is not None else self.evidence,
            parents=(*self.parents, self.identity)[-16:],
            version=self.version + 1,
            note=note,
        )

    def mutated_by(self, organ: str, payload: Any) -> CognitiveStateRef:
        """Replace the payload, refusing when ``organ`` does not own the state."""
        if organ != self.owner:
            raise StateOwnershipError(
                f"{organ!r} tried to mutate {self.kind!r} state owned by {self.owner!r}; "
                "derive a new ref instead, so the change has a causal parent"
            )
        return replace(self, payload=payload, at=time.time(), version=self.version + 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "identity": self.identity,
            "owner": self.owner,
            "at": self.at,
            "version": self.version,
            "parents": list(self.parents),
            "confidence": self.confidence,
            "evidence": self.evidence.to_dict() if self.evidence else None,
            "note": self.note,
        }


# ── adoption ledger ───────────────────────────────────────────────────────
#
# A contract nobody uses is a docstring. These count what actually flows.

_ledger_lock = checked_lock("core.evidence.state_ref.singleton")
_registered: dict[str, str] = {}
_wrapped: dict[str, int] = {}
_bare: dict[str, int] = {}


def register_handoff(name: str, description: str) -> None:
    """Declare a cross-organ handoff that ought to carry the envelope."""
    with _ledger_lock:
        _registered[name] = description


def record_handoff(name: str, state: Any) -> Any:
    """Record one handoff and whether it carried a ref. Returns ``state``."""
    with _ledger_lock:
        if isinstance(state, CognitiveStateRef):
            _wrapped[name] = _wrapped.get(name, 0) + 1
        else:
            _bare[name] = _bare.get(name, 0) + 1
    return state


def handoff_coverage() -> dict[str, Any]:
    """What fraction of observed handoffs carried the contract.

    ``by_handoff`` is the useful half: a handoff at 0.0 names the adapter that
    still drops provenance, which is a work item rather than a statistic.
    """
    with _ledger_lock:
        names = sorted(set(_registered) | set(_wrapped) | set(_bare))
        by_handoff = {}
        for name in names:
            wrapped, bare = _wrapped.get(name, 0), _bare.get(name, 0)
            total = wrapped + bare
            by_handoff[name] = {
                "wrapped": wrapped,
                "bare": bare,
                "coverage": (wrapped / total) if total else None,
                "description": _registered.get(name, ""),
            }
        total_wrapped = sum(_wrapped.values())
        total_bare = sum(_bare.values())
        total = total_wrapped + total_bare
        return {
            "registered_handoffs": len(_registered),
            "observed_handoffs": len(names),
            "wrapped": total_wrapped,
            "bare": total_bare,
            "coverage": (total_wrapped / total) if total else None,
            "by_handoff": by_handoff,
        }


def reset_handoff_ledger_for_test() -> None:
    with _ledger_lock:
        _registered.clear()
        _wrapped.clear()
        _bare.clear()
