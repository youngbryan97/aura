"""core/science/replication_pack.py — a result somebody else can run.

Aura's evidence is produced by Aura's repository on Aura's machine. That is not
a criticism of the evidence; it is a statement about what it can support. A
causal effect measured only where it was discovered has one independent
observation behind it, and this repository's own evidence module is built on
the principle that one observation is one observation.

A replication pack is the smallest bundle another machine needs: the tasks, the
seeds, the arms, the expected result, and the tolerance the original author is
willing to be wrong by. It is hashed, and the hash is what makes a replication
a replication rather than a rerun of something that moved.

Declaring the tolerance first
-----------------------------
``tolerance`` is part of the pack, not part of the verdict. An author who sees
the replication and then decides how close is close enough has not been
replicated, and the ordering here makes that impossible: the pack is sealed
with its tolerance before it leaves.

What the environment has to say
-------------------------------
:class:`Environment` records the commit, the model, the hardware and the
seeds. A replication that differs on any of them is still useful and is a
different claim, so :meth:`Replication.divergence` names what differed rather
than passing or failing on it. "Reproduced on different hardware" is a
stronger result than "reproduced", and "reproduced on a different model" is a
weaker one. A boolean cannot hold that distinction.
"""

from __future__ import annotations

from core.runtime.lockdep import checked_lock
import hashlib
import json
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Environment", "ReplicationPack", "Replication", "ReplicationRegistry"]


@dataclass(frozen=True, slots=True)
class Environment:
    """Where a result was produced. Part of the claim, not context."""

    commit: str
    model: str
    hardware: str
    python: str = ""

    def to_dict(self) -> dict[str, str]:
        return {"commit": self.commit, "model": self.model,
                "hardware": self.hardware, "python": self.python}


@dataclass(frozen=True, slots=True)
class ReplicationPack:
    """Everything another machine needs, sealed with the tolerance."""

    claim: str
    tasks: tuple[str, ...]
    seeds: tuple[int, ...]
    arms: tuple[str, ...]
    expected: Mapping[str, float]
    tolerance: float
    origin: Environment
    #: How to run it. A path in the repository, not a description.
    entrypoint: str = ""

    def __post_init__(self) -> None:
        if self.tolerance <= 0:
            raise ValueError(
                "a pack with no tolerance cannot be replicated, only matched exactly; "
                "declare what you are willing to be wrong by"
            )
        if not self.seeds:
            raise ValueError("a pack with no seeds cannot be rerun the same way twice")
        if set(self.expected) != set(self.arms):
            raise ValueError(
                f"expected results {sorted(self.expected)} do not cover the arms "
                f"{sorted(self.arms)}"
            )

    @property
    def seal(self) -> str:
        """The hash that makes a replication a replication rather than a rerun."""
        return hashlib.blake2s(
            json.dumps(
                {
                    "claim": self.claim,
                    "tasks": sorted(self.tasks),
                    "seeds": sorted(self.seeds),
                    "arms": sorted(self.arms),
                    "expected": {k: round(v, 6) for k, v in sorted(self.expected.items())},
                    "tolerance": round(self.tolerance, 6),
                    "entrypoint": self.entrypoint,
                },
                sort_keys=True,
            ).encode(),
            digest_size=16,
        ).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim": self.claim, "seal": self.seal, "tasks": list(self.tasks),
            "seeds": list(self.seeds), "arms": list(self.arms),
            "expected": dict(self.expected), "tolerance": self.tolerance,
            "origin": self.origin.to_dict(), "entrypoint": self.entrypoint,
        }


@dataclass(frozen=True, slots=True)
class Replication:
    """One outside attempt, and how far it landed from the pack."""

    seal: str
    observed: Mapping[str, float]
    environment: Environment
    pack: ReplicationPack
    replicator: str = ""

    @property
    def seal_matches(self) -> bool:
        return self.seal == self.pack.seal

    @property
    def deviations(self) -> dict[str, float]:
        return {
            arm: abs(self.observed.get(arm, float("inf")) - self.pack.expected[arm])
            for arm in self.pack.arms
        }

    @property
    def within_tolerance(self) -> bool:
        return self.seal_matches and all(
            d <= self.pack.tolerance for d in self.deviations.values()
        )

    def divergence(self) -> dict[str, Any]:
        """What differed about the environment. Not a pass or a fail.

        Reproduced on different hardware is a stronger result than reproduced;
        reproduced on a different model is a weaker one. A boolean loses that.
        """
        origin, here = self.pack.origin, self.environment
        differed = [
            field for field in ("commit", "model", "hardware", "python")
            if getattr(origin, field) and getattr(origin, field) != getattr(here, field)
        ]
        return {
            "differed_on": differed,
            "independent": bool(differed),
            "reading": (
                "same machine and same build; this is a rerun, not a replication"
                if not differed
                else "reproduced across " + ", ".join(differed)
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim": self.pack.claim,
            "replicator": self.replicator,
            "seal_matches": self.seal_matches,
            "within_tolerance": self.within_tolerance,
            "deviations": self.deviations,
            "environment": self.environment.to_dict(),
            **self.divergence(),
        }


class ReplicationRegistry:
    """Packs, the attempts against them, and what each claim has earned."""

    def __init__(self) -> None:
        self._lock = checked_lock("core.science.replication_pack.ReplicationRegistry", reentrant=True)
        self._packs: dict[str, ReplicationPack] = {}
        self._attempts: list[Replication] = []

    def publish(self, pack: ReplicationPack) -> ReplicationPack:
        with self._lock:
            self._packs[pack.seal] = pack
            return pack

    def submit(
        self,
        seal: str,
        observed: Mapping[str, float],
        environment: Environment,
        *,
        replicator: str = "",
    ) -> Replication:
        with self._lock:
            pack = self._packs.get(seal)
        if pack is None:
            raise KeyError(f"no pack with seal {seal!r}; a pack that moved is a rerun")
        replication = Replication(seal, dict(observed), environment, pack, replicator)
        with self._lock:
            self._attempts.append(replication)
        return replication

    def status(self, claim: str) -> dict[str, Any]:
        """What this claim has earned: reruns, replications, and by whom."""
        with self._lock:
            attempts = [a for a in self._attempts if a.pack.claim == claim]
        independent = [
            a for a in attempts if a.within_tolerance and a.divergence()["independent"]
        ]
        return {
            "claim": claim,
            "attempts": len(attempts),
            "reruns": sum(1 for a in attempts if not a.divergence()["independent"]),
            "independent_replications": len(independent),
            "replicators": sorted({a.replicator for a in independent if a.replicator}),
            "failed": [a.to_dict() for a in attempts if not a.within_tolerance],
            "externally_replicated": len(independent) >= 1,
        }

    def report(self) -> dict[str, Any]:
        with self._lock:
            claims = sorted({p.claim for p in self._packs.values()})
        return {
            "packs": len(self._packs),
            "claims": claims,
            "by_claim": {claim: self.status(claim) for claim in claims},
        }
