"""Keep the forge's target still while the forge takes aim at it.

A forged skill is admitted when it passes the probes its own draft declared.
The probes are written before the code runs, which makes a failed expectation
a failing test rather than a second opinion, and that much was already true.

What was not true is that the target stayed put. ``_forge_verified_draft``
redrafts on failure and hands the drafter the interpreter's own traceback, and
the redraft returns *both* new code and new probes. So a drafter that cannot
make the code satisfy the expectation is free to return an expectation the
code already satisfies, and the second attempt passes a test the first one
would have failed. Every count downstream still reads "verified": the probes
were precommitted, they ran, they matched. They matched because they moved.

The invariant a candidate has to clear is that it does better under an
evaluation it was not able to shape. So the first draft that yields a usable
probe set fixes the target, and every later attempt is verified against that
set. A later attempt may *add* probes, because more tests can only raise the
bar. It may not change or drop one.

Nothing here decides whether a skill is good. It decides what the skill is
being asked, and holds the question still.
"""
from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import cycle at runtime, types only
    from core.skill_management.skill_verification import Probe

logger = logging.getLogger("Aura.WhatTheProbesWereHeldTo")

__all__ = [
    "TheTargetForOneForge",
    "WhatAnAttemptDidToTheTarget",
    "a_fingerprint_of",
    "how_the_target_has_held",
    "reset_how_the_target_has_held",
]

#: What every attempt after the first tried to do to the target it was being
#: judged against. Read by the inspector; a forge that never redrafts leaves
#: this empty, which is the honest reading of "nothing was measured".
HOW_THE_TARGET_MOVED: Counter[str] = Counter()

_HELD = "held"
_ADDED = "added to"
_ALTERED = "tried to alter"
_DROPPED = "tried to drop"


def _one(probe: Probe) -> str:
    """A content address for a single probe, stable across processes."""
    payload = json.dumps(
        {
            "params": probe.params,
            "expect": probe.expect if probe.has_expectation else "<none>",
            "expect_keys": sorted(probe.expect_keys),
            "has_expectation": bool(probe.has_expectation),
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=8).hexdigest()


def a_fingerprint_of(probes: tuple[Probe, ...]) -> str:
    """A content address for a whole probe set, independent of their order."""
    if not probes:
        return ""
    joined = ",".join(sorted(_one(p) for p in probes))
    return hashlib.blake2b(joined.encode("utf-8"), digest_size=16).hexdigest()


@dataclass(frozen=True, slots=True)
class WhatAnAttemptDidToTheTarget:
    """One redraft, and what it did to the question it was being asked."""

    attempt: int
    verdict: str
    added: int
    altered: int
    dropped: int

    @property
    def moved_it(self) -> bool:
        return bool(self.altered or self.dropped)

    def __str__(self) -> str:
        if self.verdict == _HELD:
            return f"attempt {self.attempt} kept the same probes"
        parts = []
        if self.added:
            parts.append(f"added {self.added}")
        if self.altered:
            parts.append(f"altered {self.altered}")
        if self.dropped:
            parts.append(f"dropped {self.dropped}")
        return f"attempt {self.attempt} {', '.join(parts)}"


@dataclass
class TheTargetForOneForge:
    """The probe set one forge is judged against, fixed at the first draft."""

    skill: str = ""
    _held: tuple[Probe, ...] = ()
    _seen: dict[str, Probe] = field(default_factory=dict)
    history: list[WhatAnAttemptDidToTheTarget] = field(default_factory=list)

    @property
    def fixed(self) -> bool:
        """Whether a target exists yet. It does not before the first draft."""
        return bool(self._held)

    @property
    def fingerprint(self) -> str:
        return a_fingerprint_of(self._held)

    @property
    def ever_moved(self) -> bool:
        return any(one.moved_it for one in self.history)

    def admit(self, attempt: int, offered: tuple[Probe, ...]) -> tuple[Probe, ...]:
        """What this attempt is verified against, whatever it offered.

        The first usable set becomes the target. After that, probes the target
        does not already carry are added and everything else is the target's,
        so a redraft cannot answer a failing expectation by rewriting it.
        """
        offered = tuple(offered or ())
        if not self.fixed:
            if offered:
                self._held = offered
                self._seen = {_one(p): p for p in offered}
            return self._held

        fresh = [p for p in offered if _one(p) not in self._seen]
        # A probe the attempt did not re-offer, and one it re-offered with a
        # different expectation, are the same act seen from two sides: the
        # first set had a question this set does not ask. Count both, because
        # which one it is says whether the drafter forgot the probe or
        # rewrote it.
        kept = {_one(p) for p in offered if _one(p) in self._seen}
        missing = len(self._seen) - len(kept)
        altered = min(missing, len(fresh))
        dropped = missing - altered
        added = len(fresh) - altered

        if fresh:
            self._held = (*self._held, *fresh)
            for probe in fresh:
                self._seen[_one(probe)] = probe

        verdict = _HELD
        if dropped:
            verdict = _DROPPED
        elif altered:
            verdict = _ALTERED
        elif added:
            verdict = _ADDED
        note = WhatAnAttemptDidToTheTarget(
            attempt=attempt,
            verdict=verdict,
            added=added,
            altered=altered,
            dropped=dropped,
        )
        self.history.append(note)
        HOW_THE_TARGET_MOVED[verdict] += 1
        if note.moved_it:
            logger.info(
                "🎯 Forge of %r: %s; verifying against the first set anyway.",
                self.skill or "a skill",
                note,
            )
        return self._held

    def as_dict(self) -> dict[str, Any]:
        return {
            "skill": self.skill,
            "probes": len(self._held),
            "fingerprint": self.fingerprint,
            "ever_moved": self.ever_moved,
            "attempts": [str(one) for one in self.history],
        }


def how_the_target_has_held() -> dict[str, Any]:
    """One block for the inspector: did any redraft move its own target?"""
    total = sum(HOW_THE_TARGET_MOVED.values())
    moved = HOW_THE_TARGET_MOVED[_ALTERED] + HOW_THE_TARGET_MOVED[_DROPPED]
    return {
        "redrafts": total,
        "moved_the_target": moved,
        "by_verdict": dict(sorted(HOW_THE_TARGET_MOVED.items())),
        "reading": (
            "no forge has redrafted in this process"
            if not total
            else f"{moved} of {total} redraft(s) offered a target the first draft did not set"
        ),
    }


def reset_how_the_target_has_held() -> None:
    HOW_THE_TARGET_MOVED.clear()
