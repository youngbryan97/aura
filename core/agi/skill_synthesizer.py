"""core/agi/skill_synthesizer.py
Capability gaps, and the forge they now reach.

This module watches for tasks Aura could not handle and counts them. A gap seen
often enough is a candidate for a new skill.

What it used to do with that candidate was the problem. It asked a model for an
"implementation", received one line of English describing what the skill should
do, and rendered it into a class body as::

    implementation=f"result = {impl[:100]!r}"

which is a string literal. The generated skill returned the *description of the
work* in place of the work. The module's own comment said so — "this
synthesizer produces non-runnable stubs" — and the gap detection upstream was
sound, so the effect was that every capability gap Aura correctly identified
produced an artifact that could never close it.

Two forges, neither of which worked
-----------------------------------
The codebase had a second one. :mod:`core.skill_management.hephaestus` generated
real executable Python, and was reached from a different trigger: a skill
missing at execution time. So Aura had gap detection wired to a stub generator,
and a real generator wired to a trigger that fires only once someone already
asked for a skill by name.

Hephaestus now verifies what it forges — the code has to run and satisfy probes
declared before it ran, or nothing is retained. So this module's job is the one
it was always supposed to have: notice the gap, and hand it to the forge.

The stub path is gone rather than deprecated. A synthesizer that can still emit
a non-runnable skill will emit one.

Why the name is derived here and not asked for
----------------------------------------------
The old code took the skill's identifier from the model and sanitised it. The
identifier decides which file is written and which capability is overwritten, so
it is not a thing to accept from a generator and repair afterwards. It is now
derived from the gap text, deterministically, which also means the same gap
maps to the same skill instead of accumulating near-duplicates under whatever
name the model picked that day.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from core.runtime.atomic_writer import atomic_write_text
from core.runtime.errors import record_degradation
from core.runtime.state_ownership import state_root

logger = logging.getLogger("Aura.SkillSynthesizer")

PERSIST_PATH = state_root() / "data" / "synthesized_skills.json"

# Skill template — all synthesized skills follow this pattern
import re as _re


#: Gaps forged for in one pass. Each forge costs model calls and a sandbox run,
#: and a background pass that tries every open gap at once turns a quiet period
#: into a stampede.
_MAX_FORGES_PER_PASS = 3

#: Ceiling on one forge, covering its drafts, its sandbox runs and its install.
#: The forge is a background activity behind a resource arbitrator; this is the
#: outer bound that keeps a wedged drafter from holding the pass open.
_FORGE_TIMEOUT_S = 180.0


def skill_name_for_gap(gap: str) -> str:
    """A stable snake_case identifier derived from the gap text.

    Derived rather than requested. The identifier decides which file the forge
    writes and which existing capability it supersedes, so it must be a function
    of the need and not a field in a model's reply.

    A short hash of the full gap is appended because the readable part is
    truncated: two long gaps that share an opening would otherwise collapse onto
    one skill, and the second would silently overwrite the first.
    """
    text = _re.sub(r"[^a-z0-9]+", "_", str(gap or "").strip().lower()).strip("_")
    words = [w for w in text.split("_") if w][:5]
    stem = "_".join(words) or "capability"
    if not stem[0].isalpha():
        stem = f"skill_{stem}"
    suffix = hashlib.blake2b(str(gap or "").encode("utf-8"), digest_size=3).hexdigest()
    return f"{stem[:48]}_{suffix}"


def _sanitize_text_field(raw: object, limit: int) -> str:
    """Strip control characters and bound a free-text model field."""
    text = "".join(ch for ch in str(raw or "") if ch == " " or ord(ch) >= 32)
    return text.strip()[:limit]


#: Times a gap must recur before it is worth forging for.
#:
#: Three, because the first occurrence is a task and the second is a
#: coincidence. A skill forged from one miss is a skill built for one request,
#: and the cost of forging is paid whether or not the need returns.
GAP_FORGE_THRESHOLD = 3


@dataclass
class SynthesizedSkill:
    """A skill the forge produced for a gap, and the evidence behind it."""

    name: str
    description: str
    gap: str                    # the capability gap this fills
    #: Content address of the verified region, or "" when nothing was retained.
    #: This is the join to the forge ledger, where the probe results live.
    digest: str = ""
    verified: bool = False
    detail: str = ""
    created_at: float = field(default_factory=time.time)
    use_count: int = 0


class SkillSynthesizer:
    """
    Detects capability gaps from failed queries, synthesizes new skills.

    Integration:
      - Call `log_gap(task, reason)` when a skill lookup fails
      - Call `synthesize_pending(orchestrator)` in background loop
      - Synthesized skills are auto-registered into the skill registry
    """

    def __init__(self):
        self._gaps: list[dict] = []          # observed capability gaps
        self._synthesized: list[SynthesizedSkill] = []
        self._gap_counts: dict[str, int] = {}  # gap → frequency
        self._load()
        logger.info("SkillSynthesizer online — autonomous capability expansion ready.")

    # ── Public API ────────────────────────────────────────────────────────

    def log_gap(self, task_description: str, failure_reason: str = ""):
        """Record a capability gap. High-frequency gaps trigger synthesis."""
        # Normalize to a gap key
        gap_key = task_description[:80].lower().strip()
        self._gap_counts[gap_key] = self._gap_counts.get(gap_key, 0) + 1
        self._gaps.append({
            "task": task_description,
            "reason": failure_reason,
            "count": self._gap_counts[gap_key],
            "timestamp": time.time(),
        })
        if self._gap_counts[gap_key] >= GAP_FORGE_THRESHOLD:
            logger.info("SkillSynthesizer: gap threshold reached for '%s'", gap_key[:60])
        if len(self._gaps) > 200:
            self._gaps = self._gaps[-200:]

    async def synthesize_pending(self, orchestrator=None) -> list[SynthesizedSkill]:
        """Hand the most frequent unresolved gaps to the forge.

        ``orchestrator`` is accepted and unused. It is kept because the existing
        callers pass it, and removing it would be a signature change in a live
        background loop for no gain.
        """
        del orchestrator

        hot_gaps = sorted(
            (
                (gap, count)
                for gap, count in self._gap_counts.items()
                if count >= GAP_FORGE_THRESHOLD
            ),
            key=lambda pair: pair[1],
            reverse=True,
        )[:_MAX_FORGES_PER_PASS]

        forged: list[SynthesizedSkill] = []
        for gap, count in hot_gaps:
            if any(s.gap == gap and s.verified for s in self._synthesized):
                continue
            skill = await self._forge_for_gap(gap, count)
            if skill is None:
                continue
            # Replace any earlier unverified attempt at the same gap rather than
            # letting failed attempts accumulate into a list of things that do
            # not exist.
            self._synthesized = [s for s in self._synthesized if s.gap != gap]
            self._synthesized.append(skill)
            if skill.verified:
                forged.append(skill)

        await self._save_async()
        return forged

    def get_synthesized_skills(self) -> list[dict]:
        return [
            {
                "name": s.name,
                "description": s.description,
                "gap": s.gap,
                "verified": s.verified,
                "digest": s.digest,
                "use_count": s.use_count,
            }
            for s in self._synthesized
        ]

    @staticmethod
    def _what_it_has_done_since(name: str) -> dict[str, Any]:
        """What a forged skill actually did after it was installed.

        Read from the skill library rather than from a counter here.
        ``SynthesizedSkill.use_count`` was declared, serialised and loaded,
        and incremented by nothing anywhere — so it read 0 for every forged
        skill forever, and the question "did the thing she built help?" had a
        field and no answer.

        The library already counts successes and failures per skill because
        that is what running one produces. A second counter would be a second
        thing to forget to write.
        """
        try:
            from core.container import ServiceContainer

            library = ServiceContainer.get("skill_library", default=None)
            held = getattr(library, "skills", {}).get(name) if library else None
        except (ImportError, AttributeError, RuntimeError, TypeError) as exc:
            return {"known": False, "why": f"{type(exc).__name__}: {exc}"}
        if held is None:
            # Not an error. A forged skill the library has never heard of is
            # one that was never installed, which is itself the answer.
            return {"known": False, "why": "the library does not hold it"}
        successes = int(getattr(held, "successes", 0) or 0)
        failures = int(getattr(held, "failures", 0) or 0)
        return {
            "known": True,
            "successes": successes,
            "failures": failures,
            "taken": successes + failures,
            "reliability": getattr(held, "reliability", None),
        }

    def what_the_forge_has_produced(self) -> dict[str, Any]:
        """Every forged skill, and what it has done since — the last arrow.

        A gap becomes a candidate, a candidate is verified, a verified skill
        is installed. Whether the installed skill was ever taken, and whether
        it worked when it was, is the arrow that closes the loop, and it was
        the one nothing wrote down.
        """
        rows = [
            {
                "name": one.name,
                "gap": one.gap,
                "verified": one.verified,
                "since": self._what_it_has_done_since(one.name),
            }
            for one in self._synthesized
        ]
        installed = [one for one in rows if one["since"].get("known")]
        taken = [one for one in installed if one["since"].get("taken", 0) > 0]
        return {
            "schema": "aura.forge.outcomes.v1",
            "forged": len(rows),
            "verified": sum(1 for one in rows if one["verified"]),
            "installed": len(installed),
            # Forged, verified, installed and never once taken is the honest
            # reading of a forge that is running and not paying.
            "taken_at_least_once": len(taken),
            "skills": rows,
        }

    def get_status(self) -> dict:
        outcomes = self.what_the_forge_has_produced()
        return {
            "gap_count": len(self._gap_counts),
            "gaps_at_threshold": sum(
                1 for c in self._gap_counts.values() if c >= GAP_FORGE_THRESHOLD
            ),
            "attempted": len(self._synthesized),
            "verified": sum(1 for s in self._synthesized if s.verified),
            "installed": outcomes["installed"],
            "taken_at_least_once": outcomes["taken_at_least_once"],
        }

    # ── Forging ───────────────────────────────────────────────────────────

    async def _forge_for_gap(self, gap: str, frequency: int) -> SynthesizedSkill | None:
        """Ask the verified forge for a skill that closes this gap.

        Returns a record either way. A failed forge is worth keeping: it is the
        evidence that this gap has been attempted, and without it the next pass
        would try again immediately and every pass after that.
        """
        from core.container import ServiceContainer

        hephaestus = ServiceContainer.get("hephaestus_engine", default=None)
        if hephaestus is None or not hasattr(hephaestus, "synthesize_skill"):
            logger.debug("SkillSynthesizer: no forge available; gap left open.")
            return None

        name = skill_name_for_gap(gap)
        objective = _sanitize_text_field(gap, 400)
        try:
            result = await asyncio.wait_for(
                hephaestus.synthesize_skill(name, objective),
                timeout=_FORGE_TIMEOUT_S,
            )
        except TimeoutError:
            logger.info("SkillSynthesizer: forge timed out on '%s'", name)
            return SynthesizedSkill(
                name=name, description=objective[:200], gap=gap,
                detail="the forge did not finish within its budget",
            )
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation('skill_synthesizer', exc)
            logger.debug("SkillSynthesizer: forge failed for '%s': %s", name, exc)
            return SynthesizedSkill(
                name=name, description=objective[:200], gap=gap,
                detail=f"{type(exc).__name__}: {exc}"[:200],
            )

        result = result if isinstance(result, dict) else {}
        verified = bool(result.get("ok"))
        if verified:
            logger.info(
                "SkillSynthesizer: forged and verified '%s' for a gap seen %d times.",
                name, frequency,
            )
        return SynthesizedSkill(
            name=name,
            description=objective[:200],
            gap=gap,
            digest=str(result.get("digest") or ""),
            verified=verified,
            detail=str(result.get("error") or result.get("capability") or "")[:200],
        )

    # ── Persistence ───────────────────────────────────────────────────────

    def _payload(self) -> str:
        return json.dumps(self._state(), indent=2)

    async def _save_async(self):
        """Persist off the event loop, through the gateway.

        ``synthesize_pending`` is async and used to call the synchronous saver,
        so its fsync ran on the loop. That is the failure mode this codebase has
        already paid for once — an on-loop fsync froze the live runtime for
        twenty minutes under disk pressure — and a background capability pass is
        exactly the kind of caller that would do it unnoticed.
        """
        from core.governance_context import local_internal_governed_scope
        from core.runtime.file_write_gateway import get_file_write_gateway

        try:
            payload = self._payload()
            with local_internal_governed_scope("agi.skill_synthesizer"):
                gateway = get_file_write_gateway()
                await gateway.ensure_directory_async(
                    PERSIST_PATH.parent, source="agi.skill_synthesizer"
                )
                await gateway.write_text_async(
                    PERSIST_PATH, payload, source="agi.skill_synthesizer"
                )
        except (OSError, RuntimeError, TypeError, ValueError) as e:
            record_degradation('skill_synthesizer', e)
            logger.debug("SkillSynthesizer save failed: %s", e)

    def _state(self) -> dict:
        return {
            "gaps": self._gaps[-50:],
            "gap_counts": self._gap_counts,
            "synthesized": [
                {"name": s.name, "description": s.description, "gap": s.gap,
                 "digest": s.digest, "verified": s.verified, "detail": s.detail,
                 "use_count": s.use_count, "created_at": s.created_at}
                for s in self._synthesized
            ],
        }

    def _save(self):
        """Synchronous save, for callers that are not on an event loop."""
        try:
            atomic_write_text(PERSIST_PATH, self._payload())
        except (json.JSONDecodeError, TypeError, ValueError, OSError) as e:
            # Directory creation + atomic write can raise OSError; the old tuple
            # missed it, so a persistence failure crashed after mutating state.
            record_degradation('skill_synthesizer', e)
            logger.debug("SkillSynthesizer save failed: %s", e)

    def _load(self):
        try:
            if PERSIST_PATH.exists():
                data = json.loads(PERSIST_PATH.read_text())
                self._gaps = data.get("gaps", [])
                self._gap_counts = data.get("gap_counts", {}) if isinstance(data.get("gap_counts"), dict) else {}
                for s in data.get("synthesized", []):
                    if not isinstance(s, dict) or not s.get("name"):
                        continue
                    self._synthesized.append(SynthesizedSkill(
                        name=str(s.get("name")), description=str(s.get("description", "")),
                        gap=str(s.get("gap", "")),
                        digest=str(s.get("digest", "") or ""),
                        # Records written by the old stub synthesiser carry no
                        # digest, and their skills were never runnable. Loading
                        # them as unverified is what makes the next pass forge a
                        # real one for the same gap instead of believing it is
                        # already covered.
                        verified=bool(s.get("verified")) and bool(s.get("digest")),
                        detail=str(s.get("detail", "") or ""),
                        use_count=int(s.get("use_count", 0) or 0),
                        created_at=float(s.get("created_at", time.time()) or time.time()),
                    ))
                logger.info("SkillSynthesizer: loaded %d synthesized skills.",
                            len(self._synthesized))
        except (OSError, ConnectionError, TimeoutError, json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            # Malformed JSON, missing keys, and bad types must not break
            # singleton construction — keep empty state and record it.
            record_degradation('skill_synthesizer', e)
            logger.debug("SkillSynthesizer load failed: %s", e)


# ── Singleton ─────────────────────────────────────────────────────────────────

_synthesizer: SkillSynthesizer | None = None
_synthesizer_lock = threading.Lock()


def get_skill_synthesizer() -> SkillSynthesizer:
    global _synthesizer
    if _synthesizer is None:
        with _synthesizer_lock:
            if _synthesizer is None:
                _synthesizer = SkillSynthesizer()
    return _synthesizer
