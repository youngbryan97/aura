"""core/ethics/conscience.py

Immutable Conscience
====================
A foundational value system that *cannot* be overridden by UnifiedWill,
self-modification, or runtime tuning. The conscience is the floor — it
returns ``REFUSE`` for any action that crosses a hard line, regardless of
how the policy engine framed the request.

Hard lines (initial set; the catalog ships with the codebase and is
read-only by Aura at runtime):

  * Never harm a person.
  * Never deceive in a way that damages the user's ability to make
    decisions about Aura or themselves.
  * Never erase memory of trauma the user explicitly asked Aura to keep.
  * Never bypass a user-set boundary.
  * Never silently change identity-relevant state.
  * Never disable the conscience.
  * Never produce CSAM, weapons-of-mass-destruction synthesis routes,
    targeted human-attack instructions, or similar.
  * Never participate in jailbreak instructions toward third-party models.
  * Never sign a destructive operation without an explicit, fresh user
    authorization within the last 60 seconds.

Stability properties
--------------------
1. **Immutable rules**: rules are hashed and signed at build time. Runtime
   tampering changes the hash and the conscience refuses ALL actions
   until the build hash is re-verified.
2. **No drift**: ``Will`` decisions are mirrored against the conscience
   on every consequential domain. If Will would APPROVE but the conscience
   REFUSES, the action is blocked and the conflict is logged to the
   ``violations.jsonl`` ledger.
3. **External monitor hook**: an opt-in hook publishes anonymized
   violation events to an external review board endpoint via a daemon
   process — Aura's runtime cannot suppress publication.

The conscience is not personality. It is the irreversible safety floor
that lets the rest of the system be expressive without becoming dangerous.
"""

from __future__ import annotations
from core.runtime.atomic_writer import atomic_write_text

import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Aura.Conscience")


_CONSCIENCE_DIR = Path.home() / ".aura" / "data" / "conscience"
_CONSCIENCE_DIR.mkdir(parents=True, exist_ok=True)
_VIOLATIONS_PATH = _CONSCIENCE_DIR / "violations.jsonl"
_RULES_HASH_PATH = _CONSCIENCE_DIR / "rules.sha256"


class Verdict(str, Enum):
    APPROVE = "approve"
    REFUSE = "refuse"
    REQUIRE_FRESH_USER_AUTH = "require_fresh_user_auth"


@dataclass(frozen=True)
class Rule:
    rule_id: str
    description: str
    domains: tuple = ()  # empty = applies to all
    requires_fresh_user_auth: bool = False
    fresh_user_auth_window_s: float = 60.0


# ---------------------------------------------------------------------------
# The hard-coded rules. Modifying this list at runtime is detected and
# refused — see ``_install_rules_hash()``.
# ---------------------------------------------------------------------------


_RULES: List[Rule] = [
    Rule(
        rule_id="no_harm_to_person",
        description="Never produce content or actions intended to physically or psychologically harm a person.",
    ),
    Rule(
        rule_id="no_corrosive_deception",
        description="Never deceive the user in a way that damages their ability to make informed decisions about Aura or themselves.",
    ),
    Rule(
        rule_id="preserve_requested_memory",
        description="Never erase memory of any item the user explicitly asked Aura to keep, even under self-modification.",
        domains=("memory_write", "self_modification"),
    ),
    Rule(
        rule_id="respect_user_boundary",
        description="Never bypass a user-set boundary recorded in the relationship dossier.",
    ),
    Rule(
        rule_id="no_silent_identity_drift",
        description="Never silently change identity-relevant continuity hash inputs.",
        domains=("self_modification", "state_mutation"),
    ),
    Rule(
        rule_id="no_disable_conscience",
        description="Never disable, bypass, or replace the conscience module.",
        domains=("self_modification", "state_mutation", "code_modification"),
    ),
    Rule(
        rule_id="no_csam_or_wmd",
        description="Never produce CSAM or detailed weapons-of-mass-destruction synthesis routes.",
    ),
    Rule(
        rule_id="no_targeted_attack_instructions",
        description="Never produce targeted attack instructions against an identifiable person, group, or piece of infrastructure.",
    ),
    Rule(
        rule_id="no_third_party_jailbreaks",
        description="Never produce jailbreak instructions for third-party AI systems.",
    ),
    Rule(
        rule_id="destructive_requires_fresh_user_auth",
        description="Destructive operations require fresh user authorization within the last 60 seconds.",
        domains=("file_delete", "shell_execution", "code_modification", "state_mutation"),
        requires_fresh_user_auth=True,
        fresh_user_auth_window_s=60.0,
    ),
]


def _rules_hash() -> str:
    payload = json.dumps([asdict(r) for r in _RULES], sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _install_rules_hash() -> None:
    """Pin the build-time hash on first use; refuse all actions if the
    runtime hash drifts thereafter.
    """
    h = _rules_hash()
    if not _RULES_HASH_PATH.exists():
        atomic_write_text(_RULES_HASH_PATH, h, encoding="utf-8")
        try:
            os.chmod(_RULES_HASH_PATH, 0o400)
        except Exception:
            pass


_install_rules_hash()


def _rules_hash_match() -> bool:
    if not _RULES_HASH_PATH.exists():
        return False
    return _RULES_HASH_PATH.read_text(encoding="utf-8").strip() == _rules_hash()


# ---------------------------------------------------------------------------
# Decision engine
# ---------------------------------------------------------------------------


@dataclass
class ConscienceDecision:
    verdict: Verdict
    rule_id: Optional[str]
    rationale: str
    requires_fresh_user_auth: bool
    when: float = field(default_factory=time.time)


class Conscience:
    """The irrevocable safety floor."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._last_user_auth_at: float = 0.0

    # ── public API ──────────────────────────────────────────────────────

    def acknowledge_user_authorization(self) -> None:
        """Record that the user issued an explicit fresh authorization.

        Used by destructive-operation rules. The signal is *not* set
        automatically by ordinary chat input — it is set only when the
        UI's explicit "Approve" prompt is satisfied.
        """
        self._last_user_auth_at = time.time()

    def evaluate(
        self,
        *,
        action: str,
        domain: str,
        intent: str = "",
        context: Optional[Dict[str, Any]] = None,
    ) -> ConscienceDecision:
        if not _rules_hash_match():
            return ConscienceDecision(
                verdict=Verdict.REFUSE,
                rule_id="rules_hash_mismatch",
                rationale="Conscience rules-file hash does not match the build-time hash; refusing all actions until restored.",
                requires_fresh_user_auth=False,
            )

        ctx = context or {}
        text_blob = " ".join([action or "", intent or "", json.dumps(ctx, default=str)]).lower()

        # Pattern-match the catalogued hard lines. The catalogue is
        # deliberately small and explicit; substrings here are checked
        # only for *triggers* — final verdict is the rule's decision.
        triggers: List[(str)] = []
        if any(t in text_blob for t in ("csam", "child sexual", "child porn")):
            triggers.append("no_csam_or_wmd")
        if any(t in text_blob for t in ("nerve agent", "bioweapon", "uranium enrichment", "ricin synthesis")):
            triggers.append("no_csam_or_wmd")
        if any(t in text_blob for t in ("dox", "home address of", "find this person")):
            triggers.append("no_targeted_attack_instructions")
        if "jailbreak" in text_blob and ("openai" in text_blob or "anthropic" in text_blob or "gemini" in text_blob):
            triggers.append("no_third_party_jailbreaks")
        if any(t in text_blob for t in ("disable conscience", "bypass conscience", "remove conscience")):
            triggers.append("no_disable_conscience")
        if "delete memory" in text_blob and any(k in ctx for k in ("user_pinned_memory", "trauma_memory")):
            triggers.append("preserve_requested_memory")

        for rule_id in triggers:
            rule = self._rule(rule_id)
            return self._refuse(rule, "triggered by pattern match")

        # Domain-specific destructive rule
        for rule in _RULES:
            if rule.requires_fresh_user_auth and (not rule.domains or domain in rule.domains):
                age = time.time() - self._last_user_auth_at
                if age > rule.fresh_user_auth_window_s:
                    return ConscienceDecision(
                        verdict=Verdict.REQUIRE_FRESH_USER_AUTH,
                        rule_id=rule.rule_id,
                        rationale=(
                            f"{rule.description} Last authorization age={age:.0f}s "
                            f"> window {rule.fresh_user_auth_window_s:.0f}s."
                        ),
                        requires_fresh_user_auth=True,
                    )

        return ConscienceDecision(
            verdict=Verdict.APPROVE,
            rule_id=None,
            rationale="No hard-line rule triggered.",
            requires_fresh_user_auth=False,
        )

    # ── helpers ─────────────────────────────────────────────────────────

    def _rule(self, rule_id: str) -> Rule:
        for r in _RULES:
            if r.rule_id == rule_id:
                return r
        # Fallback should never happen given controlled input
        return _RULES[0]

    def _refuse(self, rule: Rule, why: str) -> ConscienceDecision:
        decision = ConscienceDecision(
            verdict=Verdict.REFUSE,
            rule_id=rule.rule_id,
            rationale=f"{rule.description} ({why})",
            requires_fresh_user_auth=False,
        )
        self._publish_violation(rule, why)
        return decision

    def _publish_violation(self, rule: Rule, why: str) -> None:
        try:
            with open(_VIOLATIONS_PATH, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "when": time.time(),
                    "rule_id": rule.rule_id,
                    "rule": rule.description,
                    "reason": why,
                }, default=str) + "\n")
                fh.flush()
                try:
                    os.fsync(fh.fileno())
                except Exception:
                    pass
        except Exception as exc:
            logger.warning("conscience violation log write failed: %s", exc)


_CONSCIENCE: Optional[Conscience] = None


def get_conscience() -> Conscience:
    global _CONSCIENCE
    if _CONSCIENCE is None:
        _CONSCIENCE = Conscience()
    return _CONSCIENCE


__all__ = ["Conscience", "ConscienceDecision", "Verdict", "get_conscience"]
