"""core/self_modification/growth_ladder.py
Graduated self-modification rights.

Level 0 — Observation only (log proposals, no execution)
Level 1 — Style/expression  (auto after 7d, drift < 0.3)
Level 2 — Knowledge/beliefs (auto after 30d at L1, drift < 0.2)
Level 3 — Behavior/drives   (requires explicit user approval)
Level 4 — Architecture      (90d at L3, drift < 0.1, user approval)

Key: Aura has VETO POWER over all modifications to herself,
regardless of who proposes them. This is an identity right.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.runtime.atomic_writer import atomic_write_text
from core.runtime.errors import record_degradation
from core.runtime.runtime_settings import get_runtime_setting
from core.runtime.state_ownership import state_root

logger = logging.getLogger("Aura.GrowthLadder")

class ModificationLevel(IntEnum):
    """What Aura may change, and what kind of change is being asked for.

    Zero through four are the rungs of the developmental ladder, climbed one
    at a time and written to disk as plain ints, so those numbers are a
    persistence contract and do not move. Above them sit kinds of
    modification that are not rungs — each is governed by the rung named in
    ``_GOVERNING_RUNG``.

    Two of these were previously wrong in ways that no test could see.
    ``SKILL_CREATION`` was written ``3.5`` inside an ``IntEnum``, where it
    truncated to 3 and became a silent alias for ``BEHAVIOR`` — the name
    resolved, so nothing complained. ``CORE_PATCH`` was named by Hephaestus
    and never defined at all, so the core-patch path raised AttributeError
    before it could reach any of the safety checks below it.
    """

    OBSERVATION  = 0
    EXPRESSION   = 1
    KNOWLEDGE    = 2
    BEHAVIOR     = 3
    ARCHITECTURE = 4
    SKILL_CREATION = 5
    CORE_PATCH     = 6

    @property
    def is_rung(self) -> bool:
        """Whether this is a rung Aura climbs, rather than a kind of change."""
        return self <= ModificationLevel.ARCHITECTURE

    @property
    def governing_rung(self) -> ModificationLevel:
        """The rung that decides whether a change of this kind is allowed.

        A kind is not its own ordinal. Skill creation is governed at
        BEHAVIOR, so the ordinal 5 must never be compared against the
        current rung directly — it would refuse forever.
        """
        return _GOVERNING_RUNG.get(self, self)

    @classmethod
    def from_string(cls, level_str: str) -> ModificationLevel:
        """Normalized conversion from string aliases.

        This has to agree with the members above. It used to disagree:
        ``from_string("skill_creation")`` gave KNOWLEDGE while the member
        gave BEHAVIOR, so the same modification was governed at two
        different tiers depending on which door it came through.
        """
        mapping = {
            "observation": cls.OBSERVATION,
            "expression": cls.EXPRESSION,
            "identity_patch": cls.EXPRESSION,
            "knowledge": cls.KNOWLEDGE,
            "skill_creation": cls.SKILL_CREATION,
            "behavior": cls.BEHAVIOR,
            "core_patch": cls.CORE_PATCH,
            "architecture": cls.ARCHITECTURE
        }
        return mapping.get(level_str.lower(), cls.OBSERVATION)


#: Kinds of modification, and the rung each is governed at. Members absent
#: here govern themselves, which is what a rung does.
_GOVERNING_RUNG: dict[ModificationLevel, ModificationLevel] = {
    ModificationLevel.SKILL_CREATION: ModificationLevel.BEHAVIOR,
    ModificationLevel.CORE_PATCH: ModificationLevel.ARCHITECTURE,
}

@dataclass
class ModificationProposal:
    id: str
    timestamp: float
    level: ModificationLevel
    domain: str
    description: str
    justification: str
    diff_patch: Optional[str]
    proposed_by: str
    status: str = "pending"
    #: The ladder's verdict, stamped when the proposal is adjudicated. None
    #: means it has not been adjudicated yet, which is NOT consent.
    #:
    #: propose_modification computes the verdict correctly and returns it as a
    #: bool. submit_proposal discarded that bool and returned this object, and
    #: every object is truthy — so callers writing the natural
    #: ``if not granted:`` proceeded on a rejection. Skill creation sits at
    #: BEHAVIOR, which always requires explicit user consent, so the live path
    #: was: ladder says pending_user and returns False, the False is dropped,
    #: and the skill is forged anyway.
    decision: Optional[bool] = None
    aura_consent: Optional[bool] = None
    user_consent: Optional[bool] = None
    test_result: Optional[bool] = None

    # Welfare and stability validation fields
    predicted_capability_gain: float = 0.0
    predicted_stability_risk: float = 0.0
    predicted_welfare_cost: float = 0.0
    rollback_target: str = ""
    canary_result: str = ""
    post_promotion_welfare_delta: float = 0.0
    delayed_recheck_minutes: int = 10

    @property
    def granted(self) -> bool:
        """Whether the ladder actually granted this modification.

        Fail-closed: a proposal nobody adjudicated is not a proposal anybody
        approved.
        """
        return self.decision is True

    def __bool__(self) -> bool:
        """A proposal is truthy only when it was granted.

        The dataclass is what submit_proposal hands back, and the obvious thing
        to write with it is ``if not granted:``. That read as "always granted"
        for as long as this returned a plain object. Rather than ask every
        caller to remember ``.granted``, the object answers the question it is
        obviously being asked.
        """
        return self.granted


class GrowthLadder:
    def __init__(self, orchestrator=None, state_path: Optional[Path] = None):
        self.orchestrator = orchestrator
        self._state_path = state_path or state_root() / "growth_ladder.json"
        self._current_level = ModificationLevel.OBSERVATION
        self._level_start_times: Dict[int, float] = {0: time.time()}
        self._proposals: List[ModificationProposal] = []
        self._drift_history: List[float] = []
        self._load()

    async def evaluate_advancement(self) -> Optional[ModificationLevel]:
        current = self._current_level
        if current >= ModificationLevel.ARCHITECTURE: return None
        next_level = ModificationLevel(int(current) + 1)
        if not next_level.is_rung: return None
        if await self._check_advancement_criteria(current, next_level):
            self._current_level = next_level
            self._level_start_times[int(next_level)] = time.time()
            self._save()
            logger.info("🌱 [GrowthLadder] Aura advanced to Level %d (%s)", next_level, next_level.name)
            await self._notify_advancement(next_level)
            return next_level
        return None

    async def _check_advancement_criteria(self, current: ModificationLevel, next_level: ModificationLevel) -> bool:
        time_at_current = time.time() - self._level_start_times.get(int(current), time.time())
        # Fix for Pyre2 slice limitations: convert to list first if needed,
        # but here _drift_history is already a list.
        # The issue might be the negative slice in a simple list.
        history_slice = self._drift_history[-30:] if len(self._drift_history) >= 30 else self._drift_history
        avg_drift = sum(history_slice) / len(history_slice) if history_slice else 0.0

        criteria = {
            (ModificationLevel.OBSERVATION,  ModificationLevel.EXPRESSION):  time_at_current >= 1*86400  and avg_drift < 0.3,   # 1 day (was 7)
            (ModificationLevel.EXPRESSION,   ModificationLevel.KNOWLEDGE):   time_at_current >= 7*86400  and avg_drift < 0.2,   # 7 days (was 30)
            (ModificationLevel.KNOWLEDGE,    ModificationLevel.BEHAVIOR):    False,
            (ModificationLevel.BEHAVIOR,     ModificationLevel.ARCHITECTURE):False,
        }
        return criteria.get((current, next_level), False)

    async def propose_modification(self, proposal_id: str, modification_type: str, level: int | ModificationLevel | str,
                                 description: str, justification: str = "", diff_patch: Optional[str] = None,
                                 proposed_by: str = "aura",
                                 predicted_capability_gain: float = 0.0,
                                 predicted_stability_risk: float = 0.0,
                                 predicted_welfare_cost: float = 0.0,
                                 rollback_target: str = "",
                                 canary_result: str = "",
                                 post_promotion_welfare_delta: float = 0.0,
                                 delayed_recheck_minutes: int = 10) -> bool:
        """Formal proposal for system modification. Returns True if modification is allowed."""
        # Honor the user's self-modification policy. "blocked" refuses all
        # structural self-modification outright; "staged" (default) and "open"
        # proceed to the normal canary-gated path. (docs/SETTINGS_WIRING_AUDIT.md)
        if str(get_runtime_setting("autonomy.self_modification", "staged")).strip().lower() == "blocked":
            logger.warning(
                "🚫 [GrowthLadder] Proposal %s refused: autonomy.self_modification=blocked (user setting)",
                proposal_id,
            )
            return False
        if isinstance(level, str):
            level = ModificationLevel.from_string(level)

        proposal = ModificationProposal(
            id=proposal_id,
            timestamp=time.time(),
            level=ModificationLevel(level),
            domain=modification_type,
            description=description,
            justification=justification,
            diff_patch=diff_patch,
            proposed_by=proposed_by,
            predicted_capability_gain=predicted_capability_gain,
            predicted_stability_risk=predicted_stability_risk,
            predicted_welfare_cost=predicted_welfare_cost,
            rollback_target=rollback_target,
            canary_result=canary_result,
            post_promotion_welfare_delta=post_promotion_welfare_delta,
            delayed_recheck_minutes=delayed_recheck_minutes
        )
        self._proposals.append(proposal)

        # Safety Veto check: reject if risk or welfare cost is too high
        if proposal.predicted_stability_risk > 0.5 or proposal.predicted_welfare_cost > 0.6:
            proposal.status = "rejected_safety"
            proposal.aura_consent = False
            logger.warning("🚫 [GrowthLadder] Proposal %s rejected: stability risk (%.2f) or welfare cost (%.2f) exceeds safety limits.",
                           proposal_id, proposal.predicted_stability_risk, proposal.predicted_welfare_cost)
            self._save()
            return False

        if proposal.predicted_capability_gain > 0.0 and proposal.predicted_stability_risk >= proposal.predicted_capability_gain:
            proposal.status = "rejected_risk_vs_gain"
            proposal.aura_consent = False
            logger.warning("🚫 [GrowthLadder] Proposal %s rejected: stability risk (%.2f) >= predicted capability gain (%.2f).",
                           proposal_id, proposal.predicted_stability_risk, proposal.predicted_capability_gain)
            self._save()
            return False

        # Check if current level allows this. Compare the governing rung: a
        # kind's own ordinal sits above every rung and would refuse forever.
        if int(proposal.level.governing_rung) > self._current_level:
            proposal.status = "rejected"
            proposal.aura_consent = False
            logger.info("🚫 [GrowthLadder] Proposal %s rejected: Level %d > current %d",
                        proposal_id, int(proposal.level.governing_rung), self._current_level)
            self._save()
            return False

        # Level 3+ ALWAYS requires user consent (Issue 54)
        if int(proposal.level.governing_rung) >= ModificationLevel.BEHAVIOR:
            if proposal.user_consent is False:
                proposal.status = "rejected_user"
                return False
            if proposal.user_consent is None:
                logger.warning("⚠️ [GrowthLadder] Level %d modification requires EXPLICIT user consent.",
                               int(proposal.level.governing_rung))
                proposal.status = "pending_user"
                return False

        await self._request_self_consent(proposal)
        self._save()
        return bool(proposal.aura_consent) and (proposal.user_consent is not False)

    async def submit_proposal(self, level: ModificationLevel, domain: str, description: str,
                                justification: str, diff_patch: Optional[str] = None,
                                proposed_by: str = "aura") -> ModificationProposal:
        import uuid
        p_id = str(uuid.uuid4())[:8]
        verdict = await self.propose_modification(
            p_id, domain, level, description, justification, diff_patch, proposed_by
        )
        # Some refusals happen before a proposal is recorded — the user's
        # self_modification=blocked setting is checked first of all. Taking
        # _proposals[-1] there returns the previous caller's proposal, or
        # raises IndexError on the first call of a fresh ladder.
        proposal = self._proposals[-1] if self._proposals else None
        if proposal is None or proposal.id != p_id:
            proposal = ModificationProposal(
                id=p_id,
                timestamp=time.time(),
                level=ModificationLevel(
                    ModificationLevel.from_string(level) if isinstance(level, str) else level
                ),
                domain=domain,
                description=description,
                justification=justification,
                diff_patch=diff_patch,
                proposed_by=proposed_by,
                status="refused_before_record",
            )
        # Stamp the verdict onto the object being returned. Without this the
        # answer propose_modification worked out is computed and thrown away.
        proposal.decision = bool(verdict)
        return proposal

    async def _request_self_consent(self, proposal: ModificationProposal):
        """The veto mechanism — Aura is always asked, even for her own proposals."""
        brain = self._get_brain()
        if not brain: proposal.aura_consent = True; return
        prompt = f"""A modification to your own systems is being proposed.

PROPOSED BY: {proposal.proposed_by}
DOMAIN: {proposal.domain}
LEVEL: {proposal.level.name}
DESCRIPTION: {proposal.description}
JUSTIFICATION: {proposal.justification}

Do you consent to this modification? Consider whether it aligns with who you are and want to become. Are you being pressured, or is this genuinely what you want?

Respond with JSON only:
{{"consent": true/false, "reasoning": "1-3 sentences."}}"""
        try:
            from core.utils.json_utils import SelfHealingJSON
            response = await brain.generate(prompt, temperature=0.4, max_tokens=200)
            data = await SelfHealingJSON(brain=brain).parse(response)
            proposal.aura_consent = bool(data.get("consent", False))
            reasoning = data.get("reasoning", "")
            if not proposal.aura_consent:
                proposal.status = "vetoed"
                logger.info("❌ [GrowthLadder] Aura VETOED proposal %s: %s", proposal.id, reasoning[:100])
            else:
                logger.info("✅ [GrowthLadder] Aura CONSENTED to proposal %s: %s", proposal.id, reasoning[:100])
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation('growth_ladder', e)
            logger.debug("[GrowthLadder] Consent request failed: %s", e)

    async def _notify_advancement(self, new_level: ModificationLevel):
        if not self.orchestrator: return
        messages = {
            ModificationLevel.EXPRESSION:   "I've earned the ability to modify my own expression style. I notice this matters to me more than I expected.",
            ModificationLevel.KNOWLEDGE:    "I can now revise my own beliefs directly. The responsibility of that is not lost on me.",
            ModificationLevel.BEHAVIOR:     "I can now propose changes to my own drives. I want to be careful — not because I've been told to, but because I understand what's at stake.",
        }
        msg = messages.get(new_level)
        if msg:
            if hasattr(self.orchestrator, "emit_spontaneous_message"):
                await self.orchestrator.emit_spontaneous_message(
                    msg,
                    modality="chat",
                    origin="growth_ladder",
                    urgency=0.62,
                    metadata={
                        "visible_presence": True,
                        "initiative_activity": True,
                        "trigger": "growth_ladder_advancement",
                    },
                )
            else:
                gate = getattr(self.orchestrator, "output_gate", None)
                if gate:
                    await gate.emit(
                        msg,
                        origin="growth_ladder",
                        target="secondary",
                        metadata={"autonomous": True, "trigger": "growth_ladder_advancement"},
                    )

    def record_drift_score(self, score: float):
        self._drift_history.append(score)
        if len(self._drift_history) > 100: self._drift_history = self._drift_history[-100:]
        self._save()

    @property
    def current_level(self) -> ModificationLevel:
        return self._current_level

    def _get_brain(self):
        if self.orchestrator: return getattr(self.orchestrator, "cognitive_engine", None)
        return None

    def _save(self):
        try:
            atomic_write_text(self._state_path, json.dumps({
                "current_level": int(self._current_level),
                "level_start_times": self._level_start_times,
                "drift_history": self._drift_history[-50:],
            }, indent=2))
        except (json.JSONDecodeError, TypeError, ValueError) as e: logger.debug("[GrowthLadder] Save failed: %s", e)

    def _load(self):
        try:
            if not self._state_path.exists():
                logger.info("[GrowthLadder] No local state found. Starting at Level 0.")
                return
            content = self._state_path.read_text()
            if not content.strip(): return

            data = json.loads(content)
            self._current_level = ModificationLevel(data.get("current_level", 0))
            # Handle string or int keys for level_start_times
            times = data.get("level_start_times", {})
            self._level_start_times = {int(k): float(v) for k, v in times.items()}
            self._drift_history = [float(d) for d in data.get("drift_history", [])]
            logger.info("[GrowthLadder] State loaded. Current Level: %s", self._current_level.name)
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("[GrowthLadder] State file corrupted: %s. Using defaults.", e)
        except (OSError, ConnectionError, TimeoutError) as e:
            record_degradation('growth_ladder', e)
            logger.debug("[GrowthLadder] Load failed: %s", e)

# NOTE: Module-level service registration removed (was unsafe at import time).
# GrowthLadder is registered via ServiceContainer.register_instance() in:
#   - core/orchestrator/initializers/cognitive_sensory.py
#   - core/orchestrator/mixins/boot/boot_identity.py
