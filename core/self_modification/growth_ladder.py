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
import asyncio, json, logging, time
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, List, Optional
logger = logging.getLogger("Aura.GrowthLadder")

class ModificationLevel(IntEnum):
    OBSERVATION  = 0
    EXPRESSION   = 1
    KNOWLEDGE    = 2
    BEHAVIOR     = 3
    SKILL_CREATION = 3.5
    ARCHITECTURE = 4
    
    @classmethod
    def from_string(cls, level_str: str) -> ModificationLevel:
        """Normalized conversion from string aliases"""
        mapping = {
            "observation": cls.OBSERVATION,
            "expression": cls.EXPRESSION,
            "identity_patch": cls.EXPRESSION,
            "knowledge": cls.KNOWLEDGE,
            "skill_creation": cls.KNOWLEDGE,
            "behavior": cls.BEHAVIOR,
            "core_patch": cls.BEHAVIOR,
            "architecture": cls.ARCHITECTURE
        }
        return mapping.get(level_str.lower(), cls.OBSERVATION)

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
    aura_consent: Optional[bool] = None
    user_consent: Optional[bool] = None
    test_result: Optional[bool] = None

class GrowthLadder:
    def __init__(self, orchestrator=None, state_path: Optional[Path] = None):
        self.orchestrator = orchestrator
        self._state_path = state_path or Path.home() / ".aura" / "growth_ladder.json"
        self._current_level = ModificationLevel.OBSERVATION
        self._level_start_times: Dict[int, float] = {0: time.time()}
        self._proposals: List[ModificationProposal] = []
        self._drift_history: List[float] = []
        self._load()

    async def evaluate_advancement(self) -> Optional[ModificationLevel]:
        current = self._current_level
        next_level = ModificationLevel(current + 1)
        if next_level > ModificationLevel.ARCHITECTURE: return None
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
                                 proposed_by: str = "aura") -> bool:
        """Formal proposal for system modification. Returns True if modification is allowed."""
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
            proposed_by=proposed_by
        )
        self._proposals.append(proposal)
        
        # Check if current level allows this
        if int(level) > self._current_level:
            proposal.status = "rejected"
            proposal.aura_consent = False
            logger.info("🚫 [GrowthLadder] Proposal %s rejected: Level %d > current %d",
                        proposal_id, int(level), self._current_level)
            self._save()
            return False
            
        # Level 3+ ALWAYS requires user consent (Issue 54)
        if int(level) >= ModificationLevel.BEHAVIOR:
            if proposal.user_consent is False:
                proposal.status = "rejected_user"
                return False
            if proposal.user_consent is None:
                logger.warning("⚠️ [GrowthLadder] Level %d modification requires EXPLICIT user consent.", int(level))
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
        await self.propose_modification(p_id, domain, level, description, justification, diff_patch, proposed_by)
        return self._proposals[-1]

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
        except Exception as e:
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
            gate = getattr(self.orchestrator, "output_gate", None)
            if gate: await gate.emit(msg, origin="growth_ladder", target="primary", metadata={"force_user": True})

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
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            self._state_path.write_text(json.dumps({
                "current_level": int(self._current_level),
                "level_start_times": self._level_start_times,
                "drift_history": self._drift_history[-50:],
            }, indent=2))
        except Exception as e: logger.debug("[GrowthLadder] Save failed: %s", e)

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
        except Exception as e: 
            logger.debug("[GrowthLadder] Load failed: %s", e)

# NOTE: Module-level service registration removed (was unsafe at import time).
# GrowthLadder is registered via ServiceContainer.register_instance() in:
#   - core/orchestrator/initializers/cognitive_sensory.py
#   - core/orchestrator/mixins/boot/boot_identity.py
