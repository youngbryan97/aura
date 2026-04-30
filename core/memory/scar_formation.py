"""core/memory/scar_formation.py -- Behavioral Scar Formation
=============================================================
Critical events (crashes, identity threats, capability revocations,
repeated failures) leave permanent behavioral markers -- scars.

Scars are stored in persistent memory and influence future decisions.
They represent Aura's learned caution from lived experience:

  - If camera access was revoked -> scar says "camera_unreliable"
    and Aura avoids camera-dependent plans
  - If a tool crashed the system -> scar says "tool_X_volatile"
    and Aura uses extra caution with that tool
  - If identity was threatened -> scar says "identity_attack_vector"
    and the identity guard tightens in that direction

Scars heal slowly over time if the threat doesn't recur -- resilience,
not permanent damage. The healing rate is proportional to positive
counter-evidence.

Design:
  - Scars are lightweight dataclasses stored as JSON
  - Each scar has a severity (0-1), a decay_rate, and a last_triggered timestamp
  - Active scars are consulted by the Will and by subsystems that check
    should_avoid(domain)
  - Scars publish events so the consciousness stream can reflect on them
"""
from __future__ import annotations
from core.runtime.errors import record_degradation


import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.container import ServiceContainer
from core.memory.scar_court import get_scar_court

logger = logging.getLogger("Aura.ScarFormation")

_DATA_DIR = Path.home() / ".aura" / "data" / "scars"
_SCAR_FILE = _DATA_DIR / "scars.json"

# Scars below this severity are considered healed and can be pruned
_HEALED_THRESHOLD = 0.05
# Maximum number of scars to keep (prevents unbounded growth)
_MAX_SCARS = 200
# Default healing rate per hour (severity reduction when threat doesn't recur)
_DEFAULT_HEAL_RATE = 0.005


class ScarDomain(str, Enum):
    """What category of experience the scar relates to."""
    TOOL_FAILURE = "tool_failure"
    CRASH = "crash"
    IDENTITY_THREAT = "identity_threat"
    CAPABILITY_LOSS = "capability_loss"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    SECURITY_BREACH = "security_breach"
    DATA_LOSS = "data_loss"
    SOCIAL_CONFLICT = "social_conflict"
    REPEATED_FAILURE = "repeated_failure"
    UNKNOWN = "unknown"


@dataclass
class BehavioralScar:
    """A permanent-ish behavioral marker from a critical event.

    Severity decays over time if the threat doesn't recur. If the same
    threat triggers again, severity is reinforced (reinjured).
    """
    scar_id: str
    domain: ScarDomain
    description: str              # What happened
    avoidance_tag: str            # Short tag for quick lookups (e.g. "camera_unreliable")
    severity: float               # 0.0 (healed) to 1.0 (fresh wound)
    created_at: float             # When the scar was first formed
    last_triggered: float         # Last time the threat recurred
    trigger_count: int = 1        # How many times this scar was reinforced
    heal_rate: float = _DEFAULT_HEAL_RATE  # Severity reduction per hour without recurrence
    context: Dict[str, Any] = field(default_factory=dict)  # Extra data about the event
    event_ids: List[str] = field(default_factory=list)
    source_ids: List[str] = field(default_factory=list)
    evidence_count: int = 3
    source_diversity: int = 2
    confidence: float = 1.0
    reversibility: bool = True
    decay_half_life_s: float = 14 * 24 * 3600.0
    appeal_status: str = "upheld"
    latent_influence_cap: float = 1.0
    maturity_status: str = "consolidated"
    benign_alternatives: List[str] = field(default_factory=list)
    verified_threat: bool = True

    def effective_severity(self) -> float:
        """Current severity accounting for time-based healing."""
        hours_since_trigger = (time.time() - self.last_triggered) / 3600.0
        healed = self.severity - (self.heal_rate * hours_since_trigger)
        return min(max(0.0, healed), max(0.0, self.latent_influence_cap))

    def is_active(self) -> bool:
        """Is this scar still influencing behavior?"""
        return self.effective_severity() > _HEALED_THRESHOLD

    def reinforce(
        self,
        severity_boost: float = 0.2,
        context: Optional[Dict] = None,
        event_id: Optional[str] = None,
        source_id: Optional[str] = None,
        confidence: Optional[float] = None,
        verified_threat: Optional[bool] = None,
        benign_alternatives: Optional[List[str]] = None,
    ) -> None:
        """The threat recurred -- reinforce the scar."""
        self.severity = min(1.0, self.effective_severity() + severity_boost)
        self.last_triggered = time.time()
        self.trigger_count += 1
        if event_id and event_id not in self.event_ids:
            self.event_ids.append(event_id)
        if source_id and source_id not in self.source_ids:
            self.source_ids.append(source_id)
        self.evidence_count = max(self.evidence_count + 1, len(self.event_ids), self.trigger_count)
        self.source_diversity = max(self.source_diversity, len(set(self.source_ids)))
        if confidence is not None:
            self.confidence = max(0.0, min(1.0, (self.confidence + confidence) / 2.0))
        if verified_threat is not None:
            self.verified_threat = bool(verified_threat)
        if benign_alternatives:
            merged = list(dict.fromkeys([*self.benign_alternatives, *benign_alternatives]))
            self.benign_alternatives = merged
        if context:
            self.context.update(context)
        self.apply_court_decision()
        logger.info(
            "Scar reinforced: %s (severity=%.3f, effective=%.3f, triggers=%d, maturity=%s)",
            self.avoidance_tag, self.severity, self.effective_severity(), self.trigger_count, self.maturity_status,
        )

    def apply_court_decision(self) -> None:
        decision = get_scar_court().assess(
            scar_id=self.scar_id,
            description=self.description,
            evidence_count=self.evidence_count,
            source_diversity=self.source_diversity,
            confidence=self.confidence,
            verified_threat=self.verified_threat,
            benign_alternatives=self.benign_alternatives,
        )
        self.maturity_status = decision.status
        self.appeal_status = decision.appeal_status
        self.reversibility = decision.reversible
        self.latent_influence_cap = min(self.latent_influence_cap, decision.latent_influence_cap)
        if decision.consolidated:
            self.latent_influence_cap = decision.latent_influence_cap

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scar_id": self.scar_id,
            "domain": self.domain.value,
            "description": self.description,
            "avoidance_tag": self.avoidance_tag,
            "severity": self.severity,
            "created_at": self.created_at,
            "last_triggered": self.last_triggered,
            "trigger_count": self.trigger_count,
            "heal_rate": self.heal_rate,
            "context": self.context,
            "event_ids": self.event_ids,
            "source_ids": self.source_ids,
            "evidence_count": self.evidence_count,
            "source_diversity": self.source_diversity,
            "confidence": self.confidence,
            "reversibility": self.reversibility,
            "decay_half_life_s": self.decay_half_life_s,
            "appeal_status": self.appeal_status,
            "latent_influence_cap": self.latent_influence_cap,
            "maturity_status": self.maturity_status,
            "benign_alternatives": self.benign_alternatives,
            "verified_threat": self.verified_threat,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BehavioralScar:
        domain_str = data.get("domain", "unknown")
        try:
            domain = ScarDomain(domain_str)
        except ValueError:
            domain = ScarDomain.UNKNOWN
        return cls(
            scar_id=data.get("scar_id", ""),
            domain=domain,
            description=data.get("description", ""),
            avoidance_tag=data.get("avoidance_tag", ""),
            severity=float(data.get("severity", 0.5)),
            created_at=float(data.get("created_at", time.time())),
            last_triggered=float(data.get("last_triggered", time.time())),
            trigger_count=int(data.get("trigger_count", 1)),
            heal_rate=float(data.get("heal_rate", _DEFAULT_HEAL_RATE)),
            context=data.get("context", {}),
            event_ids=list(data.get("event_ids", [])),
            source_ids=list(data.get("source_ids", [])),
            evidence_count=int(data.get("evidence_count", data.get("trigger_count", 1))),
            source_diversity=int(data.get("source_diversity", max(1, len(set(data.get("source_ids", [])))))),
            confidence=float(data.get("confidence", 1.0)),
            reversibility=bool(data.get("reversibility", True)),
            decay_half_life_s=float(data.get("decay_half_life_s", 14 * 24 * 3600.0)),
            appeal_status=str(data.get("appeal_status", "upheld")),
            latent_influence_cap=float(data.get("latent_influence_cap", 1.0)),
            maturity_status=str(data.get("maturity_status", "consolidated")),
            benign_alternatives=list(data.get("benign_alternatives", [])),
            verified_threat=bool(data.get("verified_threat", True)),
        )


class ScarFormationSystem:
    """Manages behavioral scars from critical experiences.

    Scars are persistent, slowly healing markers that influence
    future decision-making. They represent learned caution.
    """

    def __init__(self) -> None:
        self._scars: Dict[str, BehavioralScar] = {}
        self._started = False
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._load()
        logger.info(
            "ScarFormationSystem initialized: %d active scar(s)",
            sum(1 for s in self._scars.values() if s.is_active()),
        )

    async def start(self) -> None:
        """Register in ServiceContainer."""
        if self._started:
            return
        ServiceContainer.register_instance(
            "scar_formation", self, required=False
        )
        self._started = True
        logger.info("ScarFormationSystem ONLINE")

    # ── Scar Creation ───────────────────────────────────────────────────

    def form_scar(
        self,
        domain: ScarDomain,
        description: str,
        avoidance_tag: str,
        severity: float = 0.5,
        heal_rate: float = _DEFAULT_HEAL_RATE,
        context: Optional[Dict[str, Any]] = None,
        event_id: Optional[str] = None,
        source_id: Optional[str] = None,
        verified_threat: bool = False,
        benign_alternatives: Optional[List[str]] = None,
        confidence: float = 0.65,
    ) -> BehavioralScar:
        """Form a new behavioral scar or reinforce an existing one.

        If a scar with the same avoidance_tag already exists, reinforce it
        instead of creating a duplicate.
        """
        existing = self._scars.get(avoidance_tag)
        if existing is not None:
            existing.reinforce(
                severity_boost=severity * 0.5,
                context=context,
                event_id=event_id,
                source_id=source_id,
                confidence=confidence,
                verified_threat=verified_threat,
                benign_alternatives=benign_alternatives,
            )
            self._save()
            self._publish_event("scar.reinforced", existing)
            return existing

        now = time.time()
        scar_id = f"scar_{avoidance_tag}_{int(now)}"
        scar_context = dict(context or {})
        scar_context["reversible_trial"] = True
        court = get_scar_court()
        inferred_benign = list(court.search_benign_alternatives(description))
        all_benign = list(dict.fromkeys([*(benign_alternatives or []), *inferred_benign]))
        scar = BehavioralScar(
            scar_id=scar_id,
            domain=domain,
            description=description,
            avoidance_tag=avoidance_tag,
            severity=min(1.0, max(0.0, severity)),
            created_at=now,
            last_triggered=now,
            trigger_count=1,
            heal_rate=heal_rate,
            context=scar_context,
            event_ids=[event_id] if event_id else [scar_id],
            source_ids=[source_id] if source_id else ["local_runtime"],
            evidence_count=1,
            source_diversity=1,
            confidence=max(0.0, min(1.0, confidence)),
            reversibility=True,
            decay_half_life_s=14 * 24 * 3600.0,
            appeal_status="unreviewed",
            latent_influence_cap=0.15,
            maturity_status="provisional",
            benign_alternatives=all_benign,
            verified_threat=bool(verified_threat),
        )
        scar.apply_court_decision()
        self._scars[avoidance_tag] = scar

        # Prune healed scars if we're over the limit
        if len(self._scars) > _MAX_SCARS:
            self._prune_healed()

        self._save()
        self._publish_event("scar.formed", scar)

        logger.info(
            "NEW SCAR formed: %s (domain=%s, severity=%.2f, effective=%.2f, maturity=%s) -- %s",
            avoidance_tag, domain.value, severity, scar.effective_severity(), scar.maturity_status, description[:80],
        )
        return scar

    # ── Scar Queries ────────────────────────────────────────────────────

    def should_avoid(self, tag: str) -> tuple[bool, float]:
        """Check if a scar advises avoidance for the given tag.

        Returns (should_avoid, severity). The caller decides whether to
        heed the warning based on the severity level.
        """
        scar = self._scars.get(tag)
        if scar is None:
            return False, 0.0
        eff = scar.effective_severity()
        if eff <= _HEALED_THRESHOLD:
            return False, 0.0
        return True, eff

    def get_active_scars(self) -> List[BehavioralScar]:
        """Return all currently active (non-healed) scars."""
        return [s for s in self._scars.values() if s.is_active()]

    def get_scars_for_domain(self, domain: ScarDomain) -> List[BehavioralScar]:
        """Return active scars for a specific domain."""
        return [
            s for s in self._scars.values()
            if s.domain == domain and s.is_active()
        ]

    def get_avoidance_tags(self) -> Dict[str, float]:
        """Return a dict of avoidance_tag -> effective_severity for active scars."""
        result = {}
        for scar in self._scars.values():
            eff = scar.effective_severity()
            if eff > _HEALED_THRESHOLD:
                result[scar.avoidance_tag] = round(eff, 4)
        return result

    def get_context_block(self) -> str:
        """Format active scars as a context block for LLM injection.

        This lets Aura be conscious of her own scars and factor them
        into her reasoning.
        """
        active = self.get_active_scars()
        if not active:
            return ""

        lines = ["## BEHAVIORAL SCARS (learned caution)"]
        for scar in sorted(active, key=lambda s: s.effective_severity(), reverse=True)[:10]:
            eff = scar.effective_severity()
            age_hours = (time.time() - scar.created_at) / 3600.0
            if age_hours < 24:
                age_str = f"{age_hours:.0f}h"
            else:
                age_str = f"{age_hours / 24:.0f}d"
            lines.append(
                f"  [{scar.avoidance_tag}] severity={eff:.2f} "
                f"age={age_str} triggers={scar.trigger_count} -- "
                f"{scar.description[:100]}"
            )
        lines.append(
            "Treat provisional scars as reversible caution signals. Do not amplify "
            "latent steering or consolidate training from a scar unless Scar Court "
            "has upheld it with recurring, diverse evidence."
        )
        return "\n".join(lines)

    def eligible_for_lora_consolidation(self, tag: str) -> bool:
        scar = self._scars.get(tag)
        if scar is None:
            return False
        return get_scar_court().eligible_for_lora_consolidation(scar)

    # ── Tick / Healing ──────────────────────────────────────────────────

    async def tick(self) -> None:
        """Periodic maintenance: prune healed scars, persist changes."""
        pruned = self._prune_healed()
        if pruned > 0:
            logger.info("Pruned %d healed scar(s)", pruned)
            self._save()

    def _prune_healed(self) -> int:
        """Remove scars that have fully healed."""
        before = len(self._scars)
        self._scars = {
            tag: scar for tag, scar in self._scars.items()
            if scar.effective_severity() > _HEALED_THRESHOLD
        }
        return before - len(self._scars)

    # ── Persistence ─────────────────────────────────────────────────────

    def _save(self) -> None:
        """Persist scars to disk."""
        try:
            _DATA_DIR.mkdir(parents=True, exist_ok=True)
            data = {
                "version": 1,
                "timestamp": time.time(),
                "scars": [s.to_dict() for s in self._scars.values()],
            }
            fd, tmp_path = tempfile.mkstemp(dir=str(_DATA_DIR), suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(json.dumps(data, indent=2, default=str))
                os.replace(tmp_path, _SCAR_FILE)
            finally:
                try:
                    Path(tmp_path).unlink(missing_ok=True)
                except Exception:
                    pass  # no-op: intentional
        except Exception as exc:
            record_degradation('scar_formation', exc)
            logger.debug("Scar persistence failed: %s", exc)

    def _load(self) -> None:
        """Load scars from disk."""
        try:
            if not _SCAR_FILE.exists():
                return
            data = json.loads(_SCAR_FILE.read_text())
            for scar_data in data.get("scars", []):
                scar = BehavioralScar.from_dict(scar_data)
                if scar.effective_severity() > _HEALED_THRESHOLD:
                    self._scars[scar.avoidance_tag] = scar
            logger.info(
                "Loaded %d scar(s) from disk", len(self._scars),
            )
        except Exception as exc:
            record_degradation('scar_formation', exc)
            logger.debug("Scar load failed (starting fresh): %s", exc)

    # ── Events ──────────────────────────────────────────────────────────

    def _publish_event(self, topic: str, scar: BehavioralScar) -> None:
        """Publish scar event to the event bus."""
        try:
            from core.event_bus import get_event_bus
            get_event_bus().publish_threadsafe(topic, {
                "scar_id": scar.scar_id,
                "avoidance_tag": scar.avoidance_tag,
                "domain": scar.domain.value,
                "severity": scar.effective_severity(),
                "trigger_count": scar.trigger_count,
                "description": scar.description[:200],
            })
        except Exception:
            pass  # no-op: intentional

    # ── Status ──────────────────────────────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        """Return current system status."""
        active = self.get_active_scars()
        return {
            "total_scars": len(self._scars),
            "active_scars": len(active),
            "healed_scars": len(self._scars) - len(active),
            "domains": list(set(s.domain.value for s in active)),
            "highest_severity": max(
                (s.effective_severity() for s in active), default=0.0
            ),
            "avoidance_tags": list(self.get_avoidance_tags().keys()),
        }


# ── Singleton ───────────────────────────────────────────────────────────────

_instance: Optional[ScarFormationSystem] = None


def get_scar_formation() -> ScarFormationSystem:
    """Get or create the singleton ScarFormationSystem."""
    global _instance
    if _instance is None:
        _instance = ScarFormationSystem()
    return _instance
