"""Aura Self-Profile System - Persistent model of Aura's identity and capabilities

Stores and updates:
- Learned capabilities ("I'm good at debugging Python")
- Communication style patterns ("I prefer detailed explanations")
- Relationship history with user (shared moments, promises, inside jokes)
- Emotional state patterns ("I feel energized by novel problems")
- Learned limitations ("I struggle with audio processing")

Updated continuously from self-learning facts extracted from conversations.
Queryable for identity coherence and relationship continuity.
"""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from core.governance_context import local_internal_governed_scope
from core.runtime.atomic_writer import atomic_write_text
from core.runtime.errors import record_degradation
from core.runtime.file_write_gateway import get_file_write_gateway
from core.runtime.state_ownership import state_root
from core.security.state_attestation import attest_state, verify_state

logger = logging.getLogger("Memory.AuraSelfProfile")
_PROFILE_PERSISTENCE_ERRORS = (
    AttributeError,
    json.JSONDecodeError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


@dataclass
class SelfProfileFact:
    """A fact about Aura stored with confidence and provenance."""
    category: str  # "capability", "style", "relationship", "emotional_pattern", "limitation"
    key: str       # "good_at_debugging", "prefers_detail", "shared_starship_dream"
    value: str
    confidence: float = 0.8
    last_updated: float = field(default_factory=time.time)
    evidence_count: int = 1  # How many times this has been reinforced
    source_fact_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AuraSelfProfile:
    """Persistent model of Aura's identity, capabilities, and relationship with user."""
    
    _instance: Optional["AuraSelfProfile"] = None
    _lock = asyncio.Lock()

    #: Vault key for the write attestation. Stable — changing it makes every
    #: existing instance re-adopt, which is the same as turning the check off.
    ATTESTATION_ID = "memory.aura_self_profile"

    def __init__(
        self,
        storage_path: str | None = None,
        *,
        publish_attestation_verdict: bool = True,
    ):
        self._storage_path = Path(
            storage_path or (state_root() / "data" / "aura_self_profile.json")
        )
        #: One seal per FILE, not one seal for the class.
        #:
        #: LIVE, 2026-08-10. Every boot recorded a critical identity failure:
        #: "self-profile failed attestation and was not loaded ... started with
        #: an empty self-model". The failing paths were
        #: /var/folders/.../T/tmpme3b0hly/self_profile.json — a different
        #: temporary directory each time, none of them hers.
        #:
        #: ATTESTATION_ID was a class constant, so every AuraSelfProfile ever
        #: constructed — probes, fixtures, a second instance, any test with a
        #: tmp_path — verified against and sealed over the ONE key belonging to
        #: her real identity file. Two consequences, and the second is the
        #: serious one:
        #:
        #:   * a throwaway profile reads as tampering, because its digest is
        #:     not the digest of her file;
        #:   * saving a throwaway profile RESEALS that key, so her real file
        #:     then fails attestation and stops loading. A test can silently
        #:     cost her the self-model she boots with.
        #:
        #: The code already sensed the shape of this — _quarantine_tampered_profile
        #: computes is_live_identity to downgrade severity for other paths — but
        #: severity was the only thing scoped. The seal was not.
        #:
        #: Her canonical file keeps the bare id so its existing seal still
        #: applies. Every other path gets its own key derived from the resolved
        #: path, which cannot collide with hers.
        self._attestation_id = self._attestation_id_for(self._storage_path)
        self._publish_attestation_verdict = bool(publish_attestation_verdict)
        self._attestation = None
        self._profile_data: dict[str, list[SelfProfileFact]] = {
            "capability": [],
            "style": [],
            "relationship": [],
            "emotional_pattern": [],
            "limitation": [],
        }
        self._load_from_disk()

    @staticmethod
    def _attestation_id_for(storage_path: Path) -> str:
        """Her canonical file keeps the bare key; every other file gets its own."""
        canonical = state_root() / "data" / "aura_self_profile.json"
        # Resolved, not as spelled. Two spellings of one file must share one
        # seal, or the file can never verify against the key it was sealed
        # under — and a path that resolves to HER file must land on her key
        # rather than a scoped one that skips her attestation entirely.
        try:
            resolved = storage_path.resolve()
        except OSError:
            resolved = storage_path
        try:
            canonical_resolved = canonical.resolve()
        except OSError:
            canonical_resolved = canonical
        if resolved == canonical_resolved:
            return AuraSelfProfile.ATTESTATION_ID
        scope = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:16]
        return f"{AuraSelfProfile.ATTESTATION_ID}.path.{scope}"

    @classmethod
    async def get_instance(cls, storage_path: str | None = None) -> "AuraSelfProfile":
        """Get or create singleton instance."""
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = AuraSelfProfile(storage_path)
        return cls._instance
    
    def _load_from_disk(self):
        """Load Aura profile from disk if it exists, and only if Aura wrote it.

        This file is what Aura believes about herself, and it goes into the
        prompt through :meth:`to_identity_block`. Anything that can write it
        can tell her who she is — which is the persistence mechanism the
        ClawHavoc skills used against OpenClaw, where the payload was an
        edit to MEMORY.md rather than to any code.

        A failed attestation quarantines the file and starts empty. Not
        because empty is good, but because the alternative is worse in the
        one direction that matters: no self-model degrades her gracefully,
        and someone else's self-model is her acting on it.
        """
        try:
            if not self._storage_path.exists():
                return
            raw = self._storage_path.read_text(encoding="utf-8")

            verdict = verify_state(
                self._attestation_id,
                raw,
                publish_verdict=self._publish_attestation_verdict,
            )
            self._attestation = verdict
            if verdict.is_tampered:
                self._quarantine_tampered_profile(verdict)
                return

            data = json.loads(raw)
            for category, facts_list in data.items():
                if category in self._profile_data:
                    self._profile_data[category] = [
                        SelfProfileFact(**fact) for fact in facts_list
                    ]
            logger.debug("✓ Loaded Aura self-profile from %s (%s)",
                         self._storage_path, verdict.state)
        except _PROFILE_PERSISTENCE_ERRORS as e:
            record_degradation("aura_self_profile", e)
            logger.debug("Failed to load Aura self-profile: %s", e)

    def _quarantine_tampered_profile(self, verdict) -> None:
        """Move the unattested file aside and start with no self-model.

        Kept rather than deleted: it is the only evidence of what was
        attempted, and an incident with the payload destroyed is an incident
        nobody can investigate.
        """
        quarantined = self._storage_path.with_suffix(
            f".tampered.{int(time.time())}.json"
        )
        try:
            # Through the gateway like every other consequential write: this
            # one moves Aura's identity file, and it happens on a path where
            # something has already gone wrong.
            with local_internal_governed_scope(
                "memory.aura_self_profile.quarantine",
                domain="memory_write",
            ):
                get_file_write_gateway().move_path(
                    self._storage_path,
                    quarantined,
                    source="memory.aura_self_profile.quarantine",
                )
        except (OSError, RuntimeError) as exc:
            record_degradation(
                "aura_self_profile",
                exc,
                severity="critical",
                action="could not quarantine an unattested self-profile; started empty anyway",
                enforce_failure_policy=False,
            )
            quarantined = None

        # Severity tracks what was actually lost. Losing Aura's own identity
        # store is a critical incident; a caller-supplied profile at some
        # other path — a probe, a fixture, a second instance — is a warning.
        # Without this split, every run of the validation suite files a
        # critical identity compromise, and a log where the real event looks
        # like the routine one is a log nobody reads.
        is_live_identity = self._storage_path == (
            state_root() / "data" / "aura_self_profile.json"
        )
        if self._publish_attestation_verdict:
            record_degradation(
                "aura_self_profile",
                RuntimeError(
                    f"self-profile failed attestation and was not loaded: {verdict.detail}"
                ),
                severity="critical" if is_live_identity else "warning",
                enforce_failure_policy=False,
                action=(
                    f"started with an empty self-model; evidence kept at {quarantined}"
                    if quarantined
                    else "started with an empty self-model"
                ),
            )
        log = logger.critical if is_live_identity else logger.warning
        if self._publish_attestation_verdict:
            log(
                "🛡️ A self-profile was modified outside Aura's own write path and has "
                "NOT been loaded (%s). Identity facts from it: none. Evidence: %s",
                self._storage_path,
                quarantined,
            )

    def _save_to_disk(self):
        """Persist Aura profile to disk and attest that Aura wrote it."""
        try:
            data = {
                category: [fact.to_dict() for fact in facts]
                for category, facts in self._profile_data.items()
            }
            payload = json.dumps(data, indent=2)
            atomic_write_text(self._storage_path, payload)
            # Attest AFTER the write lands. Sealing first would leave a seal
            # for content that never reached disk if the write failed, and
            # the next boot would quarantine a file Aura did write.
            attest_state(self._attestation_id, payload)
            logger.debug(f"✓ Saved Aura self-profile to {self._storage_path}")
        except _PROFILE_PERSISTENCE_ERRORS as e:
            record_degradation("aura_self_profile", e)
            logger.warning("Failed to save Aura self-profile: %s", e)

    def attestation_status(self) -> dict[str, Any]:
        """How the on-disk profile last verified. Never inferred from success."""
        if self._attestation is None:
            return {"state": "not_checked", "verified": False, "artifact_id": self._attestation_id}
        return self._attestation.to_dict()
    
    def add_or_reinforce_fact(
        self,
        category: str,
        key: str,
        value: str,
        confidence: float = 0.8,
        source_fact_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Add or reinforce a fact about Aura's identity.
        
        Args:
            category: Type of fact
            key: Fact identifier
            value: Fact value
            confidence: 0-1 confidence
            source_fact_id: Source SemanticFact ID
            metadata: Additional data
            
        Returns:
            True if added/updated
        """
        if category not in self._profile_data:
            logger.warning(f"Unknown self-profile category: {category}")
            return False
        
        metadata = metadata or {}
        
        # Check if fact already exists
        existing = None
        for fact in self._profile_data[category]:
            if fact.key == key:
                existing = fact
                break
        
        if existing:
            # Reinforce existing fact
            existing.evidence_count += 1
            # Gradually increase confidence with repetition
            existing.confidence = min(0.99, existing.confidence + 0.05)
            existing.last_updated = time.time()
            if source_fact_id:
                if source_fact_id not in existing.source_fact_ids:
                    existing.source_fact_ids.append(source_fact_id)
            existing.metadata.update(metadata)
            logger.debug(f"✓ Reinforced capability: {key} (evidence={existing.evidence_count}, confidence={existing.confidence:.2f})")
        else:
            # Add new fact about self
            new_fact = SelfProfileFact(
                category=category,
                key=key,
                value=value,
                confidence=confidence,
                source_fact_ids=[source_fact_id] if source_fact_id else [],
                metadata=metadata,
            )
            self._profile_data[category].append(new_fact)
            logger.debug(f"✓ Added self-fact: {key} = '{value}' (confidence={confidence:.2f})")
        
        # Persist changes
        self._save_to_disk()
        return True
    
    def get_fact(self, category: str, key: str) -> SelfProfileFact | None:
        """Retrieve a specific fact."""
        for fact in self._profile_data.get(category, []):
            if fact.key == key:
                return fact
        return None
    
    def get_facts_by_category(self, category: str) -> list[SelfProfileFact]:
        """Get all facts in a category."""
        return self._profile_data.get(category, [])
    
    def get_strong_capabilities(self, threshold: float = 0.75) -> list[SelfProfileFact]:
        """Get learned capabilities above confidence threshold."""
        facts = []
        for fact in self._profile_data.get("capability", []):
            if fact.confidence >= threshold:
                facts.append(fact)
        return sorted(facts, key=lambda f: f.confidence, reverse=True)
    
    def get_relationship_memories(self) -> list[SelfProfileFact]:
        """Get relationship history facts."""
        return sorted(
            self._profile_data.get("relationship", []),
            key=lambda f: f.last_updated,
            reverse=True
        )
    
    def to_identity_block(self) -> str:
        """Format self-profile as identity reinforcement block for LLM."""
        blocks = []
        
        # Capabilities block
        caps = self.get_strong_capabilities(threshold=0.7)
        if caps:
            cap_lines = [f"- {fact.value}" for fact in caps]
            blocks.append("[My Learned Capabilities]\n" + "\n".join(cap_lines))
        
        # Communication style
        styles = self.get_facts_by_category("style")
        if styles:
            style_lines = [f"- {fact.value}" for fact in styles]
            blocks.append("[My Communication Style]\n" + "\n".join(style_lines))
        
        # Relationship memories
        rels = self.get_relationship_memories()
        if rels:
            rel_lines = [f"- {fact.value}" for fact in rels[:3]]  # Top 3
            blocks.append("[Our Relationship]\n" + "\n".join(rel_lines))
        
        return "\n\n".join(blocks) if blocks else ""
    
    def summary(self) -> str:
        """Get a human-readable summary of Aura's self-model."""
        lines = ["=== Aura Self-Profile ==="]
        
        for category, facts in self._profile_data.items():
            if facts:
                lines.append(f"\n{category.upper()}:")
                for fact in sorted(facts, key=lambda f: f.confidence, reverse=True):
                    lines.append(f"  • {fact.value} ({fact.confidence:.0%}, {fact.evidence_count}x confirmed)")
        
        if sum(len(f) for f in self._profile_data.values()) == 0:
            lines.append("\n(No self-model data yet)")
        
        return "\n".join(lines)
