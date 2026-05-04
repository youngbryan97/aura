"""Structured affordance memory for any embodied environment.

Affordances are not brittle scripts. They are inspectable hypotheses about
what actions are possible, under what preconditions, with what effects and
risks. They can be seeded from documentation, inferred by a reasoning model,
or reinforced/disconfirmed by experience.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from core.config import config

logger = logging.getLogger("Aura.AffordanceSchema")

@dataclass
class Affordance:
    entity: str                   # e.g., "floating eye", "potion", "altar"
    action: str                   # e.g., "melee", "quaff", "pray"
    preconditions: List[str]      # e.g., ["adjacent", "can see"]
    effects: List[str]            # e.g., ["paralyzes player", "identifies BUC"]
    risk_level: float             # 0.0 (safe) to 1.0 (deadly)
    confidence: float             # 0.0 (guess) to 1.0 (verified by experience)
    source: str                   # "experience", "inference"
    tags: List[str] = field(default_factory=list)
    observations: int = 1
    last_tested: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "entity": self.entity,
            "action": self.action,
            "preconditions": self.preconditions,
            "effects": self.effects,
            "risk_level": self.risk_level,
            "confidence": self.confidence,
            "source": self.source,
            "tags": self.tags,
            "observations": self.observations,
            "last_tested": self.last_tested,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'Affordance':
        return cls(**d)


class AffordanceKnowledgeBase:
    """
    Persistent store for learned affordances.
    """

    def __init__(self, domain: str = "generic", storage_path: Optional[Path] = None):
        self.domain = domain
        self.storage_path = Path(storage_path) if storage_path else config.paths.data_dir / f"knowledge/{domain}_affordances.json"
        self.affordances: Dict[str, List[Affordance]] = {}
        self._load()
        if not self.affordances:
            self.seed_general_doctrine()

    def _load(self):
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    for entity, aff_list in data.items():
                        self.affordances[entity] = [Affordance.from_dict(a) for a in aff_list]
                logger.info(f"Loaded {sum(len(v) for v in self.affordances.values())} affordances for {self.domain}.")
        except Exception as e:
            logger.error(f"Failed to load affordances: {e}")

    def _save(self):
        try:
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            data = {entity: [a.to_dict() for a in aff_list] for entity, aff_list in self.affordances.items()}
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save affordances: {e}")

    def add_learned_affordance(self, affordance: Affordance):
        """Adds a new affordance or updates an existing one based on experience."""
        entity = affordance.entity.lower()
        if entity not in self.affordances:
            self.affordances[entity] = []

        # Check if we already have this action for this entity
        existing = next((a for a in self.affordances[entity] if a.action == affordance.action), None)
        if existing:
            existing.observations += max(1, affordance.observations)
            existing.last_tested = time.time()
            existing.confidence = min(1.0, max(existing.confidence, affordance.confidence) + 0.05)
            existing.risk_level = max(0.0, min(1.0, (existing.risk_level + affordance.risk_level) / 2.0))
            for effect in affordance.effects:
                if effect not in existing.effects:
                    existing.effects.append(effect)
            for precondition in affordance.preconditions:
                if precondition not in existing.preconditions:
                    existing.preconditions.append(precondition)
            for tag in affordance.tags:
                if tag not in existing.tags:
                    existing.tags.append(tag)
        else:
            self.affordances[entity].append(affordance)

        self._save()

    def get_affordances(self, entity: str) -> List[Affordance]:
        return self.affordances.get(entity.lower(), [])

    def query(
        self,
        *,
        entities: Iterable[str] = (),
        action: Optional[str] = None,
        tags: Iterable[str] = (),
        min_confidence: float = 0.0,
    ) -> List[Affordance]:
        requested_entities = {str(entity).lower() for entity in entities if str(entity)}
        requested_tags = {str(tag).lower() for tag in tags if str(tag)}
        results: List[Affordance] = []
        for entity, affordances in self.affordances.items():
            if requested_entities and entity not in requested_entities:
                continue
            for affordance in affordances:
                if action and affordance.action != action:
                    continue
                if affordance.confidence < min_confidence:
                    continue
                aff_tags = {tag.lower() for tag in affordance.tags}
                if requested_tags and not requested_tags.intersection(aff_tags):
                    continue
                results.append(affordance)
        return sorted(results, key=lambda a: (a.risk_level, a.confidence), reverse=True)

    def seed_general_doctrine(self) -> None:
        """Seed sparse, domain-general survival affordances.

        These are not NetHack mechanics. They are broad embodied control
        priors: unknown irreversible actions are risky, resources must be
        stabilized, prompts must be handled before normal action, and threats
        close to the body deserve caution.
        """
        seeds = [
            Affordance(
                entity="unknown object",
                action="use",
                preconditions=["identity uncertain"],
                effects=["may produce unknown irreversible effects"],
                risk_level=0.75,
                confidence=0.6,
                source="general_doctrine",
                tags=["uncertainty", "caution"],
            ),
            Affordance(
                entity="critical resource",
                action="stabilize",
                preconditions=["resource below safe threshold"],
                effects=["reduces systemic failure risk"],
                risk_level=0.15,
                confidence=0.75,
                source="general_doctrine",
                tags=["survival", "resource"],
            ),
            Affordance(
                entity="active prompt",
                action="resolve",
                preconditions=["environment awaiting modal input"],
                effects=["restores normal control loop"],
                risk_level=0.25,
                confidence=0.8,
                source="general_doctrine",
                tags=["interface", "modal"],
            ),
            Affordance(
                entity="nearby threat",
                action="engage",
                preconditions=["threat adjacent or close"],
                effects=["may reduce threat but can increase immediate harm"],
                risk_level=0.65,
                confidence=0.65,
                source="general_doctrine",
                tags=["threat", "combat", "caution"],
            ),
        ]
        for seed in seeds:
            self.affordances.setdefault(seed.entity, []).append(seed)
        self._save()

    def get_summary_for_prompt(self, visible_entities: List[str]) -> str:
        """Returns a string summarizing known affordances for visible entities."""
        lines = []
        visible = {ent.lower() for ent in visible_entities if ent}
        visible.update({"unknown object", "critical resource", "active prompt", "nearby threat"})
        for ent in sorted(visible):
            affs = self.get_affordances(ent)
            for a in affs:
                if a.confidence > 0.4:  # Only mention if somewhat confident
                    effects = ", ".join(a.effects)
                    risk = "DANGER" if a.risk_level >= 0.7 else "Safe" if a.risk_level <= 0.3 else "Caution"
                    lines.append(f"  - {ent.capitalize()} ({a.action}): {effects} [{risk}]")

        if lines:
            return "RECALLED KNOWLEDGE:\n" + "\n".join(lines)
        return ""
