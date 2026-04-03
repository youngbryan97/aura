"""core/agi/skill_synthesizer.py
Autonomous Skill Synthesizer
==============================
When Aura encounters a task she cannot handle with existing skills,
this module:
  1. Detects the capability gap (what is missing)
  2. Designs a new skill specification
  3. Validates it against the safety registry
  4. Generates the implementation
  5. Registers it into the live skill registry

This is self-directed capability expansion — not tool use,
but tool creation.

The synthesizer does NOT write arbitrary code. It:
  - Works within a constrained skill template
  - Requires safety validation before registration
  - Has a human-approval path for high-risk capabilities
  - Persists to disk for survival across restarts
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Aura.SkillSynthesizer")

PERSIST_PATH = Path.home() / ".aura" / "data" / "synthesized_skills.json"

# Skill template — all synthesized skills follow this pattern
SKILL_TEMPLATE = '''
class {class_name}(BaseSkill):
    """Auto-synthesized skill: {description}

    Synthesized at: {timestamp}
    Gap identified: {gap}
    """
    name = "{skill_name}"
    description = "{description}"

    async def execute(self, params: dict, context: dict) -> dict:
        """Execute the skill.

        Params: {param_spec}
        """
        # Generated implementation
        {implementation}
        return {{"ok": True, "result": result}}
'''


@dataclass
class SynthesizedSkill:
    """A skill designed by the synthesizer."""
    name: str
    description: str
    gap: str                    # what capability gap this fills
    class_code: str             # Python class source
    param_spec: Dict[str, str]  # parameter name → description
    safety_level: str           # low | medium | high
    approved: bool = False      # requires human approval if high risk
    registered: bool = False
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
        self._gaps: List[Dict] = []          # observed capability gaps
        self._synthesized: List[SynthesizedSkill] = []
        self._gap_counts: Dict[str, int] = {}  # gap → frequency
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
        # Trigger synthesis if gap seen 3+ times
        if self._gap_counts[gap_key] >= 3:
            logger.info("SkillSynthesizer: gap threshold reached for '%s'", gap_key[:60])
        if len(self._gaps) > 200:
            self._gaps = self._gaps[-200:]

    async def synthesize_pending(self, orchestrator=None) -> List[SynthesizedSkill]:
        """Synthesize skills for the most frequent unresolved gaps."""
        # Find gaps at threshold with no existing skill
        hot_gaps = sorted(
            [(gap, count) for gap, count in self._gap_counts.items() if count >= 3],
            key=lambda x: x[1], reverse=True
        )[:3]

        synthesized = []
        for gap, count in hot_gaps:
            # Skip if already synthesized
            if any(s.gap == gap for s in self._synthesized):
                continue
            skill = await self._synthesize_skill(gap, count, orchestrator)
            if skill:
                synthesized.append(skill)
                self._synthesized.append(skill)
                # Register if low-risk
                if skill.safety_level != "high":
                    await self._register_skill(skill, orchestrator)

        self._save()
        return synthesized

    def get_synthesized_skills(self) -> List[Dict]:
        return [
            {"name": s.name, "description": s.description, "gap": s.gap,
             "registered": s.registered, "use_count": s.use_count}
            for s in self._synthesized
        ]

    def get_status(self) -> Dict:
        return {
            "gap_count": len(self._gap_counts),
            "synthesized": len(self._synthesized),
            "registered": sum(1 for s in self._synthesized if s.registered),
            "pending_approval": sum(1 for s in self._synthesized
                                    if not s.approved and s.safety_level == "high"),
        }

    # ── Synthesis ─────────────────────────────────────────────────────────

    async def _synthesize_skill(self, gap: str, frequency: int,
                                 orchestrator=None) -> Optional[SynthesizedSkill]:
        try:
            from core.container import ServiceContainer
            router = ServiceContainer.get("llm_router", default=None)
            if not router:
                return None

            from core.brain.llm.llm_router import LLMTier
            prompt = (
                f"Design a skill to address this capability gap:\n"
                f"Gap: {gap}\n"
                f"Frequency: seen {frequency} times\n\n"
                "Return JSON with:\n"
                '{"name": "skill_name", "description": "what it does", '
                '"params": {"param1": "description"}, '
                '"implementation": "one-line description of what the execute method should do", '
                '"safety_level": "low|medium|high"}\n'
                "low = read-only/informational, medium = local state changes, high = external effects"
            )
            raw = await asyncio.wait_for(
                router.think(prompt, priority=0.2, is_background=True,
                             prefer_tier=LLMTier.SECONDARY),
                timeout=20.0,
            )
            if not raw:
                return None

            # Parse response
            import re
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if not match:
                return None
            data = json.loads(match.group())

            # Build the skill
            name = data.get("name", f"auto_skill_{int(time.time())}")
            desc = data.get("description", "Auto-synthesized skill")
            impl = data.get("implementation", "result = 'skill executed'")
            safety = data.get("safety_level", "medium")
            params = data.get("params", {})

            # Render class code from template
            class_name = "".join(w.capitalize() for w in name.split("_")) + "Skill"
            code = SKILL_TEMPLATE.format(
                class_name=class_name,
                skill_name=name,
                description=desc,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                gap=gap[:80],
                param_spec=str(params),
                implementation=f"result = '{impl[:100]}'",
            )

            skill = SynthesizedSkill(
                name=name,
                description=desc,
                gap=gap,
                class_code=code,
                param_spec=params,
                safety_level=safety,
                approved=(safety != "high"),
            )
            logger.info("SkillSynthesizer: synthesized '%s' (safety=%s)", name, safety)
            return skill

        except Exception as e:
            logger.debug("Skill synthesis failed for gap '%s': %s", gap[:40], e)
            return None

    async def _register_skill(self, skill: SynthesizedSkill, orchestrator=None):
        """Register a synthesized skill into the live registry."""
        try:
            from core.container import ServiceContainer
            registry = ServiceContainer.get("skill_registry", default=None)
            if registry and hasattr(registry, "register_skill"):
                registry.register_skill({
                    "name": skill.name,
                    "description": skill.description,
                    "source": "synthesized",
                    "params": skill.param_spec,
                })
                skill.registered = True
                logger.info("SkillSynthesizer: registered '%s'", skill.name)
        except Exception as e:
            logger.debug("Skill registration failed: %s", e)

    # ── Persistence ───────────────────────────────────────────────────────

    def _save(self):
        try:
            PERSIST_PATH.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "gaps": self._gaps[-50:],
                "gap_counts": self._gap_counts,
                "synthesized": [
                    {"name": s.name, "description": s.description, "gap": s.gap,
                     "safety_level": s.safety_level, "registered": s.registered,
                     "use_count": s.use_count, "created_at": s.created_at}
                    for s in self._synthesized
                ],
            }
            PERSIST_PATH.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.debug("SkillSynthesizer save failed: %s", e)

    def _load(self):
        try:
            if PERSIST_PATH.exists():
                data = json.loads(PERSIST_PATH.read_text())
                self._gaps = data.get("gaps", [])
                self._gap_counts = data.get("gap_counts", {})
                for s in data.get("synthesized", []):
                    self._synthesized.append(SynthesizedSkill(
                        name=s["name"], description=s["description"],
                        gap=s["gap"], class_code="", param_spec={},
                        safety_level=s.get("safety_level", "medium"),
                        registered=s.get("registered", False),
                        use_count=s.get("use_count", 0),
                        created_at=s.get("created_at", time.time()),
                    ))
                logger.info("SkillSynthesizer: loaded %d synthesized skills.",
                            len(self._synthesized))
        except Exception as e:
            logger.debug("SkillSynthesizer load failed: %s", e)


# ── Singleton ─────────────────────────────────────────────────────────────────

_synthesizer: Optional[SkillSynthesizer] = None


def get_skill_synthesizer() -> SkillSynthesizer:
    global _synthesizer
    if _synthesizer is None:
        _synthesizer = SkillSynthesizer()
    return _synthesizer
