# core/agency/safety_registry.py
import asyncio
from dataclasses import dataclass, field
from typing import Set, Optional

@dataclass
class SafetyRegistry:
    disabled_skills: Set[str] = field(default_factory=set)
    autonomy_level: float = 0.7  # 0.0 = fully supervised, 1.0 = full autonomy
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def disable_skill(self, skill_name: str):
        async with self._lock:
            self.disabled_skills.add(skill_name)
    
    async def is_allowed(self, skill_name: str) -> bool:
        async with self._lock:
            # Audit Fix 1.2: Check both revocation list and global autonomy level
            return skill_name not in self.disabled_skills and self.autonomy_level > 0.3

# Singleton
_safety: Optional[SafetyRegistry] = None

def get_safety_registry() -> SafetyRegistry:
    global _safety
    if _safety is None:
        _safety = SafetyRegistry()
    return _safety
