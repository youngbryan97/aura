"""Strategic memory tier for architecture-level improvement pressure."""
from dataclasses import dataclass, field


@dataclass
class StrategicMemoryTier:
    weaknesses: list[str] = field(default_factory=list)
    priorities: list[str] = field(default_factory=list)
    regressions: list[str] = field(default_factory=list)
