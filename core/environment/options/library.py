"""Option library with cooldown and failure suppression."""
from __future__ import annotations

from dataclasses import dataclass, field

from .base import Option, OptionRun
from .builtins import builtin_options


@dataclass
class OptionStats:
    success_count: int = 0
    failure_count: int = 0
    cooldown_until_seq: int = 0
    runs: list[OptionRun] = field(default_factory=list)


class OptionLibrary:
    def __init__(self, options: dict[str, Option] | None = None) -> None:
        self.options = options or builtin_options()
        self.stats: dict[str, OptionStats] = {name: OptionStats() for name in self.options}

    def get(self, name: str) -> Option:
        return self.options[name]

    def available(self, *, seq: int) -> list[Option]:
        return [opt for name, opt in self.options.items() if self.stats[name].cooldown_until_seq <= seq]

    def record_run(self, run: OptionRun, *, seq: int) -> None:
        stats = self.stats.setdefault(run.option_name, OptionStats())
        stats.runs.append(run)
        stats.runs = stats.runs[-100:]
        if run.status == "succeeded":
            stats.success_count += 1
        elif run.status == "failed":
            stats.failure_count += 1
            option = self.options.get(run.option_name)
            if option:
                stats.cooldown_until_seq = max(stats.cooldown_until_seq, seq + option.cooldown_steps)

    def suppressed_after_repeated_failure(self, name: str, *, threshold: int = 2) -> bool:
        stats = self.stats.get(name)
        if not stats:
            return False
        recent = stats.runs[-threshold:]
        return len(recent) == threshold and all(run.status == "failed" for run in recent)


__all__ = ["OptionStats", "OptionLibrary", "Option", "OptionRun"]
