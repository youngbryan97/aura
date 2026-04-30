"""Native acceleration planner for bottlenecked cognitive kernels.

Builds and validates C++ extension compile commands without letting Aura mutate
sealed math or bus code.  Actual compilation is explicit and sandboxable.
"""
from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.self_modification.mutation_tiers import MutationTier, classify_mutation_path


@dataclass(frozen=True)
class NativeCompilePlan:
    source_path: str
    output_path: str
    command: tuple[str, ...]
    source_hash: str
    allowed: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "output_path": self.output_path,
            "command": list(self.command),
            "source_hash": self.source_hash,
            "allowed": self.allowed,
            "reason": self.reason,
        }


class NativeCompilerPlanner:
    def plan_cpp(self, source_path: str | Path, output_path: str | Path) -> NativeCompilePlan:
        src = Path(source_path)
        out = Path(output_path)
        tier = classify_mutation_path(src.as_posix())
        allowed = tier.tier not in {MutationTier.SEALED, MutationTier.PROPOSE_ONLY}
        source = src.read_text(encoding="utf-8") if src.exists() else ""
        digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
        cmd = ("clang++", "-O3", "-shared", "-std=c++20", "-fPIC", str(src), "-o", str(out))
        return NativeCompilePlan(str(src), str(out), cmd, digest, allowed, tier.reason)

    def compile(self, plan: NativeCompilePlan, *, timeout: float = 60.0) -> subprocess.CompletedProcess[str]:
        if not plan.allowed:
            raise PermissionError(plan.reason)
        return subprocess.run(list(plan.command), capture_output=True, text=True, timeout=timeout, check=False)


__all__ = ["NativeCompilePlan", "NativeCompilerPlanner"]
