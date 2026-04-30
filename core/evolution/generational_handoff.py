"""Controlled generational handoff for long-running Aura instances."""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.runtime.atomic_writer import atomic_write_text


@dataclass(frozen=True)
class HandoffPlan:
    parent_root: str
    child_root: str
    identity_hash: str
    artifacts: tuple[str, ...]
    validation_gates: tuple[str, ...] = ("boot_probe", "behavioral_contracts", "proof_bundle")
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "parent_root": self.parent_root,
            "child_root": self.child_root,
            "identity_hash": self.identity_hash,
            "artifacts": list(self.artifacts),
            "validation_gates": list(self.validation_gates),
            "created_at": self.created_at,
        }


class GenerationalHandoff:
    def __init__(self, parent_root: str | Path = ".") -> None:
        self.parent_root = Path(parent_root).resolve()

    def plan(self, child_root: str | Path, *, artifacts: list[str | Path]) -> HandoffPlan:
        child = Path(child_root).resolve()
        identity_payload = json.dumps([str(a) for a in artifacts], sort_keys=True)
        identity_hash = hashlib.sha256(identity_payload.encode("utf-8")).hexdigest()
        return HandoffPlan(str(self.parent_root), str(child), identity_hash, tuple(str(a) for a in artifacts))

    def create_child_workspace(self, plan: HandoffPlan) -> Path:
        child = Path(plan.child_root)
        child.mkdir(parents=True, exist_ok=True)
        for name in ("core", "interface", "skills", "training", "aura_main.py", "Makefile"):
            src = self.parent_root / name
            dst = child / name
            if src.is_dir():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".venv", "node_modules"))
            elif src.exists():
                shutil.copy2(src, dst)
        atomic_write_text(child / "HANDOFF_PLAN.json", json.dumps(plan.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return child

    def validate_promotion(self, validation: dict[str, Any]) -> bool:
        return all(bool(validation.get(gate)) for gate in ("boot_probe", "behavioral_contracts", "proof_bundle"))


__all__ = ["HandoffPlan", "GenerationalHandoff"]
