"""Proof-pack and benchmark schemas."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EnvironmentBenchmarkResult:
    environment_id: str
    metrics: dict[str, float]
    trace_path: str = ""
    ablation_results: dict[str, dict[str, float]] = field(default_factory=dict)
    passed: bool = False


@dataclass
class ProofPack:
    proof_pack: str
    environments: list[dict[str, Any]]
    shared_requirements: dict[str, Any]

    @staticmethod
    def load(path: str | Path) -> "ProofPack":
        text = Path(path).read_text(encoding="utf-8")
        try:
            import yaml  # type: ignore

            data = yaml.safe_load(text)
        except Exception:
            data = _tiny_yaml(text)
        return ProofPack(
            proof_pack=str(data["proof_pack"]),
            environments=list(data.get("environments", [])),
            shared_requirements=dict(data.get("shared_requirements", {})),
        )


def _tiny_yaml(text: str) -> dict[str, Any]:
    """A tiny parser for the proof-pack fixture shape when PyYAML is absent."""
    data: dict[str, Any] = {"environments": [], "shared_requirements": {}}
    current: dict[str, Any] | None = None
    section = ""
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if line.startswith("proof_pack:"):
            data["proof_pack"] = line.split(":", 1)[1].strip()
        elif line.startswith("environments:"):
            section = "environments"
        elif line.startswith("shared_requirements:"):
            section = "shared_requirements"
        elif section == "environments" and line.strip().startswith("- "):
            current = {}
            data["environments"].append(current)
            body = line.strip()[2:]
            if ":" in body:
                k, v = body.split(":", 1)
                current[k.strip()] = v.strip().strip('"')
        elif section == "environments" and current is not None and ":" in line:
            k, v = line.strip().split(":", 1)
            current[k.strip()] = v.strip().strip('"')
        elif section == "shared_requirements" and ":" in line:
            k, v = line.strip().split(":", 1)
            data["shared_requirements"][k.strip()] = v.strip().lower() in {"true", "yes", "1"}
    return data


def write_result(path: str | Path, result: EnvironmentBenchmarkResult) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(result.__dict__, indent=2, sort_keys=True), encoding="utf-8")


__all__ = ["EnvironmentBenchmarkResult", "ProofPack", "write_result"]
