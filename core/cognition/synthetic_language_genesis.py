"""Synthetic latent token registry for recurring internal concepts."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.runtime.atomic_writer import atomic_write_text


@dataclass(frozen=True)
class SyntheticToken:
    token: str
    description: str
    vector_hash: str
    evidence_count: int
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "token": self.token,
            "description": self.description,
            "vector_hash": self.vector_hash,
            "evidence_count": self.evidence_count,
            "created_at": self.created_at,
        }


class SyntheticLanguageRegistry:
    def __init__(self, path: str | Path = Path.home() / ".aura" / "data" / "cognition" / "synthetic_tokens.json") -> None:
        self.path = Path(path)
        self.tokens: dict[str, SyntheticToken] = {}
        self._load()

    def propose(self, description: str, *, evidence_count: int, vector: list[float]) -> SyntheticToken:
        if evidence_count < 3:
            raise ValueError("synthetic tokens require at least 3 recurring examples")
        digest = hashlib.sha256(json.dumps(vector, sort_keys=True).encode("utf-8")).hexdigest()
        token = "aura_" + hashlib.sha256(f"{description}|{digest}".encode("utf-8")).hexdigest()[:10]
        rec = SyntheticToken(token, description, digest, evidence_count)
        self.tokens[token] = rec
        self._save()
        return rec

    def decode(self, token: str) -> str:
        rec = self.tokens.get(token)
        return rec.description if rec else ""

    def _load(self) -> None:
        if not self.path.exists():
            return
        data = json.loads(self.path.read_text(encoding="utf-8"))
        for item in data.get("tokens", []):
            rec = SyntheticToken(**item)
            self.tokens[rec.token] = rec

    def _save(self) -> None:
        atomic_write_text(
            self.path,
            json.dumps({"tokens": [t.to_dict() for t in self.tokens.values()]}, indent=2, sort_keys=True),
            encoding="utf-8",
        )


__all__ = ["SyntheticToken", "SyntheticLanguageRegistry"]
