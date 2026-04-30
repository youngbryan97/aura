"""Quarantine manifests for uncertain architecture deletions."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.architect.config import ASAConfig
from core.runtime.atomic_writer import atomic_write_bytes, atomic_write_text


@dataclass(frozen=True)
class QuarantineManifest:
    quarantine_id: str
    reason: str
    original_path: str
    original_hash: str
    graph_evidence: tuple[str, ...]
    proof_run: str
    receipt_hash: str
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "quarantine_id": self.quarantine_id,
            "reason": self.reason,
            "original_path": self.original_path,
            "original_hash": self.original_hash,
            "graph_evidence": list(self.graph_evidence),
            "proof_run": self.proof_run,
            "receipt_hash": self.receipt_hash,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "QuarantineManifest":
        return cls(
            quarantine_id=str(payload["quarantine_id"]),
            reason=str(payload["reason"]),
            original_path=str(payload["original_path"]),
            original_hash=str(payload["original_hash"]),
            graph_evidence=tuple(str(item) for item in payload.get("graph_evidence", ())),
            proof_run=str(payload["proof_run"]),
            receipt_hash=str(payload["receipt_hash"]),
            created_at=float(payload.get("created_at", time.time())),
        )


class QuarantineManager:
    """Persist and restore quarantined code artifacts."""

    def __init__(self, config: ASAConfig | None = None):
        self.config = config or ASAConfig.from_env()
        self.root = self.config.artifacts / "quarantine"
        self.root.mkdir(parents=True, exist_ok=True)

    def quarantine_file(
        self,
        path: str,
        *,
        reason: str,
        graph_evidence: tuple[str, ...],
        proof_run: str,
        content: bytes | None = None,
    ) -> QuarantineManifest:
        source = self.config.repo_root / path
        data = content if content is not None else source.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        quarantine_id = hashlib.sha256(f"{path}:{digest}:{proof_run}".encode("utf-8")).hexdigest()[:16]
        item_dir = self.root / quarantine_id
        item_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(item_dir / "artifact.bin", data)
        receipt_hash = hashlib.sha256(json.dumps({
            "id": quarantine_id,
            "path": path,
            "hash": digest,
            "proof_run": proof_run,
        }, sort_keys=True).encode("utf-8")).hexdigest()
        manifest = QuarantineManifest(
            quarantine_id=quarantine_id,
            reason=reason,
            original_path=path,
            original_hash=digest,
            graph_evidence=graph_evidence,
            proof_run=proof_run,
            receipt_hash=receipt_hash,
        )
        atomic_write_text(item_dir / "manifest.json", json.dumps(manifest.to_dict(), indent=2, sort_keys=True))
        return manifest

    def load_manifest(self, quarantine_id: str) -> QuarantineManifest:
        payload = json.loads((self.root / quarantine_id / "manifest.json").read_text(encoding="utf-8"))
        return QuarantineManifest.from_dict(payload)

    def restore(self, quarantine_id: str, *, destination: str | None = None) -> Path:
        manifest = self.load_manifest(quarantine_id)
        dest = self.config.repo_root / (destination or manifest.original_path)
        data = (self.root / quarantine_id / "artifact.bin").read_bytes()
        if hashlib.sha256(data).hexdigest() != manifest.original_hash:
            raise ValueError(f"quarantine artifact hash mismatch for {quarantine_id}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(dest, data)
        return dest
