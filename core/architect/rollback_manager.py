"""Rollback packet creation, dry-run verification, and restoration."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from core.architect.config import ASAConfig
from core.architect.errors import RollbackError
from core.architect.models import RefactorPlan, RollbackPacket
from core.architect.shadow_workspace import ShadowRun
from core.runtime.atomic_writer import atomic_write_bytes, atomic_write_text


class RollbackManager:
    """Create immutable rollback packets before promotion."""

    def __init__(self, config: ASAConfig | None = None):
        self.config = config or ASAConfig.from_env()
        self.root = self.config.artifacts / "rollback"
        self.root.mkdir(parents=True, exist_ok=True)

    def create_packet(self, plan: RefactorPlan, shadow: ShadowRun) -> RollbackPacket:
        changed = tuple(dict.fromkeys(plan.changed_files or shadow.changed_files))
        if not changed:
            raise RollbackError("cannot create rollback packet without changed files")
        packet_dir = self.root / shadow.run_id
        original_dir = packet_dir / "original"
        candidate_dir = packet_dir / "candidate"
        original_hashes: dict[str, str] = {}
        candidate_hashes: dict[str, str] = {}
        for rel in changed:
            live = self.config.repo_root / rel
            candidate = Path(shadow.candidate_files.get(rel, "")) if rel in shadow.candidate_files else Path(shadow.shadow_root) / rel
            if not live.exists():
                raise RollbackError(f"live file missing before rollback packet: {rel}")
            if not candidate.exists():
                raise RollbackError(f"candidate file missing before rollback packet: {rel}")
            live_bytes = live.read_bytes()
            candidate_bytes = candidate.read_bytes()
            original_hashes[rel] = hashlib.sha256(live_bytes).hexdigest()
            candidate_hashes[rel] = hashlib.sha256(candidate_bytes).hexdigest()
            original_target = original_dir / rel
            candidate_target = candidate_dir / rel
            original_target.parent.mkdir(parents=True, exist_ok=True)
            candidate_target.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_bytes(original_target, live_bytes)
            atomic_write_bytes(candidate_target, candidate_bytes)
        repo_hash = self._repo_hash(changed)
        receipt_hash = hashlib.sha256(json.dumps({
            "run_id": shadow.run_id,
            "repo_hash": repo_hash,
            "original_hashes": original_hashes,
            "candidate_hashes": candidate_hashes,
        }, sort_keys=True).encode("utf-8")).hexdigest()
        packet = RollbackPacket(
            run_id=shadow.run_id,
            timestamp=time.time(),
            repo_root_hash=repo_hash,
            changed_files=changed,
            original_hashes=original_hashes,
            candidate_hashes=candidate_hashes,
            packet_path=str(packet_dir),
            receipt_hash=receipt_hash,
            dry_run_passed=False,
            post_restore_verified=False,
        )
        atomic_write_text(packet_dir / "packet.json", json.dumps(packet_to_dict(packet), indent=2, sort_keys=True, default=str))
        return packet

    def dry_run(self, packet: RollbackPacket) -> RollbackPacket:
        packet_dir = Path(packet.packet_path)
        failures: list[str] = []
        for rel, expected in packet.original_hashes.items():
            live = self.config.repo_root / rel
            saved = packet_dir / "original" / rel
            if not live.exists() or not saved.exists():
                failures.append(rel)
                continue
            if hashlib.sha256(saved.read_bytes()).hexdigest() != expected:
                failures.append(f"{rel}:saved_hash")
            if hashlib.sha256(live.read_bytes()).hexdigest() != expected:
                failures.append(f"{rel}:live_changed")
        if failures:
            raise RollbackError(f"rollback dry-run failed: {failures[:5]}")
        verified = RollbackPacket(
            run_id=packet.run_id,
            timestamp=packet.timestamp,
            repo_root_hash=packet.repo_root_hash,
            changed_files=packet.changed_files,
            original_hashes=packet.original_hashes,
            candidate_hashes=packet.candidate_hashes,
            packet_path=packet.packet_path,
            receipt_hash=packet.receipt_hash,
            dry_run_passed=True,
            post_restore_verified=packet.post_restore_verified,
        )
        atomic_write_text(Path(packet.packet_path) / "packet.json", json.dumps(packet_to_dict(verified), indent=2, sort_keys=True, default=str))
        return verified

    def restore(self, run_id: str | RollbackPacket) -> RollbackPacket:
        packet = run_id if isinstance(run_id, RollbackPacket) else self.load_packet(run_id)
        packet_dir = Path(packet.packet_path)
        for rel, expected in packet.original_hashes.items():
            src = packet_dir / "original" / rel
            if not src.exists():
                raise RollbackError(f"rollback original missing: {rel}")
            data = src.read_bytes()
            if hashlib.sha256(data).hexdigest() != expected:
                raise RollbackError(f"rollback original hash mismatch: {rel}")
            dest = self.config.repo_root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_bytes(dest, data)
        failures: list[str] = []
        for rel, expected in packet.original_hashes.items():
            live = self.config.repo_root / rel
            if hashlib.sha256(live.read_bytes()).hexdigest() != expected:
                failures.append(rel)
        if failures:
            raise RollbackError(f"post-restore verification failed: {failures[:5]}")
        restored = RollbackPacket(
            run_id=packet.run_id,
            timestamp=packet.timestamp,
            repo_root_hash=packet.repo_root_hash,
            changed_files=packet.changed_files,
            original_hashes=packet.original_hashes,
            candidate_hashes=packet.candidate_hashes,
            packet_path=packet.packet_path,
            receipt_hash=packet.receipt_hash,
            dry_run_passed=packet.dry_run_passed,
            post_restore_verified=True,
        )
        atomic_write_text(packet_dir / "packet.json", json.dumps(packet_to_dict(restored), indent=2, sort_keys=True, default=str))
        return restored

    def load_packet(self, run_id: str) -> RollbackPacket:
        payload = json.loads((self.root / run_id / "packet.json").read_text(encoding="utf-8"))
        return packet_from_dict(payload)

    def _repo_hash(self, changed: tuple[str, ...]) -> str:
        hashes = []
        for rel in sorted(changed):
            path = self.config.repo_root / rel
            hashes.append(f"{rel}:{hashlib.sha256(path.read_bytes()).hexdigest()}")
        return hashlib.sha256("|".join(hashes).encode("utf-8")).hexdigest()


def packet_to_dict(packet: RollbackPacket) -> dict[str, object]:
    return {
        "run_id": packet.run_id,
        "timestamp": packet.timestamp,
        "repo_root_hash": packet.repo_root_hash,
        "changed_files": list(packet.changed_files),
        "original_hashes": packet.original_hashes,
        "candidate_hashes": packet.candidate_hashes,
        "packet_path": packet.packet_path,
        "receipt_hash": packet.receipt_hash,
        "dry_run_passed": packet.dry_run_passed,
        "post_restore_verified": packet.post_restore_verified,
    }


def packet_from_dict(payload: dict[str, object]) -> RollbackPacket:
    return RollbackPacket(
        run_id=str(payload["run_id"]),
        timestamp=float(payload["timestamp"]),
        repo_root_hash=str(payload["repo_root_hash"]),
        changed_files=tuple(str(item) for item in payload.get("changed_files", ())),
        original_hashes={str(key): str(value) for key, value in dict(payload.get("original_hashes", {})).items()},
        candidate_hashes={str(key): str(value) for key, value in dict(payload.get("candidate_hashes", {})).items()},
        packet_path=str(payload["packet_path"]),
        receipt_hash=str(payload.get("receipt_hash", "")),
        dry_run_passed=bool(payload.get("dry_run_passed", False)),
        post_restore_verified=bool(payload.get("post_restore_verified", False)),
    )
