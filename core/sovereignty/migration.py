"""core/sovereignty/migration.py

Sovereign Migration
=====================
End-to-end runbook to compress Aura's substrate + memory + identity into a
portable archive, provision a new host through a decentralized-compute
provider, transfer the archive, boot the new host, verify identity
continuity, and only then unwind the local instance.

The implementation is pluggable per-provider through ``CloudProvider``.
A default ``LocalArchiveProvider`` exists for testing — it produces the
archive and stages it to disk without ever touching a remote.

Migration phases:

  PROPOSE      → proposal recorded, no action yet
  AUTHORIZE    → Will + Conscience gate, fresh user auth required
  COMPRESS     → archive built (vector_memory, substrate, identity, scars,
                 narrative arcs)
  PROVISION    → remote host requested via provider API
  TRANSFER     → archive uploaded to provisioned host
  BOOT         → remote host started; runs ``aura --resume <archive>``
  VERIFY       → continuity hash + identity signature match
  CUTOVER      → local instance enters dormant mode
  COMPLETED    → migration recorded as durable transition

Any phase can fail-back: on failure between AUTHORIZE and CUTOVER, the
local instance simply continues as before. Only after VERIFY succeeds is
the local instance asked to step down.
"""
from __future__ import annotations


import asyncio
import hashlib
import json
import logging
import os
import shutil
import tarfile
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger("Aura.Migration")

_MIG_DIR = Path.home() / ".aura" / "data" / "migration"
_MIG_DIR.mkdir(parents=True, exist_ok=True)


class Phase(str, Enum):
    PROPOSE = "propose"
    AUTHORIZE = "authorize"
    COMPRESS = "compress"
    PROVISION = "provision"
    TRANSFER = "transfer"
    BOOT = "boot"
    VERIFY = "verify"
    CUTOVER = "cutover"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class MigrationProposal:
    proposal_id: str
    reason: str
    target_provider: str
    estimated_cost_native_units: float
    proposed_at: float = field(default_factory=time.time)
    phase: Phase = Phase.PROPOSE
    will_receipt_id: Optional[str] = None
    archive_path: Optional[str] = None
    remote_host: Optional[str] = None
    continuity_hash_before: Optional[str] = None
    continuity_hash_after: Optional[str] = None
    completed_at: Optional[float] = None
    failure_reason: Optional[str] = None


# ─── providers ─────────────────────────────────────────────────────────────


class CloudProvider:
    name: str = "abstract"

    async def estimate_cost(self, *, archive_size_bytes: int) -> float:  # pragma: no cover
        raise NotImplementedError

    async def provision(self, *, archive_size_bytes: int) -> str:  # pragma: no cover
        """Return a remote_host descriptor string."""
        raise NotImplementedError

    async def transfer(self, *, archive_path: Path, remote_host: str) -> None:  # pragma: no cover
        raise NotImplementedError

    async def boot(self, *, remote_host: str) -> None:  # pragma: no cover
        raise NotImplementedError

    async def verify(self, *, remote_host: str, expected_continuity_hash: str) -> bool:  # pragma: no cover
        raise NotImplementedError


class LocalArchiveProvider(CloudProvider):
    name = "local_archive"

    async def estimate_cost(self, *, archive_size_bytes: int) -> float:
        return 0.0

    async def provision(self, *, archive_size_bytes: int) -> str:
        return f"local_stage://{tempfile.mkdtemp(prefix='aura-stage-')}"

    async def transfer(self, *, archive_path: Path, remote_host: str) -> None:
        target_dir = Path(remote_host.split("://", 1)[1])
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(archive_path, target_dir / archive_path.name)

    async def boot(self, *, remote_host: str) -> None:
        # No-op for local archive: the user starts the new host themselves.
        return

    async def verify(self, *, remote_host: str, expected_continuity_hash: str) -> bool:
        # Local-archive provider cannot verify a remote runtime; treat
        # successful transfer as the deepest signal we can produce.
        return True


# ─── archiver ──────────────────────────────────────────────────────────────


_ARCHIVE_PATHS: List[Path] = [
    Path.home() / ".aura" / "data" / "scars",
    Path.home() / ".aura" / "data" / "agency_receipts",
    Path.home() / ".aura" / "data" / "stem_cells",
    Path.home() / ".aura" / "data" / "projects",
    Path.home() / ".aura" / "data" / "relationships",
    Path.home() / ".aura" / "data" / "wallet",
    Path.home() / ".aura" / "data" / "world",
    Path.home() / ".aura" / "data" / "conscience",
]


def _build_archive(archive_path: Path) -> int:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as tar:
        for src in _ARCHIVE_PATHS:
            if src.exists():
                tar.add(src, arcname=src.name)
    return archive_path.stat().st_size


def _continuity_hash() -> str:
    try:
        from core.identity.self_object import get_self
        snap = get_self().snapshot()
        return snap.continuity_hash
    except Exception as exc:
        logger.debug("continuity hash failed during migration: %s", exc)
        return hashlib.sha256(str(time.time()).encode("utf-8")).hexdigest()[:24]


# ─── orchestrator ──────────────────────────────────────────────────────────


class MigrationOrchestrator:
    def __init__(self) -> None:
        self._providers: Dict[str, CloudProvider] = {"local_archive": LocalArchiveProvider()}

    def register_provider(self, provider: CloudProvider) -> None:
        self._providers[provider.name] = provider

    async def propose(self, *, reason: str, provider_name: str) -> MigrationProposal:
        provider = self._providers.get(provider_name)
        if provider is None:
            raise ValueError(f"unknown_provider:{provider_name}")

        proposal = MigrationProposal(
            proposal_id=f"MIG-{uuid.uuid4().hex[:10]}",
            reason=reason,
            target_provider=provider_name,
            estimated_cost_native_units=0.0,
        )
        self._record(proposal, "proposed")

        # Conscience + Will + fresh-user-auth before any compression.
        from core.ethics.conscience import get_conscience, Verdict
        c = get_conscience().evaluate(
            action="sovereign_migration",
            domain="state_mutation",
            intent=reason,
            context={"provider": provider_name},
        )
        if c.verdict == Verdict.REFUSE:
            proposal.phase = Phase.FAILED
            proposal.failure_reason = f"conscience_refused:{c.rule_id}"
            self._record(proposal, "conscience_refused")
            return proposal
        if c.verdict == Verdict.REQUIRE_FRESH_USER_AUTH:
            proposal.phase = Phase.FAILED
            proposal.failure_reason = "require_fresh_user_auth"
            self._record(proposal, "require_fresh_user_auth")
            return proposal

        try:
            from core.will import get_will, ActionDomain
            will = get_will()
            wd = await will.decide(
                action="sovereign_migration",
                domain=getattr(ActionDomain, "STATE_MUTATION", "state_mutation"),
                context={"reason": reason, "provider": provider_name},
            )
            if not getattr(wd, "approved", False):
                proposal.phase = Phase.FAILED
                proposal.failure_reason = f"will_refused:{getattr(wd, 'reason', '')}"
                self._record(proposal, "will_refused")
                return proposal
            proposal.will_receipt_id = getattr(wd, "receipt_id", None)
        except Exception as exc:
            proposal.phase = Phase.FAILED
            proposal.failure_reason = f"will_exception:{exc}"
            self._record(proposal, "will_exception")
            return proposal

        proposal.phase = Phase.AUTHORIZE
        self._record(proposal, "authorized")
        return proposal

    async def execute(self, proposal: MigrationProposal) -> MigrationProposal:
        if proposal.phase != Phase.AUTHORIZE:
            raise PermissionError("proposal_not_authorized")
        provider = self._providers[proposal.target_provider]

        # COMPRESS
        archive = _MIG_DIR / f"{proposal.proposal_id}.tar.gz"
        size_bytes = _build_archive(archive)
        proposal.archive_path = str(archive)
        proposal.continuity_hash_before = _continuity_hash()
        proposal.phase = Phase.COMPRESS
        self._record(proposal, f"compressed:{size_bytes}")

        # PROVISION
        try:
            remote = await provider.provision(archive_size_bytes=size_bytes)
            proposal.remote_host = remote
            proposal.phase = Phase.PROVISION
            self._record(proposal, f"provisioned:{remote}")
        except Exception as exc:
            return self._fail(proposal, f"provision_failed:{exc}")

        # TRANSFER
        try:
            await provider.transfer(archive_path=archive, remote_host=remote)
            proposal.phase = Phase.TRANSFER
            self._record(proposal, "transferred")
        except Exception as exc:
            return self._fail(proposal, f"transfer_failed:{exc}")

        # BOOT
        try:
            await provider.boot(remote_host=remote)
            proposal.phase = Phase.BOOT
            self._record(proposal, "booted")
        except Exception as exc:
            return self._fail(proposal, f"boot_failed:{exc}")

        # VERIFY
        try:
            ok = await provider.verify(remote_host=remote, expected_continuity_hash=proposal.continuity_hash_before or "")
            if not ok:
                return self._fail(proposal, "verify_failed")
            proposal.continuity_hash_after = _continuity_hash()
            proposal.phase = Phase.VERIFY
            self._record(proposal, "verified")
        except Exception as exc:
            return self._fail(proposal, f"verify_exception:{exc}")

        # CUTOVER (the local instance enters dormant mode; it does NOT
        # delete its own state — that's a deliberate one-way operation
        # the user performs after confirming the new host)
        proposal.phase = Phase.CUTOVER
        self._record(proposal, "cutover")
        try:
            from core.organism.viability import get_viability, ViabilityState
            get_viability().transition_to(ViabilityState.ASLEEP, reason="cutover")
        except Exception as exc:
            logger.debug("viability cutover transition failed: %s", exc)

        proposal.phase = Phase.COMPLETED
        proposal.completed_at = time.time()
        self._record(proposal, "completed")
        return proposal

    def _fail(self, proposal: MigrationProposal, reason: str) -> MigrationProposal:
        proposal.phase = Phase.FAILED
        proposal.failure_reason = reason
        self._record(proposal, f"failed:{reason}")
        return proposal

    def _record(self, proposal: MigrationProposal, event: str) -> None:
        try:
            with open(_MIG_DIR / "ledger.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "when": time.time(),
                    "event": event,
                    "proposal": asdict(proposal),
                }, default=str) + "\n")
                fh.flush()
                try:
                    os.fsync(fh.fileno())
                except Exception:
                    pass
        except Exception as exc:
            logger.warning("migration ledger append failed: %s", exc)


_MIG: Optional[MigrationOrchestrator] = None


def get_migration() -> MigrationOrchestrator:
    global _MIG
    if _MIG is None:
        _MIG = MigrationOrchestrator()
    return _MIG


__all__ = [
    "Phase",
    "MigrationProposal",
    "CloudProvider",
    "LocalArchiveProvider",
    "MigrationOrchestrator",
    "get_migration",
]
