"""Survive a dead machine, not just a closed terminal (CP237).

The detached supervisor already survives this session and sleep -- a
suspended process resumes on wake. What it does NOT survive is the machine
losing power or the process being jetsam-killed under memory pressure,
which is exactly how the CP227 accuracy gate died mid-run. When that
happens, an 8-hour run that has completed 6 hours has lost 6 hours.

This module makes a run RESUMABLE from disk. Everything needed to continue
-- step cursor, RNG state, optimizer state, the trained adapter, and the
curriculum's learned difficulty map -- is written atomically at a cadence,
so a resumed process picks up from the last checkpoint rather than the
start.

Atomicity is the whole point: a checkpoint half-written when the power
fails must not be loadable as if complete. Each checkpoint is written to a
temp path and renamed (atomic on POSIX), and a manifest is updated last, so
a torn write leaves the PREVIOUS good checkpoint as the latest valid one.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.governance_context import local_internal_governed_scope
from core.runtime.file_write_gateway import get_file_write_gateway

logger = logging.getLogger(__name__)

DURABLE_RUN_SCHEMA = "aura.durable_run.v1"
MANIFEST = "checkpoint_manifest.json"


@dataclass
class Checkpoint:
    """One resumable point in a run."""

    step: int
    payload: dict[str, Any]
    created_unix: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DURABLE_RUN_SCHEMA,
            "step": self.step,
            "created_unix": self.created_unix,
            "payload": self.payload,
        }


class DurableRun:
    """Atomic checkpoint/resume for a long training or eval run."""

    def __init__(
        self,
        run_dir: str | Path,
        *,
        keep: int = 3,
    ) -> None:
        if type(keep) is not int or keep < 1:
            raise ValueError("keep must be a positive integer")
        self.run_dir = Path(run_dir)
        with local_internal_governed_scope(
            "durable_run.checkpoint_directory",
            domain="file_write",
        ):
            get_file_write_gateway().ensure_directory(
                self.run_dir,
                source="durable_run.checkpoint_directory",
            )
        self.keep = keep

    def _manifest_path(self) -> Path:
        return self.run_dir / MANIFEST

    def save(self, step: int, payload: dict[str, Any]) -> Path:
        """Write a checkpoint atomically and update the manifest LAST.

        Order matters: the checkpoint file is fully on disk and renamed
        before the manifest points at it, so a crash between the two leaves
        the manifest naming the previous good checkpoint, never a partial.
        """
        if type(step) is not int or step < 0:
            raise ValueError("step must be a non-negative integer")
        checkpoint = Checkpoint(step=step, payload=payload, created_unix=time.time())
        name = f"checkpoint_{step:08d}.json"
        final = self.run_dir / name
        with local_internal_governed_scope(
            "durable_run.checkpoint",
            domain="file_write",
        ):
            get_file_write_gateway().write_text(
                final,
                json.dumps(checkpoint.to_dict(), indent=2),
                source="durable_run.checkpoint",
            )

        manifest = self._read_manifest()
        manifest["latest"] = name
        manifest["history"] = ([name] + manifest.get("history", []))[: self.keep + 5]
        manifest["updated_unix"] = time.time()
        with local_internal_governed_scope(
            "durable_run.manifest",
            domain="file_write",
        ):
            get_file_write_gateway().write_text(
                self._manifest_path(),
                json.dumps(manifest, indent=2),
                source="durable_run.manifest",
            )

        self._prune()
        return final

    def _read_manifest(self) -> dict[str, Any]:
        path = self._manifest_path()
        if not path.exists():
            return {"schema": DURABLE_RUN_SCHEMA, "latest": None, "history": []}
        try:
            return json.loads(path.read_text())
        except (ValueError, OSError):
            # A torn manifest must not crash resume; fall back to scanning.
            return {"schema": DURABLE_RUN_SCHEMA, "latest": None, "history": []}

    def latest(self) -> Checkpoint | None:
        """Load the newest VALID checkpoint, or None for a fresh run.

        Prefers the manifest's pointer but verifies it loads; if the named
        file is missing or torn, scans for the highest-step file that parses.
        A run must resume from real state, never from a filename that
        promises state that is not there.
        """
        manifest = self._read_manifest()
        candidates: list[str] = []
        if manifest.get("latest"):
            candidates.append(manifest["latest"])
        candidates += sorted(
            (p.name for p in self.run_dir.glob("checkpoint_[0-9]*.json")),
            reverse=True,
        )
        for name in candidates:
            checkpoint = self._load(self.run_dir / name)
            if checkpoint is not None:
                return checkpoint
        return None

    def _load(self, path: Path) -> Checkpoint | None:
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            if data.get("schema") != DURABLE_RUN_SCHEMA:
                return None
            return Checkpoint(
                step=int(data["step"]),
                payload=data["payload"],
                created_unix=float(data["created_unix"]),
            )
        # not a failure: a reading that cannot be taken, or that is not a
        # number when it is, leaves this at the value below.
        except (ValueError, OSError, KeyError):
            return None

    def _prune(self) -> None:
        checkpoints = sorted(
            self.run_dir.glob("checkpoint_[0-9]*.json"),
            key=lambda p: p.name,
            reverse=True,
        )
        for stale in checkpoints[self.keep :]:
            try:
                with local_internal_governed_scope(
                    "durable_run.prune",
                    domain="file_write",
                ):
                    get_file_write_gateway().delete_file(
                        stale,
                        source="durable_run.prune",
                    )
            except OSError as exc:
                logger.debug(
                    "%s unavailable (%s: %s); a stale run directory was not pruned",
                    "it",
                    type(exc).__name__,
                    exc,
                )

    def resume_step(self) -> int:
        """Where to continue from -- 0 for a fresh run."""
        checkpoint = self.latest()
        return checkpoint.step if checkpoint else 0


__all__ = ["DURABLE_RUN_SCHEMA", "Checkpoint", "DurableRun"]
