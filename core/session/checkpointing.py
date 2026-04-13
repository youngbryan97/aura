"""
Conversation Checkpointing — Ported from gemini-cli checkpoint feature

Serializes full conversation state (messages, tool results, compression state,
file records) to JSON. Enables restore-on-boot for session continuity across
reboots and memory purges.

Features:
  - Auto-checkpoint every N turns (configurable)
  - Manual checkpoint via API/skill
  - Restore from latest or specific checkpoint
  - Rotational cleanup (keep last 10 checkpoints)
"""

import glob
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Aura.Checkpointing")

# ── Configuration ────────────────────────────────────────────────────────────

DEFAULT_CHECKPOINT_DIR = os.path.expanduser("~/.aura_snapshots")
AUTO_CHECKPOINT_INTERVAL = 10    # Auto-save every N turns
MAX_CHECKPOINTS_KEPT = 10        # Rotational cleanup


@dataclass
class CheckpointData:
    """Serializable conversation checkpoint."""
    timestamp: float = 0.0
    turn_count: int = 0
    model_name: str = ""
    messages: List[Dict[str, str]] = field(default_factory=list)
    system_prompt: str = ""
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    compression_state: Dict[str, Any] = field(default_factory=dict)
    file_records: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class CheckpointService:
    """Manages conversation state persistence across sessions.

    Provides save/restore lifecycle for the full conversation context,
    enabling stable recovery from reboots, crashes, and memory pressure events.
    """

    def __init__(self, checkpoint_dir: str = None):
        self._dir = checkpoint_dir or DEFAULT_CHECKPOINT_DIR
        os.makedirs(self._dir, exist_ok=True)
        self._turn_count = 0
        self._auto_interval = AUTO_CHECKPOINT_INTERVAL
        self._last_checkpoint_turn = 0

    # ── Save ─────────────────────────────────────────────────────────────

    def save(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str = "",
        model_name: str = "",
        tool_results: List[Dict[str, Any]] = None,
        compression_state: Dict[str, Any] = None,
        file_records: Dict[str, Any] = None,
        metadata: Dict[str, Any] = None,
        label: str = "",
    ) -> str:
        """Save a checkpoint to disk.

        Returns: path to the saved checkpoint file.
        """
        checkpoint = CheckpointData(
            timestamp=time.time(),
            turn_count=self._turn_count,
            model_name=model_name,
            messages=messages,
            system_prompt=system_prompt,
            tool_results=tool_results or [],
            compression_state=compression_state or {},
            file_records=file_records or {},
            metadata=metadata or {},
        )

        # Generate filename
        ts = int(checkpoint.timestamp)
        label_part = f"_{label}" if label else ""
        filename = f"checkpoint_{ts}_turn{self._turn_count}{label_part}.json"
        filepath = os.path.join(self._dir, filename)

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(asdict(checkpoint), f, indent=2, default=str)

            self._last_checkpoint_turn = self._turn_count
            logger.info("Checkpoint saved: %s (%d messages, turn %d)",
                       filename, len(messages), self._turn_count)

            # Rotational cleanup
            self._cleanup_old_checkpoints()

            return filepath
        except Exception as e:
            logger.error("Checkpoint save failed: %s", e)
            return ""

    def _cleanup_old_checkpoints(self):
        """Keep only the most recent N checkpoints."""
        pattern = os.path.join(self._dir, "checkpoint_*.json")
        files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
        for old_file in files[MAX_CHECKPOINTS_KEPT:]:
            try:
                os.remove(old_file)
                logger.debug("Removed old checkpoint: %s", os.path.basename(old_file))
            except Exception as e:
                logger.warning("Failed to remove old checkpoint: %s", e)

    # ── Restore ──────────────────────────────────────────────────────────

    def restore_latest(self) -> Optional[CheckpointData]:
        """Restore the most recent checkpoint."""
        pattern = os.path.join(self._dir, "checkpoint_*.json")
        files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)

        if not files:
            logger.info("No checkpoints found in %s", self._dir)
            return None

        return self._load_checkpoint(files[0])

    def restore_by_label(self, label: str) -> Optional[CheckpointData]:
        """Restore a checkpoint by label."""
        pattern = os.path.join(self._dir, f"checkpoint_*_{label}.json")
        files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)

        if not files:
            logger.info("No checkpoint found with label '%s'", label)
            return None

        return self._load_checkpoint(files[0])

    def _load_checkpoint(self, filepath: str) -> Optional[CheckpointData]:
        """Load a checkpoint from disk."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            checkpoint = CheckpointData(
                timestamp=data.get("timestamp", 0),
                turn_count=data.get("turn_count", 0),
                model_name=data.get("model_name", ""),
                messages=data.get("messages", []),
                system_prompt=data.get("system_prompt", ""),
                tool_results=data.get("tool_results", []),
                compression_state=data.get("compression_state", {}),
                file_records=data.get("file_records", {}),
                metadata=data.get("metadata", {}),
            )

            self._turn_count = checkpoint.turn_count
            self._last_checkpoint_turn = checkpoint.turn_count

            logger.info(
                "Checkpoint restored: %s (turn %d, %d messages, %.0fs ago)",
                os.path.basename(filepath),
                checkpoint.turn_count,
                len(checkpoint.messages),
                time.time() - checkpoint.timestamp,
            )
            return checkpoint

        except Exception as e:
            logger.error("Checkpoint restore failed for %s: %s", filepath, e)
            return None

    # ── Auto-Checkpoint ──────────────────────────────────────────────────

    def advance_turn(self):
        """Advance the turn counter. Called after each user message."""
        self._turn_count += 1

    def should_auto_checkpoint(self) -> bool:
        """Check if an auto-checkpoint should be triggered."""
        turns_since = self._turn_count - self._last_checkpoint_turn
        return turns_since >= self._auto_interval

    # ── Listing ──────────────────────────────────────────────────────────

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints with metadata."""
        pattern = os.path.join(self._dir, "checkpoint_*.json")
        files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)

        checkpoints = []
        for filepath in files:
            try:
                stat = os.stat(filepath)
                # Quick peek at the JSON for metadata
                with open(filepath, "r") as f:
                    data = json.load(f)
                checkpoints.append({
                    "filename": os.path.basename(filepath),
                    "path": filepath,
                    "turn_count": data.get("turn_count", 0),
                    "message_count": len(data.get("messages", [])),
                    "model": data.get("model_name", ""),
                    "timestamp": data.get("timestamp", stat.st_mtime),
                    "age_seconds": time.time() - data.get("timestamp", stat.st_mtime),
                    "size_bytes": stat.st_size,
                })
            except Exception:
                continue

        return checkpoints
