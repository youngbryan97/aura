"""
Context Compression Service — Ported from gemini-cli/contextCompressionService.ts

Routes files to 4 compression levels based on relevance:
  FULL     — complete content preserved (recently accessed or highly relevant)
  PARTIAL  — first N lines + summary
  SUMMARY  — LLM-generated summary only
  EXCLUDED — completely removed from context

Protects files read in the last 2 turns from any compression.
Caches summaries with content hashes for change detection.
"""

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger("Aura.ContextCompression")


class CompressionLevel(IntEnum):
    """Compression levels ordered from most to least preserved."""
    FULL = 0
    PARTIAL = 1
    SUMMARY = 2
    EXCLUDED = 3


@dataclass
class FileRecord:
    """Tracks a file's compression state and cached summary."""
    path: str
    content_hash: str = ""
    compression_level: CompressionLevel = CompressionLevel.FULL
    summary: str = ""
    last_accessed_turn: int = 0
    char_count: int = 0


@dataclass
class CompressionState:
    """Serializable compression state."""
    files: Dict[str, FileRecord] = field(default_factory=dict)
    current_turn: int = 0


# ── Configuration ────────────────────────────────────────────────────────────

# Files accessed within this many turns are protected from compression
PROTECTED_TURN_WINDOW = 2

# Maximum chars for PARTIAL level (first N lines)
PARTIAL_MAX_CHARS = 2000

# Batch routing prompt
ROUTING_PROMPT = """You are a context optimization assistant. Given a user's current task and a list of files in memory, categorize each file's relevance.

For each file, assign ONE level:
- FULL: Essential to the current task — user is actively working with this file
- PARTIAL: Related but not actively needed — keep first section for reference
- SUMMARY: Background context — a one-line summary is sufficient
- EXCLUDED: Completely irrelevant to the current task

Respond as JSON:
{{"files": [{{"path": "...", "level": "FULL|PARTIAL|SUMMARY|EXCLUDED", "reason": "..."}}]}}

Current task context: {task_context}

Files to evaluate:
{file_list}"""

STATE_FILE = os.path.expanduser("~/.aura_runtime/compression_state.json")


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()[:16]


class ContextCompressionService:
    """Routes file content through compression levels based on task relevance.

    Key behaviors:
    - Recently accessed files (last 2 turns) are always FULL
    - Files are re-evaluated when the task context changes
    - Summaries are cached with content hashes — regenerated only on file change
    - State can be persisted to disk for session recovery
    """

    def __init__(self):
        self._state = CompressionState()
        self._load_state()

    def _load_state(self):
        """Load compression state from disk if available."""
        try:
            if os.path.exists(STATE_FILE):
                with open(STATE_FILE, "r") as f:
                    data = json.load(f)
                self._state.current_turn = data.get("current_turn", 0)
                for path, fdata in data.get("files", {}).items():
                    self._state.files[path] = FileRecord(
                        path=path,
                        content_hash=fdata.get("content_hash", ""),
                        compression_level=CompressionLevel(fdata.get("compression_level", 0)),
                        summary=fdata.get("summary", ""),
                        last_accessed_turn=fdata.get("last_accessed_turn", 0),
                        char_count=fdata.get("char_count", 0),
                    )
        except Exception as e:
            logger.debug("Could not load compression state: %s", e)

    def _save_state(self):
        """Persist compression state to disk."""
        try:
            os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
            data = {
                "current_turn": self._state.current_turn,
                "files": {
                    path: {
                        "content_hash": rec.content_hash,
                        "compression_level": rec.compression_level.value,
                        "summary": rec.summary,
                        "last_accessed_turn": rec.last_accessed_turn,
                        "char_count": rec.char_count,
                    }
                    for path, rec in self._state.files.items()
                }
            }
            with open(STATE_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.debug("Could not save compression state: %s", e)

    def advance_turn(self):
        """Advance the current turn counter."""
        self._state.current_turn += 1

    def register_file_access(self, path: str, content: str):
        """Register that a file was accessed in the current turn.

        Updates the content hash and marks the file as recently accessed.
        """
        content_hash = _content_hash(content)
        existing = self._state.files.get(path)

        if existing and existing.content_hash == content_hash:
            # Same content — just update access time
            existing.last_accessed_turn = self._state.current_turn
            existing.compression_level = CompressionLevel.FULL
        else:
            # New or changed content — reset compression
            self._state.files[path] = FileRecord(
                path=path,
                content_hash=content_hash,
                compression_level=CompressionLevel.FULL,
                summary="",  # Invalidate cached summary
                last_accessed_turn=self._state.current_turn,
                char_count=len(content),
            )

    def _is_protected(self, record: FileRecord) -> bool:
        """Check if a file is protected from compression."""
        return (self._state.current_turn - record.last_accessed_turn) <= PROTECTED_TURN_WINDOW

    async def route_files(
        self,
        file_contents: Dict[str, str],
        task_context: str,
        brain: Any = None,
    ) -> Dict[str, CompressionLevel]:
        """Route all registered files to compression levels.

        Args:
            file_contents: {path: content} of all files in context
            task_context: Current task description for relevance routing
            brain: LocalBrain for LLM-powered routing decisions

        Returns:
            {path: CompressionLevel} mapping
        """
        # Register all files
        for path, content in file_contents.items():
            if path not in self._state.files:
                self.register_file_access(path, content)

        # Separate protected vs evaluatable files
        protected = {}
        to_evaluate = {}

        for path, record in self._state.files.items():
            if path not in file_contents:
                continue
            if self._is_protected(record):
                protected[path] = CompressionLevel.FULL
            else:
                to_evaluate[path] = record

        # If no brain or nothing to evaluate, keep everything FULL
        if not brain or not to_evaluate:
            result = {**protected}
            for path in to_evaluate:
                result[path] = CompressionLevel.FULL
            return result

        # Batch LLM routing
        try:
            file_list = "\n".join(
                f"- {path} ({record.char_count} chars, last accessed turn {record.last_accessed_turn})"
                for path, record in to_evaluate.items()
            )
            prompt = ROUTING_PROMPT.format(
                task_context=task_context,
                file_list=file_list,
            )

            result_data = await brain.generate(
                prompt, options={"num_predict": 1024, "temperature": 0.1}
            )
            response_text = result_data.get("response", "")

            # Parse JSON response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                routing = json.loads(json_match.group())
                level_map = {"FULL": CompressionLevel.FULL, "PARTIAL": CompressionLevel.PARTIAL,
                             "SUMMARY": CompressionLevel.SUMMARY, "EXCLUDED": CompressionLevel.EXCLUDED}

                for file_info in routing.get("files", []):
                    path = file_info.get("path", "")
                    level_str = file_info.get("level", "FULL")
                    if path in to_evaluate:
                        to_evaluate[path].compression_level = level_map.get(level_str, CompressionLevel.FULL)

        except Exception as e:
            logger.warning("LLM file routing failed, keeping all FULL: %s", e)

        # Build final result
        result = {**protected}
        for path, record in to_evaluate.items():
            result[path] = record.compression_level

        self._save_state()
        return result

    def apply_compression(
        self, path: str, content: str, level: CompressionLevel
    ) -> str:
        """Apply the specified compression level to file content.

        Args:
            path: File path
            content: Full file content
            level: Target compression level

        Returns:
            Compressed content string
        """
        if level == CompressionLevel.FULL:
            return content

        if level == CompressionLevel.EXCLUDED:
            return f"[File excluded from context: {path}]"

        if level == CompressionLevel.PARTIAL:
            lines = content.split("\n")
            partial = "\n".join(lines[:50])
            if len(partial) > PARTIAL_MAX_CHARS:
                partial = partial[:PARTIAL_MAX_CHARS]
            remaining = len(lines) - 50
            if remaining > 0:
                partial += f"\n\n... [{remaining} more lines in {path}]"
            return partial

        if level == CompressionLevel.SUMMARY:
            record = self._state.files.get(path)
            if record and record.summary:
                return f"[Summary of {path}]: {record.summary}"
            # No summary cached — fall back to PARTIAL
            return self.apply_compression(path, content, CompressionLevel.PARTIAL)

        return content

    async def generate_summary(self, path: str, content: str, brain: Any) -> str:
        """Generate and cache an LLM summary for a file."""
        try:
            result = await brain.generate(
                f"Summarize this file in 2-3 sentences. Focus on its purpose, key functions, and important constants.\n\n{content[:4000]}",
                options={"num_predict": 256, "temperature": 0.2}
            )
            summary = result.get("response", "").strip()
            if summary:
                record = self._state.files.get(path)
                if record:
                    record.summary = summary
                self._save_state()
                return summary
        except Exception as e:
            logger.warning("Summary generation failed for %s: %s", path, e)
        return ""
