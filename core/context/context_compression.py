"""Context Compression Service — Ported from gemini-cli/contextCompressionService.ts.

Routes files to 4 compression levels based on relevance:
  FULL     — complete content preserved (recently accessed or highly relevant)
  PARTIAL  — first N lines + summary
  SUMMARY  — LLM-generated summary only
  EXCLUDED — completely removed from context

Protects files read in the last 2 turns from any compression.
Caches summaries with content hashes for change detection.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any

from core.runtime.errors import FallbackClassification, record_degradation

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

    files: dict[str, FileRecord] = field(default_factory=dict)
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

MAX_STATE_FILES = 5000
MAX_PATH_CHARS = 600
MAX_SUMMARY_CHARS = 2000
MAX_TASK_CONTEXT_CHARS = 2000
MAX_ROUTING_FILES = 200
MAX_ROUTING_RESPONSE_CHARS = 20000
SUMMARY_SOURCE_CHARS = 4000


def _emit_context_fault(
    error: BaseException,
    *,
    action: str,
    severity: str = "degraded",
    stage: str = "",
    extra: dict[str, Any] | None = None,
) -> None:
    """Record a context-compression fault with explicit recovery semantics."""
    metadata = dict(extra or {})
    if stage:
        metadata["stage"] = stage
    try:
        record_degradation(
            "context_compression",
            error,
            severity=severity,  # type: ignore[arg-type]
            action=action,
            classification=FallbackClassification.SAFE_FALLBACK,
            extra=metadata or None,
        )
    except TypeError:
        record_degradation("context_compression", error)


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _safe_text(value: Any, default: str = "", *, max_chars: int = 1000) -> str:
    if value is None:
        return default
    try:
        text = str(value)
    except (RuntimeError, TypeError, ValueError):
        return default
    text = text.replace("\x00", "")
    if len(text) > max_chars:
        text = text[:max_chars]
    return text


def _safe_content(value: Any) -> str:
    if isinstance(value, str):
        return value.replace("\x00", "")
    try:
        return str(value).replace("\x00", "")
    except (RuntimeError, TypeError, ValueError):
        return ""


def _coerce_response_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        for key in ("response", "content", "text", "answer"):
            value = result.get(key)
            if isinstance(value, str):
                return value
        return ""
    for attr in ("response", "content", "text", "answer"):
        value = getattr(result, attr, None)
        if isinstance(value, str):
            return value
    return ""


async def _call_generate(brain: Any, prompt: str, *, options: dict[str, Any]) -> Any:
    generate = getattr(brain, "generate", None)
    if not callable(generate):
        raise AttributeError("brain does not expose generate()")
    result = generate(prompt, options=options)
    if inspect.isawaitable(result):
        return await result
    return result


def _extract_json_object(text: str) -> dict[str, Any]:
    if not text:
        raise ValueError("empty routing response")
    bounded = text[:MAX_ROUTING_RESPONSE_CHARS]
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", bounded, re.DOTALL):
        try:
            parsed, _end = decoder.raw_decode(bounded[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
        raise TypeError("routing response root must be an object")
    raise ValueError("routing response did not contain a JSON object")


def _deterministic_summary(path: str, content: str) -> str:
    normalized_path = _safe_text(path, "unknown file", max_chars=MAX_PATH_CHARS)
    lines = content.splitlines()
    nonempty = [line.strip() for line in lines if line.strip()]
    excerpt = " ".join(nonempty[:3])
    if not excerpt:
        excerpt = "No textual content was available."
    return _safe_text(
        f"{normalized_path}: {len(lines)} lines, {len(content)} chars. {excerpt}",
        max_chars=MAX_SUMMARY_CHARS,
    )


class ContextCompressionService:
    """Routes file content through compression levels based on task relevance.

    Key behaviors:
    - Recently accessed files (last 2 turns) are always FULL
    - Files are re-evaluated when the task context changes
    - Summaries are cached with content hashes — regenerated only on file change
    - State can be persisted to disk for session recovery
    """

    def __init__(self, state_file: str | os.PathLike[str] | None = None):
        self._state_file = Path(state_file or STATE_FILE).expanduser()
        self._state = CompressionState()
        self._load_state()

    def _load_state(self):
        """Load compression state from disk if available."""
        path = self._state_file
        try:
            if not path.exists():
                return
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise TypeError("compression state root must be an object")

            self._state.current_turn = max(0, int(data.get("current_turn", 0) or 0))
            files = data.get("files", {})
            if not isinstance(files, dict):
                raise TypeError("compression state files must be an object")

            for raw_path, fdata in list(files.items())[:MAX_STATE_FILES]:
                if not isinstance(fdata, dict):
                    continue
                safe_path = _safe_text(raw_path, max_chars=MAX_PATH_CHARS)
                if not safe_path:
                    continue
                self._state.files[safe_path] = FileRecord(
                    path=safe_path,
                    content_hash=_safe_text(fdata.get("content_hash", ""), max_chars=64),
                    compression_level=self._coerce_level(fdata.get("compression_level", 0)),
                    summary=_safe_text(fdata.get("summary", ""), max_chars=MAX_SUMMARY_CHARS),
                    last_accessed_turn=max(0, int(fdata.get("last_accessed_turn", 0) or 0)),
                    char_count=max(0, int(fdata.get("char_count", 0) or 0)),
                )
        except (json.JSONDecodeError, OSError, RuntimeError, TypeError, ValueError) as e:
            self._quarantine_state(path)
            _emit_context_fault(
                e,
                action="quarantined unreadable compression state and started fresh",
                severity="degraded",
                stage="load_state",
            )
            logger.debug("Could not load compression state: %s", e)

    def _save_state(self):
        """Persist compression state to disk."""
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "current_turn": self._state.current_turn,
                "files": {
                    path: {
                        "content_hash": rec.content_hash,
                        "compression_level": rec.compression_level.value,
                        "summary": _safe_text(rec.summary, max_chars=MAX_SUMMARY_CHARS),
                        "last_accessed_turn": rec.last_accessed_turn,
                        "char_count": rec.char_count,
                    }
                    for path, rec in list(self._state.files.items())[:MAX_STATE_FILES]
                },
            }
            fd, tmp_name = tempfile.mkstemp(
                dir=str(self._state_file.parent),
                prefix=".compression_state.",
                suffix=".tmp",
                text=True,
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)
                os.replace(tmp_name, self._state_file)
            except (OSError, RuntimeError, TypeError, ValueError):
                try:
                    os.unlink(tmp_name)
                except FileNotFoundError:
                    pass
                raise
        except (OSError, TypeError, ValueError, RuntimeError) as e:
            _emit_context_fault(
                e,
                action="continued without persisting compression state",
                severity="degraded",
                stage="save_state",
            )
            logger.debug("Could not save compression state: %s", e)

    def _quarantine_state(self, path: Path) -> None:
        if not path.exists():
            return
        quarantine = path.with_name(f"{path.stem}.corrupt.{int(time.time())}{path.suffix}")
        try:
            path.replace(quarantine)
        except OSError as exc:
            _emit_context_fault(
                exc,
                action="continued fresh compression state after quarantine rename failed",
                severity="warning",
                stage="quarantine_state",
            )

    @staticmethod
    def _coerce_level(value: Any) -> CompressionLevel:
        if isinstance(value, CompressionLevel):
            return value
        if isinstance(value, str):
            normalized = value.strip().upper()
            if normalized in CompressionLevel.__members__:
                return CompressionLevel[normalized]
        try:
            return CompressionLevel(int(value))
        except (TypeError, ValueError):
            return CompressionLevel.FULL

    def advance_turn(self):
        """Advance the current turn counter."""
        self._state.current_turn += 1

    def register_file_access(self, path: str, content: str):
        """Register that a file was accessed in the current turn.

        Updates the content hash and marks the file as recently accessed.
        """
        self._ensure_record(path, content, mark_access=True)

    def _ensure_record(self, path: str, content: str, *, mark_access: bool) -> FileRecord:
        safe_path = _safe_text(path, "unknown", max_chars=MAX_PATH_CHARS)
        safe_content = _safe_content(content)
        content_hash = _content_hash(safe_content)
        existing = self._state.files.get(safe_path)

        if existing is None:
            existing = FileRecord(
                path=safe_path,
                content_hash=content_hash,
                compression_level=CompressionLevel.FULL,
                summary="",
                last_accessed_turn=self._state.current_turn if mark_access else 0,
                char_count=len(safe_content),
            )
            self._state.files[safe_path] = existing
        elif existing.content_hash != content_hash:
            existing.content_hash = content_hash
            existing.summary = ""
            existing.char_count = len(safe_content)
            if existing.compression_level in {CompressionLevel.SUMMARY, CompressionLevel.EXCLUDED}:
                existing.compression_level = CompressionLevel.PARTIAL

        if mark_access:
            existing.last_accessed_turn = self._state.current_turn
            existing.compression_level = CompressionLevel.FULL

        return existing

    def _is_protected(self, record: FileRecord) -> bool:
        """Check if a file is protected from compression."""
        age = max(0, self._state.current_turn - record.last_accessed_turn)
        return age <= PROTECTED_TURN_WINDOW

    async def route_files(
        self,
        file_contents: dict[str, str],
        task_context: str,
        brain: Any = None,
    ) -> dict[str, CompressionLevel]:
        """Route all registered files to compression levels.

        Args:
            file_contents: {path: content} of all files in context
            task_context: Current task description for relevance routing
            brain: LocalBrain for LLM-powered routing decisions

        Returns:
            {path: CompressionLevel} mapping
        """
        if not isinstance(file_contents, dict):
            raise TypeError("file_contents must be a mapping of path to content")

        # Register all files and invalidate stale hashes without treating every
        # file in the context window as explicitly accessed by the user.
        for path, content in file_contents.items():
            self._ensure_record(path, content, mark_access=path not in self._state.files)

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
            for path, record in to_evaluate.items():
                record.compression_level = CompressionLevel.FULL
                result[path] = CompressionLevel.FULL
            self._save_state()
            return result

        # Batch LLM routing
        try:
            file_list = "\n".join(
                f"- {path} ({record.char_count} chars, last accessed turn {record.last_accessed_turn})"
                for path, record in list(to_evaluate.items())[:MAX_ROUTING_FILES]
            )
            prompt = ROUTING_PROMPT.format(
                task_context=_safe_text(task_context, max_chars=MAX_TASK_CONTEXT_CHARS),
                file_list=file_list,
            )

            result_data = await _call_generate(
                brain,
                prompt, options={"num_predict": 1024, "temperature": 0.1}
            )
            response_text = _coerce_response_text(result_data)

            # Parse JSON response
            routing = _extract_json_object(response_text)
            files = routing.get("files", [])
            if not isinstance(files, list):
                raise TypeError("routing response 'files' must be a list")

            for file_info in files:
                if not isinstance(file_info, dict):
                    continue
                path = _safe_text(file_info.get("path", ""), max_chars=MAX_PATH_CHARS)
                level = self._coerce_level(file_info.get("level", "FULL"))
                if path in to_evaluate:
                    to_evaluate[path].compression_level = level

        except (
            AttributeError,
            json.JSONDecodeError,
            OSError,
            RuntimeError,
            TimeoutError,
            TypeError,
            ValueError,
        ) as e:
            for record in to_evaluate.values():
                record.compression_level = CompressionLevel.FULL
            _emit_context_fault(
                e,
                action="failed open to FULL context after routing failure",
                severity="degraded",
                stage="route_files",
                extra={"evaluated_files": len(to_evaluate)},
            )
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
            current_hash = _content_hash(_safe_content(content))
            if record and record.content_hash != current_hash:
                record.content_hash = current_hash
                record.summary = ""
                record.char_count = len(content)
            if record and record.summary:
                return f"[Summary of {path}]: {record.summary}"
            # No summary cached — fall back to PARTIAL
            return self.apply_compression(path, content, CompressionLevel.PARTIAL)

        return content

    async def generate_summary(self, path: str, content: str, brain: Any) -> str:
        """Generate and cache an LLM summary for a file."""
        record = self._ensure_record(path, content, mark_access=False)
        try:
            result = await _call_generate(
                brain,
                (
                    "Summarize this file in 2-3 sentences. Focus on its purpose, "
                    "key functions, and important constants.\n\n"
                    f"{_safe_text(content, max_chars=SUMMARY_SOURCE_CHARS)}"
                ),
                options={"num_predict": 256, "temperature": 0.2},
            )
            summary = _safe_text(
                _coerce_response_text(result).strip(),
                max_chars=MAX_SUMMARY_CHARS,
            )
            if len(summary) >= 20:
                record.summary = summary
                self._save_state()
                return summary
            raise ValueError("LLM returned an empty or too-short summary")
        except (AttributeError, OSError, RuntimeError, TimeoutError, TypeError, ValueError) as e:
            fallback = _deterministic_summary(path, content)
            record.summary = fallback
            self._save_state()
            _emit_context_fault(
                e,
                action="generated deterministic file summary fallback",
                severity="warning",
                stage="generate_summary",
                extra={"path": record.path},
            )
            logger.warning("Summary generation failed for %s: %s", path, e)
            return fallback
