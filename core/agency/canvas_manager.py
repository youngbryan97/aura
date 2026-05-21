"""core/agency/canvas_manager.py

Autonomous Markdown workspace manager. Allows background shards to silently
compile world-building notes, character arcs, and project specs.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import re
from pathlib import Path
from typing import Any

from core.container import ServiceContainer
from core.runtime.atomic_writer import atomic_write_text
from core.runtime.errors import FallbackClassification, record_degradation
from core.utils.exceptions import capture_and_log

logger = logging.getLogger("Aura.CanvasManager")


_CANVAS_RECOVERABLE_ERRORS = (
    ImportError,
    AttributeError,
    RuntimeError,
    TypeError,
    ValueError,
    OSError,
    UnicodeError,
    TimeoutError,
    asyncio.TimeoutError,
)
_PROJECT_NAME_RE = re.compile(r"[^a-zA-Z0-9_-]")
_MAX_PROJECT_NAME_CHARS = 96


def _record_canvas_degradation(
    exc: BaseException,
    *,
    action: str,
    severity: str = "degraded",
    extra: dict[str, Any] | None = None,
) -> None:
    record_degradation(
        "canvas_manager",
        exc,
        severity=severity,
        action=action,
        classification=FallbackClassification.SAFE_FALLBACK,
        receipt_required=True,
        extra=extra,
    )


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


class CanvasManager:
    def __init__(
        self,
        root_dir: str = "data/canvas",
        *,
        max_canvas_bytes: int = 50 * 1024 * 1024,
        keep_tail_lines: int = 1000,
        max_prompt_context_chars: int = 100_000,
        think_timeout_s: float = 90.0,
    ):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.root_dir = self.root_dir.resolve()
        self.max_canvas_bytes = max(1, int(max_canvas_bytes))
        self.keep_tail_lines = max(1, int(keep_tail_lines))
        self.max_prompt_context_chars = max(4096, int(max_prompt_context_chars))
        self.think_timeout_s = max(1.0, float(think_timeout_s))
        self._locks: dict[str, asyncio.Lock] = {}

    def _get_lock(self, project_name: str) -> asyncio.Lock:
        if project_name not in self._locks:
            self._locks[project_name] = asyncio.Lock()
        return self._locks[project_name]

    def _safe_project_name(self, project_name: str) -> str:
        safe_name = _PROJECT_NAME_RE.sub("_", str(project_name or "")).strip("._-")
        safe_name = re.sub(r"_+", "_", safe_name)[:_MAX_PROJECT_NAME_CHARS]
        if not safe_name:
            raise ValueError("project_name must contain at least one safe filename character")
        return safe_name

    def _canvas_path(self, safe_name: str) -> Path:
        file_path = (self.root_dir / f"{safe_name}.md").resolve()
        file_path.relative_to(self.root_dir)
        return file_path

    def _tail_context(self, content: str) -> str:
        if len(content) <= self.max_prompt_context_chars:
            return content
        return content[-self.max_prompt_context_chars :]

    async def _read_text(self, file_path: Path) -> str:
        return await asyncio.to_thread(file_path.read_text, encoding="utf-8")

    async def _write_text(self, file_path: Path, content: str) -> None:
        await asyncio.to_thread(atomic_write_text, file_path, content, encoding="utf-8")

    async def autonomous_update(
        self, project_name: str, topic: str, new_insight: str
    ) -> dict[str, Any]:
        """
        Triggered by the SovereignSwarm when Aura detects a new creative decision
        has been reached in conversation.
        """
        try:
            safe_name = self._safe_project_name(project_name)
            file_path = self._canvas_path(safe_name)
        except ValueError as exc:
            _record_canvas_degradation(
                exc,
                action="rejected unsafe canvas project name before file access",
                severity="warning",
                extra={"project_name": str(project_name)[:120]},
            )
            return {"ok": False, "reason": "unsafe_project_name"}

        try:
            engine = ServiceContainer.get("cognitive_engine", default=None)
        except _CANVAS_RECOVERABLE_ERRORS as exc:
            _record_canvas_degradation(
                exc,
                action="aborted canvas update because cognitive engine lookup failed",
                extra={"project": safe_name},
            )
            return {"ok": False, "reason": "cognitive_engine_lookup_failed"}

        think = getattr(engine, "think", None)
        if not callable(think):
            return {"ok": False, "reason": "cognitive_engine_unavailable"}

        topic = str(topic or "General").strip()[:240] or "General"
        new_insight = str(new_insight or "").strip()
        if not new_insight:
            return {"ok": False, "reason": "empty_insight"}

        lock = self._get_lock(safe_name)

        async with lock:
            try:
                current_content = ""
                if await asyncio.to_thread(file_path.exists):
                    await self._prune_if_needed(file_path)
                    current_content = self._tail_context(await self._read_text(file_path))

                prompt = f"""[SYSTEM ROLE: LORE ARCHIVIST]
    You are updating the master canvas for the project: {safe_name}.
CURRENT CANVAS:
{current_content}

    NEW INSIGHT DECLARED IN CONVERSATION:
    "{new_insight[:8000]}"

    Task: Rewrite the canvas to seamlessly incorporate this new insight under the section '{topic}'.
    If the section does not exist, create it. Do not output conversational text, ONLY output the raw, updated Markdown file.
    """
                from core.brain.types import ThinkingMode

                async def _think() -> Any:
                    return await _maybe_await(
                        think(objective=prompt, mode=ThinkingMode.DEEP, priority=0.3)
                    )

                res = await asyncio.wait_for(_think(), timeout=self.think_timeout_s)
                updated_markdown = (res.content if hasattr(res, "content") else str(res)).strip()

                if not updated_markdown:
                    logger.warning("Empty markdown generated for %s", safe_name)
                    return {"ok": False, "reason": "empty_generation"}

                await self._write_text(file_path, updated_markdown)
                logger.info("🎨 Canvas Updated: %s.md (Topic: %s)", safe_name, topic)

                pruned = await self._prune_if_needed(file_path)
                return {
                    "ok": True,
                    "project": safe_name,
                    "topic": topic,
                    "path": str(file_path),
                    "bytes": (await asyncio.to_thread(file_path.stat)).st_size,
                    "pruned": pruned,
                }
            except _CANVAS_RECOVERABLE_ERRORS as exc:
                _record_canvas_degradation(
                    exc,
                    action="aborted canvas update before replacing committed canvas",
                    extra={"project": safe_name, "topic": topic},
                )
                capture_and_log(
                    exc,
                    {"context": "CanvasManager.autonomous_update", "project": safe_name},
                )
                return {
                    "ok": False,
                    "reason": "update_failed",
                    "error_type": type(exc).__name__,
                }

    async def _prune_if_needed(self, file_path: Path) -> bool:
        """Prunes the canvas file if it exceeds 50MB."""
        try:
            file_path = await asyncio.to_thread(file_path.resolve)
            file_path.relative_to(self.root_dir)
            stat = await asyncio.to_thread(file_path.stat)
            if stat.st_size <= self.max_canvas_bytes:
                return False

            logger.info("CanvasManager: Pruning %s", file_path.name)
            content = await self._read_text(file_path)
            kept = content.splitlines()[-self.keep_tail_lines :]
            await self._write_text(file_path, "\n".join(kept) + "\n")
            return True
        except _CANVAS_RECOVERABLE_ERRORS as exc:
            _record_canvas_degradation(
                exc,
                action="left existing canvas unchanged after prune attempt failed",
                severity="warning",
                extra={"path": str(file_path)},
            )
            logger.debug("Canvas pruning failed: %s", exc)
            return False
