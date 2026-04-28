"""core/agency/canvas_manager.py

Autonomous Markdown workspace manager. Allows background shards to silently 
compile world-building notes, character arcs, and project specs.
"""
from core.runtime.errors import record_degradation
import asyncio
import logging
import os
import re
from pathlib import Path
from core.container import ServiceContainer

logger = logging.getLogger("Aura.CanvasManager")

class CanvasManager:
    def __init__(self, root_dir: str = "data/canvas"):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._locks = {}

    def _get_lock(self, project_name: str) -> asyncio.Lock:
        if project_name not in self._locks:
            self._locks[project_name] = asyncio.Lock()
        return self._locks[project_name]

    async def autonomous_update(self, project_name: str, topic: str, new_insight: str):
        """
        Triggered by the SovereignSwarm when Aura detects a new creative decision 
        has been reached in conversation.
        """
        engine = ServiceContainer.get("cognitive_engine", default=None)
        if not engine:
            return

        # CM-003: Sanitize project_name to prevent path traversal
        safe_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', project_name)
        file_path = self.root_dir / f"{safe_name}.md"
        lock = self._get_lock(safe_name)

        async with lock:
            try:
                current_content = ""
                if file_path.exists():
                    # CM-002: Force utf-8 encoding
                    current_content = await asyncio.to_thread(file_path.read_text, encoding="utf-8")

                prompt = f"""[SYSTEM ROLE: LORE ARCHIVIST]
You are updating the master canvas for the project: {safe_name}.
CURRENT CANVAS:
{current_content}

NEW INSIGHT DECLARED IN CONVERSATION:
"{new_insight}"

Task: Rewrite the canvas to seamlessly incorporate this new insight under the section '{topic}'. 
If the section does not exist, create it. Do not output conversational text, ONLY output the raw, updated Markdown file.
"""
                from core.brain.types import ThinkingMode
                # CM-001: Wrap in try/except for robustness
                res = await engine.think(objective=prompt, mode=ThinkingMode.DEEP, priority=0.3)
                updated_markdown = res.content if hasattr(res, 'content') else str(res)

                if not updated_markdown:
                    logger.warning("Empty markdown generated for %s", safe_name)
                    return

                # Write asynchronously with utf-8 encoding
                await asyncio.to_thread(file_path.write_text, updated_markdown, encoding="utf-8")
                logger.info("🎨 Canvas Updated: %s.md (Topic: %s)", safe_name, topic)
                
                # ZENITH Audit Fix 2.1: Automated Pruning
                await self._prune_if_needed(file_path)
            except Exception as e:
                record_degradation('canvas_manager', e)
                from core.utils.exceptions import capture_and_log
                capture_and_log(e, {"context": "CanvasManager.autonomous_update", "project": safe_name})

    async def _prune_if_needed(self, file_path: Path):
        """Prunes the canvas file if it exceeds 50MB."""
        try:
            if file_path.stat().st_size > 50 * 1024 * 1024:
                logger.info("CanvasManager: Pruning %s", file_path.name)
                content = await asyncio.to_thread(file_path.read_text, encoding="utf-8")
                lines = content.splitlines()
                # Keep last 1000 lines for markdown canvases
                kept = lines[-1000:]
                await asyncio.to_thread(file_path.write_text, "\n".join(kept), encoding="utf-8")
        except Exception as e:
            record_degradation('canvas_manager', e)
            logger.debug("Canvas pruning failed: %s", e)
