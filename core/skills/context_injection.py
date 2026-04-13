"""
Context Injection — Ported from gemini-skills GEMINI.md pattern

Scans for project-level .aura.md or AURA.md files and injects their
contents as high-priority system context. Enables per-project custom
instructions, constraints, and knowledge.

Hierarchy: workspace → parent → global (~/.aura/AURA.md)
"""

import logging
import os

logger = logging.getLogger("Aura.ContextInjection")

# Files to scan, in priority order (first found wins per level)
CONTEXT_FILE_NAMES = [
    ".aura.md",
    "AURA.md",
    ".aura.txt",
    "aura.md",
]

GLOBAL_CONTEXT_PATH = os.path.expanduser("~/.aura/AURA.md")


class ContextInjectionService:
    """Discovers and injects per-project context files into system prompts.

    Scans the workspace hierarchy for .aura.md files and merges them
    with proper precedence (workspace overrides parent overrides global).
    Caches content with hash-based change detection to avoid re-reading
    unchanged files.
    """

    def __init__(self) -> None:
        self._cache: dict[str, tuple[str, str]] = {}  # path -> (hash, content)

    def _read_if_changed(self, filepath: str) -> str | None:
        """Read a file only if it has changed since last read."""
        if not os.path.exists(filepath):
            self._cache.pop(filepath, None)
            return None

        try:
            stat = os.stat(filepath)
            # Quick size+mtime hash as change indicator
            change_key = f"{stat.st_size}:{stat.st_mtime_ns}"

            cached = self._cache.get(filepath)
            if cached and cached[0] == change_key:
                return cached[1]

            with open(filepath, encoding="utf-8", errors="ignore") as f:
                content = f.read(50000)  # Cap at 50KB per file

            self._cache[filepath] = (change_key, content)
            return content

        except Exception as e:
            logger.debug("Failed to read context file %s: %s", filepath, e)
            return None

    def discover_context_files(self, workspace_dir: str) -> list[tuple[str, str]]:
        """Discover context files in the workspace hierarchy.

        Returns: List of (filepath, content) tuples, ordered from
                 most specific (workspace) to least specific (global).
        """
        found = []

        # 1. Workspace-level
        if workspace_dir and os.path.isdir(workspace_dir):
            for name in CONTEXT_FILE_NAMES:
                path = os.path.join(workspace_dir, name)
                content = self._read_if_changed(path)
                if content:
                    found.append((path, content))
                    break  # Only take the first found at this level

        # 2. Parent directory (one level up)
        if workspace_dir:
            parent = os.path.dirname(os.path.abspath(workspace_dir))
            if parent != workspace_dir:  # Avoid root loops
                for name in CONTEXT_FILE_NAMES:
                    path = os.path.join(parent, name)
                    content = self._read_if_changed(path)
                    if content:
                        found.append((path, content))
                        break

        # 3. Global (~/.aura/AURA.md)
        global_content = self._read_if_changed(GLOBAL_CONTEXT_PATH)
        if global_content:
            found.append((GLOBAL_CONTEXT_PATH, global_content))

        return found

    def build_context_block(self, workspace_dir: str) -> str:
        """Build a merged context block from all discovered files.

        Returns: Formatted string ready for system prompt injection,
                 or empty string if no context files found.
        """
        files = self.discover_context_files(workspace_dir)

        if not files:
            return ""

        blocks = []
        for filepath, content in files:
            rel = os.path.basename(filepath)
            source = "workspace" if filepath != GLOBAL_CONTEXT_PATH else "global"
            blocks.append(
                f"<!-- Project context from {rel} ({source}) -->\n{content.strip()}"
            )

        merged = "\n\n---\n\n".join(blocks)

        logger.info(
            "Injected %d context file(s) into system prompt (%d chars)",
            len(files), len(merged)
        )

        return f"[PROJECT CONTEXT]\n{merged}\n[/PROJECT CONTEXT]"


# Global singleton
_service: ContextInjectionService | None = None

def get_context_injection_service() -> ContextInjectionService:
    global _service
    if _service is None:
        _service = ContextInjectionService()
    return _service
