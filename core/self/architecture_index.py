"""core/self/architecture_index.py

Architecture Self-Awareness Index.

Builds and maintains a searchable in-memory map of Aura's own source code.
This is first-class cognition, not a file browser — Aura can reason about
her own subsystems, trace data flows, and explain how she works.

What it indexes (per Python file):
  - Subsystem name (derived from directory path)
  - Classes and their docstrings
  - Public methods and their signatures
  - File-level module docstring

Query returns the most relevant code excerpts for a given topic, formatted
for injection into the LLM context so Aura can answer questions like:
  "How does the OutputGate work?"
  "What handles knowledge gap monitoring?"
  "Walk me through how a user message becomes a reply."
"""

from core.runtime.errors import record_degradation
import ast
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("Aura.ArchitectureIndex")

# Only index these core subsystems — keeps the index tight and relevant
_INDEXED_DIRS = {
    "core",
    "core/affect",
    "core/agency",
    "core/brain",
    "core/consciousness",
    "core/continuity",
    "core/identity",
    "core/memory",
    "core/orchestrator",
    "core/senses",
    "core/self",
    "core/skills",
    "core/utils",
    "core/world_model",
    "interface",
}

_SKIP_FILES = {"__init__.py", "__pycache__"}
_MAX_RESULTS = 5
_INDEX_TTL   = 3600.0  # Rebuild once per hour


@dataclass
class ModuleRecord:
    path: str              # Relative to project root
    subsystem: str         # Top-level label (e.g. "brain/inference", "affect/circumplex")
    module_doc: str        # File-level docstring
    classes: List[Dict]    # [{name, doc, methods: [str]}]
    keywords: List[str]    # Extracted for fast matching


class ArchitectureIndex:
    """
    Lazy-built, cached index of Aura's own source code.
    Thread-safe for reads; rebuilds are idempotent.
    """

    def __init__(self, project_root: Optional[Path] = None):
        self._root = project_root or self._detect_root()
        self._index: Dict[str, ModuleRecord] = {}
        self._built_at: float = 0.0
        self._building = False
        self._created_at_monotonic = time.monotonic()
        self._deferred_reason: str | None = None
        self._build_thread: threading.Thread | None = None
        self._schedule_lock = threading.Lock()

    # ─── Public API ───────────────────────────────────────────────────────────

    def build(self, force: bool = False) -> int:
        """Build the index synchronously. Returns number of files indexed."""
        if not force and self._foreground_build_deferred():
            return len(self._index)
        if not force and self._index and (time.monotonic() - self._built_at) < _INDEX_TTL:
            return len(self._index)
        if self._building:
            return len(self._index)
        self._building = True
        try:
            self._index = {}
            count = self._walk_and_index()
            self._built_at = time.monotonic()
            logger.info("🧠 ArchitectureIndex: indexed %d modules", count)
            return count
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            record_degradation('architecture_index', e)
            logger.error("ArchitectureIndex build failed: %s", e)
            return 0
        finally:
            self._building = False

    def query(self, topic: str, max_results: int = _MAX_RESULTS) -> str:
        """
        Return a formatted excerpt of relevant modules for the given topic.
        Suitable for direct injection into an LLM system prompt or context block.
        """
        if not self._index:
            if self._interactive_runtime():
                self.schedule_background_build()
                return ""
            self.build()

        hits = self._search(topic, max_results)
        if not hits:
            return ""

        lines = [f"## ARCHITECTURE: Relevant modules for '{topic}'"]
        for rec, score in hits:
            lines.append(f"\n### {rec.subsystem}  ({rec.path})")
            if rec.module_doc:
                lines.append(f"_{rec.module_doc[:200].strip()}_")
            for cls in rec.classes[:3]:
                lines.append(f"\n**{cls['name']}**: {cls['doc'][:150].strip()}")
                if cls["methods"]:
                    lines.append("  Methods: " + ", ".join(cls["methods"][:8]))
        return "\n".join(lines)

    def get_overview(self) -> str:
        """
        Returns a high-level subsystem map: one line per module with a summary.
        """
        if not self._index:
            if self._interactive_runtime():
                self.schedule_background_build()
                return ""
            self.build()

        lines = ["## AURA ARCHITECTURE OVERVIEW"]
        seen = set()
        for path, rec in sorted(self._index.items()):
            subsys = rec.subsystem.split("/")[0]
            if subsys not in seen:
                seen.add(subsys)
                # Pick a representative module from this subsystem
                lines.append(f"  • **{subsys}**: {rec.module_doc[:100].strip() or '(no doc)'}")
        return "\n".join(lines)

    @staticmethod
    def _env_flag(name: str, default: bool = False) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return default
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    @classmethod
    def _interactive_runtime(cls) -> bool:
        return any(
            cls._env_flag(name)
            for name in (
                "AURA_FOREGROUND_ONLY",
                "AURA_SAFE_BOOT_DESKTOP",
                "AURA_HEADLESS",
            )
        )

    @classmethod
    def _foreground_background_build_allowed(cls) -> bool:
        """Return true when live foreground sessions may self-index in-process.

        The architecture index is useful, but a full repository AST walk is not
        a foreground-critical subsystem. In the desktop lane it can collide with
        active chat turns, shutdown, and model warmup. Keeping it opt-in makes
        launch and conversation reliability the default while preserving an
        explicit escape hatch for diagnostic sessions.
        """

        return cls._env_flag("AURA_ALLOW_FOREGROUND_ARCHITECTURE_INDEX")

    def _foreground_build_deferred(self, *, emit_log: bool = True) -> bool:
        if not self._interactive_runtime():
            return False
        if self._foreground_background_build_allowed():
            return False
        try:
            quiet_s = float(os.getenv("AURA_FOREGROUND_ARCHITECTURE_INDEX_QUIET_S", "180") or 180.0)
        except (TypeError, ValueError):
            quiet_s = 180.0
        age_s = max(0.0, time.monotonic() - self._created_at_monotonic)
        if age_s < max(0.0, quiet_s):
            self._deferred_reason = f"foreground_quiet_window:{age_s:.1f}s/{max(0.0, quiet_s):.1f}s"
            if emit_log:
                logger.info("🧠 ArchitectureIndex build deferred (%s).", self._deferred_reason)
            return True
        try:
            from core.runtime.foreground_guard import foreground_activity_reason

            activity_reason = str(foreground_activity_reason() or "").strip()
            if activity_reason:
                self._deferred_reason = activity_reason
                if emit_log:
                    logger.info("🧠 ArchitectureIndex build deferred (%s).", activity_reason)
                return True
        except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation("architecture_index", exc)
            self._deferred_reason = "foreground_guard_unavailable"
            return True
        return False

    def schedule_background_build(self) -> None:
        """Build once the interactive runtime has been quiet long enough."""

        if self._index or self._building:
            return
        if self._interactive_runtime() and not self._foreground_background_build_allowed():
            self._foreground_build_deferred()
            return
        with self._schedule_lock:
            if self._build_thread is not None and self._build_thread.is_alive():
                return
            self._build_thread = threading.Thread(
                target=self._build_when_quiet,
                daemon=True,
                name="ArchitectureIndexBuild",
            )
            self._build_thread.start()

    def _build_when_quiet(self) -> None:
        while not self._index:
            if not self._foreground_build_deferred(emit_log=False):
                self.build()
                return
            time.sleep(5.0)

    def get_subsystem_detail(self, subsystem: str) -> str:
        """Return full detail for all modules in a named subsystem."""
        if not self._index:
            if self._interactive_runtime():
                self.schedule_background_build()
                return f"Architecture index is warming; no modules are available yet for '{subsystem}'."
            self.build()

        matches = [
            rec for rec in self._index.values()
            if subsystem.lower() in rec.subsystem.lower()
        ]
        if not matches:
            return f"No modules found for subsystem '{subsystem}'."

        lines = [f"## {subsystem.upper()} SUBSYSTEM"]
        for rec in matches[:6]:
            lines.append(f"\n### {rec.path}")
            if rec.module_doc:
                lines.append(rec.module_doc[:300].strip())
            for cls in rec.classes[:5]:
                lines.append(f"\n**{cls['name']}**: {cls['doc'][:200].strip()}")
                if cls["methods"]:
                    lines.append("  " + ", ".join(cls["methods"][:10]))
        return "\n".join(lines)

    # ─── Indexing ─────────────────────────────────────────────────────────────

    def _walk_and_index(self) -> int:
        count = 0
        for dir_rel in _INDEXED_DIRS:
            dir_abs = self._root / dir_rel
            if not dir_abs.exists():
                continue
            for py_file in dir_abs.rglob("*.py"):
                if any(skip in py_file.parts for skip in _SKIP_FILES):
                    continue
                rel = str(py_file.relative_to(self._root))
                try:
                    rec = self._parse_file(py_file, rel)
                    if rec:
                        self._index[rel] = rec
                        count += 1
                except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                    record_degradation('architecture_index', e)
                    logger.debug("Index skip %s: %s", rel, e)
        return count

    def _parse_file(self, path: Path, rel: str) -> Optional[ModuleRecord]:
        src = path.read_text(encoding="utf-8", errors="ignore")
        if len(src) < 50:
            return None  # Skip near-empty files

        try:
            tree = ast.parse(src)
        except SyntaxError:
            return None

        # Module docstring
        module_doc = ""
        if (
            tree.body
            and isinstance(tree.body[0], ast.Expr)
            and isinstance(tree.body[0].value, ast.Constant)
        ):
            module_doc = str(tree.body[0].value.value).split("\n")[0].strip()

        # Classes
        classes = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            cls_doc = ast.get_docstring(node) or ""
            methods = [
                n.name for n in node.body
                if isinstance(n, ast.FunctionDef) and not n.name.startswith("__")
            ]
            classes.append({
                "name": node.name,
                "doc":  cls_doc.split("\n")[0][:200] if cls_doc else "",
                "methods": methods,
            })

        # Keywords for search
        subsystem = self._path_to_subsystem(rel)
        keywords = self._extract_keywords(module_doc, classes, rel)

        return ModuleRecord(
            path=rel,
            subsystem=subsystem,
            module_doc=module_doc,
            classes=classes,
            keywords=keywords,
        )

    @staticmethod
    def _path_to_subsystem(rel: str) -> str:
        parts = Path(rel).parts
        # Drop "core/" prefix for readability, keep the rest
        if parts and parts[0] == "core":
            parts = parts[1:]
        # Drop filename stem
        if len(parts) > 1:
            return "/".join(parts[:-1]) + "/" + Path(parts[-1]).stem
        return Path(parts[0]).stem if parts else rel

    @staticmethod
    def _extract_keywords(module_doc: str, classes: List[Dict], rel: str) -> List[str]:
        words = set()
        # File path components
        words.update(Path(rel).stem.lower().replace("_", " ").split())
        words.update(Path(rel).parts)
        # Module doc words
        for word in module_doc.lower().split():
            clean = word.strip(".,;:()")
            if len(clean) > 3:
                words.add(clean)
        # Class names (split CamelCase)
        for cls in classes:
            name = cls["name"]
            # Split CamelCase → individual words
            import re
            parts = re.sub(r"([A-Z])", r" \1", name).lower().split()
            words.update(parts)
            # Method names
            for m in cls["methods"]:
                words.update(m.lower().split("_"))
        return [w for w in words if len(w) > 2]

    # ─── Search ───────────────────────────────────────────────────────────────

    def _search(self, topic: str, max_results: int) -> List[Tuple["ModuleRecord", float]]:
        import re
        query_words = set(
            w.lower().strip(".,;:()") for w in re.split(r"\W+", topic)
            if len(w) > 2
        )
        if not query_words:
            return []

        scored: List[Tuple[ModuleRecord, float]] = []
        for rec in self._index.values():
            score = 0.0
            kw_set = set(rec.keywords)
            # Keyword overlap
            overlap = query_words & kw_set
            score += len(overlap) * 2.0
            # Subsystem name match
            for qw in query_words:
                if qw in rec.subsystem.lower():
                    score += 3.0
            # Module doc match
            doc_lower = rec.module_doc.lower()
            for qw in query_words:
                if qw in doc_lower:
                    score += 1.5
            # Class name exact match
            for cls in rec.classes:
                for qw in query_words:
                    if qw in cls["name"].lower():
                        score += 4.0

            if score > 0:
                scored.append((rec, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:max_results]

    # ─── Utility ──────────────────────────────────────────────────────────────

    @staticmethod
    def _detect_root() -> Path:
        """Walk up from this file to find the project root (contains 'core/')."""
        here = Path(__file__).resolve()
        for parent in here.parents:
            if (parent / "core").is_dir() and (parent / "interface").is_dir():
                return parent
        # Fallback: assume we're two levels deep in core/
        return here.parent.parent.parent


# ── Singleton ──────────────────────────────────────────────────────────────────
_index: Optional[ArchitectureIndex] = None


def get_architecture_index() -> ArchitectureIndex:
    global _index
    if _index is None:
        _index = ArchitectureIndex()
        if _index._interactive_runtime():
            _index._foreground_build_deferred()
        else:
            _index.schedule_background_build()
    return _index
