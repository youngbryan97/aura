"""Code Graph: AST-based symbol index for Aura's self-knowledge.

Parses the codebase into a graph of symbols (functions, classes, methods,
imports, calls) and their relationships. Aura can query this to understand
her own architecture before making changes.

No external dependencies — uses Python's built-in ast module and SQLite.

Usage:
    graph = CodeGraph(root=Path("~/Desktop/aura"))
    await graph.build()  # Parse entire codebase (~2-5s for 900+ files)

    # Queries
    graph.who_calls("_commit_vault")
    graph.what_does("UnitaryResponsePhase")
    graph.dependencies_of("core/kernel/aura_kernel.py")
    graph.dependents_of("core/affect/damasio_v2.py")
    graph.search_symbols("phi")
"""
import ast
import asyncio
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.CodeGraph")

# Symbol types
SYM_FUNCTION = "function"
SYM_CLASS = "class"
SYM_METHOD = "method"
SYM_MODULE = "module"

# Relationship types
REL_CALLS = "calls"
REL_IMPORTS = "imports"
REL_INHERITS = "inherits"
REL_DEFINES = "defines"
REL_CONTAINS = "contains"

_DEFAULT_EXCLUDED_PARTS = frozenset(
    {
        ".git",
        ".agents",
        ".aura_architect",
        ".claude",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "archive",
        "artifacts",
        "build",
        "data",
        "dist",
        "htmlcov",
        "logs",
        "node_modules",
        "scratch",
        "test_vdb",
        "venv",
    }
)


class CodeGraph:
    """AST-based code graph for codebase self-knowledge."""

    def __init__(self, root: Path | None = None, db_path: str | None = None):
        if root is None:
            root = Path(__file__).resolve().parent.parent.parent
        self.root = Path(root).resolve()
        self.db_path = db_path or str(self.root / "data" / "code_graph.db")
        self._conn: sqlite3.Connection | None = None
        self._built = False
        self._stats = {"files": 0, "symbols": 0, "relationships": 0, "errors": 0}

    # ── Database ─────────────────────────────────────────────────────────

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._init_schema()
        return self._conn

    def _init_schema(self):
        conn = self._conn
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS symbols (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                qualified_name TEXT NOT NULL,
                type TEXT NOT NULL,
                file_path TEXT NOT NULL,
                line_start INTEGER,
                line_end INTEGER,
                docstring TEXT,
                signature TEXT,
                UNIQUE(qualified_name, file_path)
            );
            CREATE TABLE IF NOT EXISTS relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_name TEXT NOT NULL,
                target_name TEXT NOT NULL,
                rel_type TEXT NOT NULL,
                source_file TEXT,
                target_file TEXT,
                line_number INTEGER
            );
            CREATE TABLE IF NOT EXISTS file_index (
                file_path TEXT PRIMARY KEY,
                module_name TEXT,
                last_modified REAL,
                line_count INTEGER,
                symbol_count INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_sym_name ON symbols(name);
            CREATE INDEX IF NOT EXISTS idx_sym_type ON symbols(type);
            CREATE INDEX IF NOT EXISTS idx_sym_file ON symbols(file_path);
            CREATE INDEX IF NOT EXISTS idx_rel_source ON relationships(source_name);
            CREATE INDEX IF NOT EXISTS idx_rel_target ON relationships(target_name);
            CREATE INDEX IF NOT EXISTS idx_rel_type ON relationships(rel_type);
        """)
        conn.commit()

    # ── Building ─────────────────────────────────────────────────────────

    async def build(self, incremental: bool = True) -> dict[str, int]:
        """Parse the codebase and build the symbol graph.

        Args:
            incremental: Only reparse files modified since last build.
        """
        start = time.time()
        conn = self._get_conn()
        self._stats = {"files": 0, "symbols": 0, "relationships": 0, "errors": 0}

        if not incremental:
            conn.executescript("DELETE FROM symbols; DELETE FROM relationships; DELETE FROM file_index;")

        py_files = list(self.root.rglob("*.py"))
        # Skip generated, archived, cache, and local-runtime directories. The
        # graph is for active architecture, not historical repair debris.
        py_files = [
            f for f in py_files
            if not any(part in _DEFAULT_EXCLUDED_PARTS for part in f.relative_to(self.root).parts)
        ]

        # Check which files need reparsing
        existing = {}
        if incremental:
            for row in conn.execute("SELECT file_path, last_modified FROM file_index"):
                existing[row["file_path"]] = row["last_modified"]

        to_parse = []
        for f in py_files:
            rel = str(f.relative_to(self.root))
            mtime = f.stat().st_mtime
            if not incremental or existing.get(rel, 0) < mtime:
                to_parse.append((f, rel))

        logger.info("CodeGraph: %d files to parse (%d total, %d cached)", len(to_parse), len(py_files), len(py_files) - len(to_parse))

        # Parse in thread to not block event loop
        await asyncio.to_thread(self._parse_files, to_parse)

        elapsed = time.time() - start
        self._built = True
        self._stats["build_time_s"] = round(elapsed, 2)

        logger.info(
            "CodeGraph built: %d files, %d symbols, %d relationships (%.1fs)",
            self._stats["files"], self._stats["symbols"],
            self._stats["relationships"], elapsed,
        )
        return dict(self._stats)

    def _parse_files(self, files: list[tuple[Path, str]]):
        conn = self._get_conn()
        symbol_rows: list[tuple[Any, ...]] = []
        relationship_rows: list[tuple[Any, ...]] = []
        file_rows: list[tuple[Any, ...]] = []
        stale_file_rows: list[tuple[str]] = []
        for filepath, rel_path in files:
            try:
                source = filepath.read_text(encoding="utf-8", errors="ignore")
                tree = ast.parse(source, filename=rel_path)
                line_count = source.count("\n") + 1
                modified_at = filepath.stat().st_mtime
                method_nodes = {
                    child
                    for parent in ast.walk(tree)
                    if isinstance(parent, ast.ClassDef)
                    for child in ast.iter_child_nodes(parent)
                    if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef)
                }

                # Clear old data for this file
                stale_file_rows.append((rel_path,))

                module_name = rel_path.replace("/", ".").replace(".py", "")
                symbols_in_file = 0

                # Extract symbols and relationships
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                        sym_type = SYM_METHOD if node in method_nodes else SYM_FUNCTION

                        sig = self._extract_signature(node)
                        doc = ast.get_docstring(node) or ""
                        qualified = f"{module_name}.{node.name}"

                        symbol_rows.append(
                            (
                                node.name,
                                qualified,
                                sym_type,
                                rel_path,
                                node.lineno,
                                node.end_lineno or node.lineno,
                                doc[:500],
                                sig,
                            )
                        )
                        symbols_in_file += 1
                        self._stats["symbols"] += 1

                        # Extract calls from function body
                        for child in ast.walk(node):
                            if isinstance(child, ast.Call):
                                call_name = self._extract_call_name(child)
                                if call_name:
                                    relationship_rows.append(
                                        (node.name, call_name, REL_CALLS, rel_path, child.lineno)
                                    )
                                    self._stats["relationships"] += 1

                    elif isinstance(node, ast.ClassDef):
                        doc = ast.get_docstring(node) or ""
                        qualified = f"{module_name}.{node.name}"

                        symbol_rows.append(
                            (
                                node.name,
                                qualified,
                                SYM_CLASS,
                                rel_path,
                                node.lineno,
                                node.end_lineno or node.lineno,
                                doc[:500],
                                "",
                            )
                        )
                        symbols_in_file += 1
                        self._stats["symbols"] += 1

                        # Extract inheritance
                        for base in node.bases:
                            base_name = self._extract_call_name(base) if isinstance(base, ast.Call) else self._get_name(base)
                            if base_name:
                                relationship_rows.append(
                                    (node.name, base_name, REL_INHERITS, rel_path, node.lineno)
                                )
                                self._stats["relationships"] += 1

                    elif isinstance(node, (ast.Import, ast.ImportFrom)):
                        for alias in node.names:
                            import_name = alias.name
                            if isinstance(node, ast.ImportFrom) and node.module:
                                import_name = f"{node.module}.{alias.name}"
                            relationship_rows.append(
                                (module_name, import_name, REL_IMPORTS, rel_path, node.lineno)
                            )
                            self._stats["relationships"] += 1

                file_rows.append(
                    (rel_path, module_name, modified_at, line_count, symbols_in_file),
                )
                self._stats["files"] += 1

            except SyntaxError:
                self._stats["errors"] += 1
            except Exception as e:
                record_degradation('code_graph', e)
                self._stats["errors"] += 1
                logger.debug("CodeGraph parse error in %s: %s", rel_path, e)

        if stale_file_rows:
            conn.executemany("DELETE FROM symbols WHERE file_path = ?", stale_file_rows)
            conn.executemany("DELETE FROM relationships WHERE source_file = ?", stale_file_rows)
        if symbol_rows:
            conn.executemany(
                "INSERT OR REPLACE INTO symbols (name, qualified_name, type, file_path, line_start, line_end, docstring, signature) VALUES (?,?,?,?,?,?,?,?)",
                symbol_rows,
            )
        if relationship_rows:
            conn.executemany(
                "INSERT INTO relationships (source_name, target_name, rel_type, source_file, line_number) VALUES (?,?,?,?,?)",
                relationship_rows,
            )
        if file_rows:
            conn.executemany(
                "INSERT OR REPLACE INTO file_index (file_path, module_name, last_modified, line_count, symbol_count) VALUES (?,?,?,?,?)",
                file_rows,
            )
        conn.commit()

    # ── Queries ──────────────────────────────────────────────────────────

    def who_calls(self, function_name: str, limit: int = 20) -> list[dict[str, Any]]:
        """Find all callers of a function/method."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT source_name, source_file, line_number FROM relationships WHERE target_name = ? AND rel_type = ? LIMIT ?",
            (function_name, REL_CALLS, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def what_calls(self, function_name: str, limit: int = 20) -> list[dict[str, Any]]:
        """Find all functions called by a given function."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT target_name, source_file, line_number FROM relationships WHERE source_name = ? AND rel_type = ? LIMIT ?",
            (function_name, REL_CALLS, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def what_does(self, symbol_name: str) -> list[dict[str, Any]]:
        """Get info about a symbol (function, class, method)."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT name, qualified_name, type, file_path, line_start, line_end, docstring, signature FROM symbols WHERE name = ?",
            (symbol_name,),
        ).fetchall()
        return [dict(r) for r in rows]

    def dependencies_of(self, file_path: str) -> list[dict[str, Any]]:
        """Find all imports from a given file."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT target_name, line_number FROM relationships WHERE source_file = ? AND rel_type = ?",
            (file_path, REL_IMPORTS),
        ).fetchall()
        return [dict(r) for r in rows]

    def dependents_of(self, file_path: str) -> list[dict[str, Any]]:
        """Find all files that import from a given module."""
        conn = self._get_conn()
        module = file_path.replace("/", ".").replace(".py", "")
        rows = conn.execute(
            "SELECT source_file, source_name FROM relationships WHERE target_name LIKE ? AND rel_type = ?",
            (f"%{module}%", REL_IMPORTS),
        ).fetchall()
        return [dict(r) for r in rows]

    def who_inherits(self, class_name: str, limit: int = 20) -> list[dict[str, Any]]:
        """Find all classes that inherit from a given class."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT source_name, source_file, line_number FROM relationships WHERE target_name = ? AND rel_type = ? LIMIT ?",
            (class_name, REL_INHERITS, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def search_symbols(self, query: str, sym_type: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        """Search symbols by name (partial match)."""
        conn = self._get_conn()
        if sym_type:
            rows = conn.execute(
                "SELECT name, qualified_name, type, file_path, line_start, docstring FROM symbols WHERE name LIKE ? AND type = ? ORDER BY name LIMIT ?",
                (f"%{query}%", sym_type, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT name, qualified_name, type, file_path, line_start, docstring FROM symbols WHERE name LIKE ? ORDER BY name LIMIT ?",
                (f"%{query}%", limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_stats(self) -> dict[str, Any]:
        """Get graph statistics."""
        conn = self._get_conn()
        return {
            "files": conn.execute("SELECT COUNT(*) FROM file_index").fetchone()[0],
            "symbols": conn.execute("SELECT COUNT(*) FROM symbols").fetchone()[0],
            "relationships": conn.execute("SELECT COUNT(*) FROM relationships").fetchone()[0],
            "functions": conn.execute("SELECT COUNT(*) FROM symbols WHERE type = ?", (SYM_FUNCTION,)).fetchone()[0],
            "classes": conn.execute("SELECT COUNT(*) FROM symbols WHERE type = ?", (SYM_CLASS,)).fetchone()[0],
            "methods": conn.execute("SELECT COUNT(*) FROM symbols WHERE type = ?", (SYM_METHOD,)).fetchone()[0],
            "call_edges": conn.execute("SELECT COUNT(*) FROM relationships WHERE rel_type = ?", (REL_CALLS,)).fetchone()[0],
            "import_edges": conn.execute("SELECT COUNT(*) FROM relationships WHERE rel_type = ?", (REL_IMPORTS,)).fetchone()[0],
            "inherit_edges": conn.execute("SELECT COUNT(*) FROM relationships WHERE rel_type = ?", (REL_INHERITS,)).fetchone()[0],
            "built": self._built,
        }

    def hotspots(self, limit: int = 15) -> list[dict[str, Any]]:
        """Find most-called functions (highest in-degree)."""
        conn = self._get_conn()
        rows = conn.execute("""
            SELECT target_name, COUNT(*) as call_count
            FROM relationships WHERE rel_type = ?
            GROUP BY target_name ORDER BY call_count DESC LIMIT ?
        """, (REL_CALLS, limit)).fetchall()
        return [dict(r) for r in rows]

    def orphans(self, limit: int = 20) -> list[dict[str, Any]]:
        """Find functions never called by anything (potential dead code)."""
        conn = self._get_conn()
        rows = conn.execute("""
            SELECT s.name, s.file_path, s.line_start
            FROM symbols s
            LEFT JOIN relationships r ON r.target_name = s.name AND r.rel_type = 'calls'
            WHERE s.type IN ('function', 'method')
            AND r.id IS NULL
            AND s.name NOT LIKE '_%'
            AND s.name NOT IN ('main', 'setup', 'teardown', 'setUp', 'tearDown')
            ORDER BY s.file_path
            LIMIT ?
        """, (limit,)).fetchall()
        return [dict(r) for r in rows]

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _extract_signature(node: ast.FunctionDef) -> str:
        args = []
        for arg in node.args.args:
            ann = ""
            if arg.annotation:
                try:
                    ann = f": {ast.unparse(arg.annotation)}"
                except Exception:
                    pass  # no-op: intentional
            args.append(f"{arg.arg}{ann}")
        ret = ""
        if node.returns:
            try:
                ret = f" -> {ast.unparse(node.returns)}"
            except Exception:
                pass  # no-op: intentional
        return f"({', '.join(args)}){ret}"

    @staticmethod
    def _extract_call_name(node) -> str | None:
        if isinstance(node, ast.Call):
            return CodeGraph._get_name(node.func)
        return CodeGraph._get_name(node)

    @staticmethod
    def _get_name(node) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return node.attr
        elif isinstance(node, ast.Subscript):
            return CodeGraph._get_name(node.value)
        return None

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None


# ── Singleton ────────────────────────────────────────────────────────────────

_instance: CodeGraph | None = None


def get_code_graph() -> CodeGraph:
    global _instance
    if _instance is None:
        _instance = CodeGraph()
    return _instance
