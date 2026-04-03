"""InternalRepositorySearch: Deep Technical Navigation for Aura
Allows Aura to grep her own source code and map dependencies autonomously.
"""
import asyncio
import os
import re
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Set

logger = logging.getLogger("SelfEvolution.RepoSearch")

class InternalRepositorySearch:
    """A tool for high-precision searching and dependency mapping within Aura."""

    def __init__(self, project_root: Optional[Path] = None):
        from core.config import config
        self.root = project_root or config.paths.project_root

    async def grep(self, pattern: str, includes: List[str] = None, excludes: List[str] = None, 
                   exclude_file: Optional[str] = None, multiline: bool = False) -> List[Dict[str, Any]]:
        """Search the repository for a specific pattern with technical context (Async)."""
        flags = re.MULTILINE | re.DOTALL if multiline else 0
        try:
            regex = re.compile(pattern, flags)
        except re.error as e:
            logger.error("Invalid regex pattern '%s': %s", pattern, e)
            return []
            
        results = []
        includes = includes or ["*.py", "*.js", "*.html", "*.css", "*.md"]
        excludes = excludes or [".git", "__pycache__", "node_modules", ".venv", "dist", "build"]

        def _sync_search():
            for root, dirs, files in os.walk(self.root):
                dirs[:] = [d for d in dirs if d not in excludes]
                for file in files:
                    if not any(file.endswith(ext.replace("*", "")) for ext in includes):
                        continue
                    
                    file_path = Path(root) / file
                    # Issue 72: Self-dependency suppression
                    rel_path = str(file_path.relative_to(self.root))
                    if exclude_file and rel_path == exclude_file:
                        continue
                        
                    try:
                        content = file_path.read_text(encoding='utf-8', errors='ignore')
                        if multiline:
                            for match in regex.finditer(content):
                                line_no = content.count('\n', 0, match.start()) + 1
                                results.append({
                                    "file": rel_path,
                                    "line": line_no,
                                    "content": match.group(0).strip(),
                                    "snippet": self._get_sync_snippet(file_path, line_no)
                                })
                        else:
                            for i, line in enumerate(content.splitlines(), 1):
                                if regex.search(line):
                                    results.append({
                                        "file": rel_path,
                                        "line": i,
                                        "content": line.strip(),
                                        "snippet": self._get_sync_snippet(file_path, i)
                                    })
                    except Exception as e:
                        logger.debug("Grepping failed for %s: %s", file_path, e)

        await asyncio.to_thread(_sync_search)
        return results

    def _get_sync_snippet(self, file_path: Path, line_no: int, context: int = 2) -> str:
        """Synchronous version of snippet helper for use in thread."""
        try:
            lines = file_path.read_text(encoding='utf-8', errors='ignore').splitlines()
            start = max(0, line_no - context - 1)
            end = min(len(lines), line_no + context)
            return "\n".join(lines[start:end])
        except Exception:
            return ""

    def _get_snippet(self, file_path: Path, line_no: int, context: int = 2) -> str:
        """Helper to get a small context snippet around a match."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            start = max(0, line_no - context - 1)
            end = min(len(lines), line_no + context)
            return "".join(lines[start:end])
        except Exception:
            return ""

    async def map_dependencies(self, module_name: str) -> Dict[str, Any]:
        """Maps what other modules depend on a given module and what it depends on."""
        dependents = []
        dependencies = []
        
        # 1. Find what THIS module depends on
        from .ast_analyzer import ASTAnalyzer
        analyzer = ASTAnalyzer(self.root)
        
        # Resolve module name to path
        module_path = self.root / (module_name.replace(".", "/") + ".py")
        if module_path.exists():
            analysis = await analyzer.analyze_file(module_path)
            dependencies = analysis.get("imports", [])

        # 2. Find what depends on THIS module (Grep for imports)
        # Search for 'from module_name import' or 'import module_name'
        search_pattern = rf"(from {re.escape(module_name)} import|import {re.escape(module_name)})"
        grep_results = await self.grep(search_pattern, exclude_file=str(module_path.relative_to(self.root)) if module_path.exists() else None)
        
        for res in grep_results:
            dependents.append(res["file"])

        return {
            "module": module_name,
            "dependencies": dependencies,
            "dependents": list(set(dependents))
        }

    async def audit_architectural_footprint(self) -> Dict[str, List[str]]:
        """Identifies pervasive patterns across the entire system (Async)."""
        indicators = {
            "monkey_patches": await self.grep(r"setattr\(.*,") ,
            "hardcoded_paths": await self.grep(r"['\"]/Users/.*['\"]|['\"]C:\\.*['\"]"),
            "sync_io_in_async": await self.grep(r"async def.*[\s\S]*?(time\.sleep|requests\.get|subprocess\.run)", multiline=True)
        }
        
        report = {}
        for key, results in indicators.items():
            report[key] = sorted(list(set(r["file"] for r in results)))
            
        return report