"""ASTAnalyzer: Structural Self-Awareness for Aura
Provides deep analysis of Python source code via Abstract Syntax Trees.
"""
import ast
import logging
from pathlib import Path
import asyncio
from typing import Dict, List, Any, Optional, Set

logger = logging.getLogger("SelfEvolution.ASTAnalyzer")

class ASTAnalyzer:
    """Analyzes Python code to extract architectural patterns and smells."""

    def __init__(self, project_root: Optional[Path] = None):
        from core.config import config
        self.root = project_root or config.paths.project_root
        self._parent_map: Dict[ast.AST, ast.AST] = {}

    async def analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """Performs a comprehensive structural audit of a file (Async)."""
        if not file_path.is_absolute():
            file_path = self.root / file_path

        try:
            source = await asyncio.to_thread(file_path.read_text, encoding='utf-8')
            tree = ast.parse(source)
            self._build_parent_map(tree) # SM-02: Build map once per file
        except Exception as e:
            logger.error("Failed to parse %s: %s", file_path, e)
            return {"error": str(e)}

        results = {
            "classes": self._get_classes(tree),
            "functions": self._get_functions(tree),
            "imports": self._get_imports(tree),
            "smells": self._detect_smells(tree, source),
            "complexity": self._calculate_complexity(tree)
        }
        return results

    def _get_classes(self, tree: ast.AST) -> List[Dict[str, Any]]:
        classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append({
                    "name": node.name,
                    "methods": [m.name for m in node.body if isinstance(m, ast.FunctionDef)],
                    "bases": [ast.unparse(b) if hasattr(ast, 'unparse') else "unknown" for b in node.bases],
                    "lineno": node.lineno
                })
        return classes

    def _get_functions(self, tree: ast.AST) -> List[Dict[str, Any]]:
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Skip methods (already in classes)
                is_method = any(isinstance(p, ast.ClassDef) for p in self._get_parents(node))
                if not is_method:
                    functions.append({
                        "name": node.name,
                        "args": [arg.arg for arg in node.args.args],
                        "is_async": isinstance(node, ast.AsyncFunctionDef),
                        "lineno": node.lineno
                    })
        return functions

    def _get_imports(self, tree: ast.AST) -> List[str]:
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
        return list(set(imports))

    def _detect_smells(self, tree: ast.AST, source: str) -> List[Dict[str, Any]]:
        smells = []
        
        # 1. Monkey-Patching Sensor
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == "setattr":
                    smells.append({
                        "type": "monkey_patch",
                        "severity": "high",
                        "message": "Dynamic attribute setting (setattr) detected. Architectural debt risk.",
                        "lineno": node.lineno
                    })
                elif isinstance(node.func, ast.Attribute) and node.func.attr == "update":
                    if isinstance(node.func.value, ast.Attribute) and node.func.value.attr == "__dict__":
                        smells.append({
                            "type": "monkey_patch",
                            "severity": "critical",
                            "message": "Manual __dict__ update detected. Dangerous architectural subversion.",
                            "lineno": node.lineno
                        })

        # 2. Async-Await & Stall Sensor
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                has_async = isinstance(node, ast.AsyncFunctionDef)
                has_await_or_sleep = False
                
                for subnode in ast.walk(node):
                    if not has_async and isinstance(subnode, ast.Await):
                        smells.append({
                            "type": "async_mismatch",
                            "severity": "high",
                            "message": f"'await' used in non-async function '{node.name}'",
                            "lineno": subnode.lineno
                        })
                    
                    if isinstance(subnode, ast.Await):
                        has_await_or_sleep = True
                    
                    # Detect common sync blockers in async def
                    if has_async and isinstance(subnode, ast.Call):
                        func_name = ast.unparse(subnode.func) if hasattr(ast, 'unparse') else ""
                        if any(blocker in func_name for blocker in ["time.sleep", "requests.get", "subprocess.run"]):
                            smells.append({
                                "type": "blocking_call",
                                "severity": "medium",
                                "message": f"Potential blocking call '{func_name}' in async function '{node.name}'",
                                "lineno": subnode.lineno
                            })
                        if "asyncio.sleep" in func_name:
                            has_await_or_sleep = True

                # 3. Deadlock Sensor (Nested Locks) - SM-01 FIX: Scoped Traversal
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    smells.extend(self._check_nested_locks(node))

        # 4. Resource Leak Sensor
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == "open":
                    # Check if it's inside a 'with' statement
                    is_in_with = False
                    parents = self._get_parents(node)
                    if any(isinstance(p, ast.With) for p in parents):
                        is_in_with = True
                    
                    if not is_in_with:
                        smells.append({
                            "type": "resource_leak",
                            "severity": "medium",
                            "message": "Unmanaged 'open()' call detected. Use 'with' statement for resource safety.",
                            "lineno": node.lineno
                        })

        return smells

    def _calculate_complexity(self, tree: ast.AST) -> Dict[str, int]:
        """Calculates basic cyclomatic complexity equivalents."""
        complexity = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                score = 1
                for sub in ast.walk(node):
                    if isinstance(sub, (ast.If, ast.For, ast.While, ast.And, ast.Or, ast.ExceptHandler)):
                        score += 1
                complexity[node.name] = score
        return complexity

    def _build_parent_map(self, tree: ast.AST):
        """SM-02: Pre-calculate parent relationships for O(1) lookup."""
        self._parent_map = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                self._parent_map[child] = node

    def _get_parents(self, target_node: ast.AST) -> List[ast.AST]:
        """SM-02: Optimized parent lookup using the pre-built map."""
        parents = []
        curr = target_node
        while curr in self._parent_map:
            curr = self._parent_map[curr]
            parents.append(curr)
        return parents

    def _check_nested_locks(self, func_node: ast.AST) -> List[Dict]:
        """SM-01: Correct nested lock detection using scoped recursive visitor."""
        findings = []
        
        def _visit_with_scope(nodes, active_locks: set):
            for node in nodes:
                if isinstance(node, ast.With):
                    acquired = set()
                    for item in node.items:
                        lock_name = ast.unparse(item.context_expr) if hasattr(ast, 'unparse') else ""
                        if "lock" in lock_name.lower():
                            if lock_name in active_locks:
                                findings.append({
                                    "type": "deadlock_risk",
                                    "severity": "critical",
                                    "message": f"Nested acquisition of SAME lock '{lock_name}' in '{func_node.name}'",
                                    "lineno": node.lineno
                                })
                            acquired.add(lock_name)
                    # Recurse into the WITH body with the newly acquired locks in scope
                    _visit_with_scope(node.body, active_locks | acquired)
                
                # Broaden to other nodes that have body blocks (If, For, While, Try)
                elif hasattr(node, 'body') and isinstance(node.body, list):
                    _visit_with_scope(node.body, active_locks)
                    if hasattr(node, 'orelse') and isinstance(node.orelse, list):
                        _visit_with_scope(node.orelse, active_locks)
                    if hasattr(node, 'finalbody') and isinstance(node.finalbody, list):
                        _visit_with_scope(node.finalbody, active_locks)
        
        _visit_with_scope(func_node.body, set())
        return findings

    async def map_repository(self) -> Dict[str, Any]:
        """Scans the entire core codebase to build an architectural map (Async)."""
        repo_map = {}
        core_dir = self.root / "core"
        if not core_dir.exists():
            return {"error": "Core directory not found"}

        py_files = await asyncio.to_thread(list, core_dir.rglob("*.py"))
        for py_file in py_files:
            rel_path = py_file.relative_to(self.root)
            repo_map[str(rel_path)] = await self.analyze_file(py_file)
            
        return repo_map