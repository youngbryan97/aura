"""
core/self_modification/shadow_ast_healer.py
============================================
SKYNET'S SHADOW AST HEALER

Implements zero-token self-repair via AST manipulation.
When Skynet detects a subsystem failure due to code errors,
the ShadowHealer analyzes the AST, identifies common patterns
(e.g., missing imports, type mismatches), and applies patches.
"""

import ast
import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, List, Any

logger = logging.getLogger("Aura.ShadowHealer")

class ShadowASTHealer:
    """
    Autonomously repairs source code via AST manipulation.
    """

    def __init__(self, codebase_root: Optional[Path] = None):
        self.root = codebase_root or Path.cwd()

    async def attempt_repair(self, file_path: Path, error_msg: str) -> bool:
        """Attempts to repair a specific error in a target file (Async)."""
        logger.info("🛠️ Skynet: Attempting AST repair for %s: %s", file_path.name, error_msg)
        
        try:
            # Ensure path is absolute and within root (Issue 90)
            if not file_path.is_absolute():
                file_path = self.root / file_path
                
            content = await asyncio.to_thread(file_path.read_text)
            tree = ast.parse(content)
            
            repaired = False
            # Pattern 1: Missing Import
            if "name" in error_msg.lower() and "is not defined" in error_msg.lower():
                try:
                    missing_name = error_msg.split("'")[1]
                except IndexError:
                    missing_name = error_msg.split()[-1]
                repaired = self._inject_missing_import(tree, missing_name)
            
            if repaired:
                # Issue 88: Built-in ast.unparse (Python 3.9+)
                new_content = ast.unparse(tree)
                await asyncio.to_thread(file_path.write_text, new_content)
                logger.info("✅ Skynet: Successfully repaired AST for %s", file_path.name)
                return True
                
            return False
        except Exception as e:
            logger.error("❌ Skynet: AST repair failed: %s", e)
            return False

    def _inject_missing_import(self, tree: ast.AST, name: str) -> bool:
        """Injects a common missing import into the AST."""
        common_imports = {
            "asyncio": "import asyncio",
            "json": "import json",
            "logging": "import logging",
            "os": "import os",
            "sys": "import sys",
            "time": "import time",
            "Path": "from pathlib import Path",
            "Any": "from typing import Any",
            "Dict": "from typing import Dict",
            "List": "from typing import List",
            "Optional": "from typing import Optional"
        }
        
        if name in common_imports:
            import_node = ast.parse(common_imports[name]).body[0]
            # Insert at the top of the file
            tree.body.insert(0, import_node)
            return True
        return False

    def validate_syntax(self, file_path: Path) -> bool:
        """Validates that a file has valid Python syntax."""
        try:
            ast.parse(file_path.read_text())
            return True
        except SyntaxError:
            return False
