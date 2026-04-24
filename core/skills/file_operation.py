import logging
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from core.skills.base_skill import BaseSkill


class FileOpInput(BaseModel):
    action: str = Field(..., description="Action to perform: 'read', 'write', 'append', 'list', 'exists', 'delete', 'move', 'copy', 'patch'")
    path: str = Field(..., description="Target file or directory path.")
    content: Optional[str] = Field(None, description="Content for write, append, or patch actions.")
    destination: Optional[str] = Field(None, description="Destination path for move or copy actions.")
    start_line: Optional[int] = Field(None, description="Starting line number for 'patch' action (inclusive, 1-indexed).")
    end_line: Optional[int] = Field(None, description="Ending line number for 'patch' action (inclusive, 1-indexed).")

class FileOperationSkill(BaseSkill):
    name = "file_operation"
    description = "Read, write, append, or list files in the allowed workspace."
    input_model = FileOpInput

    def __init__(self):
        self.logger = logging.getLogger(f"Skills.{self.name}")
        # Define allowable root (e.g. scratch dir or current dir)
        self.root_dir = os.path.realpath(os.getcwd())

    def _safe_resolve(self, path: str) -> str:
        """v5.0: SECURITY — Resolve path and enforce it stays within root_dir.
        Prevents path traversal via ../ or absolute paths.
        """
        if not path:
            return self.root_dir
        # Join relative to root and resolve symlinks
        if os.path.isabs(path):
            full = os.path.realpath(path)
        else:
            full = os.path.realpath(os.path.join(self.root_dir, path))
        # Check containment
        if not full.startswith(self.root_dir):
            raise PermissionError(f"Access denied: path '{path}' resolves outside workspace")
        return full

    def match(self, goal: Dict[str, Any]) -> bool:
        obj = goal.get("objective", "").lower()
        return "file" in obj or "read" in obj or "write" in obj or "save" in obj or "log" in obj

    async def execute(self, params: FileOpInput, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Standard execution entry point."""
        import asyncio
        if isinstance(params, dict):
            try:
                params = FileOpInput(**params)
            except Exception as e:
                return {"ok": False, "error": f"Invalid input: {e}"}
        
        action = params.action
        path = params.path
        content = params.content or ""

        if not action:
            return {"ok": False, "error": "Missing 'action' parameter (read, write, list, exists, delete, append, move, copy, patch)"}

        try:
            full_path = self._safe_resolve(path)
        except PermissionError as e:
            self.logger.warning("Path traversal blocked: %s", e)
            return {"ok": False, "error": str(e)}
            
        try:
            if action == "read":
                if not await asyncio.to_thread(os.path.exists, full_path):
                    return {"ok": False, "error": f"File not found: {path}", "path": path}
                
                def _read():
                    with open(full_path, "r", encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()
                        # Output semantic line-indexed text
                        indexed_lines = [f"{i+1:04d}: {line}" for i, line in enumerate(lines)]
                        return "".join(indexed_lines)
                
                data = await asyncio.to_thread(_read)
                return {"ok": True, "content": data[:60000], "truncated": len(data) > 60000, "path": path}
                
            elif action == "write":
                def _write():
                    with open(full_path, "w", encoding='utf-8') as f:
                        f.write(content)
                
                await asyncio.to_thread(_write)
                return {"ok": True, "summary": f"Wrote {len(content)} bytes to {path}", "path": path}
                
            elif action == "append":
                 def _append():
                     with open(full_path, "a", encoding='utf-8') as f:
                        f.write(content + "\n")
                 
                 await asyncio.to_thread(_append)
                 return {"ok": True, "summary": f"Appended to {path}", "path": path}
                 
            elif action == "list":
                if await asyncio.to_thread(os.path.isdir, full_path):
                    files = await asyncio.to_thread(os.listdir, full_path)
                    return {"ok": True, "files": files[:50], "path": path}
                else:
                    return {"ok": False, "error": "Path is not a directory", "path": path}

            elif action == "exists":
                exists = await asyncio.to_thread(os.path.exists, full_path)
                is_dir = await asyncio.to_thread(os.path.isdir, full_path) if exists else False
                kind = "directory" if is_dir else "file"
                summary = f"{path} exists." if exists else f"{path} does not exist."
                return {
                    "ok": True,
                    "path": path,
                    "exists": exists,
                    "kind": kind if exists else None,
                    "state": "present" if exists else "missing",
                    "summary": summary,
                }

            elif action == "delete":
                if await asyncio.to_thread(os.path.exists, full_path):
                    # Double-check containment before destructive ops
                    if not os.path.realpath(full_path).startswith(self.root_dir):
                        return {"ok": False, "error": "Delete blocked: path outside workspace"}
                    
                    is_dir = await asyncio.to_thread(os.path.isdir, full_path)
                    if is_dir:
                        # Safety: refuse to delete root or top-level dirs
                        if os.path.realpath(full_path) == self.root_dir:
                            return {"ok": False, "error": "Cannot delete workspace root", "path": path}
                        await asyncio.to_thread(shutil.rmtree, full_path)
                    else:
                        await asyncio.to_thread(os.remove, full_path)
                    return {"ok": True, "summary": f"Deleted {path}", "path": path}
                return {"ok": False, "error": "File not found", "path": path}

            elif action == "move":
                dest_path = params.destination
                if not dest_path:
                    return {"ok": False, "error": "Missing 'destination' for move action"}
                try:
                    full_dest = self._safe_resolve(dest_path)
                except PermissionError as e:
                    return {"ok": False, "error": str(e)}
                
                await asyncio.to_thread(shutil.move, full_path, full_dest)
                return {"ok": True, "summary": f"Moved {path} to {dest_path}", "path": path, "destination": dest_path}

            elif action == "copy":
                dest_path = params.destination
                if not dest_path:
                    return {"ok": False, "error": "Missing 'destination' for copy action"}
                try:
                    full_dest = self._safe_resolve(dest_path)
                except PermissionError as e:
                    return {"ok": False, "error": str(e)}
                
                is_dir = await asyncio.to_thread(os.path.isdir, full_path)
                if is_dir:
                    await asyncio.to_thread(shutil.copytree, full_path, full_dest)
                else:
                    await asyncio.to_thread(shutil.copy2, full_path, full_dest)
                return {"ok": True, "summary": f"Copied {path} to {dest_path}", "path": path, "destination": dest_path}

            elif action == "patch":
                start_line = params.start_line
                end_line = params.end_line
                replacement = params.content
                if start_line is None or end_line is None or replacement is None:
                    return {"ok": False, "error": "Missing 'start_line', 'end_line', or 'content' for patch action"}
                
                def _patch():
                    with open(full_path, "r", encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()
                    
                    if start_line < 1 or end_line > len(lines) or start_line > end_line:
                        raise ValueError(f"Invalid line range [{start_line}, {end_line}] for file with {len(lines)} lines")
                    
                    new_lines = replacement.splitlines(keepends=True)
                    # Ensure last line has a newline if original had one
                    if new_lines and not new_lines[-1].endswith("\n"):
                        new_lines[-1] += "\n"
                        
                    lines[start_line - 1 : end_line] = new_lines
                    new_data = "".join(lines)
                    
                    # Syntax validation pre-commit
                    if full_path.endswith(".py"):
                        import py_compile
                        with open(full_path + ".tmp", "w", encoding='utf-8') as f:
                            f.write(new_data)
                        try:
                            py_compile.compile(full_path + ".tmp", doraise=True)
                        except py_compile.PyCompileError as e:
                            os.remove(full_path + ".tmp")
                            raise ValueError(f"Syntax Error introduced by patch: {e}")
                        os.remove(full_path + ".tmp")
                    elif full_path.endswith(".json"):
                        import json
                        try:
                            json.loads(new_data)
                        except json.JSONDecodeError as e:
                            raise ValueError(f"JSON Syntax Error introduced by patch: {e}")
                    
                    with open(full_path, "w", encoding='utf-8') as f:
                        f.write(new_data)
                    return new_data

                try:
                    await asyncio.to_thread(_patch)
                    return {"ok": True, "summary": f"Patched {path}: Replaced lines {start_line}-{end_line}", "path": path}
                except ValueError as ve:
                    return {"ok": False, "error": str(ve), "path": path}

        except Exception as e:
            self.logger.error("File Op failed: %s", e)
            return {"ok": False, "error": str(e), "path": path}
