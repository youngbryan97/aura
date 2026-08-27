from core.runtime.errors import record_degradation
from core.runtime.action_executor import ActionExecutor
from core.governance.will import ActionDomain
import contextlib
import hashlib
import logging
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Literal, Optional
from pydantic import BaseModel, Field

from core.skills.base_skill import BaseSkill


#: The actions this skill performs. Declared once, so the schema, the
#: governance check and the code that dispatches them cannot disagree.
FILE_ACTIONS: tuple[str, ...] = (
    "read", "list", "exists", "write", "append", "patch", "move", "copy", "delete",
)


class FileOpInput(BaseModel):
    # An unenumerated required string makes the caller guess. These nine were
    # named only inside a description, so the generated tool schema carried
    # `type: string, enum: None` and the model had to invent an action name.
    action: Literal[
        "read", "list", "exists", "write", "append", "patch", "move", "copy", "delete"
    ] = Field(..., description="Action to perform.")
    path: str = Field(..., description="Target file or directory path.")
    content: Optional[str] = Field(None, description="Content for write, append, or patch actions.")
    destination: Optional[str] = Field(None, description="Destination path for move or copy actions.")
    start_line: Optional[int] = Field(None, description="Starting line number for 'patch' action (inclusive, 1-indexed).")
    end_line: Optional[int] = Field(None, description="Ending line number for 'patch' action (inclusive, 1-indexed).")

#: Actions that observe and change nothing.
_READING_ACTIONS = frozenset({"read", "list", "exists", "stat", "head", "tail"})


def _the_person_named_this_path(path: str) -> bool:
    """Whether this exact path appears in what the person said this turn.

    The authorisation is theirs and it is literal: the path has to be in their
    own words, not derived, expanded or guessed at by anything downstream.
    """
    wanted = str(path or "").strip()
    if not wanted:
        return False
    try:
        from core.conversation.session_scope import the_persons_own_words

        said = str(the_persons_own_words("") or "")
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        return False
    if not said:
        return False
    # Identity or containment, never substring.
    #
    # Testing whether the text appears in the sentence authorised every PARENT
    # of a named path, because a parent is a prefix of it — so naming anything
    # at all authorised "/". What a person authorises by naming a place is that
    # place and what is inside it.
    try:
        from core.language.named_paths import named_paths

        target = Path(os.path.realpath(wanted))
        for candidate in named_paths(said):
            named = Path(os.path.realpath(candidate))
            if target == named or target.is_relative_to(named):
                return True
    except (OSError, ValueError, ImportError):
        return False
    return False


class FileOperationSkill(BaseSkill):
    name = "file_operation"
    description = "Read, write, append, or list files in the allowed workspace."

    #: What each action actually does.
    #:
    #: The skill's own effect_scope is state_mutation, because it CAN delete.
    #: Scoping the whole skill by its worst action meant reading a file
    #: required permission to destroy one — and every real task begins by
    #: reading something, so the cheapest safe step in computing sat behind
    #: the most dangerous grant in the system.
    ACTION_EFFECT_SCOPES = {
        "read": "read_only",
        "list": "read_only",
        "exists": "read_only",
        "write": "state_mutation",
        "append": "state_mutation",
        "patch": "state_mutation",
        "move": "state_mutation",
        "copy": "state_mutation",
        "delete": "state_mutation",
    }
    input_model = FileOpInput

    def __init__(self):
        self.logger = logging.getLogger(f"Skills.{self.name}")
        # Define allowable root (e.g. scratch dir or current dir)
        self.root_dir = os.path.realpath(os.getcwd())

    def _is_within_root(self, full_path: str) -> bool:
        root = os.path.realpath(self.root_dir)
        target = os.path.realpath(full_path)
        try:
            return os.path.commonpath([root, target]) == root
        except ValueError:
            return False

    def _safe_resolve(self, path: str, *, for_reading: bool = False) -> str:
        """Resolve path and enforce it stays within root_dir.

        Prevents path traversal via ../ or absolute paths to unauthorized areas.

        One exception, and only for reading: a path the PERSON typed in their
        own message this turn. They named it, which is the authorisation, and
        it is the same reason `diagnose_repo` may be aimed at any directory
        somebody names — a capability this tree gained precisely because
        nothing could look at a project a person pointed at.

        LIVE, 2026-08-27: "docs and source are at <path>. Read it, then
        actually use it" was refused with "resolves outside workspace" for a
        directory in the person's own sentence. Writing outside the workspace
        stays refused, because naming a place to read is not asking for it to
        be changed.
        """
        if not path:
            return os.path.realpath(self.root_dir)
        # Join relative to root and resolve symlinks
        if os.path.isabs(path):
            full = os.path.realpath(path)
        else:
            full = os.path.realpath(os.path.join(self.root_dir, path))

        # Check containment
        if not self._is_within_root(full):
            if for_reading and _the_person_named_this_path(path):
                self.logger.info(
                    "Reading %s: outside the workspace, and the person named it.", path
                )
                return full
            raise PermissionError(f"Access denied: path '{path}' resolves outside workspace")
        return full

    @staticmethod
    def _sha256_text(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @staticmethod
    def _file_sha256(path: str) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @classmethod
    async def _file_effect(cls, full_path: str, *, expected_sha256: str = "") -> dict[str, Any]:
        import asyncio

        exists = await asyncio.to_thread(os.path.exists, full_path)
        is_file = await asyncio.to_thread(os.path.isfile, full_path) if exists else False
        evidence: dict[str, Any] = {
            "exists": exists,
            "is_file": is_file,
            "effect_verified": exists,
        }
        if is_file:
            sha256 = await asyncio.to_thread(cls._file_sha256, full_path)
            size = await asyncio.to_thread(os.path.getsize, full_path)
            evidence.update({"sha256": sha256, "bytes": size})
            if expected_sha256:
                evidence["expected_sha256"] = expected_sha256
                evidence["effect_verified"] = sha256 == expected_sha256
        elif expected_sha256:
            evidence["expected_sha256"] = expected_sha256
            evidence["effect_verified"] = False
        return evidence

    def match(self, goal: Dict[str, Any]) -> bool:
        obj = goal.get("objective", "").lower()
        return "file" in obj or "read" in obj or "write" in obj or "save" in obj or "log" in obj

    async def execute(self, params: FileOpInput, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Standard execution entry point."""
        import asyncio
        if isinstance(params, dict):
            try:
                params = FileOpInput(**params)
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                record_degradation('file_operation', e)
                return {"ok": False, "error": f"Invalid input: {e}"}
        
        action = params.action
        path = params.path
        content = params.content or ""

        if not action:
            return {"ok": False, "error": "Missing 'action' parameter (read, write, list, exists, delete, append, move, copy, patch)"}

        # Only the actions that change nothing may reach outside the
        # workspace, and only for a path the person typed themselves.
        reading = str(action).strip().lower() in _READING_ACTIONS
        try:
            full_path = self._safe_resolve(path, for_reading=reading)
        except PermissionError as e:
            self.logger.warning("Path traversal blocked: %s", e)
            return {"ok": False, "error": str(e)}
        
        # [CONSENT INTEGRATION] Check for sensitive operations
        try:
            from core.consent import get_consent_workflow
            workflow = get_consent_workflow()
            
            # Classify operation sensitivity
            if action == "delete":
                operation_type = "delete_file"
            elif action == "write" and ("/System" in full_path or "/etc" in full_path):
                operation_type = "write_system_file"
            elif action == "write":
                operation_type = "write_file_downloads" if "Downloads" in full_path else "write_file_home"
            elif action == "read" and ("/System" in full_path or "/etc" in full_path):
                operation_type = "read_system_file"
            else:
                operation_type = None
            
            # Check consent if operation is sensitive
            if operation_type:
                has_consent = await workflow.check_consent(
                    operation_type,
                    {"path": path, "action": action}
                )
                if not has_consent:
                    return {
                        "ok": False,
                        "error": f"Operation {action} blocked: requires user approval for {path}",
                        "status": "consent_denied"
                    }
        except (ImportError, AttributeError) as exc:
            self.logger.debug("Consent workflow unavailable for file operation: %s", exc)
            
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
                result = await ActionExecutor.execute(
                    domain=ActionDomain.FILE_WRITE,
                    action_name="file_operation.write",
                    params={"path": full_path, "text": content},
                    source="file_operation",
                )
                if not result.get("ok"):
                    return {"ok": False, "error": result.get("error", "write failed"), "path": path}
                expected_sha256 = self._sha256_text(content)
                effect = await self._file_effect(full_path, expected_sha256=expected_sha256)
                return {
                    "ok": bool(effect.get("effect_verified")),
                    "summary": f"Wrote {len(content)} bytes to {path}",
                    "path": path,
                    "criteria_results": {"file written": bool(effect.get("effect_verified"))},
                    **effect,
                }

            elif action == "append":
                existing = ""
                if await asyncio.to_thread(os.path.exists, full_path):
                    def _read_existing():
                        with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                            return f.read()

                    existing = await asyncio.to_thread(_read_existing)
                next_text = existing + content + "\n"
                result = await ActionExecutor.execute(
                    domain=ActionDomain.FILE_WRITE,
                    action_name="file_operation.append",
                    params={"path": full_path, "text": next_text},
                    source="file_operation",
                )
                if not result.get("ok"):
                    return {"ok": False, "error": result.get("error", "append failed"), "path": path}
                expected_sha256 = self._sha256_text(next_text)
                effect = await self._file_effect(full_path, expected_sha256=expected_sha256)
                return {
                    "ok": bool(effect.get("effect_verified")),
                    "summary": f"Appended to {path}",
                    "path": path,
                    "criteria_results": {"file appended": bool(effect.get("effect_verified"))},
                    **effect,
                }

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
                    if not self._is_within_root(full_path):
                        return {"ok": False, "error": "Delete blocked: path outside workspace"}

                    is_dir = await asyncio.to_thread(os.path.isdir, full_path)
                    real_target = await asyncio.to_thread(os.path.realpath, full_path)
                    if is_dir and real_target == self.root_dir:
                        # Safety: refuse to delete root or top-level dirs
                        return {"ok": False, "error": "Cannot delete workspace root", "path": path}
                    result = await ActionExecutor.execute(
                        domain=ActionDomain.FILE_WRITE,
                        action_name="file_operation.delete",
                        params={"op": "delete", "path": full_path, "recursive": is_dir},
                        source="file_operation",
                    )
                    if not result.get("ok"):
                        return {"ok": False, "error": result.get("error", "delete failed"), "path": path}
                    exists_after = await asyncio.to_thread(os.path.exists, full_path)
                    effect_verified = not exists_after
                    return {
                        "ok": effect_verified,
                        "summary": f"Deleted {path}",
                        "path": path,
                        "exists": exists_after,
                        "effect_verified": effect_verified,
                        "criteria_results": {"path deleted": effect_verified},
                    }
                return {"ok": False, "error": "File not found", "path": path}

            elif action == "move":
                dest_path = params.destination
                if not dest_path:
                    return {"ok": False, "error": "Missing 'destination' for move action"}
                try:
                    full_dest = self._safe_resolve(dest_path)
                except PermissionError as e:
                    return {"ok": False, "error": str(e)}

                result = await ActionExecutor.execute(
                    domain=ActionDomain.FILE_WRITE,
                    action_name="file_operation.move",
                    params={"op": "move", "path": full_path, "destination": full_dest},
                    source="file_operation",
                )
                if not result.get("ok"):
                    return {"ok": False, "error": result.get("error", "move failed"), "path": path}
                source_exists = await asyncio.to_thread(os.path.exists, full_path)
                effect = await self._file_effect(full_dest)
                effect_verified = bool(effect.get("effect_verified")) and not source_exists
                return {
                    "ok": effect_verified,
                    "summary": f"Moved {path} to {dest_path}",
                    "path": path,
                    "destination": dest_path,
                    "source_exists": source_exists,
                    "criteria_results": {"path moved": effect_verified},
                    **effect,
                    "effect_verified": effect_verified,
                }

            elif action == "copy":
                dest_path = params.destination
                if not dest_path:
                    return {"ok": False, "error": "Missing 'destination' for copy action"}
                try:
                    full_dest = self._safe_resolve(dest_path)
                except PermissionError as e:
                    return {"ok": False, "error": str(e)}

                result = await ActionExecutor.execute(
                    domain=ActionDomain.FILE_WRITE,
                    action_name="file_operation.copy",
                    params={"op": "copy", "path": full_path, "destination": full_dest},
                    source="file_operation",
                )
                if not result.get("ok"):
                    return {"ok": False, "error": result.get("error", "copy failed"), "path": path}
                source_effect = await self._file_effect(full_path)
                dest_effect = await self._file_effect(full_dest)
                effect_verified = bool(dest_effect.get("effect_verified"))
                if source_effect.get("sha256") and dest_effect.get("sha256"):
                    effect_verified = source_effect["sha256"] == dest_effect["sha256"]
                return {
                    "ok": effect_verified,
                    "summary": f"Copied {path} to {dest_path}",
                    "path": path,
                    "destination": dest_path,
                    "source_sha256": source_effect.get("sha256", ""),
                    "criteria_results": {"path copied": effect_verified},
                    **dest_effect,
                    "effect_verified": effect_verified,
                }

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
                        tmp_path = ""
                        try:
                            fd, tmp_path = tempfile.mkstemp(suffix=".py", prefix="aura_patch_")
                            os.close(fd)
                            from core.runtime.atomic_writer import atomic_write_text

                            atomic_write_text(tmp_path, new_data, encoding="utf-8")
                            py_compile.compile(tmp_path, doraise=True)
                        except py_compile.PyCompileError as e:
                            raise ValueError(f"Syntax Error introduced by patch: {e}")
                        finally:
                            if tmp_path:
                                with contextlib.suppress(OSError):
                                    os.remove(tmp_path)
                    elif full_path.endswith(".json"):
                        import json
                        try:
                            json.loads(new_data)
                        except json.JSONDecodeError as e:
                            raise ValueError(f"JSON Syntax Error introduced by patch: {e}")
                    return new_data

                try:
                    new_content = await asyncio.to_thread(_patch)
                    result = await ActionExecutor.execute(
                        domain=ActionDomain.FILE_WRITE,
                        action_name="file_operation.patch",
                        params={"path": full_path, "text": new_content},
                        source="file_operation",
                    )
                    if not result.get("ok"):
                        return {"ok": False, "error": result.get("error", "patch write failed"), "path": path}
                    expected_sha256 = self._sha256_text(new_content)
                    effect = await self._file_effect(full_path, expected_sha256=expected_sha256)
                    return {
                        "ok": bool(effect.get("effect_verified")),
                        "summary": f"Patched {path}: Replaced lines {start_line}-{end_line}",
                        "path": path,
                        "criteria_results": {"file patched": bool(effect.get("effect_verified"))},
                        **effect,
                    }
                except ValueError as ve:
                    return {"ok": False, "error": str(ve), "path": path}

        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation('file_operation', e)
            self.logger.error("File Op failed: %s", e)
            return {"ok": False, "error": str(e), "path": path}
