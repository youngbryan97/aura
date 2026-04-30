"""Durable Markdown workspace with content-addressed versioning.

This is an Aura-native workspace layer for agent-authored Markdown artifacts:
small enough to embed in the runtime, explicit enough to audit, and compatible
with agent workflows that need search, commits, bookmarks, rollback, and
non-blocking merge conflicts.
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import posixpath
import threading
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from core.runtime.atomic_writer import atomic_write_text
from core.runtime.errors import record_degradation


@dataclass(frozen=True)
class FileNode:
    content_hash: str
    owner: str = "aura"
    group: str = "aura"
    mode: int = 0o644
    updated_at: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FileNode:
        return cls(
            content_hash=str(data["content_hash"]),
            owner=str(data.get("owner", "aura")),
            group=str(data.get("group", "aura")),
            mode=int(data.get("mode", 0o644)),
            updated_at=float(data.get("updated_at", 0.0)),
        )


@dataclass(frozen=True)
class WorkspaceCommit:
    change_id: str
    message: str
    author: str
    parents: tuple[str, ...]
    tree: dict[str, dict[str, Any]]
    timestamp: float
    bookmark: str = "main"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkspaceCommit:
        return cls(
            change_id=str(data["change_id"]),
            message=str(data.get("message", "")),
            author=str(data.get("author", "aura")),
            parents=tuple(str(p) for p in data.get("parents", ())),
            tree={str(k): dict(v) for k, v in data.get("tree", {}).items()},
            timestamp=float(data.get("timestamp", 0.0)),
            bookmark=str(data.get("bookmark", "main")),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class ConflictRecord:
    path: str
    base_hash: str | None
    target_hash: str | None
    source_hash: str | None


@dataclass(frozen=True)
class MergeResult:
    merge_type: str
    change_id: str | None = None
    conflicts: tuple[ConflictRecord, ...] = ()

    @property
    def conflicted(self) -> bool:
        return bool(self.conflicts)


class MarkdownWorkspace:
    """A Markdown-only virtual workspace with Git/JJ-like versioning."""

    schema_version = 1

    def __init__(
        self,
        storage_path: str | Path | None = None,
        *,
        default_user: str = "aura",
    ) -> None:
        if storage_path is None:
            try:
                from core.config import config

                storage_path = config.paths.data_dir / "markdown_workspace" / "workspace.json"
            except Exception:
                storage_path = Path.home() / ".aura" / "data" / "markdown_workspace" / "workspace.json"
        self.storage_path = Path(storage_path)
        self.default_user = default_user
        self._lock = threading.RLock()
        self._files: dict[str, FileNode] = {}
        self._blobs: dict[str, str] = {}
        self._commits: dict[str, WorkspaceCommit] = {}
        self._bookmarks: dict[str, str] = {}
        self._conflicts: dict[str, list[ConflictRecord]] = {}
        self._groups: dict[str, set[str]] = {default_user: {default_user}}
        self._load()

    def write_file(
        self,
        path: str,
        content: str,
        *,
        user: str | None = None,
        mode: int | None = None,
    ) -> FileNode:
        user = user or self.default_user
        clean = self._normalize_file_path(path)
        with self._lock:
            existing = self._files.get(clean)
            if existing and not self._can_write(existing, user):
                raise PermissionError(f"{user} cannot write {clean}")
            digest = self._hash_content(content)
            self._blobs[digest] = str(content)
            node = FileNode(
                content_hash=digest,
                owner=existing.owner if existing else user,
                group=existing.group if existing else user,
                mode=existing.mode if existing and mode is None else int(mode or 0o644),
                updated_at=time.time(),
            )
            self._files[clean] = node
            self._persist()
            return node

    def read_file(self, path: str, *, user: str | None = None) -> str:
        user = user or self.default_user
        clean = self._normalize_file_path(path)
        with self._lock:
            node = self._require_file(clean)
            if not self._can_read(node, user):
                raise PermissionError(f"{user} cannot read {clean}")
            return self._blobs[node.content_hash]

    def delete_file(self, path: str, *, user: str | None = None) -> None:
        user = user or self.default_user
        clean = self._normalize_file_path(path)
        with self._lock:
            node = self._require_file(clean)
            if not self._can_write(node, user):
                raise PermissionError(f"{user} cannot delete {clean}")
            del self._files[clean]
            self._persist()

    def list_dir(self, path: str = "/") -> list[dict[str, str]]:
        clean = self._normalize_dir_path(path)
        prefix = f"{clean}/" if clean else ""
        entries: dict[str, str] = {}
        with self._lock:
            for file_path in self._files:
                if not file_path.startswith(prefix):
                    continue
                remainder = file_path[len(prefix):]
                if not remainder:
                    continue
                head = remainder.split("/", 1)[0]
                entry_path = f"{prefix}{head}" if prefix else head
                entries[entry_path] = "directory" if "/" in remainder else "file"
        return [
            {"path": entry_path, "type": entry_type}
            for entry_path, entry_type in sorted(entries.items())
        ]

    def tree(self, path: str = "/") -> list[str]:
        clean = self._normalize_dir_path(path)
        prefix = f"{clean}/" if clean else ""
        with self._lock:
            return sorted(file_path for file_path in self._files if file_path.startswith(prefix))

    def grep(self, pattern: str, *, path: str = "/", recursive: bool = True) -> list[dict[str, Any]]:
        clean = self._normalize_dir_path(path)
        prefix = f"{clean}/" if clean else ""
        matches: list[dict[str, Any]] = []
        with self._lock:
            for file_path, node in sorted(self._files.items()):
                if prefix and not file_path.startswith(prefix):
                    continue
                if not recursive and "/" in file_path[len(prefix):]:
                    continue
                for lineno, line in enumerate(self._blobs[node.content_hash].splitlines(), start=1):
                    if pattern in line:
                        matches.append({"path": file_path, "line": lineno, "text": line})
        return matches

    def find(self, name: str = "*.md", *, path: str = "/") -> list[str]:
        clean = self._normalize_dir_path(path)
        prefix = f"{clean}/" if clean else ""
        with self._lock:
            return sorted(
                file_path
                for file_path in self._files
                if file_path.startswith(prefix) and fnmatch.fnmatch(posixpath.basename(file_path), name)
            )

    def chmod(self, path: str, mode: int, *, user: str | None = None) -> None:
        user = user or self.default_user
        clean = self._normalize_file_path(path)
        with self._lock:
            node = self._require_file(clean)
            if node.owner != user:
                raise PermissionError(f"{user} cannot chmod {clean}")
            self._files[clean] = FileNode(
                content_hash=node.content_hash,
                owner=node.owner,
                group=node.group,
                mode=int(mode),
                updated_at=time.time(),
            )
            self._persist()

    def chown(self, path: str, owner: str, group: str | None = None, *, user: str | None = None) -> None:
        user = user or self.default_user
        clean = self._normalize_file_path(path)
        with self._lock:
            node = self._require_file(clean)
            if node.owner != user:
                raise PermissionError(f"{user} cannot chown {clean}")
            self._files[clean] = FileNode(
                content_hash=node.content_hash,
                owner=owner,
                group=group or node.group,
                mode=node.mode,
                updated_at=time.time(),
            )
            self._persist()

    def add_user_to_group(self, user: str, group: str) -> None:
        with self._lock:
            self._groups.setdefault(group, set()).add(user)
            self._persist()

    def commit(
        self,
        message: str,
        *,
        author: str | None = None,
        bookmark: str = "main",
        metadata: dict[str, Any] | None = None,
    ) -> WorkspaceCommit:
        author = author or self.default_user
        metadata = dict(metadata or {})
        with self._lock:
            parent = self._bookmarks.get(bookmark)
            parents = (parent,) if parent else ()
            tree = {path: asdict(node) for path, node in sorted(self._files.items())}
            timestamp = time.time()
            change_id = self._hash_json(
                {
                    "message": message,
                    "author": author,
                    "parents": parents,
                    "tree": tree,
                    "timestamp": timestamp,
                    "metadata": metadata,
                }
            )[:12]
            commit = WorkspaceCommit(
                change_id=change_id,
                message=message,
                author=author,
                parents=parents,
                tree=tree,
                timestamp=timestamp,
                bookmark=bookmark,
                metadata=metadata,
            )
            self._commits[change_id] = commit
            self._bookmarks[bookmark] = change_id
            self._persist()
            return commit

    def log(self, bookmark: str = "main", *, limit: int = 50) -> list[WorkspaceCommit]:
        with self._lock:
            head = self._bookmarks.get(bookmark)
            commits: list[WorkspaceCommit] = []
            seen: set[str] = set()
            while head and head not in seen and len(commits) < limit:
                seen.add(head)
                commit = self._commits.get(head)
                if commit is None:
                    break
                commits.append(commit)
                head = commit.parents[0] if commit.parents else None
            return commits

    def status(self, bookmark: str = "main") -> dict[str, list[str]]:
        with self._lock:
            head = self._bookmarks.get(bookmark)
            base_tree = self._commits[head].tree if head in self._commits else {}
            current = {path: asdict(node) for path, node in self._files.items()}
            added = sorted(set(current) - set(base_tree))
            deleted = sorted(set(base_tree) - set(current))
            modified = sorted(
                path for path in set(current) & set(base_tree)
                if current[path] != base_tree[path]
            )
            conflicts = sorted(
                conflict.path
                for records in self._conflicts.values()
                for conflict in records
            )
            return {
                "added": added,
                "modified": modified,
                "deleted": deleted,
                "conflicts": conflicts,
            }

    def revert(self, change_id: str) -> None:
        with self._lock:
            commit = self._commits.get(change_id)
            if commit is None:
                raise KeyError(f"unknown change: {change_id}")
            self._files = {
                path: FileNode.from_dict(node)
                for path, node in commit.tree.items()
            }
            self._persist()

    def create_bookmark(self, name: str, change_id: str | None = None) -> None:
        with self._lock:
            target = change_id or self._bookmarks.get("main")
            if target and target not in self._commits:
                raise KeyError(f"unknown change: {target}")
            if target:
                self._bookmarks[name] = target
            else:
                self._bookmarks.pop(name, None)
            self._persist()

    def move_bookmark(self, name: str, change_id: str) -> None:
        with self._lock:
            if change_id not in self._commits:
                raise KeyError(f"unknown change: {change_id}")
            self._bookmarks[name] = change_id
            self._persist()

    def merge(self, *, target: str, source: str, delete_source: bool = False, author: str | None = None) -> MergeResult:
        author = author or self.default_user
        with self._lock:
            target_head = self._bookmarks.get(target)
            source_head = self._bookmarks.get(source)
            if not target_head or not source_head:
                raise KeyError("both target and source bookmarks must exist")
            if target_head == source_head:
                if delete_source:
                    self._bookmarks.pop(source, None)
                    self._persist()
                return MergeResult("no_op", change_id=target_head)
            if self._is_ancestor(target_head, source_head):
                self._bookmarks[target] = source_head
                if delete_source:
                    self._bookmarks.pop(source, None)
                self._persist()
                return MergeResult("fast_forward", change_id=source_head)
            if self._is_ancestor(source_head, target_head):
                if delete_source:
                    self._bookmarks.pop(source, None)
                    self._persist()
                return MergeResult("no_op", change_id=target_head)

            base = self._common_ancestor(target_head, source_head)
            base_tree = self._commits[base].tree if base else {}
            target_tree = self._commits[target_head].tree
            source_tree = self._commits[source_head].tree
            merged_tree: dict[str, dict[str, Any]] = {}
            conflicts: list[ConflictRecord] = []

            for path in sorted(set(base_tree) | set(target_tree) | set(source_tree)):
                base_node = base_tree.get(path)
                target_node = target_tree.get(path)
                source_node = source_tree.get(path)
                if target_node == source_node:
                    if target_node is not None:
                        merged_tree[path] = target_node
                elif base_node == target_node:
                    if source_node is not None:
                        merged_tree[path] = source_node
                elif base_node == source_node:
                    if target_node is not None:
                        merged_tree[path] = target_node
                else:
                    if target_node is not None:
                        merged_tree[path] = target_node
                    conflicts.append(
                        ConflictRecord(
                            path=path,
                            base_hash=base_node.get("content_hash") if base_node else None,
                            target_hash=target_node.get("content_hash") if target_node else None,
                            source_hash=source_node.get("content_hash") if source_node else None,
                        )
                    )

            timestamp = time.time()
            change_id = self._hash_json(
                {
                    "merge": [target_head, source_head],
                    "target": target,
                    "source": source,
                    "tree": merged_tree,
                    "timestamp": timestamp,
                }
            )[:12]
            commit = WorkspaceCommit(
                change_id=change_id,
                message=f"merge {source} into {target}",
                author=author,
                parents=(target_head, source_head),
                tree=merged_tree,
                timestamp=timestamp,
                bookmark=target,
            )
            self._commits[change_id] = commit
            self._bookmarks[target] = change_id
            if conflicts:
                self._conflicts[change_id] = conflicts
            if delete_source:
                self._bookmarks.pop(source, None)
            self._files = {path: FileNode.from_dict(node) for path, node in merged_tree.items()}
            self._persist()
            return MergeResult("merge_commit", change_id=change_id, conflicts=tuple(conflicts))

    def bookmarks(self) -> dict[str, str]:
        with self._lock:
            return dict(self._bookmarks)

    def conflicts(self, change_id: str | None = None) -> list[ConflictRecord]:
        with self._lock:
            if change_id:
                return list(self._conflicts.get(change_id, ()))
            return [conflict for records in self._conflicts.values() for conflict in records]

    def _require_file(self, path: str) -> FileNode:
        node = self._files.get(path)
        if node is None:
            raise FileNotFoundError(path)
        return node

    def _can_read(self, node: FileNode, user: str) -> bool:
        return self._permission_bit(node, user, owner_bit=0o400, group_bit=0o040, other_bit=0o004)

    def _can_write(self, node: FileNode, user: str) -> bool:
        return self._permission_bit(node, user, owner_bit=0o200, group_bit=0o020, other_bit=0o002)

    def _permission_bit(self, node: FileNode, user: str, *, owner_bit: int, group_bit: int, other_bit: int) -> bool:
        if user == node.owner:
            return bool(node.mode & owner_bit)
        if user in self._groups.get(node.group, set()):
            return bool(node.mode & group_bit)
        return bool(node.mode & other_bit)

    def _is_ancestor(self, ancestor: str, descendant: str) -> bool:
        stack = [descendant]
        seen: set[str] = set()
        while stack:
            current = stack.pop()
            if current == ancestor:
                return True
            if current in seen:
                continue
            seen.add(current)
            commit = self._commits.get(current)
            if commit:
                stack.extend(commit.parents)
        return False

    def _common_ancestor(self, left: str, right: str) -> str | None:
        left_ancestors = set(self._walk_ancestors(left))
        for candidate in self._walk_ancestors(right):
            if candidate in left_ancestors:
                return candidate
        return None

    def _walk_ancestors(self, head: str) -> Iterable[str]:
        stack = [head]
        seen: set[str] = set()
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            yield current
            commit = self._commits.get(current)
            if commit:
                stack.extend(commit.parents)

    def _load(self) -> None:
        if not self.storage_path.exists():
            return
        try:
            data = json.loads(self.storage_path.read_text(encoding="utf-8"))
            if int(data.get("schema_version", 0)) != self.schema_version:
                raise ValueError("unsupported markdown workspace schema")
            self._files = {
                path: FileNode.from_dict(node)
                for path, node in data.get("files", {}).items()
            }
            self._blobs = {str(k): str(v) for k, v in data.get("blobs", {}).items()}
            self._commits = {
                change_id: WorkspaceCommit.from_dict(commit)
                for change_id, commit in data.get("commits", {}).items()
            }
            self._bookmarks = {str(k): str(v) for k, v in data.get("bookmarks", {}).items()}
            self._conflicts = {
                str(k): [ConflictRecord(**record) for record in records]
                for k, records in data.get("conflicts", {}).items()
            }
            self._groups = {
                str(group): {str(user) for user in users}
                for group, users in data.get("groups", {}).items()
            } or {self.default_user: {self.default_user}}
        except Exception as exc:
            record_degradation("markdown_workspace", exc)
            raise

    def _persist(self) -> None:
        payload = {
            "schema_version": self.schema_version,
            "files": {path: asdict(node) for path, node in sorted(self._files.items())},
            "blobs": dict(sorted(self._blobs.items())),
            "commits": {
                change_id: asdict(commit)
                for change_id, commit in sorted(self._commits.items())
            },
            "bookmarks": dict(sorted(self._bookmarks.items())),
            "conflicts": {
                change_id: [asdict(conflict) for conflict in records]
                for change_id, records in sorted(self._conflicts.items())
            },
            "groups": {
                group: sorted(users)
                for group, users in sorted(self._groups.items())
            },
        }
        atomic_write_text(self.storage_path, json.dumps(payload, indent=2, sort_keys=True))

    @staticmethod
    def _hash_content(content: str) -> str:
        return hashlib.sha256(str(content).encode("utf-8")).hexdigest()

    @staticmethod
    def _hash_json(data: dict[str, Any]) -> str:
        return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode("utf-8")).hexdigest()

    @staticmethod
    def _normalize_file_path(path: str) -> str:
        clean = MarkdownWorkspace._normalize_dir_path(path)
        if not clean.endswith(".md"):
            raise ValueError("MarkdownWorkspace only stores .md files")
        return clean

    @staticmethod
    def _normalize_dir_path(path: str) -> str:
        raw = str(path or "").replace("\\", "/").strip()
        if raw in {"", ".", "/"}:
            return ""
        if raw.startswith("/"):
            raw = raw[1:]
        clean = posixpath.normpath(raw)
        if clean in {"", "."}:
            return ""
        if clean.startswith("../") or clean == ".." or "/../" in f"/{clean}/":
            raise ValueError(f"path escapes workspace: {path}")
        return clean.rstrip("/")


__all__ = [
    "ConflictRecord",
    "FileNode",
    "MarkdownWorkspace",
    "MergeResult",
    "WorkspaceCommit",
]
