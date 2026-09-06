"""Governed durable knowledge-item library for Aura."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from core.skills.what_every_skill_gives_back import THE_SHARED_RESULT
from core.config import config
from core.runtime.atomic_writer import interprocess_file_lock
from core.runtime.file_write_gateway import get_file_write_gateway
from core.skills.base_skill import BaseSkill

_CATALOG_SCHEMA_VERSION = 1
_MAX_CATALOG_BYTES = 8 * 1024 * 1024
_ITEM_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,95}$")


class KnowledgeBaseInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    action: Literal["create", "upsert", "read", "search", "list", "delete"]
    title: str | None = Field(default=None, min_length=1, max_length=200)
    content: str | None = Field(default=None, min_length=1, max_length=2_000_000)
    query: str | None = Field(default=None, min_length=1, max_length=500)
    summary: str | None = Field(default=None, max_length=500)
    limit: int = Field(default=20, ge=1, le=100)

    @field_validator("title", "query", "summary")
    @classmethod
    def _reject_blank_optional_text(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("value must not be blank")
        return value

    @model_validator(mode="after")
    def _validate_action_fields(self) -> KnowledgeBaseInput:
        if self.action in {"create", "upsert"} and (not self.title or not self.content):
            raise ValueError(f"{self.action} requires title and content")
        if self.action in {"read", "delete"} and not self.title:
            raise ValueError(f"{self.action} requires title")
        if self.action == "search" and not self.query:
            raise ValueError("search requires query")
        return self


class KnowledgeBaseSkill(BaseSkill):
    #: What a caller gets back. The shared part only: every skill
    #: here returns `ok`, and a schema claiming to be complete
    #: would be wrong for every one that adds a field.
    result_schema = THE_SHARED_RESULT

    name = "knowledge_base"
    description = (
        "Create, update, verify, search, list, and delete durable Markdown knowledge items."
    )
    effect_scope = "state_mutation"
    input_model = KnowledgeBaseInput

    def __init__(self, store_dir: str | Path | None = None):
        self.store_dir = (
            Path(store_dir).expanduser()
            if store_dir is not None
            else Path(config.paths.data_dir) / "knowledge_base"
        )
        self._lock = threading.RLock()

    @property
    def _catalog_path(self) -> Path:
        return self.store_dir / "catalog.json"

    @contextmanager
    def _transaction_lock(self) -> Iterator[None]:
        with self._lock, interprocess_file_lock(self.store_dir / ".catalog.lock"):
            yield

    @staticmethod
    def _normalize_title(title: str) -> str:
        return " ".join(str(title).strip().split()).casefold()

    @classmethod
    def _item_id(cls, title: str) -> str:
        normalized = cls._normalize_title(title)
        slug = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")[:64] or "item"
        digest = hashlib.sha256(normalized.encode()).hexdigest()[:12]
        return f"{slug}_{digest}"

    def _item_path(self, item_id: str) -> Path:
        if not _ITEM_ID_RE.fullmatch(item_id):
            raise ValueError("invalid knowledge item identifier")
        return self.store_dir / "items" / f"{item_id}.md"

    @staticmethod
    def _empty_catalog() -> dict[str, Any]:
        return {"items": {}, "schema_version": _CATALOG_SCHEMA_VERSION}

    def _load_catalog(self) -> dict[str, Any]:
        path = self._catalog_path
        if not path.exists():
            return self._empty_catalog()
        if path.stat().st_size > _MAX_CATALOG_BYTES:
            raise ValueError("knowledge catalog exceeds its bounded size")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"knowledge catalog is unreadable: {exc}") from exc
        if not isinstance(payload, dict) or payload.get("schema_version") != _CATALOG_SCHEMA_VERSION:
            raise ValueError("knowledge catalog schema is invalid or unsupported")
        items = payload.get("items")
        if not isinstance(items, dict):
            raise ValueError("knowledge catalog items must be an object")
        for item_id, item in items.items():
            if not _ITEM_ID_RE.fullmatch(str(item_id)) or not isinstance(item, dict):
                raise ValueError("knowledge catalog contains an invalid item record")
        return payload

    @staticmethod
    def _write_text(path: Path, content: str, *, source: str) -> None:
        get_file_write_gateway().write_text(path, content, source=source)

    @staticmethod
    def _delete_file(path: Path, *, source: str) -> bool:
        return bool(get_file_write_gateway().delete_file(path, source=source))

    def _write_catalog(self, catalog: dict[str, Any]) -> None:
        self._write_text(
            self._catalog_path,
            json.dumps(catalog, ensure_ascii=False, indent=2, sort_keys=True),
            source="skills.knowledge_base.catalog",
        )

    @staticmethod
    def _content_digest(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(tz=UTC).isoformat()

    def _create_or_update(self, params: KnowledgeBaseInput) -> dict[str, Any]:
        if params.title is None or params.content is None:
            raise ValueError(f"{params.action} requires title and content")
        with self._transaction_lock():
            catalog = self._load_catalog()
            items = dict(catalog["items"])
            item_id = self._item_id(params.title)
            existing = items.get(item_id)
            if existing is not None and params.action == "create":
                return {
                    "error": "knowledge item already exists; use upsert to replace it",
                    "item_id": item_id,
                    "ok": False,
                    "status": "conflict",
                }

            path = self._item_path(item_id)
            old_content = path.read_text(encoding="utf-8") if path.exists() else None
            now = self._timestamp()
            summary = params.summary or " ".join(params.content.split())[:300]
            record = {
                "bytes": len(params.content.encode("utf-8")),
                "content_sha256": self._content_digest(params.content),
                "created_at": str((existing or {}).get("created_at") or now),
                "path": f"items/{item_id}.md",
                "summary": summary,
                "title": " ".join(params.title.split()),
                "updated_at": now,
            }
            self._write_text(path, params.content, source="skills.knowledge_base.item")
            items[item_id] = record
            updated_catalog = {"items": items, "schema_version": _CATALOG_SCHEMA_VERSION}
            try:
                self._write_catalog(updated_catalog)
            except (OSError, RuntimeError, ValueError):
                if old_content is None:
                    self._delete_file(path, source="skills.knowledge_base.compensate_create")
                else:
                    self._write_text(
                        path,
                        old_content,
                        source="skills.knowledge_base.compensate_update",
                    )
                raise
            return {
                "content_sha256": record["content_sha256"],
                "item_id": item_id,
                "ok": True,
                "status": "created" if existing is None else "updated",
                "summary": f"Knowledge item {record['title']!r} persisted and indexed.",
            }

    def _read(self, title: str) -> dict[str, Any]:
        with self._transaction_lock():
            catalog = self._load_catalog()
            item_id = self._item_id(title)
            record = catalog["items"].get(item_id)
            if not isinstance(record, dict):
                return {"error": "knowledge item not found", "item_id": item_id, "ok": False}
            path = self._item_path(item_id)
            if not path.is_file():
                return {
                    "error": "knowledge item body is missing",
                    "item_id": item_id,
                    "ok": False,
                    "status": "integrity_failed",
                }
            content = path.read_text(encoding="utf-8")
            observed_digest = self._content_digest(content)
            if observed_digest != record.get("content_sha256"):
                return {
                    "error": "knowledge item content hash does not match its catalog record",
                    "item_id": item_id,
                    "ok": False,
                    "status": "integrity_failed",
                }
            return {
                "content": content,
                "content_sha256": observed_digest,
                "item_id": item_id,
                "metadata": dict(record),
                "ok": True,
                "status": "verified",
            }

    def _search(self, query: str, limit: int) -> dict[str, Any]:
        with self._transaction_lock():
            catalog = self._load_catalog()
            needle = query.casefold()
            results: list[dict[str, Any]] = []
            integrity_failures: list[str] = []
            for item_id, record in catalog["items"].items():
                title = str(record.get("title") or "")
                summary = str(record.get("summary") or "")
                path = self._item_path(item_id)
                try:
                    content = path.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    integrity_failures.append(item_id)
                    continue
                if self._content_digest(content) != record.get("content_sha256"):
                    integrity_failures.append(item_id)
                    continue
                title_text = title.casefold()
                summary_text = summary.casefold()
                content_text = content.casefold()
                score = (
                    (8 if needle == title_text else 0)
                    + (5 if needle in title_text else 0)
                    + (3 if needle in summary_text else 0)
                    + (1 if needle in content_text else 0)
                )
                if score <= 0:
                    continue
                position = content_text.find(needle)
                start = max(0, position - 120) if position >= 0 else 0
                results.append(
                    {
                        "item_id": item_id,
                        "score": score,
                        "snippet": content[start : start + 320],
                        "summary": summary,
                        "title": title,
                        "updated_at": record.get("updated_at"),
                    }
                )
            results.sort(
                key=lambda item: (-int(item["score"]), str(item["title"]).casefold(), item["item_id"])
            )
            return {
                "count": min(len(results), limit),
                "integrity_failures": integrity_failures,
                "ok": not integrity_failures,
                "query": query,
                "results": results[:limit],
                "status": "complete" if not integrity_failures else "partial_integrity_failure",
                "total_matches": len(results),
            }

    def _list(self, limit: int) -> dict[str, Any]:
        with self._transaction_lock():
            catalog = self._load_catalog()
            items = [
                {"item_id": item_id, **dict(record)}
                for item_id, record in catalog["items"].items()
            ]
            items.sort(
                key=lambda item: (str(item.get("updated_at") or ""), item["item_id"]),
                reverse=True,
            )
            return {
                "count": min(len(items), limit),
                "items": items[:limit],
                "ok": True,
                "total": len(items),
            }

    def _delete(self, title: str) -> dict[str, Any]:
        with self._transaction_lock():
            catalog = self._load_catalog()
            items = dict(catalog["items"])
            item_id = self._item_id(title)
            record = items.get(item_id)
            if not isinstance(record, dict):
                return {"error": "knowledge item not found", "item_id": item_id, "ok": False}
            path = self._item_path(item_id)
            old_content = path.read_text(encoding="utf-8") if path.exists() else None
            self._delete_file(path, source="skills.knowledge_base.delete_item")
            items.pop(item_id, None)
            try:
                self._write_catalog({"items": items, "schema_version": _CATALOG_SCHEMA_VERSION})
            except (OSError, RuntimeError, ValueError):
                if old_content is not None:
                    self._write_text(
                        path,
                        old_content,
                        source="skills.knowledge_base.compensate_delete",
                    )
                raise
            return {
                "item_id": item_id,
                "ok": True,
                "status": "deleted",
                "title": record.get("title"),
            }

    def _execute_sync(self, params: KnowledgeBaseInput) -> dict[str, Any]:
        if params.action in {"create", "upsert"}:
            return self._create_or_update(params)
        if params.action == "read":
            if params.title is None:
                raise ValueError("read requires title")
            return self._read(params.title)
        if params.action == "search":
            if params.query is None:
                raise ValueError("search requires query")
            return self._search(params.query, params.limit)
        if params.action == "list":
            return self._list(params.limit)
        if params.action == "delete":
            if params.title is None:
                raise ValueError("delete requires title")
            return self._delete(params.title)
        raise ValueError(f"unsupported knowledge-base action: {params.action}")

    async def execute(
        self,
        params: KnowledgeBaseInput | dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del context
        try:
            validated = (
                params
                if isinstance(params, KnowledgeBaseInput)
                else KnowledgeBaseInput.model_validate(params)
            )
            return await asyncio.to_thread(self._execute_sync, validated)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            return {
                "error": f"knowledge base transaction failed: {type(exc).__name__}: {exc}",
                "ok": False,
                "status": "transaction_failed",
            }
