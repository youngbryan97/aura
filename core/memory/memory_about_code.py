"""Whether a memory about code still matches the code.

A memory that names a file, and perhaps a function or class in it, can be
checked: the file is there or it is not, and the name is in it or it is not.
Lifted whole out of `memory_facade`, whose MemoryFacade mixes this in, so the
constants below still read as MemoryFacade's own.
"""
from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any


class _ChecksMemoriesAboutCode:
    """Lifted whole out of MemoryFacade; see memory_facade.py."""

    TECHNICAL_HINTS = (
        "code", "repo", "file", "function", "method", "class",
        "module", "symbol", "technical", "api", "endpoint", "schema",
    )
    FILE_METADATA_KEYS = (
        "file_path", "path", "source_file", "source_path",
        "repo_path", "relative_path", "target_file",
    )
    SIGNATURE_METADATA_KEYS = (
        "target_signature", "signature", "symbol", "function",
        "class_name", "method", "api_name",
    )
    FILE_REFERENCE_RE = re.compile(
        r"(?<![\w/.-])((?:[\w.-]+/)+[\w.-]+\.(?:py|tsx?|jsx?|json|md|ya?ml|toml|sh|go|rs|java|c|cpp|h))(?:[:#]\d+)?"
    )
    SYMBOL_PATTERNS = (
        re.compile(r"\bdef\s+([A-Za-z_][A-Za-z0-9_]*)"),
        re.compile(r"\bclass\s+([A-Za-z_][A-Za-z0-9_]*)"),
        re.compile(r"\b(?:function|method|module|symbol)\s+`?([A-Za-z_][A-Za-z0-9_]*)`?", re.IGNORECASE),
    )

    def _extract_candidate_path(self, metadata: dict[str, Any], content: str) -> str | None:
        for key in self.FILE_METADATA_KEYS:
            raw_value = str(metadata.get(key) or "").strip()
            if raw_value:
                return raw_value
        match = self.FILE_REFERENCE_RE.search(content or "")
        if match:
            return match.group(1)
        return None

    def _resolve_candidate_path(self, raw_path: str | None) -> Path | None:
        cleaned = str(raw_path or "").strip().strip("`\"'")
        if not cleaned:
            return None
        if ":" in cleaned and not cleaned.startswith(("/", "./", "../")):
            base, suffix = cleaned.rsplit(":", 1)
            if suffix.isdigit():
                cleaned = base
        candidate = Path(cleaned).expanduser()
        if candidate.is_absolute():
            return candidate
        return (self._repo_root / candidate).resolve()

    def _extract_candidate_signature(self, metadata: dict[str, Any], content: str) -> str | None:
        for key in self.SIGNATURE_METADATA_KEYS:
            raw_value = str(metadata.get(key) or "").strip()
            if raw_value:
                return raw_value
        for pattern in self.SYMBOL_PATTERNS:
            match = pattern.search(content or "")
            if match:
                return match.group(1)
        return None

    def _looks_technical_memory(self, metadata: dict[str, Any], content: str) -> bool:
        if self._extract_candidate_path(metadata, content):
            return True
        meta_blob = " ".join(
            str(metadata.get(key) or "")
            for key in ("type", "category", "domain", "kind", "memory_type")
        )
        combined = f"{meta_blob} {content or ''}".lower()
        return any(hint in combined for hint in self.TECHNICAL_HINTS)

    async def _verify_memory_result(self, item: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(item)
        metadata = self._safe_metadata(normalized.get("metadata"))
        content = str(normalized.get("content") or normalized.get("text") or "").strip()

        verification_state = "not_applicable"
        verification_evidence = ""

        if self._looks_technical_memory(metadata, content):
            verification_state = "unverified"
            raw_path = self._extract_candidate_path(metadata, content)
            candidate_path = self._resolve_candidate_path(raw_path)
            if candidate_path is not None:
                metadata["resolved_path"] = str(candidate_path)
                if await asyncio.to_thread(candidate_path.exists):
                    signature = self._extract_candidate_signature(metadata, content)
                    if signature:
                        try:
                            live_content = await asyncio.to_thread(
                                candidate_path.read_text,
                                encoding="utf-8",
                                errors="ignore",
                            )
                        # not a failure: a file that will not read contributes no live content.
                        except (RuntimeError, AttributeError, TypeError, ValueError):
                            live_content = ""

                        if signature in live_content:
                            verification_state = "verified_live"
                            verification_evidence = f"matched '{signature}' in {candidate_path.name}"
                        else:
                            verification_state = "stale"
                            verification_evidence = f"file exists but '{signature}' was not found"
                    else:
                        verification_state = "verified_live"
                        verification_evidence = f"live file exists: {candidate_path.name}"
                else:
                    verification_state = "missing"
                    verification_evidence = f"live file missing: {candidate_path}"

        metadata["verification_state"] = verification_state
        if verification_evidence:
            metadata["verification_evidence"] = verification_evidence
        normalized["metadata"] = metadata
        return normalized
