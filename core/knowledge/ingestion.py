"""Knowledge ingestion entry points."""
from __future__ import annotations

import hashlib

from .source_registry import KnowledgeSource


def ingest_text_source(*, domain: str, title: str, text: str, allowed_for_eval: bool = False, trust_level: float = 0.7) -> KnowledgeSource:
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return KnowledgeSource(
        source_id="src_" + content_hash[:16],
        domain=domain,
        title=title,
        trust_level=trust_level,
        allowed_for_eval=allowed_for_eval,
        content_hash=content_hash,
    )


__all__ = ["ingest_text_source"]
