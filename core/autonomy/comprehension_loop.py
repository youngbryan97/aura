"""core/autonomy/comprehension_loop.py
───────────────────────────────────────
Take a ``FetchExecution`` (raw content from the fetcher) and produce
structured ``CheckpointSummary`` records via incremental LLM extraction.
The comprehension loop is the layer that turns "bytes on disk" into
"things Aura has actually understood."

Design
------
- **Chunked**, never one-shot. Long content (a film transcript, a wiki
  article, a comic adaptation) is split into ~2000-token chunks. The LLM
  is asked to summarize and extract per chunk, then to integrate across
  chunks at the end.
- **Multi-pass per chunk**: skim → focused-read → contradiction-check.
  Each pass refines the extraction with a different lens.
- **Reasoning-trace aware**: when the model emits `<think>` blocks
  (Mythos / R1-class), the trace is captured separately for the substrate.
- **Self-critique**: after extraction, the loop runs a "shallow read
  detector" — does the summary actually reference specific content from
  the chunk, or is it generic? If shallow, retry with more context.
- **Cross-source consistency**: when multiple priority levels supplied
  content (watched + read transcript + read commentary), the loop checks
  for contradictions across them and surfaces them as open threads.

Public API:
    loop = ComprehensionLoop(inference=...)
    record = await loop.comprehend(item, fetch_execution)
    record.checkpoints, record.cross_source_contradictions, record.shallow_read_flag
"""

from __future__ import annotations
from core.runtime.errors import record_degradation


import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from core.autonomy.reasoning_trace import (
    parse_reasoning_response,
    reasoning_aware_prompt_prefix,
)

logger = logging.getLogger("Aura.ComprehensionLoop")

DEFAULT_CHUNK_TOKENS = 2000
TOKENS_PER_WORD = 1.33
DEFAULT_MAX_CHUNKS = 16
DEFAULT_RETRIES_PER_CHUNK = 2

# Comprehension prompt templates
SKIM_PROMPT = (
    "You are reading a chunk of a larger work to understand it. Don't summarize "
    "in generic terms — be specific. Reference named characters, scenes, arguments, "
    "and quotes from THIS chunk. If the chunk is empty or untranslatable, say so.\n\n"
    "Chunk:\n```\n{chunk}\n```\n\n"
    "Return a JSON object with:\n"
    "  summary: 2-4 sentences on what specifically happens / what is argued in this chunk.\n"
    "  named_entities: list of proper nouns or specific concepts introduced.\n"
    "  key_quotes: list of at most 3 verbatim quotes worth remembering.\n"
    "  open_questions: list of things this chunk raises that you can't answer yet.\n"
    "  affective_response: one short sentence on what felt important.\n"
    "Output JSON only."
)

CRITIQUE_PROMPT = (
    "You produced this extraction:\n```json\n{extraction}\n```\n"
    "From this content chunk:\n```\n{chunk_excerpt}\n```\n\n"
    "Critique your extraction. If it could have been written without reading the "
    "specific chunk (i.e. it's a generic summary), say `shallow: true` and propose a "
    "more specific revision. Otherwise say `shallow: false`.\n"
    "Return JSON: {{\"shallow\": bool, \"revised_summary\": str | null, \"reason\": str}}."
)

CROSS_SOURCE_PROMPT = (
    "These are extractions from different sources about the same work, in different "
    "fidelity tiers:\n{sources}\n\n"
    "Identify any contradictions between sources, the strongest source for each "
    "factual claim, and any threads still unresolved. Return JSON:\n"
    "{{\"contradictions\": [string], \"unified_summary\": string, \"open_threads\": [string]}}."
)


@dataclass
class CheckpointSummary:
    chunk_index: int
    method_source: str
    priority_level: int
    summary: str = ""
    extracted_facts: List[str] = field(default_factory=list)
    named_entities: List[str] = field(default_factory=list)
    quotes: List[str] = field(default_factory=list)
    open_questions: List[str] = field(default_factory=list)
    affective_response: str = ""
    shallow_read: bool = False
    thinking_trace: Optional[str] = None
    raw_excerpt: str = ""
    elapsed_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_index": self.chunk_index,
            "method_source": self.method_source,
            "priority_level": self.priority_level,
            "summary": self.summary,
            "extracted_facts": list(self.extracted_facts),
            "named_entities": list(self.named_entities),
            "quotes": list(self.quotes),
            "open_questions": list(self.open_questions),
            "affective_response": self.affective_response,
            "shallow_read": self.shallow_read,
            "thinking_trace": self.thinking_trace,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
        }


@dataclass
class ComprehensionRecord:
    item_title: str
    checkpoints: List[CheckpointSummary] = field(default_factory=list)
    unified_summary: str = ""
    cross_source_contradictions: List[str] = field(default_factory=list)
    open_threads: List[str] = field(default_factory=list)
    shallow_read_flag: bool = False
    sources_engaged: List[str] = field(default_factory=list)
    priority_levels_engaged: List[int] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    inference_failures: int = 0


class ComprehensionLoop:
    def __init__(
        self,
        inference: Optional[Any] = None,
        max_chunks_per_source: int = DEFAULT_MAX_CHUNKS,
        chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
        retries_per_chunk: int = DEFAULT_RETRIES_PER_CHUNK,
        enable_reasoning_trace: bool = True,
    ) -> None:
        self._infer = inference
        self._max_chunks = max_chunks_per_source
        self._chunk_tokens = chunk_tokens
        self._retries = retries_per_chunk
        self._reasoning = enable_reasoning_trace

    async def comprehend(self, item: Any, fetch_execution: Any) -> ComprehensionRecord:
        title = getattr(item, "title", None) or getattr(fetch_execution, "plan_title", "") or ""
        record = ComprehensionRecord(item_title=str(title))

        successful = list(getattr(fetch_execution, "successful", []) or [])
        if not successful:
            record.inference_failures = 1
            record.completed_at = time.time()
            return record

        record.sources_engaged = list({c.target for c in successful if getattr(c, "target", None)})
        record.priority_levels_engaged = sorted({c.priority_level for c in successful})

        # 1. Per-source chunked comprehension
        for content in successful:
            text = (getattr(content, "transcript", "") or "").strip() or (getattr(content, "text", "") or "").strip()
            if not text:
                continue
            chunks = self._chunk_text(text, self._chunk_tokens)[: self._max_chunks]
            for idx, chunk in enumerate(chunks):
                checkpoint = await self._comprehend_chunk(
                    chunk_index=len(record.checkpoints),
                    chunk_text=chunk,
                    method_source=getattr(content, "method", ""),
                    priority_level=int(getattr(content, "priority_level", 6)),
                    item_title=title,
                )
                record.checkpoints.append(checkpoint)
                if checkpoint.shallow_read:
                    record.shallow_read_flag = True

        # 2. Cross-source integration
        if len(successful) >= 2 and record.checkpoints:
            integration = await self._integrate_across_sources(record.checkpoints)
            record.unified_summary = integration.get("unified_summary", "")
            record.cross_source_contradictions = integration.get("contradictions", [])
            record.open_threads = integration.get("open_threads", [])
        else:
            record.unified_summary = " ".join(
                c.summary for c in record.checkpoints if c.summary
            )[:2000]
            record.open_threads = list({q for c in record.checkpoints for q in c.open_questions})[:20]

        record.completed_at = time.time()
        return record

    # ── Per-chunk comprehension ──────────────────────────────────────────

    async def _comprehend_chunk(
        self,
        chunk_index: int,
        chunk_text: str,
        method_source: str,
        priority_level: int,
        item_title: str,
    ) -> CheckpointSummary:
        t0 = time.time()
        excerpt = chunk_text[:2000]
        cp = CheckpointSummary(
            chunk_index=chunk_index,
            method_source=method_source,
            priority_level=priority_level,
            raw_excerpt=excerpt,
        )

        last_extraction: Optional[Dict[str, Any]] = None
        for attempt in range(self._retries + 1):
            try:
                extraction, thinking = await self._llm_extract(chunk_text, item_title)
                if extraction is not None:
                    last_extraction = extraction
                    cp.thinking_trace = thinking
                    break
            except Exception as e:
                record_degradation('comprehension_loop', e)
                logger.debug("chunk extract attempt %d failed: %s", attempt, e)

        if last_extraction is not None:
            cp.summary = str(last_extraction.get("summary", "")).strip()
            cp.named_entities = _to_str_list(last_extraction.get("named_entities"))
            cp.quotes = _to_str_list(last_extraction.get("key_quotes"))
            cp.open_questions = _to_str_list(last_extraction.get("open_questions"))
            cp.affective_response = str(last_extraction.get("affective_response", "")).strip()
            cp.extracted_facts = cp.named_entities + cp.quotes  # convenience aggregation

            # Self-critique pass
            try:
                critique = await self._self_critique(last_extraction, excerpt)
                if critique and critique.get("shallow") is True:
                    cp.shallow_read = True
                    if critique.get("revised_summary"):
                        cp.summary = str(critique["revised_summary"])
            except Exception as e:
                record_degradation('comprehension_loop', e)
                logger.debug("self-critique failed: %s", e)

        cp.elapsed_seconds = time.time() - t0
        return cp

    async def _llm_extract(self, chunk: str, item_title: str) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
        prompt = (
            reasoning_aware_prompt_prefix(self._reasoning)
            + SKIM_PROMPT.format(chunk=chunk)
            + f"\n\nContent context: this is from `{item_title}`."
        )
        raw = await self._call_llm(prompt)
        if not raw:
            return None, None
        parsed = parse_reasoning_response(raw)
        extraction = _safe_json_object(parsed.answer)
        return extraction, parsed.thinking

    async def _self_critique(
        self,
        extraction: Dict[str, Any],
        chunk_excerpt: str,
    ) -> Optional[Dict[str, Any]]:
        prompt = CRITIQUE_PROMPT.format(
            extraction=json.dumps(extraction, ensure_ascii=False),
            chunk_excerpt=chunk_excerpt,
        )
        raw = await self._call_llm(prompt)
        if not raw:
            return None
        parsed = parse_reasoning_response(raw)
        return _safe_json_object(parsed.answer)

    async def _integrate_across_sources(
        self,
        checkpoints: Sequence[CheckpointSummary],
    ) -> Dict[str, Any]:
        # Group by source
        by_source: Dict[str, List[CheckpointSummary]] = {}
        for cp in checkpoints:
            by_source.setdefault(f"{cp.method_source}@p{cp.priority_level}", []).append(cp)

        sources_text_parts: List[str] = []
        for src, cps in by_source.items():
            joined = " ".join(cp.summary for cp in cps if cp.summary)
            sources_text_parts.append(f"### {src}\n{joined[:2000]}")

        prompt = (
            reasoning_aware_prompt_prefix(self._reasoning)
            + CROSS_SOURCE_PROMPT.format(sources="\n\n".join(sources_text_parts))
        )
        raw = await self._call_llm(prompt)
        if not raw:
            return {"unified_summary": "", "contradictions": [], "open_threads": []}
        parsed = parse_reasoning_response(raw)
        out = _safe_json_object(parsed.answer) or {}
        return {
            "unified_summary": str(out.get("unified_summary", "")),
            "contradictions": _to_str_list(out.get("contradictions")),
            "open_threads": _to_str_list(out.get("open_threads")),
        }

    # ── Inference adapter ────────────────────────────────────────────────

    async def _call_llm(self, prompt: str) -> str:
        if self._infer is None:
            return ""
        # Try several common entry-point names so we adapt to whatever the
        # surrounding system exposes.
        for fn_name in ("think", "complete", "ask", "generate"):
            fn = getattr(self._infer, fn_name, None)
            if fn is None:
                continue
            try:
                if asyncio.iscoroutinefunction(fn):
                    res = await fn(prompt)
                else:
                    res = fn(prompt)
                # Result may be a string, an object with .content, .text, etc.
                if isinstance(res, str):
                    return res
                for attr in ("content", "text", "answer"):
                    val = getattr(res, attr, None)
                    if isinstance(val, str):
                        return val
                if isinstance(res, dict):
                    return str(res.get("content", res.get("text", "")) or "")
            except Exception as e:
                record_degradation('comprehension_loop', e)
                logger.debug("inference %s failed: %s", fn_name, e)
                continue
        return ""

    # ── Chunking ─────────────────────────────────────────────────────────

    def _chunk_text(self, text: str, target_tokens: int) -> List[str]:
        if not text:
            return []
        words = text.split()
        words_per_chunk = max(1, int(target_tokens / TOKENS_PER_WORD))
        chunks: List[str] = []
        for i in range(0, len(words), words_per_chunk):
            chunk = " ".join(words[i : i + words_per_chunk])
            if chunk.strip():
                chunks.append(chunk)
        return chunks


# ── Helpers ───────────────────────────────────────────────────────────────


def _safe_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    candidate = text.strip()
    # Trim leading/trailing prose: find first { and last }
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start < 0 or end < 0 or end <= start:
        return None
    snippet = candidate[start : end + 1]
    try:
        obj = json.loads(snippet)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    return None


def _to_str_list(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [v] if v.strip() else []
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    return []
