"""
Deep Research Pipeline — Ported from gemini-fullstack-langgraph-quickstart

Implements a LangGraph-inspired iterative research graph:
  1. Query Generation  — expand user question into 1-3 diverse search queries
  2. Parallel Web Research — dispatch all queries simultaneously
  3. Reflection — analyze results, identify knowledge gaps
  4. Iterative Refinement — loop back with follow-up queries (max 3 loops)
  5. Answer Synthesis — produce cited, comprehensive answer

Replaces single-shot web search for complex research queries.
"""

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("Aura.DeepResearch")


# ── State Models ────────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    """Single search result with source metadata."""
    query: str
    content: str
    sources: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class ResearchState:
    """Full research pipeline state."""
    original_question: str
    search_queries: List[str] = field(default_factory=list)
    search_results: List[SearchResult] = field(default_factory=list)
    knowledge_gaps: List[str] = field(default_factory=list)
    follow_up_queries: List[str] = field(default_factory=list)
    loop_count: int = 0
    is_sufficient: bool = False
    final_answer: str = ""
    all_sources: List[Dict[str, str]] = field(default_factory=list)


class ResearchPhase(Enum):
    QUERY_GEN = "query_generation"
    WEB_RESEARCH = "web_research"
    REFLECTION = "reflection"
    SYNTHESIS = "synthesis"
    COMPLETE = "complete"


# ── Configuration ────────────────────────────────────────────────────────────

MAX_RESEARCH_LOOPS = 3
MAX_INITIAL_QUERIES = 3
MAX_FOLLOW_UP_QUERIES = 2


# ── Prompts ──────────────────────────────────────────────────────────────────

QUERY_GENERATION_PROMPT = """Your goal is to generate sophisticated web search queries to research the following topic.

Instructions:
- Generate 1 to {max_queries} diverse search queries
- Each query should focus on a different aspect of the topic
- Queries should target the most current information. Today is {current_date}.
- Don't generate duplicate or overlapping queries

Respond as JSON:
{{"rationale": "Brief explanation of why these queries are relevant", "queries": ["query1", "query2"]}}

Topic: {research_topic}"""

REFLECTION_PROMPT = """You are an expert research assistant analyzing search results about "{research_topic}".

Instructions:
- Evaluate if the gathered information is sufficient to comprehensively answer the question
- If gaps exist, identify what's missing and generate 1-2 focused follow-up queries
- If the information is sufficient, say so

Respond as JSON:
{{"is_sufficient": true/false, "knowledge_gap": "Description of missing info or empty string", "follow_up_queries": ["query1"] or []}}

Search results gathered so far:
{summaries}"""

SYNTHESIS_PROMPT = """Generate a comprehensive, well-structured answer based on the research results below.

Instructions:
- Today's date is {current_date}
- Synthesize all information into a coherent, accurate response
- Include citations using [Source Title](URL) markdown format where applicable
- If sources conflict, note the discrepancy
- Be thorough but concise
- Do not mention that you are synthesizing from multiple sources

User's question: {research_topic}

Research results:
{summaries}"""


# ── Pipeline Nodes ───────────────────────────────────────────────────────────

async def generate_queries(
    state: ResearchState,
    brain: Any,
) -> ResearchState:
    """Node 1: Generate diverse search queries from the user's question."""
    current_date = datetime.now().strftime("%B %d, %Y")

    prompt = QUERY_GENERATION_PROMPT.format(
        max_queries=MAX_INITIAL_QUERIES,
        current_date=current_date,
        research_topic=state.original_question,
    )

    result = await brain.generate(
        prompt,
        options={"num_predict": 512, "temperature": 1.0, "num_ctx": 4096}
    )
    response_text = result.get("response", "")

    # Parse JSON response
    try:
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            queries = parsed.get("queries", [])
            if queries:
                state.search_queries = queries[:MAX_INITIAL_QUERIES]
                logger.info("Generated %d search queries: %s", len(state.search_queries), state.search_queries)
                return state
    except (json.JSONDecodeError, AttributeError) as e:
        logger.warning("Query generation JSON parse failed: %s", e)

    # Fallback: use the original question as-is
    state.search_queries = [state.original_question]
    return state


async def web_research(
    state: ResearchState,
    search_fn: Any,
    queries: List[str] = None,
) -> ResearchState:
    """Node 2: Execute web searches in parallel for all queries.

    Args:
        state: Current research state
        search_fn: Async callable that takes a query string and returns
                   {"ok": bool, "content": str, "sources": list}
        queries: Override queries (for follow-up rounds)
    """
    queries_to_search = queries or state.search_queries

    async def _search_one(query: str) -> SearchResult:
        try:
            result = await search_fn(query)
            sources = result.get("sources", [])
            content = result.get("content", result.get("text", ""))
            return SearchResult(query=query, content=content, sources=sources)
        except Exception as e:
            logger.warning("Search failed for '%s': %s", query, e)
            return SearchResult(query=query, content=f"Search failed: {e}", sources=[])

    # Parallel dispatch
    results = await asyncio.gather(*[_search_one(q) for q in queries_to_search])

    state.search_results.extend(results)

    # Collect sources
    for r in results:
        state.all_sources.extend(r.sources)

    logger.info("Completed %d parallel searches, total results: %d",
                len(queries_to_search), len(state.search_results))
    return state


async def reflection(
    state: ResearchState,
    brain: Any,
) -> ResearchState:
    """Node 3: Analyze results for knowledge gaps and decide if more research is needed."""
    state.loop_count += 1

    # Build summaries
    summaries = "\n\n---\n\n".join(
        f"Query: {r.query}\nResult:\n{r.content[:3000]}"
        for r in state.search_results
    )

    prompt = REFLECTION_PROMPT.format(
        research_topic=state.original_question,
        summaries=summaries,
    )

    result = await brain.generate(
        prompt,
        options={"num_predict": 512, "temperature": 0.3, "num_ctx": 8192}
    )
    response_text = result.get("response", "")

    # Parse reflection JSON
    try:
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            state.is_sufficient = parsed.get("is_sufficient", False)
            gap = parsed.get("knowledge_gap", "")
            if gap:
                state.knowledge_gaps.append(gap)
            follow_ups = parsed.get("follow_up_queries", [])
            state.follow_up_queries = follow_ups[:MAX_FOLLOW_UP_QUERIES]

            logger.info(
                "Reflection loop %d: sufficient=%s, gaps=%s, follow_ups=%d",
                state.loop_count, state.is_sufficient, gap[:80] if gap else "none",
                len(state.follow_up_queries)
            )
            return state
    except (json.JSONDecodeError, AttributeError) as e:
        logger.warning("Reflection JSON parse failed: %s", e)

    # Conservative fallback: mark as sufficient to avoid infinite loops
    state.is_sufficient = True
    return state


async def synthesize_answer(
    state: ResearchState,
    brain: Any,
) -> ResearchState:
    """Node 4: Produce the final cited answer from all research results."""
    current_date = datetime.now().strftime("%B %d, %Y")

    summaries = "\n\n---\n\n".join(
        f"Query: {r.query}\nResult:\n{r.content}"
        for r in state.search_results
    )

    prompt = SYNTHESIS_PROMPT.format(
        current_date=current_date,
        research_topic=state.original_question,
        summaries=summaries,
    )

    result = await brain.generate(
        prompt,
        options={"num_predict": 4096, "temperature": 0.3, "num_ctx": 16384}
    )

    state.final_answer = result.get("response", "No answer generated.")
    logger.info("Synthesis complete: %d chars", len(state.final_answer))
    return state


# ── Main Pipeline ────────────────────────────────────────────────────────────

async def run_deep_research(
    question: str,
    brain: Any,
    search_fn: Any,
    max_loops: int = MAX_RESEARCH_LOOPS,
    on_phase: Any = None,
) -> Dict[str, Any]:
    """Run the full deep research pipeline.

    Args:
        question: User's research question
        brain: LocalBrain instance for LLM calls
        search_fn: Async callable for web search (query -> result dict)
        max_loops: Maximum reflection/refinement loops
        on_phase: Optional callback(phase: ResearchPhase, state: ResearchState)

    Returns:
        {"answer": str, "sources": list, "loops": int, "queries": list}
    """
    start_time = time.time()
    state = ResearchState(original_question=question)

    def _notify(phase: ResearchPhase):
        if on_phase:
            try:
                on_phase(phase, state)
            except Exception:
                pass

    # Phase 1: Generate queries
    _notify(ResearchPhase.QUERY_GEN)
    state = await generate_queries(state, brain)

    # Phase 2: Initial web research (parallel)
    _notify(ResearchPhase.WEB_RESEARCH)
    state = await web_research(state, search_fn)

    # Phase 3-4: Reflection loop
    for loop in range(max_loops):
        _notify(ResearchPhase.REFLECTION)
        state = await reflection(state, brain)

        if state.is_sufficient or not state.follow_up_queries:
            break

        # Follow-up research
        _notify(ResearchPhase.WEB_RESEARCH)
        state = await web_research(state, search_fn, queries=state.follow_up_queries)
        state.follow_up_queries = []  # Reset for next reflection

    # Phase 5: Synthesis
    _notify(ResearchPhase.SYNTHESIS)
    state = await synthesize_answer(state, brain)

    _notify(ResearchPhase.COMPLETE)

    duration = time.time() - start_time
    logger.info(
        "Deep research complete: %d loops, %d queries, %d sources, %.1fs",
        state.loop_count, len(state.search_queries), len(state.all_sources), duration
    )

    # Deduplicate sources
    seen_urls = set()
    unique_sources = []
    for src in state.all_sources:
        url = src.get("url", src.get("uri", ""))
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_sources.append(src)

    return {
        "answer": state.final_answer,
        "sources": unique_sources,
        "loops": state.loop_count,
        "queries_used": state.search_queries + [q for r in state.search_results for q in [r.query]],
        "knowledge_gaps": state.knowledge_gaps,
        "duration_seconds": round(duration, 1),
    }
