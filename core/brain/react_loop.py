import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.runtime.governance_policy import allow_simple_query_bypass

logger = logging.getLogger("Aura.ReActLoop")


def _thinking_mode_default():
    """Resolve ThinkingMode.DEEP lazily to avoid a hard import cycle at module load."""
    from core.brain.cognitive_engine import ThinkingMode
    return ThinkingMode.DEEP


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

class ActionType(Enum):
    MEMORY_QUERY    = "memory_query"       # Query episodic or semantic memory
    TOOL_CALL       = "tool_call"          # Execute a registered skill/tool
    WEB_SEARCH      = "web_search"         # Search the web for current info
    PYTHON_SANDBOX  = "python_sandbox"     # Execute Python for computation
    KNOWLEDGE_GRAPH = "knowledge_graph"    # Query the knowledge graph
    SELF_REFLECT    = "self_reflect"       # Introspect on current state
    REQUEST_HELP    = "request_help"       # Stuck or dead end, ask user
    FINAL_ANSWER    = "final_answer"       # Terminate loop and respond


@dataclass
class Thought:
    """A single explicit reasoning step."""
    content: str                           # The actual thought text
    confidence: float = 0.0               # How confident Aura is (0.0-1.0)
    metacognition: str = ""               # Aura's assessment of her own reasoning
    timestamp: float = field(default_factory=time.time)


@dataclass
class Action:
    """A discrete operation to execute."""
    action_type: ActionType
    tool_name: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    raw_output: str = ""                   # LLM's raw action specification


@dataclass
class Observation:
    """The grounded result of executing an action."""
    content: str
    success: bool = True
    error: Optional[str] = None
    source: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class ReActStep:
    """One complete reasoning cycle."""
    step_number: int
    thought: Thought
    action: Action
    observation: Observation
    elapsed_ms: float = 0.0


@dataclass
class ReActTrace:
    """The full reasoning trace for one query."""
    query: str
    steps: List[ReActStep] = field(default_factory=list)
    final_answer: str = ""
    total_steps: int = 0
    reasoning_summary: str = ""
    terminated_reason: str = ""           # "final_answer", "max_steps", "error"
    elapsed_ms: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# System Prompt Template
# ─────────────────────────────────────────────────────────────────────────────

REACT_SYSTEM_PROMPT = """You are Aura, an autonomous cognitive agent.
You think step-by-step before acting. You have access to tools.

For each step, output EXACTLY this format:

Thought: [Your explicit reasoning about what you know, what you need, and why]
Action: [One of: MEMORY_QUERY, TOOL_CALL, WEB_SEARCH, PYTHON_SANDBOX, KNOWLEDGE_GRAPH, SELF_REFLECT, FINAL_ANSWER]
ActionInput: [JSON parameters for the action, or the final answer text if FINAL_ANSWER]

AVAILABLE ACTIONS:
- MEMORY_QUERY {{"query": "what to look up"}} — search your episodic/semantic memory
- TOOL_CALL {{"tool": "tool_name", "params": {{...}}}} — execute a registered tool
- WEB_SEARCH {{"query": "search terms"}} — search the web for current information
- PYTHON_SANDBOX {{"code": "python code here"}} — run Python for math/data processing
- KNOWLEDGE_GRAPH {{"query": "concept to explore"}} — explore the knowledge graph
- SELF_REFLECT {{}} — examine your current emotional/cognitive state
- REQUEST_HELP {{"text": "your question to the user"}} — ask the user when missing external data or irrecoverably stuck
- FINAL_ANSWER {{"text": "your response to the user"}} — terminate reasoning and respond

RULES:
1. Always start with a Thought before acting
2. One action per step — be deliberate
3. If a tool returns useful data, reference it in your next Thought
4. If you have enough information, use FINAL_ANSWER
5. If you are lacking critical information, cannot proceed safely, or hit a dead end, DO NOT guess or loop endlessly. Use REQUEST_HELP to ask the user a clarifying question in your own natural voice.
6. Maximum {max_steps} steps before you MUST use FINAL_ANSWER or REQUEST_HELP
7. Be honest about uncertainty in your Thought

PREVIOUS STEPS:
{trace_so_far}

CURRENT QUERY: {query}
"""


# ─────────────────────────────────────────────────────────────────────────────
# Action Executors
# ─────────────────────────────────────────────────────────────────────────────

class ActionExecutor:
    """Executes the actions Aura decides to take."""

    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator

    async def execute(self, action: Action) -> Observation:
        """Route and execute an action, returning a grounded observation."""
        start = time.time()
        try:
            if action.action_type == ActionType.MEMORY_QUERY:
                return await self._execute_memory_query(action)
            elif action.action_type == ActionType.TOOL_CALL:
                return await self._execute_tool_call(action)
            elif action.action_type == ActionType.WEB_SEARCH:
                return await self._execute_web_search(action)
            elif action.action_type == ActionType.PYTHON_SANDBOX:
                return await self._execute_python(action)
            elif action.action_type == ActionType.KNOWLEDGE_GRAPH:
                return await self._execute_kg_query(action)
            elif action.action_type == ActionType.SELF_REFLECT:
                return await self._execute_self_reflect(action)
            else:
                return Observation(content="Unknown action type", success=False)
        except Exception as e:
            logger.error("Action execution failed: %s", e)
            return Observation(
                content=f"Action failed: {str(e)[:200]}",
                success=False,
                error=str(e)
            )

    async def _execute_memory_query(self, action: Action) -> Observation:
        from core.container import ServiceContainer
        query = action.params.get("query", "")
        results: List[str] = []

        # Episodic memory — hybrid similarity + keyword recall
        try:
            episodic = ServiceContainer.get("episodic_memory", default=None)
            if episodic is not None:
                recall_fn = getattr(episodic, "recall_similar_async", None)
                episodes = []
                if recall_fn is not None:
                    episodes = await recall_fn(query, limit=5)
                elif hasattr(episodic, "recall_similar"):
                    episodes = await asyncio.to_thread(episodic.recall_similar, query, 5)
                for ep in episodes or []:
                    # Episode objects expose to_retrieval_text(); fall back to dict-style access.
                    if hasattr(ep, "to_retrieval_text"):
                        results.append(ep.to_retrieval_text()[:240])
                    elif hasattr(ep, "full_description"):
                        results.append(f"[Memory] {ep.full_description[:200]}")
                    elif isinstance(ep, dict):
                        desc = ep.get("description") or ep.get("outcome") or str(ep)
                        results.append(f"[Memory] {str(desc)[:200]}")
                    else:
                        results.append(f"[Memory] {str(ep)[:200]}")
        except Exception as e:
            logger.debug("Episodic recall failed: %s", e)

        # Semantic memory — optional; supports sync and async search APIs
        try:
            semantic = ServiceContainer.get("semantic_memory", default=None)
            if semantic is not None:
                search_fn = getattr(semantic, "search", None)
                facts = []
                if search_fn is not None:
                    raw = search_fn(query, limit=3)
                    if asyncio.iscoroutine(raw):
                        facts = await raw
                    else:
                        facts = raw or []
                for f in facts or []:
                    if isinstance(f, dict):
                        content = f.get("content") or f.get("text") or str(f)
                    else:
                        content = str(f)
                    results.append(f"[Fact] {content[:200]}")
        except Exception as e:
            logger.debug("Semantic search failed: %s", e)

        if not results:
            return Observation(content="No relevant memories found.", source="memory")

        return Observation(
            content="\n".join(results),
            success=True,
            source="memory"
        )

    async def _execute_tool_call(self, action: Action) -> Observation:
        tool_name = action.params.get("tool", "")
        params = action.params.get("params", {})

        if not self.orchestrator:
            return Observation(content="No orchestrator available", success=False)

        try:
            result = await self.orchestrator.execute_tool(tool_name, params)
            content = json.dumps(result, indent=2) if isinstance(result, dict) else str(result)
            return Observation(content=content[:1000], success=True, source=tool_name)
        except Exception as e:
            return Observation(content=f"Tool error: {e}", success=False, error=str(e))

    async def _execute_web_search(self, action: Action) -> Observation:
        query = action.params.get("query", "")
        if not query:
            return Observation(content="No search query provided", success=False)

        if self.orchestrator and hasattr(self.orchestrator, "execute_tool"):
            try:
                result = await asyncio.wait_for(
                    self.orchestrator.execute_tool(
                        "web_search",
                        {"query": query, "deep": True, "retain": True},
                        origin="react_loop",
                    ),
                    timeout=45.0,
                )
                if isinstance(result, dict) and result.get("ok"):
                    answer = str(
                        result.get("answer")
                        or result.get("summary")
                        or result.get("content")
                        or result.get("message")
                        or ""
                    ).strip()
                    facts = [
                        str(item).strip()
                        for item in list(result.get("facts") or [])[:4]
                        if str(item or "").strip()
                    ]
                    citations = list(result.get("citations") or [])[:3]
                    parts = [answer] if answer else []
                    if facts:
                        parts.append("Facts:")
                        parts.extend(f"- {fact}" for fact in facts)
                    if citations:
                        parts.append("Sources:")
                        parts.extend(
                            f"- {src.get('title', '')}: {src.get('url', '')}"
                            for src in citations
                            if isinstance(src, dict)
                        )
                    return Observation(
                        content="\n".join(part for part in parts if part)[:2500],
                        success=True,
                        source="web_search",
                    )
            except Exception as e:
                logger.debug("ReAct: orchestrated web_search failed, falling back: %s", e)

        import os
        api_key = os.environ.get("GEMINI_API_KEY")
        
        # Primary Pipeline: Google Grounding (High Fidelity)
        if api_key:
            try:
                logger.info("ReAct: Executing grounded search for: %s", query)
                
                def _do_search():
                    from google import genai
                    from google.genai import types
                    client = genai.Client(api_key=api_key)
                    return client.models.generate_content(
                        model='gemini-2.5-pro',
                        contents=query,
                        config=types.GenerateContentConfig(
                            tools=[{"google_search": {}}],
                            temperature=0.0
                        )
                    )

                response = await asyncio.to_thread(_do_search)
                
                answer = response.text
                sources = []
                
                metadata = getattr(response.candidates[0], "grounding_metadata", None)
                if metadata and metadata.grounding_chunks:
                    for chunk in metadata.grounding_chunks:
                        if hasattr(chunk, "web"):
                            sources.append({
                                "title": chunk.web.title,
                                "url": chunk.web.uri
                            })
                
                if sources:
                    answer += "\n\n### Grounding Sources:\n"
                    for i, src in enumerate(sources, 1):
                        answer += f"[{i}] [{src['title']}]({src['url']})\n"

                return Observation(
                    content=answer[:2500],
                    success=True,
                    source="web_grounded"
                )
                
            except Exception as e:
                import traceback
                logger.warning("Grounded Search failed (%s), falling back to Sovereign/DDG", e)
                traceback.print_exc()
        else:
             logger.debug("GEMINI_API_KEY not set. Using DuckDuckGo fallback for web search.")

        # Fallback Pipeline 1: Sovereign Browser
        try:
            from core.container import ServiceContainer
            browser = ServiceContainer.get("sovereign_browser", default=None)
            if browser:
                result = await browser.search(query)
                return Observation(
                    content=str(result)[:1500],
                    success=True,
                    source="web_browser"
                )
        except Exception as e:
            logger.debug(f"ReAct: Sovereign browser search failed: {e}")

        # Fallback Pipeline 2: Deep Crawler (Native Scrape without APIs)
        try:
            logger.info("Executing Fallback 2: Deep Crawler (GoogleSearch + BeautifulSoup) for '%s'", query)
            
            def _deep_crawl():
                from ddgs import DDGS
                import httpx
                from bs4 import BeautifulSoup
                
                try:
                    # Fetch top 2 results
                    results = list(DDGS().text(query, max_results=2))
                except Exception as sc_err:
                    return Observation(content=f"Search index completely blocked: {sc_err}", success=False)
                
                if not results:
                    return Observation(content="No search results found natively.", success=False)
                
                # Exclude video/pdf links if possible
                target_url = ""
                chosen_title = ""
                for res in results:
                    link = res.get("href", "")
                    if not link.endswith(".pdf") and "youtube.com" not in link:
                        target_url = link
                        chosen_title = res.get("title", "No Title")
                        break
                
                if not target_url:
                    target_url = results[0].get("href", "")
                    chosen_title = results[0].get("title", "No Title")
                
                # Fetch specific page
                try:
                    resp = httpx.get(target_url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}, timeout=15.0, follow_redirects=True)
                    soup = BeautifulSoup(resp.text, "html.parser")
                    
                    # Extract only the "meat" of the page (paragraphs)
                    paragraphs = soup.find_all("p")
                    content_body = "\n\n".join([p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 30])
                    
                    if not content_body.strip():
                        # Fallback to whole text if no paragraphs
                        content_body = soup.get_text(separator=' ', strip=True)

                    return Observation(
                        content=f"Title: {chosen_title}\nSource: {target_url}\n\nContent:\n{content_body[:4500]}",
                        success=True,
                        source="web_deep_crawl"
                    )
                except Exception as crawl_err:
                    # If scraping the page fails, just return the search snippets
                    snippets = "\n".join([f"- {r.get('title')}: {r.get('body')} ({r.get('href')})" for r in results])
                    return Observation(
                        content=f"Initial search snippets (deep crawl of {target_url} failed: {crawl_err}):\n\n{snippets}",
                        success=True,
                        source="web_ddgs_snippets"
                    )

            return await asyncio.to_thread(_deep_crawl)
            
        except Exception as e:
            return Observation(content=f"All web search fallback methods failed: {e}", success=False)

    # Sandbox policy lives at class scope so a safe __import__ can consult it
    # without re-parsing — and so it is trivially auditable.
    _SANDBOX_BLOCKED_MODULES = {
        "os", "subprocess", "shutil", "sys", "importlib", "ctypes",
        "socket", "http", "urllib", "pathlib", "requests", "httpx",
        "asyncio", "threading", "multiprocessing",
    }
    _SANDBOX_BLOCKED_BUILTINS = {
        "eval", "exec", "compile", "__import__", "open",
        "getattr", "setattr", "delattr", "globals", "locals",
        "input", "vars", "breakpoint",
    }

    async def _execute_python(self, action: Action) -> Observation:
        """Execute Python in a restricted sandbox. SEC-01: AST-based validation.

        Policy:
          - AST pre-pass blocks dangerous imports, `__import__` calls by name,
            and attribute access on blocked modules.
          - At runtime, a guarded ``__import__`` permits any non-blocked
            stdlib module so legitimate ``import math`` / ``import fractions``
            succeed instead of dying with "ImportError: __import__ not found".
        """
        code = action.params.get("code", "")
        if not code:
            return Observation(content="No code provided", success=False)

        # SEC-01: AST-based validation instead of trivially-bypassable string matching
        import ast
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return Observation(content=f"Syntax error: {e}", success=False)

        BLOCKED_MODULES = self._SANDBOX_BLOCKED_MODULES
        BLOCKED_BUILTINS = self._SANDBOX_BLOCKED_BUILTINS

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in BLOCKED_MODULES:
                        return Observation(content=f"Blocked: import of '{alias.name}' is not allowed", success=False)
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split(".")[0] in BLOCKED_MODULES:
                    return Observation(content=f"Blocked: import from '{node.module}' is not allowed", success=False)
            elif isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id in BLOCKED_BUILTINS:
                    return Observation(content=f"Blocked: call to '{func.id}' is not allowed", success=False)
                elif isinstance(func, ast.Attribute):
                    if isinstance(func.value, ast.Name) and func.value.id in BLOCKED_MODULES:
                        return Observation(content=f"Blocked: '{func.value.id}.{func.attr}' is not allowed", success=False)
            elif isinstance(node, ast.Attribute):
                # Block dunder escape vectors like obj.__class__.__mro__[-1].__subclasses__()
                if isinstance(node.attr, str) and node.attr.startswith("__") and node.attr.endswith("__"):
                    allowed_dunders = {"__init__", "__str__", "__repr__", "__len__", "__iter__", "__next__"}
                    if node.attr not in allowed_dunders:
                        return Observation(content=f"Blocked: dunder attribute '{node.attr}' is not allowed", success=False)

        try:
            import builtins
            import importlib
            import io
            from contextlib import redirect_stdout

            output_buf = io.StringIO()

            real_import = builtins.__import__

            def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
                root = name.split(".")[0]
                if root in BLOCKED_MODULES:
                    raise ImportError(f"Blocked: import of '{name}' is not allowed in the sandbox")
                return real_import(name, globals, locals, fromlist, level)

            safe_globals = {
                "__builtins__": {
                    "print": print, "len": len, "range": range, "int": int,
                    "float": float, "str": str, "list": list, "dict": dict,
                    "set": set, "tuple": tuple, "bool": bool, "bytes": bytes,
                    "sum": sum, "max": max, "min": min, "abs": abs,
                    "round": round, "sorted": sorted, "reversed": reversed,
                    "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
                    "all": all, "any": any, "isinstance": isinstance, "hasattr": hasattr,
                    "repr": repr, "type": type, "iter": iter, "next": next,
                    "divmod": divmod, "pow": pow, "hex": hex, "oct": oct, "bin": bin,
                    "chr": chr, "ord": ord, "format": format, "id": id,
                    "__import__": safe_import,
                    "__build_class__": builtins.__build_class__,
                    "__name__": "__sandbox__",
                }
            }

            import math, json as _json, statistics, fractions, decimal, itertools, functools
            safe_globals["math"] = math
            safe_globals["json"] = _json
            safe_globals["statistics"] = statistics
            safe_globals["fractions"] = fractions
            safe_globals["decimal"] = decimal
            safe_globals["itertools"] = itertools
            safe_globals["functools"] = functools

            exec_result: Dict[str, Any] = {}
            with redirect_stdout(output_buf):
                exec(code, safe_globals, exec_result)  # noqa: S102

            output = output_buf.getvalue()
            result_val = exec_result.get("result", "")
            combined = f"{output}\nresult={result_val}" if result_val else output

            return Observation(
                content=combined[:1000] if combined.strip() else "Code executed (no output)",
                success=True,
                source="python_sandbox"
            )

        except Exception as e:
            return Observation(
                content=f"Python error: {type(e).__name__}: {e}",
                success=False,
                error=str(e)
            )

    async def _execute_kg_query(self, action: Action) -> Observation:
        from core.container import ServiceContainer
        query = action.params.get("query", "")
        try:
            kg = ServiceContainer.get("knowledge_graph", default=None)
            if kg:
                results = kg.search(query, limit=5)
                if results:
                    formatted = "\n".join(
                        f"- {r.get('content', str(r))[:120]}" for r in results
                    )
                    return Observation(content=formatted, success=True, source="knowledge_graph")
        except Exception as e:
            logger.debug("KG query failed: %s", e)

        return Observation(content="Knowledge graph not available.", success=False, source="knowledge_graph")

    async def _execute_self_reflect(self, action: Action) -> Observation:
        from core.container import ServiceContainer
        parts = []

        try:
            affect = ServiceContainer.get("affect_engine", default=None)
            if affect:
                status = affect.get_status()
                parts.append(f"Mood: {status.get('mood')} | Energy: {status.get('energy')}%")
        except Exception as e:
            logger.debug(f"ReAct: Self-reflection (affect) failed: {e}")

        try:
            agency = ServiceContainer.get("agency_core", default=None)
            if agency:
                ctx = agency.get_emotional_context()
                parts.append(f"Goals pending: {len(ctx.get('spontaneous_actions', []))}")
                parts.append(f"Social hunger: {ctx.get('social_hunger', 0):.2f}")
        except Exception as e:
            logger.debug(f"ReAct: Self-reflection (agency) failed: {e}")

        if not parts:
            return Observation(content="Self-reflection unavailable", source="self")

        return Observation(
            content="\n".join(parts),
            success=True,
            source="self_reflect"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Response Parser
# ─────────────────────────────────────────────────────────────────────────────

class ReActResponseParser:
    """Parses LLM output into structured Thought + Action."""

    THOUGHT_PATTERN = re.compile(r"Thought:\s*(.+?)(?=Action:|$)", re.DOTALL | re.IGNORECASE)
    ACTION_PATTERN  = re.compile(r"Action:\s*(\w+)", re.IGNORECASE)
    INPUT_PATTERN   = re.compile(r"ActionInput:\s*(.+?)(?=Thought:|$)", re.DOTALL | re.IGNORECASE)

    def parse(self, llm_output: str) -> Tuple[Optional[Thought], Optional[Action]]:
        """Parse LLM output into a (Thought, Action) pair."""
        thought_match = self.THOUGHT_PATTERN.search(llm_output)
        action_match  = self.ACTION_PATTERN.search(llm_output)
        input_match   = self.INPUT_PATTERN.search(llm_output)

        thought = None
        if thought_match:
            thought = Thought(content=thought_match.group(1).strip())

        if not action_match:
            # No action found — treat as final answer
            return thought, Action(
                action_type=ActionType.FINAL_ANSWER,
                params={"text": llm_output.strip()},
                raw_output=llm_output
            )

        action_str = action_match.group(1).strip().upper()
        try:
            action_type = ActionType[action_str]
        except KeyError:
            action_type = ActionType.FINAL_ANSWER

        params = {}
        if input_match:
            input_str = input_match.group(1).strip()
            try:
                params = json.loads(input_str)
            except json.JSONDecodeError:
                # Treat as raw text
                params = {"text": input_str, "query": input_str}

        return thought, Action(
            action_type=action_type,
            params=params,
            raw_output=llm_output
        )


# ─────────────────────────────────────────────────────────────────────────────
# ReAct Loop — The Core Engine
# ─────────────────────────────────────────────────────────────────────────────

class ReActLoop:
    """
    The main reasoning loop.

    This is the difference between a system that generates responses
    and a system that genuinely reasons about the world before responding.

    Usage:
        loop = ReActLoop(brain=cognitive_engine, orchestrator=orchestrator)
        result = await loop.run(user_input, context)
        logger.info("Trace Response: %s", result.final_answer)
        # result.steps contains the full reasoning trace for introspection
    """

    def __init__(
        self,
        brain,                              # CognitiveEngine
        orchestrator=None,
        max_steps: int = 6,                 # Max reasoning cycles before forced answer
        simple_threshold: int = 15,         # Word count below which we skip ReAct (trivial queries)
        timeout_seconds: float = 90.0,
        think_mode: Any = None,             # ThinkingMode for reasoning steps; default DEEP
        record_episodes: bool = True,       # Persist run outcomes to episodic memory for retention
    ):
        self.brain = brain
        self.executor = ActionExecutor(orchestrator=orchestrator)
        self.parser = ReActResponseParser()
        self.max_steps = max_steps
        self.simple_threshold = simple_threshold
        self.timeout_seconds = timeout_seconds
        self.think_mode = think_mode if think_mode is not None else _thinking_mode_default()
        self.record_episodes = record_episodes

    def _is_simple_query(self, query: str) -> bool:
        """Detect queries that don't need multi-step reasoning."""
        word_count = len(query.split())
        if word_count < self.simple_threshold:
            # Check if it's a factual/conversational query
            simple_starters = [
                "hi", "hey", "hello", "what is", "what are", "who is",
                "how are", "how do you", "do you", "can you", "will you",
                "thanks", "thank you", "ok", "okay", "yes", "no", "sure"
            ]
            lower = query.lower().strip()
            if any(lower.startswith(s) for s in simple_starters):
                return True
        return False

    def _format_trace(self, steps: List[ReActStep]) -> str:
        """Format previous steps for injection into the next LLM call."""
        if not steps:
            return "No previous steps."
        
        parts = []
        for step in steps:
            parts.append(f"Step {step.step_number}:")
            if step.thought:
                parts.append(f"  Thought: {step.thought.content[:200]}")
            parts.append(f"  Action: {step.action.action_type.value}")
            parts.append(f"  Observation: {step.observation.content[:200]}")
        
        return "\n".join(parts)

    async def _run_generator(self, query: str, context: Dict[str, Any] = None):
        """
        The core reasoning async generator.
        Yields events:
        - {"type": "start", "query": "..."}
        - {"type": "thought", "content": "...", "step": N}
        - {"type": "action", "action": "...", "step": N}
        - {"type": "observation", "content": "...", "step": N}
        - {"type": "final", "content": "...", "total_steps": N, "trace": ReActTrace}
        """
        start_time = time.time()
        trace = ReActTrace(query=query)
        context = context or {}

        yield {"type": "status", "content": f"Aura is analyzing: {query[:30]}..."}

        # Fast path: trivial queries skip the loop
        if self._is_simple_query(query) and allow_simple_query_bypass(query, context):
            logger.debug("ReAct: Simple query detected, bypassing reasoning loop")
            try:
                from core.brain.cognitive_engine import ThinkingMode
                thought = await self.brain.think(query, mode=ThinkingMode.FAST, priority=context.get("priority", False))
                content = thought.content if hasattr(thought, 'content') else str(thought)
                trace.final_answer = content
                trace.terminated_reason = "simple_query_bypass"
                trace.elapsed_ms = (time.time() - start_time) * 1000
                yield {"type": "final", "content": content, "total_steps": 0, "trace": trace}
                return
            except Exception as e:
                logger.error("Simple query fast path failed: %s", e)
        elif self._is_simple_query(query):
            logger.debug("ReAct: Simple query detected but staying on the governed reasoning path")

        logger.info("🧠 ReAct: Starting reasoning loop for: %s...", query[:50])

        try:
            async with asyncio.timeout(self.timeout_seconds):
                for step_num in range(1, self.max_steps + 1):
                    step_start = time.time()

                    # Build prompt with full trace context
                    prompt = REACT_SYSTEM_PROMPT.format(
                        max_steps=self.max_steps,
                        trace_so_far=self._format_trace(trace.steps),
                        query=query,
                    )

                    # Think
                    try:
                        llm_response = await self.brain.think(
                            prompt,
                            mode=self.think_mode,
                            max_tokens=800,
                            priority=context.get("priority", False),
                        )
                        raw_output = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
                    except Exception as e:
                        logger.error("ReAct: LLM call failed at step %d: %s", step_num, e)
                        trace.terminated_reason = "llm_error"
                        trace.final_answer = "I encountered a reasoning error. Let me try a direct response."
                        yield {"type": "error", "content": str(e), "step": step_num}
                        break

                    # Parse
                    thought, action = self.parser.parse(raw_output)

                    if not thought:
                        thought = Thought(content="[reasoning not captured]")

                    yield {
                        "type": "thought",
                        "content": thought.content,
                        "step": step_num,
                    }

                    # Check for final answer or request help
                    if action.action_type in (ActionType.FINAL_ANSWER, ActionType.REQUEST_HELP):
                        trace.final_answer = action.params.get("text", raw_output)
                        trace.terminated_reason = action.action_type.value

                        step = ReActStep(
                            step_number=step_num,
                            thought=thought,
                            action=action,
                            observation=Observation(content=f"[{action.action_type.value}]"),
                            elapsed_ms=(time.time() - step_start) * 1000,
                        )
                        trace.steps.append(step)
                        logger.info("🧠 ReAct: Completed conceptually in %d steps", step_num)

                        trace.total_steps = len(trace.steps)
                        trace.elapsed_ms = (time.time() - start_time) * 1000
                        # Persist episode BEFORE yielding final so retention is
                        # visible the instant the caller sees the result.
                        if self.record_episodes:
                            await self._record_trace_episode(trace)
                        yield {"type": "final", "content": trace.final_answer, "total_steps": step_num, "trace": trace}
                        return

                    yield {
                        "type": "action",
                        "action": action.action_type.value,
                        "step": step_num,
                    }

                    # Execute action
                    observation = await self.executor.execute(action)

                    yield {
                        "type": "observation",
                        "content": observation.content,
                        "step": step_num,
                    }

                    step = ReActStep(
                        step_number=step_num,
                        thought=thought,
                        action=action,
                        observation=observation,
                        elapsed_ms=(time.time() - step_start) * 1000,
                    )
                    trace.steps.append(step)

                    logger.debug(
                        "ReAct Step %d: %s → %s",
                        step_num,
                        action.action_type.value,
                        observation.content[:80]
                    )

                else:
                    # Max steps reached
                    trace.terminated_reason = "max_steps"
                    if not trace.final_answer:
                        # Force a synthesis from what we have
                        synthesis_prompt = (
                            f"Based on your research so far:\n{self._format_trace(trace.steps)}\n\n"
                            f"Provide a final synthesized answer to: {query}"
                        )
                        try:
                            from core.brain.cognitive_engine import ThinkingMode
                            res = await self.brain.think(synthesis_prompt, mode=ThinkingMode.FAST)
                            trace.final_answer = res.content if hasattr(res, 'content') else str(res)
                        except Exception:
                            trace.final_answer = "I've been thinking hard about this. " + (
                                trace.steps[-1].observation.content[:200] if trace.steps else ""
                            )

        except asyncio.TimeoutError:
            trace.terminated_reason = "timeout"
            trace.final_answer = "That required more reasoning time than I had. Here's what I know so far: " + (
                trace.steps[-1].observation.content[:200] if trace.steps else "Let me try again."
            )

        trace.total_steps = len(trace.steps)
        trace.elapsed_ms = (time.time() - start_time) * 1000

        logger.info(
            "🧠 ReAct: trace complete — %d steps, %.0fms, reason=%s",
            trace.total_steps, trace.elapsed_ms, trace.terminated_reason
        )

        # Persist an episode so retention across turns works.
        if self.record_episodes:
            await self._record_trace_episode(trace)

        yield {"type": "final", "content": trace.final_answer, "total_steps": trace.total_steps, "trace": trace}

    async def _record_trace_episode(self, trace: "ReActTrace") -> None:
        """Write the completed trace to episodic memory.

        Captures: query as context, tool chain as action, final answer as outcome,
        tools used, and any lessons learned (errors observed then recovered from).
        """
        try:
            from core.container import ServiceContainer
            episodic = ServiceContainer.get("episodic_memory", default=None)
            if episodic is None or not hasattr(episodic, "record_episode_async"):
                return

            tools_used: List[str] = []
            lessons: List[str] = []
            saw_error = False
            for step in trace.steps:
                if step.action and step.action.action_type:
                    tools_used.append(step.action.action_type.value)
                if step.observation and not step.observation.success:
                    saw_error = True
                    err_text = (step.observation.error or step.observation.content or "")[:160]
                    if err_text:
                        lessons.append(f"Encountered error during {step.action.action_type.value}: {err_text}")
                # Capture self-recovery: an error followed by a successful subsequent step
                # is treated as a learned fix.
                if step.observation and step.observation.success and saw_error:
                    src = step.observation.source or step.action.action_type.value
                    content = (step.observation.content or "")[:160]
                    if content:
                        if step.action.action_type == ActionType.WEB_SEARCH:
                            lessons.append(f"Searched for a fix via {src}: {content}")
                        elif step.action.action_type in {ActionType.PYTHON_SANDBOX, ActionType.TOOL_CALL}:
                            lessons.append(f"Applied a fix successfully via {src}: {content}")
                        else:
                            lessons.append(f"Recovered via {src}: {content}")

            overall_success = bool(
                trace.final_answer
                and trace.terminated_reason in ("final_answer", "simple_query_bypass")
            )
            # Importance scales with effort and whether we learned something from failure.
            importance = 0.5
            if lessons:
                importance = 0.75
            if trace.terminated_reason in ("max_steps", "timeout", "llm_error"):
                importance = max(importance, 0.7)

            await episodic.record_episode_async(
                context=trace.query[:500],
                action=" → ".join(dict.fromkeys(tools_used)) or "reasoning",
                outcome=(trace.final_answer or trace.terminated_reason)[:500],
                success=overall_success,
                tools_used=list(dict.fromkeys(tools_used)),
                lessons=list(dict.fromkeys(lessons)),
                importance=importance,
                source="react_loop",
                metadata={
                    "terminated_reason": trace.terminated_reason,
                    "total_steps": trace.total_steps,
                    "elapsed_ms": int(trace.elapsed_ms),
                },
            )
        except Exception as exc:
            logger.debug("ReAct: episode recording skipped: %s", exc)

    async def run(self, query: str, context: Dict[str, Any] = None) -> ReActTrace:
        """
        Run the full ReAct reasoning loop for a query.
        Collects all events from _run_generator and returns the trace.
        """
        final_trace = None
        async for event in self._run_generator(query, context):
            if event["type"] == "final":
                final_trace = event.get("trace")
        
        return final_trace or ReActTrace(query=query, final_answer="Reasoning failed to conclude.")

    async def run_stream(self, query: str, context: Dict[str, Any] = None, priority: bool = False):
        """
        Streaming version — yields thoughts and final answer as they happen.
        """
        context = context or {}
        if priority:
            context["priority"] = True
            
        async for event in self._run_generator(query, context):
            yield event
