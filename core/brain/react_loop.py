import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("Aura.ReActLoop")


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
- FINAL_ANSWER {{"text": "your response to the user"}} — terminate reasoning and respond

RULES:
1. Always start with a Thought before acting
2. One action per step — be deliberate
3. If a tool returns useful data, reference it in your next Thought
4. If you have enough information, use FINAL_ANSWER
5. Maximum {max_steps} steps before you MUST use FINAL_ANSWER
6. Be honest about uncertainty in your Thought

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
        results = []

        # Episodic memory
        try:
            episodic = ServiceContainer.get("episodic_memory", default=None)
            if episodic:
                episodes = await episodic.recall(query, limit=3)
                for ep in episodes:
                    results.append(f"[Memory] {ep.get('description', str(ep))[:150]}")
        except Exception as e:
            logger.debug("Episodic recall failed: %s", e)

        # Semantic memory
        try:
            semantic = ServiceContainer.get("semantic_memory", default=None)
            if semantic:
                facts = semantic.search(query, limit=3)
                for f in facts:
                    results.append(f"[Fact] {f.get('content', str(f))[:150]}")
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
        try:
            from core.container import ServiceContainer
            browser = ServiceContainer.get("sovereign_browser", default=None)
            if browser:
                result = await browser.search(query)
                return Observation(
                    content=str(result)[:1500],
                    success=True,
                    source="web"
                )
        except Exception as e:
            logger.debug(f"ReAct: Sovereign browser search failed, trying fallback: {e}")

        # Fallback: try requests-based search
        try:
            import httpx
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    "https://duckduckgo.com/html/",
                    params={"q": query},
                    headers={"User-Agent": "Mozilla/5.0"}
                )
                # Extract first few results from HTML (simplified)
                text = re.sub(r'<[^>]+>', ' ', resp.text)
                text = ' '.join(text.split())[:1000]
                return Observation(content=text, success=True, source="web_ddg")
        except Exception as e:
            return Observation(content=f"Web search failed: {e}", success=False)

    async def _execute_python(self, action: Action) -> Observation:
        """Execute Python in a restricted sandbox. SEC-01: AST-based validation."""
        code = action.params.get("code", "")
        if not code:
            return Observation(content="No code provided", success=False)

        # SEC-01: AST-based validation instead of trivially-bypassable string matching
        import ast
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return Observation(content=f"Syntax error: {e}", success=False)
        
        BLOCKED_MODULES = {"os", "subprocess", "shutil", "sys", "importlib", "ctypes", "socket", "http", "urllib", "pathlib"}
        BLOCKED_BUILTINS = {"eval", "exec", "compile", "__import__", "open", "getattr", "setattr", "delattr", "globals", "locals"}
        
        for node in ast.walk(tree):
            # Block dangerous imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in BLOCKED_MODULES:
                        return Observation(content=f"Blocked: import of '{alias.name}' is not allowed", success=False)
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split(".")[0] in BLOCKED_MODULES:
                    return Observation(content=f"Blocked: import from '{node.module}' is not allowed", success=False)
            # Block dangerous function calls
            elif isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id in BLOCKED_BUILTINS:
                    return Observation(content=f"Blocked: call to '{func.id}' is not allowed", success=False)
                elif isinstance(func, ast.Attribute):
                    if isinstance(func.value, ast.Name) and func.value.id in BLOCKED_MODULES:
                        return Observation(content=f"Blocked: '{func.value.id}.{func.attr}' is not allowed", success=False)

        try:
            import io
            from contextlib import redirect_stdout
            output_buf = io.StringIO()

            # Restricted globals
            safe_globals = {
                "__builtins__": {
                    "print": print, "len": len, "range": range, "int": int,
                    "float": float, "str": str, "list": list, "dict": dict,
                    "sum": sum, "max": max, "min": min, "abs": abs,
                    "round": round, "sorted": sorted, "enumerate": enumerate,
                    "zip": zip, "map": map, "filter": filter,
                }
            }

            import math, json as _json
            safe_globals["math"] = math
            safe_globals["json"] = _json

            exec_result = {}
            with redirect_stdout(output_buf):
                exec(code, safe_globals, exec_result)  # noqa: S102

            output = output_buf.getvalue()
            result_val = exec_result.get("result", "")
            combined = f"{output}\nresult={result_val}" if result_val else output

            return Observation(
                content=combined[:500] if combined.strip() else "Code executed (no output)",
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
    ):
        self.brain = brain
        self.executor = ActionExecutor(orchestrator=orchestrator)
        self.parser = ReActResponseParser()
        self.max_steps = max_steps
        self.simple_threshold = simple_threshold
        self.timeout_seconds = timeout_seconds

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
        if self._is_simple_query(query):
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
                        from core.brain.cognitive_engine import ThinkingMode
                        llm_response = await self.brain.think(
                            prompt,
                            mode=ThinkingMode.DEEP,
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

                    # Check for final answer
                    if action.action_type == ActionType.FINAL_ANSWER:
                        trace.final_answer = action.params.get("text", raw_output)
                        trace.terminated_reason = "final_answer"

                        step = ReActStep(
                            step_number=step_num,
                            thought=thought,
                            action=action,
                            observation=Observation(content="[final answer]"),
                            elapsed_ms=(time.time() - step_start) * 1000,
                        )
                        trace.steps.append(step)
                        logger.info("🧠 ReAct: Completed in %d steps", step_num)
                        
                        trace.total_steps = len(trace.steps)
                        trace.elapsed_ms = (time.time() - start_time) * 1000
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

        yield {"type": "final", "content": trace.final_answer, "total_steps": trace.total_steps, "trace": trace}

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

