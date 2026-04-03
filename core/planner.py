"""Intelligent Planning Engine with Goal Decomposition and Validation.

Features:
1. LLM-based goal decomposition with structured output
2. Tool validation and parameter inference
3. Execution plan optimization
4. Error recovery and fallback strategies
5. Structured logging and metrics
"""

import asyncio
import hashlib
import json
import logging
import re
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, Field
from core.config import config

logger = logging.getLogger("Kernel.Planner")


class ToolType(Enum):
    """Supported tool types with validation schemas."""

    WEB_SEARCH = "web_search"
    BROWSER_ACTION = "browser_action"
    CODING = "coding"
    FILE_OPERATION = "file_operation"
    DATA_ANALYSIS = "data_analysis"
    NATIVE_CHAT = "native_chat"
    INTERACTIVE_SEARCH = "interactive_search"


@dataclass
class ToolSchema:
    """Tool schema definition for validation."""

    name: str
    description: str
    required_params: List[str]
    optional_params: List[str] = field(default_factory=list)
    param_schemas: Dict[str, Any] = field(default_factory=dict)

# ─── 1. CONSTRAINED DECODING SCHEMAS ─────────────────────────────────────────

class ToolCallSchema(BaseModel):
    """Pydantic model for LLM-generated tool calls."""
    tool: str = Field(..., description="The exact name of the tool from available schema.")
    params: Dict[str, Any] = Field(default_factory=dict, description="Arguments required by tool.")
    output_var: Optional[str] = Field(None, description="Variable name to store result.")

class PlanSchema(BaseModel):
    """Pydantic model for LLM-generated execution plans."""
    plan_steps: List[str] = Field(..., description="High-level reasoning steps.")
    tool_calls: List[ToolCallSchema] = Field(..., description="Sequential tool executions.")


@dataclass
class ToolCall:
    """Structured tool call with validation."""

    tool: str
    params: Dict[str, Any]
    output_var: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate tool call structure."""
        if not self.tool or not isinstance(self.tool, str):
            raise ValueError("Tool name must be a non-empty string")
        
        if not isinstance(self.params, dict):
            raise ValueError("Params must be a dictionary")
        
        if self.output_var and not isinstance(self.output_var, str):
            raise ValueError("Output variable must be a string")
    
    def get_param(self, key: str, default: Any = None) -> Any:
        """Safely get parameter value."""
        return self.params.get(key, default)
    
    def validate(self, available_tools: Dict[str, ToolSchema]) -> List[str]:
        """Validate tool call against schema."""
        errors = []
        
        if self.tool not in available_tools:
            errors.append(f"Unknown tool: {self.tool}")
            return errors
        
        schema = available_tools[self.tool]
        
        # Check required parameters
        for param in schema.required_params:
            if param not in self.params:
                errors.append(f"Missing required parameter: {param}")
        
        # Validate parameter types if schemas provided
        for param, value in self.params.items():
            if param in schema.param_schemas:
                param_type = schema.param_schemas[param]
                if not isinstance(value, param_type):
                    errors.append(f"Parameter '{param}' should be {param_type}, got {type(value)}")
        
        return errors


@dataclass
class ExecutionPlan:
    """Complete execution plan with metadata."""

    goal: str
    plan_steps: List[str]
    tool_calls: List[ToolCall]
    replan_budget: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    plan_hash: str = None
    
    def __post_init__(self):
        """Generate plan hash for deduplication."""
        if self.plan_hash is None:
            self.plan_hash = self._generate_hash()
    
    def _generate_hash(self) -> str:
        """Generate hash for plan deduplication."""
        plan_data = {
            "goal": self.goal,
            "plan_steps": self.plan_steps,
            "tool_calls": [
                {"tool": tc.tool, "params": tc.params}
                for tc in self.tool_calls
            ]
        }
        plan_str = json.dumps(plan_data, sort_keys=True)
        return hashlib.sha256(plan_str.encode()).hexdigest()
    
    def is_valid(self) -> Tuple[bool, List[str]]:
        """Validate plan structure."""
        errors = []
        
        if not self.goal or not isinstance(self.goal, str):
            errors.append("Goal must be a non-empty string")
        
        if not isinstance(self.plan_steps, list):
            errors.append("Plan steps must be a list")
        elif not all(isinstance(step, str) for step in self.plan_steps):
            errors.append("All plan steps must be strings")
        
        if not isinstance(self.tool_calls, list):
            errors.append("Tool calls must be a list")
        elif not all(isinstance(tc, ToolCall) for tc in self.tool_calls):
            errors.append("All tool calls must be ToolCall instances")
        
        return len(errors) == 0, errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "goal": self.goal,
            "plan": self.plan_steps,
            "tool_calls": [
                {
                    "tool": tc.tool,
                    "params": tc.params,
                    "output_var": tc.output_var,
                    "metadata": tc.metadata
                }
                for tc in self.tool_calls
            ],
            "metadata": self.metadata,
            "created_at": self.created_at,
            "plan_hash": self.plan_hash
        }


class PlanCache:
    """O(1) LRU cache for execution plans using OrderedDict."""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.cache: OrderedDict[str, ExecutionPlan] = OrderedDict()
    
    def get(self, goal_hash: str) -> Optional[ExecutionPlan]:
        """Get plan from cache, moving it to most-recent position."""
        if goal_hash in self.cache:
            self.cache.move_to_end(goal_hash)
            return self.cache[goal_hash]
        return None
    
    def put(self, goal_hash: str, plan: ExecutionPlan) -> None:
        """Add plan to cache with O(1) LRU eviction."""
        if goal_hash in self.cache:
            self.cache.move_to_end(goal_hash)
        else:
            if len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)  # O(1) evict oldest
            self.cache[goal_hash] = plan
    
    def clear(self) -> None:
        """Clear cache."""
        self.cache.clear()


class Planner:
    """Intelligent planning engine with caching, validation, and optimization.
    
    Responsibilities:
    1. Decompose high-level goals into executable plans
    2. Validate tool calls against schemas
    3. Cache frequently used plans
    4. Handle planning failures with fallback strategies
    5. Provide planning metrics and insights
    """
    
    # Intent patterns for shortcut matching
    INTENT_PATTERNS = {
        "latest_news": r"(newest|latest|top)\s+(story|article|news)\s+(on|from)\s+([a-zA-Z0-9.-]+\.[a-z]{2,})",
        "search_query": r"^(search for|find|look up|what is|who is|tell me about)\s+(.+)$",
        "greeting": r"^(hello|hi|hey|greetings|sup|what's up|how are you)$",
        "browse_site": r"^(go to|visit|browse|open)\s+(https?://\S+|www\.\S+\.\w{2,})"
    }
    
    def __init__(self, cognitive_engine, registry=None):
        """Initialize planner.
        
        Args:
            cognitive_engine: LLM interface for planning
            registry: Skill registry for tool validation

        """
        if cognitive_engine is None:
            raise ValueError("Cognitive engine is required")
        
        self.brain = cognitive_engine
        self.registry = registry
        self.plan_cache = PlanCache()
        self.planning_stats = defaultdict(int)
        
        # Load tool schemas from skills
        self.refresh_tool_schemas()
        
        # Initialize JSON optimizer if available
        try:
            from core.json_repair import SelfHealingJSON
            self.json_optimizer = SelfHealingJSON(self.brain)
        except ImportError:
            self.json_optimizer = None
            logger.warning("JSON optimizer not available")
            
        # Hook Phase 25 Critic Engine
        from core.container import ServiceContainer
        self.critic = ServiceContainer.get("critic_engine", default=None)
    
    def refresh_tool_schemas(self):
        """Rebuild tool schemas from the registry."""
        self.tool_schemas = self._load_tool_schemas()

    def _load_tool_schemas(self) -> Dict[str, ToolSchema]:
        """Load tool schemas from registry or defaults."""
        schemas = {}
        
        # Default core tool schemas
        core_schemas = {
            "web_search": ToolSchema(
                name="web_search",
                description="Search the web for information",
                required_params=["query"],
                optional_params=["deep"],
                param_schemas={"query": str, "deep": bool}
            ),
            "native_chat": ToolSchema(
                name="native_chat",
                description="Engage in conversation",
                required_params=["message"],
                optional_params=["context"],
                param_schemas={"message": str}
            )
        }
        
        # Merge with registry if available
        if self.registry:
            for name, skill in self.registry.skills.items():
                if hasattr(skill, "description"):
                     # Support class-based schemas if they define required_params etc
                     schemas[name] = ToolSchema(
                         name=name,
                         description=skill.description,
                         required_params=getattr(skill, "required_params", []),
                         optional_params=getattr(skill, "optional_params", []),
                         param_schemas=getattr(skill, "param_schemas", {})
                     )
                else:
                    # Fallback for function-based skills
                    schemas[name] = ToolSchema(
                        name=name,
                        description="External function skill",
                        required_params=[]
                    )
        
        # Core always overrides
        schemas.update(core_schemas)
        return schemas
    
    def _detect_intent(self, goal_text: str) -> Optional[Dict[str, Any]]:
        """Detect intent patterns for shortcut planning.
        
        Args:
            goal_text: User goal text
            
        Returns:
            Shortcut plan if pattern matched, None otherwise

        """
        goal_text_lower = goal_text.lower().strip()
        
        # FIX: Don't shortcut if goal involves complex actions (save, extract, write, file)
        # This prevents complex autonomous goals from being truncated to just shallow shortcuts.
        complex_keywords = ["save", "write", "file", "extract", "store", "log", " and ", "deconstruct", "research", "comprehensively"]
        if any(keyword in goal_text_lower for keyword in complex_keywords):
            logger.info("Ignoring shortcuts due to complex goal structure.")
            return None
        
        # Check for latest news pattern
        news_match = re.search(self.INTENT_PATTERNS["latest_news"], goal_text_lower)
        if news_match:
            domain = news_match.group(4)
            logger.info("Shortcut: Latest news intent for %s", domain)
            
            # Smart URL inference
            if "space.com" in domain:
                url = "https://www.space.com/news"
            elif "hacker news" in goal_text_lower or "ycombinator" in domain:
                url = "https://news.ycombinator.com"
            else:
                url = f"https://www.{domain}/news"
            
            return {
                "plan": [
                    f"Navigate to {url} for latest stories",
                    f"Extract top headlines from {domain}"
                ],
                "tool_calls": [
                    ToolCall(
                        tool="browser_action",
                        params={
                            "url": url,
                            "steps": [
                                {"type": "wait", "value": 2},
                                {"type": "extract_headlines", "selector": "article h2"}
                            ],
                            "headless": True
                        },
                        output_var="headlines"
                    )
                ]
            }
        
        # Check for search pattern
        search_match = re.match(self.INTENT_PATTERNS["search_query"], goal_text_lower)
        if search_match:
            query = search_match.group(2)
            logger.info("Shortcut: Search intent for '%s'", query)
            
            return {
                "plan": [
                    f"Search for information about '{query}'",
                    "Analyze search results for relevant content"
                ],
                "tool_calls": [
                    ToolCall(
                        tool="web_search",
                        params={"query": query, "num_results": 5},
                        output_var="search_results"
                    )
                ]
            }
        
        # Check for greetings
        if re.match(self.INTENT_PATTERNS["greeting"], goal_text_lower):
            logger.info("Shortcut: Greeting intent")
            
            return {
                "plan": ["Respond to greeting conversationally"],
                "tool_calls": [
                    ToolCall(
                        tool="native_chat",
                        params={"message": goal_text},
                        output_var="response"
                    )
                ]
            }
        
        return None
    
    async def decompose(self, goal_text: str) -> ExecutionPlan:
        """Decompose high-level goal into an executable plan using Constrained Decoding."""
        # Reliability Check
        try:
            from core.container import ServiceContainer
            reliability = ServiceContainer.get("reliability_engine", default=None)
            if reliability:
                svc_info = reliability.services.get("planner")
                if svc_info and svc_info.circuit_open:
                    logger.warning("🔴 Planner circuit is OPEN. Using fallback plan.")
                    return self._create_fallback_plan(goal_text)
                await reliability.heartbeat("planner", stability=0.95)
        except Exception as _e:
            logger.debug('Ignored Exception in planner.py: %s', _e)

        # 1. Validation & Sanitization
        if not goal_text or not isinstance(goal_text, str):
            raise ValueError("Goal text must be a non-empty string")
        
        goal_text = goal_text.strip()
        if len(goal_text) > 1000:
            logger.warning("Goal text exceeds optimal length (%d chars). Truncating.", len(goal_text))
            goal_text = goal_text[:1000]

        # 2. O(1) Cache Retrieval
        tool_keys = sorted(self.tool_schemas.keys())
        state_str = goal_text + "".join(tool_keys)
        goal_hash = hashlib.sha256(state_str.encode()).hexdigest()[:16]
        cached_plan = self.plan_cache.get(goal_hash)
        if cached_plan:
            logger.info("⚡ Cache Hit: Executing known plan for '%s...'", goal_text[:50])
            self.planning_stats["cache_hits"] += 1
            return cached_plan

        self.planning_stats["total_plans"] += 1

        # 3.5 Complex Goal Detection (Phase 29: Strategic Synthesis)
        strategic_keywords = ["architect", "security", "analyze core", "stress-test", "consensus", "strategic", "multi-agent", "deep analysis"]
        is_strategic = len(goal_text.split()) > 20 or any(kw in goal_text.lower() for kw in strategic_keywords)
        
        if is_strategic and config.get("collective_intelligence_enabled", True):
            try:
                from core.collective.strategic_synthesis import get_strategic_synthesizer
                synthesizer = get_strategic_synthesizer(self.brain.orchestrator if hasattr(self.brain, "orchestrator") else None)
                strategic_plan = await synthesizer.synthesize_strategic_plan(goal_text)
                if strategic_plan:
                    logger.info("⚡ Strategic Synthesis SUCCESS for '%s...'", goal_text[:50])
                    self.plan_cache.put(goal_hash, strategic_plan)
                    return strategic_plan
            except Exception as e:
                logger.warning("Strategic synthesis bypass failed, falling back to LLM: %s", e)

        # 3. Intent Shortcuts (Zero-Compute Bypasses)
        shortcut_plan = self._detect_intent(goal_text)
        if shortcut_plan:
            logger.info("⚡ Intent Match: Bypassing LLM for '%s...'", goal_text[:50])
            self.planning_stats["shortcut_plans"] += 1
            plan = ExecutionPlan(
                goal=goal_text,
                plan_steps=shortcut_plan["plan"],
                tool_calls=shortcut_plan["tool_calls"]
            )
            self.plan_cache.put(goal_hash, plan)
            return plan

        # 4. Strict LLM Generation
        logger.info("🧠 Generating novel execution plan for '%s...'", goal_text[:50])
        
        max_retries = 3
        attempt = 0
        last_error = None
        
        while attempt < max_retries:
            try:
                # Use a fresh working goal to avoid stack contamination (Focus Area 1)
                working_goal = goal_text
                
                # Phase 29: Recall memories before deep reasoning
                try:
                    from core.container import ServiceContainer
                    memory_engine = ServiceContainer.get("long_term_memory_engine", default=None)
                    if memory_engine:
                        relevant_memories = await memory_engine.recall_relevant(working_goal, limit=3)
                        if relevant_memories:
                            memory_context = "\n".join([f"- {m.content}" for m in relevant_memories])
                            working_goal = f"{working_goal}\n\n[Relevant Long-Term Memories]:\n{memory_context}"
                except Exception as e:
                    logger.debug(f"Long-term memory recall failed in planner: {e}")
                    
                prompt = self._build_planning_prompt(working_goal)
                
                # If we had a previous error, instruct the LLM to fix it
                if last_error:
                    prompt += f"\n\nCRITICAL FIX REQUIRED: Your previous attempt failed with error:\n{last_error}\nEnsure valid JSON format and complete all strings. Do not truncate the JSON output."
                
                # AWAITING COGNITIVE ENGINE (Using strict Pydantic response_format)
                thought = await self.brain.think(
                    prompt, 
                    response_format=PlanSchema,
                    mode="deep" # Hardened planning requires deep reasoning
                )
                
                # Because we used a Pydantic constraint, thought.content is
                # guaranteed to adhere to PlanSchema.
                validated_data = thought.content if isinstance(thought.content, dict) else thought.content.model_dump()

                # Map to native dataclasses
                native_tool_calls = [
                    ToolCall(
                        tool=tc["tool"],
                        params=tc["params"],
                        output_var=tc.get("output_var")
                    )
                    for tc in validated_data["tool_calls"]
                ]

                plan = ExecutionPlan(
                    goal=goal_text,
                    plan_steps=validated_data["plan_steps"],
                    tool_calls=native_tool_calls,
                    metadata={"source": "llm_constrained_generation", "retries": attempt}
                )

                # 5. Schema Alignment Validation
                is_valid, errors = plan.is_valid()
                if not is_valid:
                    last_error = f"Capability mismatch: {errors}"
                    attempt += 1
                    logger.error("Capability mismatch in generated plan (attempt %d): %s", attempt, errors)
                    continue

                # 6. Pre-execution Critique (Phase 25)
                if self.critic:
                    judgment = await self.critic.critique_plan(plan, [])
                    if judgment.recommendation == "backtrack":
                        logger.warning("🧠 Critic REJECTED initial plan: %s", judgment.evidence)
                        attempt += 1
                        last_error = f"Critic rejection: {judgment.evidence}"
                        continue

                # 7. State Persistence (Offloaded)
                self.plan_cache.put(goal_hash, plan)
                self.planning_stats["successful_plans"] += 1
                
                # Background the blocking disk operation
                t = asyncio.create_task(asyncio.to_thread(self.save_to_disk, plan))
                t.add_done_callback(lambda t: t.exception() if not t.cancelled() and t.exception() else None)
                
                return plan

            except Exception as e:
                attempt += 1
                last_error = str(e)
                logger.error("Planning Failure on attempt %d: %s", attempt, e)
                
        # Fallback if all retries fail
        logger.error("Fatal Planning Failure after %d retries: %s", max_retries, last_error)
        self.planning_stats["failed_plans"] += 1
        return self._create_fallback_plan(goal_text)
            
    async def revise_plan(self, original_plan: ExecutionPlan, failure_reason: str, failed_step_index: int) -> ExecutionPlan:
        """Revise a plan based on failure feedback.
        Audit Requirement 2: Plan Revision.
        """
        # Reliability Check
        try:
            from core.container import ServiceContainer
            reliability = ServiceContainer.get("reliability_engine", default=None)
            if reliability:
                svc_info = reliability.services.get("planner")
                if svc_info and svc_info.circuit_open:
                    return self._create_fallback_plan(original_plan.goal)
                await reliability.heartbeat("planner", stability=0.85)
        except Exception as _e:
            logger.debug('Ignored Exception in planner.py: %s', _e)

        logger.info("Revising plan due to failure at step %s: %s", failed_step_index, failure_reason)
        
        # Context for LLM
        plan_steps = list(original_plan.plan_steps)
        completed_steps = plan_steps[:failed_step_index]
        failed_step = plan_steps[failed_step_index] if failed_step_index < len(plan_steps) else "Unknown"
        remaining_steps = plan_steps[failed_step_index+1:]

        # 1. Check replan budget to prevent infinite storms (Focus Area 1)
        budget = getattr(original_plan, "replan_budget", 3)
        if budget <= 0:
            logger.error("🛑 Replan budget EXHAUSTED for goal: %s. Halting.", original_plan.goal)
            return self._create_fallback_plan(original_plan.goal)
        
        prompt = f"""You are an Autonomous Planner. A previous plan failed. You must revise the remaining steps.

ORIGINAL GOAL: {original_plan.goal}
COMPLETED STEPS: {completed_steps}
FAILED STEP: {failed_step}
FAILURE REASON: {failure_reason}

REQUIREMENTS:
1. Provide a NEW list of steps starting from the current situation to achieve the goal.
2. Do NOT include completed steps.
3. Try a DIFFERENT approach for the failed step.

AVAILABLE TOOLS:
{self._build_tool_list()}

OUTPUT JSON:
{{
  "plan": ["new step 1", "new step 2"],
  "tool_calls": [ ... ]
}}
"""
        try:
            # Define Plan Schema
            plan_schema = {
                "plan": ["step 1", "step 2"],
                "tool_calls": [
                    {
                        "tool": "tool_name",
                        "params": {},
                        "output_var": "var_name"
                    }
                ]
            }

            # AWAITING COGNITIVE ENGINE (Revision with constraint)
            thought = await self.brain.think(
                prompt,
                response_format=PlanSchema,
                mode="critical"
            )
            
            validated_data = thought.content if isinstance(thought.content, dict) else thought.content.model_dump()
            
            native_tool_calls = [
                ToolCall(tool=tc["tool"], params=tc["params"], output_var=tc.get("output_var"))
                for tc in validated_data["tool_calls"]
            ]
            
            # Create new derived plan with decremented budget
            new_plan = ExecutionPlan(
                goal=original_plan.goal,
                plan_steps=completed_steps + validated_data["plan_steps"], 
                tool_calls=list(original_plan.tool_calls)[:failed_step_index] + native_tool_calls,
                replan_budget=budget - 1,
                metadata={
                    "source": "revision",
                    "original_plan_hash": original_plan.plan_hash,
                    "failure_reason": failure_reason
                }
            )
            # 6. Pre-execution Critique of REVISED plan (Phase 25)
            if self.critic:
                judgment = await self.critic.critique_plan(new_plan, [])
                if judgment.recommendation == "backtrack":
                    logger.warning("🧠 Critic REJECTED revised plan: %s", judgment.evidence)
                    # Fixed Asymmetry: In revision, backtrack also consumes budget and causes retry
                    if budget > 0:
                        return await self.revise_plan(original_plan, failed_step_index, f"Critic rejection: {judgment.evidence}")
                    else:
                        logger.error("🛑 Replan budget EXHAUSTED after critic rejection.")
                        new_plan.metadata["critic_warning"] = judgment.evidence
            
            # Persist revision (Offloaded)
            t = asyncio.create_task(asyncio.to_thread(self.save_to_disk, new_plan))
            t.add_done_callback(lambda t: t.exception() if not t.cancelled() and t.exception() else None)
            return new_plan
            
        except Exception as e:
            logger.error("Plan revision failed: %s", e)
            return self._create_fallback_plan(original_plan.goal)

    def _build_planning_prompt(self, goal_text: str) -> str:
        """Build planning prompt for LLM."""
        available_tools = "\n".join([
            f"- {name}: {schema.description} | Required: {', '.join(schema.required_params)}"
            for name, schema in self.tool_schemas.items()
        ])
        
        return f"""You are an Autonomous Agent Planner. Decompose the goal into a structured execution plan.

GOAL: {goal_text}

AVAILABLE TOOLS:
{available_tools}

REQUIREMENTS:
1. Create a step-by-step plan (list of strings)
2. Generate tool calls with all required parameters
3. Use output variables to chain tool results when needed
4. Be specific with parameter values
5. Ensure plan is executable and complete

OUTPUT FORMAT (JSON):
{{
  "plan": ["step 1 description", "step 2 description"],
  "tool_calls": [
    {{
      "tool": "tool_name",
      "params": {{"param1": "value1", "param2": "value2"}},
      "output_var": "result_variable_name"
    }}
  ]
}}

CRITICAL OUTPUT INSTRUCTIONS:
You must output RAW, valid JSON. Do not wrap the parameters in a secondary "params" object. 
CORRECT: {{"tool": "search", "params": {{"query": "test"}}}}
INCORRECT: {{"tool": "search", "params": {{"params": {{"query": "test"}}}}}}

Return ONLY the JSON object, no additional text."""

    async def _parse_llm_response(self, response: str, goal_text: str) -> Dict[str, Any]:
        """Parse LLM response into structured plan."""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                response = json_match.group(0)
            
            # Parse JSON
            if self.json_optimizer:
                # AWAITING JSON OPTIMIZER
                parsed = await self.json_optimizer.parse(response)
            else:
                parsed = json.loads(response)
            
            # Normalize structure
            plan_steps = parsed.get("plan", [])
            if not isinstance(plan_steps, list):
                plan_steps = [str(plan_steps)]
            
            tool_calls = []
            raw_tool_calls = parsed.get("tool_calls", [])
            
            for raw_tc in raw_tool_calls:
                try:
                    tool_call = ToolCall(
                        tool=raw_tc.get("tool"),
                        params=raw_tc.get("params", {}),
                        output_var=raw_tc.get("output_var"),
                        metadata={"source": "llm_parsed"}
                    )
                    
                    # Validate against schema
                    errors = tool_call.validate(self.tool_schemas)
                    if errors:
                        logger.warning("Tool call validation errors: %s", errors)
                        continue
                    
                    tool_calls.append(tool_call)
                    
                except ValueError as e:
                    logger.warning("Invalid tool call skipped: %s", e)
                    continue
            
            return {
                "plan_steps": plan_steps,
                "tool_calls": tool_calls
            }
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.error("Failed to parse LLM response: %s", e)
            return self._extract_plan_from_text(response, goal_text)

    def _extract_plan_from_text(self, text: str, goal_text: str) -> Dict[str, Any]:
        """Extract plan from unstructured text as fallback."""
        # Simple heuristic extraction
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        plan_steps = []
        tool_calls = []
        
        for line in lines:
            if len(line) < 100:  # Likely a plan step
                plan_steps.append(line)
            elif any(tool in line.lower() for tool in self.tool_schemas.keys()):
                # Try to extract tool call
                for tool_name in self.tool_schemas.keys():
                    if tool_name in line.lower():
                        tool_calls.append(
                            ToolCall(
                                tool=tool_name,
                                params={"query": goal_text},  # Fallback param
                                metadata={"extracted": True}
                            )
                        )
                        break
        
        return {
            "plan_steps": plan_steps or [f"Execute: {goal_text}"],
            "tool_calls": tool_calls or [
                ToolCall(
                    tool="native_chat",
                    params={"message": goal_text},
                    metadata={"fallback": True}
                )
            ]
        }

    def _create_fallback_plan(self, goal_text: str) -> ExecutionPlan:
        """Create fallback plan when planning fails."""
        try:
            from core.synthesis import strip_meta_commentary
        except ImportError:
            strip_meta_commentary = lambda x: x
            
        goal_text = strip_meta_commentary(goal_text)
        logger.info("Creating fallback plan")
        
        return ExecutionPlan(
            goal=goal_text,
            plan_steps=[f"Respond conversationally to: {goal_text}"],
            tool_calls=[
                ToolCall(
                    tool="native_chat",
                    params={"message": goal_text},
                    metadata={"fallback": True}
                )
            ],
            metadata={"source": "fallback"}
        )

    def validate_tool_call(self, tool_call: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate tool call structure."""
        try:
            tc = ToolCall(**tool_call)
            errors = tc.validate(self.tool_schemas)
            return len(errors) == 0, errors
        except ValueError as e:
            return False, [str(e)]

    def get_stats(self) -> Dict[str, Any]:
        """Get planning statistics."""
        return {
            "total_plans": self.planning_stats["total_plans"],
            "successful_plans": self.planning_stats["successful_plans"],
            "failed_plans": self.planning_stats["failed_plans"],
            "cache_hits": self.planning_stats["cache_hits"],
            "shortcut_plans": self.planning_stats["shortcut_plans"],
            "cache_size": len(self.plan_cache.cache),
            "cache_hit_rate": (
                self.planning_stats["cache_hits"] / self.planning_stats["total_plans"] * 100
                if self.planning_stats["total_plans"] > 0 else 0
            )
        }

    def clear_cache(self) -> None:
        """Clear plan cache."""
        self.plan_cache.clear()
        logger.info("Plan cache cleared")

    def _build_tool_list(self) -> str:
        """Helper to formatting tool list."""
        return "\n".join([
            f"- {name}: {schema.description}"
            for name, schema in self.tool_schemas.items()
        ])


    def save_to_disk(self, plan: ExecutionPlan) -> None:
        """Save active plan to disk for resilience."""
        import tempfile
        import os
        try:
            plan_path = config.paths.data_dir / "active_plan.json"
            plan_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write to a secure temp file first
            fd, temp_path = tempfile.mkstemp(dir=plan_path.parent)
            with os.fdopen(fd, 'w') as f:
                json.dump(plan.to_dict(), f, indent=2)
            
            # Atomic rename guarantees file integrity
            os.replace(temp_path, plan_path)
            logger.info("Plan persisted to disk: %s", plan_path)
        except Exception as e:
            logger.error("Failed to persist plan: %s", e)

    def load_from_disk(self) -> Optional[ExecutionPlan]:
        """Load active plan from disk after restart."""
        try:
            from core.config import config
            plan_path = config.paths.data_dir / "active_plan.json"
            if not plan_path.exists():
                return None
                
            with open(plan_path, "r") as f:
                data = json.load(f)
                
            # Reconstruct ExecutionPlan object
            # Note: ToolCall reconstruction is needed
            tool_calls = []
            for tc_data in data.get("tool_calls", []):
                tool_calls.append(ToolCall(
                    tool=tc_data["tool"],
                    params=tc_data["params"],
                    output_var=tc_data.get("output_var"),
                    metadata=tc_data.get("metadata", {})
                ))
                
            return ExecutionPlan(
                goal=data["goal"],
                plan_steps=data["plan"],
                tool_calls=tool_calls,
                metadata=data.get("metadata", {}),
                created_at=data.get("created_at", time.time()),
                plan_hash=data.get("plan_hash")
            )
        except Exception as e:
            logger.error("Failed to load plan from disk: %s", e)
            return None

    def clear_persisted_plan(self):
        """Remove plan from disk after completion."""
        try:
            from core.config import config
            plan_path = config.paths.data_dir / "active_plan.json"
            if plan_path.exists():
                plan_path.unlink()
        except Exception as e:
            logger.error("Failed to clear persisted plan: %s", e)

class PlanningError(Exception):
    """Planning-related exception."""

    pass