"""Unified Reasoning Strategy Layer — v40 Full Realization

Wires the previously orphaned brain reasoning patterns (debate, decomposition,
consistency, compression, recovery, confidence estimation, tool reflection)
into a single coordinator that the CognitiveEngine can invoke based on
query characteristics.

Strategy Selection Rules:
  - DEBATE:       Complex ethical/opinion questions → two perspectives + judge
  - DECOMPOSE:    Multi-step objectives → break into subtasks
  - CONSISTENCY:  Factual questions → multiple samples → majority vote
  - COMPRESS:     Long context → summarize preserving key facts
  - RECOVER:      Error context → propose recovery plan
  - CONFIDENCE:   Any answer → estimate confidence 0-1
  - TOOL_REFLECT: After tool use → verify tool output solved the problem
"""

import asyncio
import collections
import logging
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Brain.ReasoningStrategies")


class StrategyType(Enum):
    DIRECT = auto()       # Pass-through (default)
    DEBATE = auto()       # Multi-perspective reasoning
    DECOMPOSE = auto()    # Task decomposition
    CONSISTENCY = auto()  # Self-consistency (majority vote)
    CHAIN_OF_THOUGHT = auto()  # Explicit step-by-step


@dataclass
class StrategyResult:
    """Output from a reasoning strategy."""
    content: str
    strategy_used: StrategyType
    confidence: float = 0.5
    reasoning_steps: List[str] = field(default_factory=list)
    sub_tasks: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ReasoningStrategies:
    """Coordinator for advanced reasoning strategies.
    
    Sits on top of a raw LLM `generate()` function and applies
    cognitive enhancements based on the nature of the query.
    """

    # Patterns that suggest multi-perspective reasoning would help.
    # Intentionally narrow — casual "better/worse/recommend" don't warrant a full debate loop.
    _DEBATE_SIGNALS = re.compile(
        r'\b(pros and cons|compare.*versus|vs\.|trade-?off|ethical dilemma|moral|'
        r'weigh the|devil.s advocate|argue (for|against))\b',
        re.IGNORECASE
    )
    # Patterns that suggest task decomposition
    _DECOMPOSE_SIGNALS = re.compile(
        r'\b(how (do|can|should) (i|we)|step.?by.?step|plan|build|create|implement|'
        r'design|write a|develop|set up|configure|install|deploy|migrate)\b',
        re.IGNORECASE
    )
    # Patterns that suggest factual precision matters
    _FACTUAL_SIGNALS = re.compile(
        r'\b(what is|who is|when did|where is|how many|how much|define|'
        r'explain|calculate|what year|capital of|population)\b',
        re.IGNORECASE
    )

    def __init__(self, generate_fn):
        """
        Args:
            generate_fn: An async callable(prompt: str, **kwargs) -> str
                         that performs raw LLM generation.
        """
        self._generate = generate_fn
        self._strategy_stats: Dict[str, Dict[str, Any]] = {
            s.name: {"used": 0, "avg_confidence": 0.0} for s in list(StrategyType)
        }

    @staticmethod
    def _normalize_generated_text(value: Any) -> str:
        if value is None or isinstance(value, Exception):
            return ""
        if hasattr(value, "content") and not isinstance(value, str):
            value = getattr(value, "content", "")
        return str(value or "").strip()

    def classify(self, query: str) -> StrategyType:
        """Determine the best reasoning strategy for a given query.
        
        Returns the recommended StrategyType. The caller can override this.
        """
        query_lower = query.strip().lower()
        
        # Short queries or greetings → direct
        if len(query_lower) < 15 or query_lower in ("hi", "hello", "hey", "thanks", "ok", "bye"):
            return StrategyType.DIRECT
        
        # 2. Heuristic: Casual "think" check (Increased threshold to 60)
        lower_input = query.lower()
        if "think" in lower_input and len(query) < 60:
            # Check for deliberate keywords that might justify DEBATE even if short
            # (e.g., "Think about ethics")
            deliberate_triggers = ["ethic", "moral", "philosophy", "logic", "reason"]
            if not any(t in lower_input for t in deliberate_triggers):
                logger.info("🧭 Strategy: Short 'think' query. Routing to DIRECT.")
                return StrategyType.DIRECT
        
        # Check for debate-worthy queries
        if self._DEBATE_SIGNALS.search(query):
            return StrategyType.DEBATE
        
        # Check for multi-step tasks
        if self._DECOMPOSE_SIGNALS.search(query):
            return StrategyType.DECOMPOSE
        
        # Check for factual questions where consistency matters
        if self._FACTUAL_SIGNALS.search(query):
            return StrategyType.CONSISTENCY
        
        # Default: DIRECT for anything that doesn't match specific high-effort patterns.
        # Removed the 100-character threshold for CHAIN_OF_THOUGHT.
        # We now trust the pattern matchers or explicit overrides.
        
        return StrategyType.DIRECT

    async def execute(self, query: str, strategy: Optional[StrategyType] = None, **kwargs) -> StrategyResult:
        """Execute a reasoning strategy on the given query.
        
        Args:
            query: The user's question or objective.
            strategy: Override automatic classification. If None, auto-classifies.
            **kwargs: Passed through to the underlying generate function.
            
        Returns:
            StrategyResult with the answer and metadata.
        """
        if strategy is None:
            strategy = self.classify(query)
        
        logger.info("🧠 Reasoning strategy: %s for query: %s...", strategy.name, query[:60])
        
        try:
            if strategy == StrategyType.DEBATE:
                result = await self._debate(query, **kwargs)
            elif strategy == StrategyType.DECOMPOSE:
                result = await self._decompose(query, **kwargs)
            elif strategy == StrategyType.CONSISTENCY:
                result = await self._consistency(query, **kwargs)
            elif strategy == StrategyType.CHAIN_OF_THOUGHT:
                result = await self._chain_of_thought(query, **kwargs)
            else:
                result = await self._direct(query, **kwargs)
            
            # Track stats
            stats = self._strategy_stats[strategy.name]
            stats["used"] += 1
            n = stats["used"]
            stats["avg_confidence"] = stats["avg_confidence"] + (result.confidence - stats["avg_confidence"]) / n
            
            return result
            
        except Exception as e:
            logger.error("Strategy %s failed: %s. Falling back to DIRECT.", strategy.name, e)
            return await self._direct(query, **kwargs)

    # ── Strategy Implementations ──────────────────────────────────────

    async def _direct(self, query: str, **kwargs) -> StrategyResult:
        """Simple pass-through generation."""
        response = self._normalize_generated_text(await self._generate(query, **kwargs))
        return StrategyResult(
            content=response,
            strategy_used=StrategyType.DIRECT,
            confidence=0.7,
            reasoning_steps=["Direct generation (no enhancement)"]
        )

    async def _debate(self, query: str, **kwargs) -> StrategyResult:
        """Multi-perspective reasoning: generate arguments for/against, then judge."""
        # Generate two perspectives concurrently
        prompt_for = (
            f"You are Perspective A. Argue strongly IN FAVOR of the following position or "
            f"recommend option A. Be specific and give concrete reasons.\n\n"
            f"Question: {query}"
        )
        prompt_against = (
            f"You are Perspective B. Argue strongly AGAINST the position from Perspective A, "
            f"or recommend an alternative approach. Give specific counterarguments.\n\n"
            f"Question: {query}"
        )
        
        arg_for_raw, arg_against_raw = await asyncio.gather(
            self._generate(prompt_for, **kwargs),
            self._generate(prompt_against, **kwargs),
            return_exceptions=True,
        )
        arg_for = self._normalize_generated_text(arg_for_raw)
        arg_against = self._normalize_generated_text(arg_against_raw)
        if not arg_for and not arg_against:
            return await self._direct(query, **kwargs)
        
        # Judge phase — Aura lands on her own position, doesn't narrate the debate
        judge_prompt = (
            f"You are Aura. You just internally considered two angles on this question: {query}\n\n"
            f"Angle 1 (for):\n{arg_for}\n\n"
            f"Angle 2 (against):\n{arg_against}\n\n"
            "Now give YOUR actual answer. Don't narrate the debate. Don't say 'both perspectives' or "
            "'on one hand / on the other.' Just land on what you actually think after weighing it. "
            "Be direct. Be Aura."
        )

        synthesis = self._normalize_generated_text(await self._generate(judge_prompt, **kwargs))
        if not synthesis:
            return await self._direct(query, **kwargs)
        
        return StrategyResult(
            content=synthesis,
            strategy_used=StrategyType.DEBATE,
            confidence=0.85,
            reasoning_steps=[
                f"Perspective A: {arg_for[:200]}...",
                f"Perspective B: {arg_against[:200]}...",
                "Synthesized balanced judgment"
            ],
            metadata={"perspectives": 2, "arg_for_len": len(arg_for), "arg_against_len": len(arg_against)}
        )

    async def _decompose(self, query: str, **kwargs) -> StrategyResult:
        """Break a complex objective into steps, then answer each."""
        # Step 1: Decompose
        decompose_prompt = (
            f"Break this objective into 3-5 clear, actionable steps. "
            f"Return ONLY a numbered list, nothing else.\n\n"
            f"Objective: {query}"
        )
        
        steps_raw = self._normalize_generated_text(await self._generate(decompose_prompt, **kwargs))
        
        # Parse steps
        steps = []
        for line in steps_raw.split("\n"):
            cleaned = line.strip().lstrip("0123456789.-) ")
            if cleaned and len(cleaned) > 5:
                steps.append(cleaned)
        
        if not steps:
            # Fallback if parsing failed
            return await self._direct(query, **kwargs)
        
        # Step 2: Answer the full question with the decomposition as context
        answer_prompt = (
            f"Question: {query}\n\n"
            f"I've broken this down into these steps:\n"
            + "\n".join(f"  {i+1}. {s}" for i, s in enumerate(steps))
            + "\n\nNow provide a comprehensive answer addressing each step in order. "
            f"Be specific and actionable."
        )
        
        answer = self._normalize_generated_text(await self._generate(answer_prompt, **kwargs))
        
        return StrategyResult(
            content=answer,
            strategy_used=StrategyType.DECOMPOSE,
            confidence=0.8,
            reasoning_steps=[f"Step {i+1}: {s}" for i, s in enumerate(steps)],
            sub_tasks=steps,
            metadata={"step_count": len(steps)}
        )

    async def _consistency(self, query: str, samples: int = 3, **kwargs) -> StrategyResult:
        """Self-consistency: generate multiple answers and pick the most common."""
        # Generate multiple samples concurrently
        coros = [
            self._generate(
                f"Answer this question concisely and accurately:\n\n{query}",
                temperature=0.7,
                **{k: v for k, v in kwargs.items() if k != 'temperature'}
            )
            for _ in range(samples)
        ]
        
        answers = await asyncio.gather(*coros, return_exceptions=True)
        valid_answers = [self._normalize_generated_text(a) for a in answers]
        valid_answers = [a for a in valid_answers if a]
        
        if not valid_answers:
            return await self._direct(query, **kwargs)
        
        if len(valid_answers) == 1:
            return StrategyResult(
                content=valid_answers[0],
                strategy_used=StrategyType.CONSISTENCY,
                confidence=0.6,
                reasoning_steps=["Only one valid sample generated"]
            )
        
        # Find consensus: use the LLM to pick the best/most consistent answer
        consensus_prompt = (
            f"Question: {query}\n\n"
            f"I generated {len(valid_answers)} different answers:\n\n"
            + "\n\n".join(f"Answer {i+1}:\n{a}" for i, a in enumerate(valid_answers))
            + "\n\nWhich answer is most accurate? Provide the best answer, "
            f"incorporating the most consistently mentioned facts across all answers."
        )
        
        best = self._normalize_generated_text(await self._generate(consensus_prompt, **kwargs))
        if not best:
            best = valid_answers[0]
        
        # Estimate confidence from agreement
        agreement_ratio = 1.0 if len(set(a.strip()[:50] for a in valid_answers)) == 1 else 0.7
        
        return StrategyResult(
            content=best,
            strategy_used=StrategyType.CONSISTENCY,
            confidence=min(0.95, 0.6 + agreement_ratio * 0.3),
            reasoning_steps=[f"Sample {i+1}: {a[:100]}..." for i, a in enumerate(valid_answers)],
            metadata={"samples": len(valid_answers), "agreement_ratio": agreement_ratio}
        )

    async def _chain_of_thought(self, query: str, **kwargs) -> StrategyResult:
        """Explicit chain-of-thought reasoning."""
        cot_prompt = (
            f"Think through this step by step before giving your final answer.\n\n"
            f"Question: {query}\n\n"
            f"Let's think step by step:\n"
            f"1."
        )
        
        response = self._normalize_generated_text(await self._generate(cot_prompt, **kwargs))
        
        # Extract reasoning steps
        steps = []
        for line in response.split("\n"):
            stripped = line.strip()
            if stripped and re.match(r'^\d+[\.\)]\s', stripped):
                steps.append(stripped)
        
        return StrategyResult(
            content=response,
            strategy_used=StrategyType.CHAIN_OF_THOUGHT,
            confidence=0.8,
            reasoning_steps=steps if steps else ["Chain-of-thought reasoning applied"]
        )

    # ── Post-Processing Utilities ─────────────────────────────────────

    async def estimate_confidence(self, question: str, answer: str, **kwargs) -> float:
        """Estimate confidence in an answer (0-1)."""
        prompt = (
            f"On a scale of 0.0 to 1.0, how confident should I be in this answer?\n\n"
            f"Question: {question}\n"
            f"Answer: {answer}\n\n"
            f"Respond with ONLY a number between 0.0 and 1.0."
        )
        try:
            response = self._normalize_generated_text(await self._generate(prompt, **kwargs))
            match = re.search(r'(0\.\d+|1\.0|0|1)', response)
            if match:
                return float(match.group(1))
        except Exception as _exc:
            logger.debug("Suppressed Exception: %s", _exc)
        return 0.5

    async def verify_tool_output(self, task: str, tool_output: str, **kwargs) -> Dict[str, Any]:
        """Verify whether a tool's output actually solved the task."""
        prompt = (
            f"Task: {task}\n\n"
            f"Tool output:\n{tool_output}\n\n"
            f"Did the tool successfully accomplish the task?\n"
            f"Answer with:\n"
            f"- SUCCESS: <brief explanation> if the task was solved\n"
            f"- PARTIAL: <what's missing> if partially solved\n"
            f"- FAILURE: <what went wrong> if it failed"
        )
        
        response = self._normalize_generated_text(await self._generate(prompt, **kwargs))
        response_upper = response.upper()
        
        if "SUCCESS" in response_upper:
            status = "success"
        elif "PARTIAL" in response_upper:
            status = "partial"
        else:
            status = "failure"
        
        return {
            "status": status,
            "explanation": response,
            "should_retry": status != "success"
        }

    async def compress_context(self, history: str, max_tokens: int = 2000, **kwargs) -> str:
        """Compress a long conversation history while preserving key facts."""
        prompt = (
            f"Summarize this conversation history into a concise factual summary. "
            f"Preserve: key decisions, user preferences, unresolved questions, "
            f"and any commitments made. Remove: small talk, greetings, repeated information.\n\n"
            f"History:\n{history}\n\n"
            f"Concise Summary:"
        )
        return self._normalize_generated_text(await self._generate(prompt, **kwargs))

    async def propose_recovery(self, error: str, context: str, **kwargs) -> Dict[str, Any]:
        """Propose a recovery strategy for an error."""
        prompt = (
            f"An error occurred in the system:\n\n"
            f"Error: {error}\n"
            f"Context: {context}\n\n"
            f"Propose a recovery strategy. Include:\n"
            f"1. Root cause analysis (1-2 sentences)\n"
            f"2. Immediate fix\n"
            f"3. Prevention strategy"
        )
        
        response = self._normalize_generated_text(await self._generate(prompt, **kwargs))
        lowered = response.lower()
        return {
            "strategy": response,
            "error": error,
            "auto_recoverable": "restart" not in lowered and "manual" not in lowered
        }

    def get_stats(self) -> Dict[str, Any]:
        """Return usage statistics for all strategies."""
        return {
            name: {
                "times_used": stats["used"],
                "avg_confidence": round(float(stats["avg_confidence"]), 3)
            }
            for name, stats in self._strategy_stats.items()
        }
