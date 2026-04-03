# core/brain/planner.py
from typing import Callable, List, Any, Dict, Optional
import asyncio
import heapq
import math

from core.brain.llm_interface import LLMInterface
from core.brain.trace_logger import TraceLogger

class Plan:
    def __init__(self, steps: List[str], score: float = 0.0, metadata: Optional[Dict] = None):
        self.steps = steps
        self.score = score
        self.metadata = metadata or {}

    def __repr__(self):
        return f"Plan(score={self.score:.3f}, steps={self.steps})"

class Planner:
    """
    Produces candidate plans by expanding actions up to depth and scores them using the llm scorer.
    - action_generator(step_context) -> List[str] supplied by user or simple expansion
    - score_fn(plan: Plan) -> float : uses LLMInterface to score a plan (higher better)
    """

    def __init__(self, llm: LLMInterface, trace: Optional[TraceLogger] = None):
        self.llm = llm
        self.trace = trace

    async def score_plan(self, context: str, plan: Plan, temperature: float = 0.0) -> float:
        """
        Ask LLM to score plan quality (0.0 - 1.0). Keep prompts short for efficiency.
        """
        prompt = f"Rate this plan on a scale 0.0-1.0 for the goal/context.\nCONTEXT:\n{context}\nPLAN:\n" + "\n".join(f"- {s}" for s in plan.steps) + "\n\nScore:"
        raw = await self.llm.generate(prompt, temperature=temperature)
        # parse first float
        s = 0.0
        try:
            # find first float-like substring
            import re
            m = re.search(r"([0-9]*\.?[0-9]+)", raw)
            if m:
                s = float(m.group(1))
                # clamp
                s = max(0.0, min(1.0, s))
        except Exception:
            s = 0.0
        plan.score = s
        if self.trace:
            self.trace.log({"type": "plan_score", "plan": plan.steps, "score": s, "raw": raw})
        return s

    async def generate(self, context: str, action_generator: Callable[[str], List[str]], beam: int = 4, depth: int = 3, max_candidates: int = 16) -> List[Plan]:
        """
        Simple breadth-beam expansion:
        - Start with empty plan
        - Expand each leaf with action_generator(step_context)
        - Keep top-k candidate plans by heuristic (length then score via LLM)
        """
        candidates = [Plan(steps=[] , score=0.0)]
        for d in range(depth):
            nxt = []
            for p in candidates:
                # last context = join of steps
                step_context = context + "\n" + "\n".join(p.steps)
                actions = action_generator(step_context)
                for a in actions:
                    new_steps = p.steps + [a]
                    nxt.append(Plan(steps=new_steps, score=0.0))
            # score candidates quickly with simple heuristic to prune (prefer shorter then LLM score)
            # limit for synchronous performance: sample up to max_candidates then score via llm
            if len(nxt) > max_candidates:
                nxt = nxt[:max_candidates]
            # score with llm in parallel
            scored = []
            coros = [self.score_plan(context, plan) for plan in nxt]
            if coros:
                await asyncio.gather(*coros)
            # sort by score desc
            nxt.sort(key=lambda p: p.score, reverse=True)
            candidates = nxt[:beam]
        return candidates
