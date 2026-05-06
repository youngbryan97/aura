# core/brain/planner.py
from typing import Callable, List, Any, Dict, Optional
import asyncio
import heapq
import math

from core.brain.llm_interface import LLMInterface
from core.brain.trace_logger import TraceLogger
from core.runtime.errors import record_degradation

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
            await self._native_system2_rescore(context, nxt)
            # sort by score desc
            nxt.sort(key=lambda p: p.score, reverse=True)
            candidates = nxt[:beam]
        return candidates

    async def _native_system2_rescore(self, context: str, plans: List[Plan]) -> None:
        """Let Aura's native System 2 substrate arbitrate plan candidates.

        This does not execute plans. It only adds a governed deliberate-search
        score so the ordinary planner benefits from the same backtracking,
        value, simulation, and audit path as the rest of cognition.
        """
        if len(plans) < 2:
            return
        try:
            from core.container import ServiceContainer
            from core.reasoning.native_system2 import SearchAlgorithm, System2SearchConfig

            system2 = ServiceContainer.get("native_system2", default=None)
            if system2 is None:
                return
            ranked = await system2.rank_actions(
                context=context[:1200],
                actions=[
                    {
                        "name": " -> ".join(plan.steps) or "(empty plan)",
                        "prior": max(0.05, plan.score),
                        "metadata": {"index": idx, "score_hint": plan.score},
                    }
                    for idx, plan in enumerate(plans)
                ],
                config=System2SearchConfig(
                    algorithm=SearchAlgorithm.HYBRID,
                    budget=max(12, min(64, len(plans) * 10)),
                    max_depth=2,
                    branching_factor=len(plans),
                    beam_width=min(5, len(plans)),
                    confidence_threshold=0.55,
                ),
                source="planner",
            )
            root = ranked.tree.nodes[ranked.root_id]
            for child_id in root.children_ids:
                child = ranked.tree.nodes.get(child_id)
                if not child or not child.action:
                    continue
                idx = child.action.metadata.get("index")
                if idx is None:
                    continue
                plan = plans[int(idx)]
                original = plan.score
                plan.score = max(0.0, min(1.0, (0.55 * plan.score) + (0.45 * child.mean_value)))
                plan.metadata["native_system2"] = {
                    "search_id": ranked.search_id,
                    "value": child.mean_value,
                    "original_score": original,
                    "will_receipt_id": ranked.receipt.will_receipt_id,
                }
        except Exception as exc:
            record_degradation("planner.native_system2", exc)
