"""core/collective/strategic_synthesis.py
Strategic Synthesis Engine: Coordinates specialized swarm agents to solve complex problems.
Produces a unified ExecutionPlan from multi-agent consensus.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional
from core.container import ServiceContainer
from core.planner import ExecutionPlan, ToolCall, PlanSchema
from core.collective.delegator import AgentDelegator

logger = logging.getLogger("Aura.StrategicSynthesis")

class StrategicSynthesizer:
    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator
        self.delegator: Optional[AgentDelegator] = None
        
    def _resolve_delegator(self) -> bool:
        """Lazily resolve the AgentDelegator from the container or orchestrator."""
        if self.delegator:
            return True
        
        # Try ServiceContainer first
        self.delegator = ServiceContainer.get("agent_delegator", default=None)
        if not self.delegator and self.orchestrator:
            # Try orchestrator attribute
            self.delegator = getattr(self.orchestrator, "agent_delegator", None)
            
        return self.delegator is not None

    async def synthesize_strategic_plan(self, goal: str, context: str = "") -> Optional[ExecutionPlan]:
        """
        Orchestrates a multi-agent debate and synthesizes a final ExecutionPlan.
        This is used for high-complexity goals that require architectural, security, or optimization review.
        """
        if not self._resolve_delegator():
            logger.error("StrategicSynthesizer: AgentDelegator not available. Synthesis failed.")
            return None
            
        logger.info("🧠 Beginning Strategic Synthesis for complex goal: %s", goal[:60])
        
        # 1. Define specializing roles for the swarm
        roles = ["architect", "critic", "researcher"]
        
        # 2. Trigger the multi-agent debate
        # We use AgentDelegator's debate capability, but we want the raw outputs for specialized synthesis
        agent_results = []
        tasks = []
        
        for role in roles:
            prompt = f"Goal: {goal}\nContext: {context}\nProvide your specialized analysis and a high-level sequence of steps from your perspective."
            tasks.append(self.delegator.delegate(role, prompt))
            
        # Wait for agents to be spawned (not necessarily finished)
        agent_ids = await asyncio.gather(*tasks)
        agent_ids = [aid for aid in agent_ids if aid]
        
        if not agent_ids:
            logger.warning("StrategicSynthesizer: No agents spawned. Falling back to standard planning.")
            return None
            
        # 3. Wait for agents to complete (with timeout)
        start_time = time.time()
        timeout = 90.0
        while time.time() - start_time < timeout:
            all_done = True
            for aid in agent_ids:
                agent = self.delegator.active_agents.get(aid)
                if not agent or agent.status not in ["COMPLETED", "FAILED"]:
                    all_done = False
                    break
            if all_done:
                break
            await asyncio.sleep(1.0)
            
        # 4. Gather results
        analyses = []
        for aid in agent_ids:
            agent = self.delegator.active_agents.get(aid)
            if agent and agent.status == "COMPLETED" and agent.result:
                analyses.append(f"[{agent.specialty.upper()} PERSPECTIVE]:\n{agent.result}")
            elif agent and agent.status == "FAILED":
                logger.warning("Swarm Agent %s failed during synthesis.", aid)
                
        if not analyses:
            logger.error("StrategicSynthesizer: All swarm agents failed to produce results.")
            return None
            
        # 5. Final Synthesis into an ExecutionPlan
        # We use 'think' with PlanSchema to ensure we get a valid plan out of the synthesis
        brain = ServiceContainer.get("cognitive_engine", default=None)
        if not brain:
            logger.error("StrategicSynthesizer: Cognitive engine not available.")
            return None
            
        combined_perspectives = "\n\n---\n\n".join(analyses)
        
        synthesis_prompt = f"""You are the Master Strategic Architect. You have received specialized analyses from three swarm agents (Architect, Critic, Researcher) regarding a complex goal.
        
        GOAL: {goal}
        
        SWARM ANALYSES:
        {combined_perspectives}
        
        TASK: Synthesize these conflicting or complementary insights into a SINGLE, OPTIMIZED execution plan. 
        Ensure you address the Critic's concerns and follow the Architect's structural guidance.
        
        Provide the plan in RAW JSON following the PlanSchema.
        """
        
        try:
            from core.brain.cognitive_engine import ThinkingMode
            # Use DEEP thinking for the final synthesis to ensure quality
            thought = await brain.think(
                synthesis_prompt,
                response_format=PlanSchema,
                mode=ThinkingMode.DEEP
            )
            
            validated_data = thought.content if isinstance(thought.content, dict) else thought.content.model_dump()
            
            # Convert to ExecutionPlan
            native_tool_calls = [
                ToolCall(
                    tool=tc["tool"],
                    params=tc["params"],
                    output_var=tc.get("output_var")
                )
                for tc in validated_data["tool_calls"]
            ]

            plan = ExecutionPlan(
                goal=goal,
                plan_steps=validated_data["plan_steps"],
                tool_calls=native_tool_calls,
                metadata={
                    "source": "strategic_synthesis",
                    "swarm_agents": agent_ids,
                    "swarm_count": len(analyses)
                }
            )
            
            logger.info("✅ Strategic Synthesis complete. Produced plan with %d steps.", len(plan.plan_steps))
            return plan
            
        except Exception as e:
            logger.error("StrategicSynthesizer: Final synthesis failed: %s", e)
            return None

_synthesizer_instance = None

def get_strategic_synthesizer(orchestrator=None) -> StrategicSynthesizer:
    global _synthesizer_instance
    if _synthesizer_instance is None:
        _synthesizer_instance = StrategicSynthesizer(orchestrator)
    return _synthesizer_instance
