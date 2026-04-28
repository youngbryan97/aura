from core.runtime.errors import record_degradation
import json
import logging
from typing import Any, Dict, List, Optional

from core.brain.cognitive_engine import CognitiveEngine

logger = logging.getLogger("Brain.Simulator")

class MentalSimulator:
    """Imagine the consequences of actions before taking them.
    (World Model v2.0 - Graph-Integrated)
    """
    
    def __init__(self, cognitive_engine: CognitiveEngine):
        self.brain = cognitive_engine
        try:
            from core.world_model.belief_graph import belief_graph
            self.world_model = belief_graph
        except ImportError:
            from core.consciousness.world_model import EpistemicState
            self.world_model = EpistemicState()
        
    async def simulate_action(self, action: Dict[str, Any], context: str = "") -> Dict[str, Any]:
        """Simulate an action and return predicted outcome + risk.
        Uses the internal world model to ground the simulation.
        """
        tool_name = action.get("tool")
        params = action.get("params", {})
        
        # Optimization: Skip read-only tools
        if tool_name in ["read_file", "list_dir", "grep_search", "view_file", "search_web"]:
            return {"risk": 0.0, "prediction": "Read-only action. Safe."}
            
        # Get current beliefs to ground the simulation
        beliefs = self.world_model.get_beliefs()
        beliefs_summary = json.dumps(beliefs, indent=2) if beliefs else "No prior world state data."
            
        # Get historical precedents from ACG (v5.2)
        history = []
        try:
            from core.world_model.acg import acg
            history = acg.query_consequences(tool_name, params)
        except Exception as _e:
            record_degradation('mental_simulator', _e)
            logger.debug("ACG lookup failed: %s", _e)
            
        history_summary = "No historical precedents found for this action."
        if history:
            history_summary = json.dumps([
                {"outcome": h["outcome"], "success": h["success"]} 
                for h in history[-3:] # Last 3 matches
            ], indent=2)

        prompt = f"""
        YOU ARE A WORLD SIMULATOR FOR AN AGI.
        
        CURRENT WORLD BELIEFS:
        {beliefs_summary}
        
        HISTORICAL PRECEDENTS (Empirical Data):
        {history_summary}
        
        The agent intends to run the following action:
        Tool: {tool_name}
        Params: {json.dumps(params)}
        
        Context: {context[:500]}
        
        Task:
        1. Predict the STDOUT/Result based on the current world beliefs.
        2. Predict file system changes (files created, modified, deleted).
        3. Predict how this action changes the WORLD BELIEFS (Source:Relation:Target).
        4. Assign a RISK SCORE (0.0 - 1.0), where 1.0 is catastrophic.
        
        Return JSON:
        {{
            "prediction_output": "...",
            "file_changes": ["..."],
            "belief_updates": [
                {{"source": "...", "relation": "...", "target": "...", "confidence": 0.9}}
            ],
            "risk_score": 0.5,
            "risk_reason": "..."
        }}
        """
        
        try:
            # We use FAST mode for simulation to be snappy
            response = await self.brain.think(
                objective=prompt, 
                context={"role": "simulator"},
                mode="fast"
            )
            
            # Parse JSON from response
            import re
            json_match = re.search(r"\{.*\}", response.content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                
                # Proactively update world model with predicted changes if risk is low
                if data.get("risk_score", 1.0) < 0.3:
                    for update in data.get("belief_updates", []):
                        self.world_model.update_belief(
                            update["source"], 
                            update["relation"], 
                            update["target"], 
                            update["confidence"]
                        )
                
                return data
            else:
                return {"risk": 0.1, "prediction": response.content}
                
        except Exception as e:
            record_degradation('mental_simulator', e)
            logger.error("Simulation failed: %s", e)
            return {"risk": 0.5, "error": str(e)}
            
    async def evaluate_risk(self, simulation_result: Dict[str, Any]) -> bool:
        """Return True if safe to proceed, False if unsafe.
        """
        score = simulation_result.get("risk_score", 0.0)
        reason = simulation_result.get("risk_reason", "Unknown")
        
        if score > 0.8: # Adjusted threshold for AGI autonomy
            logger.warning("🛑 HIGH RISK ACTION BLOCKED (%s): %s", score, reason)
            return False
            
        return True