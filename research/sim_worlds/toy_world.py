"""research/sim_worlds/toy_world.py — Resource & Knowledge Simulation
========================================================================
A simple grid-like or state-based environment where Aura must balance 
'energy' and 'knowledge' points. 
Used by RSILab to evaluate resource-management heuristics.
"""

from typing import Dict, Any, Tuple
import time

class ToyWorld:
    """A resource management simulation environment."""
    
    def __init__(self):
        self.state = {
            "energy": 100,
            "knowledge": 0,
            "day": 1
        }
        self.history = []

    def reset(self) -> Dict[str, Any]:
        self.state = {"energy": 100, "knowledge": 0, "day": 1}
        self.history = []
        return self.state.copy()

    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, Dict]:
        """
        Actions: 'explore', 'rest', 'synthesize'
        Returns: (next_state, reward, done, info)
        """
        reward = 0.0
        done = False
        
        self.history.append({"day": self.state["day"], "action": action, "pre_state": self.state.copy()})
        
        if action == "explore":
            if self.state["energy"] >= 20:
                self.state["energy"] -= 20
                self.state["knowledge"] += 10
                reward = 1.0
            else:
                reward = -5.0 # Penalty for exhaustion
                
        elif action == "rest":
            self.state["energy"] = min(100, self.state["energy"] + 40)
            
        elif action == "synthesize":
            if self.state["knowledge"] >= 30:
                self.state["knowledge"] -= 30
                reward = 10.0 # Huge payoff for synthesis
            else:
                reward = -2.0 # Wasting time
                
        self.state["day"] += 1
        self.state["energy"] -= 5 # Daily cost of living
        
        if self.state["energy"] <= 0:
            done = True
            reward -= 50 # Starvation/Failure
            
        if self.state["day"] > 30:
            done = True
            
        info = {"history": self.history} if done else {}
        return self.state.copy(), reward, done, info
