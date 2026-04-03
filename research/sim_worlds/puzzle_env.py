"""research/sim_worlds/puzzle_env.py — Strategic Planning Simulation
========================================================================
A discrete graph traversal puzzle (like a simplified Towers of Hanoi or maze).
Used by RSILab to evaluate multi-step planning heuristics and causal foresight.
"""

from typing import Dict, Any, Tuple, List

class PuzzleEnv:
    """A logic puzzle environment."""
    
    def __init__(self):
        # 1D lock picking puzzle: pins 0-3 must all be 'up'
        self.pins = [False, False, False, False]
        self.steps = 0
        
    def reset(self) -> List[bool]:
        self.pins = [False, False, False, False]
        self.steps = 0
        return self.pins.copy()

    def step(self, pin_index: int) -> Tuple[List[bool], float, bool, Dict]:
        """
        Action: integer index of the pin to toggle (0-3).
        Twist: toggling a pin also toggles the adjacent pin to the right.
        """
        if not (0 <= pin_index <= 3):
            return self.pins.copy(), -1.0, False, {"error": "Invalid action"}
            
        # Toggle target pin
        self.pins[pin_index] = not self.pins[pin_index]
        # Toggle adjacent pin if it exists
        if pin_index + 1 <= 3:
            self.pins[pin_index + 1] = not self.pins[pin_index + 1]
            
        self.steps += 1
        
        # Success if all pins are True
        done = all(self.pins)
        
        # Reward shaping: reward for getting closer to all True, penalty per step
        reward = -0.1
        if done:
            reward = 10.0
            
        return self.pins.copy(), reward, done, {"steps": self.steps}
