"""research/sim_worlds/market_sim.py — Economic Strategy Simulation
========================================================================
A stochastic trading environment simulating market cycles.
Used by RSILab to evaluate risk-management heuristics and trading algorithms.
"""

from typing import Dict, Any, Tuple
import random

class MarketSim:
    """A stochastic market simulation environment."""
    
    def __init__(self):
        self.cash = 1000.0
        self.shares = 0
        self.price = 100.0
        self.volatility = 0.05
        self.tick = 0
        
    def reset(self) -> Dict[str, Any]:
        self.cash = 1000.0
        self.shares = 0
        self.price = 100.0
        self.volatility = 0.05
        self.tick = 0
        return self._get_state()

    def _get_state(self) -> Dict[str, Any]:
        return {
            "portfolio_value": self.cash + (self.shares * self.price),
            "cash": self.cash,
            "shares": self.shares,
            "price": self.price,
            "tick": self.tick
        }

    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, Dict]:
        """
        Actions: 'buy', 'sell', 'hold'
        """
        reward = 0.0
        # Determine market movement before executing action
        change = random.normalvariate(0, self.volatility)
        self.price = max(1.0, self.price * (1.0 + change))
        
        pre_value = self.cash + (self.shares * self.price)
        
        if action == "buy":
            # Buy 1 share
            if self.cash >= self.price:
                self.cash -= self.price
                self.shares += 1
                
        elif action == "sell":
            # Sell 1 share
            if self.shares > 0:
                self.cash += self.price
                self.shares -= 1
                
        # Calculate reward as change in portfolio value
        post_value = self.cash + (self.shares * self.price)
        reward = post_value - pre_value
        
        self.tick += 1
        
        # Volatility clustering (simplified)
        self.volatility = max(0.01, min(0.15, self.volatility + random.uniform(-0.01, 0.01)))
        
        done = self.tick >= 100 or post_value <= 0
        
        return self._get_state(), reward, done, {}
