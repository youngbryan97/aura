"""core/embodiment/games/nethack/env.py — NetHack Environment Wrapper.

Connects the NetHack game process (via pexpect or similar) to Aura's sensory gate.
Handles step-by-step execution and returns parsed observations.
"""
from __future__ import annotations

import os
import pexpect
import logging
import asyncio
from typing import Dict, List, Any, Tuple, Optional

from .parser import NetHackParser

logger = logging.getLogger("Aura.Embodiment.NetHack.Env")

class NetHackEnv:
    """Wrapper for the NetHack process."""
    
    def __init__(self, cmd: str = "nethack"):
        self.cmd = cmd
        self.child: Optional[pexpect.spawn] = None
        self.parser = NetHackParser()
        self.last_obs: Dict[str, Any] = {}
        
    async def reset(self) -> Dict[str, Any]:
        """Start a new game."""
        if self.child:
            self.child.close(force=True)
            
        # Run nethack in a pseudo-terminal with a stable TERM and preserved PATH
        import os
        merged_env = os.environ.copy()
        merged_env["TERM"] = "vt100"
        self.child = pexpect.spawn("/opt/homebrew/bin/nethack", encoding='utf-8', dimensions=(24, 80), env=merged_env)
        # Wait for the first screen to populate buffer
        try:
            self.child.expect(pexpect.TIMEOUT, timeout=1.0)
        except:
            pass
            
        raw_text = str(self.child.before or "") + str(self.child.after or "")
        logger.debug(f"NetHack Initial Screen: {raw_text[:200]}...")
        self.last_obs = self.parser.parse(raw_text)
        return self.last_obs
        
    async def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Execute one action (keystroke)."""
        if not self.child:
            raise RuntimeError("Env not reset")
            
        # Map action strings to keystrokes
        keys = self._map_action(action)
        if not self.child.isalive():
            logger.error("NetHack process died unexpectedly.")
            return self.last_obs, 0.0, True, {"error": "process_died"}
            
        self.child.send(keys)
        
        # Wait for update
        try:
            self.child.expect(pexpect.TIMEOUT, timeout=0.1)
        except:
            pass
        
        raw_text = str(self.child.before or "") + str(self.child.after or "")
        obs = self.parser.parse(raw_text)
        
        # Simple reward: HP change + Gold change
        reward = 0.0
        if self.last_obs:
            reward += (obs["vitals"]["hp"] - self.last_obs["vitals"]["hp"]) * 0.1
            
        done = "Do you want your possessions identified?" in raw_text or "DYWYPI" in raw_text
        self.last_obs = obs
        
        return obs, reward, done, {}
        
    def _map_action(self, action: str) -> str:
        # Standard NetHack movement
        mapping = {
            "move_n": "k", "move_s": "j", "move_e": "l", "move_w": "h",
            "move_ne": "u", "move_nw": "y", "move_se": "n", "move_sw": "b",
            "attack_n": "k", "attack_s": "j", "attack_e": "l", "attack_w": "h",
            "wait": ".", "search": "s", "pray": "#pray\n", "eat": "e",
            "pick_up": ",", "inventory": "i", "descend": ">", "ascend": "<"
        }
        return mapping.get(action, action) # Default to raw string if not mapped
