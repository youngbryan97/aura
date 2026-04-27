from core.utils.task_tracker import get_task_tracker
import asyncio
import logging
import time
from typing import Dict, Any, List, Optional
from core.state.aura_state import AuraState
from core.container import ServiceContainer

logger = logging.getLogger("Aura.CellularSubstrate")

class CellularSubstrate:
    """Unified mutation aggregator for Aura's state.
    
    Instead of multiple components competing to commit their version of the 
    entire state, they submit granular 'patches'. The substrate merges these 
    patches at high frequency (~10Hz) and commits the unified state.
    """
    
    def __init__(self, commit_interval: float = 0.1):
        self._patch_queue: asyncio.Queue = asyncio.Queue()
        self._commit_interval = commit_interval
        self._is_active = False
        self._substrate_task: Optional[asyncio.Task] = None
        
    async def initialize(self):
        self._is_active = True
        self._substrate_task = get_task_tracker().create_task(self._substrate_loop())
        logger.info("♾️ [CELLULAR] Substrate loop active (%.1fHz).", 1/self._commit_interval)
        
    async def shutdown(self):
        self._is_active = False
        if self._substrate_task:
            self._substrate_task.cancel()

    def submit_patch(self, patch: Dict[str, Any]):
        """Submit a granular state delta (e.g., {'motivation': {'energy': {'level': 90}}})."""
        self._patch_queue.put_nowait(patch)

    async def _substrate_loop(self):
        while self._is_active:
            try:
                await asyncio.sleep(self._commit_interval)
                
                # 1. Collect all pending patches
                patches = []
                while not self._patch_queue.empty():
                    patches.append(self._patch_queue.get_nowait())
                
                if not patches:
                    continue
                
                # 2. Fetch current memory state (The Truth)
                repo = ServiceContainer.get("state_repository")
                if not repo: continue
                
                state: AuraState = await repo.get_state()
                if not state: continue
                
                # 3. Apply all patches (Sequential Merge)
                # Note: In Ph4 we'll use three_way_merge for complex branches,
                # but for most 'cellular' metabolic updates, sequential is fine.
                for patch in patches:
                    self._apply_patch_recursive(state, patch)
                
                # 4. Single Unified Commit
                await repo.commit(state, cause="cellular_unification")
                
            except Exception as e:
                logger.error("🛑 [CELLULAR] Substrate crash: %s", e)
                await asyncio.sleep(1.0)

    def _apply_patch_recursive(self, target: Any, patch: Dict[str, Any]):
        """Deeply apply a dictionary patch to a dataclass or dict."""
        for key, value in patch.items():
            if isinstance(value, dict) and hasattr(target, key):
                sub_target = getattr(target, key)
                if isinstance(sub_target, dict):
                    sub_target.update(value)
                else:
                    self._apply_patch_recursive(sub_target, value)
            elif hasattr(target, key):
                # Direct assignment
                setattr(target, key, value)
            elif isinstance(target, dict):
                # Dictionary path
                if isinstance(value, dict) and isinstance(target.get(key), dict):
                    target[key].update(value)
                else:
                    target[key] = value

    def three_way_merge(self, base: Dict, ours: Dict, theirs: Dict) -> Dict:
        """
        Cooperative state merging. 
        Note: Currently used for manual conflict resolution if sequential patches 
        fail version checks, but cellular submit_patch usually bypasses this.
        """
        import copy
        result = copy.deepcopy(theirs)
        
        # Simple implementation: Diffs from base -> ours are applied to theirs
        # if they don't overwrite their existing recent changes.
        for key, our_val in ours.items():
            if key not in base or our_val != base[key]:
                # We changed it
                if key not in theirs or theirs[key] == base.get(key):
                    # They didn't change it, safe to apply
                    result[key] = our_val
                else:
                    # Conflict! Both changed it.
                    # Policy: Metabolic data (motivation) prefers repository 'theirs'
                    # Cognitive data (goals) prefers 'ours'.
                    if key in ["motivation", "affect"]:
                        result[key] = theirs[key]
                    else:
                        result[key] = our_val
        return result
                    
def get_cellular_substrate() -> CellularSubstrate:
    """Access the unified substrate singleton."""
    return ServiceContainer.get("cellular_substrate")
