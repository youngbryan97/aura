"""core/self_modification/kernel_refiner.py — v1.0 Kernel Refiner

Proactive analyzer for the CognitiveKernel. 
Hunts for bottlenecks, redundant logic, and regex ulcers.
"""

import ast
import logging
import time
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import asyncio
import hashlib

logger = logging.getLogger("SelfModification.KernelRefiner")

class KernelRefiner:
    """Specialized engine for optimizing the CognitiveKernel's reasoning logic."""
    
    def __init__(self, cognitive_engine, code_base_path: str = "."):
        self.brain = cognitive_engine
        self.code_base = Path(code_base_path)
        self.kernel_path = self.code_base / "core" / "cognitive_kernel.py"
        self._last_audit_time = 0
        self._content_cache = None
        self._cache_hash = None
        
    async def analyze_kernel_health(self) -> List[Dict[str, Any]]:
        """Hunts for optimization opportunities in the CognitiveKernel.
        
        Returns:
            List of optimization proposals.
        """
        logger.info("💎 Refiner: Initiating CognitiveKernel health audit...")
        
        if not self.kernel_path.exists():
            logger.error("Kernel file not found: %s", self.kernel_path)
            return []
            
        ops = []
        
        # 1. Static Analysis (Regex ulcers & Complexity)
        content = await self._get_kernel_content()
        if not content: return []
        
        static_issues = self._perform_static_audit(content)
        ops.extend(static_issues)
        
        # 2. Semantic Analysis (Logic Redundancy)
        if not ops: # Only do deep semantic check if no obvious static ones found
            semantic_issues = await self._perform_deep_brain_audit(content)
            ops.extend(semantic_issues)
            
        return ops

    async def _get_kernel_content(self) -> Optional[str]:
        """Read kernel content with basic caching (Async)."""
        try:
            mtime = self.kernel_path.stat().st_mtime
            if self._content_cache and mtime <= self._last_audit_time:
                return self._content_cache
                
            content = await asyncio.to_thread(self.kernel_path.read_text, encoding='utf-8')
            self._content_cache = content
            self._last_audit_time = mtime
            return content
        except Exception as e:
            logger.error("Failed to read kernel: %s", e)
            return None

    def _perform_static_audit(self, content: str) -> List[Dict[str, Any]]:
        """Static pattern matching for known 'code ulcers'."""
        issues = []
        lines = content.splitlines()
        
        # A. Large pattern lists (potential latency hit)
        pattern_matches = re.finditer(r'(_DOMAIN_PATTERNS|_CHALLENGE_TRIGGERS|_INQUIRY_TRIGGERS)\s*=\s*\[', content)
        for match in pattern_matches:
            # Check length of the list roughly
            start_pos = match.end()
            bracket_count = 1
            idx = start_pos
            while bracket_count > 0 and idx < len(content):
                if content[idx] == '[': bracket_count += 1
                elif content[idx] == ']': bracket_count -= 1
                idx += 1
            
            list_content = content[start_pos:idx]
            items_count = list_content.count(',') + 1
            if items_count > 30:
                line_no = content.count('\n', 0, match.start()) + 1
                issues.append({
                    "type": "performance",
                    "file": "core/cognitive_kernel.py",
                    "line": line_no,
                    "message": f"Large trigger list detected ({items_count} items). High-frequency regex scanning may cause latency spikes.",
                    "priority": "medium"
                })
        
        # B. Nested loops or complex comprehensions
        # (Simplified check for now)
        
        return issues

    async def _perform_deep_brain_audit(self, content: str) -> List[Dict[str, Any]]:
        """uses LLM to look for 'cognitive ulcers' in the reasoning flow."""
        logger.info("🧠 Refiner: Running deep semantic audit via LLM...")
        
        # Read evaluate() method specifically
        tree = ast.parse(content)
        
        evaluate_method = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "evaluate":
                evaluate_method = ast.get_source_segment(content, node)
                break
        
        if not evaluate_method:
            return []
            
        prompt = f"""You are the Kernel Refiner for Aura's CognitiveKernel.
Your task is to find one specific architectural optimization in the reasoning pipeline.

CORE REASONING LOGIC (evaluate() method):
```python
{evaluate_method}
```

Hunt for:
1. Redundant logic: Are there steps that could be consolidated?
2. Latency killers: Complex logic that runs for every single user message.
3. Logical dead-ends: Code that calculates values but doesn't use them effectively.

If you find a valid optimization, return it in this JSON format:
{{
    "found": true,
    "line": <approximate line number in core/cognitive_kernel.py>,
    "type": "optimization",
    "message": "<technical explanation of the bottleneck>",
    "plan": "<briefly how you will refactor it>"
}}
If no refinement is needed, return {{"found": false}}.
"""

        try:
            thought = await self.brain.think(prompt, priority=0.1)
            raw = thought.content
            # Basic JSON extraction
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if not match: return []
            
            data = __import__('json').loads(match.group(0))
            if data.get("found"):
                return [{
                    "type": data["type"],
                    "file": "core/cognitive_kernel.py",
                    "line": data["line"],
                    "message": data["message"],
                    "plan": data["plan"],
                    "priority": "high"
                }]
        except Exception as e:
            logger.error("Deep audit failed: %s", e)
            
        return []

    async def refine_kernel(self, proposal: Dict[str, Any]) -> bool:
        """Executes the proposed refinement using the logic transplantation protocol."""
        logger.info("🧬 Refiner: Initiating Logic Transplantation for: %s", proposal['message'])
        
        # Phase 15 Integration: Delegate to SelfModificationEngine
        # We need the SME instance to apply the fix safely (testing, swarm review, etc.)
        from core.container import ServiceContainer
        sme = ServiceContainer.get("self_modification_engine", default=None)
        
        if not sme:
            logger.error("Refinement failed: SelfModificationEngine not found in ServiceContainer.")
            return False
            
        try:
            # report_optimization handles LLM fix generation, sandboxed testing, and permanent application
            success = await sme.report_optimization(proposal)
            
            if success:
                logger.info("✅ Refinement successful: %s applied via SME.", proposal['type'])
                # Emit success to thought stream if available
                try:
                    from core.thought_stream import get_emitter
                    get_emitter().emit("Kernel Evolved 💎", f"Logic optimized in {proposal['file']}", level="success")
                except Exception: pass
            else:
                logger.warning("❌ Refinement rejected or failed during SME application loop.")
                
            return success
        except Exception as e:
            logger.error("Critical error during kernel refinement: %s", e)
            return False
