"""skills/train_self.py - Neuroplasticity / Self-Fine-Tuning Skill
Provides the architecture for Aura to learn from her own high-value experiences.
"""
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import asyncio
from core.config import config
from core.skills.base_skill import BaseSkill

logger = logging.getLogger("Aura.Training")

class TrainSelfSkill(BaseSkill):
    """Sleep and Learn: Neuroplasticity simulation.
    Orchestrates the preparation and trigger for local model fine-tuning.
    """

    name = "train_self"
    description = (
        "Neuroplasticity / self-fine-tuning. Collects high-value conversation memories "
        "and distills them into long-term knowledge for future sessions. "
        "Use action='collect_memories' to gather recent interactions, or "
        "action='trigger_tuning' to consolidate them into the knowledge base."
    )

    def __init__(self):
        super().__init__()
        # Issue 86: Use config.paths for dataset location
        self.dataset_path = config.paths.data_dir / "training" / "dataset.jsonl"
        self.dataset_path.parent.mkdir(parents=True, exist_ok=True)

    async def execute(self, goal: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        action = goal.get("action", "collect_memories")
        
        if action == "collect_memories":
            return await self._collect_high_value_memories(context)
        elif action == "trigger_tuning":
            return await self._trigger_finetuning(goal.get("params", {}))
            
        return {"ok": False, "error": f"Unknown action: {action}"}

    async def _collect_high_value_memories(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Gathers successful interactions for future training (v13: no fake data)."""
        try:
            logger.info("Collecting high-value memories for neuroplasticity...")
            
            # Query actual conversation history from context
            history = context.get("history", [])
            collected = 0
            
            if not history:
                return {
                    "ok": True,
                    "message": "No conversation history available to collect memories from.",
                    "collected": 0
                }
            
            with open(self.dataset_path, "a") as f:
                for turn in history[-10:]:  # Last 10 turns
                    # Issue 86: Handle varying turn formats (dict or object with 'role'/'content')
                    role = turn.get("role") if isinstance(turn, dict) else getattr(turn, "role", None)
                    content = turn.get("content") if isinstance(turn, dict) else getattr(turn, "content", "")
                    
                    if role == "assistant":
                        entry = {
                            "instruction": "Continue the conversation naturally.",
                            "input": "",
                            "output": str(content)[:500]
                        }
                        f.write(json.dumps(entry) + "\n")
                        collected += 1
                
            return {
                "ok": True,
                "message": f"Collected {collected} high-value memories for future consolidation.",
                "collected": collected
            }
        except Exception as e:
            logger.error("Memory collection failed: %s", e)
            return {"ok": False, "error": str(e)}

    async def _trigger_finetuning(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Consolidate memories into permanent knowledge (Self-Learning).
        Instead of full fine-tuning (expensive), we distill high-value memories
        into a 'Self-Learned Knowledge' file that is injected into context.
        """
        try:
            logger.info("🧠 Consolidating short-term memories into long-term knowledge...")
            
            # 1. Read dataset
            if not self.dataset_path.exists():
                return {"ok": False, "error": "No memories to consolidate."}
                
            knowledge_path = config.paths.project_root / "core" / "knowledge" / "self_learned.md"
            knowledge_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 2. Distill memories using the actual LLM (Cognitive Engine)
            new_knowledge = []
            def read_memories():
                lines = []
                with open(self.dataset_path, "r") as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            if "output" in data:
                                lines.append(data['output'][:500])
                        except Exception as e:
                            logger.debug("TrainSelf: failed to parse memory line: %s", e)
                return lines

            raw_memories = await asyncio.to_thread(read_memories)
            
            if not raw_memories:
                return {"ok": False, "error": "No valid memories found in dataset."}
                
            try:
                from core.container import ServiceContainer
                from core.brain.types import ThinkingMode
                brain = ServiceContainer.get("cognitive_engine", default=None)
                
                if brain:
                    nl = chr(10)
                    prompt = (
                        "Review these recent conversational interactions. Distill the underlying patterns, "
                        "user preferences, and factual corrections into 3-5 concise bullet points of permanent knowledge. "
                        "Do not just summarize the conversation; extract the core *learned* principles.\n\n"
                        f"Memories:\n{nl.join(raw_memories)}"
                    )
                    
                    logger.info("🧠 Requesting LLM Neuroplastic Distillation...")
                    result = await brain.think(objective=prompt, mode=ThinkingMode.SLOW)
                    
                    if result and hasattr(result, 'content') and result.content:
                        lines = result.content.strip().split("\n")
                        new_knowledge = [line for line in lines if line.strip() and len(line) > 5]
                    else:
                        new_knowledge = ["- [System Event] Distillation yielded empty analytical response."]
                else:
                    logger.warning("Cognitive Engine absent. Falling back to raw aggregation.")
                    new_knowledge = [f"- **Raw Pattern**: {m[:100]}..." for m in raw_memories]
            except Exception as e:
                logger.error("Distillation execution failed: %s", e)
                new_knowledge = [f"- **Failed Pattern Extraction**: {m[:100]}..." for m in raw_memories]
            
            # 3. Append to Knowledge Base
            timestamp = datetime.now().isoformat()
            with open(knowledge_path, "a") as f:
                f.write(f"\n\n### Consolidation {timestamp}\n")
                f.write("\n".join(new_knowledge))
                
            # 4. Clear buffer
            Path(self.dataset_path).write_text("")
            
            return {
                "ok": True,
                "message": f"Consolidated {len(new_knowledge)} insights into Long-Term Memory.",
                "path": knowledge_path
            }
        except Exception as e:
            logger.error("Consolidation failed: %s", e)
            return {"ok": False, "error": str(e)}

    def _get_dataset_size(self) -> int:
        if not os.path.exists(self.dataset_path): return 0
        with open(self.dataset_path, "r") as f:
            return sum(1 for _ in f)