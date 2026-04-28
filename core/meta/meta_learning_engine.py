from core.runtime.errors import record_degradation
from core.utils.task_tracker import get_task_tracker
import json
import logging
import time
import uuid
import asyncio
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("Meta.Learning")

from core.adaptation.finetune_pipe import get_finetune_pipe

class MetaLearningEngine:
    """Enables Aura to learn from past experiences by identifying structural similarities
    between tasks and applying successful strategies.
    
    Integrated with CognitiveEngine and VectorMemory.
    """
    
    def __init__(self, vector_memory, cognitive_engine):
        self.vectors = vector_memory
        self.brain = cognitive_engine
        
    async def fingerprint_task(self, task_description: str) -> Optional[List[float]]:
        """Generate a semantic embedding of the task structure using the brain's client.
        """
        if not task_description:
            return None
            
        try:
            # Use the brain's legacy client which has generate_embedding
            if hasattr(self.brain, "client") and self.brain.client:
                return self.brain.client.generate_embedding(task_description)
            else:
                logger.warning("No embedding provider available for fingerprinting.")
                return None
        except Exception as e:
            record_degradation('meta_learning_engine', e)
            logger.error("Failed to fingerprint task: %s", e)
            return None

    async def recall_strategy(self, task_description: str) -> Optional[Dict[str, Any]]:
        """Retrieve relevant past strategies for a new task using semantic search.
        """
        if not self.vectors:
            return None
            
        # We use the text-based search directly now that VectorMemory handles embeddings
        results = self.vectors.search(
            query=task_description, 
            limit=3
        )
        
        # Filter for 'experience' type and check similarity
        for match in results:
            if match.get("metadata", {}).get("type") == "experience":
                # distance in ChromaDB is usually cosine distance, 0.0 is perfect match
                distance = match.get("distance", 1.0)
                if distance < 0.3: # Threshold for high relevance
                    logger.info("🧠 Meta-Learning: Found relevant past experience (Dist: %.2f)", distance)
                    return match
        
        return None

    async def index_experience(self, task: str, outcome: str, successful_tools: List[str], strategy_note: str = ""):
        """Save a completed task and its outcome as a learning experience.
        """
        if not self.vectors:
            return
            
        experience_data = {
            "task": task,
            "outcome": outcome,
            "successful_tools": successful_tools,
            "strategy_note": strategy_note,
            "timestamp": time.time(),
            "type": "experience"
        }
        
        # Create a text representation for the vector store
        text_rep = f"Task: {task}\nOutcome: {outcome}\nTools: {', '.join(successful_tools)}\nNote: {strategy_note}"
        
        # VectorMemory.add handles the embedding internally
        res = self.vectors.add(content=text_rep, metadata=experience_data)
        if asyncio.iscoroutine(res):
            await res
        logger.info("🧠 Meta-Learning: Indexed experience for '%s...'", task[:30])
        
        # ACTIVE LEARNING: Route to LoRA dataset generator if tools were successfully used
        if successful_tools:
            try:
                # We fire and forget this async task so it doesn't block the critical path
                pipe = get_finetune_pipe()
                get_task_tracker().create_task(
                    pipe.register_success(
                        task_description=task,
                        context=strategy_note or "Standard execution context.",
                        reasoning=f"Analyzed objective and selected tools: {successful_tools}",
                        final_action=outcome
                    )
                )
            except Exception as e:
                record_degradation('meta_learning_engine', e)
                logger.error("Failed to route to FinetunePipe: %s", e)