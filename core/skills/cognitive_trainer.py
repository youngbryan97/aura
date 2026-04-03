import asyncio
import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from core.memory.memory_facade import MemoryFacade
from core.skills.base_skill import BaseSkill

logger = logging.getLogger("Skills.CognitiveTrainer")

class TrainingInput(BaseModel):
    dataset_name: str = Field(..., description="Name of the dataset: 'MemoryAgentBench' or 'AgentDrive'")
    limit: int = Field(100, description="Max number of samples to process.")
    dry_run: bool = Field(False, description="Whether to simulate ingestion without writing to memory.")

class CognitiveTrainerSkill(BaseSkill):
    """Integrates high-fidelity 2026 benchmarks into Aura's cognitive substrate."""
    
    name = "cognitive_trainer"
    description = "Ingest training data from MemoryAgentBench or AgentDrive to improve reasoning and memory."
    input_model = TrainingInput
    
    def __init__(self, memory_facade: Optional[MemoryFacade] = None):
        super().__init__()
        self.facade = memory_facade
        self._initialized = False

    async def _add_bench_memory(self, content: str, source: str, mtype: str, dry_run: bool):
        """Standardized helper for adding benchmark memories."""
        if not dry_run:
            await self.facade.add_memory(content, metadata={"source": source, "type": mtype})

    async def execute(self, params: TrainingInput, context: Dict[str, Any]) -> Dict[str, Any]:
        """Entry point for cognitive training."""
        if not self.facade:
            from core.container import ServiceContainer
            self.facade = ServiceContainer.get("memory_facade")
        
        if not self.facade:
            return {"ok": False, "error": "MemoryFacade not available for training."}

        try:
            if params.dataset_name.lower() == "memoryagentbench":
                return await self._ingest_memory_agent_bench(params.limit, params.dry_run)
            elif params.dataset_name.lower() == "agentdrive":
                return await self._ingest_agent_drive(params.limit, params.dry_run)
            else:
                return {"ok": False, "error": f"Unsupported dataset: {params.dataset_name}"}
        except Exception as e:
            logger.error("Training failed: %s", e)
            return {"ok": False, "error": str(e)}

    async def _ingest_memory_agent_bench(self, limit: int, dry_run: bool) -> Dict[str, Any]:
        """Load and ingest MemoryAgentBench from HuggingFace."""
        logger.info("📚 Ingesting MemoryAgentBench (ICLR 2026)...")
        try:
            from datasets import load_dataset
            # Using the identified HF path and one of the valid splits (Accurate_Retrieval)
            dataset = await asyncio.to_thread(load_dataset, "ai-hyz/MemoryAgentBench", split="Accurate_Retrieval", streaming=True)
            
            # Issue 59 Fix: Wrap blocking iteration in a thread
            def _iterate_dataset(ds, limit):
                items = []
                count = 0
                for item in ds:
                    if count >= limit: break
                    items.append(item)
                    count += 1
                return items

            items = await asyncio.to_thread(_iterate_dataset, dataset, limit)
            
            count = 0
            for item in items:
                # Format for memory ingestion
                instruction = item.get("instruction", "")
                reference = item.get("reference", "")
                
                content = f"Cognitive Training Case: {instruction}\nReference Reasoning: {reference}"
                await self._add_bench_memory(content, "MemoryAgentBench", "training_case", dry_run)
                
                count += 1
                if count % 10 == 0:
                    logger.info("Processed %d samples...", count)

            return {"ok": True, "count": count, "message": f"Successfully ingested {count} samples from MemoryAgentBench."}
        except Exception as e:
            logger.error("HuggingFace ingestion failed: %s", e)
            return {"ok": False, "error": f"HF Load Error: {e}"}

    async def _ingest_agent_drive(self, limit: int, dry_run: bool) -> Dict[str, Any]:
        """Ingest AgentDrive-MCQ from the locally unzipped repository."""
        logger.info("🚦 Ingesting AgentDrive-MCQ (2026 Reasoning Bench)...")
        from pathlib import Path
        import json
        from core.config import config
        
        # Issue 60 Fix: Use config paths instead of hardcoded developer path
        base_path = config.paths.project_root / "data" / "training" / "agent_drive" / "data" / "AgentDrive-MCQ" / "extracted_final" / "ALL_JSONS_AgentDrive_MCQ_COLLECTED"
        
        if not base_path.exists():
            return {"ok": False, "error": f"AgentDrive data path not found: {base_path}"}
            
        count = 0
        try:
            # Issue 59: Wrap directory iteration if large, but glob is usually okay.
            # For safety, let's collect files first.
            json_files = list(base_path.glob("*.json"))
            
            for json_file in json_files:
                if count >= limit: break
                
                # Offload blocking IO
                data = await asyncio.to_thread(json_file.read_text, encoding='utf-8')
                item = json.loads(data)
                    
                scenario = item.get("scenario_name", "Unknown")
                desc = item.get("description", "")
                question = item.get("question", "")
                correct = item.get("correct_choice", "")
                reasoning = item.get("reason", "")
                
                content = (
                    f"Driving Reasoning Case ({scenario}):\n"
                    f"Context: {desc}\n"
                    f"Problem: {question}\n"
                    f"Optimal Decision: {correct}\n"
                    f"Rationale: {reasoning}"
                )
                await self._add_bench_memory(content, "AgentDrive", "driving_reasoning", dry_run)
                
                count += 1
                if count % 10 == 0:
                    logger.info("Processed %d AgentDrive samples...", count)

            return {"ok": True, "count": count, "message": f"Successfully ingested {count} scenarios from AgentDrive."}
        except Exception as e:
            logger.error("AgentDrive ingestion failed: %s", e)
            return {"ok": False, "error": str(e)}