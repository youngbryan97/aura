import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.skills.base_skill import BaseSkill
from core.config import config

from pydantic import BaseModel, Field

class EvolutionInput(BaseModel):
    action: str = Field("propose", description="Action: 'propose' | 'apply' | 'scramble'")
    objective: str = Field(..., description="The feature or improvement to implement.")
    files: Optional[List[str]] = Field(None, description="Specific files to target.")

class SelfEvolutionSkill(BaseSkill):
    """Skill for autonomous self-improvement. Performs the Research -> Synthesize -> Propose loop for code updates."""
    
    name = "self_evolution"
    description = "Research and propose code updates for the agent's own codebase."
    input_model = EvolutionInput
    output = "Proposed implementation plan or patch."
    MAX_PREVIEW = 1000  # Max characters to include in proposal preview

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(f"Skills.{self.name}")
        self.code_base = config.paths.project_root
        self.error_registry = self._load_error_registry()

    @staticmethod
    def _evolution_dir() -> Path:
        path = config.paths.data_dir / "evolution"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _load_error_registry(self) -> Dict:
        """Load or initialize error registry for self-correction."""
        filepath = self._evolution_dir() / "error_registry.json"
        if filepath.exists():
            try:
                with open(filepath, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                self.logger.warning("Could not load error registry: %s", e)
        return {}

    def _save_error_registry(self):
        """Save updated error registry."""
        filepath = self._evolution_dir() / "error_registry.json"
        with open(filepath, "w") as f:
            json.dump(self.error_registry, f)

    async def execute(self, params: EvolutionInput, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute self-evolution loop."""
        if isinstance(params, dict):
            try:
                params = EvolutionInput(**params)
            except Exception as e:
                return {"ok": False, "error": f"Invalid input: {e}"}

        action = params.action
        objective = params.objective

        if action == "scramble":
            return self._perform_scrambling()
            
        if not objective:
            return {"ok": False, "error": "No objective provided for self-evolution."}

        self.logger.info("Initiating Self-Evolution for: %s", objective)

        # 1. RESEARCH - Grounded in Proprioception
        proprioception = context.get("proprioception", {})
        research_summary = f"System Status: {json.dumps(proprioception)}"
        
        # 2. SYNTHESIZE
        try:
            # Look for cognitive_engine in container if not in context
            brain = None
            if "brain" in context:
                brain = context["brain"]
            else:
                from core.container import get_container
                brain = get_container().get("cognitive_engine")
                
            if not brain:
                 return {"ok": False, "error": "No brain accessible for synthesis."}
            
            prompt = (
                f"Objective: {objective}\n"
                f"Body Proprioception: {research_summary}\n\n"
                f"TASK: Perform a self-evolution cycle. Propose code changes to achieve the objective.\n"
                f"Format: Provide a detailed Implementation Plan in Markdown.\n"
                f"Security: Ensure all code matches Aura v14 'Sovereign' protocols."
            )
            from core.brain.cognitive_engine import ThinkingMode
            thought = await brain.think(prompt)
            proposal = thought.content
        except Exception as e:
            error_msg = f"Failed to synthesize proposal: {e}"
            self.error_registry[objective] = error_msg
            self._save_error_registry()
            return {"ok": False, "error": error_msg}

        if action == "apply":
            return await self._apply_evolution(objective, params.files, brain, context)

        # 3. PROPOSE
        timestamp = int(time.time())
        filename = f"evolution_proposal_{timestamp}.md"
        filepath = self._evolution_dir() / filename
        try:
            with open(filepath, "w") as f:
                f.write(f"# Self-Evolution Proposal\n\n**Objective**: {objective}\n\n{proposal}")
            self.logger.info("Proposal saved to %s", filepath)
            return {
                "ok": True,
                "summary": f"Proposal created: {filename}",
                "proposal_path": str(filepath),
                "results": proposal[:self.MAX_PREVIEW]
            }
        except Exception as e:
            return {"ok": False, "error": f"Failed to save proposal: {e}"}

    async def _apply_evolution(self, objective: str, files: Optional[List[str]], brain: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply the evolution objective autonomously using sandboxed verification."""
        self.logger.info("Applying Self-Evolution for: %s", objective)
        
        # 1. Identify target file/line if not provided
        if not files:
             # Use LLM to identify the best file to target
             search_prompt = f"Objective: {objective}\nTask: Identify the SINGLE most relevant file in the codebase to modify."
             search_thought = await brain.think(search_prompt)
             # Basic extraction for now
             target_file = search_thought.content.strip().split('\n')[0].strip('` ')
        else:
             target_file = files[0]

        # 2. Use AutonomousCodeRepair to generate and test
        from core.self_modification.code_repair import AutonomousCodeRepair
        from core.self_modification.self_modification_engine import AutonomousSelfModificationEngine
        
        repair_engine = AutonomousCodeRepair(brain, str(self.code_base))
        
        # Synthetic diagnosis
        diagnosis = {
            "hypotheses": [
                {
                    "root_cause": "Evolution Objective",
                    "explanation": objective,
                    "potential_fix": "Implement requested improvement",
                    "confidence": "high"
                }
            ]
        }
        
        # We need a line number. If it's a new feature, we might target line 1 or a specific function.
        # For simplicity in this upgrade, we target line 1.
        success, fix, test_results = await repair_engine.repair_bug(target_file, 1, diagnosis)
        
        if not success or not fix:
             error_msg = test_results.get("error") if isinstance(test_results, dict) else str(test_results)
             return {"ok": False, "error": f"Evolution failed in sandbox: {error_msg}"}
        
        # 3. Apply via SafeSelfModification (integrated in SME)
        sme = AutonomousSelfModificationEngine(brain, str(self.code_base))
        proposal = {
            "bug": {"pattern": {"events": [{"error_type": "evolution"}]}},
            "fix": fix,
            "test_results": test_results
        }
        
        final_success = await sme.apply_fix(proposal, force=True)
        
        if final_success:
             return {"ok": True, "summary": f"Evolution applied successfully to {target_file}", "applied": True}
        else:
             return {"ok": False, "error": "Failed to apply validated fix to live system."}

    def _perform_scrambling(self) -> Dict[str, Any]:
        """Scramble is disabled. Returns honest status."""
        self.logger.info("Scramble requested but feature is disabled by safety protocols.")
        return {
            "ok": False,
            "status": "disabled",
            "message": "Source scrambling is disabled by Sovereign safety protocols to protect core integrity."
        }
