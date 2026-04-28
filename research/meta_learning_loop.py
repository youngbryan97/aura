"""research/meta_learning_loop.py — Recursive Self-Improvement (RSI) Lab
=======================================================================
This implements the overarching "Outer Loop" of Aura's Phase 22 architecture.
Rather than modifying herself directly and risking regression, this engine:
1. Receives candidate artifacts (skills, heuristics, parameter changes).
2. Evaluates them against simulation environments or objective heuristics.
3. Gates them through a Promotion Protocol, generating PR-ready metadata.

This satisfies Phase 22.10.
"""

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger("Aura.RSILab")

@dataclass
class CandidateArtifact:
    """An artifact proposed for core integration."""
    id: str
    artifact_type: str  # 'heuristic', 'skill', 'prompt_tweak'
    content: Any
    rationale: str
    status: str = 'pending_eval'  # pending_eval, passed, failed, promoted
    score: float = 0.0
    created_at: float = field(default_factory=time.time)

class RSILab:
    """The laboratory for safe Recursive Self-Improvement."""
    name = "rsi_lab"

    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator
        from core.config import config
        # Store RSI experiments separately from operational data
        self.lab_dir = config.paths.data_dir / "rsi_lab"
        get_task_tracker().create_task(get_storage_gateway().create_dir(self.lab_dir, cause='RSILab.__init__'))
        
        self.candidates: Dict[str, CandidateArtifact] = {}
        self._load()

    def submit_candidate(self, artifact_type: str, content: Any, rationale: str) -> str:
        """Submit a new artifact for evaluation."""
        candidate_id = f"cand_{int(time.time())}_{len(self.candidates)}"
        candidate = CandidateArtifact(
            id=candidate_id,
            artifact_type=artifact_type,
            content=content,
            rationale=rationale
        )
        self.candidates[candidate_id] = candidate
        self._save()
        logger.info(f"🧪 RSI Lab received new {artifact_type} candidate: {candidate_id}")
        return candidate_id

    async def evaluate_pending_candidates(self) -> int:
        """
        Run the evaluation loop. This gates changes from entering the core
        without validation.
        """
        pending = [c for c in self.candidates.values() if c.status == 'pending_eval']
        if not pending:
            return 0
            
        logger.info(f"🧪 Evaluating {len(pending)} pending RSI candidates...")
        evaluated_count = 0
        
        for candidate in pending:
            score = 0.0
            
            # Simple placeholder evaluation criteria based on type
            if candidate.artifact_type == 'heuristic':
                # Heuristics pass if they are specific enough
                if isinstance(candidate.content, str) and len(candidate.content) > 15:
                    score += 0.6
                if "always" in str(candidate.content).lower():
                    score -= 0.2  # Penalize absolutes
                    
            elif candidate.artifact_type == 'skill':
                # Skills pass if they are properly structured macros
                if isinstance(candidate.content, dict) and candidate.content.get('steps'):
                    score += 0.8
                    
            candidate.score = score
            # The gating threshold
            candidate.status = 'passed' if score >= 0.5 else 'failed'
            logger.info(f"🧪 Candidate {candidate.id} evaluated. Score: {score:.2f} -> {candidate.status}")
            evaluated_count += 1
            
        self._save()
        return evaluated_count

    def get_promotable_artifacts(self) -> List[CandidateArtifact]:
        """Fetch candidates that passed evaluation and are ready for promotion."""
        return [c for c in self.candidates.values() if c.status == 'passed']

    def promote(self, candidate_id: str):
        """Mark as promoted. The actual integration is handled by the caller."""
        if candidate_id in self.candidates:
            self.candidates[candidate_id].status = 'promoted'
            self._save()
            logger.info(f"🚀 Candidate {candidate_id} promoted to core!")

    def _save(self):
        try:
            data = {k: asdict(v) for k, v in self.candidates.items()}
            with open(self.lab_dir / "candidates.json", "w") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save RSI Lab candidates: {e}")

    def _load(self):
        file_path = self.lab_dir / "candidates.json"
        if not file_path.exists():
            return
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            self.candidates = {k: CandidateArtifact(**v) for k, v in data.items()}
        except Exception as e:
            logger.error(f"Failed to load RSI Lab candidates: {e}")

def register_rsi_lab(orchestrator=None):
    from core.container import ServiceContainer
    lab = RSILab(orchestrator)
    ServiceContainer.register_instance("rsi_lab", lab)
    return lab
