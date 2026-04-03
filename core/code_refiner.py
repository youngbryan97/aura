import asyncio
import logging
import ast
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger("Aura.CodeRefiner")

__all__ = ["CodeRefinerService", "RefinementProposal", "register_code_refiner"]

@dataclass
class RefinementProposal:
    """A proposal for code refinement."""
    file_path: str
    description: str
    category: str  # 'performance', 'complexity', 'consistency', 'security'
    impact_score: float = 0.5  # 0.0 to 1.0
    original_code: Optional[str] = None
    suggested_code: Optional[str] = None

class CodeRefinerService:
    """Analyzing and refining Aura's own architecture."""
    
    def __init__(self):
        from core.config import config
        self.root_dir = config.paths.project_dir / "core"
        self.proposals: List[RefinementProposal] = []
        logger.info("CodeRefinerService initialized.")

    async def analyze_file(self, file_path: Path) -> List[RefinementProposal]:
        """Analyze a specific file for code smells."""
        if not file_path.exists() or file_path.suffix != ".py":
            return []
            
        proposals = []
        try:
            content = await asyncio.to_thread(file_path.read_text, encoding="utf-8")
            
            tree = ast.parse(content)
            
            # Simple Ast-based heuristics
            # 1. Complexity check (too many functions in one file)
            functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            if len(functions) > 30:
                proposals.append(RefinementProposal(
                    file_path=str(file_path),
                    description=f"High functional density ({len(functions)} functions). Consider refactoring into sub-modules.",
                    category="complexity",
                    impact_score=0.6
                ))
            
            # 2. Large function check
            for func in functions:
                lines = func.end_lineno - func.lineno if func.end_lineno and func.lineno else 0
                if lines > 150:
                    proposals.append(RefinementProposal(
                        file_path=str(file_path),
                        description=f"Large function '{func.name}' ({lines} lines). Consider splitting.",
                        category="complexity",
                        impact_score=0.5
                    ))

            # 3. Synchronous IO in async functions
            # Build parent map for context check
            parent_map = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
            
            for node in ast.walk(tree):
                if isinstance(node, ast.AsyncFunctionDef):
                    for subnode in ast.walk(node):
                        if (
                            isinstance(subnode, ast.Call)
                            and isinstance(subnode.func, ast.Name)
                            and subnode.func.id == "open"
                        ):
                            # Only flag if NOT inside asyncio.to_thread
                            parent = parent_map.get(subnode)
                            is_to_thread = (
                                parent
                                and isinstance(parent, ast.Call)
                                and isinstance(parent.func, ast.Attribute)
                                and parent.func.attr == "to_thread"
                            )
                            if not is_to_thread:
                                proposals.append(RefinementProposal(
                                    file_path=str(file_path),
                                    description="Potential blocking IO 'open' inside async function.",
                                    category="performance",
                                    impact_score=0.7,
                                ))

        except Exception as e:
            logger.error(f"Failed to analyze {file_path}: {e}")
            
        return proposals

        """Audit all core files."""
        all_proposals = []
        files = await asyncio.to_thread(list, self.root_dir.glob("**/*.py"))
        for file in files:
            file_proposals = await self.analyze_file(file)
            all_proposals.extend(file_proposals)
            
        self.proposals = all_proposals
        logger.info(f"Audit complete. Found {len(all_proposals)} refinement targets.")
        return all_proposals

    def get_highest_impact_proposals(self, limit: int = 5) -> List[RefinementProposal]:
        """Fetch priority targets."""
        sorted_proposals = sorted(self.proposals, key=lambda x: x.impact_score, reverse=True)
        return sorted_proposals[:limit]

# Service Registration
def register_code_refiner() -> None:
    """Register the code refiner service."""
    from core.container import ServiceContainer, ServiceLifetime
    ServiceContainer.register(
        "code_refiner",
        factory=CodeRefinerService,
        lifetime=ServiceLifetime.SINGLETON,
    )
