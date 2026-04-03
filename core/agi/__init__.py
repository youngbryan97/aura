"""core/agi — AGI coordination layer.

Modules:
  curiosity_explorer    — Translates curiosity signal into actual learning
  skill_synthesizer     — Detects capability gaps, synthesizes new skills
  hierarchical_planner  — Strategic → tactical → operational goal decomposition
  causal_world_model    — Cause-effect tracking for counterfactual simulation
"""
from .curiosity_explorer import CuriosityExplorer, get_curiosity_explorer
from .skill_synthesizer import SkillSynthesizer, get_skill_synthesizer
from .hierarchical_planner import HierarchicalPlanner, get_hierarchical_planner

__all__ = [
    "CuriosityExplorer", "get_curiosity_explorer",
    "SkillSynthesizer", "get_skill_synthesizer",
    "HierarchicalPlanner", "get_hierarchical_planner",
]
