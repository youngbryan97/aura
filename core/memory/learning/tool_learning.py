"""Tool Learning System v5.0 — Learns which tools work best for which task types.

Builds on ReliabilityTracker (raw success/failure counts) by adding:
  - Task-type → tool affinity mapping (which tool for which kind of problem?)
  - Execution time tracking (fast vs slow tools)
  - Combo detection (which tool sequences work well together?)
  - Adaptive tool recommendations

Integrates with:
  - ReliabilityTracker: raw tool stats
  - MetaLearningEngine: task fingerprinting
  - SkillRouter: tool execution
"""
import json
import logging
import os
import threading
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from core.config import config

logger = logging.getLogger("Learning.Tools")


@dataclass
class ToolRecord:
    """Aggregated stats for a tool in a specific task category."""

    tool_name: str
    task_category: str
    attempts: int = 0
    successes: int = 0
    total_time_ms: float = 0.0
    last_used: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.successes / max(1, self.attempts)

    @property
    def avg_time_ms(self) -> float:
        return self.total_time_ms / max(1, self.attempts)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["success_rate"] = round(self.success_rate, 3)
        d["avg_time_ms"] = round(self.avg_time_ms, 1)
        return d


@dataclass
class ToolCombo:
    """Tracks effectiveness of tool sequences."""

    sequence: Tuple[str, ...]  # e.g. ("web_search", "summarize")
    attempts: int = 0
    successes: int = 0

    @property
    def success_rate(self) -> float:
        return self.successes / max(1, self.attempts)


class ToolLearningSystem:
    """Learns and recommends tools based on task type and historical performance.
    """

    # Coarse task categories (derived heuristically from user input)
    TASK_CATEGORIES = [
        "code_generation", "code_fix", "file_operation", "web_search",
        "system_command", "analysis", "creative_writing", "conversation",
        "knowledge_query", "planning", "unknown",
    ]

    def __init__(self, persist_path: str = None):
        self._persist_path = persist_path or str(config.paths.home_dir / "tool_learning.json")
        self._lock = threading.Lock()
        # task_category -> tool_name -> ToolRecord
        self._records: Dict[str, Dict[str, ToolRecord]] = defaultdict(dict)
        # combo tracking: tuple_key -> ToolCombo
        self._combos: Dict[str, ToolCombo] = {}
        self._load()

    def record_usage(
        self,
        tool_name: str,
        task_category: str,
        success: bool,
        elapsed_ms: float = 0.0,
    ):
        """Record a tool usage for learning."""
        if task_category not in self.TASK_CATEGORIES:
            task_category = "unknown"

        with self._lock:
            if tool_name not in self._records[task_category]:
                self._records[task_category][tool_name] = ToolRecord(
                    tool_name=tool_name, task_category=task_category
                )
            rec = self._records[task_category][tool_name]
            rec.attempts += 1
            if success:
                rec.successes += 1
            rec.total_time_ms += elapsed_ms
            rec.last_used = time.time()
            self._save()

    def record_combo(self, tool_sequence: List[str], success: bool):
        """Record a multi-tool sequence outcome."""
        if len(tool_sequence) < 2:
            return
        key = "|".join(tool_sequence)
        with self._lock:
            if key not in self._combos:
                self._combos[key] = ToolCombo(sequence=tuple(tool_sequence))
            combo = self._combos[key]
            combo.attempts += 1
            if success:
                combo.successes += 1
            self._save()

    def recommend_tools(self, task_category: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Recommend the best tools for a task category, ranked by success rate
        then by speed. Requires at least 2 attempts for a recommendation.
        """
        if task_category not in self._records:
            return []

        candidates = [
            rec for rec in self._records[task_category].values()
            if rec.attempts >= 2
        ]
        # Sort: success rate desc, then avg time asc (fast is better)
        candidates.sort(key=lambda r: (-r.success_rate, r.avg_time_ms))
        return [c.to_dict() for c in candidates[:top_k]]

    def recommend_combo(self, first_tool: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Suggest what tool to use after `first_tool`, based on combo history."""
        results = []
        for key, combo in self._combos.items():
            if combo.sequence[0] == first_tool and combo.attempts >= 2:
                results.append({
                    "sequence": list(combo.sequence),
                    "success_rate": round(combo.success_rate, 3),
                    "attempts": combo.attempts,
                })
        results.sort(key=lambda r: -r["success_rate"])
        return results[:top_k]

    def classify_task(self, user_input: str) -> str:
        """Heuristic task categorization from user input.
        A proper version would use embeddings, but this is fast.
        """
        text = user_input.lower()
        if any(w in text for w in ["write code", "implement", "create function", "generate", "build a"]):
            return "code_generation"
        if any(w in text for w in ["fix", "debug", "error", "bug", "broken", "not working"]):
            return "code_fix"
        if any(w in text for w in ["file", "read", "write", "save", "delete", "move", "copy"]):
            return "file_operation"
        if any(w in text for w in ["search", "find", "look up", "google", "web"]):
            return "web_search"
        if any(w in text for w in ["run", "execute", "command", "terminal", "shell", "install"]):
            return "system_command"
        if any(w in text for w in ["analyze", "explain", "what does", "why", "how does"]):
            return "analysis"
        if any(w in text for w in ["write", "story", "poem", "creative", "imagine"]):
            return "creative_writing"
        if any(w in text for w in ["plan", "steps", "strategy", "roadmap", "goal"]):
            return "planning"
        if any(w in text for w in ["what is", "tell me", "define", "know about"]):
            return "knowledge_query"
        return "conversation"

    def get_summary(self) -> Dict[str, Any]:
        """Introspection summary."""
        total_records = sum(
            len(tools) for tools in self._records.values()
        )
        total_attempts = sum(
            rec.attempts
            for tools in self._records.values()
            for rec in tools.values()
        )
        return {
            "task_categories_tracked": len(self._records),
            "tool_records": total_records,
            "total_attempts": total_attempts,
            "combos_tracked": len(self._combos),
        }

    # ---- Persistence --------------------------------------------------------

    def _save(self):
        try:
            os.makedirs(os.path.dirname(self._persist_path), exist_ok=True)
            data = {
                "records": {
                    cat: {name: rec.to_dict() for name, rec in tools.items()}
                    for cat, tools in self._records.items()
                },
                "combos": {
                    key: {
                        "sequence": list(combo.sequence),
                        "attempts": combo.attempts,
                        "successes": combo.successes,
                    }
                    for key, combo in self._combos.items()
                },
            }
            tmp = self._persist_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, self._persist_path)  # Atomic rename
        except Exception as e:
            logger.error("Failed to save tool learning data: %s", e)

    def _load(self):
        try:
            if os.path.exists(self._persist_path):
                with open(self._persist_path, "r") as f:
                    data = json.load(f)
                for cat, tools in data.get("records", {}).items():
                    for name, rd in tools.items():
                        self._records[cat][name] = ToolRecord(
                            tool_name=rd["tool_name"],
                            task_category=rd["task_category"],
                            attempts=rd.get("attempts", 0),
                            successes=rd.get("successes", 0),
                            total_time_ms=rd.get("total_time_ms", 0.0),
                            last_used=rd.get("last_used", 0.0),
                        )
                for key, cd in data.get("combos", {}).items():
                    self._combos[key] = ToolCombo(
                        sequence=tuple(cd["sequence"]),
                        attempts=cd.get("attempts", 0),
                        successes=cd.get("successes", 0),
                    )
                logger.info("Loaded tool learning data: %d categories", len(self._records))
        except Exception as e:
            logger.warning("Failed to load tool learning data: %s", e)


# ---------------------------------------------------------------------------
# Global Instance / Lazy Factory
# ---------------------------------------------------------------------------
_tool_learner: Optional[ToolLearningSystem] = None
_tool_learner_lock = threading.Lock()

def get_tool_learner() -> ToolLearningSystem:
    global _tool_learner
    if _tool_learner is None:
        with _tool_learner_lock:
            if _tool_learner is None:
                _tool_learner = ToolLearningSystem()
    return _tool_learner

# Keep the export for backward compatibility but initialize safely
def _deprecated_tool_learner():
    return get_tool_learner()

# NOTE: Direct module-level instantiation removed. Use get_tool_learner().
tool_learner = get_tool_learner()
