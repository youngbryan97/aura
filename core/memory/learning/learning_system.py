"""core/memory/learning/learning_system.py — Canonical location for LearningSystem"""
import json
import logging
import asyncio
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Aura.Learning")

class LearningSystem:
    """Enables Aura to learn from experience and improve over time."""
    
    def __init__(self, vector_memory=None):
        self.logger = logger
        self.vector_memory = vector_memory
        self.execution_history: list = []
        self._execution_history_max = 500
        self.strategy_stats = defaultdict(lambda: {
            "attempts": 0, "successes": 0, "failures": 0, "avg_quality": 0.0
        })
        self.learned_patterns = {
            "successful_queries": [], "failed_queries": [],
            "good_sources": set(), "bad_sources": set(),
        }
        self.skill_performance = defaultdict(lambda: {
            "total_runs": 0, "successful_runs": 0, "last_success_time": None, "common_failures": []
        })
    
    async def record_execution(self, goal: Dict[str, Any], result: Dict[str, Any], skill_used: str, strategy: str = "default"):
        execution_record = {
            "timestamp": datetime.now().isoformat(),
            "goal": goal, "result": result, "skill": skill_used, "strategy": strategy,
            "success": result.get("ok", False),
            "quality_score": self._calculate_quality_score(result)
        }
        self.execution_history.append(execution_record)
        if len(self.execution_history) > self._execution_history_max:
            self.execution_history = self.execution_history[-self._execution_history_max:]
        self._update_strategy_stats(strategy, execution_record)
        self._update_skill_performance(skill_used, execution_record)
        self._learn_from_execution(execution_record)
        if self.vector_memory:
            await self._store_in_long_term_memory(execution_record)
        status = "\u2713 Success" if execution_record["success"] else "\u2717 Failed"
        self.logger.info("Recorded execution: %s/%s - %s (Quality: %.2f)", skill_used, strategy, status, execution_record["quality_score"])
    
    def _calculate_quality_score(self, result: Dict[str, Any]) -> float:
        if not result.get("ok"): return 0.0
        score = 0.5
        if "results" in result:
            score += min(0.3, len(result.get("results", [])) * 0.06)
        if "summary" in result:
            summary_length = len(result.get("summary", ""))
            if summary_length > 100: score += 0.1
            if summary_length > 500: score += 0.1
        if result.get("validation", {}).get("is_valid"): score += 0.2
        if result.get("engine") in ["brave", "wikipedia"]: score -= 0.05
        return min(1.0, score)
    
    def _update_strategy_stats(self, strategy: str, record: Dict[str, Any]):
        stats = self.strategy_stats[strategy]
        stats["attempts"] += 1
        if record["success"]: stats["successes"] += 1
        else: stats["failures"] += 1
        quality = record["quality_score"]
        n = stats["attempts"]
        stats["avg_quality"] = (stats["avg_quality"] * (n - 1) + quality) / n
    
    def _update_skill_performance(self, skill: str, record: Dict[str, Any]):
        perf = self.skill_performance[skill]
        perf["total_runs"] += 1
        if record["success"]:
            perf["successful_runs"] += 1
            perf["last_success_time"] = record["timestamp"]
        else:
            error = record["result"].get("error", "unknown")
            if error not in perf["common_failures"]: perf["common_failures"].append(error)
    
    def _learn_from_execution(self, record: Dict[str, Any]):
        goal = record["goal"]; result = record["result"]; success = record["success"]
        query = goal.get("objective").get("query") if isinstance(goal.get("objective"), dict) else goal.get("objective")
        if query:
            if success and record["quality_score"] > 0.7:
                self.learned_patterns["successful_queries"].append({"query": query, "strategy": record["strategy"], "quality": record["quality_score"]})
                if "results" in result:
                    for res in result["results"][:3]:
                        url = res.get("url", "")
                        if url:
                            import urllib.parse
                            self.learned_patterns["good_sources"].add(urllib.parse.urlparse(url).netloc)
            elif not success:
                self.learned_patterns["failed_queries"].append({"query": query, "strategy": record["strategy"], "error": result.get("error")})
    
    async def _store_in_long_term_memory(self, record: Dict[str, Any]):
        try:
            goal_text = str(record["goal"].get("objective", ""))
            success = record["success"]
            if success and record["quality_score"] > 0.6:
                strategy_text = f"To achieve '{goal_text}', handling intent '{record['goal'].get('intent', 'unknown')}', I used skill '{record['skill']}' with strategy '{record['strategy']}'. Result quality was {record['quality_score']:.2f}."
                metadata = {"type": "strategy", "skill": record["skill"], "scope": f"component:{record['skill']}", "success": success, "timestamp": record["timestamp"]}
                if hasattr(self.vector_memory, 'add_memory'): await asyncio.to_thread(self.vector_memory.add_memory, strategy_text, metadata=metadata)
                elif hasattr(self.vector_memory, 'add'): self.vector_memory.add(strategy_text, metadata=metadata)
        except Exception as e: self.logger.error("Failed to store in long-term memory: %s", e)
    
    def get_best_strategy(self, skill: str, goal: Dict[str, Any]) -> Optional[str]:
        relevant_strategies = {name: stats for name, stats in self.strategy_stats.items() if stats["attempts"] > 0}
        if not relevant_strategies: return None
        strategy_scores = []
        for name, stats in relevant_strategies.items():
            if stats["attempts"] < 3: continue
            success_rate = stats["successes"] / stats["attempts"]
            quality = stats["avg_quality"]
            score = (success_rate * 0.6) + (quality * 0.4)
            strategy_scores.append((name, score))
        if not strategy_scores: return None
        strategy_scores.sort(key=lambda x: x[1], reverse=True)
        best_strategy, best_score = strategy_scores[0]
        if best_score > 0.5: self.logger.info("Recommending strategy '%s' (score: %.2f)", best_strategy, best_score); return best_strategy
        return None
    
    def get_performance_report(self) -> Dict[str, Any]:
        total_executions = len(self.execution_history)
        successful = sum(1 for e in self.execution_history if e["success"])
        return {
            "total_executions": total_executions, "successful_executions": successful,
            "success_rate": successful / total_executions if total_executions > 0 else 0,
            "strategies": {k: v.copy() for k, v in self.strategy_stats.items()},
            "skills": {k: v.copy() for k, v in self.skill_performance.items()},
            "patterns": {
                "successful_queries_count": len(self.learned_patterns["successful_queries"]),
                "failed_queries_count": len(self.learned_patterns["failed_queries"]),
                "trusted_sources_count": len(self.learned_patterns["good_sources"]),
            }
        }
    
    def get_learning_insights(self) -> List[str]:
        insights = []
        report = self.get_performance_report()
        insights.append(f"Current success rate: {report['success_rate'] * 100:.1f}%")
        if self.strategy_stats:
            best_strategy = max(self.strategy_stats.items(), key=lambda x: x[1]["avg_quality"])
            insights.append(f"Most effective strategy: {best_strategy[0]} (quality: {best_strategy[1]['avg_quality']:.2f})")
        if self.learned_patterns["good_sources"]:
            insights.append(f"Trusted sources discovered: {', '.join(list(self.learned_patterns['good_sources'])[:5])}")
        for skill, perf in self.skill_performance.items():
            if perf["total_runs"] > 5:
                insights.append(f"{skill}: {(perf['successful_runs'] / perf['total_runs']) * 100:.1f}% success rate")
        return insights
