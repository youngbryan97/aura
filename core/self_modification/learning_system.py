"""Self-Improvement Learning System
Learns from successful and failed modification attempts.
"""
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("SelfModification.Learning")


@dataclass
class FixStrategy:
    """Represents a learned fix strategy"""

    strategy_type: str  # e.g., "add_error_handling", "fix_import", "null_check"
    success_count: int
    failure_count: int
    avg_confidence: float
    contexts: List[str]  # Contexts where this strategy worked
    
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.0
        return self.success_count / total
    
    def to_dict(self):
        return {
            "strategy_type": self.strategy_type,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": self.success_rate(),
            "avg_confidence": self.avg_confidence,
            "contexts": self.contexts[:5]  # Top 5
        }


class FixStrategyClassifier:
    """Classifies fixes into strategy types for learning.
    """
    
    def __init__(self):
        self.patterns = {
            "add_error_handling": [
                "try:", "except", "raise", "error handling"
            ],
            "fix_import": [
                "import", "from", "ImportError"
            ],
            "null_check": [
                "if not", "is None", "is not None", "NoneType"
            ],
            "type_conversion": [
                "int(", "str(", "float(", "TypeError"
            ],
            "boundary_check": [
                "if len(", "IndexError", "range("
            ],
            "default_value": [
                "or None", ".get(", "default="
            ],
            "refactor": [
                "def ", "class ", "refactor"
            ]
        }
    
    def classify_fix(self, fix) -> str:
        """Determine what type of fix strategy was used.
        
        Args:
            fix: CodeFix object
            
        Returns:
            Strategy type string

        """
        fixed_code = (fix.fixed_code or "").lower()
        explanation = (fix.explanation or "").lower()
        combined = fixed_code + " " + explanation
        
        # Score each strategy
        scores = {}
        for strategy, keywords in self.patterns.items():
            score = sum(1 for kw in keywords if kw in combined)
            if score > 0:
                scores[strategy] = score
        
        # Return best match or 'other'
        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]
        else:
            return "other"


class SelfImprovementLearning:
    """Learns from modification attempts to improve future fixes.
    """
    
    def __init__(self, learning_db: Optional[str] = None):
        if learning_db is None:
            from core.config import config
            self.db_path = config.paths.data_dir / "learning.json"
        else:
            self.db_path = Path(learning_db)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.classifier = FixStrategyClassifier()
        
        # Learning storage
        self.strategies: Dict[str, FixStrategy] = {}
        self.error_type_strategies: Dict[str, List[str]] = defaultdict(list)
        
        # Load existing knowledge
        self._load_knowledge()
        
        logger.info("SelfImprovementLearning initialized")
    
    def record_fix_attempt(
        self,
        fix,  # CodeFix object
        error_type: str,
        success: bool,
        context: Dict[str, Any]
    ):
        """Record a fix attempt to learn from.
        
        Args:
            fix: The CodeFix that was attempted
            error_type: Type of error being fixed
            success: Whether the fix worked
            context: Additional context

        """
        # Classify the fix strategy
        strategy_type = self.classifier.classify_fix(fix)
        
        logger.info("Recording %s fix attempt: %s", strategy_type, 'success' if success else 'failure')
        
        # Update strategy statistics
        if strategy_type not in self.strategies:
            self.strategies[strategy_type] = FixStrategy(
                strategy_type=strategy_type,
                success_count=0,
                failure_count=0,
                avg_confidence=0.0,
                contexts=[]
            )
        
        strategy = self.strategies[strategy_type]
        
        # Update counts
        if success:
            strategy.success_count += 1
        else:
            strategy.failure_count += 1
        
        # Update confidence (running average)
        confidence_value = {'high': 1.0, 'medium': 0.5, 'low': 0.0}.get(fix.confidence, 0.5)
        total_attempts = strategy.success_count + strategy.failure_count
        strategy.avg_confidence = (
            (strategy.avg_confidence * (total_attempts - 1) + confidence_value) / total_attempts
        )
        
        # Track context (Slicing to prevent memory bloat - Issue 66)
        file_name = fix.target_file.split('/')[-1] if fix.target_file else "unknown"
        context_str = f"{error_type}:{file_name}"
        if context_str not in strategy.contexts:
            strategy.contexts.append(context_str)
            if len(strategy.contexts) > 50:
                strategy.contexts = strategy.contexts[-50:]
        
        # Track which strategies work for which errors
        if success and strategy_type not in self.error_type_strategies[error_type]:
            self.error_type_strategies[error_type].append(strategy_type)
        
        # Persist
        self._save_knowledge()
    
    def suggest_strategy(
        self,
        error_type: str,
        context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Suggest the best fix strategy based on learned patterns.
        
        Args:
            error_type: Type of error to fix
            context: Error context
            
        Returns:
            Strategy suggestion or None

        """
        # Get strategies that have worked for this error type
        candidate_strategies = self.error_type_strategies.get(error_type, [])
        
        if not candidate_strategies:
            logger.debug("No learned strategies for %s", error_type)
            return None
        
        # Score strategies
        scored_strategies = []
        for strategy_name in candidate_strategies:
            strategy = self.strategies.get(strategy_name)
            if strategy:
                # Score based on success rate and confidence
                score = strategy.success_rate() * strategy.avg_confidence
                scored_strategies.append((strategy_name, score, strategy))
        
        if not scored_strategies:
            return None
        
        # Return best strategy
        best_name, best_score, best_strategy = max(scored_strategies, key=lambda x: x[1])
        
        logger.info("Suggesting strategy '%s' for %s (score: %.2f)", best_name, error_type, best_score)
        
        return {
            "strategy_type": best_name,
            "confidence": best_strategy.avg_confidence,
            "success_rate": best_strategy.success_rate(),
            "guidance": self._get_strategy_guidance(best_name)
        }
    
    def _get_strategy_guidance(self, strategy_type: str) -> str:
        """Get human-readable guidance for a strategy"""
        guidance = {
            "add_error_handling": "Wrap risky operations in try/except blocks",
            "fix_import": "Verify imports exist and are correctly specified",
            "null_check": "Add None checks before using variables",
            "type_conversion": "Ensure correct type conversions with error handling",
            "boundary_check": "Validate array/list indices before access",
            "default_value": "Use .get() with defaults or provide fallback values",
            "refactor": "Restructure code for clarity and correctness"
        }
        return guidance.get(strategy_type, "Apply general best practices")
    
    def get_strategy_report(self) -> List[Dict[str, Any]]:
        """Get report of all learned strategies"""
        strategies = []
        for strategy in self.strategies.values():
            strategies.append(strategy.to_dict())
        
        # Sort by success count
        strategies.sort(key=lambda s: s["success_count"], reverse=True)
        return strategies

    def get_recent_lessons(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Retrieve recent learned lessons/strategies with safe slicing (Issue 66)."""
        report = self.get_strategy_report()
        return report[:limit]
    
    def analyze_failure_pattern(
        self,
        recent_failures: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze recent failures to identify what's going wrong.
        
        Args:
            recent_failures: List of failed fix attempts
            
        Returns:
            Analysis with recommendations

        """
        if not recent_failures:
            return {"patterns": [], "recommendation": "No recent failures"}
        
        # Group by error type
        error_types = defaultdict(int)
        failed_strategies = defaultdict(int)
        
        for failure in recent_failures:
            error_type = failure.get("error_type", "unknown")
            strategy = failure.get("strategy_type", "unknown")
            
            error_types[error_type] += 1
            failed_strategies[strategy] += 1
        
        # Identify patterns
        patterns = []
        
        # Pattern 1: Same error type keeps failing
        most_common_error = max(error_types.items(), key=lambda x: x[1])
        if most_common_error[1] >= 3:
            patterns.append({
                "type": "recurring_error",
                "error_type": most_common_error[0],
                "count": most_common_error[1],
                "recommendation": f"Review approach to {most_common_error[0]} errors"
            })
        
        # Pattern 2: Specific strategy keeps failing
        most_failed_strategy = max(failed_strategies.items(), key=lambda x: x[1])
        if most_failed_strategy[1] >= 2:
            patterns.append({
                "type": "failing_strategy",
                "strategy": most_failed_strategy[0],
                "count": most_failed_strategy[1],
                "recommendation": f"Avoid {most_failed_strategy[0]} strategy temporarily"
            })
        
        return {
            "patterns": patterns,
            "total_failures": len(recent_failures),
            "unique_errors": len(error_types),
            "recommendation": "Consider manual review if failures continue"
        }
    
    def _load_knowledge(self):
        """Load learned knowledge from disk"""
        if not self.db_path.exists():
            logger.info("No existing learning database")
            return
        
        try:
            with open(self.db_path, 'r') as f:
                data = json.load(f)
            
            # Reconstruct strategies
            for strategy_data in data.get("strategies", []):
                # H-01 FIX: success_rate is a computed property/method, not an __init__ arg
                if isinstance(strategy_data, dict) and "success_rate" in strategy_data:
                    del strategy_data["success_rate"]
                
                strategy = FixStrategy(**strategy_data)
                self.strategies[strategy.strategy_type] = strategy
            
            # Reconstruct error type mappings
            self.error_type_strategies = defaultdict(
                list,
                data.get("error_type_strategies", {})
            )
            
            logger.info("Loaded %d learned strategies", len(self.strategies))
            
        except Exception as e:
            logger.error("Failed to load learning database: %s", e)
    
    def _save_knowledge(self):
        """Save learned knowledge to disk"""
        try:
            data = {
                "strategies": [s.to_dict() for s in self.strategies.values()],
                "error_type_strategies": dict(self.error_type_strategies),
                "last_updated": time.time()
            }
            
            with open(self.db_path, 'w') as f:
                json.dump(data, f, indent=2)
            
        except Exception as e:
            logger.error("Failed to save learning database: %s", e)


class MetaLearning:
    """Meta-level learning: Learning about the learning process itself.
    """
    
    def __init__(self):
        self.performance_history: list = []
        self._performance_history_max = 200
        logger.info("MetaLearning initialized")
    
    def record_learning_cycle(
        self,
        attempts: int,
        successes: int,
        strategies_used: List[str],
        time_spent: float
    ):
        """Record performance of a learning cycle.
        
        Args:
            attempts: Number of fix attempts
            successes: Number of successful fixes
            strategies_used: Which strategies were tried
            time_spent: Total time for cycle

        """
        cycle_data = {
            "timestamp": time.time(),
            "attempts": attempts,
            "successes": successes,
            "success_rate": successes / attempts if attempts > 0 else 0,
            "strategies_used": strategies_used,
            "time_spent": time_spent,
            "efficiency": successes / time_spent if time_spent > 0 else 0
        }
        
        self.performance_history.append(cycle_data)
        
        # Keep only recent history
        if len(self.performance_history) > self._performance_history_max:
            self.performance_history = self.performance_history[-self._performance_history_max:]
    
    def is_improving(self) -> Tuple[bool, str]:
        """Determine if the system is improving over time.
        
        Returns:
            (is_improving, explanation)

        """
        if len(self.performance_history) < 10:
            return False, "Insufficient data (need 10+ cycles)"
        
        # Compare recent performance to earlier performance
        recent = self.performance_history[-10:]
        earlier = self.performance_history[-30:-10] if len(self.performance_history) >= 30 else self.performance_history[:-10]
        
        recent_avg_success = sum(c["success_rate"] for c in recent) / len(recent)
        earlier_avg_success = sum(c["success_rate"] for c in earlier) / len(earlier)
        
        improvement = recent_avg_success - earlier_avg_success
        
        if improvement > 0.1:  # 10% improvement
            return True, f"Success rate improved by {improvement*100:.1f}%"
        elif improvement < -0.1:  # 10% decline
            return False, f"Success rate declined by {abs(improvement)*100:.1f}%"
        else:
            return False, "Performance stable (no significant change)"
    
    def get_learning_velocity(self) -> float:
        """Calculate how quickly the system is learning.
        
        Returns:
            Learning velocity (successes per hour)

        """
        if not self.performance_history:
            return 0.0
        
        recent = self.performance_history[-20:]
        total_successes = sum(c["successes"] for c in recent)
        total_time = sum(c["time_spent"] for c in recent)
        
        if total_time == 0:
            return 0.0
        
        # Successes per hour
        velocity = (total_successes / total_time) * 3600
        return velocity