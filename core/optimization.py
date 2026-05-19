import asyncio
import json
import logging
from typing import Any

from core.runtime.errors import record_degradation

try:
    from .memory_store import Strategy, StrategyStore, UserMemory, UserMemoryStore
except ImportError:
    # Legacy module — memory_store was consolidated into the unified memory subsystem
    Strategy = StrategyStore = UserMemory = UserMemoryStore = None  # type: ignore

logger = logging.getLogger("Kernel.MetaOptimizer")

class StrategyMetaOptimizer:
    """The 'Meta-Cortex' that reflects on execution to improve system behavior.
    Handles:
    1. Extracting memories/strategies from turns.
    2. Evaluating strategy performance (offline/batch).
    """
    
    def __init__(self, brain, strategy_store: StrategyStore, user_store: UserMemoryStore):
        self.brain = brain
        self.strategy_store = strategy_store
        self.user_store = user_store

    async def process_turn(self, turn_record: dict[str, Any]):
        """Analyze a completed turn to extract potential learnings.
        """
        try:
            extraction = await self._extract_memories_from_turn(turn_record)
            
            # Save User Memories
            for um in extraction.get("user_memories", []):
                mem = UserMemory.new(
                    user_id=turn_record.get("user_id", "default_user"),
                    kind=um.get("kind", "preference"),
                    text=um["text"],
                    metadata={"source": "auto_extracted_from_turn", "turn_msg": turn_record.get("user_message")}
                )
                self.user_store.add(mem)
                logger.info("💾 Learned User Memory: %s", um['text'])

            # Save Strategies
            for st in extraction.get("strategies", []):
                strat = Strategy.new(
                    scope=st["scope"],
                    target=st["target"],
                    key=st["key"],
                    description=st["description"],
                    params=st.get("params", {})
                )
                self.strategy_store.add(strat)
                logger.info("💡 Learned Strategy: %s", st['description'])
                
        except (OSError, ConnectionError, TimeoutError) as e:
            record_degradation('optimization', e)
            logger.error("Meta-optimization loop failed: %s", e, exc_info=True)

    async def _extract_memories_from_turn(self, turn: dict[str, Any]) -> dict[str, Any]:
        """Decide which user memories and strategies to create from a turn.
        """
        # Simplify turn for LLM consumption
        lean_turn = {
            "user_message": turn.get("user_message"),
            "final_answer": turn.get("final_answer"),
            "outcome": turn.get("outcome"),
            "tool_results": str(turn.get("tool_results", []))[:1000] # Truncate for token limits
        }
        
        prompt = f"""
You are Aura's meta-cortex. Given this turn (JSON), extract:
1) Up to 2 persistent user preferences (style, limits, topics).
2) Up to 2 reusable strategies (for planner/tools).

You MUST reply in valid JSON only:

{{
  "user_memories": [
    {{"kind": "preference", "text": "..."}},
    ...
  ],
  "strategies": [
    {{
      "scope": "component", 
      "target": "web_search", 
      "key": "merge_tool_outputs", 
      "description": "If user complains about duplicate info, merge tool outputs.",
      "params": {{ "enabled": true }} 
    }},
    {{
      "scope": "domain", 
      "target": "space.com", 
      "key": "news_entrypoint", 
      "description": "For Space.com, use /news.", 
      "params": {{ "url": "https://www.space.com/news" }}
    }}
  ]
}}

TURN:
```json
{json.dumps(lean_turn, indent=2)}
```
"""
        try:
            if asyncio.iscoroutinefunction(self.brain.generate):
                raw = await self.brain.generate(prompt)
            else:
                raw = self.brain.generate(prompt)
            # Safe cleaning: handle if raw is already a dict (some clients auto-parse)
            if isinstance(raw, dict):
                return raw
                
            if hasattr(raw, 'strip'):
                clean_json = raw.strip()
            else:
                clean_json = str(raw).strip()
                
            if "```json" in clean_json:
                clean_json = clean_json.split("```json")[1].split("```")[0].strip()
            elif "```" in clean_json:
                clean_json = clean_json.split("```")[1].split("```")[0].strip()
                
            return json.loads(clean_json)
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            record_degradation('optimization', e)
            logger.warning("Failed to extract memories: %s", e)
            return {"user_memories": [], "strategies": []}

    def evaluate_strategies(self, logs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Evaluate observed strategy outcomes and return ranked scorecards.

        The optimizer accepts either explicit ``strategy_key`` fields or common
        legacy names such as ``strategy``/``action``. Success can be supplied as
        a bool, numeric score, or textual outcome. When the attached strategy
        store exposes an ``update_stats`` or ``record_evaluation`` method, the
        scorecard is persisted there as well.
        """
        scorecards: dict[str, dict[str, Any]] = {}

        for entry in logs or []:
            strategy_key = (
                entry.get("strategy_key")
                or entry.get("strategy")
                or entry.get("action")
                or entry.get("key")
            )
            if not strategy_key:
                continue

            score = self._coerce_outcome_score(entry)
            card = scorecards.setdefault(
                str(strategy_key),
                {"attempts": 0, "successes": 0, "score_total": 0.0, "mean_score": 0.0},
            )
            card["attempts"] += 1
            card["successes"] += int(score >= 0.5)
            card["score_total"] += score
            card["mean_score"] = card["score_total"] / card["attempts"]

        ranked = dict(
            sorted(
                scorecards.items(),
                key=lambda item: (item[1]["mean_score"], item[1]["attempts"]),
                reverse=True,
            )
        )
        self._persist_strategy_evaluations(ranked)
        return ranked

    @staticmethod
    def _coerce_outcome_score(entry: dict[str, Any]) -> float:
        raw = entry.get("score", entry.get("success", entry.get("ok", entry.get("outcome", 0.0))))
        if isinstance(raw, bool):
            return 1.0 if raw else 0.0
        if isinstance(raw, (int, float)):
            return max(0.0, min(1.0, float(raw)))
        text = str(raw).strip().lower()
        if text in {"success", "succeeded", "ok", "passed", "complete", "completed"}:
            return 1.0
        if text in {"partial", "degraded", "retry", "mixed"}:
            return 0.5
        return 0.0

    def _persist_strategy_evaluations(self, scorecards: dict[str, dict[str, Any]]) -> None:
        if not self.strategy_store:
            return
        try:
            if hasattr(self.strategy_store, "record_evaluation"):
                for key, card in scorecards.items():
                    self.strategy_store.record_evaluation(key, dict(card))
            elif hasattr(self.strategy_store, "update_stats"):
                for key, card in scorecards.items():
                    self.strategy_store.update_stats(key, dict(card))
        except (RuntimeError, AttributeError, TypeError) as exc:
            record_degradation("optimization", exc)
            logger.warning("Failed to persist strategy evaluations: %s", exc)
