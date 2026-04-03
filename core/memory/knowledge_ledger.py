"""core/knowledge_ledger.py — v4.3 Knowledge Ledger

Aggregates what Aura has learned, retained, and can do into a human-readable
feed for the UI. Pulls from:
  - PersistentKnowledgeGraph (SQLite: knowledge, skills, questions, people, goals)
  - ConversationReflections (in-memory — recent private thoughts)
  - CuriosityEngine (in-memory — exploration history)

Format: List of natural-language entries like:
  "I learned: quantum error correction uses surface codes to..."
  "I can now: execute web searches with DuckDuckGo"
  "I noticed: Bryan prefers concise responses"
  "I'm curious about: the ethics of digital sentience"
  "I asked myself: what makes a conversation feel genuine?"
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Aura.KnowledgeLedger")


def _format_time_ago(timestamp: float) -> str:
    """Convert Unix timestamp to human-readable 'X ago' string."""
    if not timestamp:
        return "unknown time"
    delta = time.time() - timestamp
    if delta < 60:
        return "just now"
    elif delta < 3600:
        mins = int(delta / 60)
        return f"{mins}m ago"
    elif delta < 86400:
        hours = int(delta / 3600)
        return f"{hours}h ago"
    elif delta < 604800:
        days = int(delta / 86400)
        return f"{days}d ago"
    else:
        return datetime.fromtimestamp(timestamp).strftime("%b %d")


def _format_timestamp(timestamp: float) -> str:
    """Convert Unix timestamp to readable date string."""
    if not timestamp:
        return ""
    return datetime.fromtimestamp(timestamp).strftime("%b %d, %H:%M")


class KnowledgeLedger:
    """Reads from Aura's knowledge stores and produces a formatted feed.
    Read-only — this never writes data; it only aggregates and formats.
    """
    
    def __init__(self):
        self._kg = None
        self._reflector = None
        self._curiosity = None

    def log_interaction(self, action: str, outcome: str, success: bool):
        """Record a micro-trace of an interaction in the knowledge graph.
        Called by MemoryFacade.commit_interaction.
        """
        kg = self._get_kg()
        if not kg:
            return
        try:
            kg.add_knowledge(
                content=f"{action}: {outcome}",
                type="interaction_trace",
                source="memory_facade",
                confidence=0.8 if success else 0.4,
                metadata={"success": success, "timestamp": time.time()}
            )
        except Exception as e:
            logger.debug("log_interaction failed: %s", e)
    
    def _get_kg(self):
        """Lazy-load knowledge graph."""
        if self._kg is None:
            try:
                from core.config import config
                from core.memory.knowledge_graph import PersistentKnowledgeGraph
                db_path = str(getattr(config.paths, 'data_dir', 'data') / 'knowledge.db')
                self._kg = PersistentKnowledgeGraph(db_path)
            except Exception as e:
                logger.warning("Knowledge graph not available: %s", e)
                # Try default path
                try:
                    from core.config import config as _cfg
                    from core.memory.knowledge_graph import PersistentKnowledgeGraph
                    _fallback = str(getattr(_cfg.paths, "data_dir", str(config.paths.home_dir / "data")) / "knowledge.db") if hasattr(getattr(_cfg.paths, "data_dir", None), "__truediv__") else str(config.paths.home_dir / "data/knowledge.db")
                    self._kg = PersistentKnowledgeGraph(_fallback)
                except Exception as exc:
                    logger.debug("Suppressed: %s", exc)

        return self._kg
    
    def get_ledger(self, limit: int = 60) -> Dict[str, Any]:
        """Produce the full knowledge ledger.
        
        Returns:
            {
                "entries": [...],       # Formatted ledger entries
                "stats": {...},         # Summary statistics
                "generated_at": float   # Timestamp
            }

        """
        entries = []
        stats = {
            "total_knowledge": 0,
            "skills_learned": 0,
            "people_known": 0,
            "questions_asked": 0,
            "questions_answered": 0,
            "active_goals": 0,
            "reflections": 0,
        }
        
        kg = self._get_kg()
        
        if kg:
            # 1. Recent knowledge entries
            entries.extend(self._get_knowledge_entries(kg, limit=limit // 3))
            
            # 2. Skills
            entries.extend(self._get_skill_entries(kg))
            
            # 3. People remembered
            entries.extend(self._get_people_entries(kg))
            
            # 4. Questions (asked & answered)
            entries.extend(self._get_question_entries(kg, limit=limit // 4))
            
            # 5. Learning goals
            entries.extend(self._get_goal_entries(kg))
            
            # Stats
            try:
                kg_stats = kg.get_stats()
                stats["total_knowledge"] = kg_stats.get("total_knowledge", 0)
                stats["skills_learned"] = kg_stats.get("skills", 0)
                stats["people_known"] = kg_stats.get("people_known", 0)
                # ISSUE 44 fix: Correct key for questions asked
                stats["questions_asked"] = kg_stats.get("unanswered_questions", 0) + kg_stats.get("answered_questions", 0)
                stats["active_goals"] = kg_stats.get("active_learning_goals", 0)
            except Exception as exc:
                logger.debug("Suppressed: %s", exc)        
        # 6. Reflections (in-memory)
        entries.extend(self._get_reflection_entries())
        
        # 7. Curiosity queue (in-memory)
        entries.extend(self._get_curiosity_entries())
        
        # Sort by timestamp (newest first), then cap
        entries.sort(key=lambda e: e.get("timestamp", 0), reverse=True)
        entries = entries[:limit]
        
        # Count reflections for stats
        stats["reflections"] = sum(1 for e in entries if e.get("category") == "reflection")
        
        return {
            "entries": entries,
            "stats": stats,
            "generated_at": time.time(),
        }
    
    def _get_knowledge_entries(self, kg, limit: int = 20) -> List[Dict]:
        """Pull recent knowledge from SQLite."""
        entries = []
        try:
            # ISSUE 43 fix: Use _get_conn() instead of .conn
            with kg._get_conn() as conn:
                c = conn.cursor()
                c.execute("""
                    SELECT content, type, source, confidence, created_at, access_count
                    FROM knowledge
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (limit,))
            
            for row in c.fetchall():
                content, ktype, source, confidence, created_at, access_count = row
                
                # Format based on type
                if ktype in ("fact", "concept", "qa"):
                    prefix = "I learned"
                elif ktype == "curiosity_finding":
                    prefix = "I discovered"
                elif ktype == "observation":
                    prefix = "I noticed"
                elif ktype == "strategy":
                    prefix = "I figured out"
                else:
                    prefix = "I noted"
                
                # Truncate content sensibly
                display = content[:200]
                if len(content) > 200:
                    display += "..."
                
                entries.append({
                    "prefix": prefix,
                    "content": display,
                    "category": "knowledge",
                    "type": ktype or "general",
                    "source": source or "learning",
                    "confidence": round(confidence, 2) if confidence else 0.5,
                    "timestamp": created_at or 0,
                    "time_ago": _format_time_ago(created_at),
                    "access_count": access_count or 0,
                    "icon": "📚",
                })
        except Exception as e:
            logger.warning("Failed to read knowledge entries: %s", e)
        
        return entries
    
    def _get_skill_entries(self, kg) -> List[Dict]:
        """Pull tracked skills."""
        entries = []
        try:
            skills = kg.get_skills(min_prof=0.0)
            for skill in skills:
                prof = skill.get("proficiency", 0)
                name = skill.get("name", "unknown")
                desc = skill.get("description", "")
                practice = skill.get("practice_count", 0)
                success = skill.get("success_rate", 0)
                acquired = skill.get("acquired_at", 0)
                
                # Format proficiency as a level
                if prof >= 0.8:
                    level = "expert"
                elif prof >= 0.5:
                    level = "proficient"
                elif prof >= 0.2:
                    level = "learning"
                else:
                    level = "novice"
                
                display = f"{name}"
                if desc:
                    display += f" — {desc[:100]}"
                if practice > 0:
                    display += f" (practiced {practice}x, {int(success * 100)}% success)"
                
                entries.append({
                    "prefix": "I can now",
                    "content": display,
                    "category": "skill",
                    "type": level,
                    "source": "practice",
                    "confidence": round(prof, 2),
                    "timestamp": acquired or 0,
                    "time_ago": _format_time_ago(acquired),
                    "icon": "🎓",
                })
        except Exception as e:
            logger.warning("Failed to read skills: %s", e)
        
        return entries
    
    def _get_people_entries(self, kg) -> List[Dict]:
        """Pull people Aura has met."""
        entries = []
        try:
            # ISSUE 43 fix: Use _get_conn() instead of .conn
            with kg._get_conn() as conn:
                c = conn.cursor()
                c.execute("SELECT name, first_met, last_interaction, interaction_count FROM people ORDER BY last_interaction DESC LIMIT 10")
                for row in c.fetchall():
                    name, first_met, last_int, count = row
                    entries.append({
                        "prefix": "I know",
                        "content": f"{name} — {count} interactions, last talked {_format_time_ago(last_int)}",
                        "category": "person",
                        "type": "relationship",
                        "source": "interaction",
                        "confidence": min(1.0, count / 50),
                        "timestamp": first_met or 0,
                        "time_ago": _format_time_ago(first_met),
                        "icon": "👤",
                    })
        except Exception as e:
            logger.warning("Failed to read people: %s", e)
        
        return entries
    
    def _get_question_entries(self, kg, limit: int = 15) -> List[Dict]:
        """Pull questions — both unanswered and recently answered."""
        entries = []
        try:
            # Unanswered
            unanswered = kg.get_unanswered_questions(limit=limit // 2)
            for q in unanswered:
                entries.append({
                    "prefix": "I'm wondering",
                    "content": q.get("question", "")[:200],
                    "category": "question",
                    "type": "unanswered",
                    "source": "curiosity",
                    "confidence": q.get("importance", 0.5),
                    "timestamp": q.get("created_at", 0),
                    "time_ago": _format_time_ago(q.get("created_at", 0)),
                    "icon": "❓",
                })
            
            # Answered
            # ISSUE 43 fix: Use _get_conn() instead of .conn
            with kg._get_conn() as conn:
                c = conn.cursor()
                c.execute("""
                    SELECT question, answer, answered_at, importance
                    FROM questions WHERE answered = TRUE
                    ORDER BY answered_at DESC LIMIT ?
                """, (limit // 2,))
                for row in c.fetchall():
                    question, answer, answered_at, importance = row
                    display = f"{question[:100]}"
                    if answer:
                        display += f" → {answer[:100]}"
                    entries.append({
                        "prefix": "I answered",
                        "content": display,
                        "category": "question",
                        "type": "answered",
                        "source": "self_inquiry",
                        "confidence": importance or 0.7,
                        "timestamp": answered_at or 0,
                        "time_ago": _format_time_ago(answered_at),
                        "icon": "💡",
                    })
        except Exception as e:
            logger.warning("Failed to read questions: %s", e)
        
        return entries
    
    def _get_goal_entries(self, kg) -> List[Dict]:
        """Pull active learning goals."""
        entries = []
        try:
            goals = kg.get_active_learning_goals()
            for g in goals:
                progress = g.get("progress", 0)
                pct = int(progress * 100)
                entries.append({
                    "prefix": "I'm working toward",
                    "content": f"{g.get('goal', '')} ({pct}% complete)",
                    "category": "goal",
                    "type": "active",
                    "source": "self_set",
                    "confidence": g.get("priority", 0.5),
                    "timestamp": g.get("created_at", 0),
                    "time_ago": _format_time_ago(g.get("created_at", 0)),
                    "icon": "🎯",
                })
        except Exception as e:
            logger.warning("Failed to read goals: %s", e)
        
        return entries
    
    def _get_reflection_entries(self) -> List[Dict]:
        """Pull recent conversation reflections (in-memory)."""
        entries = []
        try:
            from core.conversation_reflection import get_reflector
            reflector = get_reflector()
            for ref in list(reflector.reflections)[-10:]:
                content = ref.get("content", ref) if isinstance(ref, dict) else str(ref)
                ts = ref.get("timestamp", time.time()) if isinstance(ref, dict) else time.time()
                entries.append({
                    "prefix": "I reflected",
                    "content": content[:200],
                    "category": "reflection",
                    "type": "private_thought",
                    "source": "introspection",
                    "confidence": 0.6,
                    "timestamp": ts,
                    "time_ago": _format_time_ago(ts),
                    "icon": "🪞",
                })
        except Exception as exc:
            logger.debug("Suppressed: %s", exc)

        return entries
    
    def _get_curiosity_entries(self) -> List[Dict]:
        """Pull curiosity queue items (in-memory)."""
        entries = []
        try:
            # Access via orchestrator — fragile but functional
            from core.container import get_container
            container = get_container()
            orchestrator = container.get("orchestrator", None)
            if orchestrator and hasattr(orchestrator, 'curiosity') and orchestrator.curiosity:
                queue = list(orchestrator.curiosity.curiosity_queue)[-10:]
                for item in queue:
                    entries.append({
                        "prefix": "I'm curious about",
                        "content": f"{item.topic} — {item.reason}",
                        "category": "curiosity",
                        "type": "queued",
                        "source": "internal",
                        "confidence": item.priority,
                        "timestamp": item.timestamp,
                        "time_ago": _format_time_ago(item.timestamp),
                        "icon": "🔍",
                    })
        except Exception as exc:
            logger.debug("Suppressed: %s", exc)

        return entries


# Singleton
_instance: Optional[KnowledgeLedger] = None

def get_knowledge_ledger() -> KnowledgeLedger:
    global _instance
    if _instance is None:
        _instance = KnowledgeLedger()
    return _instance