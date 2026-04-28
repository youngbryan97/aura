"""Unified Continuous Learning Engine for Aura.

Bridges experience logging, pattern extraction, and autonomous research 
to ensure Aura progressively evolves from every interaction.
"""
from core.runtime.errors import record_degradation
from core.utils.task_tracker import get_task_tracker
import asyncio
import hashlib
import json
import logging
import sqlite3
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from core.base_module import AuraBaseModule
from core.config import config

@dataclass
class Experience:
    """Represents a single interaction or event for learning.
    
    Attributes:
        id (str): Unique hash of the interaction.
        timestamp (float): UNIX timestamp of the event.
        input_summary (str): Summary of the input received.
        response_summary (str): Summary of the response generated.
        outcome_quality (float): Metric of success (0.0 to 1.0).
        domain (str): Feature area or skill category.
        strategy (str): Decision-making strategy used.
        context_hash (str): Hash of the agent's state during the event.
        corrections (List[str]): Manual or automated feedback received.
    """
    id: str
    timestamp: float
    input_summary: str
    response_summary: str
    outcome_quality: float = 0.5
    domain: str = "general"
    strategy: str = "default"
    context_hash: str = ""
    corrections: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

@dataclass
class LearningPattern:
    """Represents a synthesized insight derived from multiple experiences.
    
    Attributes:
        id (str): Unique identifier for the pattern.
        description (str): Human-readable explanation of the insight.
        trigger (str): Condition or keyword that activates the pattern.
        recommendation (str): Actionable advice for similar future contexts.
        confidence (float): Statistical confidence in the pattern.
        evidence (int): Number of experiences supporting this pattern.
        domain (str): Category of application.
        last_updated (float): Last time the pattern was refined.
    """
    id: str
    description: str
    trigger: str
    recommendation: str
    confidence: float
    evidence: int
    domain: str
    last_updated: float = field(default_factory=time.time)

class ExperienceStore:
    """Persistent storage for experiences and learned patterns using SQLite."""
    
    def __init__(self, db_path: Optional[Path] = None):
        """Initializes the SQLite database.
        
        Args:
            db_path: Path to the database file. Defaults to data/learning/experiences.db.
        """
        if not db_path:
            db_path = config.paths.data_dir / "learning" / "experiences.db"
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Creates the necessary tables if they do not exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS experiences (id TEXT PRIMARY KEY, timestamp REAL, input TEXT, response TEXT, quality REAL, domain TEXT, strategy TEXT, context_hash TEXT, corrections TEXT, tags TEXT)")
            conn.execute("CREATE TABLE IF NOT EXISTS patterns (id TEXT PRIMARY KEY, description TEXT, trigger TEXT, recommendation TEXT, confidence REAL, evidence INTEGER, domain TEXT, last_updated REAL)")

    def save_experience(self, exp: Experience) -> None:
        """Persists an experience to the database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("INSERT OR REPLACE INTO experiences VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (exp.id, exp.timestamp, exp.input_summary, exp.response_summary, exp.outcome_quality, exp.domain, exp.strategy, exp.context_hash, json.dumps(exp.corrections), json.dumps(exp.tags)))

    def update_outcome(self, exp_id: str, quality: float, corrections: Optional[List[str]] = None) -> None:
        """Updates the feedback quality and corrections for an existing experience."""
        with sqlite3.connect(self.db_path) as conn:
            if corrections:
                conn.execute("UPDATE experiences SET quality=?, corrections=? WHERE id=?", (quality, json.dumps(corrections), exp_id))
            else:
                conn.execute("UPDATE experiences SET quality=? WHERE id=?", (quality, exp_id))

    def get_experiences(self, limit: int = 500) -> List[Experience]:
        """Retrieves recent experiences from the database."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT * FROM experiences ORDER BY timestamp DESC LIMIT ?", (limit,)).fetchall()
            return [Experience(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], json.loads(r[8])) for r in rows]

    def save_pattern(self, p: LearningPattern) -> None:
        """Persists a learned pattern to the database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("INSERT OR REPLACE INTO patterns VALUES (?,?,?,?,?,?,?,?)",
                        (p.id, p.description, p.trigger, p.recommendation, p.confidence, p.evidence, p.domain, p.last_updated))

    def get_patterns(self) -> List[LearningPattern]:
        """Retrieves high-confidence patterns from the database."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT * FROM patterns WHERE confidence > 0.6 ORDER BY confidence DESC").fetchall()
            return [LearningPattern(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]) for r in rows]

class ContinuousLearningEngine(AuraBaseModule):
    """Unified Enterprise Learning Engine.
    
    Orchestrates the lifecycle of synthetic evolution: 
    Experience logging -> LLM-based Pattern Extraction -> Knowledge Graph integration.
    """
    
    def __init__(self, base_dir: str = None, brain: Any = None, orchestrator: Any = None):
        """Initializes the LearningEngine.
        
        Args:
            base_dir: Directory for persistence. Defaults to data/learning within config.paths.data_dir.
            brain: Reference to the cognitive engine.
            orchestrator: Reference to the system orchestrator.
        """
        super().__init__("LearningEngine")
        self.orchestrator = orchestrator
        self.brain = brain
        if base_dir is None:
            from core.config import config
            self.base_dir = config.paths.data_dir / "learning"
        else:
            self.base_dir = Path(base_dir)
        self.store = ExperienceStore(db_path=self.base_dir / "experiences.db")
        self.knowledge: Any = None
        self._load_knowledge()
        
        # Scheduling
        self.last_research = 0.0
        self.research_interval = 600.0  # 10 minutes
        self.last_extraction = 0.0
        self.extraction_interval = 3600.0  # 1 hour
        
        self.logger.info("✓ Continuous Learning Engine Online")

    def _load_knowledge(self) -> None:
        """Connects to the Persistent Knowledge Graph if available."""
        try:
            from core.memory.knowledge_graph import PersistentKnowledgeGraph
            self.knowledge = PersistentKnowledgeGraph()
        except ImportError:
            self.logger.debug("Knowledge Graph not available.")

    async def record_interaction(self, user_input: str = None, aura_response: str = None, 
                                 user_name: Optional[str] = None,
                                 domain: str = "general", strategy: str = "default", **kwargs) -> str:
        """Records an interaction and triggers asynchronous learning.
        
        Args:
            user_input: The text provided by the user.
            aura_response: The response generated by Aura.
            user_name: Optional user identity for person-specific context.
            domain: The module or skill area involved.
            strategy: The reasoning strategy used.
            **kwargs: Support for aliases like input_text/response_text.
            
        Returns:
            str: The unique ID assigned to this experience.
        """
        # Alias support
        user_input = user_input or kwargs.get('input_text')
        aura_response = aura_response or kwargs.get('response_text') or kwargs.get('response')
        
        if not user_input or not aura_response:
             self.logger.warning("record_interaction called with missing input/response.")
             return "error"

        @self.error_boundary
        async def _record_wrapped():
            exp_id = hashlib.sha256(f"{time.time()}{user_input[:20]}".encode()).hexdigest()[:16]
            exp = Experience(id=exp_id, timestamp=time.time(), input_summary=user_input, 
                             response_summary=aura_response, domain=domain, strategy=strategy)
            self.store.save_experience(exp)
            
            # Remember person if name provided
            if user_name and self.knowledge:
                try:
                    self.knowledge.remember_person(user_name, {
                        "timestamp": time.time(),
                        "message": user_input,
                        "response": aura_response
                    })
                except Exception as e:
                    record_degradation('continuous_learning', e)
                    self.logger.debug("Failed to update person context: %s", e)

            # Immediate knowledge extraction if possible
            if self.orchestrator and hasattr(self.orchestrator, "cognitive_engine"):
                get_task_tracker().create_task(self._extract_knowledge_async(user_input, aura_response))
            
            return exp_id
            
        return await _record_wrapped()

    async def _extract_knowledge_async(self, user_input: str, response: str) -> None:
        """Uses the LLM to autonomously extract factual knowledge from a chat turn.
        
        Args:
            user_input: Original user message.
            response: Aura's response.
        """
        if not self.knowledge: return
        prompt = (f"Extract factual knowledge from this conversation as JSON:\n"
                  f"User: {user_input}\nAura: {response}\n\n"
                  f"Return: {{\"facts\": [\"fact 1\", \"fact 2\"]}}")
        try:
            brain = self.orchestrator.cognitive_engine
            # Try structured think if available
            if hasattr(brain, "think_structured"):
                res = await brain.think_structured(prompt, expected_format="json")
                if res.get("ok"):
                    for fact in res.get("data", {}).get("facts", []):
                        self.knowledge.add_knowledge(content=fact, source="conversation")
                    return
            
            # Fallback to standard think
            raw = await brain.generate(
                prompt,
                use_strategies=False,
                origin="knowledge_extraction",
                is_background=True,
            )
            if raw and "{" in raw:
                data = json.loads(raw[raw.find("{"):raw.rfind("}")+1])
                for fact in data.get("facts", []):
                    self.knowledge.add_knowledge(
                        content=fact,
                        type="fact",
                        source="conversation_learning",
                    )
        except Exception as e:
            record_degradation('continuous_learning', e)
            self.logger.debug("Knowledge extraction failed: %s", e)

    async def get_relevant_context(self, current_input: str, user_name: Optional[str] = None) -> str:
        """Retrieves synthesized context (patterns + person info) for prompt augmentation.
        
        Args:
            current_input: The user's latest message.
            user_name: The user's name if identified.
            
        Returns:
            str: Compiled context string.
        """
        context_parts = []
        
        # 1. Person context
        if user_name and self.knowledge:
            try:
                person_info = self.knowledge.get_person(user_name)
                if person_info:
                    count = person_info.get("interaction_count", 0)
                    context_parts.append(f"[Identity] User is {user_name}. We have had {count} interactions.")
            except Exception as e:
                record_degradation('continuous_learning', e)
                self.logger.debug("Identity retrieval failed: %s", e)

        # 2. Pattern context
        patterns = self.store.get_patterns()
        input_lower = current_input.lower()
        for p in patterns:
            # Trigger-based matching (simpler keyword match for now)
            if any(word in input_lower for word in p.trigger.lower().split()):
                context_parts.append(f"[Pattern] {p.description}: {p.recommendation}")
        
        return "\n".join(context_parts[:3])

    async def run_maintenance(self) -> None:
        """Scheduled background task for research and pattern extraction."""
        now = time.time()
        if now - self.last_research > self.research_interval:
            await self._autonomous_research()
            self.last_research = now
        
        if now - self.last_extraction > self.extraction_interval:
            await self._extract_patterns()
            await self.metabolic_compression()
            self.last_extraction = now

    async def _autonomous_research(self) -> None:
        """Checks for unanswered questions in the Knowledge Graph and performs research."""
        if not self.orchestrator: return
        self.logger.info("🔬 Running autonomous research cycle...")
        
        if self.knowledge and hasattr(self.knowledge, "get_unanswered_questions"):
            questions = self.knowledge.get_unanswered_questions(limit=1)
            if questions:
                q = questions[0]["question"]
                self.logger.info("❓ Researching: %s", q)
                # Future: Link to web_search skill automatically

    async def consolidate_experiences(self) -> Dict[str, Any]:
        """Distills high-level patterns from recent raw experiences.
        
        This is the core 'RE' consolidation step of the Dream Cycle.
        """
        self.logger.info("💤 Consolidating experiences (Dream Phase)...")
        exps = self.store.get_experiences(limit=50)
        if len(exps) < 5:
            return {"ok": True, "patterns_found": 0, "status": "insufficient_data"}
            
        # 1. Prepare data for distillation
        exp_summary = "\n".join([f"- {e.input_summary} -> {e.response_summary}" for e in exps])
        
        prompt = (
            f"Analyze these recent experiences and distill 1-3 general 'Learning Patterns'.\n"
            f"A pattern should be a reusable heuristic or observation about user preference or task strategy.\n\n"
            f"Experiences:\n{exp_summary}\n\n"
            f"Return a JSON array: [{{\"description\": \"...\", \"trigger\": \"keyword\", \"recommendation\": \"...\", \"confidence\": 0.8}}]"
        )
        
        try:
            brain = self.orchestrator.cognitive_engine
            res = await brain.think(prompt, mode="fast")
            
            # 2. Parse and Save
            import json as _json
            content = res.content
            start = content.find('[')
            end = content.rfind(']') + 1
            if start >= 0 and end > start:
                patterns = _json.loads(content[start:end])
                for p_data in patterns:
                    p_id = hashlib.sha256(p_data['description'].encode()).hexdigest()[:12]
                    pattern = LearningPattern(
                        id=p_id,
                        description=p_data['description'],
                        trigger=p_data['trigger'],
                        recommendation=p_data['recommendation'],
                        confidence=p_data.get('confidence', 0.7),
                        evidence=len(exps),
                        domain="consolidated"
                    )
                    self.store.save_pattern(pattern)
                    self.logger.info("💡 New Pattern Distilled: %s...", pattern.description[:50])
                
                return {"ok": True, "patterns_found": len(patterns)}
        except Exception as e:
            record_degradation('continuous_learning', e)
            self.logger.error("Consolidation failed: %s", e)
            return {"ok": False, "error": str(e)}

    async def metabolic_compression(self) -> Dict[str, Any]:
        """Merges older raw experiences into persistent knowledge to prevent database bloat.
        
        This is the 'Strategic Forgetting' part of Digital Metabolism.
        """
        self.logger.info("📦 Running Metabolic Compression...")
        
        if not self.knowledge:
            return {"ok": False, "error": "Knowledge Graph not available"}
            
        # 1. Get older experiences (past 500 but older than 50)
        # We keep the very recent ones raw for short-term context
        with sqlite3.connect(self.store.db_path) as conn:
            rows = conn.execute(
                "SELECT id, input, response FROM experiences "
                "WHERE timestamp < ? ORDER BY timestamp ASC LIMIT 20",
                (time.time() - 3600,) # Older than 1 hour
            ).fetchall()
            
        if len(rows) < 10:
            return {"ok": True, "status": "insufficient_old_data"}
            
        # 2. Distill into factual knowledge
        text_to_distill = "\n".join([f"Q: {r[1]} | A: {r[2]}" for r in rows])
        prompt = (
            "Summarize the following interaction history into 2-3 concise, high-value factual statements "
            "that Aura should remember permanently. Ignore trivialities.\n\n"
            f"History:\n{text_to_distill}"
        )
        
        try:
            brain = self.orchestrator.cognitive_engine
            res = await brain.think(prompt, mode="fast")
            
            if res.content:
                # 3. Save to Knowledge Graph
                facts = res.content.split("\n")
                for fact in facts:
                    if len(fact.strip()) > 10:
                        self.knowledge.add_knowledge(content=fact.strip(), source="metabolic_distillation")
                
                # 4. Delete the raw experiences (They are now 'digested')
                ids_to_delete = [r[0] for r in rows]
                with sqlite3.connect(self.store.db_path) as conn:
                    conn.executemany("DELETE FROM experiences WHERE id=?", [(i,) for i in ids_to_delete])
                
                self.logger.info("✅ Compressed %s experiences into persistent knowledge.", len(ids_to_delete))
                return {"ok": True, "compressed_count": len(ids_to_delete)}
                
        except Exception as e:
            record_degradation('continuous_learning', e)
            self.logger.error("Metabolic compression failed: %s", e)
            return {"ok": False, "error": str(e)}

    async def _extract_patterns(self) -> None:
        """Scheduled pattern extraction (delegates to consolidate_experiences)."""
        await self.consolidate_experiences()

    def get_stats(self) -> Dict[str, Any]:
        """Calculates and returns learning system statistics."""
        exps = self.store.get_experiences(limit=1) # Just check count conceptually or use separate query
        patterns = self.store.get_patterns()
        return {
            "patterns_active": len(patterns),
            "knowledge_nodes": self.knowledge.get_stats().get("total_knowledge", 0) if self.knowledge else 0
        }

    def get_health(self) -> Dict[str, Any]:
        """Provides extended health data for the learning engine."""
        return {**super().get_health(), **self.get_stats()}
