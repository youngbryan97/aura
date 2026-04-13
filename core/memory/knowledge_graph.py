"""Persistent Knowledge Graph - SQLite backend
Aura's knowledge persists forever across sessions.
v5.0: Thread-safe with WAL mode.
"""
import hashlib
import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from core.config import config

logger = logging.getLogger("Knowledge.Graph")


class PersistentKnowledgeGraph:
    """SQLite-backed knowledge that never forgets. v5.0: Thread-safe."""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = str(config.paths.data_dir / "knowledge.db")
        self.db_path = db_path
        
        # Issue 13 Hardening: SafeDatabaseLock with timeout protection
        from infrastructure.resilience import SafeDatabaseLock
        self.safe_lock = SafeDatabaseLock(name="KnowledgeGraph")
        self._lock = threading.Lock() # Legacy compatibility
        
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()
        
        logger.info("✓ Knowledge Graph: %s", db_path)
        logger.info("   Nodes: %s", self.count_nodes())
    
    def _get_conn(self) -> sqlite3.Connection:
        """Get an optimized thread-safe connection."""
        from core.memory import db_config
        return db_config.configure_connection(self.db_path)

    def _approve_memory_write(
        self,
        memory_type: str,
        content: str,
        *,
        source: str,
        importance: float,
        metadata: Optional[Dict[str, Any]] = None,
        return_decision: bool = False,
    ) -> bool | tuple[bool, Any]:
        runtime_live = False
        try:
            from core.container import ServiceContainer
            from core.constitution import get_constitutional_core, unpack_governance_result

            runtime_live = bool(
                getattr(ServiceContainer, "_registration_locked", False)
                or ServiceContainer.has("executive_core")
                or ServiceContainer.has("aura_kernel")
                or ServiceContainer.has("kernel_interface")
            )
            approved, reason, decision = unpack_governance_result(
                get_constitutional_core().approve_memory_write_sync(
                    memory_type=memory_type,
                    content=content,
                    source=source,
                    importance=importance,
                    metadata=metadata,
                    return_decision=True,
                )
            )
            if approved:
                if return_decision:
                    return True, decision
                return True

            logger.warning("🚫 KnowledgeGraph write blocked: %s (%s)", memory_type, reason)
            try:
                from core.health.degraded_events import record_degraded_event

                record_degraded_event(
                    "knowledge_graph",
                    "memory_write_blocked",
                    detail=memory_type,
                    severity="warning",
                    classification="background_degraded",
                    context={"source": source, "reason": reason},
                )
            except Exception as exc:
                logger.debug("KnowledgeGraph degraded-event logging skipped: %s", exc)
            if return_decision:
                return False, None
            return False
        except Exception as exc:
            logger.debug("KnowledgeGraph constitutional gate skipped: %s", exc)
            if return_decision:
                return (not runtime_live), None
            return not runtime_live

    async def check_health(self) -> Dict[str, Any]:
        """Diagnostic health check for the persistence layer."""
        health = {
            "status": "healthy",
            "db_path": self.db_path,
            "wal_mode": False,
            "integrity_ok": False,
            "node_count": 0
        }
        try:
            with self._get_conn() as conn:
                # 1. Check Journal Mode
                res = conn.execute("PRAGMA journal_mode;").fetchone()
                health["wal_mode"] = (res[0].lower() == "wal") if res else False
                
                # 2. Check Integrity
                res = conn.execute("PRAGMA integrity_check;").fetchone()
                health["integrity_ok"] = (res[0].lower() == "ok") if res else False
                
                # 3. Quick Stats
                health["node_count"] = self.count_nodes()
                
            if not health["wal_mode"] or not health["integrity_ok"]:
                health["status"] = "degraded"
                
        except Exception as e:
            health["status"] = "error"
            health["error"] = str(e)
            
        return health

    def _init_schema(self):
        """Create tables"""
        # ISSUE 37 fix: Keep logic inside context manager to prevent closed connection errors
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            
            # ISSUE 42 fix: Standardize on 'created_at' (knowledge: learned_at -> created_at)
            conn.execute("""CREATE TABLE IF NOT EXISTS knowledge (
                id TEXT PRIMARY KEY, content TEXT, type TEXT, source TEXT,
                confidence REAL, created_at REAL, last_accessed REAL,
                access_count INTEGER, metadata TEXT)""")
            
            conn.execute("""CREATE TABLE IF NOT EXISTS relationships (
                from_id TEXT, to_id TEXT, relation_type TEXT,
                strength REAL, created_at REAL,
                PRIMARY KEY (from_id, to_id, relation_type))""")
            
            conn.execute("""CREATE TABLE IF NOT EXISTS people (
                id TEXT PRIMARY KEY, name TEXT, first_met REAL,
                last_interaction REAL, interaction_count INTEGER, data TEXT)""")
            
            conn.execute("""CREATE TABLE IF NOT EXISTS skills (
                id TEXT PRIMARY KEY, name TEXT, description TEXT,
                acquired_at REAL, proficiency REAL, practice_count INTEGER,
                success_rate REAL, last_used REAL, data TEXT)""")
            
            conn.execute("""CREATE TABLE IF NOT EXISTS learning_goals (
                id TEXT PRIMARY KEY, goal TEXT, created_at REAL,
                target_completion REAL, priority REAL, progress REAL,
                completed BOOLEAN, notes TEXT)""")
            
            conn.execute("""CREATE TABLE IF NOT EXISTS questions (
                id TEXT PRIMARY KEY, question TEXT, created_at REAL,
                importance REAL, answered BOOLEAN, answer TEXT, answered_at REAL)""")
            conn.commit()
    
    def add_knowledge(self, content: str, type: str, source: str = "learning",
                      confidence: float = 0.7, metadata: Optional[Dict] = None) -> str:
        """Add knowledge — thread-safe."""
        node_id = hashlib.sha256(content.encode()).hexdigest()[:16]

        approved, governance_decision = self._approve_memory_write(
            f"knowledge:{type or 'observation'}",
            content,
            source=source or "learning",
            importance=max(0.0, min(1.0, float(confidence or 0.0))),
            metadata=metadata,
            return_decision=True,
        )
        if not approved:
            return node_id

        if governance_decision is not None:
            from core.governance_context import governed_scope_sync

            with governed_scope_sync(governance_decision):
                with self._lock:
                    with self._get_conn() as conn:
                        conn.row_factory = sqlite3.Row
                        row = conn.execute("SELECT id FROM knowledge WHERE id = ?", (node_id,)).fetchone()
                        if row:
                            conn.execute("""UPDATE knowledge SET last_accessed = ?, 
                                         access_count = access_count + 1, 
                                         confidence = MIN(1.0, confidence + 0.05) 
                                         WHERE id = ?""", (time.time(), node_id))
                            conn.commit()
                            return node_id
                        
                        conn.execute("""INSERT INTO knowledge (id, content, type, source, confidence, 
                                                              created_at, last_accessed, access_count, metadata) 
                                        VALUES (?,?,?,?,?,?,?,?,?)""",
                                  (node_id, content, type, source, confidence, time.time(),
                                   time.time(), 1, json.dumps(metadata or {})))
                        conn.commit()
        else:
            with self._lock:
                with self._get_conn() as conn:
                    conn.row_factory = sqlite3.Row
                    row = conn.execute("SELECT id FROM knowledge WHERE id = ?", (node_id,)).fetchone()
                    if row:
                        conn.execute("""UPDATE knowledge SET last_accessed = ?, 
                                     access_count = access_count + 1, 
                                     confidence = MIN(1.0, confidence + 0.05) 
                                     WHERE id = ?""", (time.time(), node_id))
                        conn.commit()
                        return node_id
                    
                    conn.execute("""INSERT INTO knowledge (id, content, type, source, confidence, 
                                                          created_at, last_accessed, access_count, metadata) 
                                    VALUES (?,?,?,?,?,?,?,?,?)""",
                              (node_id, content, type, source, confidence, time.time(),
                               time.time(), 1, json.dumps(metadata or {})))
                    conn.commit()
        
        logger.info("📚 Learned: %s...", content[:80])
        return node_id
    
    def get_knowledge(self, node_id: str) -> Optional[Dict]:
        """Get knowledge by ID"""
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            c.execute("SELECT * FROM knowledge WHERE id = ?", (node_id,))
            row = c.fetchone()
            return dict(row) if row else None
    
    def search_knowledge(self, query: str, type: Optional[str] = None, limit: int = 10) -> List[Dict]:
        """Search knowledge"""
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            if type:
                c.execute("""SELECT * FROM knowledge WHERE type = ? AND content LIKE ?
                            ORDER BY confidence DESC LIMIT ?""",
                         (type, f"%{query}%", limit))
            else:
                c.execute("""SELECT * FROM knowledge WHERE content LIKE ?
                            ORDER BY confidence DESC LIMIT ?""",
                         (f"%{query}%", limit))
            return [dict(row) for row in c.fetchall()]
    
    def _update_knowledge_access(self, node_id: str, confidence_boost: float = 0.0):
        """Update access — caller must hold _lock or call within with self._lock."""
        with self._get_conn() as conn:
            c = conn.cursor()
            c.execute("""UPDATE knowledge SET last_accessed = ?, access_count = access_count + 1,
                        confidence = MIN(1.0, confidence + ?) WHERE id = ?""",
                     (time.time(), confidence_boost, node_id))
            conn.commit()
    
    def count_nodes(self, type: Optional[str] = None) -> int:
        """Count nodes"""
        with self._get_conn() as conn:
            c = conn.cursor()
            if type:
                c.execute("SELECT COUNT(*) FROM knowledge WHERE type = ?", (type,))
            else:
                c.execute("SELECT COUNT(*) FROM knowledge")
            return c.fetchone()[0]
    
    def remember_person(self, name: str, interaction_data: Dict) -> str:
        """Remember person"""
        person_id = hashlib.sha256(name.lower().encode()).hexdigest()[:16]
        if not self._approve_memory_write(
            "knowledge:person",
            name,
            source="conversation_identity",
            importance=0.75,
            metadata=interaction_data,
        ):
            return person_id

        with self._lock:
            # ISSUE 39 fix: Ensure all DB operations happen within the connection context
            with self._get_conn() as conn:
                row = conn.execute("SELECT data FROM people WHERE id = ?", (person_id,)).fetchone()
            
                if row:
                    data = json.loads(row[0])
                    data["last_interaction"] = time.time()
                    data["interaction_count"] += 1
                    data["conversation_history"].append(interaction_data)
                    data["conversation_history"] = data["conversation_history"][-100:]
                
                    conn.execute("""UPDATE people SET last_interaction = ?,
                                interaction_count = ?, data = ? WHERE id = ?""",
                             (time.time(), data["interaction_count"], json.dumps(data), person_id))
                else:
                    data = {
                        "name": name, "first_met": time.time(),
                        "last_interaction": time.time(), "interaction_count": 1,
                        "conversation_history": [interaction_data],
                        "preferences": {},
                        "topics_discussed": []
                    }
                    conn.execute("""INSERT INTO people VALUES (?,?,?,?,?,?)""",
                             (person_id, name, time.time(), time.time(), 1, json.dumps(data)))
                conn.commit()
            
            logger.info("👤 Remembered: %s", name)
            return person_id
    
    def get_person(self, name: str) -> Optional[Dict]:
        """Get person"""
        person_id = hashlib.sha256(name.lower().encode()).hexdigest()[:16]
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            c.execute("SELECT * FROM people WHERE id = ?", (person_id,))
            row = c.fetchone()
            if row:
                data = dict(row)
                data.update(json.loads(data["data"]))
                del data["data"]
                return data
        return None
    
    def add_skill(self, name: str, description: str, proficiency: float = 0.1) -> str:
        """Add skill"""
        with self._lock:
            skill_id = hashlib.sha256(name.lower().encode()).hexdigest()[:16]
            with self._get_conn() as conn:
                c = conn.cursor()
                c.execute("""INSERT INTO skills VALUES (?,?,?,?,?,?,?,?,?)""",
                         (skill_id, name, description, time.time(), proficiency,
                          0, 0.0, None, json.dumps({})))
                conn.commit()
            logger.info("🎓 Skill: %s", name)
            return skill_id
    
    def practice_skill(self, skill_id: str, success: bool):
        """Practice skill"""
        with self._lock:
            with self._get_conn() as conn:
                row = conn.execute("SELECT practice_count, success_rate, proficiency FROM skills WHERE id = ?", (skill_id,)).fetchone()
                if not row:
                    return
            
                count, rate, prof = row
                new_rate = (rate * count + (1.0 if success else 0.0)) / (count + 1)
                new_prof = min(1.0, prof + (0.05 if success else 0.01) * (1.0 - prof))
            
                conn.execute("""UPDATE skills SET practice_count = practice_count + 1,
                            success_rate = ?, proficiency = ?, last_used = ? WHERE id = ?""",
                         (new_rate, new_prof, time.time(), skill_id))
                # ISSUE 41 fix: Explicit commit
                conn.commit()
    
    def get_skills(self, min_prof: float = 0.0) -> List[Dict]:
        """Get skills"""
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            c.execute("SELECT * FROM skills WHERE proficiency >= ? ORDER BY proficiency DESC", (min_prof,))
            return [dict(row) for row in c.fetchall()]
    
    def add_learning_goal(self, goal: str, priority: float = 0.5, target_days: int = 7) -> str:
        """Add goal"""
        goal_id = hashlib.sha256(goal.encode()).hexdigest()[:16]
        if not self._approve_memory_write(
            "knowledge:learning_goal",
            goal,
            source="learning_goal",
            importance=max(0.0, min(1.0, float(priority or 0.0))),
            metadata={"target_days": target_days},
        ):
            return goal_id

        with self._lock:
            with self._get_conn() as conn:
                c = conn.cursor()
                c.execute("""INSERT OR REPLACE INTO learning_goals VALUES (?,?,?,?,?,?,?,?)""",
                         (goal_id, goal, time.time(), time.time() + target_days * 86400,
                          priority, 0.0, False, ""))
                conn.commit()
            logger.info("🎯 Goal: %s", goal)
            return goal_id
    
    def get_active_learning_goals(self) -> List[Dict]:
        """Get active goals"""
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            c.execute("SELECT * FROM learning_goals WHERE completed = FALSE ORDER BY priority DESC")
            return [dict(row) for row in c.fetchall()]
    
    def ask_question(self, question: str, importance: float = 0.5) -> str:
        """Ask question"""
        q_id = hashlib.sha256(question.encode()).hexdigest()[:16]
        if not self._approve_memory_write(
            "knowledge:question",
            question,
            source="self_inquiry",
            importance=max(0.0, min(1.0, float(importance or 0.0))),
        ):
            return q_id

        with self._lock:
            with self._get_conn() as conn:
                c = conn.cursor()
                c.execute("""INSERT OR REPLACE INTO questions VALUES (?,?,?,?,?,?,?)""",
                         (q_id, question, time.time(), importance, False, None, None))
                conn.commit()
            logger.info("❓ Question: %s", question)
            return q_id
    
    def answer_question(self, q_id: str, answer: str):
        """Answer question"""
        if not self._approve_memory_write(
            "knowledge:answer",
            answer,
            source="self_inquiry",
            importance=0.7,
            metadata={"question_id": q_id},
        ):
            return

        with self._lock:
            # ISSUE 40 fix: Combine redundant connection blocks
            with self._get_conn() as conn:
                conn.execute("UPDATE questions SET answered = TRUE, answer = ?, answered_at = ? WHERE id = ?",
                         (answer, time.time(), q_id))
                
                row = conn.execute("SELECT question FROM questions WHERE id = ?", (q_id,)).fetchone()
                question = row[0] if row else "Unknown"
                conn.commit()
                
            self.add_knowledge(f"Q: {question}\nA: {answer}", type="qa",
                               source="self_inquiry", confidence=0.8)
    
    def get_unanswered_questions(self, limit: int = 10) -> List[Dict]:
        """Get unanswered"""
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            c.execute("SELECT * FROM questions WHERE answered = FALSE ORDER BY importance DESC LIMIT ?", (limit,))
            return [dict(row) for row in c.fetchall()]
    
    # ── Relational Edge Types ──────────────────────────────────────────────
    VALID_RELATION_TYPES = {
        "causes", "is_part_of", "contradicts", "supports",
        "requires", "evolved_from", "similar_to", "associated_with",
        "defined_as", "seen_in", "related_to"
    }

    def add_relationship(self, from_id: str, to_id: str,
                         relation_type: str, strength: float = 1.0) -> bool:
        """Add or strengthen a typed relationship between two knowledge nodes."""
        if relation_type not in self.VALID_RELATION_TYPES:
            logger.warning("Invalid relation type '%s'. Valid: %s",
                           relation_type, self.VALID_RELATION_TYPES)
            return False
        if not self._approve_memory_write(
            "knowledge:relationship",
            f"{from_id}:{relation_type}:{to_id}",
            source="graph_relationship",
            importance=max(0.0, min(1.0, float(strength or 0.0) / 5.0)),
        ):
            return False
        strength = max(0.0, min(5.0, strength))  # Clamp to [0, 5]
        with self._lock:
            with self._get_conn() as conn:
                c = conn.cursor()
                # Upsert: strengthen existing or create new
                c.execute(
                    """INSERT INTO relationships (from_id, to_id, relation_type, strength, created_at)
                       VALUES (?, ?, ?, ?, ?)
                       ON CONFLICT(from_id, to_id, relation_type)
                       DO UPDATE SET strength = MIN(5.0, strength + ?)""",
                    (from_id, to_id, relation_type, strength, time.time(), strength * 0.5),
                )
                conn.commit()
        logger.info("🔗 Relationship: %s -[%s]-> %s (strength=%.2f)",
                    from_id[:8], relation_type, to_id[:8], strength)
        return True

    def upsert_relationship(self, entity1: str, relation: str, entity2: str, weight: float = 1.0):
        """Phase 10: High-level convenience to link two text entities."""
        # 1. Ensure nodes exist
        id1 = self.add_knowledge(entity1, "concept", source="graph_contraction")
        id2 = self.add_knowledge(entity2, "concept", source="graph_contraction")
        
        # 2. Normalize relation
        rel = relation.lower().replace(" ", "_")
        if rel not in self.VALID_RELATION_TYPES:
            rel = "associated_with"
            
        # 3. Link
        return self.add_relationship(id1, id2, rel, strength=weight)

    def get_relationships(self, node_id: str,
                          direction: str = "both",
                          relation_type: Optional[str] = None) -> List[Dict]:
        """Get relationships for a node.
        
        Args:
            node_id: The node to query
            direction: "outgoing", "incoming", or "both"
            relation_type: Optional filter by relation type
        """
        results = []
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()

            queries = []
            if direction in ("outgoing", "both"):
                q = "SELECT from_id, to_id, relation_type, strength FROM relationships WHERE from_id = ?"
                params = [node_id]
                if relation_type:
                    q += " AND relation_type = ?"
                    params.append(relation_type)
                queries.append((q, params))

            if direction in ("incoming", "both"):
                q = "SELECT from_id, to_id, relation_type, strength FROM relationships WHERE to_id = ?"
                params = [node_id]
                if relation_type:
                    q += " AND relation_type = ?"
                    params.append(relation_type)
                queries.append((q, params))

            for q, params in queries:
                c.execute(q, params)
                for row in c.fetchall():
                    results.append(dict(row))
        return results

    def traverse(self, start_id: str, max_depth: int = 3,
                 relation_filter: Optional[str] = None) -> List[Dict]:
        """BFS traversal from a start node, returning all reachable nodes with depth.
        
        Returns list of {"node_id": str, "depth": int, "via_relation": str, "content": str}
        """
        visited = {start_id}
        queue = [(start_id, 0)]
        result = []

        while queue:
            current_id, depth = queue.pop(0)
            if depth >= max_depth:
                continue

            rels = self.get_relationships(current_id, direction="outgoing",
                                          relation_type=relation_filter)
            for rel in rels:
                neighbor_id = rel["to_id"]
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    # Fetch content for the neighbor
                    node = self.get_knowledge(neighbor_id)
                    content = node["content"][:100] if node else "(unknown)"
                    result.append({
                        "node_id": neighbor_id,
                        "depth": depth + 1,
                        "via_relation": rel["relation_type"],
                        "strength": rel["strength"],
                        "content": content,
                    })
                    queue.append((neighbor_id, depth + 1))

        return result

    async def find_path_async(self, from_id: str, to_id: str,
                             max_depth: int = 5, max_nodes: int = 2000) -> Optional[List[Dict]]:
        """Asynchronous, strictly bounded BFS to prevent event-loop freezing.
        
        Aura Hardening:
          1. Yields to event loop every 100 nodes.
          2. Circuit-breaker limit (max_nodes) prevents runaway resource consumption.
          3. Uses asyncio.to_thread for DB relationship fetching.
        """
        if from_id == to_id:
            return []

        visited = {from_id}
        queue = [(from_id, [])]
        nodes_explored = 0

        while queue:
            nodes_explored += 1
            
            # Yield control back to the orchestrator to prevent CPU starvation
            if nodes_explored % 100 == 0:
                await asyncio.sleep(0)
                
            # Circuit breaker: Graph is too dense, abort search to preserve VRAM/CPU
            if nodes_explored > max_nodes:
                logger.warning(f"Knowledge Graph traversal aborted: exceeded max node limit ({max_nodes}).")
                return None 

            current_id, path = queue.pop(0)
            
            if len(path) >= max_depth:
                continue
                
            # Fetch neighbors in a thread-safe way
            def _get_neighbors_sync():
                with self.safe_lock.acquire():
                    return self.get_relationships(current_id, direction="outgoing")
            
            rels = await asyncio.to_thread(_get_neighbors_sync)

            for rel in rels:
                neighbor_id = rel["to_id"]
                if neighbor_id == to_id:
                    return path + [{"node_id": neighbor_id, "relation": rel["relation_type"]}]
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, path + [
                        {"node_id": neighbor_id, "relation": rel["relation_type"]}
                    ]))

        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get stats"""
        with self._get_conn() as conn:
            rel_count = conn.execute("SELECT COUNT(*) FROM relationships").fetchone()[0]
            return {
                "total_knowledge": self.count_nodes(),
                "facts": self.count_nodes("fact"),
                "concepts": self.count_nodes("concept"),
                "skills": len(self.get_skills()),
                "relationships": rel_count,
                "people_known": conn.execute("SELECT COUNT(*) FROM people").fetchone()[0],
                "active_learning_goals": len(self.get_active_learning_goals()),
                "unanswered_questions": len(self.get_unanswered_questions())
            }

    def to_vis_data(self) -> Dict[str, List[Dict]]:
        """Export graph data for vis-network visualization."""
        nodes = []
        edges = []
        try:
            with self._get_conn() as conn:
                conn.row_factory = sqlite3.Row
                c = conn.cursor()
                
                # Fetch nodes
                c.execute("SELECT id, content, type, confidence, metadata FROM knowledge LIMIT 500")
                for row in c.fetchall():
                    meta = json.loads(row["metadata"]) if row["metadata"] else {}
                    if meta.get("test"):
                        continue
                        
                    # Color map based on type
                    colors = {
                        "fact": "#8a2be2",
                        "concept": "#00ffa3",
                        "qa": "#e6a800",
                        "learning": "#00e5ff",
                        "reflection": "#ff00ff",
                        "observation": "#b44dff",
                        "preference": "#00e5ff"
                    }
                    nodes.append({
                        "id": row["id"],
                        "label": row["content"][:20] + "..." if len(row["content"]) > 20 else row["content"],
                        "title": row["content"][:100],
                        "type": row["type"],
                        "color": colors.get(row["type"], "#4a4a4a"),
                        "value": row["confidence"] * 5
                    })
                
                # Fetch edges
                c.execute("SELECT from_id, to_id, relation_type, strength FROM relationships LIMIT 1000")
                for row in c.fetchall():
                    edges.append({
                        "from": row["from_id"],
                        "to": row["to_id"],
                        "label": row["relation_type"],
                        "width": row["strength"] * 2
                    })
            
            # If empty, add a primary node
            if not nodes:
                nodes.append({"id": "aura-core", "label": "Aura Core", "color": "#ff00ff"})

            return {"nodes": nodes, "edges": edges}
        except Exception as e:
            logger.error("Failed to generate vis data: %s", e)
            return {"nodes": [{"id": "error", "label": "Graph Error", "color": "red"}], "edges": []}

    def get_random_node(self) -> Optional[str]:
        """Returns a random knowledge node's content."""
        with self._get_conn() as conn:
            c = conn.cursor()
            c.execute("SELECT content FROM knowledge ORDER BY RANDOM() LIMIT 1")
            row = c.fetchone()
            return row[0] if row else None

    def get_sparse_nodes(self, limit: int = 5) -> List[str]:
        """Identify nodes with the few relationships (novelty targets)."""
        query = """
        SELECT k.content, COUNT(r.from_id) as rel_count
        FROM knowledge k
        LEFT JOIN relationships r ON k.id = r.from_id OR k.id = r.to_id
        GROUP BY k.id
        ORDER BY rel_count ASC
        LIMIT ?
        """
        with self._get_conn() as conn:
            c = conn.cursor()
            c.execute(query, (limit,))
            return [row[0] for row in c.fetchall() if row[0]]

    def get_recent_nodes(self, limit: int = 5, type: Optional[str] = None) -> List[Dict]:
        """Fetch the most recently created or accessed knowledge nodes.
        Used by AgencyCore to drive autonomous interests.
        """
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            if type:
                c.execute("""SELECT * FROM knowledge WHERE type = ? 
                            ORDER BY last_accessed DESC LIMIT ?""", (type, limit))
            else:
                c.execute("""SELECT * FROM knowledge 
                            ORDER BY last_accessed DESC LIMIT ?""", (limit,))
            return [dict(row) for row in c.fetchall()]

# Alias for compatibility
KnowledgeGraph = PersistentKnowledgeGraph
