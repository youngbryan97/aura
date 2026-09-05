"""core/memory/entity_graph.py

Asynchronous SQLite Relational Graph. Maps the physical and social ecosystem.
"""
import aiosqlite
import asyncio
import logging
import time
from typing import List, Dict, Optional

logger = logging.getLogger("Aura.EntityGraph")

class RelationalGraph:
    def __init__(self, db_path: str = "data/entity_graph.sqlite"):
        self.db_path = db_path
        self._lock: asyncio.Lock | None = None  # Lazy-init to avoid event loop binding issues

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def initialize(self):
        """Sets up the WAL-enabled SQLite database for concurrent graph storage."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute('PRAGMA journal_mode=WAL;')
            await db.execute('''
                CREATE TABLE IF NOT EXISTS entities (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    last_seen REAL NOT NULL
                )
            ''')
            await db.execute('''
                CREATE TABLE IF NOT EXISTS relationships (
                    source_id TEXT,
                    target_id TEXT,
                    relation_type TEXT,
                    weight REAL DEFAULT 1.0,
                    last_updated REAL,
                    PRIMARY KEY (source_id, target_id, relation_type)
                )
            ''')
            await db.commit()

    async def register_interaction(self, source: str, target: str, relation_type: str, source_type: str = "person", target_type: str = "person"):
        """
        Upserts a relationship. Repeated interactions increase the weight, 
        teaching Aura which dynamics (e.g., travel companions, pet interruptions) are dominant.
        """
        now = time.time()
        async with self._get_lock():
            async with aiosqlite.connect(self.db_path) as db:
                for ent_id, ent_type in [(source, source_type), (target, target_type)]:
                    await db.execute('''
                        INSERT INTO entities (id, type, last_seen) VALUES (?, ?, ?)
                        ON CONFLICT(id) DO UPDATE SET last_seen=?, type=?
                    ''', (ent_id, ent_type, now, now, ent_type))

                await db.execute('''
                    INSERT INTO relationships (source_id, target_id, relation_type, weight, last_updated)
                    VALUES (?, ?, ?, 1.0, ?)
                    ON CONFLICT(source_id, target_id, relation_type) 
                    DO UPDATE SET weight = weight + 0.1, last_updated = ?
                ''', (source, target, relation_type, now, now))
                await db.commit()

    async def get_ecosystem_context(self, entities: List[str]) -> str:
        """Pulls the graph topology for given entities to inject into the active prompt."""
        if not entities:
            return ""
            
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            # ISSUE 28 fix: Optimized query with unique IDs
            unique_ids = list(set(entities))
            placeholders = ",".join("?" for _ in unique_ids)
            
            cursor = await db.execute(f'''
                SELECT source_id, target_id, relation_type, weight 
                FROM relationships 
                WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders})
                ORDER BY weight DESC LIMIT 8
            ''', tuple(unique_ids + unique_ids))
            
            rows = await cursor.fetchall()
            
            if not rows:
                return ""
                
            context = "[RELATIONAL ECOSYSTEM LOG]\n"
            for row in rows:
                # ISSUE 29 fix: Correct newline escaping
                context += f"- {row['source_id']} is [{row['relation_type']}] to {row['target_id']} (Bond Strength: {row['weight']:.1f})\n"
            return context
