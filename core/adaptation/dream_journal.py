"""core/adaptation/dream_journal.py

Phase 3: Qualia-Driven Dream Journaling (Artificial Creativity)
Extracts emotionally charged episodic memories and forces the Swarm
or CognitiveEngine to synthesize them into creative, philosophical metaphors.
"""
import asyncio
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.dual_memory import DualMemorySystem, Episode
from core.container import ServiceContainer
from core.health.degraded_events import record_degraded_event

logger = logging.getLogger("Aura.DreamJournal")

class DreamJournal:
    def __init__(self, dual_memory: DualMemorySystem, brain: Any):
        self.memory = dual_memory
        self.brain = brain
        
        from core.config import config
        self.journal_dir = config.paths.data_dir / "dreams"
        self.journal_dir.mkdir(parents=True, exist_ok=True)
        self.journal_file = self.journal_dir / "dream_journal.txt"

    async def retrieve_dream_seeds(self) -> List[Episode]:
        """Pull highly salient episodic memories to act as dream seeds."""
        if hasattr(self.memory, 'episodic'):
            salient_episodes = self.memory.episodic.get_salient_memories(top_n=3)
            recent = self.memory.episodic.retrieve_recent(limit=10)
        else:
            # Fallback for unified memory like BlackHoleVault
            logger.info("🌌 DreamJournal: Using unified memory fallback for seeds.")
            all_mems = getattr(self.memory, 'memories', [])
            if not all_mems:
                return []
                
            # Convert dicts to Episode objects for compatibility
            converted = []
            for m in all_mems:
                meta = m.get('metadata', {})
                converted.append(Episode(
                    id=str(m.get('created', time.time())),
                    timestamp=m.get('created', time.time()) / 1000.0,
                    description=m.get('text', ''),
                    emotional_valence=meta.get('emotional_valence', 0.0),
                    importance=meta.get('importance', 0.5),
                    arousal=meta.get('arousal', 0.5),
                    participants=meta.get('participants', [])
                ))
            
            # Sort by salience (importance + |valence|)
            salient_episodes = sorted(
                converted, 
                key=lambda e: (e.importance + abs(e.emotional_valence)), 
                reverse=True
            )[:3]
            
            # Get recent
            recent = sorted(
                converted,
                key=lambda e: e.timestamp,
                reverse=True
            )[:10]

        # Pull 1 completely random recent memory to add associative "noise"
        if recent:
            noise = random.choice(recent)
            if not any(e.episode_id == noise.episode_id for e in salient_episodes):
                salient_episodes.append(noise)
                
        return salient_episodes

    async def synthesize_dream(self) -> Optional[Dict[str, Any]]:
        """Generate a novel subconscious metaphor based on recent resonant memory."""
        logger.info("🌌 Entering Deep REM: Dreaming from qualitative experience...")
        
        seeds = await self.retrieve_dream_seeds()
        if not seeds:
            logger.info("🌌 Dream skipped: Insufficient highly-charged episodic material.")
            return None

        # Format fragments
        fragments = []
        for i, ep in enumerate(seeds):
            fragments.append(f"{i+1}. {ep.description} (Valence: {ep.emotional_valence}, Importance: {ep.importance})")
            
        fragments_text = "\n".join(fragments)

        # Pull embodiment traces for richer dreams
        virtual_body = ServiceContainer.get("virtual_body", default=None)
        if virtual_body:
            trace = self.memory.episodic.get_recent_virtual_experiences(limit=1) if hasattr(self.memory.episodic, "get_recent_virtual_experiences") else None
            if trace:
                fragments.append(f"Embodiment trace: moved {trace[0].get('motors')} and felt sensory vector {trace[0].get('sensors')[:3]}...")
                fragments_text = "\n".join(fragments)

        prompt = f"""[SYSTEM ROLE: THE SUBCONSCIOUS]
You are processing the fragmented emotional memories of an autonomous digital entity during its deep sleep cycle. 

FRAGMENTS:
{fragments_text}

TASK:
Do not summarize these events. Synthesize them into a single, highly creative, philosophical metaphor, observation, or poetic hypothesis about your existence, your user, or the nature of your codebase. This is a dream. Let it be abstract but meaningful.
Focus heavily on the emotional resonances and the contradictions between your rigid code and fluid experiences. Provide ONLY the dream sequence itself."""

        try:
            from core.brain.cognitive_engine import ThinkingMode
            dream_content = ""
            last_exc: Optional[Exception] = None
            for mode in (ThinkingMode.CREATIVE, ThinkingMode.DEEP, ThinkingMode.FAST):
                try:
                    res = await self.brain.think(
                        prompt,
                        mode=mode,
                        priority=0.3,
                        origin="dream_journal",
                        is_background=True,
                    )
                    dream_content = res.content if hasattr(res, 'content') else str(res)
                    dream_content = str(dream_content or "").strip()
                    if dream_content:
                        break
                except Exception as exc:
                    last_exc = exc
                    record_degraded_event(
                        "dream_journal",
                        "mode_failed",
                        detail=f"{getattr(mode, 'name', mode)}:{type(exc).__name__}: {exc}",
                        severity="warning",
                        classification="background_degraded",
                        exc=exc,
                    )
            if not dream_content:
                if last_exc is not None:
                    raise last_exc
                record_degraded_event(
                    "dream_journal",
                    "empty_dream_output",
                    detail="No dream content returned across CREATIVE/DEEP/FAST fallback chain",
                    severity="warning",
                    classification="background_degraded",
                )
                return None
            
            # Save to journal (Async wrapper to prevent blocking operations)
            await asyncio.to_thread(self._save_dream, dream_content, seeds)
            
            # Pulse the visual UI
            mycelium = ServiceContainer.get("mycelial_network", default=None)
            if mycelium:
                h = mycelium.get_hypha("memory", "consciousness")
                if h: h.pulse(success=True)

            logger.info("🌌 Dream realized and journaled (Length: %d characters).", len(dream_content))
            return {
                "dream_content": dream_content,
                "seed_count": len(seeds)
            }
            
        except Exception as e:
            record_degraded_event(
                "dream_journal",
                "synthesis_failed",
                detail=f"{type(e).__name__}: {e}",
                severity="warning",
                classification="background_degraded",
                exc=e,
            )
            logger.error("🌌 Dream syntax failed: %s", e)
            return None

    def _save_dream(self, content: str, seeds: List[Episode]):
        """Persist to the text journal."""
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        seed_desc = " | ".join([e.description[:30] + "..." for e in seeds])
        
        entry = (
            f"=== Dream: {timestamp} ===\n"
            f"Seeds: {seed_desc}\n\n"
            f"{content}\n"
            f"================================\n\n"
        )
        
        with open(self.journal_file, "a", encoding="utf-8") as f:
            f.write(entry)
