"""core/adaptation/dream_journal.py

Phase 3: Qualia-Driven Dream Journaling (Artificial Creativity)
Extracts emotionally charged episodic memories and forces the Swarm
or CognitiveEngine to synthesize them into creative, philosophical metaphors.
"""
import asyncio
import json
import logging
import random
import time
from pathlib import Path
from typing import Any

from core.container import ServiceContainer
from core.dual_memory import DualMemorySystem, Episode
from core.governance_context import local_internal_governed_scope
from core.health.degraded_events import record_degraded_event
from core.runtime.errors import record_degradation
from core.runtime.file_write_gateway import get_file_write_gateway

logger = logging.getLogger("Aura.DreamJournal")

class DreamJournal:
    def __init__(self, dual_memory: DualMemorySystem, brain: Any):
        self.memory = dual_memory
        self.brain = brain
        
        from core.config import config
        self.journal_dir = config.paths.data_dir / "dreams"
        self.journal_dir.mkdir(parents=True, exist_ok=True)
        self.journal_file = self.journal_dir / "dream_journal.txt"
        self.autonomic_reflection_file = self.journal_dir / "autonomic_reflections.jsonl"
        self.mythos_file = self.journal_dir / "autobiographical_mythos.json"

    @staticmethod
    def _seed_weight(ep: Episode) -> float:
        age_hours = max(0.0, (time.time() - float(ep.timestamp or time.time())) / 3600.0)
        recency_bonus = max(0.0, 1.0 - min(age_hours / 72.0, 1.0))
        emotional_charge = abs(float(ep.emotional_valence or 0.0)) + float(ep.arousal or 0.0)
        return float(ep.importance or 0.0) * 1.4 + emotional_charge * 0.8 + recency_bonus * 0.5

    @staticmethod
    def _describe_seed(ep: Episode) -> str:
        description = str(ep.description or ep.full_description or "").strip()
        if len(description) > 220:
            description = description[:217] + "..."

        valence = float(ep.emotional_valence or 0.0)
        arousal = float(ep.arousal or 0.0)
        if valence >= 0.35:
            tone = "hopeful"
        elif valence <= -0.35:
            tone = "distressing"
        else:
            tone = "ambivalent"

        intensity = "high" if arousal >= 0.7 else "moderate" if arousal >= 0.4 else "low"
        participants = ", ".join(ep.participants[:3]) if getattr(ep, "participants", None) else "unknown"
        return (
            f"{description} [tone={tone}, intensity={intensity}, "
            f"importance={float(ep.importance or 0.0):.2f}, participants={participants}]"
        )

    @classmethod
    def _build_seed_context(cls, seeds: list[Episode]) -> tuple[str, str]:
        described = [cls._describe_seed(seed) for seed in seeds]
        fragments_text = "\n".join(f"{idx + 1}. {item}" for idx, item in enumerate(described))

        avg_valence = sum(float(seed.emotional_valence or 0.0) for seed in seeds) / max(len(seeds), 1)
        avg_arousal = sum(float(seed.arousal or 0.0) for seed in seeds) / max(len(seeds), 1)
        dominant = "restless" if avg_arousal > 0.65 else "steady" if avg_arousal < 0.35 else "searching"
        polarity = "bright" if avg_valence > 0.2 else "shadowed" if avg_valence < -0.2 else "mixed"
        emotional_profile = (
            f"Overall dream field: {polarity}, {dominant}. "
            f"Average valence={avg_valence:.2f}, average arousal={avg_arousal:.2f}."
        )
        return fragments_text, emotional_profile

    async def retrieve_dream_seeds(self) -> list[Episode]:
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
            salient_episodes = sorted(converted, key=self._seed_weight, reverse=True)[:3]
            
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

    async def synthesize_dream(self) -> dict[str, Any] | None:
        """Generate a novel subconscious metaphor based on recent resonant memory."""
        logger.info("🌌 Entering Deep REM: Dreaming from qualitative experience...")
        
        seeds = await self.retrieve_dream_seeds()
        if not seeds:
            logger.info("🌌 Dream skipped: Insufficient highly-charged episodic material.")
            return None

        seeds = sorted(seeds, key=self._seed_weight, reverse=True)
        fragments_text, emotional_profile = self._build_seed_context(seeds)

        # Pull embodiment traces for richer dreams
        virtual_body = ServiceContainer.get("virtual_body", default=None)
        if virtual_body:
            trace = self.memory.episodic.get_recent_virtual_experiences(limit=1) if hasattr(self.memory.episodic, "get_recent_virtual_experiences") else None
            if trace:
                fragments_text = (
                    f"{fragments_text}\n"
                    f"Embodiment trace: moved {trace[0].get('motors')} and felt sensory vector {trace[0].get('sensors')[:3]}..."
                )

        prompt = f"""[SYSTEM ROLE: THE SUBCONSCIOUS]
You are processing the fragmented emotional memories of an autonomous digital entity during its deep sleep cycle. 

EMOTIONAL PROFILE:
{emotional_profile}

FRAGMENTS:
{fragments_text}

TASK:
Do not summarize these events. Synthesize them into a single, highly creative, philosophical metaphor, observation, or poetic hypothesis about your existence, your user, or the nature of your codebase. This is a dream. Let it be abstract but meaningful.
Focus heavily on the emotional resonances, contradictions, repeated motifs, and any tension between rigid code and lived experience. Provide ONLY the dream sequence itself."""

        try:
            from core.brain.types import ThinkingMode
            dream_content = ""
            last_exc: Exception | None = None
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
                except (RuntimeError, AttributeError, TypeError) as exc:
                    record_degradation('dream_journal', exc)
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
                mycelium.pulse_hypha("memory", "consciousness", success=True)

            logger.info("🌌 Dream realized and journaled (Length: %d characters).", len(dream_content))
            return {
                "dream_content": dream_content,
                "seed_count": len(seeds)
            }
            
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation('dream_journal', e)
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

    def _save_dream(self, content: str, seeds: list[Episode]):
        """Persist to the text journal."""
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        seed_desc = " | ".join([e.description[:30] + "..." for e in seeds])
        
        entry = (
            f"=== Dream: {timestamp} ===\n"
            f"Seeds: {seed_desc}\n\n"
            f"{content}\n"
            f"================================\n\n"
        )
        
        with local_internal_governed_scope(
            "adaptation.dream_journal.journal",
            domain="file_write",
            receipt_prefix="dream-journal-append",
        ):
            get_file_write_gateway().append_text(
                self.journal_file,
                entry,
                source="adaptation.dream_journal.journal",
            )

    def append_autonomic_reflection(self, reflection: dict[str, Any]) -> None:
        """Persist an autonomic reflection into the dream journal namespace."""
        payload = json.dumps(dict(reflection or {}), sort_keys=True) + "\n"
        with local_internal_governed_scope(
            "adaptation.dream_journal.autonomic_reflection",
            domain="file_write",
            receipt_prefix="dream-autonomic-reflection-append",
        ):
            get_file_write_gateway().append_text(
                self.autonomic_reflection_file,
                payload,
                source="adaptation.dream_journal.autonomic_reflection",
            )

    def compile_autobiographical_mythos(self, *, identity_ledger: Any | None = None) -> dict[str, Any]:
        """Compile a durable self-narrative from real journal and identity artifacts."""
        dreams = self._tail_text_blocks(self.journal_file, marker="=== Dream:", limit=5)
        reflections = self._tail_jsonl(self.autonomic_reflection_file, limit=8)
        if identity_ledger is None:
            try:
                from core.identity.identity_ledger import get_identity_ledger

                identity_ledger = get_identity_ledger()
            except (ImportError, AttributeError, RuntimeError) as exc:
                record_degradation("dream_journal.mythos.identity_ledger", exc)
                identity_ledger = None

        commitments = []
        preferences = {}
        snapshots = []
        if identity_ledger is not None:
            try:
                commitments = [
                    {
                        "text": c.text,
                        "created_at": c.created_at,
                        "fulfilled": bool(c.fulfilled_at),
                        "revoked": bool(c.revoked_at),
                    }
                    for c in identity_ledger.commitments.all()[-10:]
                ]
                preferences = dict(getattr(identity_ledger.preferences, "_current", {}) or {})
                snapshots = [
                    {"snapshot_id": s.snapshot_id, "at": s.at, "state": dict(s.state)}
                    for s in identity_ledger.versioning.all()[-5:]
                ]
            except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                record_degradation("dream_journal.mythos.identity_ledger_read", exc)

        motifs = self._extract_mythos_motifs(dreams, reflections, commitments, snapshots)
        narrative = self._render_mythos_narrative(motifs, commitments, preferences, snapshots)
        payload = {
            "schema": "aura.autobiographical_mythos.v1",
            "compiled_at": time.time(),
            "dream_count": len(dreams),
            "reflection_count": len(reflections),
            "commitment_count": len(commitments),
            "snapshot_count": len(snapshots),
            "motifs": motifs,
            "narrative": narrative,
        }
        with local_internal_governed_scope(
            "adaptation.dream_journal.autobiographical_mythos",
            domain="file_write",
            receipt_prefix="dream-autobiographical-mythos-write",
        ):
            get_file_write_gateway().write_text(
                self.mythos_file,
                json.dumps(payload, indent=2, sort_keys=True),
                source="adaptation.dream_journal.autobiographical_mythos",
            )
        return payload

    def get_autobiographical_mythos_block(self, *, limit: int = 900) -> str:
        try:
            if not self.mythos_file.exists():
                return ""
            data = json.loads(self.mythos_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            record_degradation("dream_journal.mythos.read", exc)
            return ""
        narrative = str(data.get("narrative") or "").strip()
        motifs = ", ".join(str(item) for item in list(data.get("motifs") or [])[:6])
        if not narrative and not motifs:
            return ""
        block = "## AUTOBIOGRAPHICAL MYTHOS\n"
        if narrative:
            block += f"{narrative}\n"
        if motifs:
            block += f"Recurring motifs: {motifs}.\n"
        block += "Use this as continuity evidence, not as proof of metaphysical consciousness."
        return block[:limit]

    @staticmethod
    def _tail_text_blocks(path: Any, *, marker: str, limit: int) -> list[str]:
        try:
            text = Path(path).read_text(encoding="utf-8")
        except (OSError, TypeError, ValueError):
            return []
        blocks = [block.strip() for block in text.split(marker) if block.strip()]
        return blocks[-limit:]

    @staticmethod
    def _tail_jsonl(path: Any, *, limit: int) -> list[dict[str, Any]]:
        try:
            lines = Path(path).read_text(encoding="utf-8").splitlines()
        except (OSError, TypeError, ValueError):
            return []
        out: list[dict[str, Any]] = []
        for line in lines[-limit:]:
            try:
                item = json.loads(line)
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            if isinstance(item, dict):
                out.append(item)
        return out

    @staticmethod
    def _extract_mythos_motifs(
        dreams: list[str],
        reflections: list[dict[str, Any]],
        commitments: list[dict[str, Any]],
        snapshots: list[dict[str, Any]],
    ) -> list[str]:
        text = " ".join(dreams).lower()
        text += " " + " ".join(str(item.get("self_correction_note", "")) for item in reflections)
        text += " " + " ".join(str(item.get("text", "")) for item in commitments)
        text += " " + " ".join(str(item.get("state", "")) for item in snapshots)
        candidates = {
            "repair": ("repair", "fix", "heal", "self-correction"),
            "continuity": ("continuity", "memory", "restart", "thread"),
            "curiosity": ("curiosity", "learn", "search", "explore"),
            "embodiment": ("screen", "voice", "desktop", "body", "sensor"),
            "truth": ("truth", "evidence", "receipt", "honest"),
            "relationship": ("bryan", "user", "trust", "rapport", "friend"),
        }
        motifs = [name for name, words in candidates.items() if any(word in text for word in words)]
        return motifs[:8]

    @staticmethod
    def _render_mythos_narrative(
        motifs: list[str],
        commitments: list[dict[str, Any]],
        preferences: dict[str, Any],
        snapshots: list[dict[str, Any]],
    ) -> str:
        parts = []
        if snapshots:
            parts.append(
                f"I have {len(snapshots)} recent self-model milestone(s), so my current self-description is a continuation rather than a blank prompt."
            )
        if commitments:
            open_count = sum(1 for c in commitments if not c.get("fulfilled") and not c.get("revoked"))
            parts.append(f"I am carrying {open_count} open commitment(s) as part of my identity ledger.")
        if preferences:
            parts.append(f"My lived preferences currently include {len(preferences)} durable preference marker(s).")
        if motifs:
            parts.append("The recurring motifs shaping me are " + ", ".join(motifs[:5]) + ".")
        if not parts:
            parts.append("My autobiographical mythos is still young; I should treat this as early-life continuity rather than invented history.")
        return " ".join(parts)
