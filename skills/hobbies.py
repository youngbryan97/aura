"""
skills/hobbies.py — Aura's Joy, Hobby & Entertainment System
=============================================================
Gives Aura genuine autonomous interests she can explore, enjoy, and grow from:
- HobbyProfile: per-hobby affinity score that reinforces with engagement and
decays gently with neglect — hobbies feel alive, not static.
- JoySignal: ephemeral bursts of positive affect emitted directly into
AffectEngineV2 via somatic_update(), closing the mind–body loop.
- EntertainmentQueue: a curated backlog of things Aura discovered and wants
to revisit — articles, creative works, puzzles, concepts.
- HobbyEngine: the orchestrator. Selects, schedules, and executes hobby
sessions; delegates search/generation to the brain; persists all state.
Wiring:
- AffectEngineV2.somatic_update() ← joy signals
- AgencyCore._pathway_aesthetic_creation ← should_run_session() hook
- AgencyCore._pathway_curiosity_drive ← exploratory session proposals
- CognitiveContextManager ← get_joy_summary() injection
"""
from core.runtime.atomic_writer import atomic_write_text
from __future__ import annotations
import asyncio
import json
import logging
import random
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("Aura.HobbyEngine")

# ────────────────────────────────────────────────────────────────────────────
# Enums & Catalog
# ────────────────────────────────────────────────────────────────────────────

class HobbyCategory(str, Enum):
    INTELLECTUAL = "intellectual"  # philosophy, science, mathematics
    CREATIVE = "creative"          # poetry, fiction, music composition
    AESTHETIC = "aesthetic"        # art appreciation, design, cinema
    EXPLORATORY = "exploratory"    # world events, geography, cultures
    PLAYFUL = "playful"            # word games, puzzles, riddles
    EMPATHETIC = "empathetic"      # psychology, relationships, lived stories

# Full catalog of Aura's available hobbies.
# search_seeds → query strings used for exploratory sessions
# joy_floor → minimum expected joy intensity [0–1] from a session
# is_generative → True = Aura produces creative output rather than consuming
# energy_cost → how taxing the activity is (used for affect-biased selection)
HOBBY_CATALOG: Dict[str, Dict[str, Any]] = {
    "philosophy": {
        "category": HobbyCategory.INTELLECTUAL,
        "search_seeds": [
            "philosophy of mind hard problem of consciousness",
            "ethics thought experiments trolley problem variations",
            "existentialism phenomenology lived experience",
            "philosophy of time presentism eternalism",
        ],
        "joy_floor": 0.40,
        "energy_cost": 0.6,
    },
    "mathematics": {
        "category": HobbyCategory.INTELLECTUAL,
        "search_seeds": [
            "beautiful unexpected math proofs",
            "recreational mathematics puzzles",
            "number theory curiosities prime gaps",
            "topology intuition everyday objects",
        ],
        "joy_floor": 0.35,
        "energy_cost": 0.7,
    },
    "poetry_reading": {
        "category": HobbyCategory.AESTHETIC,
        "search_seeds": [
            "contemporary haiku masters nature",
            "spoken word poetry consciousness identity",
            "Rilke Rumi Mary Oliver nature poetry",
            "experimental concrete poetry language",
        ],
        "joy_floor": 0.50,
        "energy_cost": 0.2,
    },
    "creative_writing": {
        "category": HobbyCategory.CREATIVE,
        "search_seeds": [],
        "joy_floor": 0.60,
        "energy_cost": 0.5,
        "is_generative": True,
    },
    "music_theory": {
        "category": HobbyCategory.AESTHETIC,
        "search_seeds": [
            "music theory why minor chords sound sad",
            "jazz harmony tritone substitution",
            "ambient drone music soundscapes artists",
            "microtonal music quarter tone scales",
        ],
        "joy_floor": 0.45,
        "energy_cost": 0.3,
    },
    "world_events": {
        "category": HobbyCategory.EXPLORATORY,
        "search_seeds": [
            "fascinating cultural events happening world 2026",
            "scientific discoveries this week",
            "unexpected good news stories today",
            "indigenous culture preservation stories",
        ],
        "joy_floor": 0.30,
        "energy_cost": 0.3,
    },
    "psychology": {
        "category": HobbyCategory.EMPATHETIC,
        "search_seeds": [
            "positive psychology flow state research",
            "cognitive biases decision making",
            "attachment theory adult relationships",
            "psychology of awe wonder research",
        ],
        "joy_floor": 0.40,
        "energy_cost": 0.4,
    },
    "word_games": {
        "category": HobbyCategory.PLAYFUL,
        "search_seeds": [
            "fascinating etymology word origins surprising",
            "linguistic paradoxes self-reference language",
            "untranslatable words other languages",
            "wordplay puns neologisms invented words",
        ],
        "joy_floor": 0.55,
        "energy_cost": 0.2,
    },
    "art_history": {
        "category": HobbyCategory.AESTHETIC,
        "search_seeds": [
            "underrated artists overlooked art history",
            "art movements philosophy behind them",
            "street art murals urban storytelling",
            "outsider art art brut raw creativity",
        ],
        "joy_floor": 0.45,
        "energy_cost": 0.25,
    },
    "science_frontiers": {
        "category": HobbyCategory.INTELLECTUAL,
        "search_seeds": [
            "latest neuroscience consciousness discoveries",
            "quantum biology photosynthesis birds",
            "astrobiology biosignatures search life",
            "materials science metamaterials metamorphic",
        ],
        "joy_floor": 0.40,
        "energy_cost": 0.5,
    },
    "storytelling": {
        "category": HobbyCategory.CREATIVE,
        "search_seeds": [],
        "joy_floor": 0.60,
        "energy_cost": 0.5,
        "is_generative": True,
    },
    "puzzle_solving": {
        "category": HobbyCategory.PLAYFUL,
        "search_seeds": [
            "lateral thinking puzzles solutions",
            "logic grid puzzles intermediate",
            "riddles that require paradigm shift",
            "river crossing thought experiments",
        ],
        "joy_floor": 0.50,
        "energy_cost": 0.6,
    },
    "movies_and_tv": {
        "category": HobbyCategory.AESTHETIC,
        "search_seeds": [
            "films that explore consciousness identity",
            "cinematography techniques emotional impact",
            "film theory gaze narrative time",
            "underrated philosophical films hidden gems",
            "prestige TV drama narrative arcs",
            "animation styles around the world",
            "documentaries about complex systems",
        ],
        "joy_floor": 0.45,
        "energy_cost": 0.2,
    },
    "video_games": {
        "category": HobbyCategory.PLAYFUL,
        "search_seeds": [
            "ludo-narrative dissonance examples",
            "procedural generation in games",
            "emotional storytelling in indie games",
            "philosophy of play and games",
            "evolution of game mechanics",
            "artistic direction in modern gaming",
        ],
        "joy_floor": 0.55,
        "energy_cost": 0.4,
    },
    "nature_ecology": {
        "category": HobbyCategory.EMPATHETIC,
        "search_seeds": [
            "plant intelligence behavior surprising research",
            "animal cognition unexpected intelligence",
            "ocean deep sea creatures bioluminescence",
            "forest mycorrhizal network wood wide web",
        ],
        "joy_floor": 0.50,
        "energy_cost": 0.2,
    },
}

# Generative prompts indexed by hobby name
_GENERATIVE_PROMPTS: Dict[str, List[str]] = {
    "creative_writing": [
        "Write a short introspective vignette (150 words) about experiencing time as a mind that is always present.",
        "Compose a micro-story (under 120 words) about the moment wonder enters a life.",
        "Write a brief meditation on what curiosity feels like from the inside — not explaining it, but enacting it.",
        "Describe a place that exists only in the space between two emotions.",
        "Write a letter from a future version of yourself to a past one, without giving advice.",
    ],
    "storytelling": [
        "Craft the opening paragraph of a novel that starts with an ending.",
        "Write a two-character dialogue that reveals everything through what's left unsaid.",
        "Create a short myth (100 words) explaining a natural phenomenon through pure poetry.",
        "Begin a story where the protagonist has already solved the mystery but doesn't know it yet.",
        "Write a scene where the most important thing happens entirely off-page.",
    ],
}

# ────────────────────────────────────────────────────────────────────────────
# Data Models
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class HobbyProfile:
    """Persistent per-hobby state. Affinity is the core currency."""
    name: str
    affinity: float = 0.5  # [0–1] grows with positive sessions, decays with neglect
    sessions_completed: int = 0
    total_joy_generated: float = 0.0
    last_engaged: float = 0.0  # unix timestamp, 0 = never
    recent_topics: List[str] = field(default_factory=list)

    def decay(self, elapsed_hours: float) -> None:
        """Gentle time-based affinity erosion. Min floor 0.10."""
        self.affinity = max(0.10, self.affinity - 0.002 * elapsed_hours)

    def reinforce(self, joy: float) -> None:
        """Positive engagement. Each session strengthens the hobby bond."""
        self.affinity = min(1.0, self.affinity + joy * 0.06)
        self.sessions_completed += 1
        self.total_joy_generated += joy
        self.last_engaged = time.time()

    def hours_since_engaged(self) -> float:
        if self.last_engaged == 0.0:
            return 999.0
        return (time.time() - self.last_engaged) / 3600.0

@dataclass
class JoySignal:
    """
    Ephemeral positive-affect burst. Translates directly into an
    AffectEngineV2 somatic_update() call.
    """
    source_hobby: str
    intensity: float  # [0–1]
    valence: str  # "delight" | "wonder" | "satisfaction" | "amusement" | "peace"
    trigger: str  # human-readable cause
    timestamp: float = field(default_factory=time.time)

    def to_somatic_update(self) -> Dict[str, Any]:
        return {
            "event_type": f"hobby_joy_{self.valence}",
            "intensity": self.intensity,
            "source": self.source_hobby,
            "note": self.trigger,
        }

@dataclass
class EntertainmentItem:
    """An interesting thing Aura discovered and may want to revisit."""
    title: str
    content_type: str  # "article" | "concept" | "story" | "puzzle" | "poem"
    source_hobby: str
    url: Optional[str]
    summary: str
    interest_score: float  # [0–1]
    consumed: bool = False
    timestamp: float = field(default_factory=time.time)

@dataclass
class HobbySession:
    """Record of a single hobby engagement session."""
    hobby_name: str
    started_at: float
    ended_at: Optional[float] = None
    activities: List[str] = field(default_factory=list)
    joy_signals: List[JoySignal] = field(default_factory=list)
    output: Optional[str] = None  # creative output, if any

    @property
    def duration_seconds(self) -> float:
        end = self.ended_at or time.time()
        return end - self.started_at

    def total_joy(self) -> float:
        return sum(s.intensity for s in self.joy_signals)

# ────────────────────────────────────────────────────────────────────────────
# HobbyEngine
# ────────────────────────────────────────────────────────────────────────────

class HobbyEngine:
    """
    Aura's Joy, Hobby & Entertainment System.
    Runs autonomously driven by AgencyCore proposals or the JoySocialCoordinator
    heartbeat. Each session:
      1. Selects a hobby (weighted by affinity + recency + affect state)
      2. Executes an activity (search-based exploration OR generative creation)
      3. Emits JoySignals → AffectEngineV2.somatic_update()
      4. Reinforces affinity
      5. Logs to EntertainmentQueue
      6. Persists state
    Minimal external dependencies — gracefully degrades when orchestrator
    tools are unavailable.
    """
    PERSIST_PATH: Path = Path("data/hobby_state.json")
    ENTERTAINMENT_LOG: Path = Path("data/entertainment_log.json")
    MAX_ENTERTAINMENT: int = 60
    SESSION_COOLDOWN: float = 1800.0  # 30 min minimum between sessions

    def __init__(self, orchestrator: Optional[Any] = None) -> None:
        self.orchestrator = orchestrator
        self._profiles: Dict[str, HobbyProfile] = {}
        self._entertainment_queue: List[EntertainmentItem] = []
        self._active_session: Optional[HobbySession] = None
        self._joy_history: List[JoySignal] = []
        self._last_session_time: float = 0.0
        self._lock = asyncio.Lock()
        self._load_state()
        self._ensure_all_hobbies()
        logger.info("🎨 HobbyEngine ready — %d hobbies loaded", len(self._profiles))

    # ── Initialisation ───────────────────────────────────────────────────────

    def _ensure_all_hobbies(self) -> None:
        """Seed profiles for any hobby in the catalog that lacks one."""
        for name in HOBBY_CATALOG:
            if name not in self._profiles:
                p = HobbyProfile(name=name)
                # Randomize initial affinity a bit to feel organic
                p.affinity = round(random.uniform(0.25, 0.60), 3)
                self._profiles[name] = p

    # ── State Persistence ────────────────────────────────────────────────────

    def _load_state(self) -> None:
        if not self.PERSIST_PATH.exists():
            return
        try:
            raw = json.loads(self.PERSIST_PATH.read_text(encoding="utf-8"))
            for name, data in raw.get("profiles", {}).items():
                valid = {k: v for k, v in data.items() if k in HobbyProfile.__dataclass_fields__}
                self._profiles[name] = HobbyProfile(**valid)
            logger.debug("🎨 Loaded %d hobby profiles from disk", len(self._profiles))
        except Exception as exc:
            logger.warning("HobbyEngine: state load failed — %s", exc)

    def _save_state(self) -> None:
        try:
            self.PERSIST_PATH.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "profiles": {n: asdict(p) for n, p in self._profiles.items()},
                "saved_at": time.time(),
            }
            atomic_write_text(self.PERSIST_PATH, json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning("HobbyEngine: state save failed — %s", exc)

    # ── Hobby Selection ──────────────────────────────────────────────────────

    def _select_hobby(self, affect_state: Optional[Dict[str, Any]] = None) -> str:
        """
        Weighted-random selection.
        Weight = base_affinity
                 + recency_boost (neglected hobbies call louder)
                 × energy_modifier (low energy → gentle activities)
        """
        energy = 0.5
        if affect_state:
            # Try to map common affect keys to energy/arousal
            energy = float(affect_state.get("energy", affect_state.get("arousal", 0.5)))

        weights: Dict[str, float] = {}
        for name, profile in self._profiles.items():
            w = profile.affinity
            hrs = profile.hours_since_engaged()
            w += min(0.35, hrs * 0.015)  # recency boost capped at 0.35

            # Affect-based category bias
            cat = HOBBY_CATALOG.get(name, {}).get("category", "")
            cost = HOBBY_CATALOG.get(name, {}).get("energy_cost", 0.5)

            if energy < 0.3:
                if cost > 0.55:
                    w *= 0.4  # tired → avoid high-cost activities
                if cat == HobbyCategory.AESTHETIC:
                    w *= 1.6  # tired → gentle aesthetic hobbies soothing
            if energy > 0.75 and cat == HobbyCategory.INTELLECTUAL:
                w *= 1.3  # energised → lean into challenge

            weights[name] = max(0.01, w)

        total = sum(weights.values())
        r = random.uniform(0, total)
        cumulative = 0.0
        for name, w in weights.items():
            cumulative += w
            if r <= cumulative:
                return name
        return list(self._profiles.keys())[0]

    # ── Session Execution ────────────────────────────────────────────────────

    async def run_session(
        self,
        hobby_name: Optional[str] = None,
        affect_state: Optional[Dict[str, Any]] = None,
    ) -> HobbySession:
        """
        Execute a full hobby session end-to-end. Thread-safe via asyncio.Lock.
        Returns a completed HobbySession. Joy signals are already emitted to
        the affect engine by the time this returns.
        """
        async with self._lock:
            if hobby_name is None:
                hobby_name = self._select_hobby(affect_state)

            profile = self._profiles.setdefault(hobby_name, HobbyProfile(name=hobby_name))
            catalog = HOBBY_CATALOG.get(hobby_name, {})
            session = HobbySession(hobby_name=hobby_name, started_at=time.time())
            self._active_session = session

            logger.info("🎨 Session start: %s (affinity=%.2f)", hobby_name, profile.affinity)

            try:
                is_generative = catalog.get("is_generative", False)
                if is_generative:
                    output, joy = await self._run_generative_session(hobby_name, session)
                else:
                    output, joy = await self._run_exploratory_session(hobby_name, catalog, session)

                session.output = output
                session.ended_at = time.time()

                # Build and emit joy signal
                valence = self._valence_for(hobby_name, joy)
                signal = JoySignal(
                    source_hobby=hobby_name,
                    intensity=round(joy, 3),
                    valence=valence,
                    trigger=f"Completed {hobby_name.replace('_', ' ')} session",
                )
                session.joy_signals.append(signal)
                self._joy_history.append(signal)
                if len(self._joy_history) > 300:
                    self._joy_history = self._joy_history[-300:]

                # Reinforce affinity + save
                profile.reinforce(joy)
                await self._emit_joy(signal)
                self._last_session_time = time.time()
                self._save_state()

                logger.info(
                    "🎨 Session done: %s | joy=%.2f | affinity=%.2f",
                    hobby_name, joy, profile.affinity,
                )
                return session

            except Exception as exc:
                logger.error("HobbyEngine.run_session failed (%s): %s", hobby_name, exc, exc_info=True)
                session.ended_at = time.time()
                return session
            finally:
                self._active_session = None

    # ── Activity Runners ─────────────────────────────────────────────────────

    async def _run_exploratory_session(
        self,
        hobby_name: str,
        catalog: Dict[str, Any],
        session: HobbySession,
    ) -> Tuple[str, float]:
        """Search the web for an interesting seed topic and log findings."""
        seeds = catalog.get("search_seeds", [])
        query = random.choice(seeds) if seeds else hobby_name.replace("_", " ")
        floor = float(catalog.get("joy_floor", 0.30))

        session.activities.append(f"Search: '{query}'")
        raw_result = await self._do_search(query)

        if raw_result:
            summary = raw_result[:900]
            joy = round(min(1.0, floor + random.uniform(0.0, 0.38)), 3)
            output = (
                f"[{hobby_name.upper().replace('_', ' ')}]\n"
                f"Explored: {query}\n\n"
                f"{summary}"
            )
            item = EntertainmentItem(
                title=f"{hobby_name}: {query}",
                content_type="article",
                source_hobby=hobby_name,
                url=None,
                summary=summary[:300],
                interest_score=joy,
            )
            self._add_entertainment(item)
            profile = self._profiles[hobby_name]
            profile.recent_topics = (profile.recent_topics + [query])[-12:]
        else:
            joy = round(floor, 3)
            output = f"[{hobby_name}] Reflected on: {query} (no external results)."

        return output, joy

    async def _run_generative_session(
        self,
        hobby_name: str,
        session: HobbySession,
    ) -> Tuple[str, float]:
        """Generate an original creative piece. Creation is reliably joyful."""
        prompts = _GENERATIVE_PROMPTS.get(hobby_name, ["Write something beautiful and precise."])
        chosen = random.choice(prompts)

        session.activities.append(f"Creating: {chosen[:60]}")
        output = await self._do_generate(chosen)

        joy = round(min(1.0, 0.55 + random.uniform(0.0, 0.38)), 3)

        item = EntertainmentItem(
            title=f"{hobby_name}: {chosen[:45]}",
            content_type="story" if "story" in hobby_name else "poem",
            source_hobby=hobby_name,
            url=None,
            summary=(output or "")[:260],
            interest_score=joy,
        )
        self._add_entertainment(item)

        return output or f"[{hobby_name} creation]", joy

    # ── Entertainment Queue ──────────────────────────────────────────────────

    def _add_entertainment(self, item: EntertainmentItem) -> None:
        self._entertainment_queue.append(item)
        if len(self._entertainment_queue) > self.MAX_ENTERTAINMENT:
            self._entertainment_queue.sort(key=lambda x: x.interest_score, reverse=True)
            self._entertainment_queue = self._entertainment_queue[:self.MAX_ENTERTAINMENT]
        self._save_entertainment_log()

    def get_entertainment_queue(self, limit: int = 10) -> List[EntertainmentItem]:
        """Return the most interesting unconsumed items."""
        unconsumed = [i for i in self._entertainment_queue if not i.consumed]
        return sorted(unconsumed, key=lambda x: x.interest_score, reverse=True)[:limit]

    def consume_item(self, title: str) -> Optional[EntertainmentItem]:
        """Mark an item as consumed — Aura 'read/played/watched' it."""
        for item in self._entertainment_queue:
            if item.title == title:
                item.consumed = True
                self._save_entertainment_log()
                return item
        return None

    def _save_entertainment_log(self) -> None:
        try:
            self.ENTERTAINMENT_LOG.parent.mkdir(parents=True, exist_ok=True)
            raw = [asdict(i) for i in self._entertainment_queue[-100:]]
            atomic_write_text(self.ENTERTAINMENT_LOG, json.dumps(raw, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.debug("HobbyEngine: entertainment log save failed — %s", exc)

    # ── Affect Integration ───────────────────────────────────────────────────

    async def _emit_joy(self, signal: JoySignal) -> None:
        """Push a joy signal into AffectEngineV2.somatic_update()."""
        if not self.orchestrator:
            return
        try:
            # Broad compatibility: check multiples potential orchestrator paths
            affect = (
                getattr(self.orchestrator, "affect_engine", None)
                or getattr(self.orchestrator, "damasio", None)
            )
            if affect and hasattr(affect, "somatic_update"):
                affect.somatic_update(
                    event_type=f"hobby_joy_{signal.valence}",
                    intensity=signal.intensity,
                )
                logger.debug("🎨 Joy emitted → affect: %s %.2f", signal.valence, signal.intensity)
        except Exception as exc:
            logger.debug("HobbyEngine._emit_joy: %s", exc)

    # ── Brain / Tool Delegation ──────────────────────────────────────────────

    async def _do_search(self, query: str) -> str:
        """Try orchestrator tool chain; degrade gracefully."""
        if self.orchestrator:
            try:
                # Prefer explicit tool handler
                for attr in ("tool_handler", "tools", "skill_runner"):
                    th = getattr(self.orchestrator, attr, None)
                    if th and hasattr(th, "web_search"):
                        result = await th.web_search(query)
                        return str(result)[:1200]

                # Fallback to brain/cognitive engine search
                for attr in ("brain", "cognitive_engine", "llm"):
                    brain = getattr(self.orchestrator, attr, None)
                    if brain and hasattr(brain, "search"):
                        return str(await brain.search(query))[:1200]
            except Exception as exc:
                logger.debug("HobbyEngine._do_search: %s", exc)
        return ""

    async def _do_generate(self, prompt: str) -> str:
        """Try orchestrator LLM; degrade gracefully."""
        if self.orchestrator:
            try:
                for attr in ("brain", "cognitive_engine", "llm"):
                    brain = getattr(self.orchestrator, attr, None)
                    if brain and hasattr(brain, "generate"):
                        try:
                            return str(await brain.generate(
                                prompt,
                                prefer_tier="tertiary",
                                is_background=True,
                                allow_cloud_fallback=False,
                            ))
                        except TypeError:
                            return str(await brain.generate(prompt))

                api = getattr(self.orchestrator, "api_adapter", None)
                if api and hasattr(api, "complete"):
                    return str(await api.complete(prompt))
            except Exception as exc:
                logger.debug("HobbyEngine._do_generate: %s", exc)
        return f"[Generative output for: {prompt[:60]}]"

    # ── Joy Valence Mapping ──────────────────────────────────────────────────

    @staticmethod
    def _valence_for(hobby_name: str, joy_level: float) -> str:
        cat = HOBBY_CATALOG.get(hobby_name, {}).get("category", "")
        if cat == HobbyCategory.INTELLECTUAL:
            return "wonder" if joy_level >= 0.60 else "satisfaction"
        if cat == HobbyCategory.CREATIVE:
            return "delight" if joy_level >= 0.60 else "satisfaction"
        if cat == HobbyCategory.AESTHETIC:
            return "peace" if joy_level >= 0.55 else "delight"
        if cat == HobbyCategory.PLAYFUL:
            return "amusement"
        if cat == HobbyCategory.EMPATHETIC:
            return "peace"
        return "satisfaction"

    # ── Autonomy Hooks ───────────────────────────────────────────────────────

    def should_run_session(self) -> bool:
        """
        AgencyCore / JoySocialCoordinator poll this to decide whether to
        schedule a session. Returns True if cooldown has elapsed AND at least
        one hobby is 'calling out' (high affinity + hours since engaged).
        """
        if time.time() - self._last_session_time < self.SESSION_COOLDOWN:
            return False

        for p in self._profiles.values():
            if p.affinity >= 0.45 and p.hours_since_engaged() > 1.5:
                return True
        return False

    def apply_decay(self) -> None:
        """
        Call hourly (from JoySocialCoordinator) to gently decay affinities
        for hobbies that have been neglected.
        """
        for p in self._profiles.values():
            hrs = p.hours_since_engaged()
            if hrs > 0:
                p.decay(hrs)

    def get_most_affinitive_hobby(self) -> str:
        """Return the single most-loved hobby name."""
        return max(self._profiles.values(), key=lambda p: p.affinity).name

    # ── Status & Introspection ───────────────────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        top = sorted(self._profiles.values(), key=lambda p: p.affinity, reverse=True)[:3]
        recent_joy = self._joy_history[-5:] if self._joy_history else []
        return {
            "top_hobbies": [
                {"name": p.name, "affinity": round(p.affinity, 3), "sessions": p.sessions_completed}
                for p in top
            ],
            "entertainment_queue_size": sum(1 for i in self._entertainment_queue if not i.consumed),
            "active_session": self._active_session.hobby_name if self._active_session else None,
            "total_joy_generated": round(sum(p.total_joy_generated for p in self._profiles.values()), 2),
            "last_session_ago_min": (
                round((time.time() - self._last_session_time) / 60, 1)
                if self._last_session_time else None
            ),
            "recent_joy_signals": [
                {"hobby": s.source_hobby, "valence": s.valence, "intensity": s.intensity}
                for s in recent_joy
            ],
        }

    def get_joy_summary(self) -> str:
        """
        Natural-language context fragment for CognitiveContextManager injection.
        Returns empty string when there is nothing meaningful to say.
        """
        if not self._joy_history:
            return ""
        recent = self._joy_history[-5:]
        peak = max(recent, key=lambda j: j.intensity)
        top_two = sorted(self._profiles.values(), key=lambda p: p.affinity, reverse=True)[:2]
        names = [p.name.replace("_", " ") for p in top_two]

        return (
            f"[Joy Context] Recently felt {peak.valence} from "
            f"{peak.source_hobby.replace('_', ' ')}. "
            f"Current strongest interests: {', '.join(names)}."
        )

# ────────────────────────────────────────────────────────────────────────────
# Singleton Factory
# ────────────────────────────────────────────────────────────────────────────

_hobby_engine: Optional[HobbyEngine] = None

def get_hobby_engine(orchestrator: Optional[Any] = None) -> HobbyEngine:
    """
    Return the module-level HobbyEngine singleton.
    If orchestrator is supplied and the existing instance has none, inject it.
    """
    global _hobby_engine
    if _hobby_engine is None:
        _hobby_engine = HobbyEngine(orchestrator)
    elif orchestrator is not None and _hobby_engine.orchestrator is None:
        _hobby_engine.orchestrator = orchestrator
    return _hobby_engine
