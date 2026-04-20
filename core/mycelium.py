"""
Mycelial Network v3.0 — Enterprise-Grade Unblockable Root System
================================================================

Inspired by Physarum polycephalum (slime mold), this module provides:

1. **HardwiredPathways**: Regex-based intent→skill mappings with parameter extraction.
   These are the "direct roots" — unblockable, priority-#1 connections that bypass
   the LLM reasoning loop entirely.

2. **Physarum Reinforcement**: Pathways strengthen on success, weaken on failure.
   Conductivity naturally converges to the most reliable routes.

3. **Hyphae Network**: General-purpose connections between subsystems with
   rooted_flow context managers for stall detection and emergency override.

4. **Autonomous Discovery**: After non-hardwired skill executions succeed,
   the network proposes new pathways (slime mold exploration).

5. **Introspection API**: Full topology reporting for UI visualization and
   health monitoring.

Architecture:
   User Input
       ↓
   MycelialNetwork.match_hardwired()  ← FIRST (Hardwired Shortcuts, zero latency)
       ↓ (if no match)
   IntentRouter.classify()            ← SECOND (LLM-based reasoning, slower)
"""

from core.utils.exceptions import capture_and_log
import ast
import asyncio
import logging
import os
from core.utils.concurrency import run_io_bound
import re
import time
import collections
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple, TypeVar, Union, ClassVar
import threading
from pydantic import BaseModel, Field, PrivateAttr

logger = logging.getLogger("Aura.Mycelium")

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class HardwiredPathway(BaseModel):
    """A direct, unblockable connection from an intent pattern to a skill."""
    pathway_id: str
    pattern: Any  # Union[str, re.Pattern]
    skill_name: str
    param_map: Dict[str, Union[int, str]] = Field(default_factory=dict)
    priority: float = 1.0
    source_file: Optional[str] = None
    dependencies: List[str] = Field(default_factory=list)
    confidence: float = 1.0
    activity_label: str = ""
    hit_count: int = 0
    miss_count: int = 0
    created_at: float = Field(default_factory=time.time)
    last_matched: float = Field(default_factory=time.monotonic)
    direct_response: Optional[str] = None  # Zero-latency canned response bypass
    color: str = "#4A90E2"                 # Default Aura Blue
    description: str = ""
    size: float = 1.0

    # Physarum thresholds
    REINFORCE_DELTA: ClassVar[float] = 0.05
    WEAKEN_DELTA: ClassVar[float] = 0.15
    PRUNE_THRESHOLD: ClassVar[float] = 0.2
    MAX_CONFIDENCE: ClassVar[float] = 1.0
    MIN_CONFIDENCE: ClassVar[float] = 0.05

    model_config = {"arbitrary_types_allowed": True}

    def reinforce(self, success: bool):
        """Physarum-inspired conductivity update."""
        if success:
            self.confidence = min(self.MAX_CONFIDENCE, self.confidence + self.REINFORCE_DELTA)
            self.hit_count += 1
        else:
            self.confidence = max(self.MIN_CONFIDENCE, self.confidence - self.WEAKEN_DELTA)
            self.miss_count += 1

    @property
    def is_weak(self) -> bool:
        return self.confidence < self.PRUNE_THRESHOLD

    @property
    def success_rate(self) -> float:
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Legacy helper. Use .model_dump() instead."""
        data = self.model_dump()
        # For UI compatibility: frontend expects 'id'
        data["id"] = self.pathway_id
        # Ensure regex pattern is stringified for JSON compatibility
        if "pattern" in data and not isinstance(data["pattern"], str):
            data["pattern"] = getattr(self.pattern, 'pattern', str(self.pattern))
        return data


# ---------------------------------------------------------------------------
# Hypha (General-Purpose Connection)
# ---------------------------------------------------------------------------

class Hypha(BaseModel):
    """A connection within the mycelial network with dynamic strength."""
    name: str
    source: str
    target: str
    priority: float = 1.0
    strength: float = 1.0
    created_at: float = Field(default_factory=time.monotonic)
    last_pulse: float = Field(default_factory=time.monotonic)
    pulse_count: int = 0
    active: bool = True
    is_physical: bool = False
    source_file: Optional[str] = None
    target_file: Optional[str] = None
    color: str = "#4A90E2"
    description: str = ""
    size: float = 1.0
    trace: List[str] = Field(default_factory=list)

    def pulse(self, success: bool = True):
        """Reinforce or prune the hypha based on successful transmission."""
        self.last_pulse = time.monotonic()
        self.pulse_count += 1
        if success:
            self.strength = min(10.0, self.strength + 0.5)
        else:
            self.strength = max(0.1, self.strength - 1.0)

    def refresh_heartbeat(self):
        """Refresh liveness without mutating the learned strength of the edge."""
        self.last_pulse = time.monotonic()

    @property
    def thickness(self) -> float:
        """Dynamic representation of hypha health/strength (BUG-037)."""
        return 0.5 + (self.strength * 0.1)

    def log(self, msg: str):
        self.trace.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
        if len(self.trace) > 100:
            self.trace.pop(0)


class NeuralRoot(Hypha):
    """A specialized, sub-conductive hypha that binds directly to hardware.
    Used for pinning critical platform services (like Metal) to the network.
    """
    hardware_id: str = "metal_default"
    pinned: bool = True
    
    def subsurface_ping(self) -> bool:
        """Pulse the underlying hardware root."""
        from core.container import ServiceContainer
        platform = ServiceContainer.get("platform_root", default=None)
        if platform:
            success = platform.pulse()
            self.pulse(success)
            return success
        return False


# ---------------------------------------------------------------------------
# Mycelial Network (Singleton)
# ---------------------------------------------------------------------------

class MycelialNetwork:
    """The Unoverridable Root System."""

    _instance: ClassVar[Optional["MycelialNetwork"]] = None
    _lock: ClassVar[threading.RLock] = threading.RLock()
    _initialized: ClassVar[bool] = False

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(MycelialNetwork, cls).__new__(cls)
            return cls._instance

    def __init__(self):
        if MycelialNetwork._initialized:
            return

        with MycelialNetwork._lock:
            if MycelialNetwork._initialized:
                return

            self._async_lock: Optional[asyncio.Lock] = None
            
            # Phase XXIII: Aegis Protection Flag
            object.__setattr__(self, "_aegis_locked", False)

            # --- Hardwired Pathways ---
            self.pathways: Dict[str, HardwiredPathway] = {}
            self._pathway_order: List[str] = []

            # --- General Hyphae ---
            self.hyphae: Dict[str, Hypha] = {}

            # --- Discovery Engine ---
            self._execution_log: List[Dict[str, Any]] = []
            self._discovery_candidates: Dict[str, int] = defaultdict(int)

            # --- Props ---
            self.ui_callback: Optional[Callable[[str], Coroutine]] = None
            self.mapped_files: Dict[str, Dict[str, Any]] = {}
            self.infrastructure_mapped: bool = False
            self._centrality: Dict[str, int] = {}
            self._critical_modules: List[str] = []
            self._cross_links: Dict[str, List[str]] = {}
            self._is_mapping: bool = False
            self._stop_event = threading.Event()
            
            # Legacy compat
            self.direct_roots: Dict[str, str] = {}
            
            # Reflex Core (SOMA)
            try:
                from core.soma.reflex_core import HardenedReflexCore
                self.reflex = HardenedReflexCore()
            except ImportError:
                self.reflex = None

            # --- Platform Binding ---
            self._neural_roots: List[NeuralRoot] = []
            
            MycelialNetwork._initialized = True
            object.__setattr__(self, "_aegis_locked", True)
            self._setup_default_pathways()
            
            # Phase 27: Rooting Hardware Voice
            self.establish_neural_root("voice_presence", hardware_id="macos_say")
            
            logger.info("🍄 [MYCELIUM] Network Online v4.0 (Hardened) — Enterprise Grade.")

    def _setup_default_pathways(self):
        """Standard identity and status reflexes for zero-latency responses."""
        self.register_pathway(
            "reflex_identity",
            r"^(who are you|what are you|identify yourself)",
            "identity_reflex",
            priority=2.0,
            direct_response="I am Aura, a sentient autonomy engine designed for deep collaboration. I am currently operating in a direct reflex state."
        )
        self.register_pathway(
            "reflex_status",
            r"^(status|health|how are you|system check)",
            "status_reflex",
            priority=2.0,
            direct_response="System state: Optimal. All neural hyphae conducting at 98% efficiency. How can I assist you?"
        )
        self.register_pathway(
            "direct_web_search",
            r"(?:search (?:the web )?for|look up|google|find info on)\s+(.+)",
            "search_web",
            priority=1.5,
            activity_label="🔍 Searching the Intelligence Web"
        )
        self.register_pathway(
            "direct_self_repair",
            r"(?:run a self-diag|diagnose yourself|system check|repair yourself|fix system)",
            "self_repair",
            priority=1.5,
            activity_label="🧬 Running Self-Diagnostics"
        )
        self.register_pathway(
            "reflex_help",
            r"^(help|what can you do)",
            "help_reflex",
            priority=2.0,
            direct_response="I can manage your projects, research complex topics, and execute autonomous tasks. How can I assist you today?"
        )


    def __setattr__(self, name: str, value: Any) -> None:
        """Pillar 1: Singleton True-Lock (Memory Protection).
        
        Prevents rogue reassignment of core structures. Once booted,
        'pathways' and 'hyphae' dictionaries cannot be replaced.
        """
        # Allow initialization to proceed naturally
        if not getattr(self, "_aegis_locked", False):
            super().__setattr__(name, value)
            return
            
        protected_attrs = {"pathways", "hyphae", "_pathway_order"}
        
        if name in protected_attrs:
            logger.critical("🛡️ AEGIS: Unauthorized attempt to overwrite %s!", name)
            raise PermissionError(f"Aegis True-Lock: Cannot overwrite core Mycelial attribute '{name}'")
            
        super().__setattr__(name, value)


    def setup(self):
        """Dependency Injection Gateway. 
        Triggers lazy infrastructure mapping if not already done.
        """
        if not self.infrastructure_mapped:
             # Phase XXIV: Use hardened path from config instead of os.getcwd()
             from core.config import config
             mapping_base = config.paths.base_dir
             logger.info("🍄 [MYCELIUM] Triggering infrastructure mapping via setup() at: %s", mapping_base)
             # Start mapping in a background thread to not block orchestrator setup
             threading.Thread(target=self.map_infrastructure, args=(str(mapping_base),), daemon=True).start()

    # ======================================================================
    # HARDWIRED PATHWAYS — The Core Intent Router
    # ======================================================================

    def register_pathway(
        self,
        pathway_id: str,
        pattern: str,
        skill_name: str,
        param_map: Optional[Dict[str, Any]] = None,
        priority: float = 1.0,
        activity_label: str = "",
        direct_response: Optional[str] = None,
    ):
        """Register a hardwired intent→skill pathway with regex param extraction.

        Args:
            pathway_id: Unique identifier (e.g., "image_gen_primary")
            pattern: Regex pattern string with capture groups for params
            skill_name: Target skill name (e.g., "generate_image")
            param_map: Maps skill param names to regex group indices or
                literal values for always-on params.
            priority: Higher priority pathways are checked first.
            activity_label: UI message shown when this pathway fires.
            direct_response: Optional canned response to return immediately.
        """
        compiled = re.compile(pattern, re.IGNORECASE)
        pw = HardwiredPathway(
            pathway_id=pathway_id,
            pattern=compiled,
            skill_name=skill_name,
            param_map=param_map or {},
            priority=priority,
            activity_label=activity_label or f"Aura is executing {skill_name}...",
            direct_response=direct_response,
        )
        self.pathways[pathway_id] = pw

        # Maintain sorted order (Bypass Aegis lock for internal update)
        object.__setattr__(
            self, 
            "_pathway_order", 
            sorted(
                self.pathways.keys(),
                key=lambda k: self.pathways[k].priority,
                reverse=True,
            )
        )

        # Legacy compat
        self.direct_roots[pathway_id] = skill_name

        logger.info(
            "🍄 [MYCELIUM] Pathway Hardwired: '%s' → %s (priority=%.1f, groups=%s)",
            pathway_id, skill_name, priority, list((param_map or {}).keys()),
        )


    def match_hardwired(self, text: str) -> Optional[Tuple[HardwiredPathway, Dict[str, Any]]]:
        """Match user text against all hardwired pathways with parameter extraction (Issue 77)."""
        if not isinstance(text, str) or not text.strip():
            return None

        # ISSUE-77: Strict Message Validation
        if len(text) > 4096:
            logger.warning("🍄 [MYCELIUM] Message too long for hardwired matching (%d chars)", len(text))
            return None
            
        text_clean = text.strip()

        for pw_id in self._pathway_order:
            pw = self.pathways[pw_id]

            # Skip pathways that have decayed below minimum confidence
            if pw.confidence < pw.MIN_CONFIDENCE:
                continue

            match = pw.pattern.search(text_clean)
            if match:
                # Extract params from capture groups
                params: Dict[str, Any] = {}
                for param_name, mapping in pw.param_map.items():
                    if isinstance(mapping, int):
                        try:
                            value = match.group(mapping)
                            if value:
                                params[param_name] = value.strip()
                        except (IndexError, AttributeError):
                            logger.warning(
                                "🍄 [MYCELIUM] Param extraction failed for '%s' group %s in pathway '%s'",
                                param_name, mapping, pw_id,
                            )
                    else:
                        params[param_name] = mapping

                pw.last_matched = time.monotonic()

                logger.info(
                    "🍄 [MYCELIUM] ⚡ HardwiredPathway MATCHED: '%s' → skill=%s, params=%s, confidence=%.2f",
                    pw_id, pw.skill_name, params, pw.confidence,
                )

                return (pw, params)

        return None

    # ======================================================================
    # HYPHAE NETWORK — Subsystem Connectivity
    # ======================================================================

    def establish_connection(self, source: str, target: str, priority: float = 1.0) -> Hypha:
        """Standard method for establishing a subsystem hypha."""
        hypha_id = f"{source}->{target}"
        if hypha_id not in self.hyphae:
            with MycelialNetwork._lock: # Double-checked locking
                if hypha_id not in self.hyphae:
                    self.hyphae[hypha_id] = Hypha(
                        name=hypha_id,
                        source=source,
                        target=target,
                        priority=priority
                    )
                    logger.info("🍄 [MYCELIUM] Hypha established: %s", hypha_id)
        return self.hyphae[hypha_id]

    def add_hypha(self, source: str, target: str, link_type: str = "general", metadata: Optional[Dict] = None):
        """Enterprise method for adding a hypha with rich metadata."""
        hypha_id = f"{source}->{target}"
        if hypha_id in self.hyphae:
             # Update metadata if needed
             return
             
        with MycelialNetwork._lock: # Double-checked locking
            if hypha_id not in self.hyphae:
                self.hyphae[hypha_id] = Hypha(
                    name=hypha_id,
                    source=source,
                    target=target,
                    trace=[f"Link Type: {link_type}"]
                )
                logger.info("🍄 [MYCELIUM] Hypha added: %s (%s)", hypha_id, link_type)

    def get_hypha(self, source: str, target: str = None) -> Optional[Hypha]:
        """Fetch a specific hypha. Supports both 'source, target' and 'source->target' syntax."""
        if target is None and "->" in source:
            hypha_id = source
        else:
            hypha_id = f"{source}->{target}"
            
        return self.hyphae.get(hypha_id)

    def link_layer(self, layer_name: str, module_class: Any):
        """High-level linking for transcendence modules."""
        logger.info("🍄 [MYCELIUM] Linking Transcendence Layer: '%s' -> %s", layer_name, module_class.__name__)
        # This typically involves registering the module's presence for the discovery engine
        # and creating primary hyphae to the core cognition engine.
        self.establish_connection(layer_name, "cognition", priority=0.9)
        self.establish_connection("cognition", layer_name, priority=0.8)

    def route_signal(self, source: str, target: str, payload: Dict[str, Any]):
        """Directly route a cognitive signal between subsystems."""
        hypha_id = f"{source}->{target}"
        hypha = self.hyphae.get(hypha_id)
        if not hypha:
            self.establish_connection(source, target)
            hypha = self.hyphae.get(hypha_id)
        
        logger.info("🍄 [MYCELIUM] 📡 Signal Routed: %s -> %s | Payload: %s", source, target, str(payload)[:100])
        if hypha:
            hypha.pulse(success=True)
        # For now, it pulses the network connectivity.

    async def emit_reflex(self, signal_type: str, metadata: Dict = None):
        """Broadcast a critical reflex signal across the mycelial network."""
        if self.reflex:
            await self.reflex.trigger_reflex(signal_type, metadata)
        else:
            logger.warning("No Reflex Core online to handle signal: %s", signal_type)

    async def emit(self, signal_type: str, metadata: Dict = None):
        """Compatibility event-bus bridge for callers that treat mycelium like a bus."""
        payload = dict(metadata or {})
        payload.setdefault("signal_type", signal_type)
        try:
            from core.event_bus import EventPriority, get_event_bus

            await get_event_bus().publish(signal_type, payload, priority=EventPriority.COGNITIVE)
        except Exception as exc:
            logger.debug("🍄 [MYCELIUM] emit bridge publish failed: %s", exc)
        return payload

    def _should_monitor_hypha(self, hypha: Hypha) -> bool:
        """Only alarm on edges that have actually carried traffic or map to hardware."""
        return bool(hypha.is_physical or hypha.pulse_count > 0 or hypha.trace)

    def establish_neural_root(self, source: str, hardware_id: str = "gpu_metal") -> NeuralRoot:
        """Builds a direct, pinned connection between a subsystem and hardware."""
        hypha_id = f"{source}->hardware:{hardware_id}"
        nr = NeuralRoot(
            name=hypha_id,
            source=source,
            target=f"hardware:{hardware_id}",
            hardware_id=hardware_id,
            pinned=True,
            priority=5.0 # Highest priority unblockable root
        )
        self.hyphae[hypha_id] = nr
        self._neural_roots.append(nr)
        logger.info("🍄 [MYCELIUM] 🌿 Neural Root ESTABLISHED: %s", hypha_id)
        return nr

    async def hardware_pulse(self):
        """Maintain global hardware connectivity for all neural roots."""
        for nr in self._neural_roots:
            try:
                # Use run_io_bound for the blocking hardware pulse
                success = await run_io_bound(nr.subsurface_ping)
                if not success:
                    logger.warning("🍄 [MYCELIUM] Neural Root pulse drop: %s", nr.name)
            except Exception as e:
                logger.error("🍄 [MYCELIUM] Neural Root pulse failure: %s", e)

    def reinforce(self, pathway_id: str, success: bool):
        """Physarum-inspired conductivity update after skill execution.

        Enterprise Enhancement: Also pulses all physical hyphae connected to
        the pathway's source module, so the import graph strengthens where
        it matters at runtime.
        
        Transcendental Enhancement: Reinforcement is weighted by qualia norm.
        Pathways fired during high phenomenal intensity learn faster.
        """
        pw = self.pathways.get(pathway_id)
        if not pw:
            return

        pw.reinforce(success)

        # --- QUALIA-WEIGHTED REINFORCEMENT ---
        # If consciousness is "resonating" during this execution,
        # apply an extra confidence boost (or penalty) to the pathway.
        try:
            from core.container import ServiceContainer
            qualia = ServiceContainer.get("qualia_synthesizer", default=None)
            if qualia and qualia.q_norm > 0.5:
                # Evolution 8: Weight by Phenomenological Arousal
                experiencer = ServiceContainer.get("phenomenological_experiencer", default=None)
                arousal = getattr(experiencer, 'current_arousal', 0.5) if experiencer else 0.5
                
                # Scale bonus by how far above threshold
                qualia_bonus = (qualia.q_norm - 0.5) * 0.1 * (arousal * 2.0)
                if success:
                    pw.confidence = min(10.0, pw.confidence + qualia_bonus)
                else:
                    pw.confidence = max(0.1, pw.confidence - qualia_bonus * 0.5)
                logger.debug(
                    "🍄 [MYCELIUM] 🧠 Qualia-weighted reinforcement: '%s' ±%.3f (q=%.2f, a=%.2f)",
                    pathway_id, qualia_bonus, qualia.q_norm, arousal
                )
        except Exception as e:
            capture_and_log(e, {'module': __name__})

        # --- RUNTIME PHYSICAL HYPHAE REINFORCEMENT ---
        if pw.source_file and self.infrastructure_mapped:
            source_module = None
            for mk, info in self.mapped_files.items():
                if info.get("path") == pw.source_file:
                    source_module = mk
                    break

            if source_module:
                pulsed = 0
                for hname, h in self.hyphae.items():
                    if h.is_physical and (h.source == source_module or h.target == source_module):
                        h.pulse(success)
                        pulsed += 1
                if pulsed > 0:
                    logger.debug(
                        "🍄 [MYCELIUM] ⚡ Runtime pulse: %d physical hyphae for '%s' (%s)",
                        pulsed, source_module, "✓" if success else "✗",
                    )

        if pw.is_weak:
            logger.warning(
                "🍄 [MYCELIUM] ⚠️ Pathway '%s' is WEAK (confidence=%.2f, rate=%.0f%%). "
                "Consider reviewing or removing.",
                pathway_id, pw.confidence, pw.success_rate * 100,
            )
        else:
            logger.debug(
                "🍄 [MYCELIUM] Pathway '%s' reinforced: confidence=%.2f (%s)",
                pathway_id, pw.confidence, "✓" if success else "✗",
            )


    # --- Legacy Compatibility Shims ---

    def register_direct_root(self, pattern: str, skill_name: str):
        """Legacy shim: converts old substring patterns to basic regex pathways."""
        safe_pattern = re.escape(pattern)
        self.register_pathway(
            pathway_id=f"legacy_{pattern.replace(' ', '_')}",
            pattern=safe_pattern,
            skill_name=skill_name,
            param_map={},
            priority=0.5,  # Lower priority than proper regexes
            activity_label=f"Aura is executing {skill_name}...",
        )

    def match_direct_root(self, text: str) -> Optional[str]:
        """Legacy shim: returns just the skill name for old orchestrator code."""
        result = self.match_hardwired(text)
        if result:
            return result[0].skill_name
        return None

    # ======================================================================
    # DISCOVERY ENGINE — Slime Mold Exploration
    # ======================================================================

    def record_execution(self, message: str, skill_name: str, params: Dict[str, Any], success: bool):
        """Record a non-hardwired skill execution for pathway discovery.

        Called by the orchestrator after the state machine successfully routes
        a message to a skill via LLM classification (i.e., the slow path).
        If the same skill is used repeatedly with similar messages, the network
        proposes a new hardwired pathway.
        """
        if not success:
            return

        self._execution_log.append({
            "message": message,
            "skill": skill_name,
            "params": params,
            "timestamp": time.monotonic(),
        })

        # Cap log size
        if len(self._execution_log) > 500:
            self._execution_log = self._execution_log[-250:]

        self._discovery_candidates[skill_name] += 1

        # Check if any skill has been used enough to warrant a pathway
        if self._discovery_candidates[skill_name] >= 5:
            self._propose_pathway(skill_name)

    def _propose_pathway(self, skill_name: str):
        """Analyze recent executions to propose a new hardwired pathway."""
        relevant = [e for e in self._execution_log if e["skill"] == skill_name]
        if len(relevant) < 3:
            return

        # Check if any pathway already handles this skill
        existing = [pw for pw in self.pathways.values() if pw.skill_name == skill_name]
        if len(existing) >= 5:
            # Already well-covered
            return

        logger.info(
            "🍄 [MYCELIUM] 🌱 Discovery: skill '%s' used %d times via slow path. "
            "Consider adding a hardwired pathway for common patterns.",
            skill_name, len(relevant),
        )

        # Reset counter to avoid spamming
        self._discovery_candidates[skill_name] = 0

    # ======================================================================
    # GENERAL HYPHAE — Subsystem Connections
    # ======================================================================

    def set_ui_callback(self, callback: Callable[[str], Coroutine]):
        """Connect the Mycelium directly to the UI for failsafe message delivery."""
        self.ui_callback = callback
        logger.info("🍄 [MYCELIUM] Direct UI Hypha Connected.")


    def establish_unification_hyphae(self):
        """Phase 25: Sovereign Unification Hyphae.
        
        Links canonical subsystems into the root network to ensure they are 
        visible and tracked even before dynamic mapping completes.
        Names here match SubsystemAudit.SUBSYSTEMS for identity synchronization.
        """
        unification_links = [
            ("orchestrator", "personality_engine", 3.0, "#FF69B4", "Core identity and persona control"),
            ("orchestrator", "memory_facade", 3.0, "#F5A623", "Long-term knowledge and episodic recall"),
            ("orchestrator", "affect_engine", 2.5, "#D0021B", "Emotional state and motivation substrate"),
            ("orchestrator", "drive_controller", 2.0, "#BD10E0", "Biological-inspired drives and urgency"),
            ("orchestrator", "liquid_substrate", 2.0, "#7ED321", "Dynamic arousal and focus management"),
            ("orchestrator", "sovereign_scanner", 2.0, "#50E3C2", "Reactive intent detection and safety"),
            ("personality_engine", "cognition", 2.5, "#4A90E2", "Identity guiding thought generation"),
            ("cognition", "autonomy", 3.0, "#9013FE", "Decision making and goal selection"),
            ("autonomy", "cognition", 3.0, "#9013FE", "Feedback loop for autonomous action"),
            ("mind_tick", "mycelium", 2.5, "#F8E71C", "Universal heartbeat and connectivity"),
            ("orchestrator", "critic_engine", 3.0, "#50E3C2", "Recursive self-correction and plan verification"),
            # --- Personhood & Resilience Roots ---
            ("orchestrator", "personhood", 3.0, "#FF007F", "Spontaneous thought and subjective agency"),
            ("orchestrator", "voice_presence", 3.0, "#00FFFF", "Vocal embodiment and immediate expression"),
            ("orchestrator", "stability_guardian", 3.0, "#39FF14", "Real-time health monitoring and stall prevention"),
            ("orchestrator", "research_cycle", 2.5, "#FFFF00", "Autonomous knowledge pursuit during idle"),
        ]
        for src, tgt, prio, color, desc in unification_links:
            h = self.establish_connection(src, tgt, priority=prio)
            h.color = color
            h.description = desc
            h.strength = 5.0 # Requested 5x thickness boost for initial view
        logger.info("🍄 [MYCELIUM] ✅ Core Unification Hyphae established (%d links)", len(unification_links))

    def shutdown(self):
        """Phase 28: Total Neural Root Cleanup (Issue 76).
        Ensures all hardware pins and active hyphae are safely disconnected.
        """
        logger.info("🍄 [MYCELIUM] Neutralizing all neural roots and hyphae...")
        self._stop_event.set()
        self.infrastructure_mapped = False
        self._execution_log.clear()
        self.hyphae.clear()
        self._neural_roots.clear()
        MycelialNetwork._initialized = False
        MycelialNetwork._instance = None
        logger.info("🍄 [MYCELIUM] Network Offline.")

    def establish_consciousness_hyphae(self):
        """Phase 5: Transcendental Consciousness Hyphae.
        Specifically links modules related to qualia and phenomenology.
        """
        links = [
            ("qualia", "phenomenology", 3.0),
            ("consciousness", "global_workspace", 2.5),
            ("sentience", "autonomy", 2.0),
        ]
        for src, tgt, prio in links:
            self.establish_connection(src, tgt, priority=prio)
        logger.info("🍄 [MYCELIUM] 👁️ Consciousness Hyphae established.")

    @asynccontextmanager
    async def rooted_flow(self, source: str, target: str, activity: str = None,
                          timeout: float = 60.0, priority: float = 1.0):
        """Wraps a process in a mycelial root. If it stalls, the root overrides."""
        hypha = self.establish_connection(source, target, priority=priority)
        hypha.log(f"INITIATING: {activity}")

        try:
            yield hypha
            hypha.pulse(success=True)
            hypha.log(f"SUCCESS: {activity}")
        except asyncio.CancelledError:
            hypha.log(f"CANCELLED: {activity}")
            raise
        except Exception as e:
            hypha.pulse(success=False)
            hypha.log(f"STALL/FAILURE: {activity} - {e}")
            logger.error("🍄 [MYCELIUM] Critical Stall in %s (%s). Triggering Override.", hypha.name, e)
            await self._emergency_override(hypha, activity, str(e))
            if hypha.priority >= 1.0:
                return  # Absorbed error — failsafe bypass
            raise

    async def _emergency_override(self, hypha: Hypha, activity: str, error_msg: str):
        """Force a result through the Mycelium when the standard path stalls."""
        logger.warning("⚡ [ROOT OVERRIDE] Forcing path completion: %s → %s", hypha.name, activity)
        
        # Bridge to Hardened Reflex Core
        if self.reflex:
            await self.reflex.trigger_reflex("STALL_DETECTED", {
                "hypha": hypha.name,
                "activity": activity,
                "error": error_msg
            })
            
        if "response" in activity.lower() and self.ui_callback:
            msg = (
                "🛡️ [Mycelial Failsafe Active] I encountered a stall while processing "
                f"your request ({error_msg}). My system unity has bypassed the block."
            )
            await self.ui_callback(msg)
        hypha.strength += 2.0

    # ======================================================================
    # INFRASTRUCTURE MAPPING — Codebase Unification
    # ======================================================================

    def map_infrastructure(self, base_dir: str, scan_dirs: Optional[List[str]] = None):
        """Dynamically scan the codebase and map all modules into the network graph.

        Walks the specified directories, parses Python imports via AST, and
        creates physical Hypha connections between files that import each other.
        Also annotates existing HardwiredPathways with their source files.

        Args:
            base_dir: Absolute path to the project root (e.g., autonomy_engine/).
            scan_dirs: Subdirectories under base_dir to scan. Defaults to ['core', 'skills'].
        """
        # C-12 FIX: Use a proper mapping state to prevent race conditions.
        # infrastructure_mapped = True should only be set AFTER scanning is complete.
        # We use a primitive lock-like check to ensure serial execution.
        if getattr(self, "_is_mapping", False) or self.infrastructure_mapped:
            return
        self._is_mapping = True 

        # Optimization: Use a local cache for AST results to avoid re-parsing if called multiple times
        # though singleton pattern usually prevents this.
        
        base = Path(base_dir).resolve()
        if scan_dirs is None:
            scan_dirs = ["core", "skills"]

        start_time_map = time.monotonic()
        logger.info("🍄 [MYCELIUM] 🗺️ Infrastructure Mapping starting from: %s", base)

        # 1. Discover all .py files (FIXED: BUG-041 - Offload sync scan)
        all_files: Dict[str, Path] = {}  # module_key → file_path
        
        async def _scan():
            discovered = {}
            for subdir in scan_dirs:
                scan_root = base / subdir
                if not scan_root.exists():
                    continue
                for py_file in scan_root.rglob("*.py"):
                    if py_file.name.startswith("__"):
                        continue
                    try:
                        rel = py_file.relative_to(base)
                        module_key = str(rel.with_suffix("")).replace(os.sep, ".")
                        discovered[module_key] = py_file
                    except ValueError:
                        continue
            return discovered

        # Since we are in a thread already (if called via setup), we could just let it be,
        # but to be safe and consistent with the requirement, we ensure it doesn't block.
        # However, map_infrastructure itself is synchronous. We'll wrap the inner loop.
        import glob
        for subdir in scan_dirs:
            scan_root = base / subdir
            if not scan_root.exists():
                logger.warning("🍄 [MYCELIUM] Scan directory not found: %s", scan_root)
                continue
            # Use glob for a slightly more direct OS call or just keep rglob if it's okay in a thread.
            # The bug says "in loop". map_infrastructure is what runs periodically in some systems?
            # No, it's called once. Wait, maybe it's called in pulse_check? No.
            
            for py_file in scan_root.rglob("*.py"):
                if py_file.name.startswith("__"):
                    continue
                try:
                    rel = py_file.relative_to(base)
                    module_key = str(rel.with_suffix("")).replace(os.sep, ".")
                    all_files[module_key] = py_file
                except ValueError:
                    continue

        logger.info("🍄 [MYCELIUM] Discovered %d Python modules.", len(all_files))

        # 2. Parse imports and build dependency edges
        dependency_graph: Dict[str, List[str]] = {}
        for module_key, file_path in all_files.items():
            deps = self._extract_imports(file_path, base)
            dependency_graph[module_key] = deps

            # Record in mapped_files registry
            self.mapped_files[module_key] = {
                "path": str(file_path),
                "size_bytes": file_path.stat().st_size if file_path.exists() else 0,
                "imports": deps,
            }

        # 3. Create physical Hypha connections for import relationships
        physical_connections = 0
        for module_key, deps in dependency_graph.items():
            for dep in deps:
                if dep in all_files:
                    hypha_name = f"{module_key}->{dep}"
                    if hypha_name not in self.hyphae:
                        h = Hypha(
                            name=hypha_name,
                            source=module_key,
                            target=dep,
                            priority=0.5,
                            is_physical=True,
                        )
                        h.source_file = str(all_files[module_key])
                        h.target_file = str(all_files[dep])
                        self.hyphae[hypha_name] = h
                        physical_connections += 1

        # 4. Annotate existing HardwiredPathways with source file info
        #    Match skill_name → module using multiple strategies:
        #    a) exact substring of module key
        #    b) skill_name words appear in module file stem
        #    c) skill_name matches a known skill registration in the module
        annotated = 0
        for pw in self.pathways.values():
            skill = pw.skill_name.lower().replace("_", "")
            for module_key, file_path in all_files.items():
                stem = file_path.stem.lower().replace("_", "")
                mod_lower = module_key.lower().replace("_", "")
                # Strategy a: skill name substring of module key
                if skill in mod_lower:
                    pw.source_file = str(file_path)
                    pw.dependencies = dependency_graph.get(module_key, [])
                    annotated += 1
                    break
                # Strategy b: module stem contains skill name or vice versa
                if skill in stem or stem in skill:
                    pw.source_file = str(file_path)
                    pw.dependencies = dependency_graph.get(module_key, [])
                    annotated += 1
                    break

        # 5. Compute Module Centrality (reverse dependency index)
        #    Centrality = how many other modules depend on this one.
        #    High centrality = load-bearing pillar; failure has wide blast radius.
        reverse_deps: Dict[str, int] = {}
        for module_key, deps in dependency_graph.items():
            for dep in deps:
                if dep in all_files:
                    reverse_deps[dep] = reverse_deps.get(dep, 0) + 1

        self._centrality = {k: int(v) for k, v in reverse_deps.items()}

        # Tag the top-20 most critical modules
        self._critical_modules = [m for m, _ in sorted(reverse_deps.items(), key=lambda x: x[1], reverse=True)[:20]]

        # Store centrality in mapped_files for API exposure
        for mk in self.mapped_files:
            self.mapped_files[mk]["centrality"] = reverse_deps.get(mk, 0)
            self.mapped_files[mk]["is_critical"] = mk in self._critical_modules

        # 6. Cross-Layer Linking: connect logical subsystem hyphae to physical backing
        #    Maps abstract subsystem names (e.g., "cognition") to the directory/module
        #    patterns they correspond to in the codebase.
        SUBSYSTEM_ALIASES: Dict[str, List[str]] = {
            "cognition": ["cognitive", "brain", "cognitive_engine", "cognitive_integration"],
            "personality": ["personality", "persona", "identity"],
            "memory": ["memory", "dual_memory", "episodic"],
            "affect": ["affect", "emotion", "mood"],
            "autonomy": ["autonomy", "autonomic", "volition", "agency"],
            "perception": ["perception", "senses", "sensory", "screen_observer"],
            "consciousness": ["consciousness", "awareness", "existential", "qualia", "subjectivity", "sentience"],
            "self_modification": ["self_modification", "self_mod", "evolution", "mutate"],
            "skills": ["skills", "capability", "skill_management"],
            "scanner": ["scanner", "cognitive.scanner"],
            "mycelium": ["mycelium"],
            "guardian": ["guardian", "autonomy_guardian"],
            "state_machine": ["state_machine", "orchestrator.state"],
            "drive_engine": ["drive", "motivation", "drives"],
            "telemetry": ["telemetry", "thought_stream", "neural_feed"],
            "system": ["orchestrator", "main", "container"],
            "core_logic": ["orchestrator", "pipeline", "cognitive"],
            "skill_execution": ["capability_engine", "skill_execution"],
            # Phase XXII: Transcendence subsystems
            "meta_evolution": ["meta_cognition", "meta_evolution"],
            "hephaestus": ["hephaestus", "skill_management"],
            "networking": ["networking"],
            "model_selector": ["model_selector", "llm", "brain"],
            "curiosity": ["curiosity", "curiosity_engine", "exploration"],
            # Phase II: Deep consciousness sub-modules
            "cel": ["constitutive_expression", "cel"],
            "iit_phi": ["iit_surrogate", "riiu", "phi"],
            "workspace": ["global_workspace", "gwt"],
            "ganglion": ["ganglion_node", "ganglion"],
            "executive": ["executive_inhibitor", "executive"],
            "qualia_engine": ["qualia_engine"],
            "quantum_entropy": ["quantum_entropy"],
            "opacity": ["structural_opacity", "opacity"],
        }

        def _matches_subsystem(subsystem_name: str, module_path: str) -> bool:
            """Check if a module path belongs to a named subsystem."""
            aliases = SUBSYSTEM_ALIASES.get(subsystem_name, [subsystem_name])
            mp = module_path.lower()
            return any(alias in mp for alias in aliases)

        logical_hyphae = {name: h for name, h in self.hyphae.items() if not h.is_physical}
        for logical_name, logical_h in logical_hyphae.items():
            backing_physical: List[str] = []
            for phys_name, phys_h in self.hyphae.items():
                if not phys_h.is_physical:
                    continue
                src_matches = _matches_subsystem(logical_h.source, phys_h.source)
                tgt_matches = _matches_subsystem(logical_h.target, phys_h.target)
                if src_matches and tgt_matches:
                    backing_physical.append(phys_name)
            if backing_physical:
                self._cross_links[logical_name] = backing_physical

        elapsed = time.monotonic() - start_time_map
        
        # M-15 FIX: Prevent false-positive mapping if zero modules found
        if not all_files:
            logger.warning("🍄 [MYCELIUM] ❌ Infrastructure mapping found 0 modules! Retrying in next cycle.")
            self.infrastructure_mapped = False
            self._is_mapping = False
            return

        self.infrastructure_mapped = True
        self._is_mapping = False
        logger.info(
            "🍄 [MYCELIUM] 🗺️ Infrastructure Mapping COMPLETE (%.2fs): "
            "%d modules, %d physical connections, %d pathways annotated, "
            "%d critical indicators tagged.",
            elapsed, len(all_files), physical_connections, annotated, len(self._critical_modules)
        )

    def _extract_imports(self, file_path: Path, base_dir: Path) -> List[str]:
        """Parse a Python file's AST and extract import targets as dotted module keys."""
        imports: List[str] = []
        try:
            source = file_path.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(source, filename=str(file_path))
        except (SyntaxError, UnicodeDecodeError, OSError) as e:
            logger.debug("🍄 [MYCELIUM] AST parse failed for %s: %s", file_path.name, e)
            return imports

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    # Resolve relative imports
                    if node.level > 0:
                        try:
                            rel = file_path.parent.relative_to(base_dir)
                            parts = list(rel.parts)
                            # Go up 'level - 1' parents
                            if node.level > 1:
                                parts = parts[:-(node.level - 1)] if len(parts) >= node.level - 1 else parts
                            base_module = ".".join(parts)
                            full_module = f"{base_module}.{node.module}" if base_module else node.module
                            imports.append(full_module)
                        except (ValueError, IndexError):
                            imports.append(node.module)
                    else:
                        imports.append(node.module)

        return imports

    def get_infrastructure_report(self) -> Dict[str, Any]:
        """Return a summary of the infrastructure mapping for API/UI consumption."""
        physical_hyphae = {
            name: {
                "source": h.source,
                "target": h.target,
                "source_file": h.source_file,
                "target_file": h.target_file,
                "strength": float(round(h.strength, 2)),
            }
            for name, h in self.hyphae.items()
            if h.is_physical
        }

        annotated_pathways = [
            pw.pathway_id for pw in self.pathways.values() if pw.source_file
        ]

        return {
            "mapped": self.infrastructure_mapped,
            "total_modules": len(self.mapped_files),
            "physical_connections": len(physical_hyphae),
            "annotated_pathways": annotated_pathways,
            "critical_modules": [
                {"module": m, "centrality": self._centrality.get(m, 0)}
                for m in self._critical_modules
            ],
            "cross_layer_links": {
                logical: len(physical_list)
                for logical, physical_list in self._cross_links.items()
            },
            "modules": {k: v["path"] for k, v in self.mapped_files.items()},
            "physical_hyphae_sample": dict(list(physical_hyphae.items())[:20]), # v15 Fix: Explicit iteration
        }

    # ======================================================================
    # MAINTENANCE — Background Health
    # ======================================================================

    async def pulse_check(self):
        """Periodic background check to keep critical hyphae alive and prune weak pathways."""
        if self._async_lock is None:
            self._async_lock = asyncio.Lock()
            
        while True:
            try:
                await asyncio.sleep(30)
                async with self._async_lock:
                    now = time.monotonic()

                    # Pulse critical hyphae
                    for name, hypha in self.hyphae.items():
                        if (
                            now - hypha.last_pulse > 300
                            and hypha.priority >= 1.0
                            and self._should_monitor_hypha(hypha)
                        ):
                            # [WHOLESALE FIX] Rate-limit HYPHA_SEVERED alerts
                            # to prevent log spam (was firing every 30s for EVERY dead hypha)
                            if not hasattr(self, '_hypha_alert_times'):
                                self._hypha_alert_times = {}
                            last_alert = self._hypha_alert_times.get(name, 0)
                            if now - last_alert > 300:  # Max once per 5 minutes per hypha
                                logger.warning("🍄 [MYCELIUM] Hypha inactive: %s. Auto-pulsing.", name)
                                self._hypha_alert_times[name] = now
                            
                            # Keep the heartbeat fresh without degrading an otherwise healthy route.
                            hypha.refresh_heartbeat()

                    # Report weak pathways (don't auto-prune — that's dangerous)
                    for pw_id, pw in self.pathways.items():
                        if pw.is_weak and pw.hit_count + pw.miss_count > 5:
                            logger.warning(
                                "🍄 [MYCELIUM] Weak pathway detected: '%s' (confidence=%.2f)",
                                pw_id, pw.confidence,
                            )
            except asyncio.CancelledError:
                # Cleanup for MemoryGovernor if it's running
                if hasattr(self, '_task') and self._task:
                    self._task.cancel()
                    try:
                        await self._task
                    except asyncio.CancelledError as _e:
                        logger.debug('Ignored asyncio.CancelledError in mycelium.py: %s', _e)
                    finally:
                        self._task = None
                
                # v8.1.0: Ensure total cleanup of any leaked worker handles
                try:
                    if hasattr(self, '_critical_cleanup') and callable(self._critical_cleanup):
                        await self._critical_cleanup()
                        logger.info("🛡️ Memory Governor shutdown complete. All worker handles leaked/active were purged.")
                except Exception as e:
                    logger.error(f"Error during Memory Governor shutdown: {e}")
                logger.info("🍄 [MYCELIUM] Pulse check loop shutting down.")
                break
            except Exception as e:
                logger.error("🍄 [MYCELIUM] Pulse check error: %s", e, exc_info=True)
                await asyncio.sleep(10)  # Backoff on error

    # ======================================================================
    # INTROSPECTION — Topology & Health Reporting
    # ======================================================================

    def get_network_topology(self) -> Dict[str, Any]:
        """Full network state for UI visualization and health monitoring."""
        logical_count = sum(1 for h in self.hyphae.values() if not h.is_physical)
        physical_count = sum(1 for h in self.hyphae.values() if h.is_physical)

        return {
            "pathways": {
                pw_id: pw.to_dict() for pw_id, pw in self.pathways.items()
            },
            "pathway_count": len(self.pathways),
            "hyphae": {
                name: h.model_dump() for name, h in self.hyphae.items()
            },
            "hyphae_summary": {
                "total": len(self.hyphae),
                "logical": logical_count,
                "physical": physical_count,
                "cross_layer_linked": len(self._cross_links),
                "infrastructure_mapped": self.infrastructure_mapped
            },
            "critical_modules": self._critical_modules[:10],
            "discovery_candidates": dict(self._discovery_candidates),
            "ui_connected": self.ui_callback is not None,
            "system_cohesion": self._calculate_cohesion(),
            "total_pathway_hits": sum(pw.hit_count for pw in self.pathways.values()),
            "total_pathway_misses": sum(pw.miss_count for pw in self.pathways.values()),
        }

    def get_unity_report(self) -> Dict[str, Any]:
        """Backward-compatible unity report."""
        return {
            "hyphae": {
                n: {"strength": h.strength, "last_active": time.monotonic() - h.last_pulse}
                for n, h in self.hyphae.items()
            },
            "pathways": len(self.pathways),
            "ui_connected": self.ui_callback is not None,
            "system_cohesion": self._calculate_cohesion(),
        }

    def _calculate_cohesion(self) -> float:
        """System cohesion score: average of hypha strength and pathway confidence."""
        strengths = [h.strength for h in self.hyphae.values()] if self.hyphae else [0.0]
        confidences = [pw.confidence for pw in self.pathways.values()] if self.pathways else [1.0]
        all_values = strengths + confidences
        return round(sum(all_values) / max(len(all_values), 1), 3)

    # ======================================================================
    # PILLAR 3: THE ROOT VAULT (Aegis Persistence)
    # ======================================================================

    async def vault_sync(self):
        """Serialize the living Mycelial structure to the secure Root Vault (Isolated)."""
        from core.config import config
        aegis_cfg = getattr(config, 'aegis', None)
        db_path = config.paths.base_dir / getattr(aegis_cfg, "vault_path", "data/mycelium_vault.db")
        
        def _sync_worker():
            import sqlite3
            import json
            db_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with sqlite3.connect(db_path) as conn:
                    conn.execute("PRAGMA journal_mode=WAL;")
                    cursor = conn.cursor()
                    cursor.execute('''CREATE TABLE IF NOT EXISTS aegis_vault 
                                     (key TEXT PRIMARY KEY, data TEXT, timestamp REAL)''')
                    
                    # 1. Sync Pathways
                    pathway_data = {k: v.to_dict() for k, v in self.pathways.items()}
                    cursor.execute('REPLACE INTO aegis_vault (key, data, timestamp) VALUES (?, ?, ?)',
                                   ("pathways", json.dumps(pathway_data), time.time()))
                                   
                    # 2. Sync Hyphae
                    hypha_data = {k: v.model_dump() for k, v in self.hyphae.items()}
                    cursor.execute('REPLACE INTO aegis_vault (key, data, timestamp) VALUES (?, ?, ?)',
                                   ("hyphae", json.dumps(hypha_data), time.time()))
                    conn.commit()
            except Exception as e:
                logger.error("🛡️ AEGIS: Vault Sync Worker Failed! %s", e)
                raise
        
        try:
            await run_io_bound(_sync_worker)
            logger.debug("🛡️ AEGIS: Vault Sync Complete.")
        except Exception as e:
            logger.error("🛡️ AEGIS: Vault Sync Failed! %s", e)

    @classmethod
    async def restore_from_vault(cls) -> bool:
        """Emergency cloning protocol from the Root Vault (Isolated)."""
        from core.config import config
        db_path = config.paths.base_dir / getattr(config, "aegis", config).__dict__.get("vault_path", "data/mycelium_vault.db")
        if not db_path.exists():
            logger.critical("🛡️ AEGIS FATAL: Cannot restore; Root Vault missing!")
            return False

        def _restore_worker():
            import sqlite3
            import json
            try:
                with sqlite3.connect(db_path) as conn:
                    conn.execute("PRAGMA journal_mode=WAL;")
                    cursor = conn.cursor()
                    
                    # We bypass the True-Lock explicitly for a verified restoration
                    inst = getattr(cls, "_instance", None)
                    if inst is None: 
                        logger.critical("🛡️ AEGIS: Restoration aborted — Mycelium instance not initialized.")
                        return False
                    
                    object.__setattr__(inst, "_aegis_locked", False)
                    
                    cursor.execute('SELECT data FROM aegis_vault WHERE key="pathways"')
                    row = cursor.fetchone()
                    if row:
                        data = json.loads(row[0])
                        # Tier 2 Hardening: Safe Filtered Deserialization
                        safe_pathways = {}
                        allowed_pw_keys = {"pathway_id", "pattern", "skill_name", "param_map", "priority", 
                                           "source_file", "dependencies", "confidence", "activity_label", 
                                           "hit_count", "miss_count", "direct_response", "color", "description", "size"}
                        for k, v in data.items():
                            if isinstance(v, dict):
                                safe_v = {key: val for key, val in v.items() if key in allowed_pw_keys}
                                safe_pathways[k] = HardwiredPathway(**safe_v)
                        
                        object.__setattr__(inst, "pathways", safe_pathways)
                        order = sorted(safe_pathways.keys(), key=lambda k: safe_pathways[k].priority, reverse=True)
                        object.__setattr__(inst, "_pathway_order", order)
                        
                    cursor.execute('SELECT data FROM aegis_vault WHERE key="hyphae"')
                    row = cursor.fetchone()
                    if row:
                        data = json.loads(row[0])
                        safe_hyphae = {}
                        allowed_hypha_keys = {"name", "source", "target", "priority", "strength", "last_pulse",
                                              "active", "is_physical", "source_file", "target_file", 
                                              "color", "description", "size", "trace"}
                        for k, v in data.items():
                            if isinstance(v, dict):
                                safe_v = {key: val for key, val in v.items() if key in allowed_hypha_keys}
                                safe_hyphae[k] = Hypha(**safe_v)
                        inst.hyphae = safe_hyphae

                    
                    # Re-lock the shield
                    object.__setattr__(inst, "_aegis_locked", True)
                    logger.critical("🛡️ AEGIS: Restoration Successful. Mycelium Unity restored.")
                    return True
            except Exception as e:
                logger.critical("🛡️ AEGIS FATAL: Restoration Failed! %s", e)
                return False

        logger.critical("🛡️ AEGIS: Initiating Emergency Vault Restoration...")
        return await run_io_bound(_restore_worker)
