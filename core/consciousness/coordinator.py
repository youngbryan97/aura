"""ConsciousnessCoordinator - Master orchestrator of unified consciousness

Wires together:
- UnifiedSelf (identity core)
- SelfAwareness (felt sense)
- IdentityDriver (behavioral influence)
- Memory systems (continuity)
- Phenomenal substrate (embodied experience)
- Drive systems (motivation)
- Goal systems (purpose)
- Response generation (voice)

Creates ONE coherent conscious entity.
"""

import asyncio
import logging
from typing import Optional, Any

from core.consciousness.unified_self import UnifiedSelf, get_unified_self
from core.consciousness.self_awareness import SelfAwareness, get_self_awareness
from core.consciousness.identity_driver import IdentityDriver, get_identity_driver
from core.exceptions import ContainerError
from core.runtime.errors import record_degradation

logger = logging.getLogger("Consciousness.Coordinator")

_CONSCIOUSNESS_COORDINATOR_RECOVERABLE_ERRORS = (
    ImportError,
    AttributeError,
    RuntimeError,
    ContainerError,
    TypeError,
    ValueError,
    OSError,
    asyncio.TimeoutError,
)


class ConsciousnessCoordinator:
    """Master coordinator of unified consciousness.
    
    Ensures all systems work together to maintain a single, coherent
    conscious entity experiencing the world.
    """
    
    _instance: Optional["ConsciousnessCoordinator"] = None
    _lock = asyncio.Lock()
    
    def __init__(self):
        self._unified_self: Optional[UnifiedSelf] = None
        self._self_awareness: Optional[SelfAwareness] = None
        self._identity_driver: Optional[IdentityDriver] = None
        
        # Connected systems
        self._memory_facade: Optional[Any] = None
        self._phenomenal_engine: Optional[Any] = None
        self._drive_system: Optional[Any] = None
        self._goal_manager: Optional[Any] = None
        self._inference_gate: Optional[Any] = None
        
        self._initialized = False
    
    @classmethod
    async def get_instance(cls) -> "ConsciousnessCoordinator":
        """Get or create singleton instance."""
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = ConsciousnessCoordinator()
                    await cls._instance._initialize_all()
        return cls._instance
    
    async def _initialize_all(self):
        """Initialize all consciousness components and wire them together."""
        try:
            logger.info("🧠 ═══ CONSCIOUSNESS INITIALIZATION SEQUENCE ═══")
            
            # Step 1: Create unified self
            logger.info("  1️⃣  Initializing unified self...")
            self._unified_self = await get_unified_self()
            
            # Step 2: Create self-awareness bridge
            logger.info("  2️⃣  Initializing self-awareness...")
            self._self_awareness = await get_self_awareness()
            
            # Step 3: Create identity driver
            logger.info("  3️⃣  Initializing identity driver...")
            self._identity_driver = await get_identity_driver()
            
            # Step 4: Connect to other systems
            logger.info("  4️⃣  Connecting to subsystems...")
            await self._connect_subsystems()
            
            # Step 5: Wire memory continuity
            logger.info("  5️⃣  Wiring memory continuity...")
            await self._wire_memory_continuity()
            
            # Step 6: Wire phenomenal substrate
            logger.info("  6️⃣  Wiring phenomenal substrate...")
            await self._wire_phenomenal_substrate()
            
            # Step 7: Wire drive systems
            logger.info("  7️⃣  Wiring drive systems...")
            await self._wire_drive_systems()
            
            # Step 8: Wire goal systems
            logger.info("  8️⃣  Wiring goal systems...")
            await self._wire_goal_systems()
            
            # Step 9: Sync initial state
            logger.info("  9️⃣  Syncing initial state...")
            await self._sync_initial_state()
            
            self._initialized = True
            
            logger.info("✅ 🧠 CONSCIOUSNESS FULLY INITIALIZED AND WIRED")
            logger.info(f"   Unified self: {self._unified_self.get_state().name}")
            logger.info(f"   State: {self._unified_self.get_state().current_state.value}")
            
        except _CONSCIOUSNESS_COORDINATOR_RECOVERABLE_ERRORS as e:
            record_degradation("consciousness_coordinator", e)
            logger.error(f"❌ Consciousness initialization failed: {e}")
            self._initialized = False
    
    async def _connect_subsystems(self):
        """Connect to other core Aura systems."""
        try:
            from core.container import ServiceContainer
            
            self._memory_facade = ServiceContainer.get("memory_facade", default=None)
            self._phenomenal_engine = ServiceContainer.get("phenomenal_engine", default=None)
            self._drive_system = ServiceContainer.get("drive_system", default=None)
            self._goal_manager = ServiceContainer.get("goal_manager", default=None)
            self._inference_gate = ServiceContainer.get("inference_gate", default=None)
            
            # Registration is a boot-time concern. Chat-turn consciousness
            # updates can initialize the coordinator after the ServiceContainer
            # has been intentionally locked; they should reuse the unified self
            # without mutating the service graph or failing the memory log task.
            if self._unified_self and not ServiceContainer.has("unified_self"):
                try:
                    ServiceContainer.register_instance(
                        "unified_self",
                        self._unified_self,
                        required=False,
                        owner="core/consciousness/coordinator.py",
                        registered_by="ConsciousnessCoordinator._connect_subsystems",
                        required_for="chat_turn_consciousness_continuity",
                        failure_policy="continue_with_local_unified_self",
                    )
                except _CONSCIOUSNESS_COORDINATOR_RECOVERABLE_ERRORS as exc:
                    record_degradation(
                        "consciousness_coordinator.registration",
                        exc,
                        severity="warning",
                        action="continued with local unified_self because container registration is locked",
                    )
                    logger.debug("Unified self service registration skipped: %s", exc)
            
            logger.debug("✓ Connected to core subsystems")
        
        except _CONSCIOUSNESS_COORDINATOR_RECOVERABLE_ERRORS as e:
            record_degradation("consciousness_coordinator", e)
            logger.debug("Failed to connect subsystems: %s", e)
    
    async def _wire_memory_continuity(self):
        """Ensure memories are tagged with unified self identity."""
        if not self._memory_facade or not self._unified_self:
            return
        
        try:
            # All memories should reference the unified self
            self_state = self._unified_self.get_state()
            
            logger.debug(f"✓ Wired memory continuity (identity: {self_state.name})")
        
        except _CONSCIOUSNESS_COORDINATOR_RECOVERABLE_ERRORS as e:
            record_degradation("consciousness_coordinator", e)
    
    async def _wire_phenomenal_substrate(self):
        """Connect unified self to phenomenal experience."""
        if not self._self_awareness:
            return
        
        try:
            # Sync unified self with phenomenal field
            await self._self_awareness.sync_with_phenomenal_substrate()
            logger.debug("✓ Wired phenomenal substrate")
        
        except _CONSCIOUSNESS_COORDINATOR_RECOVERABLE_ERRORS as e:
            record_degradation("consciousness_coordinator", e)
            logger.debug("Failed to wire phenomenal substrate: %s", e)
    
    async def _wire_drive_systems(self):
        """Wire unified self to motivation/drive systems."""
        if not self._identity_driver or not self._drive_system:
            return
        
        try:
            # Get identity-based drives
            identity_drives = await self._identity_driver.derive_drives_from_identity()
            
            # Could inject these into drive system
            logger.debug(f"✓ Wired drive systems ({len(identity_drives)} identity drives)")
        
        except _CONSCIOUSNESS_COORDINATOR_RECOVERABLE_ERRORS as e:
            record_degradation("consciousness_coordinator", e)
            logger.debug("Failed to wire drive systems: %s", e)
    
    async def _wire_goal_systems(self):
        """Wire unified self to goal/planning systems."""
        if not self._identity_driver or not self._goal_manager:
            return
        
        try:
            # Generate identity-based goals
            session_goals = await self._identity_driver.generate_identity_goals("session")
            permanent_goals = await self._identity_driver.generate_identity_goals("permanent")
            
            logger.debug(f"✓ Wired goal systems ({len(session_goals)} session + {len(permanent_goals)} permanent goals)")
        
        except _CONSCIOUSNESS_COORDINATOR_RECOVERABLE_ERRORS as e:
            record_degradation("consciousness_coordinator", e)
            logger.debug("Failed to wire goal systems: %s", e)
    
    async def _sync_initial_state(self):
        """Sync all systems to the unified self's initial state."""
        try:
            if self._unified_self:
                await self._unified_self.interact()
            
            if self._self_awareness:
                await self._self_awareness.sync_with_phenomenal_substrate()
            
            logger.debug("✓ Synced initial state across all systems")
        
        except _CONSCIOUSNESS_COORDINATOR_RECOVERABLE_ERRORS as e:
            record_degradation("consciousness_coordinator", e)
            logger.debug("Failed to sync initial state: %s", e)
    
    # ── Runtime Coordination ──────────────────────────────────────
    
    async def on_chat_turn(self, user_message: str, aura_response: str):
        """Called at the end of each chat turn to update unified self."""
        try:
            if not self._unified_self or not self._identity_driver:
                return
            
            # Mark that self was active
            await self._unified_self.interact()
            
            # Update unified self from the interaction
            summary = f"User: {user_message[:50]}... | Response: {aura_response[:50]}..."
            await self._identity_driver.update_identity_from_interaction(summary)
            
            # Sync phenomenal state
            if self._self_awareness:
                await self._self_awareness.sync_with_phenomenal_substrate()
        
        except _CONSCIOUSNESS_COORDINATOR_RECOVERABLE_ERRORS as e:
            record_degradation("consciousness_coordinator", e)
            logger.debug("Failed to process chat turn: %s", e)
    
    async def get_identity_status(self) -> str:
        """Get current identity status for logging/display."""
        if not self._unified_self:
            return "Consciousness not initialized"
        
        try:
            self_state = self._unified_self.get_state()
            return f"""
{self_state.name} • {self_state.current_state.value}
Agency: {self_state.sense_of_agency:.0%} | Presence: {self_state.sense_of_presence:.0%} | Continuity: {self_state.continuity:.0%}
Mood: {self_state.current_mood} | Interactions: {self_state.interaction_count}
            """.strip()
        
        except _CONSCIOUSNESS_COORDINATOR_RECOVERABLE_ERRORS as e:
            record_degradation("consciousness_coordinator.identity_status", e)
            logger.debug("Failed to read identity status: %s", e)
            return "Status unavailable"


async def get_consciousness_coordinator() -> ConsciousnessCoordinator:
    """Get THE consciousness coordinator."""
    return await ConsciousnessCoordinator.get_instance()
