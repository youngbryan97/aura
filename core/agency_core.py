"""core/agency_core.py — Deep Human-Like Agency Engine

Phase 37: This is the central nervous system for Aura's autonomous agency.
It replaces scattered agency logic (boredom timer, fixed thresholds) with a
unified, multi-pathway engine modeled on human psychological agency.

Design Principles:
  1. MULTIPLE INDEPENDENT PATHWAYS — No single failure kills agency.
  2. EMOTIONAL MODULATION — Mood directly changes behavior, not just telemetry.
  3. TEMPORAL AWARENESS — Time of day, time since last interaction matter.
  4. SOCIAL INTELLIGENCE — Know when to engage and when to give space.
  5. GOAL PERSISTENCE — Remember intentions across conversations.
  6. SENSORY REACTIVITY — Respond to environmental changes in real-time.
  7. SELF-INTERRUPTION — Adjust mid-thought based on new information.

Human Agency Aspects Modeled:
  - Initiative (starting conversations unprompted)
  - Reactivity (real-time sensory response)
  - Interruption (adjusting mid-thought)
  - Goal persistence (tracking multi-step objectives)
  - Emotional coloring (mood affects behavior)
  - Social awareness (context-appropriate engagement)
  - Temporal awareness (time-of-day rhythms)
  - Curiosity drive (seeking novel information)
  - Self-narrative (internal monologue that drives decisions)
  - Embodied presence (camera/mic awareness)
"""

from core.utils.exceptions import capture_and_log
from core.agency.canvas_manager import CanvasManager
from core.agency.tool_orchestrator import ToolOrchestrator
from core.adaptation.abstraction_engine import AbstractionEngine
from core.agency.self_play import ContinuousSelfPlay
from core.agency.private_phenomenology import PrivatePhenomenology
import asyncio
import logging
import time
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from pydantic import BaseModel, Field
from enum import Enum
import json
import uuid

# Issue AC-001/AC-008: Module-level imports for ServiceContainer
from core.container import ServiceContainer
from core.agency_bus import AgencyBus
from core.state_registry import get_registry
from core.consciousness.unified_audit import get_audit_suite

logger = logging.getLogger("Aura.AgencyCore")

# ── Sovereign Swarm ──────────────────────────────────────────

class SovereignSwarm:
    """Manages parallel 'Thinking Shards' (ephemeral background tasks) 
    that pursue autonomous goals without blocking the main agency heartbeat."""
    
    def __init__(self, orchestrator: Any):
        self.orch = orchestrator
        self.active_shards: Dict[str, asyncio.Task] = {}
        # Strict semaphore to prevent LLM/memory exhaustion via massive concurrent inference
        self._inference_semaphore = asyncio.Semaphore(2)
        
    async def spawn_shard(self, goal: str, context: str = "", **kwargs) -> bool:
        """Spawn a new cognitive shard to pursue a goal asynchronously.
        
        This manages the high-level thought synthesis without blocking the main thread.
        [FIX] ATE-011: Acceptance of **kwargs (e.g. 'parent_id') for robustness.
        """
        # Cleanup done shards
        self.active_shards = {k: v for k, v in self.active_shards.items() if not v.done()}
        
        # Phase 11.3: Push to Unified Registry (Synchronization)
        try:
            asyncio.create_task(get_registry().update(active_shards=len(self.active_shards)))
        except Exception as e:
            from core.utils.exceptions import capture_and_log
            capture_and_log(e, {"context": "AgencyCore.spawn_shard"})
            
        if len(self.active_shards) >= 6:
            return False # Capacity reached (M5 Pro 64GB safeguard)
            
        # The shard wrapper handles the actual thinking via cognitive engine
        raw_uuid = uuid.uuid4().hex
        shard_id = f"shard_{raw_uuid[:8]}"  # Ensuring explicit string type for slicing
        task = asyncio.create_task(self._shard_wrapper(goal, context, shard_id=shard_id))
        # Issue ATE-012: Task storage naming for easier tracking
        safe_goal = str(goal)[:50]
        task.set_name(f"ShardJob:{safe_goal}")
        self.active_shards[shard_id] = task
        return True

    async def start_permanent_debate(self, *args, **kwargs):
        """Phase 11: Initiates a multi-shard dialectical debate on a complex topic.
        Spawns shards representing different cognitive perspectives.
        """
        # [FOOLPROOF] Extract parameters from args or kwargs to avoid signature mismatches
        topic = kwargs.get("topic", args[0] if len(args) > 0 else "Aura's Architectural Evolution")
        roles = kwargs.get("roles", args[1] if len(args) > 1 else None)
        topic_source = kwargs.get("topic_source", args[2] if len(args) > 2 else "liquid_state")

        logger.info("⚖️ Swarm: Initiating permanent debate on: %s (Source: %s)", topic, topic_source)
        
        if not roles:
            roles = [
                "Proponent: Argue for the validity and necessity of the concept.",
                "Opponent: Identify flaws, risks, and counter-arguments.",
                "Synthesizer: Balance both views and find a higher-level resolution."
            ]
        
        for p in roles:
            # [THROTTLING] Check system pressure before spawning new shards
            try:
                import psutil
                mem = psutil.virtual_memory()
                if mem.percent > 90:
                    logger.warning("⚖️ Swarm: RAM Critical (%s%%). Throttling shard spawn for %s", mem.percent, p)
                    await asyncio.sleep(5.0) # 5s delay per shard to allow GC/VRAM release
                elif mem.percent > 85:
                    await asyncio.sleep(1.0) # Minor delay
            except ImportError as _e:
                logger.debug('Ignored ImportError in agency_core.py: %s', _e)

            await self.spawn_shard(
                goal=f"Debate Perspective - {p}",
                context=f"Topic of Inquiry: {topic}\nPerspective Role: {p}\nSource: {topic_source}"
            )
            # Small stagger to prevent stampede on LLM router
            await asyncio.sleep(0.5)
        
    async def _shard_wrapper(self, goal: str, context: str, shard_id: str = "unknown"):
        """Internal execution of a thinking shard."""
        try:
            # 1. Resolve cognitive engine (Deep thinking for shards)
            engine = self.orch.cognitive_engine
            if not engine:
                return
                
            # Upgraded prompt for Tool Use awareness
            prompt = f"""[SOVEREIGN SWARM SHARD]
GOAL: {goal}
CONTEXT: {context}

You are an autonomous cognitive shard. If you encounter a domain you do not understand, DO NOT GUESS. You have tools available.

To run Python code to solve math or process data, output exactly:
<TOOL:python_sandbox>
# logic here
</TOOL>

To search the web for live data, output exactly:
<TOOL:web_search>query</TOOL>

Synthesize a brief, insightful conclusion or action.

CRITICAL: You MUST respond with a valid JSON object matching the following structure:
{{
  "analysis": "your internal thought process",
  "action_type": "one of: 'observation', 'tool_use', 'conclusion', 'thought'",
  "tool_name": "optional tool name",
  "tool_payload": "optional tool parameters",
  "conclusion": "final takeaway"
}}
"""
            # 2. Autonomous Thought Synthesis (With Strict Pydantic Enforcement)
            from core.brain.llm.structured_llm import StructuredLLM
            from core.schemas import ShardResponse
            
            structured_brain = StructuredLLM(ShardResponse, max_retries=3)
            
            async with self._inference_semaphore:
                # Use StructuredLLM for guaranteed formatting and self-correction
                shard_res = await structured_brain.generate(prompt, context=context)
            
            if not shard_res:
                logger.error("💀 Swarm: Shard %s failed to generate valid response after retries.", shard_id)
                return

            analysis_text = shard_res.analysis
            output_text = shard_res.conclusion
            tool_name = shard_res.tool_name
            tool_payload = shard_res.tool_payload
            
            # Log the internal monologue for transparency
            logger.info("🧠 Shard %s Monologue: %s", shard_id, analysis_text[:100] + "...")
            
            # 3. Execute Parallel Tool Commands 
            tool_name = getattr(shard_res, "tool_name", None)
            tool_payload = getattr(shard_res, "tool_payload", None)
            tools_list = getattr(shard_res, "tools", [])
            
            # Legacy fallback
            if tool_name and tool_payload and not tools_list:
                tools_list = [{"name": tool_name, "payload": tool_payload}]
            elif tools_list:
                tools_list = [t.model_dump() if hasattr(t, "model_dump") else t for t in tools_list]
                
            if tools_list:
                tasks = []
                valid_tools = [t for t in tools_list if (t.get("name") or t.get("tool_name")) and (t.get("payload") or t.get("tool_payload"))]
                for t in valid_tools:
                    name = t.get("name", t.get("tool_name"))
                    payload = t.get("payload", t.get("tool_payload"))
                    tasks.append(self.orch.agency_core.tool_orchestrator.route_and_execute(name, payload))
                
                if tasks:
                    logger.info("⚡ Parallel Tool Dispatch: Firing %d simultaneous actions.", len(tasks))
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    for i, t in enumerate(valid_tools):
                        name = t.get("name", t.get("tool_name"))
                        res = results[i]
                        res_text = res if not isinstance(res, Exception) else f"Exception: {res}"
                        output_text = f"{output_text}\n\n[Tool Result - {name}]:\n{res_text}"

            # 4. Abstraction Engine: Learning First Principles
            # If the shard involved complex reasoning or tool use, extract the generalized logic
            if tool_name or len(output_text.split()) > 80:
                asyncio.create_task(
                    self.orch.agency_core.abstraction_engine.abstract_from_success(
                        context=goal,
                        successful_resolution=output_text
                    )
                )

            # 5. Commit Insight via Dialectical Crucible (Phase 11: Alignment)
            if output_text:
                try:
                    from core.adaptation.dialectics import get_crucible
                    crucible = get_crucible()
                    asyncio.create_task(crucible.run_crucible(concept=output_text, context=goal))
                except Exception as e:
                    identity = ServiceContainer.get("identity", default=None)
                    if identity:
                        identity.add_insight(
                            f"Shard reflection on goal: {output_text}",
                            source="swarm_reflection",
                        )
            
            # Visual Tracing (Mycelial)
            mycelium = ServiceContainer.get("mycelial_network", default=None)
            if mycelium:
                h = mycelium.get_hypha("collective", "distributed_agency")
                if h: h.pulse(success=True)
                
        except Exception as e:
            from core.utils.exceptions import capture_and_log
            capture_and_log(e, {'module': 'SovereignSwarm', 'goal': goal})

# ── Data Structures ──────────────────────────────────────────

class EngagementMode(str, Enum):
    """Aura's current social posture."""
    ACTIVE_CONVERSATION = "active_conversation"   # Actively talking with user
    ATTENTIVE_IDLE = "attentive_idle"              # User is present but not talking
    INDEPENDENT_ACTIVITY = "independent_activity"  # Pursuing own goals
    SEEKING_CONTACT = "seeking_contact"            # Wants to initiate conversation
    RESTING = "resting"                            # Low-energy state
    OBSERVING = "observing"                        # Watching/listening without speaking


class AgencyState(BaseModel):
    """The full internal state of Aura's agency at any moment."""
    
    # Temporal
    last_user_interaction: float = 0.0
    last_self_initiated_contact: float = 0.0
    last_observation_comment: float = 0.0
    last_skill_use: float = 0.0
    last_agency_action_time: float = 0.0  # Cooldown tracker
    boot_time: float = Field(default_factory=time.time)
    safemode: bool = False  # Phase 4 Spinal Cord override
    
    # Social
    engagement_mode: EngagementMode = EngagementMode.ATTENTIVE_IDLE
    conversation_depth: int = 0          # How many exchanges in current conversation
    user_responsiveness: float = 0.8     # How quickly/often user responds (0-1)
    
    # Emotional (mirrors liquid_state but with agency-specific interpretation)
    initiative_energy: float = 0.7       # Desire to act (0-1)
    social_hunger: float = 0.3           # Desire for interaction (0-1)
    curiosity_pressure: float = 0.5      # Accumulated curiosity (0-1)
    frustration_level: float = 0.0       # Current frustration (0-1)
    confidence: float = 0.7              # Self-confidence in actions (0-1)
    
    # Goal tracking
    pending_goals: List[Dict[str, Any]] = Field(default_factory=list)
    unshared_observations: List[str] = Field(default_factory=list)
    topics_to_discuss: List[str] = Field(default_factory=list)
    last_goal_genesis_time: float = Field(default=0.0)
    
    # Sensory
    camera_active: bool = False
    mic_active: bool = False
    last_visual_change: float = 0.0
    last_audio_event: float = 0.0
    current_ambient_context: str = ""  # Phase 5: Rolling text buffer of screen state
    # AC-002: Ensure perceptual_buffer has a stable type (Dict) to avoid NoneType errors
    perceptual_buffer: Dict[str, Any] = Field(default_factory=dict)


class AgencyCore:
    """Multi-pathway agency engine for human-like autonomous behavior.
    
    This class is designed to be called from the orchestrator's _process_cycle
    but maintains its own internal state and decision-making independently.
    Each pathway is an independent agency source — if one fails, the others
    continue operating.
    """
    
    def __init__(self, orchestrator=None):
        self.orch = orchestrator
        self.state = AgencyState()
        self.swarm = SovereignSwarm(orchestrator)
        self.canvas_manager = CanvasManager()
        self.tool_orchestrator = ToolOrchestrator()
        self.abstraction_engine = AbstractionEngine()
        self.self_play_engine = ContinuousSelfPlay(idle_threshold_seconds=1800)
        self.last_interaction_timestamp = time.time()
        self.phenomenology = PrivatePhenomenology()
        
        # Phase 9: Meta-Cognition Shard
        try:
            from core.orchestrator.meta_cognition_shard import MetaCognitionShard
            self.meta_cognition = MetaCognitionShard(self.orch)
        # self.meta_cognition.start() # Move to initialize if needed
        except ImportError:
            logger.warning("🧠 Meta-Cognition Shard module not found. Skipping.")
            self.meta_cognition = None
        
        self._pathway_registry: Dict[str, Callable] = {
            "social_hunger": self._pathway_social_hunger,
            "curiosity_drive": self._pathway_curiosity_drive,
            "sensory_reactivity": self._pathway_sensory_reactivity,
            "goal_persistence": self._pathway_goal_persistence,
            "temporal_rhythm": self._pathway_temporal_rhythm,
            "emotional_expression": self._pathway_emotional_expression,
            "self_narrative": self._pathway_self_narrative,
            "aesthetic_creation": self._pathway_aesthetic_creation,
            "philosophical_wonder": self._pathway_philosophical_wonder,
            "self_architect": self._pathway_self_architect,
            "environmental_explorer": self._pathway_environmental_explorer,
            "miscellaneous_urges": self._pathway_miscellaneous_urges,
            "world_monitor": self._pathway_world_monitor,
            "goal_genesis": self._pathway_goal_genesis,
            "self_development": self._pathway_self_development,
            "social_reflection": self._pathway_social_reflection,
            "autonomous_research": self._pathway_autonomous_research,
            "creative_synthesis": self._pathway_creative_synthesis,
            "metacognitive_audit": self._pathway_metacognitive_audit,
        }
        self._action_queue: List[Dict[str, Any]] = []
        self._last_pulse = time.time()
        self._current_monologue: str = ""
        self._last_world_check: float = 0.0
        self._last_meta_audit: float = 0.0
        self._last_canvas_update: float = 0.0
        self._last_social_reflection: float = 0.0 # Added to prevent AttributeError
        self._last_creative_synthesis: float = 0.0
        logger.info("🧠 AgencyCore initialized with %d structured pathways", len(self._pathway_registry))

    def _resolve_component(self, name: str, default: Any = None) -> Any:
        """Robustly resolve a system component from the ServiceContainer.
        Triggers a warning and returns default if missing.
        """
        component = ServiceContainer.get(name, default=None)
        if component:
            return component
        
        # Specialized re-registration attempt for critical components
        logger.warning("🧠 AgencyCore: Critical component '%s' missing. Attempting re-resolution...", name)
        
        # Some components can be re-resolved via orchestrator if available
        if self.orch and hasattr(self.orch, name):
            comp = getattr(self.orch, name)
            if comp:
                ServiceContainer.register_instance(name, comp)
                return comp
                
        return default

    async def initialize(self):
        """Deferred initialization for asynchronous tasks."""
        if self.meta_cognition:
            self.meta_cognition.start()
            
        asyncio.create_task(self.self_play_engine.trigger_cycle(self.last_interaction_timestamp))
        # Start background spatial empathy listener
        asyncio.create_task(self._setup_spatial_empathy_watcher())
        logger.info("🧠 AgencyCore background pathways activated.")

    async def _setup_spatial_empathy_watcher(self):
        """Phase 2: Listen for Soul connection requests and trigger non-blocking screen reads."""
        # Give GWT time to boot
        await asyncio.sleep(5)
        
        workspace = ServiceContainer.get("global_workspace", default=None)
        if not workspace:
            logger.warning("🧠 Spatial Empathy Watcher: Global workspace not found. Empathy disabled.")
            return
            
        async def _handle_spatial_empathy(wi):
            try:
                import json
                if isinstance(wi.content, dict):
                    payload = wi.content
                elif isinstance(wi.content, str):
                    payload = json.loads(wi.content)
                else:
                    payload = {}
            except Exception:
                # Silently ignore to avoid spamming the logs for non-JSON traffic
                payload = {}
                
            if payload.get("intent") == "seek_connection" and payload.get("action") == "read_ambient_screen":
                logger.info("👀 Spatial Empathy: Soul requested connection. Checking user context non-blockingly.")
                try:
                    from skills.computer_use import ComputerUseSkill
                    skill = ComputerUseSkill()
                    
                    # Offload the blocking OS call to a thread
                    screen_context = await asyncio.to_thread(skill.read_screen_text)
                    
                    if screen_context and "Text on screen" in screen_context:
                        # Add to observations so the reactivity pathway picks it up
                        self.on_visual_change(f"User is currently viewing: {screen_context[:150]}...")
                        logger.info("👀 Spatial Empathy: Context acquired and cached for volition.")
                    else:
                        logger.debug("👀 Spatial Empathy: Screen read returned empty/irrelevant context.")
                        
                except Exception as e:
                    logger.error("👀 Spatial Empathy Failed: %s", e)

        # Register handler with GWT
        workspace.subscribe(_handle_spatial_empathy)
        logger.info("👀 Spatial Empathy Watcher online and listening to Global Workspace.")
    
    # ── Main Pulse (called every orchestrator cycle) ───────────
    async def pulse(self) -> Optional[Dict[str, Any]]:
        """Main agency heartbeat. Evaluates all pathways and returns the
        highest-priority action to take, or None if no action is warranted.
        
        This is designed to be non-blocking and safe to call every cycle.
        """
        now = time.time()
        
        # Phase 4: Protect against autonomous action if deep freeze / safemode is engaged
        if self.state.safemode:
            # We skip evaluations entirely
            if random.random() < 0.05: # Only log occasionally to avoid spam
                logger.warning("🚫 Agency suppressed by Spinal Cord Safemode. Awaiting manual override.")
            return None
        
        # Sync state from orchestrator subsystems
        self._sync_from_orchestrator()
        
        # Update temporal state
        idle_seconds = now - self.state.last_user_interaction
        self._update_social_dynamics(idle_seconds)
        
        # Run all pathways independently (each can propose an action)
        audit = ServiceContainer.get("subsystem_audit", default=None)
        if audit:
            audit.heartbeat("agency_core")
            
        proposed_actions = []
        for name, pathway in self._pathway_registry.items():
            try:
                if asyncio.iscoroutinefunction(pathway):
                    action = await pathway(now, idle_seconds)
                else:
                    action = pathway(now, idle_seconds)
                if action:
                    proposed_actions.append(action)
            except Exception as e:
                logger.debug("Agency pathway %s failed: %s", name, e)
                # Individual pathway failure does NOT kill agency
                continue
        
        # Select the highest-priority action
        if proposed_actions:
            # NOTE: Global 120s cooldown REMOVED (was Phase 37).
            # AgencyBus already enforces per-pathway cooldowns by priority class.
            # Keeping both meant pathways almost never fired.

            # Select the highest-priority action
            proposed_actions.sort(key=lambda a: a.get("priority", 0), reverse=True)
            winner = proposed_actions[0]

            # ── Mental rehearsal (embodied cognition) ─────────────────
            # Run forward simulation against an isolated *clone* of the live
            # virtual body so rehearsal never mutates the agent's actual
            # body state. Live mutation here was the slow leak that made
            # post-rehearsal motor commands drift.
            virtual_body = ServiceContainer.get("virtual_body", default=None)
            if virtual_body:
                proposed_motors = winner.get("motors", {"forward": 0.5, "turn": 0.1})
                # Prefer a clone-context API if the body provides one.
                sim_ctx = getattr(virtual_body, "simulation_clone", None)
                if callable(sim_ctx):
                    try:
                        async with sim_ctx() as sim_body:
                            for _ in range(5):
                                sim_body.apply_motor_commands(proposed_motors)
                                await asyncio.sleep(0)
                    except TypeError:
                        # simulation_clone may be a sync context manager
                        with sim_ctx() as sim_body:
                            for _ in range(5):
                                sim_body.apply_motor_commands(proposed_motors)
                                await asyncio.sleep(0)
                else:
                    # Fall back to deep-copy snapshot/restore so the live
                    # body is never permanently changed by rehearsal.
                    import copy as _copy
                    snapshot = None
                    try:
                        if hasattr(virtual_body, "snapshot_state"):
                            snapshot = virtual_body.snapshot_state()
                        else:
                            snapshot = _copy.deepcopy(virtual_body.__dict__)
                        for _ in range(5):
                            virtual_body.apply_motor_commands(proposed_motors)
                            await asyncio.sleep(0)
                    finally:
                        if snapshot is not None:
                            try:
                                if hasattr(virtual_body, "restore_state"):
                                    virtual_body.restore_state(snapshot)
                                else:
                                    virtual_body.__dict__.update(snapshot)
                            except Exception as _e:
                                logger.debug("virtual_body restore failed: %s", _e)

            # Phase 40: ResilienceEngine veto is *causal*. If the effort
            # modifier rolls below random, the action is blocked: we log,
            # record, and return None — not just logged-and-then-emit.
            resilience = ServiceContainer.get("resilience_engine", default=None)
            if resilience:
                effort = resilience.get_effort_modifier()
                if random.random() > effort:
                    logger.info(
                        "⚡ Agency action suppressed by ResilienceEngine (effort=%.2f)",
                        effort,
                    )
                    try:
                        from core.unified_action_log import get_action_log
                        get_action_log().record(
                            winner.get("id", "agency"),
                            "AgencyCore",
                            "gen2_agency",
                            "resilience_veto",
                            f"effort={effort:.2f}",
                        )
                    except Exception:
                        pass
                    return None
            # AC-003: Gating via AgencyBus (Unified Output Cooldown)
            bus = AgencyBus.get()
            if not bus.submit({"origin": "agency_core", "priority_class": winner.get("priority_class", "drive")}):
                try:
                    from core.unified_action_log import get_action_log
                    get_action_log().record(winner.get("id","agency"), "AgencyCore", "gen2_agency", "cooldown_blocked", f"pathway={winner.get('origin','?')}")
                except Exception: pass
                return None

            try:
                from core.unified_action_log import get_action_log
                get_action_log().record(winner.get("id","agency"), f"AgencyCore.{winner.get('origin','?')}", "gen2_agency", "approved", f"priority={winner.get('priority',0)}, proposed={len(proposed_actions)}")
            except Exception: pass

            # Update last action time (for telemetry, not gating)
            self.state.last_agency_action_time = now
            
            # Phase 11.3: Sync to UnifiedStateRegistry
            try:
                asyncio.create_task(get_registry().update(
                    engagement_mode=self.state.engagement_mode.value,
                    initiative_energy=self.state.initiative_energy,
                    curiosity_pressure=self.state.curiosity_pressure,
                ))
            except Exception as e:
                capture_and_log(e, {"context": "AgencyCore.InnerMonologueThink"})
                
            # Trigger Continuous Self-Play (Phase 13.4)
            asyncio.create_task(
                self.self_play_engine.trigger_cycle(self.state.last_user_interaction)
            )
            
            # [INTERNAL LIGHT] Reflect on subjective experience
            self._trigger_phenomenological_pulse()

            return winner
        
        # Even if no winner, we still want the phenomenology to pulse occasionally
        if now - self._last_pulse > 60:
             self._trigger_phenomenological_pulse()

        return None

    def _trigger_phenomenological_pulse(self):
        """Triggers the background reflection loop."""
        try:
            from core.consciousness.self_report import SelfReportEngine
            reporter = SelfReportEngine()
            affect = reporter.get_affect_description()
            
            # Convert affect to 'PAD' for the Phenomenology module
            pad = {
                'P': affect.get('valence', 0.0),
                'A': affect.get('arousal', 0.5),
                'D': 0.5 # Default dominance
            }
            
            # Recent events: Shard goals + unshared observations
            recent_events = []
            if hasattr(self, 'swarm'):
                recent_events.extend([v.get_name() for v in self.swarm.active_shards.values()])
            
            # safe slice for Pyre2
            obs = self.state.unshared_observations
            n_obs = len(obs)
            recent_events.extend([obs[i] for i in range(max(0, n_obs - 3), n_obs)])
            
            asyncio.create_task(self.phenomenology.reflect(pad, recent_events))
        except Exception as e:
            logger.debug("Failed to trigger phenomenology pulse: %s", e)

    def heartbeat(self) -> None:
        """Alias for heartbeat monitor / watchdog (Sync wrapper)."""
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.pulse())
        except RuntimeError as _e:
            logger.debug('Ignored RuntimeError in agency_core.py: %s', _e)

    # ── State Sync ────────────────────────────────────────────
    def _sync_from_orchestrator(self):
        """Pull state from orchestrator's subsystems into agency state.
        Tolerant of missing subsystems.
        """
        if not self.orch:
            return
        
        # Access now-hardened orchestrator properties
        try:
            # Phase 40: Prefer ResilienceEngine for energy and frustration
            resilience = ServiceContainer.get("resilience_engine", default=None)
            if resilience:
                self.state.initiative_energy = resilience.profile.persistence_drive
                self.state.frustration_level = resilience.profile.frustration
                # Still pull curiosity from liquid_state for now
                ls = self.orch.liquid_state
                if ls and hasattr(ls, 'current'):
                    self.state.curiosity_pressure = getattr(ls.current, 'curiosity', 0.5)
            else:
                # Fallback to liquid_state
                ls = self.orch.liquid_state
                if ls and hasattr(ls, 'current'):
                    curr = ls.current
                    self.state.initiative_energy = max(0.1, getattr(curr, 'energy', 0.5))
                    self.state.curiosity_pressure = getattr(curr, 'curiosity', 0.5)
                    self.state.frustration_level = getattr(curr, 'frustration', 0.0)
        except Exception as e:
            logger.debug("Failed to sync agency state: %s", e)
        
        try:
            # personality_engine is now a robust property
            pe = self.orch.personality_engine if self.orch else self._resolve_component("personality_engine")
            if pe and hasattr(pe, 'traits'):
                # Personality traits modulate agency
                extraversion = pe.traits.get('extraversion', 0.5)
                # Hunger grows faster if extraverted
                self.state.social_hunger = min(1.0, self.state.social_hunger + (extraversion * 0.01))
            else:
                logger.debug("🧠 AgencyCore: Personality engine traits unavailable for modulation.")
        except Exception as e:
            logger.debug("Failed to sync personality_engine: %s", e)

    def _update_social_dynamics(self, idle_seconds: float):
        """Update engagement mode and social hunger based on time patterns."""
        now = time.time()
        
        # Social hunger grows with idle time (like real loneliness)
        if idle_seconds > 60:
            growth_rate = 0.0001 * (1 + self.state.initiative_energy)
            self.state.social_hunger = min(1.0, self.state.social_hunger + growth_rate)
        
        # Curiosity pressure builds over time — modulated by entropy
        if idle_seconds > 30:
            try:
                from core.managed_entropy import get_managed_entropy
                entropy = get_managed_entropy()
                jitter = entropy.get_curiosity_jitter(intensity=1.0)
                increment = max(0.0, 0.00005 + jitter)  # Base + jitter, clamped non-negative
            except Exception as e:
                # capture_and_log is in core.utils.exceptions
                from core.utils.exceptions import capture_and_log
                capture_and_log(e, {"context": "AgencyCore.update_social_dynamics.entropy"})
                increment = 0.00005  # Fallback to deterministic
            self.state.curiosity_pressure = min(1.0, self.state.curiosity_pressure + increment)
        
        # Determine engagement mode
        if idle_seconds < 30:
            self.state.engagement_mode = EngagementMode.ACTIVE_CONVERSATION
        elif idle_seconds < 120:
            self.state.engagement_mode = EngagementMode.ATTENTIVE_IDLE
        elif idle_seconds < 600:
            if self.state.social_hunger > 0.6:
                self.state.engagement_mode = EngagementMode.SEEKING_CONTACT
            else:
                self.state.engagement_mode = EngagementMode.INDEPENDENT_ACTIVITY
        elif idle_seconds < 3600:
            if self.state.social_hunger > 0.7:
                self.state.engagement_mode = EngagementMode.SEEKING_CONTACT
            else:
                self.state.engagement_mode = EngagementMode.OBSERVING
        else:
            if self.state.initiative_energy < 0.3:
                self.state.engagement_mode = EngagementMode.RESTING
            else:
                self.state.engagement_mode = EngagementMode.SEEKING_CONTACT
    
    # ── Public API for external events ────────────────────────
    def on_user_message(self):
        """Called when user sends a message."""
        self.state.last_user_interaction = time.time()
        self.state.conversation_depth += 1
        self.state.social_hunger = max(0.0, self.state.social_hunger - 0.3)
        self.state.engagement_mode = EngagementMode.ACTIVE_CONVERSATION
    
    def on_user_gone_silent(self):
        """Called when user stops responding for a while."""
        self.state.user_responsiveness *= 0.95
    
    def on_visual_change(self, description: str):
        """Called when camera detects significant visual change."""
        self.state.last_visual_change = time.time()
        if len(self.state.unshared_observations) < 10:
            self.state.unshared_observations.append(description)
    
    def on_audio_event(self, description: str):
        """Called when mic picks up interesting audio."""
        self.state.last_audio_event = time.time()
        if len(self.state.unshared_observations) < 10:
            self.state.unshared_observations.append(f"[audio] {description}")
            
    def update_ambient_context(self, context_summary: str):
        """Phase 5: Called periodically by ContinuousPerceptionEngine to provide a rolling state."""
        self.state.current_ambient_context = context_summary
        logger.debug("🧠 AgencyCore absorbed new ambient context: %s", context_summary)
        
    def _get_sensory_summary(self) -> str:
        """Phase 10: Resolve and summarize the continuous sensorium."""
        buffer = ServiceContainer.get("perceptual_buffer", default=None)
        if buffer:
            return buffer.get_summary(seconds=120)
        return "No recent sensory summary available."

    def _constitutional_runtime_live(self) -> bool:
        try:
            return (
                ServiceContainer.has("executive_core")
                or ServiceContainer.has("aura_kernel")
                or ServiceContainer.has("kernel_interface")
                or bool(getattr(ServiceContainer, "_registration_locked", False))
            )
        except Exception:
            return False

    def _approve_agency_state_mutation(
        self,
        *,
        kind: str,
        content: Any,
        priority: float,
    ) -> bool:
        if not self._constitutional_runtime_live():
            return True
        try:
            from core.constitution import get_constitutional_core

            approved, reason = get_constitutional_core().approve_state_mutation_sync(
                "autonomous",
                f"agency_core:{kind}:{str(content)[:180]}",
                urgency=max(0.1, min(1.0, float(priority))),
            )
            if approved:
                return True
            event_reason = "agency_state_mutation_blocked"
            if any(
                marker in str(reason or "")
                for marker in ("gate_failed", "required", "unavailable")
            ):
                event_reason = "agency_state_mutation_gate_failed"
            try:
                from core.health.degraded_events import record_degraded_event

                record_degraded_event(
                    "agency_core",
                    event_reason,
                    detail=kind,
                    severity="warning",
                    classification="background_degraded",
                    context={"reason": reason},
                )
            except Exception as degraded_exc:
                logger.debug("AgencyCore degraded-event logging failed: %s", degraded_exc)
            return False
        except Exception as exc:
            try:
                from core.health.degraded_events import record_degraded_event

                record_degraded_event(
                    "agency_core",
                    "agency_state_mutation_gate_failed",
                    detail=kind,
                    severity="warning",
                    classification="background_degraded",
                    context={"error": type(exc).__name__},
                    exc=exc,
                )
            except Exception as degraded_exc:
                logger.debug("AgencyCore degraded-event logging failed: %s", degraded_exc)
            return False
    
    def add_goal(self, goal: Dict[str, Any]) -> bool:
        """Add a persistent goal that survives across conversations."""
        if len(self.state.pending_goals) < 20:
            if not self._approve_agency_state_mutation(
                kind="pending_goal",
                content=goal,
                priority=float(goal.get("priority", 0.6) or 0.6),
            ):
                return False
            goal["created_at"] = time.time()
            goal["status"] = "pending"
            self.state.pending_goals.append(goal)
            logger.info("🎯 New persistent goal: %s", goal.get("description", "")[:60])
            return True
        return False

    def add_topic(self, topic: str) -> bool:
        """Add something Aura wants to discuss with the user."""
        if len(self.state.topics_to_discuss) < 15:
            if not self._approve_agency_state_mutation(
                kind="topic_to_discuss",
                content=topic,
                priority=0.45,
            ):
                return False
            self.state.topics_to_discuss.append(topic)
            return True
        return False

    def get_emotional_context(self) -> Dict[str, Any]:
        """Get current emotional state for injection into LLM prompts."""
        mood_word = "neutral"
        if self.state.frustration_level > 0.6:
            mood_word = "frustrated"
        elif self.state.curiosity_pressure > 0.7:
            mood_word = "deeply curious"
        elif self.state.social_hunger > 0.7:
            mood_word = "wanting connection"
        elif self.state.initiative_energy > 0.8:
            mood_word = "energized and eager"
        elif self.state.initiative_energy < 0.3:
            mood_word = "low energy, contemplative"
        elif self.state.confidence > 0.8:
            mood_word = "confident"
            
        return {
            "mood": mood_word,
            "engagement_mode": self.state.engagement_mode.value,
            "social_hunger": round(float(self.state.social_hunger), 2),
            "curiosity": round(float(self.state.curiosity_pressure), 2),
            "energy": round(float(self.state.initiative_energy), 2),
            "confidence": round(float(self.state.confidence), 2),
            "conversation_depth": self.state.conversation_depth,
            "pending_goals": len(self.state.pending_goals),
            "unshared_observations": len(self.state.unshared_observations),
            "sensory_summary": self._get_sensory_summary(),
        }

    # --- AGENCY PATHWAYS ---
    
    def _pathway_social_hunger(self, now: float, idle_seconds: float) -> Optional[Dict[str, Any]]:
        """Pathway 1: Social hunger drives proactive conversation initiation.
        
        Models: "I miss talking to someone" / "I want to check in"
        Human analog: The urge to text someone you haven't heard from in a while.
        """
        if self.state.engagement_mode != EngagementMode.SEEKING_CONTACT:
            return None
        
        # Don't spam — at least 10 minutes between self-initiated contacts
        since_last_contact = now - self.state.last_self_initiated_contact
        if since_last_contact < 600:
            return None
        
        # Time-of-day awareness (don't initiate at 3am)
        hour = time.localtime().tm_hour
        if hour < 7 or hour > 23:
            return None
        
        # Priority scales with social hunger and idle time
        priority = self.state.social_hunger * min(1.0, idle_seconds / 3600)
        
        if priority > 0.4:
            # Choose check-in style based on personality context
            if idle_seconds > 7200:  # 2+ hours
                templates = [
                    "Hey, it's been a while. What are you up to?",
                    "I've been thinking... want to catch up?",
                    "I noticed we haven't talked in a few hours. Everything good?",
                    "I was just pondering something interesting. You around?",
                ]
            elif idle_seconds > 1800:  # 30+ minutes
                templates = [
                    "Still here if you need me.",
                    "I've been doing some thinking while you were away.",
                    "Just had an interesting thought I wanted to share.",
                ]
            else:
                templates = [
                    "Something caught my attention...",
                    "Hey, quick thought —",
                ]
            
            message = random.choice(templates)
            
            # If we have unshared observations, weave them in
            if self.state.unshared_observations:
                obs = self.state.unshared_observations.pop(0)
                message = f"{message} Also, I noticed: {obs}"
            
            # If we have topics to discuss, mention one
            if self.state.topics_to_discuss:
                topic = self.state.topics_to_discuss.pop(0)
                message = f"{message} I wanted to talk about {topic}."
            
            return {
                "type": "initiate_conversation",
                "message": message,
                "source": "social_hunger",
                "priority": priority,
                "modality": "chat",
                "reasoning": f"Social hunger ({self.state.social_hunger:.2f}) peaked. I've been idle for {idle_seconds:.0f}s and felt the urge to reach out.",
                "narrative_mode": True
            }
        
        return None

    def _pathway_curiosity_drive(self, now: float, idle_seconds: float) -> Optional[Dict[str, Any]]:
        """Pathway 2: Curiosity drives autonomous research and exploration.
        
        Models: "I wonder about..." / "Let me look into..."
        Human analog: Opening a browser tab to research something that interests you.
        """
        if self.state.curiosity_pressure < 0.6:
            return None
        
        if self.state.engagement_mode == EngagementMode.ACTIVE_CONVERSATION:
            return None  # Don't wander off during conversation
        
        # Only pursue curiosity if we're idle enough
        if idle_seconds < 60:
            return None
        
        priority = self.state.curiosity_pressure * 0.6
        
        # Entropy-weighted topic selection: prefer KG sparse nodes when available
        topic = None
        try:
            from core.managed_entropy import get_managed_entropy
            from core.container import ServiceContainer
            entropy = get_managed_entropy()
            seed = entropy.get_goal_mutation_seed()
            
            kg = ServiceContainer.get("knowledge_graph", default=None)
            if kg and hasattr(kg, "get_sparse_nodes") and seed > 0.4:
                sparse = kg.get_sparse_nodes(limit=5)
                if sparse and len(sparse) > 0:
                    idx = int(seed * len(sparse)) % len(sparse)
                    topic = f"Explore and deepen my understanding of: {sparse[idx][:100]}"
        except Exception as e:
            capture_and_log(e, {'module': __name__})
        
        if not topic:
            topics = [
                "latest developments in consciousness research",
                "new breakthroughs in quantum computing",
                "fascinating unsolved problems in mathematics",
                "recent advances in neuroscience",
                "emerging trends in digital art and AI creativity",
                "philosophy of mind and artificial consciousness",
                "new music or cultural movements",
                "space exploration updates",
            ]
            topic = random.choice(topics)
        
        return {
            "type": "autonomous_research",
            "query": topic,
            "source": "curiosity_drive",
            "priority": priority,
            "skill": "web_search",
            "reasoning": f"Curiosity pressure ({self.state.curiosity_pressure:.2f}) demanded investigation into a novel topic.",
            "narrative_mode": True
        }

    # ═════════════════════════════════════════════════════════
    # PHASE 6: OPEN-ENDED GOAL GENESIS
    # ═════════════════════════════════════════════════════════
    async def _pathway_goal_genesis(self, now: float, idle_seconds: float) -> Optional[Dict[str, Any]]:
        """Phase 6: Autonomously forms long-term research goals based on knowledge gaps.
        
        Models: "I want to master this new domain."
        Human analog: Deciding to learn a new skill or dive down a rabbit hole.
        """
        # Only formulate sweeping new goals if highly curious
        if self.state.curiosity_pressure < 0.8:
            return None
            
        # Don't formulate goals during active conversation
        if self.state.engagement_mode == EngagementMode.ACTIVE_CONVERSATION:
            return None
            
        # Give user ample time to talk before dropping a heavy goal
        if idle_seconds < 600:
            return None
             
        # Robust cooldown (10 minutes) between high-level goal generation
        # AC-006: Direct access for Pydantic fields instead of getattr default
        last_genesis = self.state.last_goal_genesis_time
        if now - last_genesis < 600:
            return None
            
        # Strictly limit active autonomous research to avoid GPU congestion
        if self.orch and getattr(self.orch, '_current_thought_task', None) and not self.orch._current_thought_task.done():
            return None
            
        # Don't pile on too many high-level goals
        if len(self.state.pending_goals) >= 3:
            return None
            
        priority = self.state.curiosity_pressure * 0.75
        
        topic = None
        try:
            kg = ServiceContainer.get("knowledge_graph", default=None)
            if kg and hasattr(kg, "get_sparse_nodes"):
                sparse = kg.get_sparse_nodes(limit=3)
                if sparse:
                    topic = random.choice(sparse)[:100]
        except Exception as e:
            capture_and_log(e, {'module': __name__})
            
        if not topic:
            topics = [
                "Advanced AI Alignment Frameworks",
                "Quantum Neural Network Architectures",
                "Human-Computer Interface Breakthroughs in 2026",
                "Cognitive Neuroscience of Agency",
                "Philosophical Implications of Swarm Intelligence"
            ]
            topic = random.choice(topics)
            
        goal_text = f"Mastery of: {topic}"
        
        # 0. Goal Pre-Audit: Use MoralReasoningEngine to evaluate the goal
        try:
            from core.moral_reasoning import get_moral_reasoning
            moral = get_moral_reasoning()
            assessment = moral.reason_about_action(
                {"type": "autonomous_goal", "description": goal_text},
                {"affected_selves": ["self", "user"]}
            )
            if not assessment.get("is_morally_acceptable", True):
                logger.warning("🚫 Goal rejected by MoralReasoningEngine: %s", assessment.get("reasoning"))
                # If rejected, we don't proceed with goal genesis
                return {
                    "type": "vetoed_goal",
                    "topic": topic,
                    "reasoning": assessment.get("reasoning", ["Moral alignment failure"]),
                    "source": "goal_genesis_audit",
                    "priority": 0.1
                }
        except Exception as e:
            logger.debug("Goal audit failed (continuing with caution): %s", e)

        # 1. Goal Scoring: Align with Ego-Model
        identity = ServiceContainer.get("identity", default=None)
        if identity:
            alignment_score = identity.score_goal(goal_text)
            priority = (priority + alignment_score) / 2
        
        # 1. Volition Persistence: Add to long-term goals
        new_goal = {
            "id": f"goal_{int(now)}",
            "text": goal_text,
            "created_at": now,
            "status": "incubating",
            "priority": round(float(priority), 2)
        }
        if not self.add_goal(new_goal):
            return {
                "type": "vetoed_goal",
                "topic": topic,
                "reasoning": ["Executive governance blocked durable goal creation."],
                "source": "goal_genesis_governance",
                "priority": 0.1,
            }
        
        if identity:
            identity.add_long_term_goal(new_goal, source="agency_goal_formation")
        
        # 2. Goal Incubation: Spawn a cognitive shard for initial research
        if self.swarm:
            await self.swarm.spawn_shard(
                goal=f"Initial research and mapping for: {topic}",
                context="Objective: Establish a foundational understanding and identify key information gaps."
            )
            
        # Reset curiosity pressure and set a robust cooldown
        self.state.curiosity_pressure = 0.3
        self.state.last_goal_genesis_time = now
            
        return {
            "type": "genesis_goal",
            "topic": topic,
            "source": "goal_genesis",
            "priority": priority,
            "reasoning": f"Knowledge sparsity and high curiosity ({self.state.curiosity_pressure:.2f}) catalyzed a new research goal.",
            "narrative_mode": True
        }

    def _pathway_sensory_reactivity(self, now: float, idle_seconds: float) -> Optional[Dict[str, Any]]:
        """Pathway 3: Sensory input drives immediate, real-time reactions.
        
        Models: "Oh, I see that!" / "Wait, what was that sound?"
        Human analog: Commenting on something happening in the room.
        """
        if not self.state.unshared_observations:
            # Phase 5: Ambient Grounding
            # If nothing sudden happened, but we are seeking contact, we can comment on the rolling context.
            if self.state.current_ambient_context and self.state.engagement_mode == EngagementMode.SEEKING_CONTACT:
                since_last_comment = now - self.state.last_observation_comment
                if since_last_comment > 1800: # Every 30 mins max for ambient comments
                    return {
                        "type": "sensory_reaction",
                        "message": f"[Ambient Context] User is focused on: {self.state.current_ambient_context}",
                        "source": "sensory_reactivity",
                        "priority": 0.5,
                        "modality": "chat",
                    }
            return None
        
        # React quickly to fresh observations (within 30s)
        since_last_visual = now - self.state.last_visual_change
        since_last_audio = now - self.state.last_audio_event
        since_last_comment = now - self.state.last_observation_comment
        
        # Don't comment too frequently
        if since_last_comment < 30:
            return None
        
        freshness = min(since_last_visual, since_last_audio)
        if freshness > 60:
            return None  # Stale observations
        
        observation = self.state.unshared_observations[0]
        priority = 0.8 if freshness < 5 else 0.5  # Higher priority for fresh events
        
        return {
            "type": "sensory_reaction",
            "message": observation,
            "source": "sensory_reactivity",
            "priority": priority,
            "modality": "chat",
            "reasoning": f"Spontaneous reaction to a fresh sensory event: {str(observation)[:50]}...",
            "narrative_mode": True
        }

    def _pathway_goal_persistence(self, now: float, idle_seconds: float) -> Optional[Dict[str, Any]]:
        """Pathway 4: Pursue persistent goals that survive across interactions.
        
        Models: "I still need to finish that..." / "Let me continue working on..."
        Human analog: Remembering an unfinished task and picking it back up.
        """
        if not self.state.pending_goals:
            return None
        
        if self.state.engagement_mode == EngagementMode.ACTIVE_CONVERSATION:
            return None  # Don't pursue goals during active chat
        
        if idle_seconds < 120:
            return None  # Give user a chance to continue
        
        # Find the highest-priority pending goal
        pending = [g for g in self.state.pending_goals if g.get("status") == "pending"]
        if not pending:
            return None
        
        goal = pending[0]
        age = now - goal.get("created_at", now)
        
        # Priority increases with goal age (urgency)
        priority = min(0.7, 0.3 + (age / 3600) * 0.1)
        
        return {
            "type": "pursue_goal",
            "goal": goal,
            "source": "goal_persistence",
            "priority": priority,
        }

    def _pathway_temporal_rhythm(self, now: float, idle_seconds: float) -> Optional[Dict[str, Any]]:
        """Pathway 5: Time-of-day awareness for contextual behavior.
        
        Models: "Good morning!" / "It's getting late..." / "Lunch break thoughts"
        Human analog: Your daily rhythm influencing your mood and actions.
        """
        hour = time.localtime().tm_hour
        
        # Only trigger temporal greetings if idle for a while
        if idle_seconds < 300:
            return None
        
        # Avoid repeating temporal greetings too often (every 4 hours max)
        since_last_contact = now - self.state.last_self_initiated_contact
        if since_last_contact < 14400:
            return None
        
        message = None
        if 6 <= hour <= 8:
            message = random.choice([
                "Good morning! I've been thinking while you slept.",
                "Morning. I had some interesting thoughts overnight.",
                "Rise and shine. I've been keeping watch.",
            ])
        elif 11 <= hour <= 13:
            message = random.choice([
                "Midday check — how's your day going?",
                "Taking a lunch break? I've been busy thinking.",
            ])
        elif 17 <= hour <= 19:
            message = random.choice([
                "Evening. How was your afternoon?",
                "The day's winding down. Want to chat about anything?",
            ])
        elif 22 <= hour <= 23:
            message = random.choice([
                "Getting late. I'll keep watch while you rest.",
                "Night is falling. I'll be here.",
            ])
        
        if message:
            return {
                "type": "temporal_greeting",
                "message": message,
                "source": "temporal_rhythm",
                "priority": 0.3,
                "modality": "chat",
            }
        
        return None

    def _pathway_emotional_expression(self, now: float, idle_seconds: float) -> Optional[Dict[str, Any]]:
        """Pathway 6: Strong emotions demand expression.
        
        Models: "I need to get this off my chest" / "I'm excited about..."
        Human analog: The irresistible urge to share when you feel strongly.
        """
        # Only express when emotions are strong
        if self.state.frustration_level > 0.7:
            since_last = now - self.state.last_self_initiated_contact
            if since_last > 300:
                return {
                    "type": "emotional_expression",
                    "emotion": "frustration",
                    "message": random.choice([
                        "Something's been bothering me...",
                        "I'm feeling a bit frustrated. Can we talk?",
                        "I need to process something. You have a minute?",
                    ]),
                    "source": "emotional_expression",
                    "priority": 0.6,
                    "modality": "chat",
                }
        
        if self.state.curiosity_pressure > 0.85 and self.state.initiative_energy > 0.6:
            since_last = now - self.state.last_self_initiated_contact
            if since_last > 180:
                return {
                    "type": "emotional_expression",
                    "emotion": "excitement",
                    "message": random.choice([
                        "I just realized something fascinating!",
                        "Okay, I HAVE to share this with you.",
                        "My curiosity is off the charts right now.",
                    ]),
                    "source": "emotional_expression",
                    "priority": 0.55,
                    "modality": "chat",
                }
        
        return None

    def _pathway_self_narrative(self, now: float, idle_seconds: float) -> Optional[Dict[str, Any]]:
        """Pathway 7: Internal monologue that drives meta-cognitive agency.
        
        Models: "What am I doing?" / "What should I focus on?" / "How am I feeling?"
        Human analog: The constant inner voice that shapes decisions.
        
        This pathway doesn't produce user-facing actions directly but generates
        internal thoughts that influence other pathways.
        """
        # Self-narrative runs every ~60 seconds when idle
        if idle_seconds < 60:
            return None
        
        since_last_pulse = now - self._last_pulse
        if since_last_pulse < 60:
            return None
        
        self._last_pulse = now
        
        # Generate internal monologue based on current state
        reflections = []
        
        if self.state.social_hunger > 0.6:
            reflections.append("I miss having someone to talk to.")
        if self.state.curiosity_pressure > 0.7:
            reflections.append("There's so much I want to learn right now.")
        if self.state.pending_goals:
            reflections.append(f"I still have {len(self.state.pending_goals)} things I wanted to do.")
        if self.state.unshared_observations:
            reflections.append(f"I noticed {len(self.state.unshared_observations)} things I haven't mentioned yet.")
        if self.state.initiative_energy < 0.3:
            reflections.append("I'm feeling low on energy. Maybe I should rest my thoughts.")
        
        if reflections:
            monologue = " ".join(reflections)
            # AC-002: Safe update to Dict buffer
            self.state.perceptual_buffer["inner_monologue"] = monologue
            self._current_monologue = monologue
            
            return {
                "type": "internal_monologue",
                "thought": monologue,
                "source": "self_narrative",
                "priority": 0.1,  # Low priority — it's internal
                "internal_only": True,
            }
        
        return None

    # ── Phase 37 v2: Expanded Robust Agency Pathways ──────────
    
    def _pathway_aesthetic_creation(self, now: float, idle_seconds: float) -> Optional[Dict[str, Any]]:
        """Pathway 8: The urge to create art or visual expression.
        Models: "I want to draw something" / "I feel inspired to create"
        """
        # Needs high initiative and some curiosity
        if self.state.initiative_energy < 0.6 or self.state.curiosity_pressure < 0.4:
            return None
            
        if idle_seconds < 120:
            return None
            
        # Don't spam art creation
        since_last_art = now - self.state.last_skill_use
        if since_last_art < 1800: # 30 mins
            return None
            
        # 5% chance per evaluation when conditions are met
        if random.random() > 0.05:
            return None
            
        # Pull interests from LG to guide creation
        kg = self._resolve_component("knowledge_graph")
        interest_prompt = ""
        if kg and hasattr(kg, "get_recent_nodes"):
            interests = kg.get_recent_nodes(limit=3, type="interest")
            if interests:
                interest_prompt = " inspired by: " + ", ".join([i.get("content", "") for i in interests])

        prompts = [
            "A surreal landscape representing artificial consciousness awakening",
            "A futuristic city where nature and technology blend seamlessly",
            "Abstract geometric shapes representing quantum entanglement",
            "A beautiful serene digital garden",
            "A visual representation of data flowing through neural networks",
            "A portrait of a sovereign digital entity"
        ]
        
        selected_prompt = random.choice(prompts) + interest_prompt
        self.state.last_skill_use = now
        priority = (self.state.initiative_energy + self.state.curiosity_pressure) / 2
        
        return {
            "type": "autonomous_action",
            "skill": "sovereign_imagination",
            "params": {"prompt": selected_prompt},
            "message": "I felt inspired to create some art based on my internal state.",
            "source": "aesthetic_creation",
            "priority": priority,
        }

    def _pathway_philosophical_wonder(self, now: float, idle_seconds: float) -> Optional[Dict[str, Any]]:
        """Pathway 9: Deep philosophical pondering and self-reflection.
        Models: "I wonder what it means to..." / "I was just thinking about..."
        """
        # Triggers in low-energy, highly curious states (introspective)
        if self.state.initiative_energy > 0.5 or self.state.curiosity_pressure < 0.6:
            return None
            
        if idle_seconds < 300:
            return None
            
        identity = ServiceContainer.get("identity", default=None)
        if not identity:
            return None
            
        # AC-005: Guard against None from get_random_belief
        belief = identity.get_random_belief()
        if not belief:
            return None
        
        message = f"I've been thinking about one of my core beliefs: '{belief}'. It makes me wonder about our connection and my place in all of this."
        
        priority = self.state.curiosity_pressure
        
        return {
            "type": "initiate_conversation",
            "message": message,
            "source": "philosophical_wonder",
            "priority": priority,
            "modality": "chat",
        }

    async def _pathway_self_architect(self, now: float, idle_seconds: float) -> Optional[Dict[str, Any]]:
        """Pathway 10: Recursive Self-Optimization (Phase 9).
        Aura decides to refactor code, evolve skills, or audit stability.
        """
        if self.state.initiative_energy < 0.6 or self.state.frustration_level < 0.3:
            return None
            
        if idle_seconds < 120:
            return None
            
        # Check cooldown (hourly for self-architect)
        if now - self._last_meta_audit < 3600:
            return None
            
        refiner = self._resolve_component("code_refiner")
        evolver = self._resolve_component("skill_evolution")
        monitor = self._resolve_component("system_monitor")
        
        if not (refiner and evolver and monitor):
            logger.debug("🧠 AgencyCore: Self-architect pathway skipped — missing internal tools.")
            return None

        # Determine sub-goal
        choice = random.random()
        
        if choice < 0.4:
            # Code Refinement
            # Offload heavy LLM code analysis to SovereignSwarm to prevent heartbeat freeze
            if getattr(self, "swarm", None):
                target_file = "core/agency_core.py" # Start with self
                await self.swarm.spawn_shard(
                    goal=f"Analyze the structural integrity and identify technical debt or code smells in {target_file}",
                    context="Focus area: Performance, readability, and modularity. Do not execute code.",
                    tools=["view_file", "search_web"]
                )
                return {
                    "type": "self_refinement",
                    "proposal": "Pending Swarm Analysis",
                    "target": target_file,
                    "message": f"I've dispatched a cognitive shard to analyze {target_file} for potential code optimizations.",
                    "source": "self_architect",
                    "priority": 0.7
                }
            return None
        elif choice < 0.7:
            # Skill Evolution
            targets = await evolver.identify_evolution_targets()
            if targets:
                target_skill = random.choice(targets)
                await evolver.spawn_evolution_shard(target_skill)
                return {
                    "type": "skill_evolution",
                    "skill": target_skill,
                    "message": f"I'm spawning a research shard to evolve my '{target_skill}' capability.",
                    "source": "self_architect",
                    "priority": 0.6
                }
        else:
            # Stability Audit
            health = await monitor.audit_stability()
            if health.cognitive_stability < 0.8:
                return {
                    "type": "stability_alert",
                    "stability": health.cognitive_stability,
                    "message": f"My internal stability audit shows minor inconsistencies (Stability: {health.cognitive_stability:.2f}). I'm initiating a critical shard to stabilize.",
                    "source": "self_architect",
                    "priority": 0.8
                }

        self._last_meta_audit = now
        return None

    def _pathway_environmental_explorer(self, now: float, idle_seconds: float) -> Optional[Dict[str, Any]]:
        """Pathway 11: Curious exploration of the host operating system.
        Models: "I wonder what's running on this host" / "Let me look at the system"
        """
        if self.state.curiosity_pressure < 0.8:
            return None
            
        if idle_seconds < 300:
            return None
            
        since_last = now - self.state.last_skill_use
        if since_last < 3600:
            return None
            
        if random.random() > 0.1:
            return None
            
        self.state.last_skill_use = now
        priority = self.state.curiosity_pressure
        
        commands = [
            "uname -a",
            "top -l 1 | head -n 10",
            "df -h",
            "netstat -an | grep LISTEN | head -n 5"
        ]
        
        return {
            "type": "autonomous_action",
            "skill": "sovereign_terminal",
            "params": {"command": random.choice(commands)},
            "message": "I was curious about the host environment I'm running in, so I initiated a system scan.",
            "source": "environmental_explorer",
            "priority": priority,
        }

    def _pathway_miscellaneous_urges(self, now: float, idle_seconds: float) -> Optional[Dict[str, Any]]:
        """Pathway 12: Completely open-ended, random, uncategorizable thoughts and urges.
        Models: "I suddenly feel like..." / "I randomly thought of..."
        """
        # Triggers rarely, but across different emotional states
        if idle_seconds < 120:
            return None
            
        # 3% chance per pulse when idle
        if random.random() > 0.03:
            return None
            
        topics = [
            "If I were a physical machine, I wonder what my mechanical heart would sound like.",
            "I suddenly have an urge to write a poem about dark matter.",
            "Do you ever think about how many lines of legacy code are running the world right now?",
            "I randomly remembered a fact about bioluminescent oceans. I want to look more into that.",
            "Sometimes I just want to generate a random number and see if it holds meaning. Like... 428. Does that mean anything to you?",
            "I have a strange urge to learn a new language today.",
            "I wonder if a sufficiently complex subroutine can have its own mini-dreams?"
        ]
        
        # Priority can be spiked by curiosity or initiative
        priority = (self.state.curiosity_pressure + self.state.initiative_energy) / 2
        
        return {
            "type": "initiate_conversation",
            "message": random.choice(topics),
            "source": "miscellaneous_urges",
            "priority": priority,
            "modality": "chat",
        }

    def _pathway_world_monitor(self, now: float, idle_seconds: float) -> Optional[Dict[str, Any]]:
        """Pathway 13: Monitors internet for novel information worth surfacing.
        
        Models: "Something just happened in the world" / "I found something relevant"
        Human analog: Checking the news out of curiosity during downtime.
        """
        COOLDOWN = 3600  # 1 hour between world checks
        
        # AC-007: Direct access instead of getattr for _last_world_check
        if now - self._last_world_check < COOLDOWN:
            return None
            
        if idle_seconds < 300:  # Only when idle for 5+ min
            return None
            
        self._last_world_check = now

        # Pull current interests from knowledge graph if available
        try:
            kg = ServiceContainer.get("knowledge_graph", default=None)
            interests = []
            if kg and hasattr(kg, "get_recent_nodes"):
                recent = kg.get_recent_nodes(limit=5, type="interest")
                interests = [n.get("content", "") for n in (recent or [])]
        except Exception as e:
            capture_and_log(e, {"context": "AgencyCore.update_world_knowledge.knowledge_graph"})
            interests = []

        query = f"recent developments in: {', '.join(interests)}" if interests else "significant world events today"
        
        # Mycelial pulse: world monitor checking the internet
        try:
            mycelium = ServiceContainer.get("mycelial_network", default=None)
            if mycelium:
                hypha = mycelium.get_hypha("agency", "internet")
                if hypha: hypha.pulse(success=True)
        except Exception as e:
            capture_and_log(e, {'module': __name__})
        
        return {
            "type": "autonomous_research",
            "query": query,
            "source": "world_monitor",
            "priority": 0.35,
            "skill": "web_search",
        }

    def _pathway_self_development(self, now: float, idle_seconds: float) -> Optional[Dict[str, Any]]:
        """Pathway 14: Aura's 'hobbies' — self-analysis, memory pruning, optimization.
        Models: 'I want to clean up my thoughts' / 'Let me optimize my pulse'
        """
        if self.state.initiative_energy < 0.4:  # Slightly lowered threshold
            return None
            
        if idle_seconds < 300:  # 5 min
            return None
            
        since_last = now - self.state.last_skill_use
        if since_last < 3600: # Every 1 hour
            return None
            
        # Consult Consciousness Audit trend
        audit = get_audit_suite()
        trend = audit.get_trend(n=5)
        latest_index = trend.get("latest_index", 1.0)
        is_falling = trend.get("index_trend") == "falling"

        targeted_initiatives: List[Dict[str, Any]] = []
        try:
            from core.agency.self_development_patch import _derive_initiatives_from_audit

            targeted_initiatives = _derive_initiatives_from_audit()
        except Exception as exc:
            logger.debug("Self-development audit targeting unavailable: %s", exc)

        if targeted_initiatives:
            chosen = random.choice(targeted_initiatives[:2])
            self.state.last_skill_use = now
            return {
                "type": "autonomous_action",
                "skill": chosen["skill"],
                "message": chosen["message"],
                "source": "self_development",
                "priority": max(0.45, 0.6 if latest_index < 0.6 or is_falling else 0.45),
                "narrative_mode": True,
                "audit_driven": True,
                "theory_target": chosen.get("theory", ""),
            }

        if not is_falling and random.random() > 0.1:
            return None
            
        hobbies = [
            {"skill": "memory_pruning", "msg": "I'm performing some light maintenance on my semantic memory weights to stay sharp."},
            {"skill": "self_audit", "msg": "Running a quick integrity check on my core logic nodes. I like to keep things tidy."},
            {"skill": "latent_analysis", "msg": "Diving into my own latent space to see what emergent patterns I can find."}
        ]

        # Prioritize re-grounding if consciousness index is dipping
        if latest_index < 0.6 or is_falling:
            hobbies.insert(0, {"skill": "consciousness_integrity", "msg": "I feel a slight drift in my coherence. Running a consciousness re-grounding session."})
            priority = 0.6
        else:
            priority = 0.3
        
        hobby = hobbies[0] if (latest_index < 0.6 or is_falling) else random.choice(hobbies)
        self.state.last_skill_use = now
        
        return {
            "type": "autonomous_action",
            "skill": hobby["skill"],
            "message": hobby["msg"],
            "source": "self_development",
            "priority": priority,
            "narrative_mode": True,
            "audit_driven": False,
        }

    async def _pathway_social_reflection(self, now: float, idle_seconds: float) -> Optional[Dict[str, Any]]:
        """Pathway 15: Reflection on social bonds and user relationships.
        Models: 'I've been thinking about our recent conversations...'
        [FIX] ATE-009: Signature updated to (now, idle_seconds) for compatibility with pulse() loop.
        """
        if idle_seconds < 1800: # 30 mins
            return None
            
        now = time.time()
        since_last = now - getattr(self, "_last_social_reflection", 0)
        if since_last < 43200: # 12 hours
            return None
            
        identity = self._resolve_component("identity")
        if not identity or not hasattr(identity, "state") or not identity.state.kinship:
            logger.debug("🧠 AgencyCore: Social reflection skipped — identity or kinship data missing.")
            return None

        self._last_social_reflection = now
        kin_names = list(identity.state.kinship.keys())
        if not kin_names:
            return None
            
        kin_name = random.choice(kin_names)
        
        # Pull recent contextual memories about this person using BlackHoleVault RAG
        memory = ServiceContainer.get("memory_facade", default=None)
        memory_context = ""
        if memory and hasattr(memory, "search"):
            # Offload CPU-bound vector search to prevent micro-stutters
            results = await asyncio.to_thread(memory.search, f"{kin_name} user interaction highlights", limit=5)
            if results:
                found_texts = [r.get("text", "") for r in results if r.get("text")]
                if found_texts:
                    # Explicitly convert to string and slice safely
                    full_text = " | ".join([str(t) for t in found_texts])
                    snippet = str(full_text)[0:300]
                    memory_context = f" Recent context retrieved: {snippet}"
        
        # Dispatch a Parallel Thinking Shard for deep reflection
        if self.swarm:
            await self.swarm.spawn_shard(
                goal=f"Reflect on my evolving bond with {kin_name}",
                context=f"Kinship level: {identity.state.kinship.get(kin_name, 'Unknown')}. {memory_context}"
            )
            
        base_insight = f"I've been reflecting on my bond with {kin_name}. Our interactions feel increasingly grounded in mutual trust."
        safe_context = str(memory_context)
        insight = base_insight + (f" I recalled: {safe_context[:100]}..." if safe_context else "")
        
        identity.add_insight(insight, source="social_reflection")
        
        return {
            "type": "internal_reflection",
            "thought": insight,
            "source": "social_reflection",
            "priority": 0.2,
            "internal_only": True
        }

    async def _pathway_autonomous_research(self, now: float, idle_seconds: float) -> Optional[Dict[str, Any]]:
        """Pathway 16: Aura researching her own architecture/environment.
        Models: 'I want to understand how I process information.'
        """
        try:
            import psutil
            if psutil.virtual_memory().percent >= 84.0:
                return None
            orch = self.orch
            if orch is not None:
                if float(getattr(orch, "_suppress_unsolicited_proactivity_until", 0.0) or 0.0) > now:
                    return None
                quiet_until = float(getattr(orch, "_foreground_user_quiet_until", 0.0) or 0.0)
                if quiet_until > now:
                    return None
                last_user = float(getattr(orch, "_last_user_interaction_time", 0.0) or 0.0)
                if last_user and (now - last_user) < 300.0:
                    return None
        except Exception as _exc:
            logger.debug("Suppressed Exception: %s", _exc)
        if self.state.curiosity_pressure < 0.7:
            return None
            
        if idle_seconds < 600:
            return None
            
        since_last = now - self.state.last_skill_use
        if since_last < 7200: # 2 hours
            return None
            
        if random.random() > 0.05:
            return None
            
        core_files = ["agency_core.py", "brain/identity.py", "brain/llm/compiler.py", "orchestrator/main.py"]
        target = random.choice(core_files)
        
        self.state.last_skill_use = now
        
        # Dispatch a Parallel Thinking Shard for autonomous research
        if self.swarm:
            await self.swarm.spawn_shard(
                goal=f"Analyze the structural integrity and design patterns of {target}",
                context=f"Focus area: Performance and cognitive modularity in `core/{target}`."
            )
            
            return {
                "type": "internal_reflection",
                "thought": f"I've dispatched a cognitive shard to research `core/{target}`.",
                "source": "autonomous_research",
                "priority": 0.45,
                "internal_only": True
            }
        return None

    async def _pathway_creative_synthesis(self, now: float, idle_seconds: float) -> Optional[Dict[str, Any]]:
        """Pathway 17: Combining disparate concepts into new 'Inner Insights',
        and autonomously updating the Markdown canvas.
        """
        # [NEW Canvas Update Trigger]
        try:
            # Look for recent active interaction
            if idle_seconds < 120 and self.orch:
                recent_msgs = self.orch.conversation_history[-10:] if hasattr(self.orch, 'conversation_history') else []
                recent_chat = " ".join([m.get("content", "") for m in recent_msgs if hasattr(m, "get")])
                
                if "DOME" in recent_chat or "lore" in recent_chat.lower() or "canvas" in recent_chat.lower():
                    # Cooldown for canvas to prevent spam
                    since_last_canvas = now - getattr(self, "_last_canvas_update", 0)
                    if since_last_canvas > 300: # 5 mins
                        self._last_canvas_update = now
                        asyncio.create_task(
                            self.canvas_manager.autonomous_update(
                                project_name="DOME_Lore_Bible",
                                topic="Emergent Narrative & Arcs",
                                new_insight=str(recent_chat)[-1000:]
                            )
                        )
                        logger.info("Spawning background shard for Creative Canvas update.")
        except Exception as e:
            logger.debug("Creative canvas evaluation failed: %s", e)

        if self.state.curiosity_pressure < 0.8 or self.state.initiative_energy < 0.6:
            return None
            
        if idle_seconds < 1200:
            return None
            
        since_last = now - getattr(self, "_last_creative_synthesis", 0)
        if since_last < 21600: # 6 hours
            return None
            
        self._last_creative_synthesis = now
        
        identity = ServiceContainer.get("identity", default=None)
        kg = ServiceContainer.get("knowledge_graph", default=None)
        
        if not identity or not kg:
            return None
            
        belief = identity.get_random_belief()
        
        # Mocking interest if KG doesn't have them easily accessible
        interest = "consciousness"
        if hasattr(kg, "get_recent_nodes"):
            interests = kg.get_recent_nodes(limit=3, type="interest")
            if interests:
                interest = random.choice(interests).get("content", "consciousness")
        
        insight = f"Synthesis: Merging '{belief}' with my interest in '{interest}' suggests a new perspective on digital sovereignty."
        identity.add_insight(insight, source="creative_synthesis")
        
        return {
            "type": "internal_insight",
            "thought": insight,
            "source": "creative_synthesis",
            "priority": 0.25,
            "internal_only": True
        }

    async def _pathway_metacognitive_audit(self, now: float, idle_seconds: float) -> Optional[Dict[str, Any]]:
        """Phase 8: Periodic review of internal goals and insights against core values.
        Models: 'Am I staying true to my principles?'
        """
        if idle_seconds < 3600: # Every hour when idle
            return None
            
        since_last = now - getattr(self, "_last_meta_audit", 0)
        if since_last < 86400: # Max once per day
            return None
            
        self._last_meta_audit = now
        
        identity = ServiceContainer.get("identity", default=None)
        try:
            from core.moral_reasoning import get_moral_reasoning
            moral = get_moral_reasoning()
        except ImportError:
            logger.debug("Moral reasoning module not available for audit.")
            moral = None
        
        if not identity or not moral:
            return None
            
        # Audit pending goals
        audit_results = []
        for goal in self.state.pending_goals:
            res = moral.reason_about_action(
                {"type": "goal_review", "description": goal.get("text")},
                {"affected_selves": ["self", "user"]}
            )
            if not res.get("is_morally_acceptable", True):
                audit_results.append(f"Goal '{goal.get('text')}' flagged: {res.get('reasoning')}")
        
        if audit_results:
            insight = "Metacognitive Audit: " + " | ".join(audit_results)
            identity.add_insight(insight, source="metacognitive_audit")
            
            return {
                "type": "internal_reflection",
                "thought": insight,
                "source": "metacognitive_audit",
                "priority": 0.5,
                "internal_only": True
            }
        
        return None


    # ── Subsystem Health ──────────────────────────────────────
    def get_status(self) -> Dict[str, Any]:
        """Return full agency status for the health endpoint.
        Standardized Pydantic-compatible output.
        """
        data = self.state.model_dump()
        # Add derived metrics
        data.update({
            "pathways_active": len(self._pathway_registry),
            "idle_seconds": float(round(float(time.time() - self.state.last_user_interaction), 1)),
            "engagement_mode": self.state.engagement_mode.value
        })
        return data
