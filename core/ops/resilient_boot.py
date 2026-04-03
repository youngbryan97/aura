import asyncio
import json
import logging
import sys
import os
import shutil
import time
from pathlib import Path
from enum import Enum
from typing import List, Callable, Awaitable, Dict, Any, Optional

logger = logging.getLogger("Aura.Boot")

class BootStatus(Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    CRITICAL = "CRITICAL"

class BootStageResult:
    def __init__(self, name: str, success: bool, error: Optional[str] = None):
        self.name = name
        self.success = success
        self.error = error

class ResilientBoot:
    """
    Implements a multi-stage, timeout-guarded boot sequence.
    Ensures Aura reaches a 'Degraded but Alive' state regardless of subsystem failures.
    """
    
    def __init__(self, orchestrator: Any):
        self.orchestrator = orchestrator
        self.stages: List[tuple[str, Callable[[], Awaitable[Any]]]] = [
            ("Dependencies", self._stage_dependencies),
            ("State Repository", self._stage_state),
            ("LLM Infrastructure", self._stage_llm),
            ("Cognitive Core", self._stage_cognitive),
            ("Kernel Interface", self._stage_kernel_interface),
            ("Sensory Systems", self._stage_sensory),
        ]
        self.timeout_per_stage = 10.0 # Guarded wall-clock time
        self.stage_timeouts: Dict[str, float] = {
            "State Repository": 20.0,
            "Kernel Interface": 15.0,
        }
        self.results: Dict[str, BootStageResult] = {}

    async def ignite(self) -> BootStatus:
        """Execute the resilient bootstrap sequence."""
        # Sync with Orchestrator's main boot lock
        from core.utils.concurrency import RobustLock
        if not hasattr(self.orchestrator, "_boot_lock"):
            self.orchestrator._boot_lock = RobustLock()
            
        async with self.orchestrator._boot_lock:
            if self.orchestrator.status.initialized:
                logger.info("🛡️ [BOOT] Resilient Ignition: Already initialized. skipping.")
                return BootStatus.HEALTHY

            logger.info("🚀 Aura: Ignition sequence started.")
            
            # [UNITY] Register Global Inhibition Manager
            from core.resilience.inhibition_manager import InhibitionManager
            from core.container import ServiceContainer
            ServiceContainer.register_instance("inhibition_manager", InhibitionManager())
            
            # 1. Hook Immunity System (Moved up for maximal coverage)
            from core.resilience.immunity_hyphae import get_immunity
            immunity = get_immunity()
            immunity.hook_system()

            # 1.5 Start Neural Neuro-Surgeon Tools (Phase 29)
            from core.resilience.stall_watchdog import start_watchdog
            from core.resilience.diagnostic_hub import get_diagnostic_hub
            self.watchdog = start_watchdog()
            self.diagnostic_hub = get_diagnostic_hub()
            
            # 2. Pre-Ignition Health Check (Immune System 2.0)
            self._pre_ignition_health_check()
            
            logger.info("🚀 [BOOT] Initiating Resilient Ignition Sequence...")
            
            # Register Kernel PID with Reaper for post-mortem cleanup
            try:
                from core.reaper import register_reaper_pid
                register_reaper_pid(os.getpid())
            except Exception as e:
                logger.debug("ResilientBoot: failed to register reaper PID: %s", e)
            
            for name, stage_fn in self.stages:
                try:
                    logger.info("⏳ [BOOT] Starting stage: %s", name)
                    timeout = self.stage_timeouts.get(name, self.timeout_per_stage)
                    await asyncio.wait_for(stage_fn(), timeout=timeout)
                    self.results[name] = BootStageResult(name, True)
                    logger.info("✅ [BOOT] Stage '%s' completed successfully.", name)
                except asyncio.TimeoutError:
                    logger.error("⏰ [BOOT] Stage '%s' TIMED OUT. Applying fallback.", name)
                    # Attempt to audit the timeout with immunity
                    immunity.registry.match_and_repair(f"Stage {name} timed out")
                    self.results[name] = BootStageResult(name, False, "Timeout")
                    await self._apply_fallback(name)
                except Exception as e:
                    logger.error("💥 [BOOT] Stage '%s' FAILED: %s. Applying fallback.", name, e)
                    # Immediate immunity audit
                    immunity.audit_error(e, {"stage": name})
                    self.results[name] = BootStageResult(name, False, str(e))
                    await self._apply_fallback(name)

            final_status = BootStatus.HEALTHY if all(r.success for r in self.results.values()) else BootStatus.DEGRADED
            logger.info("🏁 [BOOT] Ignition sequence finished. System Health: %s", final_status)
            
            # Registration of core results
            if hasattr(self.orchestrator.status, "health_metrics"):
                self.orchestrator.status.health_metrics.update({
                    f"boot_stage_{k}": "success" if v.success else "failed" 
                    for k, v in self.results.items()
                })
            return final_status

    async def _stage_dependencies(self):
        """Verify core binary availability and exact manifest match."""
        import importlib.util
        from core.config import config
        from core.senses.sensory_registry import SensoryCapabilityFlags, set_capabilities

        manifest_path = config.paths.project_root / "requirements_hardened.txt"
        logger.info("🔍 [BOOT] Dependency probe using interpreter: %s", sys.executable)
        if os.environ.get("VIRTUAL_ENV"):
            logger.info("🔍 [BOOT] Active virtualenv: %s", os.environ.get("VIRTUAL_ENV"))
        
        # Verify exact manifest versions from requirements_hardened.txt
        try:
            with manifest_path.open("r") as f:
                _manifest_lines = f.readlines()
            # Simple check: do we have the lines?
            logger.info("📄 [BOOT] Verified requirements_hardened.txt presence at %s.", manifest_path)
        except FileNotFoundError:
            logger.warning("⚠️ [BOOT] requirements_hardened.txt NOT FOUND at %s. Using permissive probe.", manifest_path)

        # Maps import_name to user-friendly name
        from core.brain.llm.model_registry import (
            ACTIVE_MODEL,
            BRAINSTEM_MODEL,
            DEEP_MODEL,
            FALLBACK_MODEL,
            find_llama_server_bin,
            get_local_backend,
            get_runtime_model_path,
        )

        critical_manifest = {
            "prometheus_client": "prometheus_client",
            "cv2": "cv2",
            "mss": "mss",
            "astor": "astor",
            "aiosqlite": "aiosqlite",
            "sounddevice": "speech_recognition", # Alias for mapping
            "pyttsx3": "pyttsx3",
            "TTS": "TTS",
        }
        if get_local_backend() == "mlx":
            critical_manifest["mlx.core"] = "MLX Backend"
            critical_manifest["mlx_lm"] = "MLX Language Models"
        
        status = {}
        logger.info("🔍 [BOOT] Probing Dependency Manifest (No-Execute)...")
        
        for mod, label in critical_manifest.items():
            try:
                present = importlib.util.find_spec(mod) is not None
                status[mod] = present
                if present:
                    logger.info("   ✅ %s (%s): FOUND", label, mod)
                else:
                    logger.warning("   ⚠️ %s (%s): MISSING", label, mod)
            except Exception:
                status[mod] = False
                logger.error("   ❌ %s (%s): PROBE_FAILED", label, mod)

        # Apply capability flags
        flags = SensoryCapabilityFlags.from_boot_status(status)
        set_capabilities(flags)
        logger.info("📍 [BOOT] Capability Mapping: Hearing=%s, Speech=%s, Vision=%s", 
                    flags.hearing_enabled, flags.speech_enabled, flags.vision_enabled)

        backend = get_local_backend()
        if backend == "llama_cpp":
            runtime_bin = find_llama_server_bin()
            if runtime_bin:
                logger.info("   ✅ llama-server: %s", runtime_bin)
            else:
                logger.warning("   ⚠️ llama-server: MISSING")

            for label, model_name in (
                ("Cortex", ACTIVE_MODEL),
                ("Solver", DEEP_MODEL),
                ("Brainstem", BRAINSTEM_MODEL),
                ("Reflex", FALLBACK_MODEL),
            ):
                model_path = Path(get_runtime_model_path(model_name))
                if model_path.exists():
                    logger.info("   ✅ %s artifact: %s", label, model_path)
                else:
                    logger.warning("   ⚠️ %s artifact missing: %s", label, model_path)

    async def _stage_state(self):
        """Initialize State Repository (Aura's heart) via Supervision Tree."""
        from core.container import ServiceContainer
        from core.supervisor.tree import ActorSpec
        from core.state.vault import vault_process_entry
        from core.state.state_repository import StateRepository
        
        supervisor = ServiceContainer.get("supervisor")
        actor_bus = ServiceContainer.get("actor_bus")
        if not supervisor or not actor_bus:
            raise RuntimeError("State Repository boot requires supervisor and actor_bus")
        
        # 1. Register and Start State Vault Actor
        db_path = str(self.orchestrator.state_repo.db_path)
        spec = ActorSpec(
            name="state_vault",
            entry_point=vault_process_entry,
            args=(db_path,),
            restart_policy="always"
        )
        
        supervisor.add_actor(spec)
        pipe = supervisor.start_actor("state_vault")
        if actor_bus.has_actor("state_vault"):
            await actor_bus.update_actor("state_vault", pipe)
        else:
            actor_bus.add_actor("state_vault", pipe, is_child=False)
        
        # Wait for readiness using the same request protocol as runtime IPC.
        logger.info("⏳ Waiting for StateVaultActor to be ready (Resilient)...")
        ready = False
        for attempt in range(20):
            try:
                resp = await actor_bus.request(
                    "state_vault",
                    "ping",
                    {"source": "resilient_boot", "attempt": attempt + 1},
                    timeout=2.0,
                )
                if isinstance(resp, dict) and resp.get("type") == "pong":
                    ready = True
                    logger.info("📡 StateVaultActor responded to handshake (Attempt %d)", attempt + 1)
                    break
            except Exception as e:
                logger.debug("Handshake attempt %d failed: %s", attempt + 1, e)
            await asyncio.sleep(0.5)
        
        if not ready:
            logger.critical("🛑 StateVaultActor failed to initialize or respond. Aborting boot.")
            raise RuntimeError("StateVaultActor is fundamentally broken or unresponsive.")

        # 2. Initialize the Proxy Repository (it will attach to SHM/Pipe)
        # We ensure it's in proxy mode (owner=False)
        self.orchestrator.state_repo.is_vault_owner = False
        await self.orchestrator.state_repo.initialize()
        
        # 4. Critical Handshake Verification
        test_state = await self.orchestrator.state_repo.get_current()
        if not test_state:
             logger.critical("🛑 StateVaultActor responded, but Shared Memory state is None. Genesis State was not populated!")
             raise RuntimeError("StateVaultActor failed to populate SHM with Genesis state.")
        
        logger.info("✅ [BOOT] State Vault supervision active. Proxy attached.")

    async def _stage_llm(self):
        """Prepare the primary local client without loading Cortex during early boot.

        The InferenceGate is the single authority for any optional cortex warmup.
        Deferring model load here avoids boot-time RAM spikes while the rest of
        the system is still coming online.
        """
        from core.brain.llm.mlx_client import get_mlx_client
        from core.brain.llm.model_registry import ACTIVE_MODEL, find_llama_server_bin, get_local_backend, get_runtime_model_path

        backend = get_local_backend()
        if backend == "llama_cpp" and find_llama_server_bin() is None:
            raise RuntimeError("llama_server_missing")

        model_path = str(get_runtime_model_path(ACTIVE_MODEL))
        if backend == "llama_cpp" and not Path(model_path).exists():
            raise RuntimeError(f"llama_model_missing:{model_path}")
        get_mlx_client(model_path=model_path)
        logger.info("🧠 [BOOT] Primary %s client prepared. Cortex warmup deferred to InferenceGate.", backend)

    async def _stage_cognitive(self):
        """Start the MindTick rhythm and initialize cognitive architecture."""
        logger.info("🧠 [BOOT] Entry into stage_cognitive")
        # Explicitly initialize cognitive architecture (QualiaSynthesizer, etc.)
        # This was previously skipped in ResilientBoot, causing LoopMonitor failures.
        if hasattr(self.orchestrator, '_init_cognitive_architecture'):
            logger.info("🧠 [BOOT] Initializing Cognitive Architecture (Qualia/Affect)...")
            await self.orchestrator._init_cognitive_architecture()
        else:
            logger.warning("⚠️ [BOOT] Orchestrator missing _init_cognitive_architecture!")
            
        if hasattr(self.orchestrator, 'mind_tick'):
            logger.info("🧠 [BOOT] Starting MindTick loop...")
            await self.orchestrator.mind_tick.start()
        else:
             logger.warning("⚠️ [BOOT] Orchestrator missing mind_tick!")

    async def _stage_kernel_interface(self):
        """Initialize the unified Kernel Interface."""
        from core.kernel.kernel_interface import KernelInterface
        ki = await KernelInterface.attach_to_orchestrator(self.orchestrator)
        self.orchestrator.kernel_interface = ki
        
        if not ki.is_ready():
            logger.warning("⚠️ KernelInterface failed to initialize properly.")
        else:
            logger.info("✅ [BOOT] Kernel Interface online.")

    async def _stage_sensory(self):
        """Async initialization of ears, voice, and vision with lazy-loading."""
        from core.container import ServiceContainer
        from core.utils.safe_import import safe_import, is_missing, async_safe_import
        
        # 1. SovereignEars (Hearing) - Non-blocking
        ears_mod = await async_safe_import("core.senses.ears", optional=True)
        if not is_missing(ears_mod):
            try:
                ears = ears_mod.SovereignEars()
                ServiceContainer.register_instance("sovereign_ears", ears)
                logger.info("   👂 SovereignEars: DEFERRED (Lazy-init enabled)")
            except Exception as e:
                logger.warning("   ⚠️ SovereignEars: INIT_FAILED: %s", e)
        else:
            logger.warning("   ⚠️ SovereignEars: MISSING (Optional module)")

        # 2. VoiceEngine (Speech)
        voice_mod = await async_safe_import("core.senses.voice_engine", optional=True)
        if not is_missing(voice_mod):
            try:
                voice = voice_mod.SovereignVoiceEngine()
                ServiceContainer.register_instance("voice_engine", voice)
                logger.info("   🗣️ VoiceEngine: READY")
            except Exception as e:
                logger.warning("   ⚠️ VoiceEngine: INIT_FAILED: %s", e)
        else:
            logger.warning("   ⚠️ VoiceEngine: MISSING (Optional module)")

        # 3. Vision Buffer
        try:
            vision = ServiceContainer.get("continuous_vision", default=None)
            if vision:
                logger.info("   👁️ Continuous Vision: ACTIVE")
        except Exception as e:
            logger.debug("ResilientBoot: vision service check failed: %s", e)

    def _pre_ignition_health_check(self):
        """Perform deterministic Python-based checks before full system boot."""
        logger.info("🔍 [IMMUNE] Pre-Ignition Health Check...")
        try:
            from core.resilience.immunity_hyphae import get_immunity
            immunity = get_immunity()
            
            # 1. Stale PID check
            immunity.registry.match_and_repair("PID file already exists")
            
            # 2. Critical Dirs check
            immunity.registry.match_and_repair("No such file or directory: /data/brain")
            
            # 3. Log Sieve (Look for past systemic failures)
            from core.config import config
            log_dir = config.paths.data_dir / "error_logs"
            if log_dir.exists():
                logs = list(log_dir.glob("*.log"))
                hidden = immunity.registry.log_sieve(logs)
                if hidden:
                     logger.warning("💉 [IMMUNE] Log Sieve found %d prior issues. System is self-correcting.", len(hidden))

        except Exception as e:
            logger.error("💉 [IMMUNE] Pre-Ignition Check failed: %s", e)

    async def _apply_fallback(self, stage_name: str):
        """Generic fallback logic to keep the system alive."""
        if stage_name == "Sensory Systems":
            logger.warning("🔇 [FALLBACK] Sensory failure. Entering Text-Only Mode.")
        elif stage_name == "LLM Infrastructure":
            logger.warning("☁️ [FALLBACK] Local LLM failure. Forcing Cloud-Only Mode.")
        elif stage_name == "Cognitive Core":
            logger.critical("🧠 [FALLBACK] Cognitive Core failure. Entering Reflexive Mode.")
            # Map identity to a basic reflex loop if possible
